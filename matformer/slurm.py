"""SLURM behind one command interpreter: the server, the GUI, the launcher and a person in a
terminal all drive the cluster through the same `SlurmCommands` object, and nothing outside this
module has to know that `sbatch` exists.

It covers the cluster state (queue, nodes, partitions, GRES), the environments a job needs (a venv
per node architecture, communication variables), the lifecycle (submit, cancel, requeue, tail,
live status), survival (checkpoint and requeue when the time slot ends, restart when the training
crashes), a control channel to a training that is already running, and diagnostics that explain a
failed submission in terms of what to change.

A job lives in one directory on the shared filesystem, so the submitting host, the compute nodes and
any GUI agree on its state without a socket:

    <job_dir>/<matformer_id>/  job.sh      the generated wrapper
                               job.out     its output (sbatch stdout and the training's)
                               spec.json   the launch spec, read inside the allocation
                               control    a command mailbox the trainer polls
                               status.json rank 0 heartbeat: step, loss, host, memory
                               saved.json  last checkpoint written by that trainer

`python -m matformer.slurm run <spec.json>` is the entry point used inside an allocation;
`python -m matformer.slurm <line>` is the console.

Settings: slurm.toml next to this file, optionally overridden by $CONFIG_ASSETS/slurm.toml.
Every slurm call degrades: a host without slurm, a refused field or an unreachable controller gives
empty tables and a note in `cluster()`, never a traceback. Nothing here imports Tk or the server."""
import json
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
import tomllib
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from matformer.utils import matformer_assets
from matformer.utils.table import Table

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
USER = os.environ.get("USER") or os.environ.get("LOGNAME") or "unknown"

DEFAULTS = {
    "cluster": {"default_partition": "", "account": "", "qos": ""},
    "submit": {"time": "01:00:00", "cpus_per_task": 8, "signal_warn": 180, "requeue": True,
               "crash_restarts": 1, "max_requeues": 10, "job_dir": "$ASSETS/logs/slurm",
               "repo": str(REPO)},
    "venv": {"root": "$ASSETS/venvs", "extra": [], "by_arch": {}, "by_partition": {}},
    "gateway": {"port": 8766},
    "env": {},
}
DEFAULT_PROBE = r"""
echo "== host"; hostname; uname -m; head -2 /etc/os-release
echo "== cpus"; nproc; free -g | head -2
echo "== slurm"; env | grep '^SLURM' | sort
echo "== venv"; echo "$VIRTUAL_ENV"; "$VENV/bin/python" -VV 2>&1
"$VENV/bin/python" -c "import sys, site; print(' prefix', sys.prefix); print(' site', site.getsitepackages())" 2>&1
echo "== nvidia"; nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader 2>&1
echo "== net"; ip -o addr | awk '$2!="lo" {print "  "$2" "$4}'
echo "== python"
"$VENV/bin/python" - <<'PY'
import importlib, platform, sys
print("machine", platform.machine(), "python", sys.version.split()[0], "prefix", sys.prefix)
for name in ("torch", "lightning", "transformers", "numpy"):
    try:
        importlib.import_module(name)
        print("  ok  ", name)
    except Exception as problem:
        print("  FAIL", name, type(problem).__name__, str(problem)[:160])
try:
    import torch
    print("  torch", torch.__version__, "cuda", torch.version.cuda,
          "available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("  gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0),
              "count", torch.cuda.device_count())
except Exception as problem:
    print("  FAIL torch detail", type(problem).__name__, str(problem)[:160])
try:
    import torch
    from flash_attn import flash_attn_func
    q, k, v = (torch.randn(1, 64, 4, 64, dtype=torch.float16, device="cuda") for _ in range(3))
    out = flash_attn_func(q, k, v)
    torch.cuda.synchronize()
    print("  ok   flash_attn forward", tuple(out.shape), float(out.float().abs().mean()))
except Exception as problem:
    print("  FAIL flash_attn", type(problem).__name__, str(problem)[:200])
try:
    sys.path.insert(0, "$REPO")
    import matformer.slurm  # noqa: F401
    print("  ok   matformer imports")
except Exception as problem:
    print("  FAIL matformer", type(problem).__name__, str(problem)[:200])
PY
"""


def _merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for key, value in (extra or {}).items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


_CONFIG = None


def config(refresh: bool = False) -> dict:
    """slurm.toml from the package, then $CONFIG_ASSETS/slurm.toml, then the environment; every
    string gets its $*_ASSETS tokens expanded."""
    global _CONFIG
    if _CONFIG is not None and not refresh:
        return _CONFIG
    cfg = DEFAULTS
    files = [HERE / "slurm.toml"]
    if matformer_assets.enabled():
        files.append(matformer_assets.area("configs") / "slurm.toml")
    for file in files:
        if file.is_file():
            try:
                cfg = _merge(cfg, tomllib.loads(file.read_text()))
            except (OSError, tomllib.TOMLDecodeError) as problem:
                print(f"slurm: ignoring {file} ({problem})", file=sys.stderr)
    cfg = json.loads(json.dumps(cfg),
                     object_hook=lambda d: {k: matformer_assets.expand(v) if isinstance(v, str) else v
                                            for k, v in d.items()})
    if os.environ.get("SLURM_PARTITION"):
        cfg["cluster"]["default_partition"] = os.environ["SLURM_PARTITION"]
    _CONFIG = cfg
    return cfg


def job_root() -> Path:
    path = Path(config()["submit"]["job_dir"] or str(REPO / ".matformer_server" / "slurm"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def jobs_dir() -> Path:
    path = job_root() / "jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def available() -> bool:
    """Is there a SLURM client here at all (a laptop usually has none)."""
    return shutil.which("scontrol") is not None


# The state of the cluster is asked for constantly (tables, dialogs, watchdogs) and changes slowly.
_CACHE: dict = {}
_LOCK = threading.Lock()
ERRORS: list = []


def _slurm(argv: list, timeout: int = 20) -> str:
    """One slurm call that never raises: text out, empty string and a note in ERRORS when it fails."""
    if not available():
        _note("slurm is not installed on this host")
        return ""
    try:
        proc = subprocess.run([str(a) for a in argv], capture_output=True, text=True,
                              timeout=timeout)
    except (OSError, subprocess.SubprocessError) as problem:
        _note(f"{argv[0]}: {type(problem).__name__}: {problem}")
        return ""
    if proc.returncode != 0:
        _note(f"{argv[0]}: {(proc.stderr or proc.stdout).strip()[:200]}")
        return ""
    return proc.stdout.strip()


def _run(argv: list, timeout: int = 120) -> tuple[int, str, str]:
    """A slurm call whose result matters (submit, cancel, requeue): the caller sees the failure."""
    if not available():
        return 127, "", "slurm is not installed on this host"
    try:
        proc = subprocess.run([str(a) for a in argv], capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except (OSError, subprocess.SubprocessError) as problem:
        return 1, "", f"{type(problem).__name__}: {problem}"


def _note(text: str):
    ERRORS[:] = (ERRORS + [f"{datetime.now():%H:%M:%S} {text}"])[-12:]


def _cached(name: str, seconds: int, producer):
    with _LOCK:
        stamp, value = _CACHE.get(name, (0, None))
        if value is not None and time.time() - stamp < seconds:
            return value
    value = producer()
    with _LOCK:
        _CACHE[name] = (time.time(), value)
    return value


def cluster() -> dict:
    """Health of the cluster view itself: whether slurm answered, and what it complained about."""
    out = {"available": available(), "reachable": False, "name": "", "errors": list(ERRORS)}
    if not out["available"]:
        return out
    text = _slurm(["scontrol", "show", "config"], timeout=15)
    for line in (text or "").splitlines():
        match = re.match(r"\s*ClusterName\s*=\s*(\S+)", line)
        if match:
            out["name"], out["reachable"] = match.group(1), True
    return out


def _parse_pipe(text: str, names: list) -> list[dict]:
    rows = []
    for line in (text or "").splitlines():
        if not line.strip():
            continue
        values = [v.strip() for v in line.split("|")]
        rows.append({name: (values[i] if i < len(values) else "") for i, name in enumerate(names)})
    return rows


def _sinfo(spec: list, argv: list | None = None) -> list[dict]:
    names = [name for name, _ in spec]
    text = _slurm(["sinfo", "-h", "-o", "|".join(f"%{code}" for _, code in spec)] + list(argv or []))
    return _parse_pipe(text, names)


def _scontrol(query: list) -> list[dict]:
    """`scontrol show ... -o` is the reliable source: one line per object, key=value pairs."""
    text = _slurm(["scontrol", "show"] + list(query) + ["-o"])
    return [dict(re.findall(r"(\w+)=([^\s]+)", line)) for line in text.splitlines() if line.strip()]


def _squeue(names: list, argv: list | None = None) -> list[dict]:
    """Slurm builds disagree on which squeue columns exist. A refused one is reported on stderr
    while the command still succeeds, and the row simply comes back without it: so the refused
    names are collected and the query repeated until the columns and the data agree."""
    columns, refused = list(names), set()
    while columns:
        if not available():
            _note("slurm is not installed on this host")
            return []
        try:
            proc = subprocess.run(["squeue", "-h", "-O", ",".join(columns)] + list(argv or []),
                                  capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError) as problem:
            _note(f"squeue: {type(problem).__name__}: {problem}")
            return []
        refused |= set(re.findall(r"Invalid job format specification: (\w+)", proc.stderr or ""))
        kept = [one for one in columns if one not in refused]
        if proc.returncode != 0:
            _note(f"squeue: {(proc.stderr or proc.stdout).strip()[:200]}")
            return []
        if len(kept) < len(columns):
            columns = kept
            continue
        return _parse_pipe(proc.stdout, columns)
    return []


def to_seconds(value) -> int:
    """SLURM walltime text to seconds; -1 means unlimited or unknown."""
    text = str(value or "").strip()
    if not text or text.lower() in ("n/a", "none", "unlimited", "infinite", "unknown"):
        return -1
    days, _, rest = text.partition("-")
    parts = [int(p) for p in re.findall(r"\d+", rest or days)]
    while len(parts) < 3:
        parts.insert(0, 0)
    total = (parts[-3] * 3600 + parts[-2] * 60 + parts[-1])
    if days.isdigit() and "-" in text:
        total += int(days) * 86400
    return total


def from_seconds(total) -> str:
    if total is None or total < 0:
        return "-"
    days, rest = divmod(int(total), 86400)
    clock = f"{rest // 3600:02d}:{rest % 3600 // 60:02d}:{rest % 60:02d}"
    return f"{days}-{clock}" if days else clock


def _int(value, default: int = 0) -> int:
    try:
        return int(float(str(value).rstrip("MKGmkg")))
    except (TypeError, ValueError):
        return default


def _ts(text: str):
    try:
        return datetime.strptime(str(text), "%Y-%m-%dT%H:%M:%S")
    except (TypeError, ValueError):
        return None


# cluster state
def nodes() -> list[dict]:
    """Every node: architecture, GPU model, how many are free, state and reason if it is broken."""
    def produce():
        out = []
        for node in _scontrol(["nodes"]):
            gpus, model = 0, ""
            for item in (node.get("Gres") or "").split(","):
                parts = item.split(":")
                if parts[0] == "gpu" and len(parts) > 2:
                    model, gpus = parts[1], _int(parts[2])
            state = node.get("State", "UNKNOWN")
            out.append({"node": node.get("NodeName", ""), "state": state.split("+")[0].lower(),
                        "flags": state, "down": any(s in state for s in ("DOWN", "DRAIN", "FAIL")),
                        "arch": node.get("Arch", ""), "os": node.get("OSVersion", ""),
                        "partitions": [p for p in (node.get("Partitions") or "").split(",") if p],
                        "gpu_model": model, "gpus": gpus,
                        "gpus_used": _tres(node.get("AllocTRES", "")).get("gres/gpu", 0),
                        "cpus": _int(node.get("CPUTot")), "cpus_alloc": _int(node.get("CPUAlloc")),
                        "memory": _int(node.get("RealMemory")), "free_memory": _int(node.get("FreeMem")),
                        "load": _float((node.get("LoadAvg") or "0").split("/")[0]),
                        "features": node.get("AvailableFeatures", ""), "reason": node.get("Reason", ""),
                        "reservation": node.get("ReservationName", "")})
        for one in out:
            one["gpus_free"] = max(one["gpus"] - one["gpus_used"], 0)
            one["cpus_free"] = max(one["cpus"] - one["cpus_alloc"], 0)
        return out
    return _cached("nodes", 5, produce)


def _float(text) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def _tres(text: str) -> dict:
    out = {}
    for item in (text or "").split(","):
        key, _, value = item.partition("=")
        if key in ("cpu", "mem", "node", "billing") or key.startswith("gres"):
            out[key] = _int(value)
    return out


def partitions() -> list[dict]:
    """One row per partition: state, size, walltime ceiling, the GPU models it owns, exclusivity.
    Built from the controller's own description, with the node inventory counted in."""
    def produce():
        inventory = nodes()
        rows = []
        for detail in _scontrol(["partition"]):
            name = detail.get("PartitionName", "")
            owned = [one for one in inventory if name in one["partitions"]]
            usable = [one for one in owned if not one["down"]]
            rows.append({"name": name,
                         "default": detail.get("Default", "NO").upper().startswith("Y"),
                         "avail": detail.get("State", "unknown").lower(),
                         "state": "/".join(sorted({one["state"] for one in owned})) or "empty",
                         "nodes": len(owned), "nodes_usable": len(usable),
                         "max_time": detail.get("MaxTime", "UNLIMITED"),
                         "exclusive": not detail.get("Exclusive", "NO").upper().startswith("NO"),
                         "hidden": detail.get("Hidden", "NO").upper().startswith("YES"),
                         "oversubscribe": detail.get("OverSubscribe", ""),
                         "preempt": detail.get("PreemptMode", ""),
                         "gres": ",".join(sorted({one["gpu_model"] for one in owned
                                                  if one["gpu_model"]})),
                         "gpus": sum(one["gpus"] for one in owned),
                         "gpus_free": sum(one["gpus_free"] for one in usable),
                         "memory": _int(detail.get("DefMemPerNode")
                                        or (owned[0]["memory"] if owned else 0)),
                         "node_list": detail.get("Nodes", ""), "booked": name.endswith("-booked")})
        return sorted(rows, key=lambda r: (r["hidden"], r["name"]))
    return _cached("partitions", 15, produce)


def summary() -> list[dict]:
    """The sinfo view, parsed: how many nodes of each partition are idle, allocated, drained."""
    return _sinfo([("partition", "P"), ("avail", "a"), ("state", "t"), ("nodes", "D"),
                   ("cpus", "c"), ("memory", "m"), ("gres", "G"), ("reason", "E")])


def gres() -> list[dict]:
    """GPU inventory by model and architecture, most free first: where can a job land right now."""
    tally = {}
    for node in nodes():
        if not node["gpu_model"]:
            continue
        row = tally.setdefault((node["gpu_model"], node["arch"]),
                               {"gpu_model": node["gpu_model"], "arch": node["arch"], "nodes": 0,
                                "nodes_idle": 0, "gpus": 0, "gpus_free": 0, "gpus_per_node": 0,
                                "partitions": set()})
        row["nodes"] += 1
        row["gpus"] += node["gpus"]
        row["gpus_free"] += node["gpus_free"]
        row["nodes_idle"] += 1 if not node["down"] and node["gpus_free"] else 0
        row["gpus_per_node"] = max(row["gpus_per_node"], node["gpus"])
        row["partitions"].update(node["partitions"])
    rows = list(tally.values())
    for row in rows:
        row["partitions"] = ",".join(sorted(row["partitions"]))
    return sorted(rows, key=lambda r: (-r["gpus_free"], r["gpu_model"]))


def shapes(partition: str = "", only_free: bool = True) -> list[dict]:
    """Node sets a multi-node job could actually use: every rank needs interchangeable hardware, so
    nodes are grouped by (arch, gpu model, gpus per node) and only usable ones are listed."""
    groups = {}
    for node in nodes():
        if partition and partition not in node["partitions"]:
            continue
        key = (node["arch"], node["gpu_model"] or "cpu-only", node["gpus"])
        group = groups.setdefault(key, {"arch": key[0], "gpu_model": key[1], "gpus_per_node": key[2],
                                        "usable": [], "all": [], "partitions": set()})
        group["all"].append(node["node"])
        free = node["gpus_free"] == node["gpus"] or not node["gpus"]
        if not node["down"] and (free or not only_free):
            group["usable"].append(node["node"])
        group["partitions"].update(node["partitions"])
    return sorted([dict(group, nodes=len(group["all"]),
                        partitions=",".join(sorted(group["partitions"]))) for group in groups.values()],
                  key=lambda g: (-len(g["usable"]), g["gpu_model"]))


def queue(states: str = "ALL", mine: bool = False, user: str = "") -> list[dict]:
    """What is queued and running: state, why it waits, walltime used and left, the nodes it holds."""
    argv = []
    if states and states != "ALL":
        argv.append(f"--states={states}")
    who = user or (USER if mine else "")
    if who:
        argv.append(f"--user={who}")
    rows = _squeue(["JobID", "UserName", "Name", "Partition", "State", "ReasonList", "NumNodes",
                    "NodeList", "NCPUS", "Timelimit", "QOS", "Account", "Priority", "Gres",
                    "SubmitTime", "StartTime", "EndTime"], argv)
    for row in rows:
        row["gpus"] = _gres_count(row.get("Gres", ""))
        row["seconds_left"] = _left(row)
        row["matformer_id"] = _tracked(row.get("JobID", ""))
    return rows


def _gres_count(text: str) -> int:
    for item in (text or "").split(","):
        parts = item.split(":")
        if parts[0].startswith("gres") or parts[0] == "gpu":
            return _int(parts[-1])
    return 0


def _left(row: dict) -> int:
    """Seconds of walltime a running job still has; -1 when the limit is unlimited or unknown."""
    limit = to_seconds(row.get("Timelimit", ""))
    start = _ts(row.get("StartTime", ""))
    if limit < 0 or not start or row.get("State") != "RUNNING":
        return -1
    return limit - int((datetime.now() - start).total_seconds())


def job_detail(job_id: str) -> dict:
    """One job, in full, straight from the controller: RunTime, exit code, TRES, hold state."""
    return _cached(f"detail:{job_id}", 5, lambda: (_scontrol(["job", str(job_id)]) or [{}])[0])


def _accounted(job_id: str) -> dict:
    return _cached(f"accounted:{job_id}", 60, lambda: _accounted_uncached(job_id))


def _accounted_uncached(job_id: str) -> dict:
    """How a finished job ended (the queue forgets it as soon as it does). Field names move between
    Slurm builds, so a refused one is dropped and the query retried."""
    columns = ["JobID", "State", "ExitCode", "End", "Elapsed", "TotalCPU", "MaxRSS", "Nodes"]
    while columns:
        rc, out, err = _run(["sacct", "-n", "-P", "-X", "-j", str(job_id),
                             f"--format={','.join(columns)}"], timeout=20)
        if rc == 0:
            return _parse_pipe(out, columns)[-1] if out.strip() else {}
        refused = re.findall(r'Invalid field requested: "?(\w+)"?', err or "")
        kept = [one for one in columns if one not in refused]
        if len(kept) == len(columns) or not kept:
            return {}
        columns = kept
    return {}


# virtualenvs
ELF_MACHINE = {62: "x86_64", 183: "aarch64", 40: "arm", 3: "i386"}


def _elf_arch(path: Path) -> str:
    """Read the target architecture straight out of a compiled extension: the only way to know what
    a venv can run, since a cluster mixes architectures and a venv is not portable between them."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(20)
        return ELF_MACHINE.get(head[18] | (head[19] << 8), "") if head[:4] == b"\x7fELF" else ""
    except OSError:
        return ""


def _sample_binary(site: Path) -> Path:
    for pattern in ("torch/lib/lib*.so", "*/*.so", "**/*.so"):
        for candidate in sorted(site.glob(pattern)):
            if candidate.is_file():
                return candidate
    return Path(os.devnull)


def _pkg_version(site: Path, *names: str) -> str:
    for name in names:
        for spelling in {name, name.replace("-", "_"), name.replace("_", "-")}:
            for meta in sorted(site.glob(f"{spelling}-*.dist-info/METADATA")):
                try:
                    for line in open(meta, encoding="utf-8", errors="replace"):
                        if line.startswith("Version:"):
                            return line.split(":", 1)[1].strip()
                        if not line.strip():
                            break
                except OSError:
                    pass
    return ""


def describe_venv(path, size: bool = False) -> dict:
    """Name, architecture, python, torch, flash attention of one venv. The footprint is measured only
    when asked: du over a torch install on a network filesystem is slow enough to be a choice."""
    real = Path(os.path.realpath(str(path)))
    info = {"name": Path(str(path)).name, "path": str(real), "exists": real.is_dir(), "arch": "",
            "python": "", "torch": "", "flash_attn": "", "size_mb": 0, "site": ""}
    for site in sorted(real.glob("lib/python*/site-packages")):
        info["site"] = str(site)
        info["python"] = site.parent.name.replace("python", "")
        info["arch"] = _elf_arch(_sample_binary(site))
        info["torch"] = _pkg_version(site, "torch")
        info["flash_attn"] = _pkg_version(site, "flash_attn", "flash-attn-v100", "flash_attn_v100")
        info["size_mb"] = _size_mb(site) if size else 0
        break
    if not info["arch"]:
        info["arch"] = _elf_arch(real / "bin" / "python3") or ""
    return info


def _size_mb(site: Path) -> int:
    """Disk footprint, asked with du: walking a torch install file by file over a network
    filesystem takes minutes."""
    try:
        proc = subprocess.run(["du", "-sx", "--block-size=1M", str(site)], capture_output=True,
                              text=True, timeout=90)
    except (OSError, subprocess.SubprocessError):
        return 0
    fields = proc.stdout.split()
    return _int(fields[0]) if proc.returncode == 0 and fields else 0


def venvs() -> list[dict]:
    """Every venv under the assets venvs folder (plus whatever slurm.toml declares by path), each
    described so the interface can offer only the ones the chosen partition can execute."""
    def produce():
        cfg = config()["venv"]
        roots = [Path(os.path.expandvars(str(cfg["root"])))] if cfg.get("root") else []
        roots += [Path(os.path.expandvars(str(p))) for p in cfg.get("extra", [])]
        found, seen = [], set()
        for root in roots:
            if not root.is_dir():
                continue
            for entry in sorted(root.iterdir()):
                if not (entry / "pyvenv.cfg").exists() and not (entry / "bin" / "python").exists():
                    continue
                described = describe_venv(entry)
                if described["path"] not in seen:
                    seen.add(described["path"])
                    found.append(described)
        for name, target in {**(cfg.get("by_arch") or {}), **(cfg.get("by_partition") or {}),
                             **(cfg.get("default") or {})}.items():
            if target and str(Path(os.path.realpath(str(target)))) not in seen:
                found.append(describe_venv(target))
        return found
    return _cached("venvs", 300, produce)


def venv(name_or_path: str = "") -> dict:
    """Resolve a venv by name (as shown by `venvs`) or by absolute path."""
    if not name_or_path:
        return {}
    text = str(name_or_path)
    if text.startswith("/") or text.startswith("$"):
        return describe_venv(matformer_assets.expand(text))
    for one in venvs():
        if one["name"] == text:
            return one
    return {"name": text, "path": text, "exists": Path(text).is_dir(), "arch": "", "python": "",
            "torch": "", "flash_attn": "", "size_mb": 0, "site": ""}


def venv_for(partition: str = "", name: str = "", arch: str = "") -> dict:
    """The venv a job should use: an explicit name wins, then the per-partition mapping, then the
    per-architecture one, then any venv whose architecture matches the partition's nodes."""
    cfg = config()["venv"]
    if name:
        return venv(name)
    if partition and not arch:
        present = [node["arch"] for node in nodes() if partition in node["partitions"] and node["arch"]]
        arch = present[0] if present else ""
    if partition:
        mapped = (cfg.get("by_partition") or {}).get(partition)
        if mapped:
            return venv(mapped)
    if arch:
        mapped = (cfg.get("by_arch") or {}).get(arch)
        if mapped:
            return venv(mapped)
        for one in venvs():
            if one["arch"] == arch:
                return one
    options = venvs()
    return options[0] if options else {}


def partitions_for(arch: str) -> list:
    return sorted({p for node in nodes() if node["arch"] == arch for p in node["partitions"]})


def pip(venv_name: str, packages, upgrade: bool = False, no_deps: bool = True,
        partition: str = "", timeout: int = 600) -> dict:
    """Install into a venv that may not even run on this machine: when the architecture matches, do
    it here; otherwise do it inside a short CPU-only job on a node that can. Refuses to touch the
    compiled deep-learning stack, which takes an hour to replace and never a minute."""
    target = venv(venv_name)
    if not target.get("path"):
        return {"code": 1, "text": f"unknown venv: {venv_name}"}
    protected = ("torch", "triton", "flash-attn", "flash_attn", "nvidia-", "transformer-engine")
    requested = [str(p) for p in (packages if isinstance(packages, (list, tuple))
                                  else shlex.split(str(packages)))]
    blocked = [p for p in requested if p.split("=")[0].split("[")[0].lower().startswith(protected)]
    if upgrade and blocked:
        return {"code": 1, "text": f"refusing to reinstall {blocked}: protect the compiled stack"}
    argv = [target["path"] + "/bin/python", "-m", "pip", "install", "--no-cache-dir"]
    argv += (["--upgrade"] if upgrade else []) + (["--no-deps"] if no_deps else []) + requested
    quoted = " ".join(shlex.quote(part) for part in argv)
    if target["arch"] in (platform.machine(), ""):
        code, out, err = _run(argv, timeout=timeout)
        return {"where": f"{socket.gethostname()} ({platform.machine()})", "code": code,
                "text": (out + "\n" + err)[-6000:]}
    where = partition or (partitions_for(target["arch"]) or [""])[0]
    if not where:
        return {"code": 1, "text": f"no partition offers {target['arch']}: name one explicitly"}
    return probe(where, venv_name=target["name"], script=quoted, time_limit="00:05:00",
                 timeout=timeout)


# what a job asks for, and what it gets
@dataclass
class Allocation:
    """The request a job makes of the cluster. Nodes and GPUs per node are the two numbers that cost
    money; everything else is correctness."""
    partition: str = ""
    nodes_count: int = 1
    gpus_per_node: int = 1
    gpu_model: str = ""
    tasks_per_node: int = 0
    cpus_per_task: int = 0
    memory: str = ""
    time: str = ""
    qos: str = ""
    account: str = ""
    venv: str = ""
    exclusive: bool = True
    nodelist: str = ""
    exclude: str = ""
    requeue: bool = True
    crash_restarts: int = 0
    max_requeues: int = 0
    signal_warn: int = 0
    extra_sbatch: list = field(default_factory=list)
    env: dict = field(default_factory=dict)

    @classmethod
    def from_flat(cls, params: dict) -> "Allocation":
        """Build from whatever flat keys a form produced; unknown keys are ignored so a dialog can
        grow without breaking the interpreter."""
        aliases = {"nodes": "nodes_count", "gpus": "gpus_per_node", "cpus": "cpus_per_task",
                   "mem": "memory", "walltime": "time", "venv_name": "venv",
                   "restarts": "crash_restarts", "gres_per_node": "gpus_per_node",
                   "gpu": "gpu_model", "gpu_type": "gpu_model"}
        known = set(cls.__dataclass_fields__)
        picked = {}
        for key, value in (params or {}).items():
            if value in (None, "", [], {}):
                continue
            name = aliases.get(key, key)
            if name in known:
                picked[name] = value
        alloc = cls(**picked)
        cfg = config()
        alloc.partition = alloc.partition or cfg["cluster"]["default_partition"] or \
            next((p["name"] for p in partitions() if not p["hidden"]), "")
        alloc.time = alloc.time or cfg["submit"]["time"]
        alloc.cpus_per_task = alloc.cpus_per_task or cfg["submit"]["cpus_per_task"]
        alloc.signal_warn = alloc.signal_warn or cfg["submit"]["signal_warn"]
        alloc.account = alloc.account or cfg["cluster"]["account"]
        alloc.qos = alloc.qos or cfg["cluster"]["qos"]
        alloc.requeue = alloc.requeue and cfg["submit"]["requeue"]
        alloc.crash_restarts = alloc.crash_restarts or cfg["submit"]["crash_restarts"]
        alloc.max_requeues = alloc.max_requeues or cfg["submit"]["max_requeues"]
        return alloc

    def as_dict(self) -> dict:
        return asdict(self)


def env_for(alloc: Allocation) -> dict:
    """The environment a job exports, before anything the job itself says. Thread counts and the
    allocator settings are the parts that silently cost throughput, so they are set here rather than
    left to luck; the network interface and master address are computed inside the wrapper, where the
    answer is known instead of guessed."""
    env = {"OMP_NUM_THREADS": str(alloc.cpus_per_task or 1),
           "MKL_NUM_THREADS": str(alloc.cpus_per_task or 1),
           "TOKENIZERS_PARALLELISM": "false",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
           "PYTORCH_ALLOC_CONF": "expandable_segments:True",
           "NCCL_NSOCKS_PERTHREAD": "4",
           "NCCL_SOCKET_NTHREADS": "2"}
    if alloc.nodes_count > 1:
        env["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "1800"
        env["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    if os.environ.get("HF_HOME"):
        env["HF_HOME"] = os.environ["HF_HOME"]
    env.update(config()["env"])
    for key, value in (config().get("partition") or {}).get(alloc.partition, {}).get("env", {}).items():
        env[key] = value
    env.update(alloc.env or {})
    return {key: str(value) for key, value in env.items() if str(value) != ""}


def gres_spec(alloc: Allocation) -> str:
    """The GRES request. Naming the GPU model is what keeps a job off the wrong hardware: a partition
    can hold several kinds, and only Slurm can pick by type."""
    if not alloc.gpus_per_node:
        return ""
    return f"gpu:{alloc.gpu_model}:{alloc.gpus_per_node}" if alloc.gpu_model \
        else f"gpu:{alloc.gpus_per_node}"


def directives(alloc: Allocation, name: str, log: Path, requeue_capable: bool = True) -> str:
    lines = [f"#SBATCH --job-name={name[:64]}", f"#SBATCH --partition={alloc.partition}",
             f"#SBATCH --nodes={alloc.nodes_count}",
             f"#SBATCH --ntasks-per-node={max(alloc.tasks_per_node or alloc.gpus_per_node, 1)}",
             f"#SBATCH --cpus-per-task={alloc.cpus_per_task or 1}",
             f"#SBATCH --time={alloc.time}",
             f"#SBATCH --output={log}", f"#SBATCH --error={log}",
             "#SBATCH --export=NONE", "#SBATCH --open-mode=append",
             f"#SBATCH --signal=USR1@{alloc.signal_warn}"]
    if alloc.gpus_per_node:
        lines.insert(3, f"#SBATCH --gres={gres_spec(alloc)}")
    if alloc.exclusive:
        lines.append("#SBATCH --exclusive")
    if requeue_capable:
        lines.append("#SBATCH --requeue" if alloc.requeue else "#SBATCH --no-requeue")
    for option, value in (("mem", alloc.memory), ("qos", alloc.qos), ("account", alloc.account),
                          ("nodelist", alloc.nodelist), ("exclude", alloc.exclude)):
        if value:
            lines.append(f"#SBATCH --{option}={value}")
    lines += [str(extra) for extra in alloc.extra_sbatch]
    return "\n".join(lines) + "\n"


def environment_block(alloc: Allocation, venv_path: str, runtime: Path, mid: str) -> str:
    return f"""RUNTIME="{runtime}"
REPO="{config()['submit']['repo']}"
VENV="{venv_path}"
SPEC="$RUNTIME/spec.json"
mkdir -p "$RUNTIME"
cd "$REPO" 2>/dev/null || echo "[matformer] repo $REPO is missing"
export PYTHONPATH="$REPO${{PYTHONPATH:+:$PYTHONPATH}}"
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
{chr(10).join(f'export {key}={shlex.quote(value)}' for key, value in env_for(alloc).items())}
export MASTER_ADDR="${{MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)}}"
export MASTER_PORT="${{MASTER_PORT:-$(( 20000 + (${{SLURM_JOB_ID:-0}} % 9973) ))}}"
export MATFORMER_CONTROL="$RUNTIME/control"
export MATFORMER_STATUS="$RUNTIME/status.json"
export MATFORMER_JOB="{mid}"
IFACE="$(ip -o route get "$MASTER_ADDR" 2>/dev/null | sed -n 's/.* dev \\([^ ]*\\).*/\\1/p')"
[ -n "$IFACE" ] && export NCCL_SOCKET_IFNAME="${{NCCL_SOCKET_IFNAME:-$IFACE}}"
echo "[matformer] {mid} job=${{SLURM_JOB_ID:-local}} nodes=${{SLURM_JOB_NODELIST:-1}} gpus/node={alloc.gpus_per_node} arch=$(uname -m) venv=$(basename "$VENV") master=$MASTER_ADDR:$MASTER_PORT iface=${{NCCL_SOCKET_IFNAME:-none}}"
"""


TRAIN_BODY = """INTERRUPTED=0
stop_for() {{
  INTERRUPTED=1
  printf '{{"cmd": "save_and_stop", "reason": "%s", "at": "%s"}}' "$1" "$(date -Iseconds)" > "$RUNTIME/control"
}}
trap 'stop_for time_limit' USR1
trap 'stop_for sigterm' TERM

# Markers of the previous run: a requeue reuses this directory on shared storage.
rm -f "$RUNTIME/stopped.json" "$RUNTIME/saved.json" "$RUNTIME/control"
RERUNS="$(cat "$RUNTIME/reruns" 2>/dev/null || echo 0)"
echo "$((RERUNS + 1))" > "$RUNTIME/reruns"

ATTEMPT=0
CODE=1
while : ; do
  ATTEMPT=$((ATTEMPT + 1))
  srun --export=ALL python -m matformer.slurm run "$SPEC"
  CODE=$?
  echo "[matformer] training step exited with $CODE on attempt $ATTEMPT"
  [ "$CODE" = "0" ] && break
  [ "$INTERRUPTED" = "1" ] && break
  [ "$ATTEMPT" -gt "{restarts}" ] && break
  sleep 10
done

# Requeue when the run did not reach its own end: the time slot expired (the training records that
# in stopped.json, because Slurm may signal the steps and not this script), or it crashed with
# restarts still in the budget. Never after a crash loop: that is a bug, not scheduling.
WANTED=0
[ "$INTERRUPTED" = "1" ] && WANTED=1
[ -f "$RUNTIME/stopped.json" ] && WANTED=1
if [ "$CODE" != "0" ] && [ "$RERUNS" -lt "{max_requeues}" ]; then WANTED=1; fi
if [ "$WANTED" = "1" ] && [ ! -f "$RUNTIME/done.json" ] && [ "{requeue}" = "1" ]; then
  for _ in $(seq 1 30); do [ -f "$RUNTIME/saved.json" ] && break; [ "$CODE" != "0" ] && break; sleep 2; done
  echo "[matformer] requeueing ${{SLURM_JOB_ID}} (rerun $((RERUNS + 1)) of {max_requeues})"
  scontrol requeue "${{SLURM_JOB_ID}}" || true
fi
exit "$CODE"
"""


def script_text(alloc: Allocation, mid: str, body: str = "", name: str = "training") -> str:
    runtime = jobs_dir() / mid
    return ("#!/bin/bash\n"
            + directives(alloc, name, runtime / "job.out")
            + f"# Matformer job {mid}, generated by matformer/slurm.py. Edit the request, not this.\n"
            + environment_block(alloc, venv_for(alloc.partition, alloc.venv)["path"], runtime, mid)
            + (body or TRAIN_BODY.format(restarts=alloc.crash_restarts,
                                         max_requeues=alloc.max_requeues,
                                         requeue="1" if alloc.requeue else "0")))


def submit(alloc: Allocation, spec: dict = None, name: str = "", dry_run: bool = False) -> dict:
    """Write the wrapper for one training and hand it to sbatch. `spec` is a plain launch spec, the
    same dict the local queue uses, so the only difference between this box and the cluster is who
    runs it."""
    spec = dict(spec or {})
    spec["name"] = name or spec.get("name") or "training"
    options = dict(spec.get("options") or {})
    options.update({"num_nodes": alloc.nodes_count, "device_count": max(alloc.gpus_per_node, 1)})
    spec["options"] = options
    mid = f"{re.sub(r'[^A-Za-z0-9_.-]', '-', spec['name'])}-{datetime.now():%y%m%d%H%M%S}"
    runtime = jobs_dir() / mid
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "spec.json").write_text(json.dumps(spec, indent=1, default=str))
    (runtime / "job.sh").write_text(script_text(alloc, mid, name=spec["name"]))
    problems = preflight(alloc, spec)
    errors = [one for one in problems if one["severity"] == "error"]
    if dry_run or errors:
        return {"matformer_id": mid, "script": str(runtime / "job.sh"),
                "submitted": False, "problems": problems,
                "error": "; ".join(one["problem"] for one in errors) or None}
    code, out, err = _run(["sbatch", str(runtime / "job.sh")], timeout=120)
    if code != 0:
        raise RuntimeError(f"sbatch failed: {err or out}")
    slurm_id = next(iter(re.findall(r"\d+", out)), "")
    entry = {"matformer_id": mid, "slurm_id": slurm_id, "name": spec["name"],
             "partition": alloc.partition, "nodes": alloc.nodes_count, "gpus": alloc.gpus_per_node,
             "venv": venv_for(alloc.partition, alloc.venv)["name"], "time": alloc.time,
             "submitted": datetime.now().isoformat(timespec="seconds"),
             "script": str(runtime / "job.sh"), "log": str(runtime / "job.out"),
             "requeue": alloc.requeue, "state": "PENDING"}
    _store(entry)
    return dict(entry, problems=problems)


def submit_training(spec: dict, params: dict = None, name: str = "") -> dict:
    """The door the server uses: a launch spec plus the flat request a form produced. Allocation knobs
    may sit at the top level or inside the form's `options`. The answer has the same shape as the
    local queue's ({run: id}), so nothing downstream cares where the job actually ran."""
    params = params or {}
    alloc = Allocation.from_flat({**params, **(params.get("options") or {}),
                                  **(params.get("cluster") or {})})
    entry = submit(alloc, spec, name=name or spec.get("name", "training"))
    if entry.get("error"):
        raise ValueError(entry["error"])
    return {"run": entry["matformer_id"], "slurm_id": entry["slurm_id"], "cluster": True,
            "problems": entry.get("problems", [])}


# the registry, on the shared filesystem
def registry_file() -> Path:
    return job_root() / "jobs.json"


def _registry() -> dict:
    try:
        return json.loads(registry_file().read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _store(entry: dict):
    with _LOCK:
        data = _registry()
        data[entry["matformer_id"]] = entry
        write_json(registry_file(), data)


def _tracked(slurm_id: str) -> str:
    if not slurm_id:
        return ""
    for mid, entry in _registry().items():
        if str(entry.get("slurm_id")) == str(slurm_id):
            return mid
    return ""


def tracked() -> dict:
    return _registry()


def job_dir(matformer_id: str) -> Path:
    path = jobs_dir() / matformer_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def write_json(path, payload: dict):
    """Atomic write with a process-unique temporary name: every rank of a distributed job writes the
    same files, and a shared name would let them steal each other's half-written file."""
    path = Path(path)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, default=str))
    os.replace(temporary, path)


def resolve(job: str) -> dict:
    """Accept a matformer id, a slurm id or a job name."""
    text = str(job)
    data = _registry()
    if text in data:
        return data[text]
    for entry in data.values():
        if text in (str(entry.get("slurm_id")), entry.get("name")):
            return entry
    if text.isdigit():
        return {"matformer_id": "", "slurm_id": text}
    raise ValueError(f"unknown job: {job}")


def slurm_id_of(job: str) -> str:
    slurm_id = resolve(job).get("slurm_id", "")
    if not slurm_id:
        raise ValueError(f"{job} has no slurm job id")
    return slurm_id


# lifecycle
def command(job: str, cmd: str, **extra) -> dict:
    """Leave one command in the job's mailbox; the trainer reads it at its next optimizer step."""
    entry = resolve(job)
    path = job_dir(entry["matformer_id"]) / "control"
    write_json(path, {"cmd": cmd, "at": datetime.now().isoformat(timespec="seconds"), **extra})
    return {"msg": f"{cmd} requested for {entry['matformer_id'] or entry.get('slurm_id')}",
            "control": str(path)}


def save(job: str) -> dict:
    """Write a checkpoint now and keep training."""
    return command(job, "save")


def stop(job: str, save_first: bool = True) -> dict:
    """Finish cleanly: checkpoint, exit, and let the wrapper decide about requeue."""
    return command(job, "save_and_stop" if save_first else "stop", reason="manual")


def setvar(job: str, path: str, value) -> dict:
    """Change a live value of a running training: lr, max_steps, save_every_n_steps."""
    return command(job, "set", field=path, value=value)


def cancel(job: str, signal_name: str = "") -> dict:
    code, out, err = _run(["scancel"] + ([f"--signal={signal_name}"] if signal_name else [])
                          + [slurm_id_of(job)])
    return {"code": code, "msg": (out or err or f"cancelled {job}").strip()}


def requeue(job: str, checkpoint: bool = True) -> dict:
    """Restart the job when it ends. With `checkpoint` the running training is told to save and
    exit first, so the restart resumes from that file instead of losing an interval."""
    slurm_id = slurm_id_of(job)
    if checkpoint:
        command(job, "save_and_stop", reason="manual_requeue")
    code, out, err = _run(["scontrol", "requeue", slurm_id])
    return {"code": code, "msg": (out or err or f"requeue requested for {slurm_id}").strip()}


def hold(job: str, release: bool = False) -> dict:
    code, out, err = _run(["scontrol", "release" if release else "hold", slurm_id_of(job)])
    return {"code": code, "msg": (out or err or "").strip()}


def prioritize(job: str, adjust: int = 1000) -> dict:
    code, out, err = _run(["scontrol", "update", f"JobId={slurm_id_of(job)}", f"Priority={adjust}"])
    return {"code": code, "msg": (out or err or f"priority update requested").strip()}


def tail(job: str, lines: int = 80) -> str:
    entry = resolve(job)
    path = Path(entry.get("log") or (job_dir(entry["matformer_id"]) / "job.out")) if entry else Path("")
    if not path.is_file():
        return f"(no log at {path})"
    return "\n".join(path.read_text(errors="replace").splitlines()[-int(lines):])


def log_text(job: str, maximum: int = 400_000) -> str:
    """The end of a job's log, by bytes rather than lines: a progress bar writes thousands of lines
    and would push the exception out of view."""
    entry = resolve(job)
    path = Path(entry.get("log") or (job_dir(entry["matformer_id"]) / "job.out")) if entry else Path("")
    if not path.is_file():
        return ""
    size = path.stat().st_size
    with open(path, "rb") as handle:
        if size > maximum:
            handle.seek(size - maximum)
        return handle.read().decode("utf-8", errors="replace")


def _job_details(ids: list[str]) -> dict:
    ids = [one for one in ids if one]
    if not ids:
        return {}
    rows = _scontrol(["job", ",".join(ids)])
    out = {}
    for row in rows:
        key = str(row.get("JobId") or row.get("JobID") or row.get("JobIDRaw") or "")
        if key:
            out[key] = row
    return out


def _accounted_many(ids: list[str]) -> dict:
    ids = [one for one in ids if one]
    if not ids:
        return {}
    columns = ["JobID", "State", "ExitCode", "End", "Elapsed", "TotalCPU", "MaxRSS", "Nodes"]
    code, out, err = _run(["sacct", "-n", "-P", "-X", "-j", ",".join(ids),
                           f"--format={','.join(columns)}"], timeout=25)
    if code != 0:
        return {}
    return {row.get("JobID", ""): row for row in _parse_pipe(out, columns)}


def status(job: str = "", fast: bool = False) -> list[dict]:
    """The live view of the jobs this cluster knows: slurm state, progress, last checkpoint, and how
    much of the time slot is left. `fast` skips per-job controller calls for jobs that are no
    longer in the queue: the table opens immediately and the watchdog fills the details later."""
    return _cached(f"status:{job or '*'}:{bool(fast)}", 5, lambda: _status(job, fast))


def _status(job: str = "", fast: bool = False) -> list[dict]:
    rows = []
    live = _cached("status_queue", 3, lambda: queue()) or []
    registry = _registry()
    selected = {mid: entry for mid, entry in registry.items()
                if not job or job in (mid, str(entry.get("slurm_id")), entry.get("name"))}
    missing = [str(entry.get("slurm_id")) for mid, entry in selected.items()
               if entry.get("slurm_id") and not read_json(job_dir(mid) / "done.json")
               and not any(str(one.get("JobID")) == str(entry.get("slurm_id")) for one in live)]
    details = _job_details(missing) if missing and not fast else {}
    ended_map = _accounted_many(missing) if missing and not fast else {}
    for mid, entry in selected.items():
        runtime = job_dir(mid)
        slurm_id = str(entry.get("slurm_id", ""))
        row = next((one for one in live if str(one.get("JobID")) == slurm_id), {})
        done = read_json(runtime / "done.json")
        detail = details.get(slurm_id, {})
        ended = ended_map.get(slurm_id, {})
        heart = read_json(runtime / "status.json")
        saved = read_json(runtime / "saved.json")
        state = row.get("State") or detail.get("JobState") or ("DONE" if done else
                                                               ended.get("State") or
                                                               (entry.get("state") if fast else "GONE"))
        rows.append({"matformer_id": mid, "slurm_id": str(entry.get("slurm_id", "")),
                     "name": entry.get("name", ""), "state": state,
                     "reason": row.get("ReasonList") or detail.get("Reason", "") or ended.get("ExitCode", ""),
                     "partition": entry.get("partition", ""), "nodes": entry.get("nodes", 1),
                     "gpus": entry.get("gpus", 0), "venv": entry.get("venv", ""),
                     "step": heart.get("step"), "total": heart.get("total"), "loss": heart.get("loss"),
                     "epoch": heart.get("epoch"), "host": heart.get("host") or row.get("NodeList", ""),
                     "nodelist": row.get("NodeList", "") or ended.get("NodeList", ""),
                     "gpu_mem_mib": heart.get("gpu_mem_mib"), "saved_step": saved.get("step"),
                     "seconds_left": _left(row) if row else -1, "used": detail.get("RunTime", ""),
                     "limit": row.get("Timelimit", "") or entry.get("time", ""),
                     "log": entry.get("log", "")})
    return sorted(rows, key=lambda r: r["matformer_id"], reverse=True)


def attach(job: str) -> dict:
    """How to reach the node running a job: a tunnel to open, and the command that starts a matformer
    server there. A laptop GUI can then read that node's own runs while the training owns it."""
    rows = status(job)
    row = rows[0] if rows else {}
    host = row.get("host", "").split(",")[0]
    port = int(config()["gateway"]["port"])
    python = str(Path(row.get("venv", "") or ".") / "bin" / "python")
    return {"host": host, "port": port, "python": python, "status": row,
            "tunnel": f"ssh -N -L {port}:localhost:{port} {USER}@{host}",
            "start": f"ssh {USER}@{host} 'cd {config()['submit']['repo']} && "
                     f"MATFORMER_SERVER_PORT={port} nohup {python} -m matformer.server "
                     f"> /tmp/matformer_server_{host}.log 2>&1 &'"}


# diagnostics
PATTERNS = [
    (r"CUDA out of memory|out of memory", "the GPU ran out of memory",
     "lower batch_size, or shorten max_seq_len; PYTORCH_CUDA_ALLOC_CONF is already set to "
     "expandable_segments:True"),
    (r"collective operation timeout|NCCL.*timed out", "a collective timed out",
     "ranks disagree about how many of them exist, or cannot talk: check that tasks_per_node matches "
     "gpus_per_node, and try a fixed NCCL_SOCKET_IFNAME / NCCL_IB_DISABLE=1"),
    (r"NCCL INFO Bootstrap|unhandled system error", "NCCL could not establish communication",
     "pin the interface the nodes can reach each other on: [env] NCCL_SOCKET_IFNAME in slurm.toml"),
    (r"No module named 'tomllib'", "the venv is being run by an older interpreter than it was built "
     "for", "this node has no python new enough for the venv: slurm probe <partition> prints the "
     "venv's interpreter on that machine"),
    (r"No module named '([\w.]+)'", "the venv is missing {}",
     "install it into that venv (slurm pip <venv> {}): the interpreter picks a node whose "
     "architecture matches"),
    (r"is non-finite, so it cannot be clipped", "gradients went infinite or NaN",
     "the learning rate is too high for this batch, or the dtype cannot represent the activations: "
     "lower lr, raise accumulate_grad_batches, or change training.precision"),
    (r"Attempting to unscale FP16 gradients", "fp16 weights with an AMP loss scaler, which torch "
     "refuses to unscale", "use training.precision=16-mixed (fp32 weights, fp16 activations under "
     "autocast) instead of 16-true"),
    (r"not implemented for 'BFloat16'", "this GPU cannot compute in bf16",
     "bf16 needs Ampere or newer: on a V100 use training.precision=16-mixed"),
    (r"undefined symbol|cannot open shared object file.*\.so'?", "a binary in the venv was built for "
     "another architecture or toolchain", "use the venv whose arch matches the partition"),
    (r"no kernel image is available|CUDA arch mismatch", "torch cannot run on this GPU",
     "the venv's CUDA build does not cover this GPU architecture"),
    (r"FlashAttention|flash_attn", "flash attention problem",
     "do not reinstall it: choose the venv that already ships a wheel compiled for this architecture"),
    (r"Permission denied: '(/[^']*)'", "cannot write {}",
     "check the folder's ownership, or move the assets folder (configuration.yaml)"),
    (r"Address already in use", "the rendezvous port is taken",
     "set MASTER_PORT; the wrapper derives it from the job id to avoid exactly this"),
    (r"FileNotFoundError: (\S+)", "{} is missing",
     "check the path and the $*_ASSETS tokens against this machine's assets folder"),
    (r"assert[^\n]*'trainer' section|No training section", "the config has no training section",
     "the YAML list must include a file that declares `training:`"),
    (r"Killed|memory cgroup out of memory", "the job exceeded its memory limit",
     "ask for more memory, or lower num_workers and batch_size"),
]


def problems_in(text: str) -> list[dict]:
    found = {}
    for pattern, problem, fix in PATTERNS:
        for match in re.finditer(pattern, text or "", re.IGNORECASE):
            detail = match.group(1) if match.groups() else ""
            key = problem.replace("{}", str(detail))
            found.setdefault(key, {"severity": "error", "problem": key,
                                   "fix": fix.replace("{}", str(detail)),
                                   "evidence": match.group(0)[:120]})
    return list(found.values())


def preflight(alloc: Allocation, spec: dict = None) -> list[dict]:
    """What can be caught for free before an allocation is paid for: a partition that does not exist,
    more GPUs than a node has, a venv built for another architecture, a walltime above the ceiling,
    configs that are not on disk — and the exclusive-node trap, where one GPU books a whole machine."""
    out = []
    parts = {one["name"]: one for one in partitions()}
    if not alloc.partition:
        out.append({"severity": "error", "problem": "no partition was asked for",
                    "fix": "set cluster.default_partition in slurm.toml, or pass partition="})
    elif parts and alloc.partition not in parts:
        out.append({"severity": "error", "problem": f"unknown partition '{alloc.partition}'",
                    "fix": "slurm partitions"})
    part = parts.get(alloc.partition, {})
    present = [node for node in nodes() if alloc.partition in node["partitions"]]
    archs = sorted({node["arch"] for node in present if node["arch"]})
    if alloc.gpu_model and present:
        models = sorted({node["gpu_model"] for node in present if node["gpu_model"]})
        if alloc.gpu_model not in models:
            out.append({"severity": "error",
                        "problem": f"{alloc.partition} has no {alloc.gpu_model} GPUs ({', '.join(models)})",
                        "fix": "pick a partition that owns that model: slurm gres"})
    if not present and parts:
        out.append({"severity": "warning", "problem": f"no node visible in '{alloc.partition}'",
                    "fix": "the partition may be hidden or drained: slurm partitions"})
    per_node = [node["gpus"] for node in present if not node["down"]]
    if alloc.gpus_per_node and per_node and alloc.gpus_per_node > max(per_node):
        out.append({"severity": "error", "problem":
                    f"{alloc.partition} nodes have at most {max(per_node)} GPUs, {alloc.gpus_per_node} asked",
                    "fix": "lower gpus_per_node or spread over more nodes"})
    limit = to_seconds(part.get("max_time", ""))
    if limit > 0 and to_seconds(alloc.time) > limit:
        out.append({"severity": "error", "problem":
                    f"time {alloc.time} exceeds the {part['max_time']} ceiling of {alloc.partition}",
                    "fix": "shorten it: the wrapper checkpoints and requeues, so a short slot only "
                           "costs a restart"})
    usable = [node for node in present if not node["down"] and node["gpus_free"] >= alloc.gpus_per_node]
    if part and len(usable) < alloc.nodes_count:
        out.append({"severity": "warning", "problem":
                    f"only {len(usable)} of {len(present)} nodes in {alloc.partition} can take "
                    f"{alloc.gpus_per_node} GPU(s) right now",
                    "fix": "the job will wait for capacity: slurm gres, slurm shapes"})
    if part.get("exclusive"):
        out.append({"severity": "info", "problem":
                    f"{alloc.partition} books whole nodes, whatever the GPU count",
                    "fix": "fill the node: gpus_per_node=" + str(max(node["gpus"] for node in present)
                                                                 if present else 1)})
    if alloc.nodes_count > 1 and present:
        fits = [group for group in shapes(alloc.partition) if len(group["usable"]) >= alloc.nodes_count]
        if not fits:
            out.append({"severity": "warning", "problem":
                        f"no set of {alloc.nodes_count} interchangeable nodes in {alloc.partition}",
                        "fix": "slurm shapes shows the node groups that are free together"})
    target = venv_for(alloc.partition, alloc.venv)
    if not target.get("path"):
        out.append({"severity": "error", "problem": "no virtualenv available",
                    "fix": f"put one under {config()['venv']['root']} or set venv.by_arch in slurm.toml"})
    else:
        if archs and target["arch"] and target["arch"] not in archs:
            out.append({"severity": "error", "problem":
                        f"venv {target['name']} is {target['arch']} but {alloc.partition} is {'/'.join(archs)}",
                        "fix": "pick the venv built for the partition's architecture: slurm venvs"})
        if not target.get("torch"):
            out.append({"severity": "warning", "problem": f"venv {target['name']} has no torch",
                        "fix": "that venv is not a training environment"})
    for path in (spec or {}).get("config") or []:
        expanded = matformer_assets.expand(path)
        if not Path(expanded).is_file():
            out.append({"severity": "error", "problem": f"config not found: {expanded}",
                        "fix": "check the path, or the assets folder this machine declares"})
    return out


def diagnose(job: str = "") -> list[dict]:
    """Why something is not training: the controller's own reason, then the job's log, then the state
    of the nodes it ran on. Each finding carries the change that fixes it."""
    out = []
    if not available():
        return [{"severity": "error", "problem": "slurm is not installed on this host",
                 "fix": "run the server on a submission host"}]
    for row in (status(job) if job else status()):
        mid = row["matformer_id"]
        state = row["state"]
        if state == "PENDING":
            reason = str(row.get("reason", ""))
            out.append({"severity": "warning", "job": mid, "problem": f"waiting: {reason}",
                        "fix": _waiting(row)})
        elif state == "RUNNING":
            if row.get("step") is None:
                out.append({"severity": "info", "job": mid,
                            "problem": "running, but no step has been reported yet",
                            "fix": "the first batches are data and model setup; look at the log"})
            left = row.get("seconds_left", -1)
            if 0 <= left <= 2 * int(config()["submit"]["signal_warn"]):
                out.append({"severity": "warning", "job": mid,
                            "problem": f"{from_seconds(left)} of the time slot is left",
                            "fix": "the wrapper checkpoints and requeues on its own"})
        elif state not in ("DONE", "GONE", "COMPLETED", "CANCELLED"):
            found = problems_in(log_text(mid))
            if not found:
                found = [{"severity": "error",
                          "problem": f"{state} ({row.get('reason', '')})",
                          "fix": "read the end of the log: slurm tail " + mid}]
            out += [dict(one, job=mid, state=state) for one in found]
    for one in out:
        one.setdefault("severity", "warning")
    return out


def _waiting(row: dict) -> str:
    reason = str(row.get("reason", "")).lower()
    if "priority" in reason:
        return f"behind other jobs: slurm prioritize {row['matformer_id']}"
    if "resources" in reason or "nodes" in reason:
        return "no free capacity for that shape: slurm gres, or fewer nodes/GPUs"
    if "qos" in reason:
        return "the QOS limit refuses the request: shorten time or lower the resource count"
    if "reservation" in reason:
        return "the nodes are held by a reservation"
    if "license" in reason or "feature" in reason:
        return "the requested features are not present on the free nodes"
    return row.get("reason", "")


# inside the allocation
def run_spec(spec_path: str) -> None:
    """What srun executes on every rank: the framework's own entry point, with a heartbeat and a
    SIGTERM answer, so a job that is taken away still leaves a checkpoint behind."""
    from matformer.training import control
    from matformer.training.launch import execute

    spec = read_json(spec_path)
    mailbox = control.channel()
    print(f"[matformer] rank={os.environ.get('SLURM_PROCID', '-')} node={socket.gethostname()} "
          f"arch={platform.machine()} python={platform.python_version()}", flush=True)
    execute(spec)
    # Only a training that reached its own end is finished: one that stopped because the time slot
    # expired must still be allowed to requeue and resume from the checkpoint it just wrote.
    if os.environ.get("SLURM_PROCID", "0") == "0":
        runtime = Path(os.environ.get("MATFORMER_STATUS", "status.json")).parent
        write_json(runtime / ("stopped.json" if mailbox.stop else "done.json"),
                   {"at": datetime.now().isoformat(timespec="seconds"),
                    "reason": mailbox.reason, "step": mailbox.step})


def probe(partition: str, nodes_count: int = 1, gpus: int = 0, gpu_model: str = "",
          venv_name: str = "", script: str = "", time_limit: str = "00:05:00", cpus: int = 4,
          timeout: int = 900) -> dict:
    """Run something small where the training would run, and bring the answer back: what a venv really
    contains on that architecture, whether the GPUs and the network behave, what pip installed.
    Nothing is guessed about the node; it is asked."""
    alloc = Allocation.from_flat({"partition": partition, "nodes": nodes_count, "gpus": gpus,
                                  "gpu_model": gpu_model, "venv": venv_name, "time": time_limit,
                                  "cpus": cpus, "requeue": False, "restarts": 0})
    mid = f"probe-{datetime.now():%H%M%S}"
    runtime = job_dir(mid)
    body = script or DEFAULT_PROBE
    (runtime / "probe.sh").write_text("#!/bin/bash\n" + environment_block(
        alloc, venv_for(alloc.partition, alloc.venv)["path"], runtime, mid) + body)
    (runtime / "job.sh").write_text("#!/bin/bash\n"
                                    + directives(alloc, mid, runtime / "job.out", requeue_capable=False)
                                    + f"# probe of {partition}\n"
                                    + f"bash {runtime}/probe.sh\n")
    code, out, err = _run(["sbatch", str(runtime / "job.sh")])
    if code != 0:
        return {"code": code, "error": err or out, "script": str(runtime / "job.sh")}
    slurm_id = next(iter(re.findall(r"\d+", out)), "")
    log = runtime / "job.out"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _squeue(["JobID"], [f"--job={slurm_id}"]):
            break
        time.sleep(5)
    text = log.read_text(errors="replace") if log.is_file() else f"(no output at {log})"
    return {"slurm_id": slurm_id, "log": str(log), "code": 0,
            "text": text[-14000:], "problems": problems_in(text)}


# the interpreter
class SlurmCommands:
    """Every cluster operation, Tables out. The server exposes it as RPC, the GUI draws what comes
    back, and a person can drive the same object from `python -m matformer.slurm <line>`."""

    def cluster(self) -> dict:
        """Is there a cluster here, which one, and what slurm complained about lately."""
        return cluster()

    def queue(self, states: str = "ALL", mine: bool = False) -> Table:
        """The queue: state, why a job waits, walltime used and left, nodes held."""
        return Table(["id", "user", "name", "partition", "state", "nodes", "gpus", "limit", "left",
                      "nodelist", "reason"],
                     [{"id": r.get("JobID", ""), "user": r.get("UserName", ""),
                       "name": r.get("Name", ""), "partition": r.get("Partition", ""),
                       "state": r.get("State", ""), "nodes": r.get("NumNodes", ""),
                       "gpus": r.get("gpus", 0), "limit": r.get("Timelimit", ""),
                       "left": from_seconds(r.get("seconds_left", -1)),
                       "nodelist": r.get("NodeList", ""), "reason": r.get("ReasonList", "")}
                      for r in queue(states=states, mine=bool(mine))])

    def jobs(self, fast: bool = False) -> Table:
        """The trainings this cluster is running: step, loss, last checkpoint, time left."""
        return Table(["job", "slurm", "state", "partition", "nodes", "gpus", "step", "loss",
                      "saved", "left", "venv", "host"],
                     [{"job": r["matformer_id"], "slurm": r["slurm_id"], "state": r["state"],
                       "partition": r["partition"], "nodes": r["nodes"], "gpus": r["gpus"],
                       "step": f"{r['step']}/{r['total']}" if r["step"] is not None else "",
                       "loss": r["loss"] if r["loss"] is not None else "",
                       "saved": r["saved_step"], "left": from_seconds(r["seconds_left"]),
                       "venv": r["venv"], "host": r["host"]} for r in status(fast=fast)])

    def nodes(self, partition: str = "", free_only: bool = False) -> Table:
        """Nodes with their architecture and free GPUs."""
        rows = [one for one in nodes() if not partition or partition in one["partitions"]]
        if free_only:
            rows = [one for one in rows if not one["down"] and one["gpus_free"]]
        return Table(["node", "state", "arch", "gpu", "gpus_free", "cpus_free", "mem_mb",
                      "partitions", "reason"],
                     [{"node": one["node"], "state": one["state"], "arch": one["arch"],
                       "gpu": one["gpu_model"], "gpus_free": one["gpus_free"],
                       "cpus_free": one["cpus_free"], "mem_mb": one["free_memory"],
                       "partitions": ",".join(one["partitions"]),
                       "reason": one["reason"] if one["down"] else ""} for one in rows])

    def partitions(self) -> Table:
        """Partitions: state, size, walltime ceiling, GPU models, exclusivity."""
        return Table(["name", "avail", "state", "nodes", "usable", "max_time", "gres", "gpus_free",
                      "exclusive", "hidden"],
                     [{k: p[k] for k in ("name", "avail", "state", "nodes", "nodes_usable",
                                         "max_time", "gres", "gpus_free", "exclusive", "hidden")}
                      for p in partitions()])

    def summary(self) -> Table:
        """The sinfo view: node states per partition, as the scheduler reports them."""
        return Table(["partition", "avail", "state", "nodes", "cpus", "memory", "gres", "reason"],
                     summary())

    def gres(self) -> Table:
        """GPU inventory by model and architecture, most free first."""
        return Table(["gpu_model", "arch", "nodes", "nodes_idle", "gpus", "gpus_free",
                      "gpus_per_node", "partitions"], gres())

    def shapes(self, partition: str = "") -> Table:
        """Groups of interchangeable nodes: what a multi-node job can land on."""
        return Table(["arch", "gpu_model", "gpus_per_node", "usable", "total", "partitions",
                      "free_nodes"],
                     [{"arch": group["arch"], "gpu_model": group["gpu_model"],
                       "gpus_per_node": group["gpus_per_node"], "usable": len(group["usable"]),
                       "total": group["nodes"], "partitions": group["partitions"],
                       "free_nodes": ",".join(group["usable"][:6])
                                     + (" …" if len(group["usable"]) > 6 else "")}
                      for group in shapes(partition)])

    def venvs(self) -> Table:
        """Virtualenvs: architecture, python, torch, flash attention. A venv only runs on nodes whose
        architecture matches the one it was compiled for."""
        return Table(["name", "arch", "python", "torch", "flash_attn", "path"],
                     [{k: one[k] for k in ("name", "arch", "python", "torch", "flash_attn",
                                          "path")} for one in venvs()])

    def venv(self, name: str = "") -> dict:
        """One venv in detail, resolved to its real path, footprint included."""
        found = venv_for("", name)
        return describe_venv(found.get("path", ""), size=True) if found.get("path") else found

    def allocation_fields(self) -> list:
        """The knobs that decide where a training runs, in the field language the forms speak, with
        the live choices of this cluster filled in."""
        from matformer.utils.schema import fields_schema
        options = [p["name"] for p in partitions() if not p["hidden"]]
        names = [one["name"] for one in venvs()]
        models = sorted({one["gpu_model"] for one in nodes() if one["gpu_model"]})
        out = []
        for one in fields_schema(Allocation, skip=("extra_sbatch", "env", "tasks_per_node")):
            if one["name"] == "partition":
                one.update(kind="choice", options=options)
            elif one["name"] == "venv":
                one.update(kind="choice", options=names)
            elif one["name"] == "gpu_model":
                one.update(kind="choice", options=[""] + models,
                           doc="empty: any model the partition has")
            elif one["name"] in ("nodes_count", "gpus_per_node", "cpus_per_task"):
                one.update(kind="int", min=0)
            elif one["name"] == "crash_restarts":
                one.update(kind="int", min=0)
            out.append(one)
        return out

    def submit(self, job: str = "", partition: str = "", nodes: int = 1, gpus: int = 1,
               venv: str = "", time: str = "", config_paths: str = "", overrides: str = "",
               dry_run: bool = False, **allocation) -> dict:
        """Submit a training to the cluster. configs and overrides are space separated; the rest of
        the keywords are allocation knobs. Returns the ids and where the files are."""
        alloc = Allocation.from_flat(dict(allocation, partition=partition, nodes=nodes, gpus=gpus,
                                         venv=venv, time=time))
        spec = {"config": shlex.split(config_paths), "overrides": shlex.split(overrides),
                "options": {}, "name": job or "training"}
        return submit(alloc, spec, name=spec["name"])

    def submit_training(self, spec: dict, allocation: dict) -> dict:
        """The programmatic door used by the server: a launch spec plus the allocation it should run
        under. Same spec, same code, different machine."""
        return submit(Allocation.from_flat(allocation), spec, name=spec.get("name", "training"))

    def info(self, job: str) -> dict:
        """Everything known about one job: registry entry, controller state, heartbeat, files."""
        entry = resolve(job)
        mid = entry.get("matformer_id") or ""
        runtime = job_dir(mid) if mid else Path(".")
        return {"entry": entry, "allocation": entry.get("allocation", {}),
                "spec": read_json(runtime / "spec.json") if mid else {},
                "detail": job_detail(entry.get("slurm_id", "")),
                "heartbeat": read_json(runtime / "status.json") if mid else {},
                "saved": read_json(runtime / "saved.json") if mid else {},
                "control": read_json(runtime / "control") if mid else {},
                "files": [f.name for f in runtime.iterdir()] if mid else []}

    def save(self, job: str) -> dict:
        """Force a checkpoint in a running training; it keeps going afterwards."""
        return save(job)

    def stop(self, job: str, save: bool = True) -> dict:
        """Ask a running training to finish: checkpoint, exit cleanly."""
        return stop(job, save_first=bool(save))

    def setvar(self, job: str, field: str, value: str) -> dict:
        """Change lr, max_steps or save_every_n_steps of a live training."""
        return setvar(job, field, value)

    def requeue(self, job: str, checkpoint: bool = True) -> dict:
        """Have SLURM restart a job, after a checkpoint when asked."""
        return requeue(job, checkpoint=bool(checkpoint))

    def cancel(self, job: str) -> dict:
        """scancel a job (matformer id, slurm id or name)."""
        return cancel(job)

    def hold(self, job: str, release: bool = False) -> dict:
        """Hold a pending job, or let it go."""
        return hold(job, release=bool(release))

    def prioritize(self, job: str, adjust: int = 1000) -> dict:
        """Ask the scheduler to look at this job sooner."""
        return prioritize(job, int(adjust))

    def tail(self, job: str, lines: int = 80) -> str:
        """The end of a job's log."""
        return tail(job, int(lines))

    def status(self, job: str = "") -> Table:
        """Live state of one job, or of all of them."""
        rows = status(job)
        return Table(["job", "slurm", "state", "step", "total", "loss", "saved", "left", "host"],
                     [{"job": r["matformer_id"], "slurm": r["slurm_id"], "state": r["state"],
                       "step": r["step"], "total": r["total"], "loss": r["loss"],
                       "saved": r["saved_step"], "left": from_seconds(r["seconds_left"]),
                       "host": r["host"]} for r in rows])

    def doctor(self, job: str = "") -> Table:
        """What is wrong, and the change that fixes it."""
        return Table(["severity", "job", "problem", "fix"], diagnose(job))

    def preflight(self, partition: str = "", nodes: int = 1, gpus: int = 1, venv: str = "",
                  time: str = "", config_paths: str = "") -> Table:
        """Test a request before paying for it."""
        alloc = Allocation.from_flat({"partition": partition, "nodes": nodes, "gpus": gpus,
                                      "venv": venv, "time": time})
        return Table(["severity", "problem", "fix"],
                     preflight(alloc, {"config": shlex.split(config_paths)}))

    def probe(self, partition: str, nodes: int = 1, gpus: int = 0, gpu_model: str = "",
              venv: str = "", time: str = "00:05:00") -> dict:
        """Ask a node what it is: hardware, venv contents, GPUs, network, imports."""
        return probe(partition, nodes_count=int(nodes), gpus=int(gpus), gpu_model=gpu_model,
                     venv_name=venv, time_limit=time)

    def pip(self, venv: str, packages: str, upgrade: bool = False) -> dict:
        """Install into a venv, on a machine that can actually run it."""
        return pip(venv, packages, upgrade=bool(upgrade))

    def attach(self, job: str) -> dict:
        """How to talk to the node a job is running on."""
        return attach(job)

    def env(self, partition: str = "", nodes: int = 1, gpus: int = 1, venv: str = "") -> dict:
        """The exact environment a job would get, and the venv it would use."""
        alloc = Allocation.from_flat({"partition": partition, "nodes": nodes, "gpus": gpus,
                                      "venv": venv})
        return {"env": env_for(alloc), "venv": venv_for(alloc.partition, alloc.venv)}


def names() -> list:
    import inspect
    return [one for one in dir(SlurmCommands)
            if not one.startswith("_") and inspect.isfunction(getattr(SlurmCommands, one))]


def line(text: str):
    """Run one interpreter line: 'queue', 'nodes partition=free', 'tail 1234 lines=200'."""
    parts = shlex.split(str(text))
    if not parts:
        return ""
    commands = SlurmCommands()
    name = parts[0].replace("-", "_")
    handler = getattr(commands, name, None)
    if handler is None or name not in names():
        raise ValueError(f"no such slurm command: {name}. Available: {', '.join(names())}")
    parameters = __import__("inspect").signature(handler).parameters
    kwargs, positional = {}, []
    index = 1
    while index < len(parts):
        token = parts[index]
        if token.startswith("--"):
            key = token[2:].replace("-", "_")
            if index + 1 < len(parts) and not parts[index + 1].startswith("--"):
                kwargs[key] = parts[index + 1]
                index += 2
                continue
            kwargs[key] = True
        elif "=" in token:
            key, _, value = token.partition("=")
            kwargs[key] = value
        else:
            positional.append(token)
        index += 1
    known = [key for key in parameters if key != "self" and key not in ("kwargs",)]
    for position, value in enumerate(positional):
        if position < len(known):
            kwargs[known[position]] = value
    clean = {}
    for key, value in kwargs.items():
        if key not in parameters and not any(p.kind == p.VAR_KEYWORD for p in parameters.values()):
            raise ValueError(f"{name} does not take '{key}'")
        clean[key] = _typed(value, parameters.get(key))
    return handler(**clean)


def _typed(value, parameter):
    """The console only ever produces strings: turn them into what the signature declares."""
    if parameter is None or not isinstance(value, str):
        return value
    annotation = parameter.annotation
    if annotation is bool:
        return value.strip().lower() in ("1", "true", "yes", "on")
    if annotation is int:
        return int(value)
    if annotation in (float,):
        return float(value)
    if annotation is list or str(annotation).startswith(("list", "typing.List")):
        return shlex.split(value)
    return value


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "run":
        run_spec(argv[1])
        return 0
    if not argv:
        print("usage: python -m matformer.slurm run <spec.json> | <command> [args]")
        print("commands:", ", ".join(names()))
        return 1
    print(line(" ".join(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
