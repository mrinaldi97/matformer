# matformer/server/memory.py
"""The memory & speed probe, asked from anywhere.

A probe is not a special kind of work: it is a training spec carrying a `probe` key, which the local
queue and SLURM already know how to run — so starting one is `train.start` with a few extra numbers,
and its answer arrives like any other job's, in its log. What is added here is the dialog's schema
(read from the trainer method that actually runs the probe, so the two can never drift apart) and the
report the trainer leaves in the run folder, which these commands collect into a table.

`memory_probe.json` holds the whole search: every attempt with the memory it peaked at, the measured
cost of a token, the batch size that is safe, and the throughput at that size."""
import json
from pathlib import Path

from matformer.server import REPO, config
from matformer.server.core import rpc
from matformer.utils.table import Table

PATTERNS = ("*/logs/*/memory_probe.json", "*/*/logs/*/memory_probe.json",
            "logs/*/memory_probe.json", "*/memory_probe.json")


@rpc("train.probeOptions")
def train_probe_options(conn, p):
    """What a probe can be told, with its defaults, read from the trainer method that runs it: the GUI
    adds these to the switches it already asks `train.options` for, so dialog and code cannot drift."""
    from matformer.training import launch
    return {"fields": launch.probe_fields()}


def _found(limit: int = 40, root: str = None) -> list:
    """Probe reports under every configured run root and under the repo, newest first.
    ``root`` narrows the scan to one directory, which is what a run in /tmp needs."""
    seen, paths = set(), []
    roots = [root] if root else list(config()["run_roots"]) + [str(REPO)]
    for root in roots:
        base = Path(root).expanduser()
        for pattern in PATTERNS:
            for path in base.glob(pattern):
                if path not in seen:
                    seen.add(path)
                    paths.append(path)
    return sorted(paths, key=lambda one: one.stat().st_mtime, reverse=True)[:limit]


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


@rpc("memory.reports")
def memory_reports(conn, p):
    """One row per saved probe: the batch size it measured, the memory it peaked at, and how fast the
    run went there. `root` narrows the search, `limit` caps it."""
    rows = []
    for path in _found(int(p.get("limit") or 40), p.get("root")):
        report = _read(path)
        if not report:
            continue
        rows.append({"at": report.get("at", ""), "run": path.parent.name,
                     "gpu": report.get("gpu", ""), "rows": report.get("rows_per_rank"),
                     "tokens": report.get("tokens_per_rank"),
                     "peak MiB": report.get("peak_mib"), "vram MiB": report.get("vram_mib"),
                     "tok/s": round(report.get("speed_tokens_per_second") or 0),
                     "MFU %": round(report.get("speed_mfu_percent") or 0, 1),
                     "path": str(path)})
    return Table(["at", "run", "gpu", "rows", "tokens", "peak MiB", "vram MiB", "tok/s", "MFU %",
                  "path"], rows)


@rpc("memory.report")
def memory_report(conn, p):
    """One probe, as rows: what the run was (device, precision, the memory it searched) and every
    attempt of the search with the memory it peaked at. `path` picks a report; without it, the most
    recent one. `json` returns the file as it stands."""
    path = Path(p["path"]).expanduser() if p.get("path") else (_found(1, p.get("root")) or [None])[0]
    if path is not None and path.is_dir():
        path = path / "memory_probe.json"
    if path is None or not path.is_file():
        raise ValueError("no memory_probe.json under the configured run roots; pass `path` or `root`")
    report = _read(path)
    if p.get("json"):
        return report
    rows = [{"metric": key, "value": value} for key, value in report.items() if key != "probes"]
    rows += [{"metric": f"attempt at {one.get('rows')} rows", "value": f"{one.get('peak_mib')} MiB · "
                        f"{'fits' if one.get('fits') else 'OOM'}"}
             for one in report.get("probes", [])]
    return Table(["metric", "value"], rows)
