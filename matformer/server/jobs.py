# matformer/server/jobs.py
"""Training jobs: the server owns a queue and runs one training at a time (one box, one GPU set).
What it runs is the framework's own entry point — `matformer.training.launch.execute(spec)` — in a
fresh process: a training owns a CUDA context, spawns its own DDP ranks and must be killable without
touching the server, so the process boundary is real. But it is entered as a library function with a
plain dict, never as a command line, and never by spawning a repository script.

Progress comes from the SQLite run database the run writes, so the queue needs nothing from the
process but its exit code. Job events are broadcast to every GUI: {id, kind, name, project, status,
progress, step}; `jobs.log` returns the terminal output on demand."""
import json
import multiprocessing
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from matformer.server import REPO
from matformer.server.core import broadcast, rpc
from matformer.server.runs import latest_run, metrics_since

JOBS = {}                              # id -> job dict
QUEUE = []                             # ids waiting for the GPU
WORKER, CURRENT, MUTEX = None, None, threading.Condition()
LOG_DIR = REPO / ".matformer_server" / "jobs"
CONTEXT = multiprocessing.get_context("spawn")


def _ensure_worker():
    global WORKER
    if WORKER is None or not WORKER.is_alive():
        WORKER = threading.Thread(target=_worker, name="job-queue", daemon=True).start()


def _worker():
    global CURRENT
    while True:
        with MUTEX:
            while not QUEUE:
                MUTEX.wait(0.5)
            CURRENT = job_id = QUEUE.pop(0)
        _run_job(JOBS[job_id])
        with MUTEX:
            CURRENT = None


def _child(spec: dict, log_path: str):
    """The job's process: its own session (so a whole DDP rank tree dies with it), its output in the
    job log, the framework in charge from here on. It never hosts a server of its own. `spec['_entry']`
    is the module:function that knows how to do the work."""
    os.environ["MATFORMER_SERVER"] = "0"
    runtime = spec.pop("_runtime", None)
    if runtime:
        os.environ["MATFORMER_CONTROL"] = os.path.join(runtime, "control")
        os.environ["MATFORMER_STATUS"] = os.path.join(runtime, "heartbeat.json")
    os.setsid()
    with open(log_path, "a", buffering=1) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        entry = spec.get("_entry", "matformer.training.launch:execute")
        if entry == "matformer.training.launch:execute":
            # Re-enter through a module, not through this server process: Fabric's multi-GPU
            # launcher re-executes __main__ in its children, and __main__ here is the server.
            spec.pop("_entry", None)
            os.environ["MATFORMER_JOB_SPEC"] = json.dumps(spec, default=str)
            os.execv(sys.executable, [sys.executable, "-m", "matformer.training.launch"])
        module, _, function = entry.partition(":")
        getattr(__import__(module, fromlist=[function]), function)(spec)


def enqueue(kind, name, project, spec, entry="matformer.training.launch:execute") -> str:
    """Put one piece of framework work on the queue — a training, an MDAT ingestion — and return its
    id. A spec is a plain dict, handed unchanged to `entry` in a fresh process; status, logs, abort
    and the broadcast of `job` events are the queue's business, whatever the work is."""
    _ensure_worker()
    job_id = f"{name}-{datetime.now():%H%M%S}"
    job = {"id": job_id, "kind": kind, "name": name, "project": project,
           "spec": dict(spec, _entry=entry),
           "status": "queued", "progress": 0, "step": None,
           "started": datetime.now().isoformat(timespec="seconds")}
    JOBS[job_id] = job
    with MUTEX:
        QUEUE.append(job_id)
        MUTEX.notify()
    broadcast("job", _public(job))
    threading.Thread(target=_monitor, args=(job,), daemon=True).start()
    return job_id


def _run_job(job):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    job["log"] = str(LOG_DIR / f"{job['id']}.log")
    runtime = LOG_DIR / job["id"]
    runtime.mkdir(parents=True, exist_ok=True)
    job["runtime"] = str(runtime)
    spec = dict(job["spec"], _runtime=str(runtime))
    _set(job, status="starting")
    proc = CONTEXT.Process(target=_child, args=(spec, job["log"]),
                           name=f"job-{job['id']}")
    proc.start()
    job["proc"] = proc
    _set(job, status="running")
    if job["status"] == "cancelling":         # aborted before the OS even started it
        _kill(proc)
    proc.join()
    ret = proc.exitcode
    job["proc"] = None
    if job["status"] not in ("cancelling", "cancelled"):
        if ret == 0:
            _set(job, status="done", exit=ret, progress=1)
        else:
            _set(job, status="failed", exit=ret)     # keep the last real progress: where it died
    else:
        _set(job, status="cancelled", exit=ret)


def _kill(proc):
    """SIGTERM to the whole session: a DDP run leaves ranks behind if only the parent is asked."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass


@rpc("train.options")
def train_options(conn, p):
    """The switches a training accepts, read from the launcher: the GUI builds its queue dialog
    from here, so what the terminal can do and what the dialog offers never drift apart. On a
    submission host the dialog also gets where to run it, generated by the SLURM interpreter."""
    from matformer.spec_fields import spec_fields
    # The cluster fields are a separate, potentially slow RPC (`slurm.fields`): the queue dialog
    # opens at once with the local switches and merges them when they arrive.
    return {"fields": spec_fields()}


@rpc("train.start")
def train_start(conn, p):
    """Queue a training. The params *are* the launch spec — {config: [YAML], overrides: [...],
    options: {device_count, num_nodes, checkpoint, compile, …}, sweep, gpus, name, project} —
    the same dict the command line builds, so both paths say the same thing; a sweep declared in
    the YAMLs runs as one job, one run per point. With backend=cluster that spec is submitted to
    SLURM instead of run here. Returns {run: id}."""
    configs = p.get("config") or []
    configs = [configs] if isinstance(configs, str) else list(configs)
    name = p.get("name") or (Path(configs[-1]).stem if configs else "training")
    project = p.get("project", "server")
    spec = {key: value for key, value in p.items() if key not in ("name", "project")}
    spec["config"] = configs
    spec["overrides"] = list(spec.get("overrides") or []) + [
        f"training.run_name={name}", f"training.project_name={project}"]
    if slurm_requested(p):
        from matformer import slurm
        return slurm.submit_training(spec, p, name=name)
    return {"run": enqueue("training", name, project, spec)}


def slurm_requested(params: dict) -> bool:
    """The job belongs to SLURM when the request says so, or when a partition was named and this
    host can talk to a controller. Forms may nest their answers under `options`."""
    from matformer import slurm
    flat = dict(params or {})
    flat.update(flat.get("options") or {})
    backend = str(flat.get("backend") or "").lower()
    if backend in ("cluster", "slurm"):
        return True
    if backend:
        return False
    return bool(flat.get("partition")) and slurm.available()


def cluster_rows() -> list:
    """The cluster's jobs, in the shape the job dock already understands: one table for local and
    remote trainings, whichever machine is actually running them."""
    from matformer import slurm
    states = {"RUNNING": "running", "PENDING": "queued", "CONFIGURING": "queued",
              "COMPLETED": "done", "DONE": "done", "CANCELLED": "cancelled",
              "FAILED": "failed", "TIMEOUT": "failed", "NODE_FAIL": "failed",
              "PREEMPTED": "failed", "GONE": "expired", "BOOT_FAIL": "failed", "DEADLINE": "failed"}
    rows = []
    for one in slurm.status():
        total, step = one.get("total"), one.get("step")
        rows.append({"id": one["matformer_id"], "kind": "training", "name": one["name"],
                     "project": one["partition"] or "cluster", "cluster": True,
                     "status": states.get(str(one.get("state", "")).upper(), "queued"),
                     "state": one.get("state"), "slurm_id": one.get("slurm_id"),
                     "step": step, "progress": (step / total) if (step and total) else 0,
                     "seconds_left": one.get("seconds_left"), "log": one.get("log"),
                     "nodes": one.get("nodes"), "gpus": one.get("gpus"),
                     "venv": one.get("venv"), "saved_step": one.get("saved_step")})
    return rows


@rpc("train.probe")
def train_probe(conn, p):
    """Measure the largest batch a config fits, without training. Same as `train.start` with a
    `probe` object, but easier to discover: ``{config, options, overrides, probe: {fill, steps,
    max_rows, apply}, name, project}``. The report lands in ``memory_probe.json`` and is read by
    `memory.reports`/`memory.report`; `apply` also trains at the measured size."""
    spec = dict(p)
    spec["probe"] = p.get("probe") or {}
    return train_start(conn, spec)


@rpc("train.sweep")
def train_sweep(conn, p):
    """What a set of configs declares, without running any of it: the points of the sweep, one per
    line. The same `build_configs` a real launch goes through, so the preview and the expansion can
    not drift apart, and no model is built to answer it."""
    from matformer.training import launch
    configs = launch.build_configs(p.get("config") or [], p.get("overrides") or [],
                                   p.get("options") or {})
    points = configs.sweep_points(p.get("sweep"))
    return {"runs": len(points),
            "dimensions": sorted({path for point in points for path in point.values}),
            "points": [dict(point.asdict(), describe=point.describe()) for point in points]}


@rpc("train.abort")
def train_abort(conn, p):
    """Kill a training cleanly: SIGTERM to its session; queued jobs just get dropped. A cluster job
    is stopped where it runs, and asked to checkpoint on the way out."""
    if p["id"] not in JOBS:
        from matformer import slurm
        try:
            return slurm.cancel(p["id"])
        except ValueError:
            pass
    job = JOBS[p["id"]]
    if job["status"] == "queued":
        with MUTEX:
            QUEUE.remove(p["id"])
        _set(job, status="cancelled")
    elif job.get("proc"):
        _set(job, status="cancelling")
        _kill(job["proc"])
    return {"msg": f"aborting {job['id']}"}


@rpc("train.command")
def train_command(conn, p):
    """Send one command to a running training without interrupting it.

    Params: ``{id, cmd, ...}``. ``cmd`` is one of ``save``, ``stop``, ``save_and_stop``,
    ``pause``, ``resume``, ``set`` (with ``field``/``value``), ``flush`` or ``task`` (with
    ``name``/``type``/``fields``). A local job receives it through the same mailbox the SLURM
    wrapper uses; a cluster job is forwarded to its wrapper there. Commands are read at the next
    optimizer-step boundary, so a checkpoint and a DDP collective are never split.
    """
    job_id = p["id"]
    payload = {key: value for key, value in p.items() if key != "id"}
    payload.setdefault("cmd", "save")
    payload["at"] = payload.get("at") or datetime.now().isoformat(timespec="microseconds")
    job = JOBS.get(job_id)
    if job is None:
        from matformer import slurm
        try:
            forwarded = slurm.command(job_id, payload.pop("cmd"), **{
                key: value for key, value in payload.items() if key != "at"})
            return {**forwarded, "remote": True}
        except ValueError:
            raise KeyError(f"unknown job '{job_id}'")
    runtime = job.get("runtime")
    if not runtime:
        raise RuntimeError(f"job '{job_id}' has no control mailbox (already finished?)")
    path = Path(runtime) / "control"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, default=str))
    os.replace(temporary, path)
    return {"msg": f"{payload['cmd']} sent to {job_id}", "control": str(path)}


@rpc("train.heartbeat")
def train_heartbeat(conn, p):
    """What a running training knows about itself right now: heartbeat, last saved checkpoint,
    and the answer to the last command. Read-only; no command is written."""
    job_id = p["id"]
    job = JOBS.get(job_id)
    if job is None:
        from matformer import slurm
        try:
            entry = slurm.resolve(job_id)
            runtime = slurm.job_dir(entry["matformer_id"])
            return _runtime_files(runtime)
        except (ValueError, KeyError):
            raise KeyError(f"unknown job '{job_id}'")
    runtime = job.get("runtime")
    if not runtime:
        return {"job": _public(job), "heartbeat": {}, "saved": {}, "result": {}}
    return {"job": _public(job), **_runtime_files(Path(runtime))}


@rpc("train.pause")
def train_pause(conn, p):
    """Pause at the next optimizer-step boundary; commands still work while paused."""
    return train_command(conn, {**p, "cmd": "pause"})


@rpc("train.resume")
def train_resume(conn, p):
    """Continue a paused run from exactly where it stopped."""
    return train_command(conn, {**p, "cmd": "resume"})


@rpc("train.save")
def train_save(conn, p):
    """Write a checkpoint now and keep training."""
    return train_command(conn, {**p, "cmd": "save"})


@rpc("train.stop")
def train_stop(conn, p):
    """Finish cleanly: checkpoint by default, then stop at the step boundary."""
    return train_command(conn, {**p, "cmd": "save_and_stop" if p.get("save", True) else "stop"})


@rpc("train.set")
def train_set(conn, p):
    """Change one live training value: lr, weight_decay, max_steps, save_every_n_steps, ..."""
    return train_command(conn, {**p, "cmd": "set", "field": p["field"], "value": p["value"]})


@rpc("train.task")
def train_task(conn, p):
    """Trigger a task that was not declared in the configs, without interrupting training."""
    return train_command(conn, {**p, "cmd": "task"})


def _runtime_files(runtime: Path) -> dict:
    def read(name):
        path = runtime / name
        try:
            return json.loads(path.read_text()) if path.is_file() else {}
        except (OSError, ValueError):
            return {}
    return {"heartbeat": read("heartbeat.json"), "saved": read("saved.json"),
            "result": read("control_result.json"), "control": read("control")}


@rpc("jobs.list")
def jobs_list(conn, p):
    """Every job this server knows (running, queued, finished), plus the trainings SLURM is running
    for it on other machines."""
    rows = [_public(j) for j in JOBS.values()]
    from matformer import slurm
    if slurm.available():
        try:
            rows += cluster_rows()
        except Exception:
            pass                      # a controller that does not answer cannot hide the local jobs
    return rows


@rpc("jobs.watch")
def jobs_watch(conn, p):
    """Subscribe: replay the current jobs, then push every change (broadcast covers all conns)."""
    for job in JOBS.values():
        conn.push("job", _public(job))


@rpc("jobs.log")
def jobs_log(conn, p):
    """Terminal output of a job (tail), for the Training page console. A cluster job's log is the
    wrapper's own file on the shared filesystem."""
    log = JOBS.get(p["id"], {}).get("log")
    if not log or not Path(log).exists():
        from matformer import slurm
        if slurm.available():
            try:
                return {"text": slurm.tail(p["id"], lines=int(p.get("lines", 400)))}
            except ValueError:
                pass
        return {"text": ""}
    data = Path(log).read_bytes()[-(p.get("bytes", 64_000)):]
    return {"text": data.decode("utf-8", errors="replace")}


def _public(job):
    return {k: v for k, v in job.items() if k not in ("proc", "spec")}
def _set(job, **fields):
    job.update(fields, updated=time.time())
    broadcast("job", _public(job))


def _monitor(job):
    """While the job runs, poll its row in the run database: step, progress, and metric events so a
    GUI watching 'runs' sees the live curve without knowing about jobs at all.

    The monitor starts when the job is queued, so it must wait for the worker instead of returning
    at once: an early return is why a running job used to show progress 0 until it finished."""
    from matformer.utils.run_db import RunDB

    seen, run_id, label, total, path = 0, None, None, 0, None
    while job["status"] not in ("done", "failed", "cancelled"):
        if job["status"] in ("starting", "running", "cancelling"):
            if run_id is None:
                path, run_id, label = latest_run(job["project"], job["name"],
                                                 since=job["started"])
                if run_id is not None:
                    try:
                        with RunDB(path) as store:
                            total = int(store.parameters(run_id).get("training/total_steps") or 0)
                    except Exception:
                        total = 0
            if run_id is not None:
                rows = metrics_since(path, run_id, seen)
                grouped = {}
                last_step = job.get("step")
                for row in rows:
                    seen = row["id"]
                    last_step = row["step"]
                    grouped.setdefault(row["key"], []).append([row["step"], row["value"]])
                if rows:
                    progress = min(0.99, last_step / total) if total else job.get("progress", 0)
                    _set(job, step=last_step, progress=progress)
                for key, points in grouped.items():
                    broadcast("metric", {"run": label, "key": key, "points": points,
                                         "time": rows[-1]["time"] if rows else None})
        time.sleep(0.5)
    _set(job, progress=1 if job["status"] == "done" else job.get("progress", 0))
