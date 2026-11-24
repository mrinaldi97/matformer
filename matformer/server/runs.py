# matformer/server/runs.py
"""Runs served from the SQLite run database.

The server does not parse log files: it opens the databases a run may be in (the shared logs area,
the configured run roots, or an explicit file) and asks ``RunDB``. Listing, hyperparameters, metric
series, task summaries and records all come from the same read API the logger writes with.
``runs.watch`` polls the metrics table for new rows and pushes them as ``metric`` events, so the
GUI's live and historical plots have exactly one intake.
"""
import json
import threading
from pathlib import Path

from matformer.server import config
from matformer.server.core import broadcast, rpc
from matformer.utils import matformer_assets
from matformer.utils.run_db import RunDB, run_label

TAILERS = {}
LOCK = threading.Lock()


def database_files():
    """Every database a connected server may hold runs in: explicit roots, then the assets logs."""
    seen, files = set(), []
    candidates = [Path(root).expanduser() for root in config()["run_roots"]]
    root = matformer_assets.assets_root()
    if root is not None:
        candidates.append(root / "logs")
    for candidate in candidates:
        found = [candidate] if candidate.suffix == ".db" else (
            sorted(candidate.rglob("*.db")) if candidate.is_dir() else [])
        for path in found:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(resolved)
    return files


def _runs():
    """(database path, run row) for every run in every database, newest first."""
    found = []
    for path in database_files():
        try:
            with RunDB(path) as store:
                found += [(path, row) for row in store.runs()]
        except Exception:
            continue
    return sorted(found, key=lambda item: item[1]["started"] or "", reverse=True)


def _locate(ref, db=None):
    """(database path, run id) of one run, looked up by id/label in every database."""
    for path in database_files():
        if db and str(path) != str(Path(db).resolve()):
            continue
        try:
            with RunDB(path) as store:
                row = store.resolve(ref)
                if row is not None:
                    return path, row["id"]
        except Exception:
            continue
    raise KeyError(f"run '{ref}' not found")


def _public(path, row):
    sweep = json.loads(row["sweep"]) if row["sweep"] else None
    with RunDB(path) as store:
        total = store.parameters(row["id"]).get("training/total_steps")
        steps = store.last_step(row["id"])
    return {"id": run_label(row), "project": row["project"], "run": row["name"],
            "version": row["version"], "host": row["host"], "started": row["started"],
            "ended": row["ended"], "elapsed": row["elapsed"], "status": row["status"],
            "dir": row["save_dir"], "db": str(path), "sweep": (sweep or {}).get("name"),
            "total_steps": total or 0, "steps": steps}


@rpc("runs.list")
def runs_list(conn, p):
    """One row per run, filtered by project/run/sweep when asked, plus its status."""
    rows = [_public(path, row) for path, row in _runs()]
    if p.get("project"):
        rows = [row for row in rows if row["project"] == p["project"]]
    if p.get("run"):
        rows = [row for row in rows if p["run"] in row["run"]]
    if p.get("sweep"):
        rows = [row for row in rows if row["sweep"] == p["sweep"]]
    return rows


@rpc("runs.query")
def runs_query(conn, p):
    """Runs matching a project/sweep/run filter, optionally with the last value of one metric."""
    rows = runs_list(conn, p)
    if p.get("metric"):
        for row in rows:
            path, run_id = _locate(row["id"], row["db"])
            with RunDB(path) as store:
                points = store.metrics(run_id, keys=[p["metric"]]).get(p["metric"]) or []
            row[p["metric"]] = points[-1][1] if points else None
    return rows


@rpc("runs.config")
def runs_config(conn, p):
    """The flattened hyperparameters of a run."""
    path, run_id = _locate(p["run"], p.get("db"))
    with RunDB(path) as store:
        return store.parameters(run_id)


@rpc("runs.metrics")
def runs_metrics(conn, p):
    """Full metric history: {key: [[step, value], ...]}."""
    path, run_id = _locate(p["run"], p.get("db"))
    with RunDB(path) as store:
        return store.metrics(run_id, keys=p.get("keys"))


@rpc("runs.get")
def runs_get(conn, p):
    """Everything stored for one run: identity, parameters, metrics, task executions, summaries
    and record count. The tasks are the evaluations it ran, with their step and epoch."""
    path, run_id = _locate(p["run"], p.get("db"))
    with RunDB(path) as store:
        tasks = _executions(store, run_id)
        return {"run": _public(path, store.resolve(run_id)),
                "parameters": store.parameters(run_id),
                "metrics": store.metrics(run_id),
                "tasks": tasks,
                "summaries": store.summaries(run_id),
                "records": store.record_count(run_id)}


@rpc("runs.tasks")
def runs_tasks(conn, p):
    """The tasks executed inside a run, in order: task name, phase, step, epoch, duration and
    the full summary of each execution. Optionally filtered by task or by phase."""
    path, run_id = _locate(p["run"], p.get("db"))
    with RunDB(path) as store:
        return _executions(store, run_id, task=p.get("task"), phase=p.get("phase"))


def _executions(store, run_id, task=None, phase=None):
    rows = store.task_runs(run_id, task=task, phase=phase)
    for row in rows:
        row["summary"] = store.task_summary(row["id"])
        row["records"] = store.record_count(run_id, task_run=row["id"])
    return rows


@rpc("runs.summaries")
def runs_summaries(conn, p):
    """The end-of-task results of a run: {task: [{task_run_id, phase, step, epoch, key, value}]}."""
    path, run_id = _locate(p["run"], p.get("db"))
    with RunDB(path) as store:
        return store.summaries(run_id)


@rpc("runs.records")
def runs_records(conn, p):
    """The per-item records a task kept, paged. ``task_run`` selects one execution."""
    path, run_id = _locate(p["run"], p.get("db"))
    with RunDB(path) as store:
        return store.records(run_id, task=p.get("task"), task_run=p.get("task_run"),
                             limit=p.get("limit", 100), offset=p.get("offset", 0))


@rpc("runs.delete")
def runs_delete(conn, p):
    """Forget a run and everything stored with it. Files on disk are untouched."""
    path, run_id = _locate(p["run"], p.get("db"))
    with RunDB(path) as store:
        return {"removed": store.delete_run(run_id)}


@rpc("projects.delete")
def projects_delete(conn, p):
    """Forget every run of a project, across every database the server can see."""
    removed = 0
    for path in database_files():
        try:
            with RunDB(path) as store:
                removed += store.delete_project(p["project"])
        except Exception:
            continue
    return {"removed": removed}


@rpc("projects.list")
def projects_list(conn, p):
    """Every project that has at least one run."""
    return sorted({row["project"] for _path, row in _runs()})


@rpc("sweeps.list")
def sweeps_list(conn, p):
    """Every sweep name seen in the run databases."""
    names = set()
    for path in database_files():
        try:
            with RunDB(path) as store:
                for raw in store.sweeps():
                    names.add((json.loads(raw) or {}).get("name"))
        except Exception:
            continue
    return sorted(name for name in names if name)


@rpc("runs.plot")
def runs_plot(conn, p):
    """Plot one metric over several runs (or all runs of a project) into a PNG."""
    metric = p["metric"]
    chosen = p.get("runs") or [run_label(row) for _path, row in _runs()
                                if not p.get("project") or row["project"] == p["project"]]
    series = []
    for ref in chosen:
        try:
            path, run_id = _locate(ref, p.get("db"))
        except KeyError:
            continue
        with RunDB(path) as store:
            points = store.metrics(run_id, keys=[metric]).get(metric) or []
            if points:
                series.append((run_label(store.resolve(run_id)), points))
    if not series:
        raise ValueError(f"no run has the metric '{metric}'")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(figsize=(8, 4.5))
    for label, points in series:
        axes.plot([step for step, _value in points], [value for _step, value in points], label=label)
    axes.set_xlabel("step")
    axes.set_ylabel(metric)
    axes.set_title(metric)
    axes.legend()
    folder = Path(p["path"]).expanduser() if p.get("path") else \
        (matformer_assets.area("experiments") or Path.home() / ".matformer")
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{metric.replace('/', '_')}.png"
    figure.savefig(target, bbox_inches="tight")
    plt.close(figure)
    return {"path": str(target), "runs": len(series)}


@rpc("runs.watch")
def runs_watch(conn, p):
    """Subscribe to one run: the current history first, then every new metric row as an event."""
    path, run_id = _locate(p["run"], p.get("db"))
    with RunDB(path) as store:
        run = run_label(store.resolve(run_id))
        history = store.metrics(run_id)
    for key, points in history.items():
        conn.push("metric", {"run": run, "key": key, "points": points})
    with LOCK:
        if run not in TAILERS:
            TAILERS[run] = Tailer(run, path, run_id).start()


@rpc("runs.unwatch")
def runs_unwatch(conn, p):
    with LOCK:
        tailer = TAILERS.pop(p["run"], None)
    if tailer:
        tailer.stop_flag.set()


class Tailer(threading.Thread):
    """Polls the metrics table for one run and broadcasts the rows it has not sent yet."""

    def __init__(self, run, path, run_id):
        super().__init__(name=f"tail-{run}", daemon=True)
        self.run, self.stop_flag = run, threading.Event()
        self.store = RunDB(path)
        self.run_id = run_id
        latest = self.store.metrics_since(run_id)
        self.after = latest[-1]["id"] if latest else 0

    def run(self):
        try:
            while not self.stop_flag.wait(0.5):
                rows = self.store.metrics_since(self.run_id, self.after)
                for row in rows:
                    self.after = row["id"]
                    broadcast("metric", {"run": self.run, "key": row["key"],
                                         "points": [[row["step"], row["value"]]],
                                         "time": row["time"]})
        finally:
            self.store.close()


def latest_run(project, name, since=None):
    """(database path, run id, public label) of the newest run with this project and name,
    optionally only counting runs that started at or after ``since`` (ISO timestamp)."""
    for path, row in _runs():
        if row["project"] == project and row["name"] == name \
                and (since is None or (row["started"] or "") >= since):
            return path, row["id"], run_label(row)
    return None, None, None


def metrics_since(path, run_id, after=0):
    with RunDB(path) as store:
        return store.metrics_since(run_id, after)


# ── shared run catalog (sqlite): each server caches the whole fleet's history ────────────────────
@rpc("catalog.merge")
def catalog_merge(conn, p):
    """Adopt another machine's list of one server's runs: that server's rows are replaced."""
    from matformer.utils.catalog import Catalog
    return {"merged": Catalog().replace_server(p["server"], p["rows"])}


@rpc("catalog.upsert")
def catalog_upsert(conn, p):
    """Absorb a catalog snapshot: every run the caller knows about, each under the server it was
    seen on. Additive, a machine never forgets a run just because another one lacks it."""
    from matformer.utils.catalog import Catalog
    return {"merged": Catalog().upsert(p["rows"])}


@rpc("catalog.list")
def catalog_list(conn, p):
    from matformer.utils.catalog import Catalog
    return Catalog().all()
