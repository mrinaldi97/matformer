# matformer/utils/run_db.py
"""The run database: one SQLite file holding every run Matformer produces.

A training writes here what it used to write to loose files: identity (runs), hyperparameters
(parameters, one row per key), step metrics (metrics), task summaries (task_summaries) and, when
asked, one row per task item (task_records). The logger accumulates rows in memory and commits in
batches, so a log step does not turn into a disk round-trip; WAL keeps the reader (server, GUI,
scripts) live while the writer works.

The same class is the read API: the server and the RPC commands talk to the database through it,
never through hand-rolled SQL. Configure the location with ``training.log_db``; the default is
``$LOG_ASSETS/matformer.db`` (or ``<save_dir>/logs/matformer.db`` without MATFORMER_ASSETS).
"""
import json
import math
import sqlite3
from datetime import datetime
from pathlib import Path

from matformer.utils import matformer_assets

DB_NAME = "matformer.db"
SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    project TEXT NOT NULL,
    name TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    checkpoint TEXT,
    host TEXT,
    save_dir TEXT,
    started TEXT,
    ended TEXT,
    elapsed REAL,
    status TEXT DEFAULT 'running',
    sweep TEXT,
    UNIQUE(project, name, version)
);
CREATE TABLE IF NOT EXISTS parameters (
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    value TEXT,
    PRIMARY KEY(run_id, key)
);
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    step INTEGER,
    epoch INTEGER,
    key TEXT NOT NULL,
    value REAL,
    time REAL
);
CREATE INDEX IF NOT EXISTS metrics_key ON metrics(run_id, key, step);
CREATE TABLE IF NOT EXISTS task_runs (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    task TEXT NOT NULL,
    phase TEXT NOT NULL DEFAULT 'training',
    step INTEGER,
    epoch INTEGER,
    started TEXT,
    ended TEXT,
    elapsed REAL
);
CREATE INDEX IF NOT EXISTS task_runs_run ON task_runs(run_id, task, phase);
CREATE TABLE IF NOT EXISTS task_summaries (
    id INTEGER PRIMARY KEY,
    task_run_id INTEGER NOT NULL REFERENCES task_runs(id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    value TEXT,
    time REAL
);
CREATE INDEX IF NOT EXISTS summaries_task ON task_summaries(task_run_id);
CREATE TABLE IF NOT EXISTS task_records (
    id INTEGER PRIMARY KEY,
    task_run_id INTEGER NOT NULL REFERENCES task_runs(id) ON DELETE CASCADE,
    item INTEGER NOT NULL,
    data TEXT NOT NULL,
    time REAL
);
CREATE INDEX IF NOT EXISTS records_task ON task_records(task_run_id);
"""


def database_path(training=None) -> Path:
    """Where a training's database lives. ``training.log_db`` wins (assets tokens expanded,
    relative paths under ``save_dir``); otherwise the shared logs area, or ``save_dir/logs``."""
    explicit = getattr(training, "log_db", None) if training is not None else None
    if explicit:
        path = Path(matformer_assets.expand(str(explicit))).expanduser()
        if not path.is_absolute() and training is not None:
            path = Path(training.save_dir) / path
        return path
    root = matformer_assets.assets_root(create=True)
    if root is not None:
        return root / "logs" / DB_NAME
    base = Path(training.save_dir) / "logs" if training is not None else Path.cwd() / "logs"
    return base / DB_NAME


def jsonable(value):
    """Tensors, arrays, numpy scalars and nested containers -> plain JSON-ready structures."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    return value


def to_json(value) -> str:
    return json.dumps(jsonable(value), default=str, ensure_ascii=False)


def as_number(value):
    """A plottable number, or None: booleans become 0/1, strings and non-finite floats are skipped."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and math.isfinite(value):
        return value
    return None


def run_label(row) -> str:
    """Stable public id of a run: ``project/name``, with ``@vN`` when it is not the first version."""
    version = row["version"] if "version" in row.keys() else None
    suffix = "" if version in (None, 1) else f"@v{version}"
    return f"{row['project']}/{row['name']}{suffix}"


class RunDB:
    """Writer of a training and reader of every run in one database file."""

    def __init__(self, path):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(self.path, timeout=5, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA synchronous=NORMAL")
        self.con.execute("PRAGMA temp_store=MEMORY")
        self.con.execute("PRAGMA cache_size=-64000")
        self.con.execute("PRAGMA foreign_keys=ON")
        self.con.executescript(SCHEMA)
        self.con.commit()

    # ── writing ──────────────────────────────────────────────────────────────────
    def start_run(self, meta: dict) -> int:
        """Insert the run, or reopen it when the same project/name/version resumes."""
        now = meta.get("started") or datetime.now().isoformat(timespec="seconds")
        version = meta.get("version") or 1
        row = self.con.execute(
            "SELECT id, started FROM runs WHERE project=? AND name=? AND version=?",
            (meta["project"], meta["name"], version)).fetchone()
        if row is not None:
            self.con.execute(
                "UPDATE runs SET checkpoint=?, host=?, save_dir=?, sweep=?, status='running', "
                "ended=NULL WHERE id=?",
                (meta.get("checkpoint"), meta.get("host"), meta.get("save_dir"),
                 to_json(meta.get("sweep")), row["id"]))
            self.con.commit()
            return row["id"]
        cursor = self.con.execute(
            "INSERT INTO runs(project, name, version, checkpoint, host, save_dir, started, "
            "status, sweep) VALUES (?,?,?,?,?,?,?, 'running', ?)",
            (meta["project"], meta["name"], version, meta.get("checkpoint"),
             meta.get("host"), meta.get("save_dir"), now, to_json(meta.get("sweep"))))
        self.con.commit()
        return cursor.lastrowid

    def finish_run(self, run_id: int, status: str = "finished") -> None:
        row = self.con.execute("SELECT started FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            return
        ended = datetime.now()
        try:
            elapsed = (ended - datetime.fromisoformat(row["started"])).total_seconds()
        except (TypeError, ValueError):
            elapsed = None
        self.con.execute("UPDATE runs SET ended=?, elapsed=?, status=? WHERE id=?",
                         (ended.isoformat(timespec="seconds"), elapsed, status, run_id))
        self.con.commit()

    def add_parameters(self, run_id: int, values: dict) -> None:
        self.con.executemany(
            "INSERT OR REPLACE INTO parameters(run_id, key, value) VALUES (?,?,?)",
            [(run_id, str(key), to_json(value)) for key, value in values.items()])
        self.con.commit()

    def add_metrics(self, rows) -> None:
        self.con.executemany(
            "INSERT INTO metrics(run_id, step, epoch, key, value, time) VALUES (?,?,?,?,?,?)", rows)
        self.con.commit()

    def start_task(self, run_id, task, phase="training", step=None, epoch=None) -> int:
        """Open one task execution: every summary and record it produces points at this row."""
        cursor = self.con.execute(
            "INSERT INTO task_runs(run_id, task, phase, step, epoch, started) "
            "VALUES (?,?,?,?,?,?)",
            (run_id, str(task), phase, step, epoch,
             datetime.now().isoformat(timespec="seconds")))
        self.con.commit()
        return cursor.lastrowid

    def finish_task(self, task_run_id: int) -> None:
        row = self.con.execute("SELECT started FROM task_runs WHERE id=?",
                               (task_run_id,)).fetchone()
        if row is None:
            return
        ended = datetime.now()
        try:
            elapsed = (ended - datetime.fromisoformat(row["started"])).total_seconds()
        except (TypeError, ValueError):
            elapsed = None
        self.con.execute("UPDATE task_runs SET ended=?, elapsed=? WHERE id=?",
                         (ended.isoformat(timespec="seconds"), elapsed, task_run_id))
        self.con.commit()

    def add_summaries(self, rows) -> None:
        self.con.executemany(
            "INSERT INTO task_summaries(task_run_id, key, value, time) VALUES (?,?,?,?)", rows)
        self.con.commit()

    def add_records(self, rows) -> None:
        self.con.executemany(
            "INSERT INTO task_records(task_run_id, item, data, time) VALUES (?,?,?,?)", rows)
        self.con.commit()

    # ── reading ──────────────────────────────────────────────────────────────────
    def resolve(self, ref):
        """A run row from an id (int), a label ('project/name[@vN]'), a name, or a dict."""
        if isinstance(ref, dict):
            ref = ref.get("id") or ref.get("run") or ref.get("label") or ref.get("name")
        if isinstance(ref, int) or (isinstance(ref, str) and ref.isdigit()):
            return self.con.execute("SELECT * FROM runs WHERE id=?", (int(ref),)).fetchone()
        text = str(ref)
        version = None
        if "@v" in text:
            text, _, version = text.partition("@v")
            version = int(version) if version.isdigit() else None
        project, name = (text.split("/", 1) if "/" in text else (None, text))
        sql = "SELECT * FROM runs WHERE name=?"
        args = [name]
        if project is not None:
            sql += " AND project=?"
            args.append(project)
        if version is not None:
            sql += " AND version=?"
            args.append(version)
        return self.con.execute(sql + " ORDER BY started DESC LIMIT 1", args).fetchone()

    def runs(self):
        return self.con.execute("SELECT * FROM runs ORDER BY started DESC").fetchall()

    def parameters(self, ref) -> dict:
        row = self.resolve(ref)
        if row is None:
            return {}
        return {item["key"]: json.loads(item["value"]) if item["value"] is not None else None
                for item in self.con.execute(
                    "SELECT key, value FROM parameters WHERE run_id=?", (row["id"],))}

    def metrics(self, ref, keys=None) -> dict:
        """{key: [[step, value], ...]}; ``keys`` narrows it to a list or a glob-free name."""
        row = self.resolve(ref)
        if row is None:
            return {}
        sql, args = "SELECT key, step, value FROM metrics WHERE run_id=?", [row["id"]]
        if keys:
            wanted = [keys] if isinstance(keys, str) else list(keys)
            sql += f" AND key IN ({','.join('?' * len(wanted))})"
            args += wanted
        out = {}
        for item in self.con.execute(sql + " ORDER BY step, id", args):
            out.setdefault(item["key"], []).append([item["step"], item["value"]])
        return out

    def metric_names(self, ref):
        row = self.resolve(ref)
        if row is None:
            return []
        return [item["key"] for item in self.con.execute(
            "SELECT DISTINCT key FROM metrics WHERE run_id=? ORDER BY key", (row["id"],))]

    def last_step(self, ref):
        row = self.resolve(ref)
        if row is None:
            return None
        found = self.con.execute("SELECT MAX(step) AS step FROM metrics WHERE run_id=?",
                                 (row["id"],)).fetchone()
        return found["step"] if found else None

    def metrics_since(self, ref, after: int = 0):
        """Rows with id greater than ``after``, for live tailing; each row carries its id."""
        row = self.resolve(ref)
        if row is None:
            return []
        return self.con.execute(
            "SELECT id, step, key, value, time FROM metrics WHERE run_id=? AND id>? ORDER BY id",
            (row["id"], int(after))).fetchall()

    def task_runs(self, ref, task=None, phase=None):
        """Every task execution of a run, in order: which task, at which step/epoch, how long."""
        row = self.resolve(ref)
        if row is None:
            return []
        sql, args = "SELECT * FROM task_runs WHERE run_id=?", [row["id"]]
        if task:
            sql += " AND task=?"
            args.append(task)
        if phase:
            sql += " AND phase=?"
            args.append(phase)
        return [dict(item) for item in self.con.execute(sql + " ORDER BY id", args)]

    def task_summary(self, task_run_id) -> dict:
        """The summary of one task execution, as a plain {key: value} dict."""
        return {item["key"]: json.loads(item["value"]) if item["value"] is not None else None
                for item in self.con.execute(
                    "SELECT key, value FROM task_summaries WHERE task_run_id=? ORDER BY id",
                    (int(task_run_id),))}

    def summaries(self, ref) -> dict:
        """{task: [{task_run_id, phase, step, epoch, key, value}, ...]}, all executions together."""
        row = self.resolve(ref)
        if row is None:
            return {}
        out = {}
        for item in self.con.execute(
                "SELECT t.id, t.task, t.phase, t.step, t.epoch, s.key, s.value "
                "FROM task_summaries s JOIN task_runs t ON t.id=s.task_run_id "
                "WHERE t.run_id=? ORDER BY s.id", (row["id"],)):
            out.setdefault(item["task"], []).append(
                {"task_run_id": item["id"], "phase": item["phase"], "step": item["step"],
                 "epoch": item["epoch"], "key": item["key"],
                 "value": json.loads(item["value"]) if item["value"] is not None else None})
        return out

    def records(self, ref, task=None, task_run=None, limit=100, offset=0):
        row = self.resolve(ref)
        if row is None:
            return []
        sql = ("SELECT t.task, t.phase, t.step, t.epoch, t.id AS task_run_id, r.item, r.data "
               "FROM task_records r JOIN task_runs t ON t.id=r.task_run_id WHERE t.run_id=?")
        args = [row["id"]]
        if task:
            sql += " AND t.task=?"
            args.append(task)
        if task_run is not None:
            sql += " AND t.id=?"
            args.append(int(task_run))
        args += [int(limit), int(offset)]
        return [{"task": item["task"], "phase": item["phase"], "step": item["step"],
                 "epoch": item["epoch"], "task_run_id": item["task_run_id"],
                 "item": item["item"], "data": json.loads(item["data"])}
                for item in self.con.execute(sql + " ORDER BY r.id LIMIT ? OFFSET ?", args)]

    def record_count(self, ref, task_run=None) -> int:
        row = self.resolve(ref)
        if row is None:
            return 0
        if task_run is None:
            found = self.con.execute(
                "SELECT COUNT(*) AS n FROM task_records r "
                "JOIN task_runs t ON t.id=r.task_run_id WHERE t.run_id=?", (row["id"],))
        else:
            found = self.con.execute(
                "SELECT COUNT(*) AS n FROM task_records WHERE task_run_id=?", (int(task_run),))
        return found.fetchone()["n"]

    # ── housekeeping ─────────────────────────────────────────────────────────────
    def delete_run(self, ref) -> bool:
        row = self.resolve(ref)
        if row is None:
            return False
        self.con.execute("DELETE FROM runs WHERE id=?", (row["id"],))
        self.con.commit()
        return True

    def delete_project(self, project: str) -> int:
        cursor = self.con.execute("DELETE FROM runs WHERE project=?", (project,))
        self.con.commit()
        return cursor.rowcount

    def projects(self):
        return [item["project"] for item in self.con.execute(
            "SELECT DISTINCT project FROM runs ORDER BY project")]

    def sweeps(self):
        return [item["sweep"] for item in self.con.execute(
            "SELECT sweep FROM runs WHERE sweep IS NOT NULL AND sweep != 'null'")]

    def close(self):
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
