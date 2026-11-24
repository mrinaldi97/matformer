# matformer/training/sqlite_logger.py
"""The default logger backend: every run goes into one SQLite database.

Writes are buffered and committed in batches (and at most once per second), so a log step costs a
list append, not a disk round-trip. The database file is shared by all runs; its location defaults
to ``$LOG_ASSETS/matformer.db`` and can be changed with ``training.log_db``. See
``matformer/utils/run_db.py`` for the schema and the read API.
"""
import socket
import time
from datetime import datetime
from pathlib import Path

from matformer.matformer_registry import registry
from matformer.training.logger import flatten_configs
from matformer.utils.run_db import RunDB, as_number, database_path, to_json


def run_dir_for(training_configs) -> Path:
    """Folder of a run's loose artifacts (the memory probe): inside the checkpoint directory."""
    return (Path(training_configs.save_dir) / "logs"
            / f"{training_configs.run_name}_v{training_configs.version}")


@registry.register("logger", "sqlite", "default")
class SQLiteLogger:
    """Log backend that gathers rows in memory and commits them in batches, on rank 0 only."""

    FLUSH_ROWS = 1024
    FLUSH_SECONDS = 1.0

    def __init__(self, fabric=None, configs=None, **_):
        self._zero = fabric is None or fabric.is_global_zero
        self._db, self._run_id = None, None
        self._metrics, self._summaries, self._records = [], [], []
        self._last_flush = time.monotonic()
        if configs is not None and self._zero:
            self._open(configs)

    @staticmethod
    def build_fabric_logger(training_configs):
        return None

    def _open(self, configs):
        training = configs.training
        self._db = RunDB(database_path(training))
        point = getattr(configs, "active_sweep", None)
        self._run_id = self._db.start_run({
            "project": training.get("project_name", "matformer"),
            "name": training.run_name,
            "version": training.version,
            "checkpoint": training.checkpoint_name,
            "save_dir": str(Path(training.save_dir).resolve()),
            "host": socket.gethostname(),
            "started": datetime.now().isoformat(timespec="seconds"),
            "sweep": point.asdict() if point is not None else None,
        })
        self.log_hyperparameters(configs)

    def log(self, metrics, step):
        if self._db is None:
            return
        # A scoped logger travels with its own '<prefix>/step': that is the x-axis of its metrics.
        scoped_step = next((as_number(value) for key, value in metrics.items()
                            if key.endswith("/step") and as_number(value) is not None), None)
        step = scoped_step if scoped_step is not None else step
        now = time.time()
        for key, value in metrics.items():
            number = as_number(value)
            if number is not None:
                self._metrics.append((self._run_id, step, None, str(key), number, now))
        self._maybe_flush()

    def start_task(self, task, phase="training", step=None, epoch=None):
        """Open a task execution: its summaries and records are linked to this training run."""
        return self._db.start_task(self._run_id, task, phase=phase, step=step, epoch=epoch) \
            if self._db is not None else None

    def finish_task(self, task_run):
        if self._db is not None and task_run is not None:
            self._db.finish_task(task_run)

    def _task_run(self, task, task_run, phase, step, epoch):
        """The execution a row belongs to; a caller without one gets a one-shot execution."""
        if self._db is None:
            return None
        if task_run is None:
            task_run = self._db.start_task(self._run_id, task, phase=phase, step=step, epoch=epoch)
            self._db.finish_task(task_run)
        return task_run

    def log_summary(self, task, summary, step=None, epoch=None, task_run=None, phase="manual"):
        if self._db is None:
            return
        task_run = self._task_run(task, task_run, phase, step, epoch)
        now = time.time()
        self._summaries.extend(
            (task_run, str(key), to_json(value), now) for key, value in summary.items())
        self._maybe_flush()

    def log_records(self, task, records, step=None, epoch=None, task_run=None, phase="manual"):
        if self._db is None:
            return
        task_run = self._task_run(task, task_run, phase, step, epoch)
        now = time.time()
        self._records.extend(
            (task_run, record.get("id", index) if isinstance(record, dict) else index,
             to_json(record), now)
            for index, record in enumerate(records))
        self._maybe_flush()

    def log_hyperparameters(self, configs):
        if self._db is not None:
            self._db.add_parameters(self._run_id, flatten_configs(configs))

    def _maybe_flush(self):
        pending = len(self._metrics) + len(self._summaries) + len(self._records)
        if pending and (pending >= self.FLUSH_ROWS
                        or time.monotonic() - self._last_flush >= self.FLUSH_SECONDS):
            self.flush()

    def flush(self):
        if self._db is None:
            return
        if self._metrics:
            self._db.add_metrics(self._metrics)
            self._metrics = []
        if self._summaries:
            self._db.add_summaries(self._summaries)
            self._summaries = []
        if self._records:
            self._db.add_records(self._records)
            self._records = []
        self._last_flush = time.monotonic()

    def close(self):
        if self._db is None:
            return
        self.flush()
        self._db.finish_run(self._run_id)
        self._db.close()
        self._db, self._run_id = None, None
