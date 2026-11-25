# matformer/training/jsonl_logger.py
"""The Matformer-native logger backend: no external service, no network.

Everything a run produces is stored inside the checkpoint directory, under
`{save_dir}/logs/{run_name}_v{version}/`:

    run.jsonl    one line per emitted row: {"step": int, "t": epoch_s, "m": {key: value}}
    config.json  the flattened hyperparameters of the run (section/entry/field keys)
    meta.json    run identity: project, run, checkpoint name/version, host, pid, timestamps

Wandb-free replacement for the WandbLogger: the same features (metric rows + hyperparameters),
plus the Matformer server can tail these files to serve live plots to the GUI.
"""
import json
import os
import socket
import time
from datetime import datetime
from pathlib import Path

from matformer.matformer_registry import registry
from matformer.training.logger import flatten_configs


def run_dir_for(training_configs) -> Path:
    """The log folder of a run: inside the checkpoint manager directory, per run/version."""
    return (Path(training_configs.save_dir) / "logs"
            / f"{training_configs.run_name}_v{training_configs.version}")


@registry.register("logger", "jsonl", "default")
class JsonlLogger:
    """Fan-out backend of MatformerLogger. Built twice by the trainer (once bare, to ask for a
    fabric logger): files are opened only when the configs are handed over, on rank 0 only."""

    def __init__(self, fabric=None, configs=None, **_):
        self._zero = fabric is None or fabric.is_global_zero
        self._run, self._fh = None, None
        if configs is not None and self._zero:
            self._open(configs)

    @staticmethod
    def build_fabric_logger(training_configs):
        return None

    def _open(self, configs):
        training = configs.training
        self._run = run_dir_for(training)
        self._run.mkdir(parents=True, exist_ok=True)
        started = datetime.now().isoformat(timespec="seconds")
        meta = {"project": training.get("project_name", "matformer"), "run": training.run_name,
                "checkpoint_name": training.checkpoint_name, "version": training.version,
                "save_dir": str(Path(training.save_dir).resolve()), "host": socket.gethostname(),
                "pid": os.getpid(), "started": started}
        point = getattr(configs, "active_sweep", None)
        if point is not None:      # which combination of the sweep this run is, spelled out
            meta["sweep"] = point.asdict()
        if self._run.joinpath("meta.json").exists():       # resume: keep the original start time
            meta["started"] = json.loads(self._read(self._run / "meta.json")).get("started", started)
        self._write(self._run / "meta.json", meta)
        self.log_hyperparameters(configs)
        self._fh = open(self._run / "run.jsonl", "a", buffering=1)

    @staticmethod
    def _read(name):
        return Path(name).read_text("utf-8")

    @staticmethod
    def _write(name, obj):
        Path(name).write_text(json.dumps(obj, indent=1, default=str), "utf-8")

    def log(self, metrics, step):
        if self._fh is None:
            return
        row = {"step": step, "t": time.time(),
               "m": {k: (v if v is None or isinstance(v, (str, int, bool))
                         else float(v) if isinstance(v, float) else str(v))
                     for k, v in metrics.items()}}
        self._fh.write(json.dumps(row, default=str) + "\n")

    def log_hyperparameters(self, configs):
        if self._run is not None:
            self._write(self._run / "config.json", flatten_configs(configs))

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None
