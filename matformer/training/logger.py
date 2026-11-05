from tqdm import tqdm
import json
from pathlib import Path


class JSONLLogger:
    """Append local metrics across restarts; no network dependency."""
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.stream = open(path, 'a', buffering=1)

    def log(self, metrics, step):
        self.stream.write(json.dumps({'step': step, **metrics}) + '\n')

    def close(self):
        self.stream.close()

class MatformerLogger:
    def __init__(self):
        self._loggers = []

    def add(self, logger):
        self._loggers.append(logger)

    def log(self, key, value, *, step=None, prog_bar=False, on_step=True, on_epoch=False, **_extra):
        for l in self._loggers:
            if not isinstance(l, TQDMBar):
                l.log({key: value}, step=step)
        if prog_bar:
            for l in self._loggers:
                if isinstance(l, TQDMBar):
                    l.buffer(key, value)

    def advance(self):
        for l in self._loggers:
            if hasattr(l, 'advance'): l.advance()

    def close(self):
        for l in self._loggers:
            if hasattr(l, 'close'): l.close()


class WandbAdapter:
    def __init__(self, fabric_wandb_logger):
        self._l = fabric_wandb_logger

    def log(self, metrics, step):
        self._l.log_metrics(metrics, step=step)

    def close(self):
        self._l.finalize('success')
        # Lightning's logger finalizes artifacts but does not close the W&B
        # process. Finish explicitly so offline files are flushed for sync.
        self._l.experiment.finish()


class TQDMBar:
    def __init__(self, total, initial=0, desc="Training"):
        self._bar = tqdm(
            total=total, initial=initial,
            dynamic_ncols=True, desc=desc,
            miniters=1, mininterval=1,
        )
        self._pending: dict = {}   

    def buffer(self, key, value):
        self._pending[key] = value

    def advance(self):
        if self._pending:
            self._bar.set_postfix(
                {k: f"{v:.4f}" if isinstance(v, float) else v
                 for k, v in self._pending.items()},
                refresh=False,   
            )
        self._bar.update(1)       

    def close(self):
        self._bar.close()
