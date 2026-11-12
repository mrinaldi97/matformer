# matformer/training/logger.py
from tqdm import tqdm
from matformer.matformer_registry import registry


class MatformerLogger:
    """Fan-out to the backends (wandb...) and the progress bars. Metrics logged during a step are collected
    into ONE row, sent to the backends when the step ends (advance), so wandb gets one row per step."""
    def __init__(self):
        self._backends, self._bars = [], []
        self._row, self._step = {}, None

    def add(self, logger):
        (self._bars if hasattr(logger, 'buffer') else self._backends).append(logger)  # duck-typed: bars have buffer()

    def log(self, key, value, *, step=None, prog_bar=False, **_extra):
        self._step = self._step if step is None else step  # e.g. task summaries don't know the step
        self._row[key] = value
        if prog_bar:
            for b in self._bars: b.buffer(key, value)

    def emit(self, row):
        for b in self._backends: b.log(row, step=self._step)

    def flush(self):
        if self._row:
            self.emit(self._row)
            self._row = {}

    def advance(self):
        self.flush()
        if self._step is not None:
            self._step += 1  # the trainer just incremented its step: later logs belong to the next one
        for b in self._bars: b.advance()

    def reset(self):
        self.flush()
        for b in self._bars: b.reset()

    def close(self):
        self.flush()
        for b in self._bars + self._backends: b.close()

    def log_hyperparameters(self, configs):
        pass

    def scoped(self, prefix):
        return _ScopedLogger(self, prefix)


class _ScopedLogger:
    """Sandbox logger: keys are namespaced under `prefix/` and plotted against their own `prefix/step`.
    It never ticks the outer bar, never advances the outer step, and never shows values in the outer bar."""
    def __init__(self, parent, prefix):
        self.parent, self.prefix, self.row, self.step = parent, prefix, {}, None
        for b in parent._backends:
            if hasattr(b, 'define_step_metric'): b.define_step_metric(prefix, f"{prefix}/step")

    def log(self, key, value, *, step=None, **_extra):  # prog_bar is deliberately ignored
        self.step = self.step if step is None else step
        self.row[f"{self.prefix}/{key}"] = value

    def flush(self):  # the sandbox's own step ended: send its row, stamped with the (frozen) outer step
        if self.row:
            self.parent.emit({**self.row, f"{self.prefix}/step": self.step})
            self.row = {}

    advance = reset = close = flush  # none of them may touch the parent's bar, step or backends

    def log_hyperparameters(self, configs):
        pass

    def scoped(self, prefix):
        return self.parent.scoped(f"{self.prefix}/{prefix}")


@registry.register("logger", "wandb", "default", requires=['wandb'])
class WandbAdapter:
    def __init__(self, fabric=None, fabric_logger=None, configs=None, **_):
        self._l = fabric_logger
        self._zero = fabric_logger is not None and (fabric is None or fabric.is_global_zero)
        if configs is not None:
            self.log_hyperparameters(configs)

    @staticmethod
    def build_fabric_logger(training_configs):
        from wandb.integration.lightning.fabric import WandbLogger
        return WandbLogger(project=training_configs.get('project_name', 'matformer'),
                           name=training_configs.get('run_name'))

    def log(self, metrics, step):
        self._l.log_metrics(metrics, step=step)

    def define_step_metric(self, prefix, step_key):
        """Plot every `prefix/*` metric against `step_key` instead of the outer global step."""
        if self._zero:
            self._l.experiment.define_metric(step_key)
            self._l.experiment.define_metric(f"{prefix}/*", step_metric=step_key)

    def close(self):
        if self._zero:
            self._l.experiment.finish()

    def log_hyperparameters(self, configs):
        """Flatten MatformerConfigs into 'section/entry/field' keys; private fields are dropped, non-plain values stringified."""
        def walk(obj, prefix=""):
            if hasattr(obj, 'objects'):                # MatformerConfigs / GroupConfig
                for name, child in obj.items():
                    yield from walk(child, f"{prefix}/{name}".lstrip("/"))
            elif hasattr(obj, 'asdict'):               # a single config
                yield from walk(obj.asdict(include_defaults=True), prefix)
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    if not str(k).startswith("_"): yield from walk(v, f"{prefix}/{k}")
            else:
                yield prefix, obj if obj is None or isinstance(obj, (str, int, float, bool)) else str(obj)
        if self._zero:
            self._l.experiment.config.update(dict(walk(configs)), allow_val_change=True)


@registry.register("logger", "tqdm", "default", requires=['tqdm'])
class TQDMBar:
    def __init__(self, fabric=None, total=None, initial=0, desc="Training", **_):
        # Disabled without a fabric (the trainer builds a throwaway instance just to call build_fabric_logger)
        # and on every rank but 0, so there is exactly one bar
        self._bar = tqdm(total=total, initial=initial, desc=desc, dynamic_ncols=True, miniters=1, mininterval=0,
                         disable=fabric is None or not fabric.is_global_zero)
        self._postfix = {}

    @staticmethod
    def build_fabric_logger(training_configs):
        return None

    def buffer(self, key, value):
        self._postfix[key] = value

    def advance(self):
        self._bar.set_postfix({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in self._postfix.items()}, refresh=False)
        self._bar.update(1)

    def reset(self):
        self._bar.refresh()
        self._bar.reset()

    def close(self):
        self._bar.close()