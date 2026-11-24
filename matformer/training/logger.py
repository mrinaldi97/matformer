# matformer/training/logger.py
from tqdm import tqdm
from matformer.matformer_registry import registry

SUMMARIES = {}
"""In-process task summaries kept by MemoryLogger, keyed by run name. Experiments that want the
metrics back as Python objects (MDL coding, K-sweeps) use this instead of a database."""


def pop_summaries(key):
    """Return (and forget) every summary a MemoryLogger collected under `key`."""
    return SUMMARIES.pop(key, [])


@registry.register("logger", "memory", "default")
class MemoryLogger:
    """Keeps task summaries in memory and writes nothing. Intended for in-process experiments."""

    def __init__(self, fabric=None, configs=None, **_):
        self._zero = fabric is None or fabric.is_global_zero
        self.key = configs.training.run_name if configs is not None else None

    @staticmethod
    def build_fabric_logger(training_configs):
        return None

    def log(self, metrics, step):
        pass

    def log_summary(self, task, values, step=None, epoch=None):
        if self._zero and self.key is not None:
            SUMMARIES.setdefault(self.key, []).append(dict(values))

    def log_records(self, task, records, step=None, epoch=None):
        pass

    def log_hyperparameters(self, configs):
        pass

    def close(self):
        pass


def flatten_configs(configs):
    """Flatten MatformerConfigs into 'section/entry/field' keys; private fields are dropped,
    non-plain values stringified. Shared by every logger backend that stores hyperparameters."""
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
    return dict(walk(configs))


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

    def start_task(self, task, *, step=None, epoch=None, phase="training"):
        """Open one task execution. Returns a token to pass back to log_summary/log_records and
        finish_task; backends that do not store tasks simply do not appear in it."""
        return {id(b): b.start_task(task, step=step, epoch=epoch, phase=phase)
                for b in self._backends if hasattr(b, 'start_task')}

    def finish_task(self, task_run):
        for b in self._backends:
            if hasattr(b, 'finish_task') and task_run:
                b.finish_task(task_run.get(id(b)))

    def log_summary(self, task, values, *, step=None, epoch=None, task_run=None):
        """The rich end-of-task result: arrays and nested objects a metric column cannot hold."""
        step = self._step if step is None else step
        for b in self._backends:
            if hasattr(b, 'log_summary'):
                b.log_summary(task, values, step=step, epoch=epoch,
                              task_run=(task_run or {}).get(id(b)))

    def log_records(self, task, records, *, step=None, epoch=None, task_run=None):
        """One row per item a task produced, when the task asked to keep them."""
        step = self._step if step is None else step
        for b in self._backends:
            if hasattr(b, 'log_records'):
                b.log_records(task, records, step=step, epoch=epoch,
                              task_run=(task_run or {}).get(id(b)))

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

    def start_task(self, task, *, step=None, epoch=None, phase="training"):
        return self.parent.start_task(f"{self.prefix}/{task}", step=step, epoch=epoch, phase=phase)

    def finish_task(self, task_run):
        self.parent.finish_task(task_run)

    def log_summary(self, task, values, *, step=None, epoch=None, task_run=None):
        self.parent.log_summary(f"{self.prefix}/{task}", values, step=step, epoch=epoch,
                                task_run=task_run)

    def log_records(self, task, records, *, step=None, epoch=None, task_run=None):
        self.parent.log_records(f"{self.prefix}/{task}", records, step=step, epoch=epoch,
                                task_run=task_run)

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
        if self._zero:
            self._l.experiment.config.update(flatten_configs(configs), allow_val_change=True)


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