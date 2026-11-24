---
title: Logging
sources:
  - matformer/training/logger.py
  - matformer/training/sqlite_logger.py
  - matformer/utils/run_db.py
---
# Logging

`MatformerLogger` collects the metrics of one optimizer step and sends the row to every backend.
Backends are registered in the `logger` category.

```yaml
training:
  logger: [sqlite, tqdm]
```

## The run database

The default backend is `sqlite`: every run is written to one SQLite database, and the framework
reads it from there (server, GUI, scripts). The location defaults to `$LOG_ASSETS/matformer.db`;
`training.log_db` picks another path:

```yaml
training:
  log_db: "$LOG_ASSETS/my_runs.db"   # optional; $LOG_ASSETS or an absolute path
```

Writes are buffered and committed in batches (1024 rows, or at most once per second), with WAL
enabled, so a log step costs a list append instead of a disk round-trip. The queue and the GUI read
the same file while the training writes it.

The schema has five tables:

| Table | Holds |
|-------|-------|
| `runs` | identity: project, run name, version, checkpoint, host, start/end, elapsed, status, sweep |
| `parameters` | flattened hyperparameters, one row per `section/entry/field` key |
| `metrics` | step metrics: run, step, key, value (numbers only) |
| `task_runs` | one row per task execution: training run, task, phase, step, epoch, start/end, elapsed |
| `task_summaries` | the end-of-task result, linked to its `task_runs` row: key, JSON value |
| `task_records` | per-item task output when `save_batch: true`, linked to its `task_runs` row |

A run is identified by `project/name`, with `@vN` appended when it is not version 1. Resuming the
same run reopens its row and keeps the original start time.

A task executed during training (the validation/evaluation tasks) opens one `task_runs` row with
`phase='training'` and the step and epoch it ran at; every summary and record it produces points
there. A task driven by a script with a logger uses `phase='standalone'`. This is what makes an
evaluation inside a training run unambiguous: one execution, one step, one epoch.

## Reading runs

`matformer.utils.run_db.RunDB` is the read and write API. The server uses it for every run
command: `runs.list`, `runs.query`, `runs.config`, `runs.metrics`, `runs.get`, `runs.tasks`, `runs.summaries`,
`runs.records`, `runs.watch`, plus the management commands `runs.delete`, `projects.delete`, `projects.list`,
`sweeps.list` and `runs.plot`. Nothing parses log files.

```python
from matformer.utils.run_db import RunDB

with RunDB("$LOG_ASSETS/matformer.db") as db:
    run = db.resolve("my-project/my-run")
    print(db.parameters(run["id"]))
    print(db.metrics(run["id"]))          # {key: [[step, value], ...]}
    print(db.task_runs(run["id"]))        # one row per evaluation, with step and epoch
    print(db.summaries(run["id"]))        # {task: [{task_run_id, phase, step, epoch, key, value}]}
```

## MatformerLogger

`log(key, value, step=None, prog_bar=False)` writes to the current row. `advance()` sends the row
to the backends and advances the step. `flush()` sends the row without advancing. `reset()` clears
the bars, and `close()` flushes and closes the backends.

A backend with a `buffer` method is treated as a progress bar. It receives `buffer`, `advance`,
`reset` and `close`. Other backends receive `log(row, step)`, and the optional `log_summary` and
`log_records` calls used by tasks.

## Backends

| Variant | Output |
|---------|--------|
| `sqlite` | the run database: runs, parameters, metrics, task summaries and records |
| `wandb` | a Fabric `WandbLogger`; hyperparameters and one row per step |
| `tqdm` | a progress bar, disabled on every rank except 0 |
| `memory` | task summaries kept in the process, for experiments that want Python objects |

`flatten_configs(configs)` converts a `MatformerConfigs` object to `section/entry/field` keys. It is
used by the backends that store hyperparameters.

## Scoped logging

`logger.scoped(prefix)` returns a logger that prefixes each key with `prefix/` and uses its own
`prefix/step`. It does not advance the parent step or close the parent backends. `sandbox_train`
uses it for the training runs it starts inside a validation step.

## Logged metrics

The wrapper logs `train/loss`, `train/aux_loss`, the learning-signal metrics (`train/accuracy` from
`post_loss`), `train/skipped_oom`, `train/EXCEPTIONS`, and `lr0`, `lr1`, ... for the schedulers. The
trainer adds `speed/*`: tokens per second, sequences per second, TFLOPs, MFU and step time, measured
over `log_speed_every_n_steps` steps. Tasks log their summary under `<task name>/<metric>`, and the
rich summary goes to `task_summaries`.
