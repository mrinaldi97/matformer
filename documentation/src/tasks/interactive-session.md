---
title: Interactive console
sources:
  - matformer/tasks/session.py
  - run_task.py
---
# Interactive console

`run_task.py` opens a `cmd2` shell around a loaded model. The shell is `TaskConsole` from `matformer/tasks/session.py`; it uses `TaskSession` to feed one batch at a time and keeps run statistics.

Example:

```text
task test
run
show test
set temperature 0.7
ask "The capital of Italy is"
stats
quit
```

## Commands

* `task <name>` selects a configured task. `describe` shows its fields and `set` changes them.
* `run` executes the task over its dataset.
* `ask` or a raw batch drives the task interactively when no dataset is configured.
* `show` renders rows through the task's `present` method.
* `stats` reports input and output tokens, time, speed and memory.
* `clear` drops the generation context and the KV cache.
* `annotate <file|mdat>` writes one record per item plus `<output>.summary.jsonl`. The output is JSONL for file inputs and an mdat for mdat inputs; `--format jsonl|mdat` overrides it, `--keys-only` writes a metadata-only mdat with just the source key and predictions, and `--dataset NAME` selects a dataset inside an mdat container.

## Drivers

A task receives one batch and returns one result. `run()` contains the dataset loop, and the console calls the task one batch at a time. Scripts and the GUI use the same task interface.

The session stores the generation context and the KV cache. It never opens the run database: interactive use stays ephemeral, and the user can annotate a file explicitly. The formatting helpers (`render`, `format_summary`, `format_rows`) only affect the console output.

## From Python

```python
from matformer.matformer_registry import registry

task_config = registry.get_definition("task", "evaluation").from_dict({"ref_data": "test"})
task = registry.create("task", "evaluation", config=task_config, model=model)
summary = task()
```

`run_task.py` builds the model and opens the console. A script can build the same task directly.
