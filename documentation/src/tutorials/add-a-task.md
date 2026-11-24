---
title: Add a task
sources:
  - matformer/tasks/task.py
  - matformer/tasks/forward.py
---
# Add a task

This example defines a task that returns the argmax of the model for each item. It needs a config dataclass, a task class and a YAML section.

## Config

```python
from dataclasses import dataclass
from matformer.matformer_registry import registry
from matformer.tasks.task import TaskConfig

@registry.register_dataclass("task", "argmax")
@dataclass
class ArgmaxTaskConfig(TaskConfig):
    save_ids: bool = True
```

`TaskConfig` provides `ref_data`, `batch_preprocessors`, `batch_postprocessors`, the `save_*` flags and `presentation`. The decorator makes `type: argmax` valid in a YAML file and adds the task to the configuration reference.

## Task

```python
import torch
from matformer.tasks.task import MatformerTask

@registry.register("task", "argmax", "default")
class ArgmaxTask(MatformerTask):
    def process_batch(self, batch):
        with torch.no_grad():
            predictions = self.model(batch)
        ids = predictions.tensor.argmax(-1)
        result = {"argmax": ids}
        if self.config.save_ids:
            result["input_ids"] = batch.pad().tensor
        return batch, result
```

`wrap_process_batch` runs the preprocessors, calls `process_batch`, runs the postprocessors and calls `save_batch`. `save_batch` splits the result into one record per item.

A task that accumulates state implements `compute()` and returns a dictionary. The dictionary becomes the summary, is logged under the task name and stored in the run database's `task_summaries` table. A task that needs code before or after the batch loop overrides `begin_task` and `end_task`; call `super()` in both methods.

## Config section

```yaml
task:
  - name: mine
    type: argmax
    ref_data: test
    save_batch: true
    save_ids: false
```

`ref_data` refers to a named entry in the `data` section. Without it, the task has no dataloader and is driven one batch at a time; `save_batch: true` also keeps one record per item in the `task_records` table.

## Run

From the console:

```text
task mine
run
show mine
```

From Python:

```python
task = configs.instantiate("task", "mine", model=model)
summary = task()
```

## Notes

* Return the batch received by `process_batch`. A postprocessor may need it, and `save_batch` uses it to count items.
* A result that is not a dictionary is saved under `result`. `flatten_dict: true` merges a dictionary result into the record.
* The base class checks the `save_*` flags before logging records and the summary.
* `self.model` is a `MatformerModelWrapper`, so `self.model(batch)` expects a `TensorDC`. Use `batch.pad()` or `batch.unpad()` for the other layout.
* The registry constructs the task with `config=...`, `model=...`, `logger=...` and `trainer=...`.
