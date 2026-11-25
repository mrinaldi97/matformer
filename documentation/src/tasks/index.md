---
title: MatformerTask
sources:
  - matformer/tasks/task.py
category_docs: [task]
---
# MatformerTask

This page is the overview of the task abstraction. The built-in tasks are described in [[tasks/built-in-tasks | Built-in tasks]], and the console that drives them is in [[tasks/interactive-session | The interactive console]].
[[dataclass:TaskConfig | Dataclass]] [[file:matformer/tasks/task.py | Code]]
Tasks are a very important concept in the MatformerArchitecture. In general, they mean "do something with a model", wheter it means perform an evaluation, generate text, run interpretability plugins or even just run a forward pass on the model. While training is currently not routed through the MatformerTask, in theory even the training process could potentially be reframed as a particular kind of task. What matters is that MatformerTask provides a unified interface to run "something" on the model, providing an abstract class and a specific interface for the experimenter to add a very large variety of tasks.
The concept of a "task" revolves around three things: data, model and loggin/reporting.
[[file:matformer/tasks/task.py:MatformerTask.__init__ | MatformerTask.__init__]]'s init initialize the [[data/data-module | dataloader]] dataloader, very similar to how the [[training/trainer | MatformerTrainer]] works; a specific dataset can be used for a task using a data section in the [[components/configuration | MatformerConfigs]] YAML file where its name correspond to the "ref_data" field in the [[dataclass:task.generation | TaskConfig]]. A task, however, is not forced to have a dataset loaded, as they are used also for the interactive mode, where the task is executed batch by batch in an interactive manner. A task that analyses the model itself can set `requires_data = False`: `run()` then goes straight to `compute()` (the [[training/analysis | analysis tasks]] do exactly this). MatformerTask can either use the model defined in the [[dataclass:model.default | model's config section]] or a [[components/virtual-models | VirtualModel]], where activations from a model were previously collected.
The task output goes through the logger: each execution opens a `task_runs` row (task, phase, step, epoch, duration), the summary goes to `task_summaries` and, when `save_batch` is enabled, one record per item goes to `task_records`; both point at that execution. The `annotate` command writes those same records to a JSONL file instead, so the run database and file annotation are two opt-in choices over one record builder.
When the task begins, by simple [[file:matformer/tasks/task.py:MatformerTask.__call__ | calling]] the task's instance, it call the begin_task() function, which tasks can subclass to execute single instruction before the task's loop begin. Then, each batch is wrapped between optionals "batch_pre/post processors", useful for more complex evaluation scenarios (noop as default).
Each specific task must implement a "process_batch" function. This function must return the batch itself (to be used for eventual postprocessors) and an optional dictionary (None if this is not relevant for this task) that can then be logged or annotated (for example the result coming from each single batch). [[file:matformer/tasks/task.py:MatformerTask.save_batch | save_batch]] is indeed called after each [[file:matformer/tasks/task.py:MatformerTask.wrap_process_batch | processed batch]].
Because many tasks can benefit from accumulating states batch-by-batch to compute results at the end, when all the data is passed through process_batch, [[file:matformer/tasks/task.py:MatformerTask.compute_summary | compute_summary]] is called, which, by internally calling the task's optional function "compute()", allow the task to perform a final compute and return an optional "summary" dictionary, that is logged under the task name and stored in `task_summaries`. An optional end_task() function is called before returning the optional summary or None to the caller.
Tasks can be explored through the Registry: [[registry_category:tasks]]
Some bult-in tasks are "generation", which autoregressively generate texts for GPT-like models, "evaluation", which by mimicking the same process used during training (the use of [[training/learning-signals | LearningSignals]]), is perfect to perform general evaluation (for example, accuracy, perplexity, f1...) as well as perform forward passes of classification models. "forward" is a dummy task, but very useful: it just perform a forward pass on the model, but this can be very useful if [[components/plugins | Plugins]] are attached to the model; a perfect use case for this dummy forward pass is to capture activations for a [[components/virtual-models | VirtualModel]].
Writing a task is very easy, and it is a very flexible way to do any kind of thing with a model loaded with Matformer, such as benchmarking or perform interpretability studies. For this reason, you can follow the [[tutorials:add_a_task | tutorial to add a task]].

## Lifecycle

```
__init__          dataloader from ref_data, model from the wrapper
begin_task()      once before the first batch (ensure_began() calls it lazily)
  for each batch:
    batch_preprocessors  ->  process_batch  ->  batch_postprocessors  ->  save_batch
compute_summary()    calls compute(), logs the summary, saves it
end_task()           after the summary
```

`__call__(batch=None)` is the interactive entry point: with a batch it runs one step without a dataset; without a batch it runs the whole dataset through `run()`. `save_results=False` keeps an interactive call out of the task's output files, which is what the GUI and the console use when they only want the prediction.

## Return values

`process_batch` returns `(batch, result)`. The batch is returned so a postprocessor can still read it; the result is split by `save_batch` into one record per item. `None` means that there is nothing to save, which is a valid return for a forward pass or for a task that only accumulates state.

`split_result` converts a batch-level result into per-item rows: a tensor whose leading dimension is the batch is split row-wise, a dictionary is split by key, and other values are repeated on every row. A task that returns `{"logits": tensor, "argmax": tensor}` produces one record per item with both fields.

## Model

The task receives a model instance. The caller (the console, the trainer or a script) builds it with [[training/wrapper#model-loading | load_model]]. The `ref_data` field selects the dataset section; without it, the task is driven interactively.
