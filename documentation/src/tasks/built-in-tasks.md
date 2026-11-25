---
title: Built-in tasks
sources:
  - matformer/tasks/forward.py
  - matformer/tasks/evaluation.py
  - matformer/tasks/generation.py
  - matformer/tasks/masking.py
  - matformer/tasks/sandbox_train.py
---
# Built-in tasks

The registry lists every task under [[registry_category:tasks | task]].

## forward

Runs one forward pass and returns no result. It is used with plugins that record activations, for example when collecting data for a [[components/virtual-models | VirtualModel]].

```yaml
task:
  - name: collect
    type: forward
    ref_data: train
    save_batch: false
```

## evaluation

Runs the evaluation pipeline. If `learning_signal` is set, it uses the same `prepare_inputs`, forward and `select_targets` path as training. If it is unset, it calls the model directly and uses the batch labels.

The task accumulates:

* `mean_confidence`, `mean_entropy`;
* `nll`, `perplexity`, `accuracy`, `brier_score`, `codelength_bits`, `num_tokens`;
* `acc@k` for each value in `topk`;
* `perplexity_by_token` when `compute_perplexity_by_token` is enabled;
* macro and micro precision, recall and F1, with per-class arrays, when `compute_confusion_matrix` is enabled.

The `save_*` flags select the fields of each record: `save_argmax`, `save_logits`, `save_logprobs`, `save_probs`, `save_input_ids`, `save_attributes`. Per-item records are stored only when `save_batch: true`; the summary is always logged to `task_summaries`. `limit_to_classes` restricts the saved logits. `logit_diff_classes: [i, j]` adds the margin between two classes. `presentation` (`probs`, `argmax`, `full`) controls the interactive output.

```yaml
task:
  - name: test
    type: evaluation
    ref_data: test
    learning_signal: token_classification
    compute_confusion_matrix: true
    num_classes: 7
    topk: [1, 3, 5]
    save_argmax: true
```

A dictionary prediction from `MultiProbeClassifier` produces one metric group per address, with the address as prefix.

## generation

Generates text for autoregressive models. The sampling fields are `temperature`, `top_k`, `top_p`, `repetition_penalty`, `no_repeat_ngram_size`, `do_sample`, `max_new_tokens` and `min_new_tokens`.

Interactive options:

* `stream: true` prints tokens as they are generated. A UI can pass `stream_callback` to receive `(tokens, step)`.
* `keep_context: true` keeps the conversation before the next prompt and returns only the new text.
* `max_context_tokens` trims the context from the left.
* `use_cache: true` uses the paged KV cache. This requires `flash_kv`, `sdpa_kv` or `flashinfer`.
* `topk_present` keeps the top-k candidates of each generated token. `0` is plain text, `-1` is the whole vocabulary.

```yaml
task:
  - name: chat
    type: generation
    prompt: "Once upon a time"
    max_new_tokens: 200
    temperature: 0.8
    top_p: 0.95
    stream: true
    keep_context: true
```

## masking

Runs a cloze test. The [[components/maskerator | Maskerator]] hides a fraction of the tokens, the model restores them, and the score is computed on the hidden positions. Over a dataset the task accumulates the same metrics as `evaluation`. Interactively it shows the alternatives for each hole.

`masking_ratio` is the fraction hidden, `bidirectional` controls the attention direction, and `topk_present` controls the number of alternatives. A class head predicts labels, so `vocab_head` is false and the original token is shown as context without being scored.

## sandbox_train

Runs a training inside a validation step. `sandbox_train` saves a snapshot of the current model, moves the outer model to CPU, drops the DDP wrapper, and runs one or more `MatformerTrainer` runs on the snapshot with their own configs and scoped logger. At the end it rebuilds the outer model and deletes the snapshot unless `keep_snapshot` is true.

The snapshot is written inside the checkpoint directory and named after the run and the step. The task needs a trainer, so it is used from `validation`.

## static_model_analysis

Reads every parameter and logs the full tensor statistics, layer/family metadata, model and family
totals, and optionally the current gradients. It needs no dataset and can therefore run from a
`validation` entry as a weight-health check. See [[training/analysis | analysis]].

## activation_analysis

Runs the model over a dataset and records the same statistics for every module output, padding
excluded, including per-token norm quantiles and outliers. With `gradients: true` it differentiates
a learning-signal loss through `torch.autograd.grad`, which does not touch the training gradients.
`module_logs: true` collects the architecture-specific `_log()` of mixers, and `profile: true`
records per-component time and CUDA allocator deltas. See [[training/analysis | analysis]].

## New tasks

Subclass `MatformerTask`, implement `process_batch`, and register the task and its config:

```python
@registry.register_dataclass("task", "my_task")
@dataclass
class MyTaskConfig(TaskConfig):
    my_option: int = 3

@registry.register("task", "my_task", "default")
class MyTask(MatformerTask):
    def process_batch(self, batch):
        result = self.model(batch)
        return batch, {"prediction": result.tensor.argmax(-1)}
```

The [[tutorials/add-a-task | add a task tutorial]] contains a complete example.
