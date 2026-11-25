---
title: Model and activation analysis
sources:
  - matformer/tasks/analysis.py
  - matformer/utils/tensor_stats.py
  - matformer/utils/profiling.py
---
# Model and activation analysis

Two analysis tasks describe a model without changing the training loop. They are ordinary
[[tasks/index | tasks]]: their summaries go to the run database, their scalar values become
metrics, and they can run standalone, inside `validation`, or on demand through the
[[training/control | control channel]].

## One definition of tensor statistics

`matformer.utils.tensor_stats.compute_tensor_stats(tensor)` is the single implementation. The
same dictionary is returned by `TensorDC.stats()`, by `ParameterEditor.stats(name)`, and by both
analysis tasks.

| Group | Fields |
|-------|--------|
| distribution | `mean`, `var`, `std`, `skewness`, `kurtosis`, `min`, `max`, quantiles |
| magnitude | `abs_mean`, `abs_max`, `rms`, `l1_norm`, `l2_norm` |
| sparsity | `zero_fraction`, `sparsity_1e-6`, `sparsity_1e-3` |
| outliers | `outlier_threshold`, `outlier_fraction`, `max_abs_z`, `outlier_vectors` |
| per-token geometry | `vector_norm_mean/std/min/max`, `vector_norm_p50/p90/p99`, `vector_max_abs_mean/max`, `top1/top2/top5/top10_share` |
| spectrum (2-D) | `frobenius`, `nuclear_norm`, `spectral_norm`, `stable_rank`, `effective_rank`, `effective_rank_centered`, `condition_number` |
| health | `shape`, `numel`, `nonfinite`, `nonfinite_fraction` |

Spectral statistics run only when asked: `spectral="auto"` allows them below four million
elements, `"always"` and `"never"` are explicit. Activations and gradients normally pass
`spectral="never"`.

`TensorDC.stats()` excludes padding by default, because a pad token is not a measurement of the
model. Pass `include_padding=True` when the padding path itself is the suspect.

`RunningStats` combines per-batch dictionaries into a dataset summary: weighted means, combined
standard deviations, exact minima/maxima, and no memory of the batches.

## static_model_analysis

```yaml
task:
  - name: weights
    type: static_model_analysis
    per_parameter: true
    gradients: true       # include parameter.grad when it exists
    spectral: auto
```

For every stable parameter it logs the full statistics, its layer, family (`attention`, `mlp`,
`norm`, `embedding`, `head`, `other`), dtype, shape and whether it is trainable. The summary also
contains model, family and layer totals in elements and bytes, trainable/frozen counts, plus
aggregate statistics per family, per layer and for the whole model. `gradients: true` adds the same description of the gradients currently
attached to the parameters, which is how a validation step can see gradient flow without keeping
a gradient history.

The task needs no dataset (`requires_data = False`), so it can be the first entry of a validation
group even when the model has no test set.

## activation_analysis

```yaml
task:
  - name: activations
    type: activation_analysis
    ref_data: test
    learning_signal: null     # null mirrors the model's own selector
    gradients: true
    outlier_sigma: 6.0
    spectral: never
    module_logs: true
    profile: true             # component timing
```

For every module that emits a `TensorDC` (the embedding head, every transformer block, and any
specialised socket-bearing module) the task records the same statistics of the output, padding
excluded. Per-token geometry is therefore present without storing a row per token: vector norms,
their quantiles, outlier counts and energy concentration.

`record_outliers: true` adds the most extreme tokens (per module) to the summary under
`outliers`, with the flat token index, the decoded text, the vector norm and the maximum absolute
activation. This is the quickest RPC answer to "which token in this dataset blows up?", and it
works without a logger because the list is part of the returned summary.

When `gradients` is true the task builds the learning-signal loss on its own batch and calls
`torch.autograd.grad`, which returns gradients *without touching the training gradients*. A
mid-training trigger can therefore never corrupt an accumulation window. Aggregate and
per-parameter gradient statistics are logged under `gradients/...`.

On a `VirtualModel` the task consumes the replayed activations directly: no hooks and no
gradients are needed, and every databank address becomes an activation name.

### Module-specific diagnostics

A module can implement `_log()` and return the numbers that only make sense for its
architecture. `MultiHeadAttention` reports head geometry, sliding window, ALiBi/RoPE choice,
QKV/clamp saturation and gate statistics; the FLA mixers report their recurrent architecture and
input/output magnitudes. `MatformerModelWrapper` remembers every such module once
(`refresh_diagnostics`) and `collect_module_diagnostics()` asks them all, so the task never loops
over the module tree and a new mixer only has to add `_log()`.

### Component profiling

With `profile: true`, `ComponentProfiler` attaches forward hooks for the duration of the task and
groups calls by component: embedding, attention, MLP, norm, head, block. The summary reports
calls, seconds, fraction of the run, mean/max milliseconds and, on CUDA, allocator deltas.
The summary also contains the task-level allocator start/end/peak and the peak growth, which is
the quick check for "is anything eating memory during this task?". No `timeit` is placed in model
code, and an ordinary training pass pays nothing because the hooks exist only while a profiler is
running.

## Batch-size probe from RPC

The trainer's memory probe is reachable without opening a shell:

```
train.probeOptions                                  # the knobs and their defaults
train.probe {"config": ["run.yaml"], "probe": {"fill": 0.9, "steps": 3}}
memory.reports {"root": "/tmp/my_run"}              # all reports under one folder
memory.report {"path": "/tmp/my_run/logs/run_v1/memory_probe.json"}
```

`train.probe` queues the same job `train.start` would, but with `probe` set and no training.
The report contains every attempted row count and its peak, the safe batch size and the measured
tokens/s/MFU. `apply: true` trains at the measured size in the same job.

## Reading the result

The summary is stored in `task_summaries` and every scalar is also a row in `metrics`, so the
same numbers are available as a table, as a plot, and through RPC:

```
runs.tasks    {"run": "project/run"}                  # executions and their summaries
runs.metrics  {"run": "project/run", "keys": ["activation_analysis/activations/.../mean"]}

For a run whose database is outside the server's configured roots (for example a test in /tmp),
pass the file explicitly: `runs.list {"db": "/tmp/run.db"}`, `runs.get {"run": "p/r",
"db": "/tmp/run.db"}`. `memory.reports {"root": "/tmp/run"}` works the same way for probes.
```

## Triggering an analysis mid-training

No restart is needed. The control channel accepts a task that was never declared:

```
train.task  {"id": "<job>", "type": "activation_analysis", "name": "extra",
             "fields": {"ref_data": "test", "profile": true}}
```

The task opens its own `task_runs` row at the current step, writes into the same database, and
the training continues. If the ad-hoc task raises, the trainer logs the failure in
`control_result.json` and keeps training.
