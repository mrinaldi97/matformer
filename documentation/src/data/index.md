---
title: Data
sources:
  - matformer/data/matformer_data_module.py
  - matformer/data/lazy_dataset.py
---
# Data

A run uses three data objects:

1. A dataset produces items. It reads files, tokenizes text and returns labels.
2. A data module builds a dataloader, batches items with `collate_fn`, handles distributed sharding and stores its position in a checkpoint.

The registry category is `data`. The trainer obtains its data module from `training.data.instantiate()`; a task obtains its data section through `ref_data`.

## Item format

A dataset item is a dictionary:

```python
{
    "object": [token_id, ...],
    "labels": 0,
    "is_same_document": True,
    "metadata": {"source": "..."},
    "worker_has_finished": False,
}
```

`object` is required. `labels` is one label for sequence tasks or one label per token for token tasks. `is_same_document` marks a chunk that continues the previous document. `metadata` is stored when `keep_metadata` is enabled. `worker_has_finished` marks the last item a worker produces.

The [[data/data-module | data module page]] describes how these fields become a batch.

## Data modules

The `lazy` module tokenizes files as they are read. It supports JSONL, CSV, CoNLL, CoNLL-U and HuggingFace datasets, and it is intended for quick experiments, benchmarks and instruction tuning.

The `mdat` module reads a pretokenized MDAT view. The dataset is built once, and training reads chunks or tokens from the MDAT stores. See [[data/mdat | MDAT]].

Both modules share the base class, `collate_fn` and item format.

## Task output

A task writes its summary to the run database (`task_summaries`) and, when `save_batch: true`, its
per-item records to `task_records`. The `annotate` command writes the same records to a JSONL file
or an mdat instead. See [[training/logger | Logging]] and [[tasks/interactive-session | Interactive console]].
