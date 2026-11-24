---
title: Lazy dataset
sources:
  - matformer/data/lazy_dataset.py
  - matformer/data/matformer_data_module.py
---
# Lazy dataset

`LazyIterableDataset` is the dataset used by the `lazy` data module. It reads a file or a HuggingFace dataset as a stream, tokenizes each item and yields the dictionary described in [[data/index | Data]]. It is intended for quick experiments, benchmarks and instruction tuning. For long pretraining runs, MDAT avoids repeated tokenization.

## Configuration

```yaml
data:
  - name: train
    type: lazy
    data_root: data/train.jsonl
    file_format: jsonl
    text_column: text
    batch_size: 8
    max_seq_len: 512
    task: lm
```

The main fields are:

| Field | Meaning |
|-------|---------|
| `data_root` | a file, a folder or a HuggingFace dataset name |
| `file_format` | `jsonl`, `csv`, `conll`, `conllu`, `hf`; `None` infers it |
| `text_column` | the column to tokenize |
| `input_ids_column` | the column to use when the data is already tokenized |
| `tokenize` | `False` when the file contains ids |
| `add_bos`, `add_eos` | add the special tokens |
| `max_seq_len` | truncation length; the data module can inherit it from the model |
| `split` | the HuggingFace split |
| `precompute_len` | read the file once to compute the length |

## Tasks

`task` selects the supervision:

* `lm`: the item is a token sequence. The learning signal creates the targets.
* `sentence-level`: `label_column` contains one label per row. `label2id` maps string labels; if it is absent, the dataset infers the mapping from a full scan.
* `token-level`: the row contains a text and a column of tags, or a tokenized `input_ids` field and one label per token. The dataset aligns labels with subwords: the first subword of a word receives the label, continuations receive `IGNORE_INDEX`.
* `regression`: the label is a float and the loss is a regression loss.

`missing_label_values` adds strings that mean "no label". The defaults include empty, `null`, `none`, `nan`, `n/a`, `na`, `nil`. A row without a label is dropped in a supervised task and kept in a language-model task.

## Labels

When `label2id` is configured, it is used directly. If it is absent, the dataset can scan the file to collect labels, counts and class weights. `scan_labels` controls the scan: `None` runs it when the task needs it, `True` always runs it, `False` skips it unless a consumer requires it.

The data module exposes `get_label_maps()`, `get_label_info()`, `get_num_labels()`, `get_label_distribution()` and `get_class_weights()`. Tasks and the trainer read the label information from the data module, so a resumed run keeps the same label order.

## Streaming and resume

The dataset is an `IterableDataset`. Each rank reads the global stream and keeps every `world_size`-th row. `skip_global` and `_global_idx` support a resume: `state_dict` stores the last processed row, and `restore_state_dict` skips to it. `partition_start` and `partition_end` restrict the stream to a window.

`compute_length()` counts the rows and caches the result. Without it, `__len__` returns `0` and the trainer disables LR scheduling.

## Limitations

The lazy dataset truncates at `max_seq_len` and does not split documents into chunks. If every token of a long document is needed, use MDAT.
