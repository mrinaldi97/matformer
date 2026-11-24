---
title: MDAT
sources:
  - matformer/data/mdat/container.py
  - matformer/data/mdat/mdat.py
  - matformer/data/mdat/strategy.py
  - matformer/data/mdat/preprocessing.py
  - matformer/data/mdat/tui.py
category_docs: [data]
---
# MDAT

Matformer comes together with its own dataset format, designed by keeping in mind the need for easily getting a big dataset formed of many independent sub-dataset, as well as the importance in experimental settings to test different tokenization strategies and having different views of the dataset, such as a smaller dataset derived from the main one for ablation studies.

MDAT is a complex and extensible dataset manager that handles all of this for you. To understand how it works, it is important to familiarize with its components.

## Container and datasets

MDAT is the name of the main container. It presents itself as a folder with a specific structure:

```
mdat_container.db
datasets/
views/
shuffling/
pretok/
functions/
```

Older datasets use `mdat.db`; both names are accepted.

Each mdat is composed of one to many submdats that live in the `datasets` folder. It is possible to add up to 255 submdats to an mdat or 65535 with the extended mode. The `dsmap_bits` field of the catalog sets the limit: `8` bits for 255 datasets and `16` bits for 65535.

A submdat can be created from many dataset formats such as JSON or HuggingFace. The dataset is automatically split into two files: `data.db`, containing the actual raw data, and `meta.db`, containing the optional metadata for each dataset row. A dataset with heavy metadata does not slow down data loading. Compression is optional.

A submdat can also be a standalone dataset. A standalone dataset is a directory created with `MDat.create_standalone(path)` or with the shell command `new PATH`; it is its own minimal container with one dataset. Any container can link it with `link_dataset(path, alias=...)` or with `link PATH --alias NAME`. The data and the dataset catalog stay in the standalone folder; the host container stores the alias and the view bookkeeping.

## Pretokenization

Once all your submdats are in the mdat, it's time to define a **pretokenization strategy**. You can look at the default pretokenization strategy in the `config/sample_strategy.json` file. If you don't like it, you can write your own custom pretokenization function and store it in the `functions` folder.

Pretokenization strategies are extremely flexible; let's have a look to the default one:

```json
{
  "strategy_name": "Minerva1024",
  "tokenizer_type": "huggingface",
  "tokenizer_name": "sapienzanlp/Minerva-350M-base-v1.0",
  "vocab_size": 32768,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "mask_token_id": null,
  "splitter_class": "split_and_tokenize_by_nltk_sentences_aligned",
  "splitter_init": {"language": "italian"},
  "splitter_arguments": null,
  "modality": "text",
  "chunk_size": 1024,
  "wants_from_db": ["data"],
  "wants_raw": true,
  "returns": ["tokens", "chunks"],
  "required_databases": ["tokens", "chunks"]
}
```

As you can see, the core function called by any pretokenization strategy is the **splitter**, that can be easily customized. The splitter will receive data from the submdat (ex `"data"`, `"meta"` but this can be extended for more complex cases), declare which databases it needs and what it will return to the caller.

The splitter is called with the `document` dictionary as argument, containing the `wants_from_db` data. MDAT expects the keys listed in `returns` to be present in the dictionary returned by the splitter.

If the strategy returns chunks (the most common scenario for transformer training), MDAT takes care of re-joining these chunks according to the model's sequence length. The model's sequence length must be a multiple of `chunk_size`. A change of context length does not require retokenizing the dataset.

The built-in splitters are in `matformer/data/mdat/splitters`. A custom splitter is a class in the `functions/` folder of the container; `splitter_class` names it.

Soon, MDAT will be expanded to handle multimodal data, such as audio, image or other kind of data.

After you've set a strategy, it's time to pretokenize the dataset. This calls the splitter for each dataset element and caches its result in the `pretok` folder. The `pretok` folder has one subfolder per strategy and dataset. `sync_strategy` pretokenizes only the rows appended after the first pretokenization; `desync` lists the missing ranges.

The shell and the TUI in `matformer/data/mdat/tui.py` expose the same commands; the [[tools/commands | command reference]] lists them. The shell registers and activates strategies:

```bash
strat-add minerva512 /configs/minerva_512.json
pretok oscar_it --strategy minerva512
set-strategy minerva512 --seq-len 4096
```

`pretok-all` runs a strategy over every dataset. `--procs` sets the worker count and `--resume` continues an interrupted run.

## Views and shuffling

The final steps are to **shuffle** the view and prepare for multi-gpu training (if necessary).

The shuffling doesn't shuffle the actual file. It just creates a very compact file with the permuted indexes of the datasets. This means that it is possible to shuffle the dataset very fastly, without moving any actual data.

This makes possible to create many coexisting **views** of the dataset that can be shuffled and, eventually, pretokenized independently from each other.

A view is a dictionary:

```json
{
  "view_name": "small",
  "selection": {
    "wikipedia": "2G",
    "finepdf": "1G",
    "laws": "500M"
  },
  "shuffle": true
}
```

The `selection` field must include every dataset of the container. Each value can be:

* an integer: the number of documents;
* a float between 0 and 1: a proportion of the documents;
* a string ending in `K`, `M`, `G` or `T`: a size in bytes of the raw `data` field;
* `preserved`: every document;
* `skipped`: no document.

A partial selection forces shuffling. The pointer file is written under `shuffling/<view>.mdat`; its width follows the catalog's `dsmap_bits` and the largest dataset. `use VIEW` activates a view, `shuffle [VIEW]` creates or recreates its pointer file, and `view-add`, `view-del`, `head` and `range` create and remove views.

You can for example create a tiny view:

```json
{
  "wikipedia": "2G",
  "finepdf": "1G",
  "laws": "500M"
}
```

for a fast ablation study; or you can decide to shuffle and pretokenize the complete `default` view (containing all the samples).

For distributed training, `precompute-dist NUM_WORKERS` writes the per-worker lengths once, and `dist-len NUM_DEVICES` returns the length for a given device count. `precompute-dist` uses a highly composite worker count to keep the partitions balanced.

## Iteration

The container can yield documents, tokens, chunks, chunked tokens or recurrent batches:

| Modality | Output |
|----------|--------|
| `document` | raw document, optionally with metadata |
| `tokens` | pretokenized token stream |
| `chunks` | chunk list |
| `chunked_tokens` | chunks joined to the model sequence length |
| `chunked_for_recurrence` | chunked tokens split into recurrent batches |
| `strategy_default` | the strategy's default output |

`MDATDataModule` uses `chunked_tokens` by default. Its data section contains `mdat_strategy`, `mdat_view`, `iteration_modality`, the batch size and the partition fields.

`export VIEW OUTFILE --limit N` writes a view to JSONL, one document per line. `state-save` and `state-load` store and restore the iteration position.

## Reading and writing rows from Python

An mdat can be treated like a JSONL file when only the `data` and `meta` databases matter. The two helpers are exported by `matformer.data.mdat`:

```python
from matformer.data.mdat import mdat_writer, mdat_reader

with mdat_writer("corpus.mdat", data_column="text") as md:
    for document in documents:
        md.write(document)
```

The writer creates a standalone dataset when the path does not exist. The data column (default `text`) goes into the raw `data` database; every other field is serialized into `meta`. Appending to an existing dataset is the default. If the path is a container with more than one dataset, pass `dataset="NAME"`; a missing dataset is created.

Reading streams in order without loading the dataset into memory:

```python
with mdat_reader("corpus.mdat") as md:
    for document in md:
        print(document)
```

`keys_only=True` reads only the metadata, `include_key=True` adds the sequential row key, and `data_column=...` renames the raw column in the yielded dictionaries. `mdat_writer(..., keys_only=True)` (or `data_column=None`) creates a metadata-only dataset with no `data` database.

### Editing and deleting rows

Keys are sequential integers, so the writer exposes two operations that deliberately break pretokenization:

```python
with mdat_writer("corpus.mdat") as md:
    md.write({"text": "new version", "label": 1}, key=4)   # overwrite row 4
    md.update(7, {"label": 2})                             # partial metadata edit
    md.edit({8: {"label": 3}, 9: {"label": 4}})            # several rows at once
    md.delete([3, 5])                                      # delete and compact keys
```

`delete` physically removes the rows and shifts every following row down, so keys remain `0..len-1`: iteration, views and shuffle files stay consistent. Because the bytes of every key can change, deletion and edit reset the coverage and statistics of every pretokenization strategy; run `sync_strategy` (or `pretok`) again afterwards. Appending new rows does not invalidate existing pretokenization and can be repaired with `sync_strategy`.

## Annotating an mdat

`annotate` accepts an mdat folder in addition to JSONL, CSV and CoNLL files. The input can be a standalone dataset or a container plus `--dataset NAME` (or `PATH::NAME`). The output format is chosen automatically: JSONL for file inputs, mdat for mdat inputs; override it with `--format jsonl|mdat`.

```bash
annotate corpus.mdat                       # -> corpus.mdat.annotated.mdat
annotate corpus.mdat -o labels.mdat --keys-only
annotate big_container --dataset train -f jsonl -o predictions.jsonl
```

With `--keys-only` the output is a metadata-only mdat with one record per source row: the original `key` plus the fields returned by the task, without copying the raw documents:

```
key  label
0    A
1    A
2    B
```

The full (non keys-only) output preserves the source metadata and text and adds the predictions to it. In both cases the source key is stored in the `key` field.

## MDAT-saetta

However, this comes with a drawback: even though the fast LMDB is used as datasets' backend, random access is usually much slower than sequential access. Mdat is great for experimentations, but it could become a bottleneck on some systems. In this case, the solution is **"Mdat-saetta"**!

A pretokenized, sharded and shuffled view can indeed be exported to a compact "saetta" file, ready to be read sequentially. *(Feature easy to implement but still WIP)*

## Commands

The MDAT shell is started with `python -m matformer.data.mdat.tui PATH`. The generated command list is in [[tools/commands | Commands]].
