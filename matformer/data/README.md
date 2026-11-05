# Data loading

MDAT keeps source documents in named **subdatasets** (domains, such as `wiki`
or `books`), token caches under named **strategies**, and document membership
under named **views**. A view can represent an existing train/validation split.
A **selection** filters one view at load time without rebuilding the corpus.

## Load a token-budgeted subset

```python
from matformer.data.mdat import MatformerDataset

with MatformerDataset.load_dataset(
    path="/path/to/corpus",
    readonly=True,
    ds_view="train",              # An existing view; omit for "default".
    strategy="Minerva1024",       # An existing pretokenization strategy.
    selection={
        "domains": ["wiki", "books"],
        "max_tokens": 10_000_000,
        "seed": 27,
    },
) as dataset:
    print(dataset.selection_summary)
    dataset.set_iteration_modality("chunked_tokens")
    for item in dataset:
        tokens = item["object"]
```

Use the names returned by `dataset.list_submdat()`, `dataset.list_views()`, and
`dataset.list_strategies()` for your corpus. No tokenizer or model download is
needed when reading cached tokens with standard chunking.

| Selection field | Meaning |
| --- | --- |
| `domains` | Names of subdatasets in the source view. Omit to include all domains, or the keys of `domain_tokens` when supplied. |
| `max_tokens` | Non-negative integer cap across all selected domains. Omit for no total cap. |
| `domain_tokens` | Mapping of domain names to independent token caps. With explicit `domains`, unlisted caps are unlimited. |
| `seed` | Integer for a reproducible, constant-memory permutation of candidate positions. Omit or use `null` to preserve the source order. |

For example, request up to 7M tokens from Wikipedia and 3M from books:

```python
selection = {
    "domain_tokens": {"wiki": 7_000_000, "books": 3_000_000},
    "seed": 27,
}
```

Whole documents are retained. If a document would exceed either cap, it is
skipped and scanning continues, allowing later smaller documents to fit.
Actual totals can be **below** the requested budget, including zero. These are
upper bounds, not exact mixture proportions. Inspect `selection_summary` for
the achieved per-domain document/token counts and a selection fingerprint.
Skipping documents to fit a budget can favor shorter documents near the limit.

Tokens mean **cached tokenizer token IDs**, before loader-added BOS/EOS tokens
and padding. They are not words, bytes, masked targets, or the total tokens
processed over repeated training epochs. A token budget requires a strategy
with stored tokens for every requested domain. Standard stored chunk boundaries
are supported; custom `splitter_decides_chunking` strategies are rejected for
selections because their output lengths cannot be inferred from these caches.

Selections always start from the exact membership of `ds_view`, including
partial views. Different seeds or budgets do not create disjoint train/test
splits: choose already-disjoint source views when separation is required.
The seeded order is an affine permutation, not a uniform random shuffle.
The same source view, caches, strategy, configuration, and seed give the same
selection. Changing those inputs can change membership.

## Training configuration

Add `mdat_selection` to the existing `data` section. Existing configurations
without this field continue to load their full named view.

```json
{
  "data": {
    "data_root": "/path/to/corpus",
    "batch_size": 48,
    "wanted_from_strategy": "chunked_tokens",
    "mdat_strategy": "Minerva1024",
    "mdat_view": "train",
    "mdat_selection": {
      "domains": ["wiki", "books"],
      "max_tokens": 10000000,
      "seed": 27
    },
    "prefetch_buffer": 256
  }
}
```

The same dictionary can be passed as `mdat_selection` to
`MatformerDataModule`. Set `prefetch_buffer` to zero to disable background
prefetching. The training data module uses one iterator per process
(`DataLoader(num_workers=0)`); additional DataLoader workers are unsupported.
Selected documents are assigned across distributed ranks without overlap;
epoch lengths use their actual chunk counts. Existing distributed end-of-data
sentinels remain in use to equalize rank iteration counts.

Selections support `document`, `tokens`, `chunks`, and `chunked_tokens`.
`chunked_for_recurrence` is rejected with a selection: the existing recurrent
loader may stop when one batch slot exhausts its documents while other slots
still contain chunks, so it cannot guarantee consuming the complete selection.
Its existing behavior without a selection is retained.

Each process scans the selected source view once at setup and writes compact
temporary pointer/count files, not document copies. This requires temporary
disk space proportional to selected document count (24 bytes per document).
Memory use does not grow with the number of documents. `close()`, the context
manager above, or `MatformerDataModule.teardown()` releases these resources.
Original SQLite, LMDB, and view files are not modified by read-only loading.
For very large corpora, setup still needs to read candidate token-cache entries;
the selection is rebuilt on reload, not persisted as a new named view.

`dataset.select({...})` replaces a temporary selection using the original active
view. `dataset.set_view(name)` clears it. Checkpoints store the selection
fingerprint and reject a different selection on resume. This fingerprint checks
membership/counts/configuration, not the entire corpus contents; use the same
unchanged corpus and tokenizer caches when resuming.

## Inspect a selection from the CLI

Save just the selection dictionary in `selection.json`, then run:

```sh
python -m matformer.data.mdat.user_interface load \
  --path /path/to/corpus --readonly --view train \
  --strategy Minerva1024 --selection-json selection.json
```

The existing `python -m matformer.data.mdat.mdat` and direct `mdat.py` entry
points remain available. Use `--help` to inspect the creation/import commands.
For Hugging Face imports through `convert_to_submdat`, choose a source split
with `dataset_args={"hf_split": "validation"}`.

## Composition and training budgets

`mdat_selection.composition` accepts percentage strings summing to 100%, together
with a total `max_tokens`: for example
`{"max_tokens": 1000000, "composition": {"wiki": "100%", "books": "0%"}}`.
Percentages are mixture shares of that total, not fractions of source corpora.
Selection keeps whole documents and may underfill; inspect the reported totals.
Do not combine composition with explicit `domain_tokens` caps.

The separate training target ceiling, packing semantics, checkpoint identity
checks, and bounded-storage data-stage workflow are described in
[the pretraining guide](../../docs/pretraining.md).

## Where the code lives

| File | Responsibility |
| --- | --- |
| `mdat/mdat.py` | Dataset orchestration, saved views, iteration, checkpoint state |
| `mdat/selection.py` | Domain/budget selection, temporary indexes, exact selected counts |
| `mdat/submdat.py` | One domain's data/cache access and source import |
| `mdat/sql_database.py` | SQLite schema and metadata queries |
| `mdat/lmdb.py` | LMDB byte storage and compression |
| `mdat/preprocessing.py` | Strategies, preprocessing workers, and built-in splitters |
| `mdat/user_interface.py` | CLI and interactive shell |
| `mdat/exceptions.py`, `mdat/utils.py` | Shared exceptions and small helpers |
| `matformer_data_module.py` | Training configuration, batching, tensor collation |

The existing files and on-disk format are retained. Historical named imports
from `matformer.data.mdat.mdat` remain available. Classification loaders keep
their existing interfaces; these selection options apply to MDAT pretraining.

Offline regression tests (using the normal data dependencies, no model):

```sh
python -m unittest discover -s tests -p 'test_mdat_selection.py' -v
```
