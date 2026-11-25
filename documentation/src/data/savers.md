---
title: Savers
sources:
  - matformer/data/savers/saver.py
  - matformer/data/savers/jsonl.py
  - matformer/data/savers/memory.py
  - matformer/dataclasses/saver_dataclasses.py
---
# Savers

A saver receives the output of a [[tasks/index | task]]. It is selected by the `ref_saver` field of the task config and registered in the `saver` category.

```yaml
saver:
  - name: results
    type: jsonl
    filename: predictions.jsonl
    summary_file: summary.jsonl

task:
  - name: test
    type: evaluation
    ref_saver: results
```

## Interface

`MatformerDataSaver` defines three methods:

| Method | When |
|--------|------|
| `save_batch(records, rank=0)` | after every batch, one record per item |
| `save_summary(summary, rank=0)` | once at the end, with the output of `compute()` |
| `end()` | when the task is done |

All ranks call these methods; rank 0 writes. The JSONL saver gathers records with `torch.distributed.gather_object`, so the output follows rank order. A run without `torch.distributed` has one rank.

## Built-in savers

* `jsonl`: one JSON object per line. It uses `orjson` when available and the standard library otherwise. Tensors and arrays become nested lists. `filename` and `summary_file` are independent, and `mode` and `summary_mode` control append or overwrite.
* `NullSaver`: writes nothing.
* `memory`: keeps records in an in-process bucket, keyed by the section name or by `key`.

## Record contents

`MatformerTask.save_batch` builds the record before the saver receives it. Each record has an `id`. The result fields follow the task's `save_*` flags. `input_ids` is added when `save_input_ids` is true, and `attributes` is added when `save_attributes` is true. `flatten_dict` merges a dictionary result into the record.

The saver writes the fields it receives. A custom saver can change the output format without changing the task.

## Writing a saver

Subclass `MatformerDataSaver`, register it, and add a dataclass if it has options:

```python
@registry.register_dataclass("saver", "MySaver", default=False)
@dataclass
class MySaverConfig(SaverConfig):
    folder: str = "output"

@registry.register("saver", "MySaver", "default")
class MySaver(MatformerDataSaver):
    def __init__(self, config):
        self.folder = Path(config.folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def save_batch(self, records, rank=0):
        if rank != 0:
            return
        with open(self.folder / "batch.jsonl", "a") as handle:
            for record in records:
                handle.write(json.dumps(record, default=str) + "\n")

    def save_summary(self, summary, rank=0):
        if rank == 0:
            (self.folder / "summary.json").write_text(json.dumps(summary, default=str))
```

The `config` argument is the dataclass instance. The task checks `save_batch` and `save_summary` before it calls the saver.
