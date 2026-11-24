---
title: Data module
sources:
  - matformer/data/matformer_data_module.py
  - matformer/dataclasses/data_dataclasses.py
---
# Data module

`MatformerDataModule` is the base class of every data module. It defines `setup`, `train_dataloader`, `__len__`, `state_dict` and `load_state_dict`, and it provides the shared `collate_fn`.

## Contract

A subclass:

1. calls `super().__init__(config, num_devices)` and defines `required_keys`;
2. calls `self._assertions(name, required_keys)` in `__init__`, so a missing field in the data section is reported before the dataset is built;
3. implements `setup(stage=None)` and calls `super().setup(stage)` first;
4. implements `train_dataloader()`, returning a `DataLoader` with `batch_size=self.config.batch_size` and `collate_fn=self.collate_fn`;
5. implements `__len__`, `state_dict()` and `load_state_dict()`.

`__len__` returns the number of examples on this rank, already divided by the world size. The trainer uses it to compute the total steps. If the length is not known, `__len__` returns `0` and the trainer disables LR scheduling.

`state_dict` and `load_state_dict` use `_gather_state` and `_restore_state`. Every rank sends its dataset position to rank 0, and a resume returns each rank its position. A resume with a different world size is accepted; the positions may overlap or skip a few examples, and the epoch still terminates.

## Configuration

`DataConfig` contains `data_root`, `batch_size`, `num_workers`, `keep_metadata`, `head_samples`, `partition_start`, `partition_end`, and the linked `tokenizer`, `model` and `training` sections. The subclasses add their own fields:

* `LazyDataModuleConfig`: `file_format`, `text_column`, `input_ids_column`, `tokenize`, `add_bos`, `add_eos`, `split`, `max_seq_len`, `hf_config`, `varlen_strategy`, `task`, `label_column`, `label2id`, `id2label`, `scan_labels`, `missing_label_values`.
* `MDATDataModuleConfig`: `mdat_strategy`, `mdat_view`, `iteration_modality`, and the partition fields.

When a model is loaded, the trainer copies `model.max_position_embeddings` into `config.max_seq_len` if the data config does not set it. It also copies `training.batch_size` and `training.varlen_strategy` into the data config for the run.

## collate_fn

`collate_fn` receives a list of items and returns a `TensorDC`.

With `varlen_strategy == "unpadding"` it returns an `UnpaddedTensor`: the sequences are concatenated into one tensor, and `indices`, `cu_seqlens`, `max_seq_len` and `original_seq_len` describe the boundaries. With any other value it returns a `PaddedTensor`: sequences are padded to `max_seq_len` with `pad_token_id`, and `padding_mask` is `True` on the pads.

A sequence label becomes `attributes["labels"]`, a tensor of shape `[B]` or `[B, ...]`. A token label becomes `following_attributes["labels"]`, so it is padded, unpadded and reversed with the tokens. A token-level batch must be either fully labelled or fully unlabelled; a mix raises `ValueError`.

`is_same_document` becomes `same_document_mask`, a boolean tensor of shape `[B]`. If no item supplies the field, the tensor is absent.

`metadata` is collected when `config.keep_metadata` is true and stored in `attributes["metadata"]`.

If an item sets `worker_has_finished`, the batch carries `attributes["worker_has_finished"] = True`. The wrapper returns a zero loss for that batch.

## Subclassing

```python
@registry.register("data", "my_data", "default")
class MyDataModule(MatformerDataModule):
    required_keys = ["data_root", "batch_size"]

    def __init__(self, config, num_devices=1, *args, **kwargs):
        super().__init__(config, num_devices)
        self._assertions("my_data", self.required_keys)
        self.dataset = None

    def setup(self, stage=None):
        super().setup(stage)
        self.dataset = MyDataset(self.config.data_root)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.config.batch_size,
                          collate_fn=self.collate_fn, shuffle=False)

    def __len__(self):
        return len(self.dataset)

    def state_dict(self):
        return self._gather_state({"position": self.dataset.position})

    def load_state_dict(self, state):
        self._restore_state(state, lambda one: setattr(self.dataset, "position", one["position"]))
```

The registry makes the module available as `data: {type: my_data, ...}`. The trainer and the tasks use it through the `data` category.
