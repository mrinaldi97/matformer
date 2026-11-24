---
title: Limitations
sources:
  - matformer/training/matformer_trainer.py
  - matformer/transformer_blocks/sequence_mixer/attention.py
---
# Limitations

Matformer is not released yet. Several parts are still being developed.

* Model parallelism is not implemented. The trainer uses `torch.fabric` with the `ddp` strategy; `parallel_strategy` accepts only `ddp`.
* `attention.py` is being reworked. The attention documentation is intentionally empty until the mixer API is stable.
* Multimodal support is in development. The `modalities` field and the modality field of a strategy exist, but the data path for audio and images is not complete.
* `MDAT-saetta`, the sequential export of a pretokenized view, is not implemented.
* The GUI is being developed and is not documented in detail. The server is described in [[tools/server | Server]]; its command interface is generated in [[tools/commands | Commands]].
* The `matformer/server`, `matformer/gui` and `matformer/remote` interfaces and command names may change.
* Some configuration comments and docstrings are still incomplete. The generated reference shows every field, including fields whose comment is missing.
