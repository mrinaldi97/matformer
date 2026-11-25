---
title: The Maskerator
sources:
  - matformer/utils/masked_models.py
  - matformer/dataclasses/model_dataclasses.py
---
# The Maskerator

`Maskerator` creates a masked sequence and records the positions the model should predict. It is used by the [[training/learning-signals | masked learning signal]] and by the [[tasks/built-in-tasks#masking | masking task]], and it accepts plain tensors and [[components/tensors | TensorDC]] wrappers.

## Substitutions

For every selected position the maskerator uses one of three substitutions:

* **cloze**: replace the token with the mask token (`cloze_prob`);
* **random**: replace it with a random vocabulary id (`random_prob`);
* **same**: keep the token but still ask the model to predict it (`same_prob`).

The three probabilities must sum to one. If `random_prob` and `same_prob` are both `None`, the maskerator is in pure-cloze mode and does not need `vocab_size`. As soon as one of them is set, `vocab_size` is required, because the random branch has to draw from somewhere.

## The masking rate

`substitution_rate` accepts three shapes:

* a float: a fixed fraction of the tokens is selected;
* a pair `(lower, upper)`: a fraction is sampled uniformly in that range for every batch;
* `None`: the caller decides the rate and passes it to the call, which is how a curriculum or a memory probe can control it step by step.

`variable_masking_rate` decides whether the sampled rate is one per batch or one per document. `random_seed` seeds both the Python RNG and the per-device `torch.Generator`, so a run is reproducible without touching the global RNG.

## The cloze mask

The output is `(masked_sequence, rate)` when the input is a `TensorDC`. The selection mask is stored in `following_attributes["cloze_mask"]`; it has the same leading dimensions as the tokens, so it is padded, unpadded and reversed with them. The learning signal then intersects it with the padding mask and uses the result to select the positions that contribute to the loss.

A `PaddedTensor` and a `NormalTensor` go through the plain-tensor path and are rebuilt with `replace`, while an `UnpaddedTensor` is masked directly on the packed layout: there is no padding to respect, and `cu_seqlens` already says where each document ends.

## Scope

The maskerator receives `pad_token_id` from the caller and does not inspect the model or the loss. The same object is used by a BERT-style model, by a token-classification cloze test and by the trainer's memory probe.
