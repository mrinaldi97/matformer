---
title: Source layout
sources:
  - matformer/transformer_blocks/__init__.py
  - matformer/modules/stream.py
  - matformer/modules/norm_positions.py
---
# Source layout

This folder contains the implementation of the neural network components of Matformer.

[!WarningBox] The annotations at the top of each class are described in [[components/matformer-module#class-annotations | MatformerModule]].
[!WarningBox] The `registry.create(...)` calls instantiate modules from the [[components/registry | registry]].

* `attention.py`: multi-head attention without a fixed kernel. The kernel is selected by the registry. The file also contains the interpretability-friendly attention implementation, which exposes the weights of each head for QK/OV studies. The attention documentation is intentionally empty until the mixer is reworked.

* `transformer.py`: [[components/transformer#nakedtransformer | NakedTransformer]] and [[components/transformer#transformerblock | TransformerBlock]]. NakedTransformer stacks the blocks described by the config. TransformerBlock supports pre-normalization, post-normalization and the other registered normalisation styles.

* `heads.py`: input and output heads. [[components/transformer#transformerwithembeddinghead | TransformerWithEmbeddingHead]] adds a linear embedding head to the NakedTransformer, with optional learnable positional embeddings. [[components/transformer#transformerwithlmhead | TransformerWithLMHead]] adds an LM head and supports weight tying.

* `mixin/`: model functions such as autoregressive generation (`GenerationMixin`) and masked-language-model inference.

* `models/`: the complete models. [[components/transformer#languagemodel | LanguageModel]] covers GPT-like and BERT-like models; `TransformerWithClassificationHead` and `MultiProbeClassifier` add classification heads through plugins.

* `sequence_mixer/`: attention and the other mixers. A sequence mixer receives a `TensorDC` and returns a `TensorDC` with the same leading dimensions. The `sequence_mixer` field of a `LayerConfig` selects it.

* `matformer/modules/norm_positions.py`: the normalisation styles (`pre`, `post`, `inner`, `pre_no_attn`, `sandwich`). A custom style is a registry entry.

All implementations in this folder are reached through the registry. Model code does not import a concrete kernel.
