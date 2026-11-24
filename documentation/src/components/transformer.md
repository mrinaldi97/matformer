---
title: Transformer architecture
sources:
  - matformer/transformer_blocks/transformer.py
  - matformer/transformer_blocks/heads.py
category_docs: [model, sequence_mixer]
---
# Transformer architecture

Matformer defines several blocks of the transformer model that can be used to define new models.

The single, highly customizable **TransformerBlock** is the core of any transformer model. Each layer is customizable independently and it contains:

- A configurable normalization (pre or post) such as LayerNorm or RMSNorm
- A sequence mixer, such as multihead self-attention layer
- A feed-forward (MLP) layer
- A plugin system ready for your wildest experimentations

These blocks can either be used independently if you are building a new model from scratch or composed in a **NakedTransformer** module.

The Naked Transformer is composed of N transformer blocks but it misses embedding and unembedding layers. This was designed to allow for models different from the classic tokenization → model → de-tokenization flow, for example for tokenizer-free models or multimodal models.

If you need the most common scenario (embedding head or Lmhead) you can just use the **TransformerWithEmbeddingHead** or **TransformerWithLMHead** classes. [[components/transformer#languagemodel | LanguageModel]] combines them for training and generation.

The TransformerWithLMHead is ready for Autoregressive or Masked models: Matformer comes with two predefined implementations: **AutoregressiveModel** and **BERTModel** ready for these two common scenario.

After you've composed your module, you just have to wrap it in the [[training/wrapper | ModelWrapper]] class, that handles all the training related part. Training is by default managed by PyTorch Lightning that provides nice utilities for important aspects such as mixed-precision or multi-gpu training.

Currently, the ModelWrapper supports Adam, AdamW and Muon as optimizers.

## Model registration

Models should be registered in the `model` category of the [[components/registry | MatformerRegistry]] using as class decorator `@registry.register("model", "model_name", "default")` and they should subclass [[components/matformer-module | MatformerModule]]. For the development of transformer models, however, it is possible to use the classes already available for this family of models.

Matformer's transformer model implementation tries to be the most abstract and compatible possible. Indeed, at the current state, it is already able to correctly define all the most important families of common open weights LLMs and this abstract design pattern allow the experimenter to have a great flexibility in defining transformer models. (see [[dataclass:model.default | model dataclasses]]) If this flexibility is not enough, it is always possible to expand the definition of the models by writing [[components/plugins | Plugins]] or, if even this is not enough, neural modules can be added into the registry and new models registered in the `model` category.

Transformer's implementations lives in the `transformer_blocks` [[dir:matformer/transformer_blocks | folder]] and, in particular, in the files [[file:matformer/transformer_blocks/transformer.py | transformer.py]], which defines the core transformer blocks, and [[file:matformer/transformer_blocks/heads.py | heads.py]] containing input and output heads required for language modelling. Multimodal support is in development. Full models, which subclass the core components contained in `transformer.py` and `heads.py`, can be found in the `transformer_blocks/models` folder, and are directly explorable using the [[registry:model | model registry]].

## NakedTransformer

[[file:matformer/transformer_blocks/transformer.py:NakedTransformer | NakedTransformer]] defines a transformer backbone without input and output heads. With the default configuration settings, it works like any standard transformer implementation: it instantiates a list of [[file:matformer/transformer_blocks/transformer.py:TransformerBlock | TransformerBlock]] and passes the output of each block to the input of the next one. However, it contains useful features such a flexible `layer_schedule`, which gives the native support for looped models, and the possibility of asking directly to the forward to have the signal to entry at a specific transformer block and exit before the full list is iterated. The, optional, final norm, with or without bias, is configurable.

The NakedTransformer defaults to a standard [[registry:stream.residual_stream | additive residual stream]] but this is configurable too (see the [[components/sockets-and-stream | Socket, Plugins & Stream]] section). The NakedTransformer exposes the following sockets:

* `layer_{i}` => by default, mounted with the i-esim TransformerBlock
* `final_norm` => if configured, mounted to the final normalisation, Identity if empty
* `before_unembed` => an optional, defaulting to Identity, socket useful for interpretability or extra functionalities. Executed after the `final_norm`.

## TransformerBlock

Each TransformerBlock can be configured with a [[registry:norm_style | normalisation style]] so that it can support pre-norm, post-norm, sandwich norm and personalized normalization position. It creates a stream with an [[registry:mlp | MLP]], for example a SwiGLU block, a [[registry:sequence_mixer | sequence_mixer]], such as a [[components/transformer#attention | self-attention]] block, and, for each normalisation slot configured in the normalisation style implementation, a [[registry:normalization | normalisation]] block (ex. RMSNorm).

The forward pass executes [[components/sockets-and-stream | sockets]] in this order:

* pre_attn
* self_attn
* post_attn
* post_add_attn
* pre_mlp
* mlp
* post_mlp
* post_add_mlp
* layer_output

### Branch and merge

A `Stream` holds the current value and the sockets it applies. `stream.branch()` opens a child stream, the child applies its sockets to its own copy, and when the context closes `merge(branch)` combines the child back into the parent. For the default `ResidualStream`, `merge` is `self.x = self.x + branch.x`. The attention branch is computed in a child stream and added to the trunk; the MLP branch is computed in a second child stream and added to the same trunk.

A different stream type can replace or gate the merge. The `stream_type` field of a `LayerConfig` selects one from the `stream` category, and the block asks the registry for it. The block's forward does not change.

## Attention

The attention section is intentionally empty for now. `attention.py` is going to be reworked, so any internal description would be obsolete. The stable part is the contract: the mixer receives a [[components/tensors | TensorDC]], the batch geometry and the cache, and returns a `TensorDC` with the same leading dimensions. The mixer asks the registry for the kernel. See [[registry:sequence_mixer | the registered sequence mixers]].

<a id="transformerwithembeddinghead"></a>
<a id="transformerwithlmhead"></a>
## Heads

`TransformerWithEmbeddingHead` is the input side. It creates the token embedding, optionally scales it (Gemma-style), optionally adds learnable positions (the `learnable` positional encoding) or token-type embeddings (BERT-style), applies an optional embedding norm, and then delegates to the NakedTransformer. `TransformerWithLMHead` is the output side: it optionally applies a BERT-style `dense -> gelu -> LayerNorm` transform, then the linear projection to the vocabulary, optionally tied to the input embedding, optionally soft-capped with `final_logit_softcapping`.

The NakedTransformer has no embedding and no unembedding. That allows a tokenizer-free model, a multimodal model and a classification model to share the same block code.

<a id="languagemodel"></a>
## Classification and language models

`TransformerWithClassificationHead` builds a `TransformerWithLMHead` and asks for the hidden states instead of the logits. The classification head is a [[components/plugins | plugin]] mounted on `*.final_norm`, so the pooling, the dropout and the optional extra transformer layers are configuration. `MultiProbeClassifier` does the same with one probe per target socket and returns `{address: logits}`.

`LanguageModel` is the general language model: the embedding head, the NakedTransformer and the LM head, plus the generation mixin (autoregressive) and the masked-inference helpers (BERT-style). The training objective is selected by the [[training/learning-signals | learning signal]].

## Layer schedule

`config.layer_schedule` is a list of layer indices, repeated as many times as the model should visit them. `None` means `[0, 1, ..., num_hidden_layers - 1]`. A schedule can loop: `[0, 1, 2, 3, 2, 1]` visits four layers six times, in that order. The forward also accepts `start` and `stop`: `start` enters at the first visit of that layer, `stop` exits after its last visit.
