# Matformer

> **WARNING:** Matformer hasn't been released yet. It works, but it's still Work In Progress.

Matformer is a library designed to train, finetune and execute state-of-the-art transformer models.

Unlike common implementations, such as HuggingFace's transformers or Nvidia's Megatron, it is designed from scratch with simplicity and modularity in mind but without renouncing to all the possible improvements necessary to make transformers' training more efficient.

As the research goes on, several novelties are introduced in the libraries. However, this often leads to an explosion of their complexity, where new functions are introduced abruptely in the codebase, for example by introducing many conditional branches or by repeating many times entire part of the code, breaking the DRY principles.

The goal of Matformer is to produce a very compact and clear code, with the smallest possible amount of repetition, in line with the principles of software design. Matformer tries to hard-code as few elements as possible: it is easily extensible with additional modules that allows important modification to the models' definitions without altering the core parts of the library.

Matformers' core is so easy to read that it is ready to be used also for didactical purposes: teachers are given with the possibility to present to their student an easy to understand codebase and guide them in the understanding of how a modern transformer model works.

Matformer's main focus is on elegancy and readability of the code, an aspect often ignored in the world of codebases for deep-learning models' training.

This allows a twofold use of Matformer: on the one hand, it is a library ready for the development of models with many different possible configurations ecompassing the most common scenario adopted in 2025 for training models; on the other hand, it is a library designed from the Academia to the Academia: this means that it gives to researchers from all over the world an easily hackable transformer implementation ready to embrace many possible experimental scenarios. If, let's say, a particular custom module has to be added or modifed with respect to the base Transformer design, it is not necessary to alter even a single row of code of the library itself, but the module can be easily imported and used through the configuration files. Matformer will also take care to pack the custom code in the models'repository ready to be uploaded in platforms such as HuggingFace.

This doesn't come at expenses of speed: it is very important to exploit as much as possible the capabilities of the hardware when training complex models such as transformers. That's why Matformer already contains the code necessary to maximize training's performances, such as FlashAttention, sequence packing and high-performance fused kernels. Currently Data Parallelism is fully supported thanks to Pytorch Lightning while model parallelism is still WIP.

Altough now available only for text, its modularity allows extension to multimodality scenario. This is one of the top priorities in the library's development.

## Features

| Category | Features |
|----------|----------|
| **Attention Implementations** | Flash Attention, SDPA, Flex Attention, WERSA |
| **Positional Encodings** | Learnable, RoPE, ALiBi, Sinusoidal, NoPE |
| **Normalizations** | LayerNorm, RMSNorm |
| **FFN Activations** | GELU, SwiGLU, GEGLU |
| **Training Objectives** | Causal (GPT-style), Masked (BERT-style) |
| **Optimizers** | Adam, AdamW, Muon |
| **Parallelism** | Data Parallelism (fully supported), Model Parallelism (WIP) |
| **Performance Features** | FlashAttention, Sequence Packing, Fused Kernels, Mixed Precision |
| **Extensibility** | Plugin System via Registry, Custom Hooks, External Modules |
| **Dataset Format** | MDAT with pretokenization, views, and shuffling |
| **Framework** | PyTorch Lightning |

## Table of Contents

- [Features](#features)
- [Getting Started: From Zero to a Trained Model](#getting-started-from-zero-to-a-trained-model)
- [The Matformer Basic Architecture](#the-matformer-basic-architecture)
- [The Matformer Registry: Customization of Modules](#the-matformer-registry-customization-of-modules)
  - [Registry Hierarchy](#registry-hierarchy)
  - [Add a Custom Implementation](#add-a-custom-implementation)
  - [Priority and Preference Override](#priority-and-preference-override)
  - [Adding Custom Modules to the Model](#adding-custom-modules-to-the-model)
- [Configuration Guide](#configuration-guide)
  - [Model Configuration](#1-model-configuration-model_config)
  - [Training Configuration](#2-training-configuration-training)
  - [Tokenizer Configuration](#3-tokenizer-configuration-tokenizer)
  - [Data Configuration](#4-data-configuration-data)
  - [Logging & Checkpoints](#5-logging--checkpoints)
- [MDAT: The Matformer Dataset](#mdat-the-matformer-dataset)

---

## Getting Started: From Zero to a Trained Model

1. Create an MDAT from an existing dataset.

---

## The Matformer Basic Architecture

Matformer defines several blocks of the transformer model that can be used to define new models.

The single, highly customizable **TransformerBlock** is the core of any transformer model. Each layer is customizable independently and it contains:

- A configurable normalization (pre or post) such as LayerNorm or RMSNorm
- A multihead self-attention layer
- A feed-forward (MLP) layer. Currently supported implementations: Swiglu and Geglu, but they are easily expandable
- A [hook system](#adding-custom-modules-to-the-model) ready for your wildest experimentations

These blocks can either be used independently if you are building a new model from scratch or composed in a **NakedTransformer** module.

The Naked Transformer is composed of N transformer blocks but it misses embedding and unembedding layers. This was designed to allow for models different from the classic tokenization → model → de-tokenization flow, for example for tokenizer-free models or multimodal models.

If you need the most common scenario (embedding head or Lmhead) you can just use the **TransformerWithEmbeddingHead** or **TransformerWithLMHead** classes.

The TransformerWithLMHead is ready for Autoregressive or Masked models: Matformer comes with two predefined implementations: **AutoregressiveModel** and **BERTModel** ready for these two common scenario.

After you've composed your module, you just have to wrap it in the **ModelWrapper** class, that handles all the training related part. Training is by default managed by PyTorch Lightning that provides nice utilities for important aspects such as mixed-precision or multi-gpu training.

Currently, the ModelWrapper supports Adam, AdamW and Muon as optimizers.

---

## The Matformer Registry: Customization of Modules

The Matformer architecture tries to avoid every direct instantiation of hard-coded classes of neural network components. Instead, every implementation of the core components is managed by the MatformerRegistry. This means that it is possible to define modules for every component of the model, such as MLP, attention implementations, layer normalizations...

In this way the model is not bound to specific implementation (such as vanilla PyTorch) but it is immediately possible to adopt different kernels without touching a single line of the core transformer implementation.

This gives extreme flexibility to the developer: just by putting python files in the "user-modules" folder and decorating the class with a decorator it is possible to substitute every module of the network.

By using the registry it is also possible to import customized experimental modules that can be hooked to many different entry points of the model, and, if necessary, they can be trained together with the rest of the network!

Adding custom components to a transformer model has never been so easy.

### Registry Hierarchy

The registry has the following hierarchy:

```
_registry (dict)
    └── category (dict) <= example "attention","norm","linear","embedding","mlp","loss"
        └── variant (dict) <= example "geglu","swiglu","rmsnorm","flash","gated_deltanet","mamba"...
            └── name (dict) <= example "torch","liger","nvidia-transformer-engine"... this is the name of the actual implementation
                └── name contains: 
                    * requires => List of required modules, useful for auto-fallback if not supported
                    * priority => The priority of different variants, ex torch 0, liger 1, nvidia 2 and so on
                    * metadata => Free dictionary of metadata
                    * params_names => *IMPORTANT* A mapping that renames the parameters to create universal checkpoints compatible also
                                        with different implementations
```

### Add a Custom Implementation

Put your script in the external_modules folder.

Then, you don't have to do anything by hand, as the registration is handled by an easy to use decorator, example:

```python
from matformer.matformer_registry import registry

@registry.register("mlp", "swiglu", "a_name_for_your_swiglu_kernel",
       params_names={'w13.weight': 'gate_proj.weight', 
                      'w2.weight': 'down_proj.weight'}, 
       priority=100,
       requires=["torch","my_custom_library"])
class MySwigluImplementation(nn.Module):
    # (your code here)
    def _is_available():
        # Test logic here, for auto-fallback
        pass
```

The MatformerRegister works in tandem with the CachedStuff object, thus it's very simple to define a module and let Matformer handle the picking of correct implementation for you:

```python
cache.registry.create("mlp", "swiglu", hidden_size=768, ffn_factor=4)
```

These are the only requirements to seamless integrate different custom implementation of basic modules into a Matformer model! Just be careful to match conventions for param_names if you want that your model correctly loads also with other implementation.

### Priority and Preference Override

By default, the registry selects implementations based on their priority value (higher priority wins), with automatic fallback to lower-priority alternatives if the preferred one is unavailable. You can override this behavior globally using `registry.set_preferences({"category": {"variant": "name"}})` or per-call with environment variables like `MATFORMER_ATTENTION_FLASH=triton`, which takes precedence over both config preferences and default priorities.

### Adding Custom Modules to the Model

The registry can also be used to add hooked modules/functions to any part of the model.

The main transformer block exposes the following entry points:

- `pre_attn`
- `pre_mlp`
- `post_mlp`
- `pre_output`

Let's say you want to add a custom module before the mlp. What you need to do is just to insert into the "external modules" directory a file `my_module.py`:

```python
from matformer.matformer_registry import registry

@register.registry("hook","name_of_my_hook","default")
class ACustomModule(nn.Module):
    # (your code...)
```

Then, in the model config's JSON file the hook should be attached in this way:

```json
{
  "default_layer": {
    "hooks": {"pre_mlp": "name_of_my_hook"}
  }
}
```

If the modules contains trainable parameters, it will be trained together with the model.

That's it!

---

## Configuration Guide

### Configuration File Structure

The config JSON has 5 main sections:

### 1. Model Configuration (`model_config`)

**Basic Architecture:**

```json
{
  "hidden_size": 512,
  "num_hidden_layers": 4,
  "num_attention_heads": 8,
  "ffn_factor": 1.0,
  "max_position_embeddings": 1024
}
```

| Parameter | Description |
|-----------|-------------|
| `hidden_size` | Model dimension |
| `num_hidden_layers` | Number of transformer layers |
| `num_attention_heads` | Attention heads (must divide hidden_size) |
| `ffn_factor` | FFN size = hidden_size × ffn_factor |
| `max_position_embeddings` | Maximum sequence length |

**Vocabulary & Tokens:**

```json
{
  "vocab_size": 32769,
  "pad_token_id": 0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "mask_token_id": 32768
}
```

| Parameter | Description |
|-----------|-------------|
| `vocab_size` | Size of vocabulary |
| `pad_token_id` | Padding token |
| `bos_token_id` | Beginning of sequence |
| `eos_token_id` | End of sequence |
| `mask_token_id` | Mask token (for MLM) |

**Layer Configuration (`default_layer`):**

```json
{
  "attn_impl": "sdpa",
  "positional_encoding": "learnable",
  "normalization": "layernorm",
  "normalization_position": "pre",
  "ffn_activation": "gelu"
}
```

| Parameter | Options | Description |
|-----------|---------|-------------|
| `attn_impl` | `"flash"`, `"sdpa"`, `"flex"`, `"wersa"` | Attention implementation |
| `positional_encoding` | `"learnable"`, `"rope"`, `"alibi"`, `"sinusoidal"`, `"nope"` | Positional encoding type |
| `normalization` | `"layernorm"`, `"rmsnorm"` | Normalization type |
| `normalization_position` | `"pre"`, `"post"` | Where to apply normalization |
| `ffn_activation` | `"gelu"`, `"swiglu"` | FFN activation function |

**Training Objective:**

```json
{
  "training_objective": "masked",
  "is_causal": false,
  "masked_substitution_rate": 0.25
}
```

| Parameter | Options | Description |
|-----------|---------|-------------|
| `training_objective` | `"masked"`, `"causal"` | BERT-style MLM or GPT-style |
| `is_causal` | `false`, `true` | false for BERT, true for GPT |
| `masked_substitution_rate` | `0.0` - `1.0` | Percentage of tokens to mask (MLM only) |

### 2. Training Configuration (`training`)

```json
{
  "optimizer": "adam",
  "lr": 5e-4,
  "final_lr": 2e-5,
  "lr_scheduling": true,
  "scheduler": "custom",
  "warmup_steps": 750,
  "hold_steps": 300,
  "weight_decay": 0.01,
  "gradient_clip_val": 1.0,
  "max_epochs": 1,
  "accumulate_grad_batches": 5,
  "save_every_n_steps": 5000,
  "seed": 27
}
```

| Parameter | Description |
|-----------|-------------|
| `optimizer` | Optimizer type: `"adam"`, `"adamw"`, `"muon"` |
| `lr` | Learning rate |
| `final_lr` | Final LR for scheduler |
| `lr_scheduling` | Enable LR scheduling |
| `scheduler` | Scheduler type |
| `warmup_steps` | Warmup steps |
| `hold_steps` | Steps to hold at max LR |
| `weight_decay` | Weight decay |
| `gradient_clip_val` | Gradient clipping |
| `max_epochs` | Number of epochs |
| `accumulate_grad_batches` | Gradient accumulation |
| `save_every_n_steps` | Checkpoint frequency |
| `seed` | Random seed |

### 3. Tokenizer Configuration (`tokenizer`)

```json
{
  "type": "huggingface",
  "pretrained_name": "sapienzanlp/Minerva-350M-base-v1.0",
  "varlen_strategy": "padding"
}
```

| Parameter | Description |
|-----------|-------------|
| `type` | Tokenizer type |
| `pretrained_name` | HuggingFace model name |
| `varlen_strategy` | `"padding"` or other strategies |

### 4. Data Configuration (`data`)

```json
{
  "data_root": "./datasets/mini_dataset_1000",
  "batch_size": 4,
  "num_workers": 1,
  "mdat_strategy": "Minerva1024",
  "mdat_view": null
}
```

| Parameter | Description |
|-----------|-------------|
| `data_root` | Path to MDAT dataset |
| `batch_size` | Batch size |
| `num_workers` | DataLoader workers |
| `mdat_strategy` | Pretokenization strategy |
| `mdat_view` | Dataset view (optional) |

### 5. Logging & Checkpoints

```json
{
  "save_dir": "./checkpoints",
  "wandb_project": "Test",
  "wandb_run_name": "Test-Model"
}
```

| Parameter | Description |
|-----------|-------------|
| `save_dir` | Checkpoint directory |
| `wandb_project` | Weights & Biases project |
| `wandb_run_name` | W&B run name |

---

## MDAT: The Matformer Dataset

Matformer comes together with its own dataset format, designed by keeping in mind the need for easily getting a big dataset formed of many independent sub-dataset, as well as the importance in experimental settings to test different tokenization strategies and having different views of the dataset, such as a smaller dataset derived from the main one for ablation studies.

MDAT is a complex and extensible dataset manager that handles all of this for you. To understand how it works, it is important to familiarize with its components:

MDAT is the name of the main container. It presents itself as a folder with a specific structure:

```
datasets/
functions/
pretok/
shuffling/
views/
mdat.db
```

Each mdat is composed of one to many "submdats" that lives in the `datasets` folder. It is possible to add up to 255 submdats to an mdat or 65535 with the extend mode.

A submdat can be created from many datasets formats such as JSON or HugginFace. The dataset is automatically splitted into two files `data.db`, containing the actual raw data and `meta.db`, containing all the optional metadata for each dataset's raw. This means that a dataset with heavy metadata will not have an impact on actual data loading. Compression is optional.

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

As you can see, the core function called by any pretokenization strategy is the **splitter**, that can be easily customized. The splitter will receive data from the submdat (ex `"data"`, `"meta"` but this can be extended for more compelx cases), declare which databases will needs and what it will return to the caller (usually, a MatformerModel during training).

The splitter will be called with the "document" dictionary as arguments, containing the `wants_from_db` data, while Mdat expects the `returns` dictionary's keys to be present in the dictionary returned by the splitter.

If the strategy returns chunks (the most common scenario for transformer training), the MDAT will automatically take care of re-joining these chunks according to the models' sequence length. The only requirement is the model's sequence length to be a multiple of `chunk_size`. In this way, it is no longer necessary to re-tokenize a dataset if you decide to change your models' context length.

Soon, Mdat will be expanded to handle multimodal data, such as audio, image or other kind of data.

After you've set a strategy, it's time to pretokenize the dataset. This will call the splitter for each datasets' element and cache its result in the `pretok` folder. The MDAT is now almost ready for models'training.

The final steps are to **shuffle** the view and prepare for multi-gpu training (if necessary).

The shuffling doesn't shuffle the actual file. It just creates a very compact file with the permuted indexes of the datasets. This means that it is possible to shuffle the dataset very fastly, without moving any actual data.

Why? Because this makes possible to create many coexisting **"views"** of the dataset that can be shuffled and, eventually, pretokenized independently from each other.

You can for example create a tiny view:

```json
{
  "wikipedia": "2G",
  "finepdf": "1G",
  "laws": "500M"
}
```

for a fast ablation study; or you can decide to shuffle and pretokenize the complete "default" view (containing all the samples).

However, this comes with a drawback: even though the fast LMDB is used as datasets' backend, random access is usually much slower than sequential access. Mdat is great for experimentations, but it could become a bottleneck on some systems. In this case, the solution is **"Mdat-saetta"**!

A pretokenized, sharded and shuffled view can indeed be exported to a compact "saetta" file, ready to be read sequentially! *(Feature easy to implement but still WIP)*.
