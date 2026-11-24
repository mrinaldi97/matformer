---
title: CheckpointHolder
sources:
  - matformer/loading/checkpoint_holder.py
  - matformer/loading/matformer_holder.py
  - matformer/loading/huggingface_holder.py
  - matformer/loading/utils.py
---
# CheckpointHolder

A `CheckpointHolder` reads one external checkpoint format. The wrapper, the trainer and `load_model` use the holder interface, so a Matformer checkpoint, a HuggingFace folder and a raw safetensors file are loaded through the same path.

The base class implements renaming, `N`-templated layer keys, weight transforms, lazy streaming and the missing/unexpected report. A subclass implements the format-specific parts:

* `_DEFS`: the folder with its adapter definitions (YAML);
* `__init__`: sets `source_config`, `definition` and `num_layers`;
* `_raw_weights`: returns a dict-like view of the raw source tensors;
* `get_configs`: converts the native config into `MatformerConfigs`.

## Choosing a holder

`make_holder(path, **kwargs)` splits a `scheme://path` string and asks the registry for the holder type:

```python
make_holder("checkpoints/run.ckpt")        # scheme absent -> matformer
make_holder("hf://meta-llama/Llama-3-8B")  # HuggingFace (local folder or hub id)
make_holder("modelscope://org/model")      # ModelScope
make_holder("gguf://path/to/model.gguf")   # GGUF
```

The `checkpoint_holder` registry category is where a new format is added. The base class is `CheckpointHolder`; the built-ins are `matformer`, `hf`, `modelscope` and `gguf`.

## The lazy weight view

`get_weights(definition=None, wanted=None, discard=None)` returns a `StableWeights` object. The tensors are materialised when the wrapper iterates it, so a large checkpoint is read one tensor at a time and unwanted tensors are not fetched.

`wanted` and `discard` are mutually exclusive. Each is a stable key, an fnmatch pattern, or a list of them:

```python
holder.get_weights(wanted=["encoder.*", "lm_head.*"])
holder.get_weights(discard="*.bias")
```

The selection applies to the plain renames and to the transform rules. A rule whose outputs are all discarded is skipped before the op runs, which is how loading only the LM head from a donor avoids materialising the backbone.

`holder.unresolved` is filled during the stream with the source keys that could not be placed. The wrapper adds it to the unexpected list, so a key that exists in the source but has no stable name is reported rather than silently dropped.

## The adapter definition

An adapter is a YAML file in `_DEFS/<model_type>.yaml`. Its schema is the same for the built-in adapters and for a user override:

```yaml
alias: mistral                 # another <model_type>.yaml to use instead
num_layers: 32                 # optional, overrides the holder's own
source_config: {...}           # optional, overrides the holder's own (for axis sizes)
weight_mapping:
  - "model.layers.N.self_attn.q_proj.weight": "encoder.blocks.layers.N.attn.q_proj.weight"
weight_transforms:
  - sources: [q_proj, k_proj, v_proj]
    outputs: [qkv_proj]
    op: stack
    pattern: "s ... -> ..."
    axes: {q: 32, k: 8, v: 8}
source_key_aliases: ["model.", "transformer."]
```

The `N` token is replaced by the layer index only when it is a whole dot-separated segment, so `LayerNorm` is not touched. `weight_transforms` uses `einx`: the `axes` values may be literals, dotted paths into `source_config`, or products of them. A rule that returns one tensor for several outputs is unbound along axis 0.

A definition that is only a pointer (`alias: mistral`) loads another adapter, with cycle detection. This is how a new architecture that is identical to a supported family needs no mapping at all.

## MatformerCheckpointHolder

This is the native holder. It accepts:

* a `.ckpt` file: weights, configs and states from a Matformer checkpoint;
* a raw weights file (`.pt` or `.safetensors`): weights only, no configs;
* a folder: all the YAML files as configs plus exactly one weight file.

`get_configs()` returns the checkpoint's `MatformerConfigs`, or reconstructs them from a legacy Lightning `hyper_parameters` block with a warning. `get_states()` returns the optimizer, scheduler, datamodule, step and epoch, unless `what_to_reset` excludes them. A raw weights file has no config and no states.

## HuggingFaceCheckpointHolder

The HF holder adds the two HF-specific things: reading `config.json` and downloading a hub repo. `model_type` selects the adapter; `num_hidden_layers`, `n_layer`, `num_layers` and `n_layers` are all accepted, at the top level or under `text_config`. The sharded index (`*.index.json`) is used when present; otherwise the single `.safetensors` or `.bin` is read directly. `get_configs(configs=None)` converts the HF config into Matformer configs and, if a configs object is passed, merges into it.

`ModelScopeCheckpointHolder` overrides only `_resolve_source`. `GGUFFCheckpointHolder` is the GGUF path.

## Additional weights

The `weights` section of a config mounts extra sources on top of the primary checkpoint. Each entry is a `WeightConfig`:

```yaml
weights:
  - name: donor
    type: safetensors
    path: donors/head.safetensors
    wanted: ["lm_head.*"]
```

`definition` overrides the adapter for that source. `wanted` and `discard` select what to inherit, as in `get_weights`. `load_additional_weights()` reports missing and unexpected keys for each source. The wrapper's use of additional weights is described in [[training/wrapper | MatformerWrapper]].

## Releasing

`release()` drops the references to the loaded tensors. `load_model` calls it after the weights have been copied into the model.
