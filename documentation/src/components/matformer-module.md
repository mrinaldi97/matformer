---
title: MatformerModule
sources:
  - matformer/matformer_module.py
  - matformer/plugins/socket_mixin.py
---
# MatformerModule

<a id="class-annotations"></a>
## Class annotations

If you look at the predefined neural-net implementations in Matformer, for example [[components/transformer#transformerblock | TransformerBlock]], you will immediately notice that at the top of the class some non-standard annotations are present.
For example:

```
    attn_norm: "param_name:attn_norm"
    mlp_norm: "param_name:mlp_norm"
    self_attn: "param_name:attn"
    mlp: "param_name:mlp"
```

Those annotations are automatically interpreted by the base [[components/matformer-module | MatformerModule]] class. For this reason, it is useful for consistency that if you write new modules for Matfomer, either permanent or temporary, such as a [[components/plugins | plugin]], you inherit MatformerModule.
MatformerModule inherits from two important base classes:
* [[components/matformer-module#parameterrenamer | ParameterRenamer]]
ParameterRenamer determines the actual names of the weights as they will be included in the checkpoint. PyTorch, by default, saves parameter inside the checkpoint by expanding their entire path inside the model. This would lead to very aestethically unpleasant and potentially confusing weights' keys in a complex architecture like matformer such as, for example, "model.model.encoder.layer.4.mlp.inner.weight". To avoid this, Matformer leaves at disposal an handy system for renaming parameters' name and make them "stable", that is, potentially independent from the actual modules' names used inside the library.
It seems hard, but is actually very easy to use:
1. Any module attribute annotated as `"param_name:stable_name"` is entered in the stable-name table under `stable_name` instead of its Python attribute name. The annotation applies to modules, parameters and buffers, and the scan walks the whole tree, so an annotation on a parent class is inherited by every subclass.
2. "transparent" means that the module itself contributes no segment to the path. Its children are promoted one level, so wrapping a component in a transparent container does not change any checkpoint key. Write `child: "transparent"` on the container attribute.

A third source of names is the implementation itself: a registry implementation can declare `params_names`, a mapping from its internal parameter names to stable names. That is how a custom kernel can load the same checkpoint as the torch implementation.

## ParameterRenamer

`ParametersRenamer` is the stable-name layer. It builds `_parameters_mappings`, a dictionary from stable name to the actual dotted path in the module tree, by scanning the tree once and consulting, in order:

1. the `_params_names` dictionary attached by the registry implementation;
2. the class annotations of the whole MRO;
3. the instance annotations (this is how a plugin mounted at runtime gets a stable name);
4. the actual attribute name, as a fallback.

The public methods are:

| Method | What it does |
|--------|--------------|
| `stable_state_dict()` | a state dict keyed by stable names, ready to be saved |
| `load_stable_state_dict(dict, include=None, exclude=None)` | loads a stable state dict, returning `(missing, unexpected)`; `include`/`exclude` are fnmatch patterns on stable keys |
| `_get_weight_pointer(stable_name)` | the live parameter for a stable name |
| `_set_weight(stable_name, tensor)` | copies a tensor into the parameter, returning `False` on a shape mismatch |

The wrapper uses these methods. A Matformer checkpoint is therefore addressed by stable names, and a HuggingFace checkpoint can be converted without changing Python attribute names.

## ParameterEditor

`ParameterEditor` is the weight-space intervention layer. It is a mixin, so every `MatformerModule` has it; it is used on plugins and on the backbone for interpretability experiments.

Every operation takes a stable parameter name. A mask uses `True` for an entry that stays trainable and `False` for an entry that is frozen or overwritten.

| Method | What it does |
|--------|--------------|
| `snapshot(name)` | a detached copy of the current parameter |
| `freeze(name, train=None)` | zeroes the gradient everywhere except where `train` is True; values are untouched |
| `edit(name, value=0, mask=None, freeze_grad=True)` | overwrites the parameter, optionally only where `mask` is False |
| `project_out(name, direction, side="in")` | removes a direction from the input or output space of a 2D parameter, and projects the gradient the same way |
| `directions(name, k=1, side="in")` | the top-k singular vectors, input or output space |
| `leak(name, direction, side="in")` | the fraction of the squared norm that still lies in a direction |
| `delta(name)` | the relative change with respect to the backup |
| `l1(name, lam)` | adds `lam * sign(w)` to the gradient, the L1 half of an elastic net |
| `stats(name)` | mean, std, absolute mean, and for matrices the top singular value and effective rank |
| `restore(name=None)` | undoes edits and removes the gradient callbacks |

`restore()` with no argument restores every edited parameter. The backup is created the first time a parameter is touched.

## Sockets

`MatformerModule` also inherits `SocketMixin`, which gives every module `init_sockets`, `add_socket`, `register_socket`, `inspect_sockets` and `add_nested_socket`. The details are in [[components/sockets-and-stream | Sockets and the stream]].
