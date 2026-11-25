---
title: Plugins
sources:
  - matformer/plugins/plugin_manager.py
  - matformer/plugins/plugin_classes.py
  - matformer/plugins/socket_mixin.py
  - matformer/transformer_blocks/models/classification_models.py
registry_docs: plugin.classification
category_docs: [plugin]
---
# Plugins

A plugin is a module mounted in a socket. It is part of the module tree and the state dict, and it can have trainable parameters.

Structural components (the MLP, the attention, a norm) are also mounted in sockets. A plugin can be inserted before them, after them, or in place of them.

## Interface

A plugin subclasses `Plugin`, which is a [[components/matformer-module | MatformerModule]]. Its constructor receives:

* `config`: the plugin dataclass, a `PluginConfig` subclass;
* `model_config`: the model configuration;
* `cache`: the shared [[components/cache | CachedStuff]].

Properties available after mounting:

| Property | Meaning |
|----------|---------|
| `address` | the socket address, e.g. `encoder.blocks.layers.3.post_add_mlp` |
| `visit` | the pass index in the current forward (greater than zero in looped models) |
| `layer_idx` | the layer index, or `None` outside a layer |
| `index` | the position in the socket |

`forward(x, **kwargs)` returns a new value for an active plugin or `None` for a passive plugin. A passive plugin observes the input and the socket continues with the same value. The socket applies components in order.

## Socket addresses

The addresses depend on the model class. `wrapper.inspect("*")` lists them.

For the classification model:

| Address suffix | Contents |
|----------------|----------|
| `lm_head` | language-model head |
| `encoder` | embedding and transformer |
| `embed_tokens` | token embedding |
| `embed_positions` | learnable positions, or empty |
| `blocks` | the NakedTransformer |
| `layer_<i>` | block `i` |
| `final_norm` | final norm |
| `layers.<i>.pre_attn` | norm before attention |
| `layers.<i>.self_attn` | attention or sequence mixer |
| `layers.<i>.post_attn` | norm after attention, inside the branch |
| `layers.<i>.post_add_attn` | norm after the residual add |
| `layers.<i>.pre_mlp`, `mlp`, `post_mlp`, `post_add_mlp` | the same for the MLP |

Block flow:

```
branch: pre_attn -> self_attn -> post_attn      added to the trunk
trunk:  post_add_attn
branch: pre_mlp -> mlp -> post_mlp              added to the trunk
trunk:  post_add_mlp
```

The `pre_*` and `post_*` sockets exist even when no norm is mounted. A probe on `post_add_mlp` reads the residual stream at the end of a layer.

## Target patterns

A target is an `fnmatch` pattern on the full address. `*` crosses dots.

```python
"*.final_norm"
"*.layers.*.post_add_mlp"
"*.layers.3.*"
"*.layers.[0-2].pre_attn"
["*.layers.0.post_attn", "*.layers.5.post_attn"]
"*"
```

`attach` raises `ValueError` when a pattern matches no socket.

## Mounting

There are three routes.

### Model defaults

A model class can declare its default plugins:

```python
class TransformerWithClassificationHead(MatformerModule):
    def default_plugins(self):
        return [("*.final_norm", classification_plugin_config(self.config, as_probe=False))]
```

`PluginManager.init_plugins()` mounts these pairs after the model is built.

### Configuration

A `plugin` section is a multi-entry section. `targets` is a pattern or a list of patterns; `type` selects the plugin; the other fields are the plugin config. `name` is the instance alias.

```yaml
plugin:
  - name: my_probe
    type: classification
    targets: "*.layers.*.post_add_mlp"
    scope: sequence
    pooling_type: mean
    num_classes: 2
    as_probe: true
```

A plugin without `targets` is not mounted automatically. It can be mounted at runtime.

### Runtime

```python
wrapper.add_end("*.layers.*.post_add_mlp", cfg)
wrapper.add_first("*.layers.3.self_attn", plugin_instance)
wrapper.attach("*.layers.3.pre_attn", cfg, at=1, name="x")
wrapper.replace("*.layers.0.pre_attn", new_component)
wrapper.replace("*.layers.0.pre_attn", other, ref="x")
wrapper.swap("*.layers.3.pre_attn", 0, 1)
wrapper.reorder("*.layers.3.pre_attn", [1, 0])
wrapper.detach("*.layers.*.post_add_mlp", "my_probe")
wrapper.inspect("*.layers.3.*")
wrapper.plugins()
```

`attach` accepts a config, a `Plugin` class, an instance or a callable. A config passed to a pattern creates one instance per matching socket. One instance passed to several sockets is shared, and `socket` refers to the last one.

## Writing a plugin

```python
from dataclasses import dataclass
from matformer.matformer_registry import registry
from matformer.plugins.plugin_classes import Plugin, PluginConfig

@registry.register_dataclass("plugin", "my_probe")
@dataclass
class MyProbeConfig(PluginConfig):
    scale: float = 1.0

@registry.register("plugin", "my_probe", "default")
class MyProbe(Plugin):
    name = "my_probe"

    def __init__(self, config, model_config, cache):
        super().__init__(config, model_config, cache)
        self.proj = registry.create("linear", "linear",
                                    in_features=model_config.hidden_size,
                                    out_features=1)

    def forward(self, x, **kwargs):
        self.cache.storage.setdefault("my_probe", []).append(self.proj(x.tensor))
        return None
```

An active plugin returns a new value:

```python
    def forward(self, x, **kwargs):
        from dataclasses import replace
        return replace(x, tensor=x.tensor * self.config.scale)
```

`x` is a [[components/tensors | TensorDC]]. `x.tensor` returns the underlying tensor; `x.apply(module)` runs a plain `nn.Module` and keeps the wrapper. Model dimensions should be read from `model_config`.

## Names and state dict

The mounted name is chosen from, in order: the `name=` argument, the config `name`, the class-level `name`, the class name, and the lowercased class name. A duplicate name gets `_1`, `_2`, and so on. The name is used by `detach`, `swap`, `replace` and freezing.

Plugin parameters are stored in the model state dict under `<owner>.<socket>_<name>.*`. A base checkpoint loaded into a model with plugins reports these keys as missing; filter `plug_*` or use the wrapper's missing/unexpected report.

## Save and restore

`plugin.save(path)` writes the plugin kind, address, stable state dict, index and config. `PluginManager.mount_from_files(path)` restores a file or every `*.pt` file in a folder. The trainer's `save_plugins` writes the mounted plugins of a checkpoint to the sibling `_plugins` folder.

## Freezing

`freeze` and `unfreeze` accept `plugin_type:classification`, a socket address, `component:mlp`, `layer:3` and stable parameter names. `trainable_param_report()` prints the backbone and plugin split.

## Examples

* `ClassificationPlugin`: pooling, optional extra transformer layers, a linear or MLP head, optional attention pooling and `as_probe`.
* `matformer/plugins/plugin_modules`: passive plugins that write to the cache or to an activation store.
* [[registry:plugin | Registered plugins]]
