---
title: Write a plugin
sources:
  - matformer/plugins/plugin_classes.py
  - matformer/plugins/plugin_manager.py
  - matformer/transformer_blocks/models/classification_models.py
---
# Write a plugin

This example records the norm of the residual stream at every layer. It does not change the computation.

## Config

```python
from dataclasses import dataclass
from matformer.matformer_registry import registry
from matformer.plugins.plugin_classes import PluginConfig

@registry.register_dataclass("plugin", "norm_probe")
@dataclass
class NormProbeConfig(PluginConfig):
    key: str = "norms"
```

`PluginConfig` provides `targets` and `name`. `targets` is the pattern used when the plugin is mounted from the config; `name` is the instance alias.

## Plugin

```python
import torch
from matformer.plugins.plugin_classes import Plugin

@registry.register("plugin", "norm_probe", "default")
class NormProbe(Plugin):
    name = "norm_probe"

    def forward(self, x, **kwargs):
        value = x.tensor.float().norm(dim=-1).mean()
        self.cache.storage.setdefault(self.config.key, []).append(value)
        return None
```

`forward` receives a [[components/tensors | TensorDC]]; `x.tensor` is the underlying tensor. Returning `None` keeps the input unchanged. Returning a new value makes the plugin active.

## Mount

From the config:

```yaml
plugin:
  - name: norms
    type: norm_probe
    targets: "*.layers.*.post_add_mlp"
    key: residual_norms
```

The plugin manager matches the pattern against every socket address and mounts one instance per match. At runtime:

```python
wrapper.add_end("*.layers.*.post_add_mlp", NormProbeConfig(key="residual_norms"))
```

The pattern uses `fnmatch`; `*` crosses dots. `attach` raises when the pattern matches no socket. `wrapper.inspect("*.layers.*")` lists the current sockets.

## Read the data

`cache.storage["residual_norms"]` contains one entry per socket and visit. A task can read it in `compute()` and return it in the summary. `visit` distinguishes the passes of a looped model.

## Parameters

A plugin that creates modules in `__init__` is an `nn.Module`. Its parameters are stored in the model state dict under `<socket>_<name>.*` and are trained with the model unless frozen. `freeze: ["plugin_type:norm_probe"]` freezes them.

## Further examples

* [[components/plugins | Plugins]] documents the runtime operations, naming rules and state dict.
* `ClassificationPlugin` is an active plugin with pooling, dropout, extra layers and a head.
* The activation-cache plugins in `matformer/plugins/plugin_modules` are passive plugins that write to an activation store.
