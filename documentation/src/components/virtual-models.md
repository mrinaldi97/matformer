---
title: Virtual models and activation stores
sources:
  - matformer/matformer_wrapper.py
  - matformer/plugins/plugin_modules/save_activations.py
  - matformer/utils/matformer_cache.py
---
# Virtual models and activation stores

A virtual model reads stored activations instead of executing the backbone. Sockets, plugins and the wrapper forward still run. The transformer is replaced by a lookup in an activation store.

This is useful in three situations:

* **Interpretability.** You want to mount a probe on every layer of a model, but running the full model for every probe configuration is wasteful. Store the activations once, then replay them.
* **Distillation or linear probing.** The backbone is frozen and only the head is trained.
* **Debugging.** You want to know what a plugin sees without depending on the model's numerical behaviour.

## Runtime modes

`MatformerModelWrapper` is constructed with a `runtime_mode`, either `"live"` or `"virtual"`, and keeps one model per mode in `self._models`. `set_runtime_mode(mode)` switches the active one:

1. every mounted plugin is unmounted from the current model;
2. the other model is built (a live model through the registry, a `VirtualModel` from the activation store);
3. sockets are re-indexed;
4. every plugin is mounted again on the socket with the same address. If the new model has no such socket, the plugin is dropped with a warning.

The plugin objects are moved to the new model, so their parameters and state survive the switch.

## The activation store

An activation store is a registered component in the `activation_store` category. It implements two things:

* `available_addresses()`: the socket addresses for which activations were stored;
* `fetch_all(indices, addresses)`: the tensors for a batch of item indices.

It also exposes `switch_databank(name, rewind=True)`, which selects which stored dataset is being replayed and rewinds its position. The trainer calls it at the end of an epoch for a virtual model, and the task base class calls it at the start of every task run, so a task that is re-run on a cached instance still reads the right databank.

The `VirtualModel` builds one nested socket per available address and, in `forward`, fetches the needed activations, applies the mounted plugins in address order and returns either `cache.storage['probe_outputs']` or the configured `default_return`. If `check_input` is on (the default), it compares the stored input activation with the ids it received and warns when they diverge.

## Configuration

```yaml
activation_store:
  type: duckdb          # or lmdb
virtual_model:
  check_input: true
  input_address: encoder.model_input
  return_multiprobe: true
training:
  virtual_only: true    # use only the virtual model, no backbone weights are loaded
```

With `training.virtual_only`, `load_model` instantiates the store before the wrapper, asks it for the model config, and never loads the backbone weights. The wrapper is built in virtual mode from the start.

Without `virtual_only`, the wrapper is built live, and `set_runtime_mode("virtual")` can switch at any time. This is the usual workflow: run a `forward` task with plugins attached to capture activations, then switch to virtual and replay them.

## Integration

The virtual model is a [[components/matformer-module | MatformerModule]] with sockets, so the [[components/plugins | plugin manager]] mounts plugins in the same way as on a live model. Its `default_plugins()` asks the live model class which plugins it would mount, so a classification head or a set of probes is created automatically.

The activation stores are registered under [[registry:activation_store | activation_store]].

An activation-analysis task runs on a virtual model in the same way: it consumes the replayed
activations directly, so `activation_analysis` can profile or describe a cached databank without
loading the backbone. The `default_return` must name an address the store actually contains; the
`task.info` RPC prints the loaded runtime and `model.internals` lists the mounted plugins.
