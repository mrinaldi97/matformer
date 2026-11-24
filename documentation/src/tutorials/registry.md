---
title: Use the registry
sources:
  - matformer/matformer_registry.py
  - matformer/utils/matformer_registry_implementation.py
---
# Use the registry

The registry maps a category and a variant to one or more implementations. A model asks for a variant; the registry returns an instance of an implementation that is available and compatible with the current call.

## Inspect it

```python
from matformer.matformer_registry import registry

registry.list_categories()
registry.print_registry()
registry.print_registry(category="mlp")
registry.get_registry_info(category="mlp", as_json=True)
```

`get_registry_info` returns the dataclass and the implementations of every variant, with file path, line range, priority, required packages and constraints. The generated pages in [[components/registry | the registry reference]] use the same data.

## How an implementation is chosen

`registry.create(category, variant, **kwargs)` selects an implementation in this order:

1. `preferred=<name>`, when the caller passes it;
2. the environment variable `MATFORMER_<CATEGORY>_<VARIANT>`, for example `MATFORMER_MLP_SWIGLU=torch`;
3. the implementation with the highest `priority`.

A candidate is skipped when a package in `requires` cannot be imported, when its class cannot be imported, or when its `_test()` raises or returns `False`. The same test is not repeated in the process: the first result is stored and reused.

Constraints filter the candidates for the current call. `constraints=["is_training"]` skips an implementation that declares `{"is_training": False}`. `registry.set_default_constraints(["is_training"])` applies the same filter to every later call. An implementation without constraints is accepted everywhere.

If a candidate fails during construction, the registry tries the next candidate. If every candidate fails, `NoImplementationAvailable` lists the reason for each attempt. A successful selection returns the implementation instance; the registry does not replace it later.

## Add an implementation

Register a class in the category and variant it implements:

```python
from matformer.matformer_registry import registry

@registry.register("mlp", "swiglu", "my_kernel",
                   priority=100,
                   requires=["torch", "my_package"],
                   params_names={"w13.weight": "gate_proj.weight",
                                 "w2.weight": "down_proj.weight"})
class MySwiGLU(nn.Module):
    @staticmethod
    def _test():
        import torch
        module = MySwiGLU(16, 4.0)
        module(torch.zeros(1, 3, 16))
        return True
```

Place the file in `matformer/modules` or in `MATFORMER_ASSETS/extra_plugins`; both are scanned at import time. `_test()` runs at the first instantiation, not during the scan. `params_names` maps the internal parameter names to the stable names used by checkpoints, so the new kernel can load a checkpoint written by another implementation.

Test a selection directly:

```python
import torch
module = registry.create("mlp", "swiglu", hidden_size=768, ffn_factor=4.0)
module(torch.zeros(1, 3, 768))
```

`registry.create` accepts the constructor arguments of the implementation. The model code passes `config`, `hidden_size` and the other fields of the layer; a custom implementation must accept the arguments of the variant it replaces.

## Conditions

`requires` is for external libraries that must exist for the implementation to be considered. `constraints` is for the context: `is_training`, `is_inference`, a softcap the kernel does not support, and so on. `metadata` is free data attached to the instance as `_matformer_metadata`; it describes how to call the selected implementation and does not affect the choice.

## Dataclasses

A variant can carry a configuration dataclass:

```python
@registry.register_dataclass("mlp", "swiglu")
@dataclass
class SwiGLUConfig(MatformerConfig):
    hidden_size: int = 768
    ffn_factor: float = 4.0
```

`registry.get_definition("mlp", "swiglu")` returns the class. A section with `type: swiglu` in YAML is parsed with that class, and `instantiate()` calls `registry.create` for the same category and variant.
