---
title: The Matformer Registry
sources:
  - matformer/utils/matformer_registry_implementation.py
  - matformer/matformer_registry.py
---
# The Matformer Registry

The Matformer architecture tries to avoid every direct instantiation of hard-coded classes of neural network components.
Instead, every implementation of the core components is managed by the MatformerRegistry. This means that it is possible
to define modules for every component of the model, such as MLP, attention implementations, layer normalizations...
In this way the model is not bound to specific implementation (such as vanilla pyTorch) but it is immediately possible
to adopt different kernels without touching a single line of the core transformer implementation
This gives extreme flexibility to the developer: just by putting python files in the repo's folders (or specific additional folders) and decorating 
the class with a decorator it is possible to substitute every module of the network as well as add new components to the framework, such as new [[MatformerTask | tasks]].
By using the registry it is also possible to import customized experimental modules that can be mounted on many different entry points of
the model, and, if necessary, they can be trained together with the rest of the network!
Adding custom components to a transformer model has never been so easy.

The registry data also feeds the documentation. Every category, every variant and every implementation has a page, and the page is generated from the same data used by `registry.create(...)`. Start from the [[components/registry | registry reference]]; each variant page links to its configuration dataclass and to the Python class.

The registry has the following hierarchy:

```
_registry (dict)
└── category (dict)        example: "attention", "norm", "linear", "embedding", "mlp", "loss"
    └── variant (dict)     example: "geglu", "swiglu", "rmsnorm", "flash", "gated_deltanet", "mamba"
        └── name (dict)    example: "torch", "liger", "nvidia-transformer-engine"
            ├── requires      list of required libraries, used for fallback
            ├── priority      order of the variant, e.g. torch 0, liger 1, nvidia 2
            ├── constraints   contexts in which the implementation can be used, e.g. ["is_training"]
            ├── metadata      free dictionary
            └── params_names  stable parameter names, used to load checkpoints across implementations
```

## How to explore the registry

From the console:

```python
from matformer.matformer_registry import registry

registry.list_categories()                 # everything that can appear in a YAML file
registry.print_registry()                  # the whole tree, with files and priorities
registry.print_registry(category="attention")
registry.get_registry_info(category="model", as_json=True)
```

The generated pages are built from `registry.get_registry_info()`, so they show exactly what this process sees: the variants, their dataclasses, the implementations, the priority, the required libraries, the constraints and the file each class lives in.

{{ registry_index }}

## Add a custom implementation

Put your script in repo's folders, or, to keep it out of the repo, in
`MATFORMER_ASSETS/extra_plugins` (see [[env_vars:MATFORMER_ASSETS | the environment variables]]): the registry
scans both.

Then, you don't have to do anything by hand, as the registration is handled by an easy to use decorator, example:

```python
from matformer.matformer_registry import registry

@registry.register("mlp", "swiglu", "a_name_for_your_swiglu_kernel",
       params_names={'w13.weight': 'gate_proj.weight',
                     'w2.weight': 'down_proj.weight'},
       priority=100, requires=["torch", "my_custom_library"])
class MySwigluImplementation(nn.Module):
    # (your code here)

    @staticmethod
    def _test():
        # Return True (or simply don't raise) if this kernel really works on this machine
        ...
```

It's very simple to define a module and let Matformer handle the picking of correct implementation for you:

```python
registry.create("mlp", "swiglu", hidden_size=768, ffn_factor=4)
```

These are the only requirements to seamless integrate different custom implementation of basic modules into a Matformer model! Just be careful to
match conventions for param_names if you want that your model correctly loads also with other implementation.

## Adding modules to the model

Modules that are not one of the registered components are mounted as [[components/plugins | plugins]]. A plugin is registered in the `plugin` category and addressed by a socket pattern. The main transformer block exposes these sockets:

* `pre_attn`
* `self_attn`
* `post_attn`
* `post_add_attn`
* `pre_mlp`
* `mlp`
* `post_mlp`
* `post_add_mlp`

A plugin is declared in the config with a target pattern and a type:

```yaml
plugin:
  - name: my_probe
    type: my_probe
    targets: "*.layers.*.pre_mlp"
```

The plugin class is registered with the usual decorator:

```python
from matformer.matformer_registry import registry
from matformer.plugins.plugin_classes import Plugin

@registry.register("plugin", "my_probe", "default")
class MyProbe(Plugin):
    def forward(self, x, **kwargs):
        # read x, optionally return a new value
        return None
```

A plugin with parameters is added to the model's state dict and is trained with the model. The complete interface is in [[components/plugins | Plugins]].

## Preferences, availability tests and constraints

Implementation choice can be steered without touching code: the environment variable
MATFORMER_<CATEGORY>_<VARIANT> (ex. MATFORMER_ATTENTION_NORMAL=sdpa) forces a specific
implementation, and create(..., preferred='liger') does the same programmatically. Otherwise the
priority queue decides. Once an implementation is chosen it is instantiated; if it cannot be
built, the registry simply tries the next one.

Availability tests: an implementation may define a class-level `_test()` static method, a few
lines that really exercise the kernel (import it and run the smallest possible call). It is run
by the registry at instantiation time - never during the AST discovery, which stays completely
lazy - and its outcome is stored globally, so a whole model pays the cost only once and a
failing kernel (ex. flash-attn installed but useless on a CPU-only machine, or compiled for
another CUDA version) is skipped forever after without repeating the test. Any raised exception
or a False return means "don't use me here".

Constraints describe what an implementation can be used for. A hypothetical FlashInfer kernel usable for inference but not for training is declared as:

```python
@registry.register("attention", "normal", "flashinfer", priority=20, constraints=["is_inference"])
```

`registry.create("attention", "normal", constraints=["is_training"], ...)` skips it even if it has the highest priority. `registry.set_default_constraints(["is_training"])` applies the constraint to every subsequent `create()` call. An implementation without constraints is accepted everywhere. A kernel can also declare the constraints it does not support, e.g. `constraints={"is_training": False}`.

## requires, constraints and metadata

They are easy to mix up, so the rule of thumb is:

* `requires`: external libraries the implementation needs to exist at all (e.g. `["flash_attn"]`). If a library cannot be imported, the implementation is not a candidate.
* `constraints`: whether the implementation can be selected in the current context (e.g. `is_training`, `softcap`). If a constraint is not satisfied, the implementation is skipped and the next candidate is tried.
* `metadata`: how to call an implementation that has already been selected (tensor order, packed-qkv support, and so on). It is attached to the instance as `_matformer_metadata` and does not affect the choice.
A capability whose absence forces a fallback (e.g. flash-attn not doing tanh softcapping) is a constraint. If it is declared as metadata, the caller discovers at forward time that the chosen kernel cannot do the job.

## Errors

Every error raised by the registry is a RegistryError subclass: ImplementationNotAvailable
(missing dependency or failed _test), ImplementationInitError (the constructor of the chosen
implementation raised; the original traceback is preserved, so it points at the user's module and
not at the registry) and NoImplementationAvailable (every candidate failed: the message lists the
reason of each attempt). A failing implementation never silently ends the search and never
returns an object that isn't the requested module.

There is exactly one registry per process. matformer/matformer_registry.py builds it once at import time; every
`from matformer.matformer_registry import registry` elsewhere in the codebase (or in your own modules) gets that
same instance, already populated. Never call MatformerRegistry() directly.

## Linking config dataclasses

A dataclass can be bound to a registered implementation with `register_dataclass`:

```python
@registry.register("logger", "sqlite", "default")
class SQLiteLogger:
    ...

@registry.register_dataclass("logger", "sqlite", default=True)
@dataclass
class SQLiteLoggerConfig(LoggerConfig):
    log_db: str = "matformer.db"
```

Every YAML section is keyed by category and picks its variant with a `type` field (`logger: {type: sqlite, ...}`). A section without `type` uses the variant registered with `default=True`. `load_config.instantiate()` resolves the config through the registry.

## Documentation helpers

`get_registry_info()` returns a nested dictionary with the default variant of every category and, for every variant, its dataclass and implementations. Each entry contains the file path and the line range of the class. The generated pages use this data for the source links.

`_get_filepath(pkgname)` resolves a package name to its file. `test_status(impl)` reports the result of an availability test. `list_categories(defined_only=False)` includes categories that have implementations but no dataclass.

A variant can have a hand-written page. A Markdown page with `registry_docs: [category.variant]` in its front matter is linked from the variant's registry page. The Python source page remains available.

## Attaching pages to variants

The documentation build reads the registry, so a page stays attached to a variant when the class moves to another module. The page metadata is hand-written; the class, its fields, its variants and its implementations are read from the registry and from the dataclasses.
