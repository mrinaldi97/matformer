# Matformer Hook System

(this documentation file was llm generated, must be manually summarized and reworked, however, it is correct)

## Working principle

Each module declares which positions it exposes:
```python
class TransformerBlock(MatformerModule):
    available_hooks = ['pre_attn', 'post_attn', 'post_mlp', 'before_output']
```

Hooks are registered in `__init__` from the layer config and fired in `forward` via `_apply_hook`.  
A hook not declared on a module is **silently ignored** — no warnings, no side effects.  
A hook declared but not configured is a **no-op**.  
Multiple hooks at the same position are applied **sequentially**, output of each fed into the next.

---

## Adding a hook in config

```python
LayerConfig(
    hooks={
        # single hook, registry key only
        "pre_attn": "activation_stats",

        # single hook with extra constructor kwargs
        "before_output": {
            "type": "scaled_residual",
            "scale": 0.5
        },

        # multiple hooks at the same position, applied left to right
        "post_attn": [
            "activation_stats",
            {"type": "scaled_residual", "scale": 0.1}
        ]
    }
)
```

Or per-layer in `ModelConfig.custom_layers`:
```python
ModelConfig(
    custom_layers={
        3: LayerConfig(hooks={"post_attn": "activation_stats"})
    }
)
```

---

## Writing a hook

A hook is any callable with signature:
```python
def hook(x, *, position, config, layer_idx, **kwargs) -> x
```

### Stateless hook (plain function, not in parameter tree)
```python
def activation_stats(x, *, position, layer_idx, **kwargs):
    print(f"[L{layer_idx}:{position}] mean={x.tensor.mean():.4f}")
    return x  # always return x
```

### Stateful hook (nn.Module, registered in parameter tree)
```python
class ScaledResidual(nn.Module):
    def __init__(self, config, cache, layer_idx, scale=1.0, **kwargs):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, x, *, position, layer_idx, **kwargs):
        return replace(x, tensor=x.tensor * self.scale)
```

### Permanent model feature with stable checkpoint name

If a hook is a **permanent architectural feature** rather than a temporary probe,
inherit from `MatformerModule` and declare `hook_name`. This gives the hook a stable
key in `stable_state_dict()`, independent of its position in the hook list:

```python
class LearnedScaling(MatformerModule):
    hook_name = "learned_scaling"   # stable checkpoint key; without this,
                                    # the key would be hook_post_attn_0, etc.
    scale: "param_name:scale"       # internal param_name annotations work normally

    def __init__(self, config, cache, layer_idx, **kwargs):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(config.hidden_size))

    def forward(self, x, *, position, **kwargs):
        return replace(x, tensor=x.tensor * self.scale)
```

Register any hook type under the `"hooks"` category in the registry:
```python
registry.register("hooks", "scaled_residual", ScaledResidual)
registry.register("hooks", "activation_stats", activation_stats)
registry.register("hooks", "learned_scaling",  LearnedScaling)
```

---

## Available positions by module

| Module                         | Positions                                                                                                   |
|--------------------------------|-------------------------------------------------------------------------------------------------------------|
| `TransformerBlock`             | `pre_attn`, `post_norm_pre_attn`, `post_attn`, `pre_mlp`, `post_norm_pre_mlp`, `post_mlp`, `before_output` |
| `TransformerWithEmbeddingHead` | `pre_embed`, `post_embed`                                                                                   |
| `MultiHeadAttention` (MI mode) | `q`, `k`, `v`, `attn_scores`, `attn_pattern`, `z`, `attn_out`                                              |