from matformer.matformer_registry import registry
import importlib
import torch.nn as nn
import torch

def _load(module: str, attr: str):
    return getattr(importlib.import_module(module), attr)
"""
layer_norm_weight
layer_norm_bias
fc1_weight
fc1_bias
fc2_weight
fc2_bias

Da implementare il Fused Layer norm mlp
"""
@registry.register(
    "norm",
    "rmsnorm",
    "transformer-engine",
    requires=["transformer_engine"],
    priority=20,
    params_names={'inner.weight': 'weight'}
)
class TE_RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, **kwargs):
        super().__init__()
        cls = _load("transformer_engine.pytorch", "RMSNorm")
        size = normalized_shape
        if isinstance(normalized_shape, (list, tuple)):
            # take last dim as inner-most
            size = int(normalized_shape[-1])
        self.inner = cls(size, eps=eps, **kwargs)

    def forward(self, x):
        return self.inner(x)


@registry.register(
    "norm",
    "layernorm",
    "transformer-engine",
    requires=["transformer_engine"],
    priority=20,
    params_names={'inner.weight': 'weight', 'inner.bias': 'bias'}
)
class TE_LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, **kwargs):
        super().__init__()
        cls = _load("transformer_engine.pytorch", "LayerNorm")
        size = normalized_shape
        if isinstance(normalized_shape, (list, tuple)):
            size = int(normalized_shape[-1])
        self.inner = cls(size, eps=eps, **kwargs)

    def forward(self, x):
        return self.inner(x)


@registry.register(
    "linear",
    "linear",
    "transformer-engine",
    requires=["transformer_engine"],
    priority=20,
    params_names={'inner.weight': 'weight', 'inner.bias': 'bias'}
)
class TE_Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super().__init__()
        cls = _load("transformer_engine.pytorch", "Linear")
        self.inner = cls(in_features, out_features, bias=bias, **kwargs)

    def forward(self, inp, *args, **kwargs):
        return self.inner(inp, *args, **kwargs)
