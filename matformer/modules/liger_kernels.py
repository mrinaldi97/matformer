from matformer.matformer_registry import registry
import importlib
import torch.nn as nn
import torch

def _load(module: str, attr: str):
    return getattr(importlib.import_module(module), attr)


@registry.register(
    "norm",
    "rmsnorm",
    "liger",
    requires=["liger_kernel"],
    priority=10,
    params_names={'inner.weight': 'weight'}
)
class LigerRMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        cls = _load("liger_kernel.transformers", "LigerRMSNorm")
        self.inner = cls(normalized_shape, eps=eps)
    #def is_available(self):
    #    return torch.cuda.is_available() 
    def forward(self, x):
        return self.inner(x)


@registry.register(
    "norm",
    "layernorm",
    "liger",
    requires=["liger_kernel"],
    priority=10,
    params_names={'inner.weight': 'weight', 'inner.bias': 'bias'}
)
class LigerLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        cls = _load("liger_kernel.transformers", "LigerLayerNorm")
        self.inner = cls(normalized_shape, eps=eps)
    #def is_available(self):
    #    return torch.cuda.is_available() 
    def forward(self, x):
        return self.inner(x)


@registry.register(
    "mlp",
    "swiglu",
    "liger",
    requires=["liger_kernel"],
    priority=10,
    params_names={
        'inner.gate_proj.weight': 'gate_proj.weight',
        'inner.up_proj.weight':   'up_proj.weight',
        'inner.down_proj.weight': 'down_proj.weight'
    }
)
class LigerSwiGLU(nn.Module):
    def __init__(self, hidden_size, ffn_factor):
        super().__init__()
        from liger_kernel.transformers import LigerSwiGLUMLP
        config = type("ConfigStub", (), {})()
        config.hidden_size = hidden_size
        config.intermediate_size = int(hidden_size * ffn_factor)
        config.hidden_act = "silu"
        self.inner = LigerSwiGLUMLP(config)
    #def is_available(self):
    #    return torch.cuda.is_available() 
    def forward(self, x):
        return self.inner(x)


@registry.register(
    "mlp",
    "geglu",
    "liger",
    requires=["liger_kernel"],
    priority=10,
    params_names={
        'inner.gate_proj.weight': 'gate_proj.weight',
        'inner.up_proj.weight':   'up_proj.weight',
        'inner.down_proj.weight': 'down_proj.weight'
    }
)
class LigerGEGLU(nn.Module):
    def __init__(self, hidden_size, ffn_factor):
        super().__init__()
        from liger_kernel.transformers import LigerGEGLUMLP
        config = type("ConfigStub", (), {})()
        config.hidden_size = hidden_size
        config.intermediate_size = int(hidden_size * ffn_factor)
        config.hidden_act = "gelu"
        self.inner = LigerGEGLUMLP(config)
    #def is_available(self):
    #    return torch.cuda.is_available() 
    def forward(self, x):
        return self.inner(x)


@registry.register(
    "loss",
    "cross_entropy_loss",
    "liger",
    requires=["liger_kernel"],
    priority=0
)
class LigerCrossEntropyLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        cls = _load("liger_kernel.transformers", "LigerCrossEntropyLoss")
        self.inner = cls(*args, **kwargs)
    #def is_available(self):
    #    return torch.cuda.is_available() 
    def forward(self, logits, targets, **kwargs):
        return self.inner(logits, targets, **kwargs)


@registry.register(
    "loss",
    "cross_entropy_loss_fused",
    "liger",
    requires=["liger_kernel"],
    priority=0
)
class LigerCrossEntropyLossFused(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        cls = _load("liger_kernel.transformers", "LigerFusedLinearCrossEntropyLoss")
        self.inner = cls(*args, **kwargs)
    #def is_available(self):
    #    return torch.cuda.is_available() 
    def forward(self, hidden, targets, **kwargs):
        return self.inner(
            _input=hidden,
            target=targets,
            lin_weight=kwargs['lm_head_weight'],
            bias=kwargs.get("lm_head_bias", None)
        )
