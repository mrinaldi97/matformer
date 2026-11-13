from matformer.matformer_registry import registry
import importlib
import torch.nn as nn
import torch
from matformer.dataclasses.tensors_dataclasses import apply_kernel
from matformer.utils.kernel_tests import kernel_runs

def _load(module: str, attr: str):
    return getattr(importlib.import_module(module), attr)


@registry.register(
    "norm",
    "rmsnorm",
    "liger",
    requires=["liger_kernel"],
    priority=10,
    params_names={'inner.weight': 'weight'},
    constraints={'is_interpretable': False}
)
class LigerRMSNorm(nn.Module):
    inner: "transparent"
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        cls = _load("liger_kernel.transformers", "LigerRMSNorm")
        self.inner = cls(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    @staticmethod
    def _test():
        return kernel_runs(lambda: LigerRMSNorm(16), torch.randn(2, 3, 16))

    def forward(self, x):
        return apply_kernel(self.inner, x)


@registry.register(
    "norm",
    "layernorm",
    "liger",
    requires=["liger_kernel"],
    priority=10,
    params_names={'inner.weight': 'weight', 'inner.bias': 'bias'},
    constraints={'is_interpretable': False}
)
class LigerLayerNorm(nn.Module):
    inner: "transparent"
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, bias=True):
        super().__init__()
        cls = _load("liger_kernel.transformers", "LigerLayerNorm")
        self.inner = cls(normalized_shape, eps=eps, bias=bias)

    @staticmethod
    def _test():
        return kernel_runs(lambda: LigerLayerNorm(16), torch.randn(2, 3, 16))

    def forward(self, x):
        return apply_kernel(self.inner, x)


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
    },
    constraints={'is_interpretable': False}
)
class LigerSwiGLU(nn.Module):
    inner: "transparent"
    def __init__(self, hidden_size, ffn_factor, bias=False):
        super().__init__()
        assert bias is not True
        from liger_kernel.transformers import LigerSwiGLUMLP
        config = type("ConfigStub", (), {})()
        config.hidden_size = hidden_size
        config.intermediate_size = int(hidden_size * ffn_factor)
        config.hidden_act = "silu"
        self.inner = LigerSwiGLUMLP(config)

    @staticmethod
    def _test():
        return kernel_runs(lambda: LigerSwiGLU(16, 2), torch.randn(2, 3, 16))

    def forward(self, x):
        return apply_kernel(self.inner, x)


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
    },
    constraints={'is_interpretable': False}
)
class LigerGEGLU(nn.Module):
    inner: "transparent"
    def __init__(self, hidden_size, ffn_factor, bias=False):
        super().__init__()
        assert bias is not True
        from liger_kernel.transformers import LigerGEGLUMLP
        config = type("ConfigStub", (), {})()
        config.hidden_size = hidden_size
        config.intermediate_size = int(hidden_size * ffn_factor)
        config.hidden_act = "gelu"
        self.inner = LigerGEGLUMLP(config)

    @staticmethod
    def _test():
        return kernel_runs(lambda: LigerGEGLU(16, 2), torch.randn(2, 3, 16))

    def forward(self, x):
        return apply_kernel(self.inner, x)


@registry.register(
    "loss",
    "cross_entropy_loss",
    "liger",
    requires=["liger_kernel"],
    priority=0
)
class LigerCrossEntropyLoss(nn.Module):
    inner: "transparent"
    def __init__(self, *args, **kwargs):
        super().__init__()
        cls = _load("liger_kernel.transformers", "LigerCrossEntropyLoss")
        self.inner = cls(*args, **kwargs)

    @staticmethod
    def _test():
        LigerCrossEntropyLoss()(torch.randn(2, 4), torch.tensor([1, 2]))
        return True

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
    inner: "transparent"
    def __init__(self, *args, **kwargs):
        super().__init__()
        cls = _load("liger_kernel.transformers", "LigerFusedLinearCrossEntropyLoss")
        self.inner = cls(*args, **kwargs)

    @staticmethod
    def _test():
        LigerCrossEntropyLossFused()(torch.randn(2, 4), torch.tensor([1, 2]), lm_head_weight=torch.randn(4, 4))
        return True

    def forward(self, hidden, targets, **kwargs):
        return self.inner(
            _input=hidden.float(),
            target=targets,
            lin_weight=kwargs['lm_head_weight'].float(),
            bias=kwargs.get("lm_head_bias", None)
        )
