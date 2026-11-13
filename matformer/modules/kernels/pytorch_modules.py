from matformer.matformer_registry import registry
from matformer.matformer_module import MatformerModule
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import RMSNorm, LayerNorm
from matformer.dataclasses.tensors_dataclasses import apply_kernel
from matformer.utils.kernel_tests import kernel_runs


@registry.register("norm", "identity", "default", constraints={'is_interpretable': False})
class Identity:
    def __call__(self, x, **kw): return x

@registry.register(
    "linear",
    "linear",
    "torch",
    requires=["torch"],
    priority=0,
    params_names={'inner.weight': 'weight', 'inner.bias': 'bias'},
    constraints={'is_interpretable': False}
)
class TorchLinear(nn.Module):
    inner: "transparent"
    def __init__(self, in_features, out_features, bias=True, *args, **kwargs):
        super().__init__()
        self.inner = nn.Linear(in_features, out_features, bias=bias)

    @staticmethod
    def _test():
        return kernel_runs(lambda: TorchLinear(16, 8), torch.randn(2, 3, 16))

    def forward(self, x, *args, **kwargs):
        return apply_kernel(self.inner, x, *args, **kwargs)


@registry.register(
    "embedding",
    "embedding",
    "torch",
    requires=["torch"],
    priority=0,
    params_names={'inner.weight': 'weight', 'inner.bias': 'bias'},
    constraints={'is_interpretable': False}
)
class TorchEmbedding(nn.Module):
    inner: "transparent"
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, *args, **kwargs):
        super().__init__()
        self.inner = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            **kwargs
        )

    @staticmethod
    def _test():
        return kernel_runs(lambda: TorchEmbedding(8, 16), torch.randint(0, 8, (2, 3)))

    def forward(self, x, *args, **kwargs):
        return apply_kernel(self.inner, x, *args, **kwargs)


@registry.register(
    "loss",
    "cross_entropy_loss",
    "torch",
    requires=["torch"],
    priority=10
)
class TorchCrossEntropyLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._args = args
        self._kwargs = kwargs

    @staticmethod
    def _test():
        TorchCrossEntropyLoss()(torch.randn(2, 4), torch.tensor([1, 2]))
        return True

    def forward(self, logits, targets, **extra_kwargs):
        kw = dict(self._kwargs)
        kw.update(extra_kwargs)
        return F.cross_entropy(logits, targets, *self._args, **kw)


@registry.register(
    "mlp",
    "swiglu",
    "torch",
    requires=["torch"],
    priority=0
)
class TorchSwiGLU(MatformerModule):
    inner: "transparent"
    def __init__(self, config=None, hidden_size=None, ffn_factor=None, bias=False):
        super().__init__()
        assert bias is not True
        if config is not None:
            hidden_size = config.hidden_size
            ffn_factor = config.ffn_factor
        ffn_internal_dim = int(hidden_size * ffn_factor)
        self.gate_proj = registry.create('linear', 'linear', in_features=hidden_size, out_features=ffn_internal_dim, bias=bias)
        self.up_proj   = registry.create('linear', 'linear', in_features=hidden_size, out_features=ffn_internal_dim, bias=bias)
        self.down_proj = registry.create('linear', 'linear', in_features=ffn_internal_dim, out_features=hidden_size, bias=bias)
        self.init_sockets()
        self.add_socket("hidden_act")

    @staticmethod
    def _test():
        return kernel_runs(lambda: TorchSwiGLU(hidden_size=16, ffn_factor=2), torch.randn(2, 3, 16))

    def forward(self, x):
        self.stream.init(apply_kernel(lambda t: F.silu(self.gate_proj(t)) * self.up_proj(t), x))
        self.stream.apply("hidden_act")
        return apply_kernel(self.down_proj, self.stream.exit())


@registry.register(
    "mlp",
    "geglu",
    "torch",
    requires=["torch"],
    priority=0
)
class TorchGEGLU(MatformerModule):
    inner: "transparent"
    def __init__(self, config=None, hidden_size=None, ffn_factor=None, bias=False):
        super().__init__()
        assert bias is not True
        if config is not None:
            hidden_size = config.hidden_size
            ffn_factor = config.ffn_factor
        ffn_internal_dim = int(hidden_size * ffn_factor)

        self.gate_proj = registry.create('linear', 'linear', in_features=hidden_size, out_features=ffn_internal_dim, bias=bias)
        self.up_proj   = registry.create('linear', 'linear', in_features=hidden_size, out_features=ffn_internal_dim, bias=bias)
        self.down_proj = registry.create('linear', 'linear', in_features=ffn_internal_dim, out_features=hidden_size, bias=bias)
        self.init_sockets()
        self.add_socket("hidden_act")

    @staticmethod
    def _test():
        return kernel_runs(lambda: TorchGEGLU(hidden_size=16, ffn_factor=2), torch.randn(2, 3, 16))

    def forward(self, x):
        self.stream.init(apply_kernel(lambda t: F.gelu(self.gate_proj(t)) * self.up_proj(t), x))
        self.stream.apply("hidden_act")
        return apply_kernel(self.down_proj, self.stream.exit())

@registry.register(
    "mlp",
    "gelu",
    "torch",
    requires=["torch"],
    priority=0
)
class TorchGELU(MatformerModule):
    inner: "transparent"
    def __init__(self, config=None, hidden_size=None, ffn_factor=None, bias=False):
        super().__init__()
        if config is not None:
            hidden_size = config.hidden_size
            ffn_factor = config.ffn_factor
        ffn_internal_dim = int(hidden_size * ffn_factor)
        self.up_proj = registry.create('linear', 'linear', in_features=hidden_size, out_features=ffn_internal_dim, bias=bias)
        self.down_proj = registry.create('linear', 'linear', in_features=ffn_internal_dim, out_features=hidden_size, bias=bias)
        self.init_sockets()
        self.add_socket("hidden_act")

    @staticmethod
    def _test():
        return kernel_runs(lambda: TorchGELU(hidden_size=16, ffn_factor=2), torch.randn(2, 3, 16))

    def forward(self, x):
        self.stream.init(apply_kernel(lambda t: F.gelu(self.up_proj(t)), x))
        self.stream.apply("hidden_act")
        return apply_kernel(self.down_proj, self.stream.exit())

@registry.register(
    "mlp",
    "gelu_tanh",
    "torch",
    requires=["torch"],
    priority=0
)
class TorchGELUTanh(TorchGELU):
    """2-matrix MLP with the tanh-approximated GELU (HF gelu_pytorch_tanh, e.g. GPT-2, Gemma)."""
    @staticmethod
    def _test():
        return kernel_runs(lambda: TorchGELUTanh(hidden_size=16, ffn_factor=2), torch.randn(2, 3, 16))

    def forward(self, x):
        self.stream.init(apply_kernel(lambda t: F.gelu(self.up_proj(t), approximate="tanh"), x))
        self.stream.apply("hidden_act")
        return apply_kernel(self.down_proj, self.stream.exit())


@registry.register(
    "mlp",
    "geglu_tanh",
    "torch",
    requires=["torch"],
    priority=0
)
class TorchGEGLUTanh(TorchGEGLU):
    """Gated 3-matrix MLP with tanh-GELU — HF Gemma's hidden_activation gelu_pytorch_tanh."""
    @staticmethod
    def _test():
        return kernel_runs(lambda: TorchGEGLUTanh(hidden_size=16, ffn_factor=2), torch.randn(2, 3, 16))

    def forward(self, x):
        self.stream.init(apply_kernel(lambda t: F.gelu(self.gate_proj(t), approximate="tanh") * self.up_proj(t), x))
        self.stream.apply("hidden_act")
        return apply_kernel(self.down_proj, self.stream.exit())


@registry.register(
    "norm",
    "rmsnorm",
    "torch",
    requires=["torch"],
    priority=0,
    params_names={'inner.weight': 'weight'},
    constraints={'is_interpretable': False}
)
class TorchRMSNorm(nn.Module):
    inner: "transparent"
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.inner = RMSNorm(
            normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine
        )

    @staticmethod
    def _test():
        return kernel_runs(lambda: TorchRMSNorm(16), torch.randn(2, 3, 16))

    def forward(self, x):
        return apply_kernel(self.inner, x)

@registry.register(
    "norm",
    "rmsnorm_zc",
    "torch",
    requires=["torch"],
    priority=0,
    params_names={'weight': 'weight'},
    constraints={'is_interpretable': False}
)
class TorchRMSNormZC(nn.Module):
    """RMSNorm with an optional zero-centered scale (1+weight, Gemma/Qwen3-style, weight
    initialized at 0) and/or a per-group weight (e.g. one scale per attention head) instead
    of a single shared one."""
    def __init__(self, normalized_shape, groups=None, zero_centered=True, eps=1e-6, elementwise_affine=False):
        super().__init__()
        shape = (groups, normalized_shape) if groups else (normalized_shape,)
        self.weight = nn.Parameter(torch.zeros(shape) if zero_centered else torch.ones(shape))
        self.zero_centered, self.eps = zero_centered, eps

    @staticmethod
    def _test():
        return kernel_runs(lambda: TorchRMSNormZC(16), torch.randn(2, 3, 16))

    def forward(self, x):
        return apply_kernel(lambda t: t * torch.rsqrt(t.float().pow(2).mean(-1, keepdim=True) + self.eps).to(t.dtype)
                            * (1 + self.weight if self.zero_centered else self.weight), x)



@registry.register(
    "norm",
    "layernorm",
    "torch",
    requires=["torch"],
    priority=0,
    params_names={'inner.weight': 'weight', 'inner.bias': 'bias'},
    constraints={'is_interpretable': False}
)
class TorchLayerNorm(nn.Module):
    inner: "transparent"
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, bias=True):
        super().__init__()
        self.inner = LayerNorm(
            normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias
        )

    @staticmethod
    def _test():
        return kernel_runs(lambda: TorchLayerNorm(16), torch.randn(2, 3, 16))

    def forward(self, x):
        return apply_kernel(self.inner, x)

@registry.register(
    "dropout",
    "dropout",
    "torch",
    requires=["torch"],
    priority=0,
    constraints={'is_interpretable': False}
    )
class TorchDropout(nn.Module):
    inner: "transparent"
    def __init__(self, p=0.5, inplace=False, *args, **kwargs):
        super().__init__()
        self.inner = nn.Dropout(p=p, inplace=inplace)

    @staticmethod
    def _test():
        return kernel_runs(lambda: TorchDropout(p=0.1), torch.randn(2, 3, 16))

    def forward(self, x, *args, **kwargs):
        return apply_kernel(self.inner, x, *args, **kwargs)
