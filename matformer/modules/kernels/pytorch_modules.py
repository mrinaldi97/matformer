from matformer.matformer_registry import registry
from matformer.matformer_module import MatformerModule
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import RMSNorm, LayerNorm

@registry.register("norm", "identity", "default", metadata={"is_interpretable": False})
class Identity:
    def __call__(self, x, **kw): return x

@registry.register(
    "linear",
    "linear",
    "torch",
    requires=["torch"],
    priority=0,
    params_names={'inner.weight': 'weight', 'inner.bias': 'bias'},
    metadata={"is_interpretable": False}
)
class TorchLinear(nn.Module):
    inner: "transparent"
    def __init__(self, in_features, out_features, bias=True, *args, **kwargs):
        super().__init__()
        self.inner = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, *args, **kwargs):
        return x.apply(self.inner, *args, **kwargs)


@registry.register(
    "embedding",
    "embedding",
    "torch",
    requires=["torch"],
    priority=0,
    params_names={'inner.weight': 'weight', 'inner.bias': 'bias'},
    metadata={"is_interpretable": False}
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

    def forward(self, x, *args, **kwargs):
        return x.apply(self.inner, *args, **kwargs)


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

    def forward(self, logits, targets, **extra_kwargs):
        kw = dict(self._kwargs)
        kw.update(extra_kwargs)
        return F.cross_entropy(logits, targets, *self._args, **kw)

@registry.register("loss", "bce", "torch", requires=["torch"], priority=10)
class BCELoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._args = args
        self._kwargs = kwargs
    def forward(self, logits, labels, **extra_kwargs):
        loss = F.binary_cross_entropy_with_logits(
            logits, labels.float(), reduction="none"
        )
        
        
@registry.register(
    "mlp",
    "swiglu",
    "torch",
    requires=["torch"],
    priority=0,
    params_names={
        'gate_proj.weight': 'gate_proj.weight',
        'up_proj.weight':   'up_proj.weight',
        'down_proj.weight': 'down_proj.weight'
    },
    metadata={"is_interpretable": True}
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
        self.gate_proj = nn.Linear(hidden_size, ffn_internal_dim, bias=bias)
        self.up_proj   = nn.Linear(hidden_size, ffn_internal_dim, bias=bias)
        self.down_proj = nn.Linear(ffn_internal_dim, hidden_size, bias=bias)
        self.init_sockets()
        self.add_socket("hidden_act")
    def forward(self, x):
        self.stream.init(x.apply(lambda t: F.silu(self.gate_proj(t)) * self.up_proj(t)))
        self.stream.apply("hidden_act")
        return self.stream.exit().apply(self.down_proj)


@registry.register(
    "mlp",
    "geglu",
    "torch",
    requires=["torch"],
    priority=0,
    params_names={
        'gate_proj.weight': 'gate_proj.weight',
        'up_proj.weight':   'up_proj.weight',
        'down_proj.weight': 'down_proj.weight'
    },
    metadata={"is_interpretable": True}
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

        self.gate_proj = nn.Linear(hidden_size, ffn_internal_dim, bias=bias)
        self.up_proj   = nn.Linear(hidden_size, ffn_internal_dim, bias=bias)
        self.down_proj = nn.Linear(ffn_internal_dim, hidden_size, bias=bias)
        self.init_sockets()
        self.add_socket("hidden_act")

    def forward(self, x):
        self.stream.init(x.apply(lambda t: F.gelu(self.gate_proj(t)) * self.up_proj(t)))
        self.stream.apply("hidden_act")
        return self.stream.exit().apply(self.down_proj)

@registry.register(
    "mlp",
    "gelu",
    "torch",
    requires=["torch"],
    priority=0,
    params_names={
        'up_proj.weight': 'up_proj.weight',
        'down_proj.weight': 'down_proj.weight'
    },
    metadata={"is_interpretable": True}
)
class TorchGELU(MatformerModule):
    inner: "transparent"
    def __init__(self, config=None, hidden_size=None, ffn_factor=None, bias=False):
        super().__init__()
        if config is not None:
            hidden_size = config.hidden_size
            ffn_factor = config.ffn_factor
        ffn_internal_dim = int(hidden_size * ffn_factor)
        self.up_proj = nn.Linear(hidden_size, ffn_internal_dim, bias=bias)
        self.down_proj = nn.Linear(ffn_internal_dim, hidden_size, bias=bias)
        self.init_sockets()
        self.add_socket("hidden_act")

    def forward(self, x):
        self.stream.init(x.apply(lambda t: F.gelu(self.up_proj(t))))
        self.stream.apply("hidden_act")
        return self.stream.exit().apply(self.down_proj)
@registry.register(
    "mlp",
    "gelu_tanh",
    "torch",
    requires=["torch"],
    priority=0,
    params_names={
        'up_proj.weight': 'up_proj.weight',
        'down_proj.weight': 'down_proj.weight'
    }
)
class TorchGELUTanh(TorchGELU):
    """2-matrix MLP with the tanh-approximated GELU (HF gelu_pytorch_tanh, e.g. GPT-2, Gemma)."""
    def forward(self, x):
        self.stream.init(x.apply(lambda t: F.gelu(self.up_proj(t), approximate="tanh")))
        self.stream.apply("hidden_act")
        return self.stream.exit().apply(self.down_proj)


@registry.register(
    "mlp",
    "geglu_tanh",
    "torch",
    requires=["torch"],
    priority=0,
    params_names={
        'gate_proj.weight': 'gate_proj.weight',
        'up_proj.weight':   'up_proj.weight',
        'down_proj.weight': 'down_proj.weight'
    }
)
class TorchGEGLUTanh(TorchGEGLU):
    """Gated 3-matrix MLP with tanh-GELU — HF Gemma's hidden_activation gelu_pytorch_tanh."""
    def forward(self, x):
        self.stream.init(x.apply(lambda t: F.gelu(self.gate_proj(t), approximate="tanh") * self.up_proj(t)))
        self.stream.apply("hidden_act")
        return self.stream.exit().apply(self.down_proj)


@registry.register(
    "norm",
    "rmsnorm",
    "torch",
    requires=["torch"],
    priority=0,
    params_names={'inner.weight': 'weight'},
    metadata={"is_interpretable": False}
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

    def forward(self, x):
        return x.apply(self.inner)

@registry.register(
    "norm",
    "rmsnorm_zc",
    "torch",
    requires=["torch"],
    priority=0,
    params_names={'weight': 'weight'},
    metadata={"is_interpretable": False}
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

    def forward(self, x):
        return x.apply(lambda t: t * torch.rsqrt(t.float().pow(2).mean(-1, keepdim=True) + self.eps).to(t.dtype)
                       * (1 + self.weight if self.zero_centered else self.weight))



@registry.register(
    "norm",
    "layernorm",
    "torch",
    requires=["torch"],
    priority=0,
    params_names={'inner.weight': 'weight', 'inner.bias': 'bias'},
    metadata={"is_interpretable": False}
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

    def forward(self, x):
        return x.apply(self.inner)

@registry.register(
    "dropout",
    "dropout",
    "torch",
    requires=["torch"],
    priority=0,
    metadata={"is_interpretable": False}
    )
class TorchDropout(nn.Module):
    inner: "transparent"
    def __init__(self, p=0.5, inplace=False, *args, **kwargs):
        super().__init__()
        self.inner = nn.Dropout(p=p, inplace=inplace)

    def forward(self, x, *args, **kwargs):
        return x.apply(self.inner, *args, **kwargs)
