from matformer.matformer_registry import registry
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import RMSNorm, LayerNorm

@registry.register("linear","linear","torch",requires=["torch"],priority=0)
class TorchLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, *args, **kwargs):
        super().__init__()
        self.inner = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, *args, **kwargs):
        return self.inner(x, *args, **kwargs)


@registry.register("embedding","embedding","torch",requires=["torch"],priority=0)
class TorchEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, *args, **kwargs):
        super().__init__()
        self.inner = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=padding_idx, **kwargs)

    def forward(self, x, *args, **kwargs):
        return self.inner(x, *args, **kwargs)
        
@registry.register("loss","cross_entropy_loss","torch",requires=["torch"],priority=10)
class TorchCrossEntropyLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._args = args
        self._kwargs = kwargs

    def forward(self, logits, targets, **extra_kwargs):
        kw = dict(self._kwargs)
        kw.update(extra_kwargs)
        return F.cross_entropy(logits, targets, *self._args, **kw)


@registry.register("mlp","swiglu","torch",requires=["torch"],priority=0)
class TorchSwiGLU(nn.Module):
    def __init__(
        self,
        config=None,
        hidden_size=None,
        ffn_factor=None
    ):
        super().__init__()
        if config is not None:
            hidden_size = config.hidden_size
            ffn_factor = config.ffn_factor
        ffn_internal_dim = int(hidden_size * ffn_factor)
        self.w13 = nn.Linear(hidden_size, 2 * ffn_internal_dim, bias=False)
        self.w2 = nn.Linear(ffn_internal_dim, hidden_size, bias=False)

    def forward(self, _input):
        x1, x3 = torch.chunk(self.w13(_input), 2, -1)
        return self.w2(F.silu(x1) * x3)


@registry.register("mlp","geglu","torch",requires=["torch"],priority=0)
class TorchGELU(nn.Module):
    def __init__(
        self,
        config=None,
        hidden_size=None,
        ffn_factor=None
    ):
        super().__init__()
        if config is not None:
            hidden_size = config.hidden_size
            ffn_factor = config.ffn_factor
        ffn_internal_dim = int(hidden_size * ffn_factor)
        self.w1 = nn.Linear(hidden_size, ffn_internal_dim, bias=False)
        self.w2 = nn.Linear(ffn_internal_dim, hidden_size, bias=False)

    def forward(self, _input):
        return self.w2(F.gelu(self.w1(_input)))


@registry.register("norm","rmsnorm","torch",requires=["torch"],priority=0)
class TorchRMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.inner = RMSNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x):
        return self.inner(x)


@registry.register("norm","layernorm","torch",requires=["torch"],priority=0)
class TorchLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.inner = LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x):
        return self.inner(x)
