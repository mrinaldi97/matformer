import math
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torch.nn import RMSNorm


def init_transformer_weights_(
    module: nn.Module,
    linear_init_style: str = "xavier_uniform",
    embedding_init_style: str = "normal",
    embedding_std: float | None = None,
    embedding_scale: float | None = None,
    embedding_trunc_bounds: tuple[float, float] = (-2.0, 2.0),
    zero_padding_idx: bool = True,
    norm_weight_init: str | float = "ones",
    norm_bias_init: str | float = "zeros",
    norm_weight_std: float = 1.0,
    norm_bias_std: float = 1.0,
    respect_custom_init: bool = True,
) -> None:
    if isinstance(module, nn.Linear):
        if linear_init_style == "xavier_uniform":
            nn.init.xavier_uniform_(module.weight, gain=1.0)
        elif linear_init_style == "xavier_normal":
            nn.init.xavier_normal_(module.weight, gain=1.0)
        elif linear_init_style == "kaiming_uniform":
            nn.init.kaiming_uniform_(
                module.weight, a=0, mode="fan_in", nonlinearity="leaky_relu"
            )
        elif linear_init_style == "kaiming_normal":
            nn.init.kaiming_normal_(
                module.weight, a=0, mode="fan_in", nonlinearity="leaky_relu"
            )
        elif linear_init_style == "uniform":
            nn.init.uniform_(module.weight, a=0.0, b=1.0)
        elif linear_init_style == "normal":
            nn.init.normal_(module.weight, mean=0.0, std=1.0)
        elif linear_init_style == "trunc_normal":
            nn.init.trunc_normal_(
                module.weight, mean=0.0, std=1.0, a=-2.0, b=2.0
            )
        elif linear_init_style == "orthogonal":
            nn.init.orthogonal_(module.weight, gain=1.0)
        elif linear_init_style == "sparse":
            nn.init.sparse_(module.weight, sparsity=0.1, std=0.01)
        elif linear_init_style == "constant":
            nn.init.constant_(module.weight, val=0.0)
        elif linear_init_style == "ones":
            nn.init.ones_(module.weight)
        elif linear_init_style == "zeros":
            nn.init.zeros_(module.weight)
        elif linear_init_style == "eye":
            nn.init.eye_(module.weight)
        else:
            # Resort to the registry if the user has specified a custom init function
            init_fn = registry.create("initialization", linear_init_style)
            init_fn(module.weight)

        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Embedding):
        if embedding_init_style == "normal":
            std = (
                embedding_std
                if embedding_std is not None
                else 1.0 / math.sqrt(module.embedding_dim)
            )
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif embedding_init_style == "trunc_normal":
            std = embedding_std if embedding_std is not None else 0.02
            a, b = embedding_trunc_bounds
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=a, b=b)
        elif embedding_init_style == "xavier_uniform":
            nn.init.xavier_uniform_(module.weight)
        elif embedding_init_style == "xavier_normal":
            nn.init.xavier_normal_(module.weight)
        elif embedding_init_style == "uniform":
            nn.init.uniform_(module.weight, a=0.0, b=1.0)
        elif embedding_init_style == "zeros":
            nn.init.zeros_(module.weight)
        else:
            registry.create("initialization", embedding_init_style)(module.weight)

        if embedding_scale is not None:
            with torch.no_grad():
                module.weight.mul_(embedding_scale)

        if zero_padding_idx and module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].zero_()

    elif isinstance(module, (LayerNorm, RMSNorm)):
        def _apply(tensor: torch.Tensor, spec: str | float, std: float) -> None:
            if isinstance(spec, (int, float)):
                nn.init.constant_(tensor, float(spec))
            elif spec == "ones":
                nn.init.ones_(tensor)
            elif spec == "zeros":
                nn.init.zeros_(tensor)
            elif spec == "constant":
                nn.init.constant_(tensor, 0.0)
            elif spec == "normal":
                nn.init.normal_(tensor, mean=0.0, std=std)
            else:
                registry.create("initialization", spec)(tensor)

        try:
            _apply(module.weight, norm_weight_init, norm_weight_std)
            if getattr(module, "bias", None) is not None:
                _apply(module.bias, norm_bias_init, norm_bias_std)
        except Exception as e:
            print(f"Can't initialize {module}: {e}")

    if respect_custom_init and hasattr(module, "init_weights"):
        module.init_weights()  # Respect custom initialization schema for specific modules
