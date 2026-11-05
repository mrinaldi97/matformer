# matformer/initialization.py

import math
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torch.nn import RMSNorm

def init_transformer_weights_(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1.0)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Embedding):
        std = 1.0 / math.sqrt(module.embedding_dim)
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].zero_()

    elif isinstance(module, (LayerNorm, RMSNorm)):
        nn.init.ones_(module.weight)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)
