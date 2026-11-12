"""
File: matformer/transformer_blocks/sequence_mixer/gated_short_conv.py
"""
#Torch
import torch
import torch.nn as nn
#Matformer
from matformer.matformer_module import MatformerModule
from matformer.dataclasses.tensors_dataclasses import TensorDC, NormalTensor, UnpaddedTensor
from matformer.dataclasses.matformer_config import MatformerConfig
from matformer.matformer_registry import registry
#Other
from dataclasses import dataclass, replace
from typing import Optional, Literal, Union, List


@registry.register_dataclass("sequence_mixer", "gated_short_conv")
@dataclass
class GatedShortConvConfig(MatformerConfig):
    conv_size: int = 3          # taps of the causal depthwise conv (HF conv_L_cache)
    conv_bias: bool = False     # bias on in_proj / conv / out_proj (HF conv_bias)
    # Position is implicit in the causal conv, but the field lives here (as in FLAConfig)
    # because the embedding head probes it to decide about learnable position embeddings.
    positional_encoding: Optional[
        Union[
            List[Literal["alibi", "rope", "sinusoidal", "nope", "learnable"]],
            Literal["alibi", "rope", "sinusoidal", "nope", "learnable"],
        ]
    ] = "nope"
    is_causal: bool = True


@registry.register("sequence_mixer", "gated_short_conv", "torch")
class GatedShortConv(MatformerModule):
    """Gated short convolution, the "conv" sequence mixer of LFM2 (Liquid AI).

    A cheaper alternative to attention, used by LFM2 on the majority of its layers:
    in_proj widens the stream by 3x and splits it into (b, c, u); a causal depthwise
    conv runs on b*u, the output is gated by c, and out_proj projects back to
    hidden_size. Numerically identical to transformers' Lfm2ShortConv, whose attribute
    names (in_proj / conv / out_proj) are kept as stable names.
    """

    def __init__(self, config, hidden_size, device=None, layer_idx=None, cache=None,
                 is_causal=False, is_hybrid=False):
        super().__init__()
        self.config = config
        self.in_proj = registry.create('linear', 'linear', in_features=hidden_size,
                                       out_features=3 * hidden_size, bias=config.conv_bias)
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=config.conv_size,
                              groups=hidden_size, bias=config.conv_bias,
                              padding=config.conv_size - 1)
        self.out_proj = registry.create('linear', 'linear', in_features=hidden_size,
                                        out_features=hidden_size, bias=config.conv_bias)

    def forward(self, x, original_x=None, is_causal=None, interpretability=False, **kwargs):
        # Packed sequences are convolved through the padded layout: pad/unpad keep every
        # token in its original position, so a causal conv over zero-filled pads can
        # never leak content across document boundaries.
        was_unpadded = isinstance(x, UnpaddedTensor)
        if was_unpadded:
            x = x.pad()
        padding_mask = getattr(x, 'padding_mask', None)
        tensor = x.tensor if isinstance(x, TensorDC) else x
        if padding_mask is not None:  # HF's apply_mask_to_padding_states
            tensor = tensor * torch.logical_not(padding_mask).unsqueeze(-1)

        projected = self.in_proj(NormalTensor(tensor=tensor)).tensor.transpose(1, 2)
        b, c, u = projected.chunk(3, dim=1)
        y = self.conv(b * u)[:, :, :b.shape[-1]]  # causal: drop the k-1 right-padded outputs
        y = (c * y).transpose(1, 2)
        out = self.out_proj(NormalTensor(tensor=y)).tensor

        out = replace(x, tensor=out)
        return out.unpad() if was_unpadded else out
