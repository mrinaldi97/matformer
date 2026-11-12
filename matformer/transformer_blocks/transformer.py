"""
File: matformer/transformer_blocks/transformer.py
"""
#Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
#Matformer
from matformer.matformer_module import MatformerModule
from matformer.dataclasses.tensors_dataclasses import TensorDC, NormalTensor, PaddedTensor, UnpaddedTensor
from matformer.dataclasses.model_dataclasses import ModelConfig
from matformer.utils.matformer_cache import ensure_cache, CachedStuff
from matformer.matformer_registry import registry

#Other
from dataclasses import replace
from typing import Optional, List, Literal, Any
def repeat(layers, times):
    return list(layers) * times

def cut(ids, start=None, stop=None):
    """start: enter at the first visit of that layer. stop: exit after its last visit."""
    if start is not None:
        ids = ids[ids.index(start):]
    if stop is not None:
        ids = ids[:len(ids) - ids[::-1].index(stop)]
    return ids
class TransformerBlock(MatformerModule):
    """Transformer self-attention block with pre/post normalization and flexible hooks.
    
    Architecture:
        - Configurable normalization position (pre/post)
        - Self-attention layer
        - Feed-forward (MLP) layer
        - Residual connections
        - Optional hook system for custom interventions
    
    Configuration is layer-specific: each layer can have different settings
    as defined in ModelConfig, or use a direct LayerConfig.
    """
    
    # Stable parameter names (independent of internal implementation changes)
    attn_norm: "param_name:attn_norm"
    mlp_norm: "param_name:mlp_norm"
    self_attn: "param_name:attn"
    mlp: "param_name:mlp"
    NORM_SLOTS = ["pre_attn", "post_attn", "post_add_attn", "pre_mlp", "post_mlp", "post_add_mlp"]

    def __init__(self, config, layer_idx=None, cache=None):
        """Initialize transformer block. 
        Args:
            config: ModelConfig or LayerConfig for this layer
            layer_idx: Layer index for layer-specific configs
            cache: Shared cache for registry and precomputed values
        """
        super().__init__()
        self.cache = ensure_cache(cache)
        
        if isinstance(config, ModelConfig):
            layer_config = (config.get_layer_config(layer_idx) 
                          if layer_idx is not None 
                          else config.default_layer)
        else:
            layer_config = config
        self.config = config
        self.layer_config = layer_config
        self.init_sockets(stream_type=layer_config.stream_type)
        # Initialize normalization layers
        # bias only when explicitly disabled: RMSNorm kernels take no bias argument
        norm_kwargs = {"normalized_shape": config.hidden_size,
                       "eps": config.rms_norm_eps,
                       "elementwise_affine": getattr(config, "norm_affine", True)}
        if not getattr(config, "norm_bias", True):
            norm_kwargs["bias"] = False
        norm_style = registry.create("norm_style", layer_config.normalization_position)
        for slot in self.NORM_SLOTS:
            if norm_style(slot):
                norm = registry.create("norm", layer_config.normalization, **norm_kwargs)
                self.register_socket(slot, norm, attr=norm_style.attr(slot))
            else:
                self.add_socket(slot)            # empty = identity

        self.register_socket("self_attn", registry.create(
            "sequence_mixer", layer_config.sequence_mixer,
            config=layer_config.sequence_mixer_config, hidden_size=config.hidden_size,
            layer_idx=layer_idx, cache=self.cache, is_causal=config.is_causal,
            is_hybrid=config.training_objective == "hybrid", device=self.device),
            attr="self_attn")
        self.register_socket("mlp", registry.create(
            "mlp", layer_config.ffn_activation, hidden_size=config.hidden_size,
            ffn_factor=config.ffn_factor, bias=layer_config.mlp_bias),
            attr="mlp")
        self.add_socket("layer_output")

    
    def forward(self, x, block_mask=None, sliding=False, is_causal=None, interpretability=False):
        """Forward pass through transformer block.     
        If self.norm_position == pre:
            pre_attn => norm => attn => residual => pre_mlp => norm => mlp => residual => output
        else (self.norm_position == post):
            pre_attn => attn => norm => residual => pre_mlp => mlp => norm => residual => output
        # WERSA attention requires original input

        """
        self.stream.init(x)
        with self.stream.branch() as s:
            s.apply("pre_attn")
            s.apply("self_attn", original_x=None, is_causal=is_causal, interpretability=interpretability)
            s.apply("post_attn")
        self.stream.apply("post_add_attn")

        with self.stream.branch() as s:
            s.apply("pre_mlp")
            s.apply("mlp")
            s.apply("post_mlp")
        self.stream.apply("post_add_mlp")
        self.stream.apply("layer_output")
        return self.stream.exit()

class NakedTransformer(MatformerModule):
    """
    This transformer module misses the embedding as well as the "unembedding" layer.
    The reason is that is a Transformer meant to run only on "patches".
    It applies n transformer blocks as defined in the ModelConfig
    """  
    # Stable parameter names
    norm: "param_name:norm"
    layers: "param_name:layers"
    def __init__(self, config: ModelConfig, cache=None, **kwargs):
        super().__init__()
        self.config = config
        self.cache = ensure_cache(cache)
        self.init_sockets(stream_type=config.stream_type)
        
        # bias only when explicitly disabled: RMSNorm kernels take no bias argument
        norm_kwargs = {
            "normalized_shape": config.hidden_size,
            "eps": config.rms_norm_eps,
            "elementwise_affine": getattr(config, "norm_affine", True)
        }
        if not getattr(config, "norm_bias", True):
            norm_kwargs["bias"] = False
        n = config.num_hidden_layers
        self.schedule = config.layer_schedule or list(range(n)) #Can be used for looped transformers
        self.layers = nn.ModuleList([
            TransformerBlock(config=config, layer_idx=i, cache=self.cache) for i in range(n)])
        for i, layer in enumerate(self.layers):
            self.register_socket(f"layer_{i}", layer)           

        # Final norm (after the last block). Models whose reference architecture ships no
        # final norm (e.g. HF BERT) set final_norm: false in the config: the socket is
        # registered empty, which is identity in the stream — no special-casing downstream.
        if getattr(config, "final_norm", True):
            self.register_socket("final_norm",
                registry.create("norm", config.default_layer["normalization"], **norm_kwargs),
                attr="norm")
        else:
            self.add_socket("final_norm")

        self.add_socket("before_unembed")

    def forward(self, x, start=None, stop=None, **kwargs):
        self.stream.init(x)
        for i in cut(self.schedule, start, stop):
            name = f"layer_{i}"
            self.stream.apply(name, **kwargs)
        self.stream.apply("final_norm")
        self.stream.apply("before_unembed")
        return self.stream.exit()

