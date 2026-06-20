"""
File: matformer/transformer_blocks/transformer.py
"""
#Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
#Matformer
from matformer.matformer_module import MatformerModule
from matformer.tensors_dataclasses import TensorDC, NormalTensor, PaddedTensor, UnpaddedTensor, ModuleWrapper
from matformer.model_config import ModelConfig  
from matformer.utils.matformer_cache import ensure_cache_and_registry, CachedStuff
from matformer.matformer_registry import registry
from matformer.transformer_blocks import MultiHeadAttention

#Other
from dataclasses import replace
from typing import Optional, List, Literal, Any

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

    available_hooks = [
        'pre_attn', 'post_norm_pre_attn', 'post_attn',
        'pre_mlp',  'post_norm_pre_mlp',  'post_mlp',
        'post_attn_norm', 'post_mlp_norm',
        'residual_stream_output'
    ]
    def __init__(self, config: ['ModelConfig', 'LayerConfig'], 
                 block_mask=None, layer_idx=None, cache=None):
        """Initialize transformer block. 
        Args:
            config: ModelConfig or LayerConfig for this layer
            block_mask: Optional block mask (TODO: maybe deprecated)
            layer_idx: Layer index for layer-specific configs
            cache: Shared cache for registry and precomputed values
        """
        super().__init__()
        self.cache = ensure_cache_and_registry(cache)
        
        # Extract layer-specific configuration
        if isinstance(config, ModelConfig):
            layer_config = (config.get_layer_config(layer_idx) 
                          if layer_idx is not None 
                          else config.default_layer)
        else:
            layer_config = config
        
        self.config = config
        self.layer_config = layer_config
        self.norm_position = layer_config['normalization_position']
        self.layer_idx=layer_idx
        # Initialize normalization layers
        norm_kwargs = {
            "normalized_shape": config.hidden_size,
            "eps": config.rms_norm_eps,
            "elementwise_affine": True
        }
        self.attn_norm = ModuleWrapper(
            self.cache.registry.create("norm", layer_config['normalization'], **norm_kwargs)
        )
        self.mlp_norm = ModuleWrapper(
            self.cache.registry.create("norm", layer_config['normalization'], **norm_kwargs)
        )
        
        # Initialize self-attention
        self.self_attn = MultiHeadAttention(
            bias=config.bias,
            q_dim=config.hidden_size,
            nheads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            cache=self.cache,
            attn_impl=layer_config['attn_impl'],
            positional_encoding=layer_config['positional_encoding'],
            is_causal=config.is_causal,
            is_hybrid=getattr(config, 'training_objective', None) == 'hybrid',
            sliding_window=layer_config['sliding_window_size'],
            device=self.device,
            layer_idx=layer_idx,
            interpretability_friendly=layer_config.get('interpretability_friendly',False),
            hooks=layer_config.get('hooks', {})
            
        )
        # Initialize MLP
        self.mlp = ModuleWrapper(
            self.cache.registry.create(
                "mlp", 
                layer_config["ffn_activation"],
                hidden_size=config.hidden_size,
                ffn_factor=config.ffn_factor
            )
        )
        
        # Initialize hook system
        self.hooks = self._init_hooks(layer_config.get('hooks', {}))

    def _norm_attn_residual(self, x, original_x, is_causal=None, interpretability=False):
        """Apply normalization, attention, and residual connection."""
        if self.norm_position == 'pre':
            normed = self.attn_norm(x)
            normed = self._apply_hook("post_norm_pre_attn", normed)
            attn_out = self.self_attn(normed, original_x=original_x, is_causal=is_causal, interpretability=interpretability)
            attn_out = self._apply_hook("post_attn", attn_out)
            return x + attn_out
        else:  # post normalization
            attn_out = self.self_attn(x, original_x=original_x, is_causal=is_causal, interpretability=interpretability)
            attn_out = self._apply_hook("post_attn", attn_out)
            x = x + attn_out
            x = self.attn_norm(x)
            return self._apply_hook("post_attn_norm", x)
    
    def _norm_mlp_residual(self, x):
        """Apply normalization, MLP, and residual connection."""
        if self.norm_position == 'pre':
            normed = self.mlp_norm(x)
            normed = self._apply_hook("post_norm_pre_mlp", normed)
            mlp_out = self.mlp(normed)
            mlp_out = self._apply_hook("post_mlp", mlp_out)
            return x + mlp_out
        else:  # post normalization
            mlp_out = self.mlp(x)
            mlp_out = self._apply_hook("post_mlp", mlp_out)
            x = x + mlp_out
            x = self.mlp_norm(x)
            return self._apply_hook("post_mlp_norm", x)
    
    def forward(self, x, block_mask=None, sliding=False, is_causal=None, interpretability=False):
        """Forward pass through transformer block.     
        If self.norm_position == pre:
            pre_attn => norm => attn => residual => pre_mlp => norm => mlp => residual => output
        else (self.norm_position == post):
            pre_attn => attn => norm => residual => pre_mlp => mlp => norm => residual => output

        """
        # WERSA attention requires original input
        original_x = x if self.layer_config['attn_impl'] == 'wersa' else None
        
        # Attention block
        x = self._apply_hook("pre_attn", x)
        x = self._norm_attn_residual(x, original_x, is_causal, interpretability)
        
        # MLP block
        x = self._apply_hook("pre_mlp", x)
        x = self._norm_mlp_residual(x)
        
        # Final output hook
        return self._apply_hook("residual_stream_output", x)

class NakedTransformer(MatformerModule):
    """
    This transformer module misses the embedding as well as the "unembedding" layer.
    The reason is that is a Transformer meant to run only on "patches".
    It applies n transformer blocks as defined in the ModelConfig
    """  
    # Stable parameter names
    norm: "param_name:norm"
    layers: "param_name:layers"
    
    def __init__(self, config: ModelConfig, cache=None):
        super().__init__()
        self.config = config
        self.cache = ensure_cache_and_registry(cache)
        
        norm_kwargs = {
            "normalized_shape": config.hidden_size,
            "eps": config.rms_norm_eps,
            "elementwise_affine": True
        }
        self.norm = ModuleWrapper(
            self.cache.registry.create("norm", config.default_layer['normalization'], **norm_kwargs)
        )
        
        self.layers = nn.ModuleList([
            TransformerBlock(config=config, layer_idx=idx, cache=self.cache)
            for idx in range(config.num_hidden_layers)
        ])
    
    def forward(self, x, document_mask=None, **kwargs):
        for layer in self.layers:
            x = layer(x, **kwargs)
        return self.norm(x)

