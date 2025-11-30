"""
File: matformer/transformer_blocks.py
"""
#Don't remember why
import sys
sys.path.append('../')
#Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
#Matformer
from matformer.matformer_module import MatformerModule
from matformer.tensors_dataclasses import TensorDC, NormalTensor, PaddedTensor, UnpaddedTensor, ModuleWrapper
from matformer.model_config import ModelConfig  
from matformer.cached_stuff import CachedStuff
from matformer.matformer_registry import registry
from matformer.masked_models import Maskerator
from matformer.matformer_tokenizers import ByteLevelTokenizer,MatformerTokenizer
#Other
from functools import partial, reduce
from tqdm import tqdm
from datetime import datetime
from dataclasses import replace
from copy import deepcopy
from warnings import warn
from typing import Optional, List, Literal, Any
#Transformers
from transformers.modeling_outputs import SequenceClassifierOutput


def ensure_cache_and_registry(cache):
    """Ensure cache and registry initialization"""
    if cache is None:
        warn("Cache not provided, initializing new cache. Consider passing cache for efficiency.")
        cache = CachedStuff()
    if not hasattr(cache, 'registry'):
        warn("Registry not found in cache, initializing. Consider initializing registry in cache.")
        cache.registry = registry
    return cache

class MultiHeadAttention(MatformerModule):
    # Stable parameter names
    packed_proj: "param_name:qkv_proj"      # Packed Q+K+V projection (when applicable)
    q_proj: "param_name:q_proj"             # Separate query projection (cross-attention)
    k_proj: "param_name:k_proj"             # Separate key projection (cross-attention)
    v_proj: "param_name:v_proj"             # Separate value projection (cross-attention)
    out_proj: "param_name:o_proj"           # Output projection
    attn_kernel: "param_name:attn_kernel"   # Attention implementation kernel
    
    def __init__(
        self,
        q_dim: int,               
        k_dim: Optional[int] = None,
        v_dim: Optional[int] = None,
        is_cross_attention: bool = False,
        nheads: int = 8,
        bias: bool = False,
        positional_encoding = ['rope'],  # 'alibi', 'rope', 'nope', 'sinusoidal', 'learnable'
        #dropout: float = 0.0, # Not supported by FlexAttention yet
        cache: Optional['CachedStuff'] = None,
        attn_variant: str = 'normal',  # 'normal', 'wersa', 'linear', etc.
        attn_impl: str = 'flash',  # Preferred implementation within variant
        is_causal: bool = True,
        sliding_window: Optional[int] = None,
        device: str = 'cuda'  
    ):
        super().__init__()
        if isinstance(positional_encoding, str):
            positional_encoding = [positional_encoding]        
        # Assertions
        assert q_dim % nheads == 0, "q_dim is not divisible by nheads"
        if is_cross_attention:
            assert k_dim is not None, "You asked for a cross attention, but you haven't provided keys dim"
            assert v_dim is not None, "You asked for a cross attention, but you haven't provided values dim"
        else:
            k_dim=q_dim
            v_dim=q_dim 

        # Initialization
        if cache is None:     
            warn("Cache not provided to MultiHeadAttention, initializing new cache. Consider passing cache for efficiency.")
            cache = CachedStuff()
        self.cache = cache
        
        # Initialize registry if not present
        if not hasattr(self.cache, 'registry'):
            warn("Registry not found in cache, initializing. Consider initializing registry in cache.")
            self.cache.registry = registry
        self.force_separed_qkv=False
        self.nheads = nheads
        self.attn_variant = attn_variant
        self.positional_encoding = positional_encoding
        self.is_causal = is_causal
        self.is_cross_attention = is_cross_attention
        self.sliding_window = sliding_window
        self.bias = bias
        self.qkv_samedim = q_dim == k_dim and q_dim == v_dim
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.head_dim = q_dim // nheads
        
        # RoPE initialization
        if 'rope' in self.positional_encoding:
            #self.rotary_embedding = self.cache.get_rotary_emb(self.head_dim)
            self.rotary_embedding = self.cache.registry.create('positional_encoding','rope')
            self.rotary_embedding_meta=self.rotary_embedding._matformer_metadata
        # Initialize attention kernel
        self.attn_kernel = self.cache.registry.create(
            'attention', 
            attn_variant,
            preferred=attn_impl,
            nheads=nheads,
            head_dim=self.head_dim,
            is_causal=is_causal,
            sliding_window=sliding_window,
            positional_encoding=positional_encoding,
            cache=cache,
            device=device
        )
        
        # Get kernel metadata
        self.kernel_meta = self.attn_kernel._matformer_metadata
        
        if not is_cross_attention or self.qkv_samedim: # Packed qkv projection for efficiency  
            self.packed_proj=self.cache.registry.create('linear','linear',in_features=self.q_dim, out_features=3*q_dim,bias=bias)
        else:
            self.q_proj=self.cache.registry.create('linear','linear',in_features=q_dim, out_features=q_dim,bias=bias)
            self.k_proj=self.cache.registry.create('linear','linear',in_features=k_dim, out_features=q_dim,bias=bias)
            self.v_proj=self.cache.registry.create('linear','linear',in_features=v_dim, out_features=q_dim,bias=bias)

        self.out_proj=self.cache.registry.create('linear','linear',in_features=q_dim,out_features=self.q_dim,bias=bias) # Out projection

    @staticmethod
    def _pack_qkv(q, k, v):
        normalize = lambda s: s.translate(str.maketrans('', '', '?B'))
        assert normalize(q.tensor_order) == normalize(k.tensor_order) == normalize(v.tensor_order), "QKV must have same tensor order"
        current_order_norm = normalize(q.tensor_order)
        
        if current_order_norm == "HSD":
            # [?, H, S, D] => [?, S, H, D]
            q_t = q.tensor.transpose(1, 2)
            k_t = k.tensor.transpose(1, 2)
            v_t = v.tensor.transpose(1, 2)
            #  [?, S, 3, H, D]
            packed_tensor = torch.stack([q_t, k_t, v_t], dim=-3)
            new_order = '?S3HD'
        elif current_order_norm == "SHD":
            # [?, S, 3, H, D]
            packed_tensor = torch.stack([q.tensor, k.tensor, v.tensor], dim=-3)
            new_order = '?S3HD'
        else:
            raise ValueError(f"Unsupported tensor order: {q.tensor_order}")
        
        return replace(q, tensor=packed_tensor, tensor_order=new_order)
    
    @staticmethod
    def _unpack_qkv(qkv_packed):
            # S3HD => ?SHD
            q_t, k_t, v_t = qkv_packed.tensor.unbind(dim=-3)
            order = qkv_packed.tensor_order.replace('3', '')
            return (replace(qkv_packed, tensor=q_t, tensor_order=order),
                    replace(qkv_packed, tensor=k_t, tensor_order=order),
                    replace(qkv_packed, tensor=v_t, tensor_order=order))
        
    @staticmethod
    def _transpose_for_kernel(tensor, wanted_from_kernel):
        """ Currently this helper function only supports conversion from and to flash style, sdpa style 
            BHSD
            BSHD
            that is, when transposing dimensions 1 and 2 
            Every other conversion wil raise an error
            Letters 'B' and '?' are trated as if they are the same, it represents equivalence of batching/unbatching in Matformer's jargon.
            The tensorDC must have a 'tensor_order' attribute, if not the conversion will fail too
        """
        if not isinstance(tensor, TensorDC) or not hasattr(tensor, 'tensor_order'):
            raise Exception       
        normalize = lambda s: s.translate(str.maketrans('', '', '?B'))
        wanted = normalize(wanted_from_kernel)
        current = normalize(tensor.tensor_order)        
        if current == wanted:
            return tensor  # No change required, great!  
        if (current=='SHD' and wanted=='HSD') or (current=='HSD' and wanted == 'SHD'):  
            # It can be transposed
            return replace(tensor, tensor=tensor.tensor.transpose(1,2), tensor_order='?' + wanted)   
        print(f"Current: {current} Wanted: {wanted}")
        raise Exception

    def forward(self, query_input, original_x=None, key_input=None, value_input=None, document_mask=None):
        """
        Input Tensors:
        query_input: (Batch, Seqlen, Qdim)
        key_input: (Batch, Seqlen, Kdim) [If omitted, self-attention]
        value_input: (Batch, Seqlen, Vdim) [If omitted, self-attention]
        Output:
        (Batch, Seqlen, Qdim) # Qdim is the query dimension as well as the output.
        """
        
        # 1. Extract max seq len and cu_seqlens (if appliable), convert to Normal Tensor in case a pytorch tensor is passed.
        cu_seqlens=None
        if isinstance(query_input, UnpaddedTensor):
            max_seq_len = query_input.max_seq_len # This variable can be used, for example, to get the correct RoPe cos and sin from cache, but in general is useful
            cu_seqlens=query_input.cu_seqlens.to(query_input.device) # Used for unpadding
        elif isinstance(query_input, PaddedTensor):
            max_seq_len = query_input.shape[1]
        else: #normal tensor
            shape=query_input.shape
            if len(shape)==2:
                max_seq_len = query_input.shape[0]
            else:
                max_seq_len=query_input.shape[1]
                
                
        #self.device=query_input.device
        #self.dtype=query_input.dtype
        
        # 2. 
        
        supports_unpadding = self.kernel_meta.get('supports_unpadding', False)
        if 'rope' in self.positional_encoding:
            supports_unpadding = supports_unpadding and self.rotary_embedding_meta.get('supports_unpadding', False) # If either the RoPe or the attn kernel do not support unpadding, sequence is repadded
        supports_packed_qkv = self.kernel_meta.get('supports_packed_qkv', False)
        rope_supports_packed = self.rotary_embedding_meta.get('supports_packed_qkv', False) if  'rope' in self.positional_encoding else True  
        
        # Set defaults for self-attention
        
        if key_input is None:
            key_input = query_input
        if value_input is None:
            value_input = query_input
                
        # If the attention implementation does not support unpadding, eventually unpadded sequences must be padded
        repadded = False
        if not supports_unpadding:
            if isinstance(query_input, UnpaddedTensor):
                repadded = True  # A flag: if original inputs were padded from unpadded tensors, they will be unpadded again at the end
                #warn('The selected attention implementation does not support unpadding. Sequence is automatically repadded. This can lead to a loss in performances')
                query_input = query_input.pad()
                key_input = key_input.pad()
                value_input = value_input.pad()


        
        # Projecting (eventually, packed projection)
        if self.qkv_samedim and not self.is_cross_attention and not self.force_separed_qkv:
            qkv_projected = NormalTensor(tensor=self.packed_proj(query_input.tensor), tensor_order='?S(3*D)')
            q, k, v = None, None, None
        elif self.qkv_samedim and self.is_cross_attention and not self.force_separed_qkv:
            qkv_projected = None
            w = torch.chunk(self.packed_proj.weight, 3, dim=0)
            b = torch.chunk(self.packed_proj.bias, 3, dim=0) if self.bias else (None, None, None)
            q = NormalTensor(tensor=F.linear(query_input.tensor, w[0], b[0]),tensor_order='?SD')
            k = NormalTensor(tensor=F.linear(key_input.tensor, w[1], b[1]),tensor_order='?SD')
            v = NormalTensor(tensor=F.linear(value_input.tensor, w[2], b[2]),tensor_order='?SD')
        else:
            qkv_projected = None
            q = NormalTensor(tensor=self.q_proj(query_input.tensor),tensor_order='?SD')
            k = NormalTensor(tensor=self.k_proj(key_input.tensor),tensor_order='?SD')
            v = NormalTensor(tensor=self.v_proj(value_input.tensor),tensor_order='?SD')
            
        # 3. Heads creation  
        if qkv_projected is not None:
            qkv_projected = replace(qkv_projected, tensor=qkv_projected.tensor.unflatten(-1, [3, self.nheads, self.head_dim]), tensor_order='?S3HD')
        else:
            q = replace(q, tensor=q.tensor.unflatten(-1, [self.nheads, self.head_dim]),tensor_order='?SHD')
            k = replace(k, tensor=k.tensor.unflatten(-1, [self.nheads, self.head_dim]),tensor_order='?SHD')
            v = replace(v, tensor=v.tensor.unflatten(-1, [self.nheads, self.head_dim]),tensor_order='?SHD')

        # 3b. (facultative) Apply RoPe
        if 'rope' in self.positional_encoding:
            repack_after_rope = False  
            if qkv_projected is not None and not rope_supports_packed: # Unpack if RoPe doesn't support packed qkv  
                q, k, v = self._unpack_qkv(qkv_projected)
                qkv_projected = None
                repack_after_rope = supports_packed_qkv  #Repack only if supported by the attention kernel
            elif qkv_projected is not None and rope_supports_packed:
                assert qkv_projected.tensor_order == self.rotary_embedding_meta['tensor_order_qkv_packed_input']
            
            if qkv_projected is None:  #non-packed branch, adapt to requested tensor order 
                q = self._transpose_for_kernel(q, self.rotary_embedding_meta['tensor_order_input'])
                k = self._transpose_for_kernel(k, self.rotary_embedding_meta['tensor_order_input'])

            # 1. Get sin and cos from cache
            cos, sin = self.cache.get_rotary_cos_sin(max_seq_len, self.head_dim, device=query_input.device, dtype=query_input.dtype)  

            # 2. Rotate query and keys
            qkv_t, q_t, k_t = self.rotary_embedding(
                qkv=qkv_projected.tensor if qkv_projected is not None else None,
                q=q.tensor if q is not None else None,
                k=k.tensor if k is not None else None,
                cos=cos, sin=sin,
                cu_seqlens=cu_seqlens, max_seq_len=max_seq_len
            )
            if qkv_projected is not None:
                qkv_projected = replace(qkv_projected, tensor=qkv_t, tensor_order=self.rotary_embedding_meta['tensor_order_qkv_packed_output'])
            else:
                q = replace(q, tensor=q_t, tensor_order=self.rotary_embedding_meta['tensor_order_output'])
                k = replace(k, tensor=k_t, tensor_order=self.rotary_embedding_meta['tensor_order_output'])


            # Repack if needed for attention kernel 
            if repack_after_rope:
                qkv_projected = self._pack_qkv(q, k, v)
                q, k, v = None, None, None
        kernel_input_order = self.kernel_meta.get('tensor_order_input', '?SHD')
        tensor_order_qkv_packed = self.kernel_meta.get('tensor_order_qkv_packed_input', None)
        
        if qkv_projected is not None and not supports_packed_qkv:
            q, k, v = self._unpack_qkv(qkv_projected)
            qkv_projected = None

        if qkv_projected is not None:
            qkv_projected = self._transpose_for_kernel(qkv_projected, tensor_order_qkv_packed if tensor_order_qkv_packed is not None else kernel_input_order)
        else:
            q = self._transpose_for_kernel(q, kernel_input_order)
            k = self._transpose_for_kernel(k, kernel_input_order)
            v = self._transpose_for_kernel(v, kernel_input_order)

        # Attention computation
        attn_output = self.attn_kernel(
            qkv=qkv_projected.tensor if qkv_projected is not None else None,
            q=q.tensor if q is not None else None,
            k=k.tensor if k is not None else None,
            v=v.tensor if v is not None else None,
            query_input=query_input, key_input=key_input,
            original_x=original_x
        )

        # Transpose from kernel's output format to expected format (B, S, H, Hd)
        kernel_output_order = self.kernel_meta.get('tensor_order_output', '?SHD')
        attn_output = NormalTensor(tensor=attn_output, tensor_order=kernel_output_order)
        attn_output = self._transpose_for_kernel(attn_output, '?SHD')

        # Post-attention stuff
        attn_output = attn_output.tensor.flatten(-2)  # (B, S, H, Hd) -> (B, S, D)
        output_tensor = self.out_proj(attn_output)

        # Handle output based on input type
        if hasattr(query_input, 'tensor'):
            query_input = replace(query_input, tensor=output_tensor)
        else:
            query_input = output_tensor

        if repadded:
            query_input = query_input.unpad()

        return query_input  




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
            cache=self.cache,
            attn_impl=layer_config['attn_impl'],
            positional_encoding=layer_config['positional_encoding'],
            is_causal=config.is_causal,
            sliding_window=layer_config['sliding_window_size'],
            device=self.device
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
    
    def _init_hooks(self, hook_specs):
        """Initialize hooks from layer config via registry."""
        if not hook_specs:
            return {}
        
        hooks = {}
        for name, hook_name in hook_specs.items():
            # Resolve hook implementation from registry
            hook = self.cache.registry.create("hooks", hook_name,config=self.config, cache=self.cache)
            # Register as submodule if it's a nn.Module
            if isinstance(hook, nn.Module):
                self.add_module(f"hook_{name}", hook)
            hooks[name] = hook
        
        return hooks
    
    def _apply_hook(self, hook_name, x, *args, **kwargs):
        """Apply hook if it exists, otherwise return input unchanged."""
        return self.hooks[hook_name](x, *args, **kwargs) if hook_name in self.hooks else x
    
    def _norm_attn_residual(self, x, original_x):
        """Apply normalization, attention, and residual connection."""
        if self.norm_position == 'pre':
            normed = self.attn_norm(x)
            normed = self._apply_hook("post_norm_pre_attn", normed)
            attn_out = self.self_attn(normed, original_x=original_x)
            attn_out = self._apply_hook("post_attn", attn_out)
            return x + attn_out
        else:  # post normalization
            attn_out = self.self_attn(x, original_x=original_x)
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
    
    def forward(self, x, block_mask=None, sliding=False):
        """Forward pass through transformer block.     
        If self.norm_position == pre:
            pre_attn => norm => attn => residual => pre_mpl => norm => mlp => residual => output
        else (self.norm_position == post):
            pre_attn => attn => norm => residual => pre_mlp => mlp => norm => residual => output

        """
        # WERSA attention requires original input
        original_x = x if self.layer_config['attn_impl'] == 'wersa' else None
        
        # Attention block
        x = self._apply_hook("pre_attn", x)
        x = self._norm_attn_residual(x, original_x)
        
        # MLP block
        x = self._apply_hook("pre_mlp", x)
        x = self._norm_mlp_residual(x)
        
        # Final output hook
        return self._apply_hook("before_output", x)

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
    
    def forward(self, x, document_mask=None, inference=False):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class TransformerWithEmbeddingHead(MatformerModule):
    
    # Stable parameter names
    embed_tokens: "param_name:embed_tokens"
    embed_positions: "param_name:embed_positions"
    blocks: "param_name:blocks"
    
    def __init__(self, config: ModelConfig, cache=None):
        super().__init__()
        self.config = config
        self.cache = ensure_cache_and_registry(cache)
        
        # Token embeddings
        self.embed_tokens = ModuleWrapper(
            self.cache.registry.create(
                "embedding", "embedding",
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                padding_idx=config.pad_token_id
            )
        )
        
        # Learnable positional embeddings (if configured)
        self.embed_positions = (
            ModuleWrapper(
                self.cache.registry.create(
                    "embedding", "embedding",
                    num_embeddings=config.max_position_embeddings,
                    embedding_dim=config.hidden_size
                )
            )
            if  "learnable" in config.default_layer['positional_encoding']
            else None
        )
        
        # Core transformer
        self.blocks = NakedTransformer(config, cache=self.cache)
    
    def forward(self, x, **kwargs):
        """Embed tokens (+ positions) then process through transformer."""
        embeddings = self.embed_tokens(x)
        
        # Add learnable positional embeddings if configured
        if self.embed_positions is not None:
            batch_size, seq_len = x.tensor.shape
            position_ids = torch.arange(seq_len, dtype=torch.long, device=x.tensor.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            position_ids_wrapped = replace(x, tensor=position_ids)
            position_embeds = self.embed_positions(position_ids_wrapped)
            embeddings = replace(embeddings, tensor=embeddings.tensor + position_embeds.tensor)
        
        return self.blocks(embeddings, **kwargs)


class TransformerWithLMHead(MatformerModule):
    """
    Complete language model with embeddings and language modeling head.
    Suitable for autoregressive (GPT-like) or masked (BERT-like) language models.
    Supports weight tying between embeddings and output projection.
    """
    
    # Stable parameter names
    lm_head: "param_name:lm_head"
    encoder: "param_name:encoder"
    
    def __init__(self, config: ModelConfig, tokenizer=None, device=None, cache=None):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.cache = ensure_cache_and_registry(cache)
        
        # Language modeling head (output projection)
        self.lm_head = ModuleWrapper(
            self.cache.registry.create(
                "linear", "linear",
                in_features=config.hidden_size,
                out_features=config.vocab_size
            )
        )
        
        # Transformer with embeddings
        self.encoder = TransformerWithEmbeddingHead(config, cache=self.cache)
        
        # Weight tying: share embeddings with output projection
        if config.tie_word_embeddings:
            self.lm_head.weight = self.encoder.embed_tokens.module.inner.weight
    
    def forward(self, x, return_type='logits', **kwargs):
        """Forward pass with optional return type.
        
        Args:
            x: Input token IDs
            return_type: 'logits' for language modeling head output,
                        'hidden' for transformer hidden states
        """
        hidden_states = self.encoder(x, **kwargs)
        
        if return_type == 'logits':
            return self.lm_head(hidden_states)
        else:
            return hidden_states
    def collate_generation_inputs(batch, pad_token_id, max_seq_len=None, varlen_strategy='unpadding'):
        if batch is None or len(batch) == 0:
            warn("Empty batch received in collate_generation_inputs")
            return None
        
        sequences = []
        for item in batch:
            if isinstance(item, dict):
                seq = item.get("input_ids", item.get("object", item))
            elif isinstance(item, torch.Tensor):
                seq = item.tolist() if item.dim() > 0 else [item.item()]
            elif isinstance(item, list):
                seq = item
            else:
                seq = [item]
            sequences.append(seq)
        
        if varlen_strategy == 'nested':
            return torch.nested.nested_tensor([torch.tensor(s) for s in sequences], layout=torch.jagged)
        
        if max_seq_len is None:
            max_seq_len = max(len(s) for s in sequences)
        
        padded_ids = []
        for seq in sequences:
            if len(seq) < max_seq_len:
                padded_ids.append(seq + [pad_token_id] * (max_seq_len - len(seq)))
            else:
                padded_ids.append(seq[:max_seq_len])
        
        tensors = torch.tensor(padded_ids, dtype=torch.long)
        padding_masks = (tensors == pad_token_id)
        
        sequence = PaddedTensor(tensor=tensors, padding_mask=padding_masks)
        
        if varlen_strategy == 'unpadding':
            sequence = sequence.unpad()
        
        return sequence

class TransformerWithClassificationHead(TransformerWithEmbeddingHead):
    def __init__(self, config: ModelConfig, tokenizer=None, pooling_type='cls', num_features=2, device=None, cache=None):
        super().__init__(config)
        self.cache = ensure_cache_and_registry(cache)   
        cache=self.cache      
        self.classifier_dropout_p = getattr(config, "classifier_dropout_p", 0.1)             
        self.classifier_dropout_inplace = getattr(config, "classifier_dropout_inplace", False)             
        self.classification_head = ModuleWrapper(self.cache.registry.create(
            "linear", "linear", in_features=config.hidden_size, out_features=num_features
        ))

        self.dropout = ModuleWrapper(self.cache.registry.create(
                "dropout", "dropout",
                p=self.classifier_dropout_p, inplace=self.classifier_dropout_inplace
            )
        )
        # Transformer with embeddings
        self.encoder = TransformerWithEmbeddingHead(config, cache=self.cache)
        
        # Weight tying: share embeddings with output projection
        if config.tie_word_embeddings:
            self.lm_head.weight = self.encoder.embed_tokens.module.inner.weight
   
        self.config = config
        self.tokenizer = tokenizer
        self.pooling_type = pooling_type
        self.num_features = num_features
    
    def change_num_labels(self, new_num_labels):
        """Change the number of output features of the classification head."""
        self.num_labels = new_num_labels
        self.classification_head = ModuleWrapper(self.cache.registry.create(
            "linear", "linear", in_features=self.config.hidden_size, out_features=new_num_labels
        ))

        # Get device and dtype from existing parameters
        reference_param = next(self.encoder.parameters())
        device = reference_param.device
        dtype = reference_param.dtype
        
        # Move to the same device and dtype as the model
        self.classification_head = self.classification_head.to(device=device, dtype=dtype)

        print(f"Number of labels changed to {new_num_labels}")

    def forward(self, x, attention_mask=None, **kwargs):
        hidden_states = self.encoder(x, **kwargs) # (B,S,D)

        if self.pooling_type == 'cls':
            # [CLS] in pos. 0
            pooled_output = hidden_states.tensor[:, 0, :]
        elif self.pooling_type == 'mean':
            #TODO: check if the mask works
            if attention_mask is None:
                pooled_output = hidden_states.tensor.mean(dim=1)
            else:
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.tensor.size()).to(hidden_states.dtype)
                sum_hidden = torch.sum(hidden_states.tensor * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled_output = sum_hidden / sum_mask
        else:
            raise ValueError(f"{self.pooling_type} not in 'cls','mean'")

        pooled_output = self.dropout(pooled_output)
        logits = self.classification_head(pooled_output)
        return logits

class TransformerWithTokenClassificationHead(TransformerWithEmbeddingHead):
    def __init__(self, config: ModelConfig, tokenizer=None, num_labels=2, device=None, cache=None):
        super().__init__(config)
        
        self.cache = ensure_cache_and_registry(cache)   
        cache=self.cache      
        self.classifier_dropout_p = getattr(config, "classifier_dropout_p", 0.1)             
        self.classifier_dropout_inplace = getattr(config, "classifier_dropout_inplace", False)             
        self.classification_head = ModuleWrapper(self.cache.registry.create(
            "linear", "linear", in_features=config.hidden_size, out_features=num_labels
        ))

        self.dropout = ModuleWrapper(self.cache.registry.create(
                "dropout", "dropout",
                p=self.classifier_dropout_p, inplace=self.classifier_dropout_inplace
            )
        )
        # Transformer with embeddings
        self.encoder = TransformerWithEmbeddingHead(config, cache=self.cache)
        
        # Weight tying: share embeddings with output projection
        if config.tie_word_embeddings:
            self.lm_head.weight = self.encoder.embed_tokens.module.inner.weight
   
        self.config = config
        self.tokenizer = tokenizer
        self.num_labels = num_labels
    
    def change_num_labels(self, new_num_labels):
        """Change the number of output features of the classification head."""
        self.num_labels = new_num_labels
        self.classification_head = ModuleWrapper(self.cache.registry.create(
            "linear", "linear", in_features=self.config.hidden_size, out_features=new_num_labels
        ))

        # Get device and dtype from existing parameters
        reference_param = next(self.encoder.parameters())
        device = reference_param.device
        dtype = reference_param.dtype
        
        # Move to the same device and dtype as the model
        self.classification_head = self.classification_head.to(device=device, dtype=dtype)

        print(f"Number of labels changed to {new_num_labels}")

    def forward(self, x, attention_mask=None, **kwargs):
        hidden_states = self.encoder(x, **kwargs).tensor # (B,S,D)

        hidden_states = self.dropout(hidden_states)
        logits = self.classification_head(hidden_states)
        return logits
        
        
class TextDiffusionModel:
    """
    The definition of a model similar to LLada
    """
    def generate(
        self,
        prompt=None,
        input_ids=None,
        num_steps=50,
        max_length=100,
        min_length=0,
        output_scores=False,
        return_type='text'):
        """
        1. Mask the input prompt at 100%
        2. 
        """
        raise NotImplementedError
        
        
class BERTModel(TransformerWithLMHead):
    
    def init_training(self,config: ModelConfig,tokenizer=None,device=None,cache=None):
        # Training a BERT-Like model
        self.config=config
        self.cache = ensure_cache_and_registry(cache) 
        super().__init__()
        
        # 1. Init the maskerator
        self.init_maskerator()
        # 2. Define the loss function
        if self.has_fused_loss:
            pass
        else:
            pass
   
    def init_maskerator(self, masking_ratio=None):
        from matformer.masked_models import Maskerator
        # Maskerator setup
        if not hasattr(self,'config'): # The bert model was not istantiated with the config, masking ratio parameter has to be passed as argument (case for inference)
             assert masking_ratio is not None
             self.masking_ratio=masking_ratio
             self.maskerator=Maskerator(mask_token=self.config.mask_token_id,substitution_rate=masking_ratio)
             print(f"Masking ratio: {self.masking_ratio}")
        else: # We get the maskerator settings from the config
            try:
                self.masking_ratio=self.config.masked_substitution_rate
            except:
                self.masking_ratio=0.15
            try:
                try:
                    cloze_prob = self.config.get("cloze_prob", 1.0)
                    random_prob = self.config.get("random_prob", None)
                    same_prob = self.config.get("same_prob", None)
                    vocab_size = self.config.get("vocab_size", None)
                except:
                    print("Maskerator fallback.")
                    cloze_prob=1.0
                    random_prob=0.0
                    same_prob=0.0
                    try:
                        vocab_size=self.config.vocab_size
                    except:
                        vocab_size=None
                
                self.maskerator=Maskerator(mask_token=self.config.mask_token_id,
                                           substitution_rate=self.config.masked_substitution_rate,
                                           pad_token_id=self.config.pad_token_id,
                                           cloze_prob=cloze_prob,
                                           random_prob=random_prob,
                                           same_prob=same_prob,
                                           vocab_size=vocab_size)
            except:
                print("Maskerator not set up. Fine for Autoregressive model")     


    def init_classification_head(self, num_labels=2, pooling_type='cls',cache=None):
        """Initialize classification head. Compatible with HuggingFace Trainer."""
        self.cache = ensure_cache_and_registry(cache)                               
        self.classification_head = ModuleWrapper(self.cache.registry.create(
            "linear", "linear", in_features=config.hidden_size, out_features=num_labels
        ))
        self.pooling_type = pooling_type
        self.num_labels = num_labels
        
    def forward_classification(self, x, attention_mask=None):
        """Forward pass for sequence classification."""
        hidden_states = self.encoder(x)
        
        if self.pooling_type == 'cls':
            pooled = hidden_states.tensor[:, 0]
        elif self.pooling_type == 'mean':
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).expand(hidden_states.tensor.size()).float()
                pooled = torch.sum(hidden_states.tensor * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
            else:
                pooled = hidden_states.tensor.mean(dim=1)
        else:
            raise ValueError(f"pooling_type must be 'cls' or 'mean', got {self.pooling_type}")
            
        return self.classification_head(pooled).tensor        
    def inference_testing(self, input_text=None, masking_ratio=0.25,datatype=torch.bfloat16, tokens=None):
        #assert (is input_text or is_tokens)
        if not hasattr(self,'maskerator') or masking_ratio!=self.masking_ratio:
            self.init_maskerator(masking_ratio)
        if not tokens:
            sequence = self.tokenizer.encode(input_text)
        else:
            sequence=tokens
        sequence = torch.tensor(sequence).to(self.device)
        masked_list, cloze_list = self.maskerator(sequence)
        masked_list.to(self.device)
        masked_sequence = NormalTensor(tensor=masked_list.unsqueeze(0))
        model_input=deepcopy(masked_sequence)
        with torch.no_grad():
            logits = self(model_input)
        predictions = torch.argmax(logits.tensor, dim=-1)
        targets = sequence.squeeze()
        mask = cloze_list.squeeze().bool()
        correct = (predictions.squeeze()[mask] == targets[mask]).sum().item()
        total = mask.sum().item()
        accuracy = correct / total if total > 0 else 0.0
        out_tokens=list()
        for i,token in enumerate(masked_sequence.tensor.squeeze().tolist()):
            if token != self.config.mask_token_id:
                out_tokens.append(self.tokenizer.decode(token))
            else:
                out_tokens.append(f"[ {self.tokenizer.decode(predictions.squeeze()[i])} ]")
        
        return accuracy, out_tokens

class Autoregressive_Model(TransformerWithLMHead):
    def generate(
        self, 
        prompt=None, 
        input_ids=None, 
        max_length=100, 
        min_length=0,
        temperature=1.0, 
        top_k=0, 
        top_p=0.9,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        num_return_sequences=1,
        pad_token_id=None,
        eos_token_id=None,
        bos_token_id=None,
        stopping_criteria=None,
        output_scores=False,
        return_type='text'
    ):
        self.eval()
        
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        if bos_token_id is None:
            bos_token_id = self.config.bos_token_id
        
        if input_ids is None:
            if prompt is None:
                current_ids = torch.tensor([[bos_token_id]], device=self.device).repeat(num_return_sequences, 1)
            else:
                assert isinstance(prompt, str), "Prompt expected as string"
                tokenizer = self.tokenizer
                prompt_ids = tokenizer.encode(text=prompt, add_bos=True, add_eos=False, add_special_tokens=False)
                current_ids = torch.tensor([prompt_ids], device=self.device).repeat(num_return_sequences, 1)
        else:
            if isinstance(input_ids, NormalTensor):
                current_ids = input_ids.tensor
            elif isinstance(input_ids, torch.Tensor):
                current_ids = input_ids
                if current_ids.dim() == 1:
                    current_ids = current_ids.unsqueeze(0)
            elif isinstance(input_ids, list):
                current_ids = torch.tensor([input_ids], device=self.device)
            else:
                raise TypeError(f"input_ids must be NormalTensor, torch.Tensor, or list, got {type(input_ids)}")
            
            if num_return_sequences > 1 and current_ids.shape[0] == 1:
                current_ids = current_ids.repeat(num_return_sequences, 1)
        
        current_ids = current_ids.to(self.device)
        batch_size = current_ids.shape[0]
        
        all_scores = [] if output_scores else None
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        for step in range(max_length):
            with torch.no_grad():
                outputs = self(NormalTensor(tensor=current_ids)).tensor
            
            next_token_logits = outputs[:, -1, :]
            
            if output_scores:
                all_scores.append(next_token_logits.clone())
            
            if repetition_penalty != 1.0:
                for batch_idx in range(batch_size):
                    for token_id in set(current_ids[batch_idx].tolist()):
                        if next_token_logits[batch_idx, token_id] < 0:
                            next_token_logits[batch_idx, token_id] *= repetition_penalty
                        else:
                            next_token_logits[batch_idx, token_id] /= repetition_penalty
            
            if no_repeat_ngram_size > 0 and current_ids.shape[1] >= no_repeat_ngram_size:
                for batch_idx in range(batch_size):
                    ngrams = {}
                    seq = current_ids[batch_idx].tolist()
                    for i in range(len(seq) - no_repeat_ngram_size + 1):
                        ngram = tuple(seq[i:i + no_repeat_ngram_size - 1])
                        next_token = seq[i + no_repeat_ngram_size - 1]
                        if ngram not in ngrams:
                            ngrams[ngram] = []
                        ngrams[ngram].append(next_token)
                    
                    if len(seq) >= no_repeat_ngram_size - 1:
                        current_ngram = tuple(seq[-(no_repeat_ngram_size - 1):])
                        if current_ngram in ngrams:
                            for banned_token in ngrams[current_ngram]:
                                next_token_logits[batch_idx, banned_token] = float('-inf')
            
            if current_ids.shape[1] < min_length:
                next_token_logits[:, eos_token_id] = float('-inf')
            
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            if top_k > 0:
                for batch_idx in range(batch_size):
                    top_k_values, top_k_indices = torch.topk(next_token_logits[batch_idx], min(top_k, next_token_logits.shape[-1]), dim=-1)
                    next_token_logits[batch_idx] = torch.full_like(next_token_logits[batch_idx], float('-inf'))
                    next_token_logits[batch_idx].scatter_(0, top_k_indices, top_k_values)
            
            if top_p < 1.0:
                for batch_idx in range(batch_size):
                    sorted_logits, sorted_indices = torch.sort(next_token_logits[batch_idx], descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[batch_idx, indices_to_remove] = float('-inf')
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            
            next_tokens = next_tokens.masked_fill(finished.unsqueeze(1), pad_token_id)
            current_ids = torch.cat([current_ids, next_tokens], dim=1)
            
            finished = finished | (next_tokens.squeeze(1) == eos_token_id)
            
            if stopping_criteria is not None:
                if stopping_criteria(current_ids, None):
                    break
            
            if finished.all():
                break
        
        if return_type == 'text':
            if batch_size == 1:
                return self.tokenizer.decode(current_ids[0].tolist())
            else:
                return [self.tokenizer.decode(seq.tolist()) for seq in current_ids]
        elif return_type == 'pt':
            if output_scores:
                return current_ids, torch.stack(all_scores, dim=1)
            return current_ids
        else:
            if output_scores:
                return [seq.tolist() for seq in current_ids], all_scores
            return [seq.tolist() for seq in current_ids]
class EntropyModel(Autoregressive_Model):
    def compute_entropy(self, prompts):
        """
        Return a tensor of size sequence length containing the entropy for each received text
        """
        self.eval()
        if isinstance(prompts, str):
            prompts = [prompts]

        prompt_ids = self.tokenizer.batch_encode(prompts)  # [B, seq_len]
        prompt_ids = prompt_ids.to(self.device)

        with torch.no_grad():
            logits = self(prompt_ids)  # [B, seq_len, vocab_size]

        epsilon = 1e-10
        probs = torch.nn.functional.softmax(logits.tensor, dim=-1)
        logprobs = torch.log(probs + epsilon)
        entropy = -torch.sum(probs * logprobs, dim=-1)  # [B, seq_len]

        return entropy[:, 1:] 
    def monotonicity_breakpoints(self, prompt=None, entropy=None, smoothing=None):
        # Da cambiare: il taglio delle patch provenienti dal text encoder potrebbe effettuarsi qui, sfruttando i nested tensor
        # Al momento l'implementazione che prevede il ciclo sui batch nel text encoder funziona ma è esageratamente inefficiente a livello di tempi di calcolo
        if smoothing is None:
            smoothing = 0
            print("WARNING: You are running the entropy model without a smoothing set.")
        if prompt is not None:
            entropy = self.compute_entropy(prompt)  # Expected shape: [B, seq_len]
        elif entropy is None:
            raise ValueError("Either provide `prompt` or `entropy`.")
        if entropy.dim() == 1:
            entropy = entropy.unsqueeze(0)  # Make it batched [1, seq_len]
        B, seq_len = entropy.shape
        cutting_masks = torch.zeros_like(entropy, device='cpu')
        group_masks = torch.zeros_like(entropy, dtype=torch.long, device='cpu')
        cutting_points_all = []
        
        # VECTORIZED VERSION - replaces the batch loop
        # Move entropy to CPU once for all operations
        ent_cpu = entropy.cpu()
        
        # Create padding for comparison: add infinity at the beginning of each sequence
        # This ensures the first element is never considered a breaking point
        inf_pad = torch.full((B, 1), float('inf'), device='cpu')
        padded_entropy = torch.cat([inf_pad, ent_cpu], dim=1)  # Shape: [B, seq_len+1]
        
        # Vectorized monotonicity violation detection
        # Compare each position with the previous one across all batches simultaneously
        # breaking_points[b, i] = True if entropy[b, i] > entropy[b, i-1] + smoothing
        breaking_points = ent_cpu > padded_entropy[:, :-1] + smoothing  # Shape: [B, seq_len]
        
        # Convert breaking points to cutting masks (same logic as before, but vectorized)
        cutting_masks_vectorized = breaking_points.float()
        
        # Generate group masks using cumulative sum (vectorized version of the previous logic)
        group_masks_vectorized = torch.cumsum(breaking_points.long(), dim=1)
        
        # Extract cutting points for each batch item
        # This part still requires some iteration, but only over breaking points, not all positions
        cutting_points_all_vectorized = []
        for b in range(B):
            # Find positions where breaking points occur
            break_positions = torch.where(breaking_points[b])[0]
            cutting_points = []
            
            if len(break_positions) > 0:
                start_point = 0
                for pos in break_positions:
                    cutting_points.append((start_point, pos.item()))  # Note: removed +1 here
                    start_point = pos.item()
                
                # Add the final segment from the last breaking point to the end
                if start_point < seq_len:
                    cutting_points.append((start_point, seq_len))
            else:
                # If no breaking points, the entire sequence is one chunk
                cutting_points.append((0, seq_len))
            
            cutting_points_all_vectorized.append(cutting_points)
        # Update the output tensors with vectorized results
        cutting_masks[:] = cutting_masks_vectorized
        group_masks[:] = group_masks_vectorized
        cutting_points_all = cutting_points_all_vectorized
        
        #print("------------- ENTROPY DEBUG ------------------")
        #print("----CUTTING MASK:")
        #print(cutting_masks)
        #print("----- GROUP MASKs")
        #print(group_masks)
        #print("CUTTING POINTS")
        #print(cutting_points_all)
        return cutting_points_all, cutting_masks, group_masks
    def old_monotonicity_breakpoints(self, prompt=None, entropy=None, smoothing=None):
        # Da cambiare: il taglio delle patch provenienti dal text encoder potrebbe effettuarsi qui, sfruttando i nested tensor
        # Al momento l'implementazione che prevede il ciclo sui batch nel text encoder funziona ma è esageratamente inefficiente a livello di tempi di calcolo
        if smoothing is None:
            smoothing = 0
            print("WARNING: You are running the entropy model without a smoothing set.")

        if prompt is not None:
            #start=datetime.now()
            entropy = self.compute_entropy(prompt)  # Expected shape: [B, seq_len]
            #end=datetime.now()
            #print(f"Esecuzione del modello {end-start}")
        elif entropy is None:
            raise ValueError("Either provide `prompt` or `entropy`.")
        #start=datetime.now()

        if entropy.dim() == 1:
            entropy = entropy.unsqueeze(0)  # Make it batched [1, seq_len]

        B, seq_len = entropy.shape
        cutting_masks = torch.zeros_like(entropy, device='cpu')
        group_masks = torch.zeros_like(entropy, dtype=torch.long, device='cpu')
        cutting_points_all = []
        #end=datetime.now()
        #print(f"Creazione delle robette {end-start}")  
        #start=datetime.now()    
        for b in range(B):
            ent = entropy[b].cpu()
            cutting_points = []
            cutting_mask = torch.zeros(seq_len, device='cpu')
            prev_entr = float('inf')
            start_point = 0
            for i in range(seq_len):
                if ent[i] > prev_entr + smoothing:
                    cutting_points.append((start_point, i+1))
                    cutting_mask[i] = 1
                    start_point = i+1
                prev_entr = ent[i]
            group_mask = torch.cumsum(cutting_mask, dim=0)
            cutting_masks[b] = cutting_mask
            group_masks[b] = group_mask
            cutting_points_all.append(cutting_points)
        #print("------------- ENTROPY DEBUG ------------------")
        #print("----CUTTING MASK:")
        #print(cutting_masks)
        #print("----- GROUP MASKs")
        #print(group_masks)
        #print("CUTTING POINTS")
        #print(cutting_points_all)
        #end=datetime.now()
        #print(f"Ciclo sui batch: {end-start}")
        return cutting_points_all, cutting_masks, group_masks

    def old_cut_text(self,text,cutting_points=None,smoothing=None):
        """
        Cut a text according to cutting points
        """
        if cutting_points is None:
            cutting_points,_=self.monotonicity_breakpoints(prompt=text,smoothing=smoothing)
        text_chunks=[text[i:j] for i,j in cutting_points]
        return text_chunks
    def cut_text(self, text, cutting_points=None, smoothing=None, hard_limit=None):
            """
            Cut a text or batch of texts according to cutting points.
            
            Args:
                text: str or list of str - single text or batch of texts
                cutting_points: list of tuples or list of list of tuples - cutting points for each text
                smoothing: float - smoothing parameter if cutting_points is None
                hard_limit: int - maximum character length for chunks, splits if exceeded
                
            Returns:
                list of str chunks (if single text) or list of list of str chunks (if batch)
            """
            # Handle single text input (backwards compatibility)
            hard_limit_violations=0
            if isinstance(text, str):
                if cutting_points is None:
                    cutting_points, _, _ = self.monotonicity_breakpoints(prompt=text, smoothing=smoothing)
                # Avoid error if cutting points is empty (ex. for a single word)
                if len(cutting_points)==0:
                    cutting_points=[0,len(text)] 
                # cutting_points is a list of lists, take the first (and only) one
                elif isinstance(cutting_points[0], list):
                    cutting_points = cutting_points[0]
                    
                text_chunks = [text[i:j] for i, j in cutting_points]
                # Apply hard limit splitting
                if hard_limit is not None:
                    final_chunks = []
                    for chunk in text_chunks:
                        if len(chunk) <= hard_limit:
                            final_chunks.append(chunk)
                        else:
                            final_chunks.extend([chunk[k:k+hard_limit] for k in range(0, len(chunk), hard_limit)])
                            hard_limit_violations+=1
                    text_chunks = final_chunks
                return text_chunks
            
            # Handle batch input
            if isinstance(text, list):
                if cutting_points is None:
                    cutting_points, _, _ = self.monotonicity_breakpoints(prompt=text, smoothing=smoothing)
                
                # Vectorized batch processing - process all texts simultaneously
                all_chunks = []
                for text_item, cutting_item in zip(text, cutting_points):
                    text_chunks = [text_item[i:j] for i, j in cutting_item]
                    # Apply hard limit splitting
                    if hard_limit is not None:
                        final_chunks = []
                        for chunk in text_chunks:
                            if len(chunk) <= hard_limit:
                                final_chunks.append(chunk)
                            else:
                                final_chunks.extend([chunk[k:k+hard_limit] for k in range(0, len(chunk), hard_limit)])
                                hard_limit_violations+=1
                        text_chunks = final_chunks
                    all_chunks.append(text_chunks)
                
                return all_chunks,hard_limit_violations
            
            else:
                raise ValueError("text must be either a string or a list of strings")
class TransformerWithCharAutoencoder(MatformerModule):
    def __init__(self, config, device=None, tokenizer=None, log=None):
        super().__init__()
        from char_autoencoder.autoencoders import TransCharAutoencoder_Encoder, TransCharAutoencoder_Decoder
        self.log=log
        self.encoder = TransCharAutoencoder_Encoder(config=config.encoder)
        self.decoder = TransCharAutoencoder_Decoder(config=config.decoder)
        self.transformer = NakedTransformer(config)
        self.projection_in = ModuleWrapper(nn.Linear(config.encoder.hidden_size, config.hidden_size))
        self.projection_out = ModuleWrapper(nn.Linear(config.hidden_size, config.decoder.hidden_size))
        self.config = config
        
        # Training phase state
        self.training_phase = 'autoencoder-training'

    def forward(self, x, skip_transformer=False, skip_decoder=False):
        """Phase 1: Train autoencoder only (bypass transformer)
        Input could be [B,S] or [B,P,S]
        If [B,S], we expect a PaddedTensor or a NormalTensor (for inference), "padding" refers to "internal" padding (per each sequence)
        If [B,P,S], we expect a PaddedTensor or an Unpadded Tensor (or NormalTensor for inference), where "padding" refers to "external" padding (per each batch). Thus, to reconstruct the internal padding, we need the internal padding mask
        
        If input is of shape B,P,S, flatten the patch with the batch => B,P,S => B*P,S but skip computation on "external" padding
        """
        """
        I expect as input a dictionary composed of
        {
        "tensor": see comment above
        "padding_masks_external": True if the entire patch is to be padded, False otherwise [B,P,S]
        "padding_masks_internal": Contains information about padding of each sequence inside patches (ready to be flattened) [B,P,S]
        }
        """
        tensor = x["tensor"]  # [B,P,S]
        padding_masks_external = x["padding_masks_external"]  # [B,P,S]
        padding_masks_internal = x["padding_masks_internal"]  # [B,P,S]
        
        # Step 1: Encoder - Filter valid patches
        valid_patch_mask = ~padding_masks_external.all(dim=-1)  # [B,P]
        valid_tensor = tensor[valid_patch_mask]  # [B*P_valid, S]
        valid_internal_masks = padding_masks_internal[valid_patch_mask]  # [B*P_valid, S]
        
        padded_tensor = PaddedTensor(
            tensor=valid_tensor,
            padding_mask=valid_internal_masks
        )
        z = self.encoder(padded_tensor)  # [B*P_valid, S] -> [B*P_valid, D]
        
        if not skip_transformer:
            # Step 2: Reshape for transformer
            B, P = valid_patch_mask.shape
            D = z.tensor.size(-1)
            z_full = torch.zeros(B, D, device=z.device, dtype=z.dtype)
            z_full[valid_patch_mask] = z.tensor  # [B, P]
            
            external_padding_mask = ~valid_patch_mask  # [B, P]
            z_padded = PaddedTensor(tensor=z_full, padding_mask=external_padding_mask)
            
            # Step 3: Transformer
            z = self.projection_in(z_padded)
            z = self.transformer(z)  # [B, P, D]
            z = self.projection_out(z)
            
            # Step 4: Back to decoder format
            z_valid = z.tensor[valid_patch_mask]  # [B*P_valid, D]
            z_valid_wrapped = replace(z, tensor=z_valid)
        else:
            # Skip transformer - encoder directly connected to decoder
            z_valid_wrapped = z
        
        # Early return for encoder-only phase
        if skip_decoder:
            return z_valid_wrapped  # [B*P_valid, D]
        
        # Step 5: Decoder
        logits, _ = self.decoder(z_valid_wrapped)  # [B*P_valid, D] -> [B*P_valid, S, vocab_size]
        
        # Step 6: Restore final output
        B, P = valid_patch_mask.shape
        S = tensor.size(-1)
        vocab_size = logits.tensor.size(-1)
        output_logits = torch.zeros(B, P, S, vocab_size, device=tensor.device, dtype=logits.tensor.dtype)
        output_logits[valid_patch_mask] = logits.tensor
        
        return output_logits

    def set_training_phase(self, phase: str):
        """Set training phase and configure module gradients"""
        self.training_phase = phase
        
        phase_configs = {
            'autoencoder-training': {'encoder': True, 'transformer': False, 'decoder': True},
            'patch-training': {'encoder': True, 'transformer': True, 'decoder': False},
            'autoencoder-final-annealing': {'encoder': False, 'transformer': True, 'decoder': True}
        }
        
        if phase not in phase_configs:
            raise ValueError(f"Unknown phase: {phase}")
        
        config = phase_configs[phase]
        self.encoder.requires_grad_(config['encoder'])
        self.transformer.requires_grad_(config['transformer'])
        self.decoder.requires_grad_(config['decoder'])

    def training_step(self, batch, phase=None, log=None):
        """Simplified training step dispatcher"""
        phase = phase or self.training_phase
        
        if phase == 'autoencoder-training':
            return self._autoencoder_loss(batch, log=log)
        elif phase == 'patch-training':
            return self._patch_loss(batch)
        elif phase == 'autoencoder-final-annealing':
            return self._fine_tuning_loss(batch)
        else:
            raise ValueError(f"Unknown training phase: {phase}")

    def _autoencoder_loss(self, batch, log):
        """Phase 1: Autoencoder reconstruction loss"""
        x = batch['input_ids'] 
        
        # Forward with autoencoder only (skip transformer)
        logits = self.forward(x, skip_transformer=True)  # [B, P, S, vocab_size]
        
        # Get target tensor
        targets = x["tensor"]  # [B, P, S]
        valid_mask = ~x["padding_masks_external"].all(dim=-1)  # [B, P]
        
        # Compute loss only on valid patches
        loss = 0
        count = 0
        for b in range(targets.size(0)):
            for p in range(targets.size(1)):
                if valid_mask[b, p]:
                    patch_logits = logits[b, p]  # [S, vocab_size]
                    patch_targets = targets[b, p]  # [S]
                    
                    # Mask internal padding
                    internal_mask = ~x["padding_masks_internal"][b, p]
                    if internal_mask.any():
                        valid_logits = patch_logits[internal_mask]
                        valid_targets = patch_targets[internal_mask]
                        loss += F.cross_entropy(valid_logits, valid_targets)
                        count += 1
        loss=loss / max(count, 1)
        log('train/autoencoder_loss', loss, prog_bar=True)
        return loss

    def _patch_loss(self, batch):
        """Phase 2: Patch-level autoregressive prediction"""
        x = batch['input_ids']
        
        # Get patch embeddings (skip decoder)
        z = self.forward(x, skip_decoder=True)  # [B*P_valid, D]
        
        # Reshape to get batch structure back
        valid_mask = ~x["padding_masks_external"].all(dim=-1)  # [B, P]
        B, P = valid_mask.shape
        D = z.tensor.size(-1)
        
        z_full = torch.zeros(B, P, D, device=z.tensor.device, dtype=z.tensor.dtype)
        z_full[valid_mask] = z.tensor
        
        # Autoregressive loss: predict next patch from previous patches
        if P <= 1:
            return torch.tensor(0.0, device=z.tensor.device, requires_grad=True)
        
        # Input: patches 0 to P-2, Target: patches 1 to P-1
        input_patches = z_full[:, :-1]  # [B, P-1, D]
        target_patches = z_full[:, 1:]  # [B, P-1, D]
        
        # Only compute loss where both input and target are valid
        input_valid = valid_mask[:, :-1]
        target_valid = valid_mask[:, 1:]
        both_valid = input_valid & target_valid
        
        if both_valid.any():
            return F.mse_loss(input_patches[both_valid], target_patches[both_valid])
        else:
            return torch.tensor(0.0, device=z.tensor.device, requires_grad=True)

    def _fine_tuning_loss(self, batch):
        """Phase 3"""
        x = batch['input_ids']
        
        # Full forward pass
        logits = self.forward(x)  # [B, P, S, vocab_size]
        targets = x["tensor"]  # [B, P, S]
        
        # Compute loss on all valid positions
        loss = 0
        count = 0
        
        valid_patch_mask = ~x["padding_masks_external"].all(dim=-1)
        
        for b in range(targets.size(0)):
            for p in range(targets.size(1)):
                if valid_patch_mask[b, p]:
                    patch_logits = logits[b, p]  # [S, vocab_size]
                    patch_targets = targets[b, p]  # [S]
                    
                    # Use internal padding mask
                    internal_mask = ~x["padding_masks_internal"][b, p]
                    if internal_mask.any():
                        valid_logits = patch_logits[internal_mask]
                        valid_targets = patch_targets[internal_mask]
                        loss += F.cross_entropy(valid_logits, valid_targets)
                        count += 1
        
        return loss / max(count, 1)
