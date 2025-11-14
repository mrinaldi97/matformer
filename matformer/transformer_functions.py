"""
File transformer_functions.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Literal
from matformer.model_config import ModelConfig
from matformer.tensors_dataclasses import TensorDC, NormalTensor, PaddedTensor, UnpaddedTensor
from dataclasses import replace
from torch.nn.functional import scaled_dot_product_attention
from matformer.cached_stuff import CachedStuff
from matformer.matformer_registry import registry
from warnings import warn


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        q_dim: int,               
        k_dim: Optional[int] = None,
        v_dim: Optional[int] = None,
        is_cross_attention: bool = False,
        nheads: int = 8,
        bias: bool = False,
        positional_encoding: str = 'rope',  # 'alibi', 'rope', 'nope', 'sinusoidal', 'learnable'
        #dropout: float = 0.0, # Not supported by FlexAttention yet
        cache: Optional['CachedStuff'] = None,
        attn_variant: str = 'normal',  # 'normal', 'wersa', 'linear', etc.
        attn_impl: str = 'flash',  # Preferred implementation within variant
        is_causal: bool = True,
        sliding_window: Optional[int] = None,
        device: str = 'cuda'  
    ):
        super().__init__()

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
        if self.positional_encoding == 'rope':
            self.rotary_emb = self.cache.get_rotary_emb(self.head_dim)
        
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
                        
        # Packed qkv projection for efficiency  
        if not is_cross_attention or self.qkv_samedim:
            self.packed_proj = nn.Linear(self.q_dim, 3 * q_dim, bias=bias)
        else:
            self.q_proj = nn.Linear(q_dim, q_dim, bias=bias)
            self.k_proj = nn.Linear(k_dim, q_dim, bias=bias)
            self.v_proj = nn.Linear(v_dim, q_dim, bias=bias)
            
        self.out_proj = nn.Linear(q_dim, self.q_dim, bias=bias)  # Out projection

    def _transpose_for_kernel(self, tensor, current_order, target_order):
        """
        Helper to transpose tensors based on metadata.
        Supports: BHSD <-> BSHD conversions
        """
        if current_order == target_order:
            return tensor
        
        if current_order == 'BHSD' and target_order == 'BSHD':
            return tensor.transpose(1, 2)
        elif current_order == 'BSHD' and target_order == 'BHSD':
            return tensor.transpose(1, 2)
        else:
            raise ValueError(f"Unsupported tensor order conversion: {current_order} -> {target_order}")

    def forward(self, query_input, original_x=None, key_input=None, value_input=None, document_mask=None):
        """
        Input Tensors:
        query_input: (Batch, Seqlen, Qdim)
        key_input: (Batch, Seqlen, Kdim) [If omitted, self-attention]
        value_input: (Batch, Seqlen, Vdim) [If omitted, self-attention]
        Output:
        (Batch, Seqlen, Qdim) # Qdim is the query dimension as well as the output.
        """
        
        supports_unpadding = self.kernel_meta.get('supports_unpadding', False)
        supports_packed_qkv = self.kernel_meta.get('supports_packed_qkv', False)
        
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
                query_input = query_input.pad()
                if self.is_cross_attention:
                    key_input = key_input.pad()
                    value_input = value_input.pad()

        # Extract tensors (handle both regular tensors and Matformer's TensorDC)
        q_tensor = query_input.tensor if hasattr(query_input, 'tensor') else query_input
        k_tensor = key_input.tensor if hasattr(key_input, 'tensor') else key_input
        v_tensor = value_input.tensor if hasattr(value_input, 'tensor') else value_input
        
        # Projecting (eventually, packed projection)
        if self.qkv_samedim and not self.is_cross_attention:
            qkv_projected = self.packed_proj(q_tensor)
            q = None
            k = None
            v = None
            if not supports_packed_qkv or self.positional_encoding == 'rope': #TODO: Ricordarsi perchè non funzionava con RoPe
                q, k, v = torch.chunk(qkv_projected, 3, dim=-1)
                qkv_projected = None
        elif self.qkv_samedim and self.is_cross_attention:
            qkv_projected = None
            w = torch.chunk(self.packed_proj.weight, 3, dim=0)
            b = torch.chunk(self.packed_proj.bias, 3, dim=0) if self.bias else (None, None, None)
            q = F.linear(q_tensor, w[0], b[0])
            k = F.linear(k_tensor, w[1], b[1])
            v = F.linear(v_tensor, w[2], b[2])
        else:
            qkv_projected = None
            q = self.q_proj(q_tensor)
            k = self.k_proj(k_tensor)
            v = self.v_proj(v_tensor)
        
        # Creating the heads (B, S, D) -> (B, H, S, Hd)
        if qkv_projected is None:
            q = q.unflatten(-1, [self.nheads, self.head_dim]).transpose(1, 2)
            k = k.unflatten(-1, [self.nheads, self.head_dim]).transpose(1, 2)
            v = v.unflatten(-1, [self.nheads, self.head_dim]).transpose(1, 2)
         
        # Apply RoPe
        if self.positional_encoding == 'rope':
            qkv_projected = None
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        
        # Transpose to kernel's expected input format
        # Current format after head creation: (B, H, S, Hd)
        kernel_input_order = self.kernel_meta.get('tensor_order_input', 'BHSD')
        
        if qkv_projected is None:
            q = self._transpose_for_kernel(q, 'BHSD', kernel_input_order)
            k = self._transpose_for_kernel(k, 'BHSD', kernel_input_order)
            v = self._transpose_for_kernel(v, 'BHSD', kernel_input_order)
        
        # Attention computation
        attn_output = self.attn_kernel(
            qkv=qkv_projected, q=q, k=k, v=v,
            query_input=query_input, key_input=key_input,
            original_x=original_x
        )
        
        # Transpose from kernel's output format to expected format (B, S, H, Hd)
        kernel_output_order = self.kernel_meta.get('tensor_order_output', 'BHSD')
        attn_output = self._transpose_for_kernel(attn_output, kernel_output_order, 'BSHD')
        
        # Post-attention stuff
        attn_output = attn_output.flatten(-2)  # The flatten expects: (B, S, H, Hd)
        output_tensor = self.out_proj(attn_output)
        
        # Handle output based on input type
        if hasattr(query_input, 'tensor'):
            query_input = replace(query_input, tensor=output_tensor)
        else:
            query_input = output_tensor
            
        if repadded:
            query_input = query_input.unpad()
            
        return query_input

class PackedSwiGLUFFN(nn.Module):
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        hidden_size: Optional[int] = None,
        ffn_factor: Optional[float] = None
    ):
        super().__init__()

        if config is not None:
            hidden_size = config.hidden_size
            ffn_factor = config.ffn_factor

        # Fail hard if still None
        ffn_internal_dim = int(hidden_size * ffn_factor)

        self.w13 = nn.Linear(hidden_size, 2 * ffn_internal_dim, bias=False)
        self.w2 = nn.Linear(ffn_internal_dim, hidden_size, bias=False)

    def forward(self, _input: TensorDC) -> TensorDC:
        x1, x3 = torch.chunk(self.w13(_input.tensor), 2, -1)
        out = self.w2(F.silu(x1) * x3)
        return replace(_input, tensor=out)


class PackedGELUFFN(nn.Module):
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        hidden_size: Optional[int] = None,
        ffn_factor: Optional[float] = None
    ):
        super().__init__()

        if config is not None:
            hidden_size = config.hidden_size
            ffn_factor = config.ffn_factor

        # Fail hard if still None
        ffn_internal_dim = int(hidden_size * ffn_factor)

        self.w1 = nn.Linear(hidden_size, ffn_internal_dim, bias=False)
        self.w2 = nn.Linear(ffn_internal_dim, hidden_size, bias=False)

    def forward(self, _input: TensorDC) -> TensorDC:
        out = self.w2(F.gelu(self.w1(_input.tensor)))
        return replace(_input, tensor=out)

"""
class RMSNorm(nn.Module):
    #From https://github.com/Emericen/tiny-qwen
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size))
        self.variance_epsilon = config.rms_norm_eps
    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)
"""
