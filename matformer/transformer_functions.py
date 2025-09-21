"""
File transformer_functions.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention
from typing import Optional, List, Literal
from matformer.model_config import ModelConfig
from matformer.tensors_dataclasses import TensorDC, NormalTensor, PaddedTensor, UnpaddedTensor
from matformer.extra import WERSAAttention
from dataclasses import replace
from torch.nn.functional import scaled_dot_product_attention
try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func, flash_attn_varlen_func, flash_attn_varlen_qkvpacked_func
    from flash_attn.modules.mha import get_alibi_slopes
    _is_flash_attn_available=True
except:
    _is_flash_attn_available=False
from matformer.cached_stuff import CachedStuff



class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        q_dim: int,               
        k_dim: Optional[int] = None,
        v_dim: Optional[int] = None,
        is_cross_attention: bool = False,
        nheads: int = 8,
        bias: bool = False,
        positional_encoding: Literal['alibi', 'rope', 'nope', 'sinusoidal'] = 'nope',
        #dropout: float = 0.0, # Not supported by FlexAttention yet
        cache: Optional['CachedStuff'] = None,
        attn_impl: Literal['flash', 'sdpa', 'xformers', 'flex', 'wersa'] = 'flash',
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
        self.cache = CachedStuff() if cache is None else cache  # Initialize the cache of attention masks and positional embeddings  
        self.nheads = nheads
        self.attn_impl = attn_impl
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
        
        # Alibi initialization
        if self.positional_encoding == 'alibi':
            self.alibi = True
            if attn_impl == 'flash':
                # For multi gpu training, register alibi_slopes as buffer instead of creating tensor 
                alibi_slopes = torch.tensor(self.cache.get_alibi_slopes(nheads, device=torch.device(device), dtype=torch.float32), dtype=torch.float32)
                self.register_buffer('alibi_slopes', alibi_slopes)        
        else:
            self.alibi = False
            self.register_buffer('alibi_slopes', None) 
        # RoPE initialization
        if self.positional_encoding == 'rope':
            self.rotary_emb = self.cache.get_rotary_emb(self.head_dim)
            
        # Packed qkv projection for efficiency  
        if not is_cross_attention or self.qkv_samedim:
            self.packed_proj = nn.Linear(self.q_dim, 3 * q_dim, bias=bias)
        else:
            self.q_proj = nn.Linear(q_dim, q_dim, bias=bias)
            self.k_proj = nn.Linear(k_dim, q_dim, bias=bias)
            self.v_proj = nn.Linear(v_dim, q_dim, bias=bias)
            
        self.out_proj = nn.Linear(q_dim, self.q_dim, bias=bias)  # Out projection

    def _flash_forward(
        self, q=None, k=None, v=None, qkv=None,
        query_input=None, key_input=None,
        sliding=False, alibi=False):
        sliding_window = (-self.sliding_window, -self.sliding_window) if sliding else (-1, -1)
        
        if key_input is None:
            key_input = query_input
            
        if qkv is None:
            q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))  # [B, H, S, Hd]
        else:
            qkv = qkv.unflatten(-1, [3, self.nheads, self.head_dim])  # (B, S, 3, H, Hd)
            
        if isinstance(query_input, UnpaddedTensor):
            # Version with Unpadding
            device = (qkv if qkv is not None else q).device
            cu_q = query_input.cu_seqlens.to(device=device, dtype=torch.int32)
            cu_k = key_input.cu_seqlens.to(device=device, dtype=torch.int32)
            
            alibi_slopes = self.alibi_slopes if alibi else None
            
            if qkv is not None:
                return flash_attn_varlen_qkvpacked_func(
                    qkv,
                    cu_seqlens=cu_q,
                    max_seqlen=query_input.max_seq_len,
                    alibi_slopes=alibi_slopes,
                    causal=self.is_causal,
                    window_size=sliding_window
                )
            else:
                return flash_attn_varlen_func(
                    q, k, v,
                    cu_seqlens_q=cu_q,
                    cu_seqlens_k=cu_k,
                    max_seqlen_q=query_input.max_seq_len,
                    max_seqlen_k=key_input.max_seq_len,
                    alibi_slopes=alibi_slopes,
                    causal=self.is_causal,
                    window_size=sliding_window
                )
        
        # Version with Padding
        alibi_slopes = self.alibi_slopes if alibi else None
        
        if qkv is not None:
            return flash_attn_qkvpacked_func(
                qkv,
                alibi_slopes=alibi_slopes,
                causal=self.is_causal,
                window_size=sliding_window
            )
        else:
            return flash_attn_func(
                q, k, v,
                alibi_slopes=alibi_slopes,
                causal=self.is_causal,
                window_size=sliding_window
            )

    def _alibi_score_mod(self, score, b, h, q_idx, kv_idx):
        """Required by Flex Attention"""
        L, S = score.shape[-2], score.shape[-1]
        device, dtype = score.device, score.dtype
        alibi_bias = self.cache.get_alibi_bias(L, S, self.nheads, device, dtype)
        return score + alibi_bias[0, h] * (kv_idx - q_idx)

    def forward(self, query_input, original_x=None,key_input=None, value_input=None, document_mask=None):
        """
        Input Tensors:
        query_input: (Batch, Seqlen, Qdim)
        key_input: (Batch, Seqlen, Kdim) [If omitted, self-attention]
        value_input: (Batch, Seqlen, Vdim) [If omitted, self-attention]
        Output:
        (Batch, Seqlen, Qdim) # Qdim is the query dimension as well as the output.
        """
        
        supports_unpadding = ['flash']
        supports_packed_qkv = ['flash']
        
        # Set defaults for self-attention
        if key_input is None:
            key_input = query_input
        if value_input is None:
            value_input = query_input
                
        # If the attention implementation does not support unpadding, eventually unpadded sequences must be padded
        repadded = False
        if self.attn_impl not in supports_unpadding:
            if isinstance(query_input,UnpaddedTensor):
                repadded = True  # A flag: if original inputs were padded from unpadded tensors, they will be unpadded again at the end
                query_input = query_input.pad()
                if self.is_cross_attention:
                    key_input = key_input.pad()
                    value_input = value_input.pad()

        # Extract tensors (handle both regular tensors and custom tensor classes)
        q_tensor = query_input.tensor if hasattr(query_input, 'tensor') else query_input
        k_tensor = key_input.tensor if hasattr(key_input, 'tensor') else key_input
        v_tensor = value_input.tensor if hasattr(value_input, 'tensor') else value_input
        
        # Projecting (eventually, packed projection)
        if self.qkv_samedim and not self.is_cross_attention:
            qkv_projected = self.packed_proj(q_tensor)
            q = None
            k = None
            v = None
            if self.attn_impl not in supports_packed_qkv or self.positional_encoding=='rope':
                q, k, v = torch.chunk(qkv_projected, 3, dim=-1)
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
			qkv_projected=None
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)
        # Generate (or get from cache) the attention mask for the attn. impl that requires it
        if self.attn_impl in ['sdpa', 'xformers', 'wersa']:
            attn_mask = self.cache.get_attention_mask(
                query=q, kv=k, nheads=self.nheads, causal=self.is_causal, 
                sliding_window=self.sliding_window, add_alibi=(self.positional_encoding == 'alibi')
            )

        # Attention computation                                     
        if self.attn_impl == 'flash':
            if self.positional_encoding=='rope':
                qkv_projected=None # Disable qkv packing for RoPe #CONTROLLARE: È NECESSARIO?
            attn_output = self._flash_forward(
                qkv=qkv_projected, q=q, k=k, v=v,
                sliding=(self.sliding_window is not None),
                query_input=query_input, key_input=key_input,
                alibi=(self.positional_encoding == 'alibi')
            )
        elif self.attn_impl == 'sdpa': 
            attn_output = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            attn_output = attn_output.transpose(1, 2)  # Transpose to match: (B, S, H, Hd)
        elif self.attn_impl == 'flex':          
            block_mask = self.cache.get_flex_blockmask(
                query=q, kv=k, nheads=self.nheads, causal=self.is_causal, 
                sliding_window=self.sliding_window, B=q.size(0)
            )
            attn_output = flex_attention(
                q, k, v, block_mask=block_mask, 
                score_mod=self._alibi_score_mod if self.alibi else None
            )
            attn_output = attn_output.transpose(1, 2)  # Transpose to match: (B, S, H, Hd)
        elif self.attn_impl == 'xformers': 
            raise NotImplementedError("xformers attention not implemented")
        elif self.attn_impl == 'wersa':
            if not hasattr(self, 'wersa_class'):
                self.wersa_class = WERSAAttention(
                    self.q_dim, self.nheads, 
                    decomp_levels=2, random_features=1024
                ).to(q.device)
            original_x=original_x.tensor if hasattr(original_x,'tensor') else original_x
            attn_output = self.wersa_class(q, k, v, original_x, self.is_causal)
            attn_output = attn_output.transpose(1, 2)  
        else:
            raise ValueError(f"Unknown attention implementation: {self.attn_impl}")
        
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
        hidden_size = int(hidden_size * ffn_factor)

        self.w13 = nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        self.w2 = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, _input: TensorDC) -> TensorDC:
        x1, x3 = torch.chunk(self.w13(_input.tensor), 2, -1)
        out = self.w2(F.silu(x1) * x3)
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
