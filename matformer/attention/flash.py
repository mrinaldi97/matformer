from matformer.matformer_registry import registry
# attention/flash.py
import torch
import torch.nn as nn
from typing import Optional
from matformer.tensors_dataclasses import PaddedTensor,UnpaddedTensor,NormalTensor
try:
    from flash_attn import flash_attn_func,flash_attn_qkvpacked_func,flash_attn_varlen_func,flash_attn_varlen_qkvpacked_func
except:
    pass

@registry.register(
    'attention', 'normal', 'flash',
    requires=['flash_attn'],
    priority=10,
    metadata={
        'tensor_order_input': 'BSHD',
        'tensor_order_qkv_packed_input': 'BS3HD'
        'tensor_order_qkv_packed_output': 'BS3HD',
        'tensor_order_output': 'BSHD',
        'supports_unpadding': True,
        'supports_packed_qkv': True,
        'supports_sliding_window': True,
        'supports_alibi': True,
    }
)
class FlashAttentionKernel(nn.Module):
    @staticmethod
    def is_available():
        try:
            from flash_attn import flash_attn_func,flash_attn_qkvpacked_func,flash_attn_varlen_func,flash_attn_varlen_qkvpacked_func
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False
    
    def __init__(self, nheads, head_dim, is_causal, sliding_window, positional_encoding, cache, device, **kwargs):
        super().__init__()
        self.nheads = nheads
        self.head_dim = head_dim
        self.is_causal = is_causal
        self.sliding_window = sliding_window
        self.cache = cache
        
        # Alibi initialization
        if positional_encoding == 'alibi':
            alibi_slopes = torch.tensor(
                cache.get_alibi_slopes(nheads, device=torch.device(device), dtype=torch.float32), 
                dtype=torch.float32
            )
            self.register_buffer('alibi_slopes', alibi_slopes)
        else:
            self.register_buffer('alibi_slopes', None)
    
    def forward(self, qkv=None, q=None, k=None, v=None, query_input=None, key_input=None, **kwargs):   
        sliding = self.sliding_window is not None
        if sliding:
            right_window = 0 if self.is_causal else -self.sliding_window
            sliding_window = (-self.sliding_window, right_window)
        else:
            sliding_window = (-1, -1)

        if key_input is None:
            key_input = query_input
            
        if qkv is not None:
            qkv = qkv.unflatten(-1, [3, self.nheads, self.head_dim])  # (B, S, 3, H, Hd)
            
        if isinstance(query_input, UnpaddedTensor):
            # Version with Unpadding
            device = (qkv if qkv is not None else q).device
            cu_q = query_input.cu_seqlens.to(device=device, dtype=torch.int32)
            cu_k = key_input.cu_seqlens.to(device=device, dtype=torch.int32)
            
            alibi_slopes = self.alibi_slopes
            
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
        alibi_slopes = self.alibi_slopes
        
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

