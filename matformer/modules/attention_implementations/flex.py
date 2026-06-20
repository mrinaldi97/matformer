from matformer.matformer_registry import registry
import torch.nn as nn
import torch
# attention/flex.py
@registry.register(
    'attention', 'normal', 'flex',
    requires=['torch>=2.5'],
    priority=1,
    metadata={
        'tensor_order_input': 'BHSD',
        'tensor_order_output': 'BHSD',
        'supports_unpadding': False,
        'supports_packed_qkv': False,
        'supports_sliding_window': True,
        'supports_alibi': True,
        'requires_score_mod': True,
    }
)
class FlexAttentionKernel(nn.Module): #TODO: Flex attention must be reviewed and tested
    @staticmethod
    def is_available():
        try:
            from torch.nn.attention.flex_attention import flex_attention
            return False #Ancora da implementare
        except Exception:
            return False
    
    def __init__(self, nheads, head_dim, is_causal, sliding_window, positional_encoding, cache, **kwargs):
        super().__init__()
        self.nheads = nheads
        self.is_causal = is_causal
        self.sliding_window = sliding_window
        self.cache = cache
        self.positional_encoding = positional_encoding
    
    def _alibi_score_mod(self, score, b, h, q_idx, kv_idx):
        """Required by Flex Attention"""
        L, S = score.shape[-2], score.shape[-1]
        device, dtype = score.device, score.dtype
        alibi_bias = self.cache.get_alibi_bias(L, S, self.nheads, device, dtype)
        return score + alibi_bias[0, h] * (kv_idx - q_idx)
    
    def forward(self, qkv=None, q=None, k=None, v=None, query_input=None, key_input=None, **kwargs):
        from torch.nn.attention.flex_attention import flex_attention
        
        block_mask = self.cache.get_flex_blockmask(
            query=q, kv=k, nheads=self.nheads, causal=self.is_causal, 
            sliding_window=self.sliding_window, B=q.size(0)
        )
        
        score_mod = self._alibi_score_mod if 'alibi' in self.positional_encoding else None
        
        attn_output = flex_attention(
            q, k, v, block_mask=block_mask, 
            score_mod=score_mod
        )
        return attn_output
