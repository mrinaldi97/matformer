from matformer.matformer_registry import registry
import torch.nn as nn
import torch
# attention/sdpa.py
@registry.register(
    'attention', 'normal', 'sdpa',
    requires=['torch'],
    priority=2,
    metadata={
        'tensor_order_input': 'BHSD',
        'tensor_order_output': 'BHSD',
        'supports_unpadding': False,
        'supports_packed_qkv': False,
        'supports_sliding_window': True,
        'supports_alibi': True,
    }
)
class SDPAKernel(nn.Module):
    def is_available():
        try:
            from torch.nn.functional import scaled_dot_product_attention
            return True
        except Exception:
            return False
    def __init__(self, nheads, head_dim, is_causal, sliding_window, positional_encoding, cache, **kwargs):
        super().__init__()
        self.nheads = nheads
        self.is_causal = is_causal
        self.sliding_window = sliding_window
        self.cache = cache
        if 'alibi' in positional_encoding:
            self.add_alibi=True
    def forward(self, qkv=None, q=None, k=None, v=None, query_input=None, key_input=None, **kwargs):
        from torch.nn.functional import scaled_dot_product_attention
        
        # Generate (or get from cache) the attention mask
        attn_mask = self.cache.get_attention_mask(
            query=q, kv=k, nheads=self.nheads, causal=self.is_causal, 
            sliding_window=self.sliding_window, add_alibi=True
        )
        
        attn_output = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return attn_output
