from matformer.matformer_registry import registry
import torch.nn as nn
import torch
# attention/wersa.py
@registry.register(
    'attention', 'wersa', 'wersa',
    requires=[],
    metadata={
        'tensor_order_input': 'BHSD',
        'tensor_order_output': 'BHSD',
        'supports_unpadding': False,
        'supports_packed_qkv': False,
    }
)
class WERSAKernel(nn.Module):
    def __init__(self, q_dim, nheads, head_dim, **kwargs):
        super().__init__()
        from .wersa_attention import WERSAAttention
        self.wersa_class = WERSAAttention(
            q_dim, nheads, 
            decomp_levels=2, random_features=1024
        )
        self.is_causal = kwargs.get('is_causal', True)
    
    def forward(self, q, k, v, original_x=None, **kwargs):
        original_x = original_x.tensor if hasattr(original_x, 'tensor') else original_x
        return self.wersa_class(q, k, v, original_x, self.is_causal)
