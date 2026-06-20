from matformer.matformer_registry import registry
import torch.nn as nn
import torch

# attention/xformers.py
@registry.register(
    'attention', 'normal', 'xformers',
    requires=['xformers'],
    priority=3,
    metadata={
        'supports_unpadding': False,
        'supports_packed_qkv': False,
    }
)
class XFormersKernel(nn.Module): #TODO: xformers must be implemented
    def __init__(self, **kwargs):
        super().__init__()
        raise NotImplementedError("xformers attention not implemented")
    
    def forward(self, **kwargs):
        raise NotImplementedError("xformers attention not implemented")
