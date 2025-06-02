import torch
import torch.nn.functional as F
import torch.nn as nn

class Patcher(nn.Module):
	"""
		This module receives a tensor as input and call a patching function.
		According to the result of the patching function, it outputs the patched 
		version of the tensor.
	"""
	

	
class LpPooling(torch.nn.Module):
    def __init__(self, dim=0, p=2, keepdim=False):
        super().__init__()
        self.dim = dim
        self.p = torch.nn.Parameter(torch.tensor(float(p)))
        self.keepdim = keepdim

    def forward(self, x, dim):
        p_clamped = torch.clamp(self.p, min=1e-6)
        return torch.pow(torch.mean(torch.pow(torch.abs(x), p_clamped), dim=dim, keepdim=self.keepdim), 1 / p_clamped)



