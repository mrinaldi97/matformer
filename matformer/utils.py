import torch
import torch.nn.functional as F
import torch.nn as nn
import torch
from functools import reduce

class MaskBuilder:
    def __init__(self, config):
        self.config = config

    def build_mask_tensor(self, attention_types, query, kv=None, batch_size=None, num_heads=None,
                          is_sliding=False, document_mask=None, nested=False, **kwargs):
        if kv is None:
            kv = query.tensor
        else:
            kv = kv.tensor
        query = query.tensor

        B, L, S = query.shape[0], query.shape[-2], kv.shape[-2]
        device = query.device

        if self.config.attn_impl == 'sdpa':

            q_idx = torch.arange(L, device=device).view(L, 1)
            k_idx = torch.arange(S, device=device).view(1, S)
            mask = torch.ones(B, L, S, dtype=torch.bool, device=device)

            if 'causal' in attention_types:
                causal_mask = q_idx >= k_idx
                mask &= causal_mask

            if is_sliding:
                sliding_mask = torch.abs(q_idx - k_idx) <= self.config.sliding_window_size
                mask &= sliding_mask

            if 'document' in attention_types and document_mask is not None:
                doc_mask = document_mask[:, :, None] == document_mask[:, None, :]
                mask |= doc_mask

            return mask

        elif self.config.attn_impl == 'flex':
            def mask_fn(b, h, q, k):
                m = torch.ones_like(q, dtype=torch.bool)

                if 'causal' in attention_types:
                    m &= q >= k
                if is_sliding:
                    m &= (torch.abs(q - k) <= self.config.sliding_window_size)
                if 'document' in attention_types and document_mask is not None:
                    m |= (document_mask[b, q] == document_mask[b, k])

                return m

            if nested:
                return create_nested_block_mask(mask_mod=mask_fn, q_nt=query, kv_nt=kv,
                                                B=batch_size, H=num_heads, device=device)
            else:
                return create_block_mask(mask_mod=mask_fn, Q_LEN=L, KV_LEN=S,
                                         B=batch_size, H=num_heads, device=device)

        else:
            print("Unsupported attention implementation!")
            return None

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



def printmem(text):
    # Una funzioncina rozza per controllare l'uso della memoria, da togliere una volta che il codice è completo
    device = torch.device('cuda:0')
    free, total = torch.cuda.mem_get_info(device)
    mem_used_MB = (total - free) / 1024 ** 2
    print(f"Memory at {text}: {mem_used_MB}")
