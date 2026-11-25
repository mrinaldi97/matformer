"""Paged KV attention in plain torch: the reference implementation of the KV cache contract.

It writes the new keys and values into their pages and attends over the prefix gathered back from
them, with no dependency beyond torch. That makes it two things at once: a fallback that gives
incremental decoding to any machine (no flash-attn, no flashinfer, no CUDA kernels), and the
reference the flash-attn and flashinfer kernels are checked against.

The contract every paged kernel honours: the batch carries a KVBatchState
(matformer.utils.kv_cache) saying which pages back each sequence, how long each one now is, and at
which absolute position the tokens of this call belong. RoPE stays outside, applied by the mixer
with the same offset the state carries.
"""
import torch
import torch.nn as nn

from matformer.matformer_registry import registry


@registry.register(
    'attention', 'normal', 'sdpa_kv',
    requires=['torch'],
    priority=0,
    constraints={'is_training': False},
    metadata={'tensor_order_input': '?SHD', 'tensor_order_output': '?SHD',
              'supports_unpadding': True, 'supports_packed_qkv': False,
              'supports_sliding_window': False, 'supports_alibi': True, 'supports_gqa': True,
              'needs_kv_cache': True, 'kv_page_multiple': 1}
)
class PagedSDPAKernel(nn.Module):
    """Reference paged attention. Quadratic in the prefix and entirely in torch: fast enough to
    check the others, portable enough to run anywhere."""

    def __init__(self, nheads, head_dim, is_causal, sliding_window, positional_encoding, cache,
                 device, layer_idx=None, num_key_value_heads=None, attn_logit_softcap=None, **kwargs):
        super().__init__()
        self.nheads, self.num_kv_heads = nheads, num_key_value_heads or nheads
        self.head_dim, self.is_causal, self.sliding_window = head_dim, is_causal, sliding_window
        self.attn_logit_softcap = attn_logit_softcap
        self.layer_idx = 0 if layer_idx is None else layer_idx
        self.alibi = 'alibi' in (positional_encoding or [])
        self.cache = cache

    def forward(self, q=None, k=None, v=None, query_input=None, key_input=None, **kwargs):
        state = (getattr(query_input, 'attributes', None) or {}).get('kv_state')
        if state is not None:
            keys, values = state.write(self.layer_idx, k, v)
            prefix_k, prefix_v = state.gather(keys), state.gather(values)
        else:                                   # no cache asked for: attend over this call only
            prefix_k, prefix_v = k, v
        repeat = self.nheads // self.num_kv_heads
        if repeat > 1:
            prefix_k, prefix_v = (t.repeat_interleave(repeat, dim=-2) for t in (prefix_k, prefix_v))
        q, prefix_k, prefix_v = (t.transpose(-3, -2) for t in (q, prefix_k, prefix_v))
        scores = q @ prefix_k.transpose(-1, -2) / self.head_dim ** 0.5
        if self.attn_logit_softcap:
            scores = self.attn_logit_softcap * torch.tanh(scores / self.attn_logit_softcap)
        if self.alibi:
            scores = scores + self.cache.get_alibi_bias(
                q.shape[-2], prefix_k.shape[-2], self.nheads, q.device, q.dtype)
        if self.is_causal:
            rows = torch.arange(q.shape[-2], device=q.device)[:, None]
            columns = torch.arange(prefix_k.shape[-2], device=q.device)[None, :]
            allowed = rows + prefix_k.shape[-2] - q.shape[-2] >= columns
            if self.sliding_window is not None:
                allowed &= (rows + prefix_k.shape[-2] - q.shape[-2]) - columns < self.sliding_window
            scores = scores.masked_fill(~allowed, float('-inf'))
        out = scores.softmax(-1) @ prefix_v                   # (B,H,L,D)
        return out.transpose(-3, -2)                          # (B,L,H,D), what the mixer expects
