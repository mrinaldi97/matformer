from matformer.matformer_registry import registry
# attention_implementations/sdpa_legacy.py
"""Pre-unpadding SDPA kernel, kept as the `sdpa_legacy` variant.

It is the reference the new `sdpa` kernel is validated against: packed batches have to be
repadded by MultiHeadAttention before reaching it (and unpadded again after), which is
exactly the round trip the new kernel avoids. Selectable with
MATFORMER_ATTENTION_NORMAL=sdpa_legacy.
"""
import torch
import torch.nn as nn
from typing import Optional
from matformer.dataclasses.tensors_dataclasses import TensorDC, NormalTensor, PaddedTensor, UnpaddedTensor


@registry.register(
    'attention', 'normal', 'sdpa_legacy',
    requires=['torch'],
    priority=1,
    metadata={
        'tensor_order_input': 'BHSD',
        'tensor_order_output': 'BHSD',
        'supports_unpadding': False,
        'supports_packed_qkv': False,
        'supports_sliding_window': True,
        'supports_alibi': True,
        'supports_gqa': False,
        'supports_softcap': True
    }
)
class SDPALegacyKernel(nn.Module):
    @staticmethod
    def is_available():
        try:
            from torch.nn.functional import scaled_dot_product_attention
            return True
        except Exception:
            return False

    def __init__(self, nheads, head_dim, is_causal, sliding_window, positional_encoding, cache, attn_logit_softcap=None, **kwargs):
        super().__init__()
        self.nheads = nheads
        self.is_causal = is_causal
        self.sliding_window = sliding_window
        self.cache = cache
        self.add_alibi = 'alibi' in positional_encoding
        self.attn_logit_softcap = attn_logit_softcap

    def _build_attention_mask(self, q, k, padding_mask, document_mask):
            B, nheads, L, _ = q.shape
            _, _, S, _ = k.shape
            device, dtype = q.device, q.dtype
            has_padding = padding_mask is not None
            if not self.add_alibi and not has_padding and document_mask is None and self.sliding_window is None: # Fast path for NormalTensor
                return None
            # True means attention mechanism active on those tokens
            mask_bool = torch.ones(B, L, S, dtype=torch.bool, device=device)
            if has_padding:
                mask_bool &= (~padding_mask).unsqueeze(1)
            if self.is_causal:
                q_idx = torch.arange(L, device=device).view(L, 1)
                k_idx = torch.arange(S, device=device).view(1, S)
                mask_bool &= (q_idx >= k_idx)
            if self.sliding_window is not None:
                q_idx = torch.arange(L, device=device).view(L, 1)
                k_idx = torch.arange(S, device=device).view(1, S)
                # HF / flash-attention semantics: attend to |q-k| < window (exclusive)
                mask_bool &= (torch.abs(q_idx - k_idx) < self.sliding_window)
            if document_mask is not None:
                doc_mask = document_mask[:, :L, None] == document_mask[:, None, :S]
                mask_bool &= doc_mask
            if self.add_alibi:
                float_mask = torch.full((B, 1, L, S), float("-inf"), device=device, dtype=dtype)
                float_mask.masked_fill_(mask_bool.unsqueeze(1), 0.0)
                bias = self.cache.get_alibi_bias(L, S, self.nheads, device, dtype)
                return float_mask + bias

            return mask_bool.unsqueeze(1)

    def forward(self, qkv=None, q=None, k=None, v=None, query_input=None, key_input=None, document_mask=None, **kwargs):
        from torch.nn.functional import scaled_dot_product_attention
        padding_mask = key_input.padding_mask if isinstance(key_input, PaddedTensor) else None
        if self.attn_logit_softcap:
            # tanh softcapping (Gemma2-style) needs materialized scores: plain math attention
            cap = self.attn_logit_softcap
            scores = torch.einsum('bhqd,bhkd->bhqk', q, k) / (q.shape[-1] ** 0.5)
            scores = cap * torch.tanh(scores / cap)
            attn_mask = self._build_attention_mask(q, k, padding_mask, document_mask)
            if attn_mask is None and self.is_causal:   # fast path returned nothing: causal only
                L, S = q.shape[-2], k.shape[-2]
                attn_mask = torch.ones(L, S, dtype=torch.bool, device=q.device).tril()
            if attn_mask is not None and attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attn_mask, float('-inf'))
            elif attn_mask is not None:
                scores = scores + attn_mask
            return scores.softmax(dim=-1) @ v
        attn_mask = self._build_attention_mask(q, k, padding_mask, document_mask)
        use_native_causal = self.is_causal if attn_mask is None else False
        attn_output = scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            is_causal=use_native_causal
        )

        return attn_output
