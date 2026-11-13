from matformer.matformer_registry import registry
# attention_implementations/sdpa.py
"""torch SDPA kernel.

Same maths as the legacy variant, plus the three things that keep packed batches from
being repadded on the way in:

* unpadding  - UnpaddedTensor arrives packed as (T,H,D) with cu_seqlens; each sequence gets
  its own SDPA call over zero-copy slices, so padding never costs FLOPs and no padded copy
  is ever materialised. Positional semantics (causal, sliding window, ALiBi) are applied
  per sequence, which is also what makes ALiBi correct on packed input.
* GQA        - heads are broadcast by SDPA (enable_gqa) instead of repeat_interleave, so K/V
  are never expanded to the query head count.
* packed QKV - the fused (...,S,3,H,D) projection is consumed as three views.

The padded path still applies the padding mask itself (supports_padding_mask), so padded
batches keep going through SDPA's single fused call instead of an unpad/repad round trip.
`sdpa_legacy` keeps the pre-existing behaviour for cross-checking.
"""
import torch
import torch.nn as nn
from matformer.dataclasses.tensors_dataclasses import TensorDC, PaddedTensor, UnpaddedTensor


@registry.register(
    'attention', 'normal', 'sdpa',
    requires=['torch'],
    priority=2,
    metadata={
        'tensor_order_input': 'BSHD',
        'tensor_order_qkv_packed_input': 'BS3HD',
        'tensor_order_qkv_packed_output': 'BS3HD',
        'tensor_order_output': 'BSHD',
        'supports_unpadding': True,
        'supports_packed_qkv': True,
        'supports_padding_mask': True,
        'supports_sliding_window': True,
        'supports_alibi': True,
        'supports_gqa': True
    }
)
class SDPAKernel(nn.Module):
    @staticmethod
    def _test():
        """One minimal attention call: cheaper to check than to discover a broken torch backend
        halfway through a model build. Runs once per process, at instantiation time."""
        from torch.nn.functional import scaled_dot_product_attention
        q, k, v = (torch.zeros(1, 1, 2, 4) for _ in range(3))
        scaled_dot_product_attention(q, k, v, is_causal=True)
        return True

    def __init__(self, nheads, head_dim, is_causal, sliding_window, positional_encoding, cache,
                 num_key_value_heads=None, attn_logit_softcap=None, **kwargs):
        super().__init__()
        self.nheads = nheads
        self.head_dim = head_dim
        self.num_kv_heads = num_key_value_heads or nheads
        self.is_causal = is_causal
        self.sliding_window = sliding_window
        self.cache = cache
        self.add_alibi = 'alibi' in positional_encoding
        self.attn_logit_softcap = attn_logit_softcap
        # Head broadcasting (enable_gqa) exists from torch 2.5 on, and not every backend
        # implements it: _attention tries it once and, if the call is rejected, expands K/V
        # itself from then on (the probe has to be the real call, since support is
        # per-backend and scaled_dot_product_attention has no introspectable signature).
        self.supports_gqa_kw = True

    # ---------------- attention call, GQA-aware ----------------
    def _expand_kv(self, q, k, v):
        """Broadcast grouped heads to the query head count, for the places where heads cannot
        be broadcast: the math/softcap path and any SDPA backend lacking enable_gqa."""
        rep = q.shape[-3] // k.shape[-3]
        if rep > 1:
            k = k.repeat_interleave(rep, dim=-3)
            v = v.repeat_interleave(rep, dim=-3)
        return k, v

    def _attention(self, q, k, v, attn_mask=None):
        """q: (B,H,L,D); k/v: (B,Hkv,S,D) with Hkv <= H. Output (B,H,L,D)."""
        if self.attn_logit_softcap:
            k, v = self._expand_kv(q, k, v)
            return self._softcap_attention(q, k, v, attn_mask)
        kwargs = {'enable_gqa': True} if q.shape[-3] != k.shape[-3] and self.supports_gqa_kw else {}
        if not kwargs:
            k, v = self._expand_kv(q, k, v)
        try:
            return torch.nn.functional.scaled_dot_product_attention(
                query=q, key=k, value=v, attn_mask=attn_mask,
                is_causal=(self.is_causal if attn_mask is None else False), **kwargs)
        except (TypeError, RuntimeError):
            if not kwargs:
                raise
            self.supports_gqa_kw = False   # backend cannot broadcast heads: do it ourselves
            return self._attention(q, k, v, attn_mask)

    # ---------------- packed (varlen) path ----------------
    def _segment_mask(self, L, S, device, dtype):
        """Mask for one sequence pair; None when SDPA can handle it natively (causal) or when
        nothing at all restricts attention (bidirectional, no window, no ALiBi)."""
        if self.sliding_window is None and not self.add_alibi:
            return None
        allowed = torch.ones(L, S, dtype=torch.bool, device=device)
        idx = torch.arange(L, device=device)
        kdx = torch.arange(S, device=device)
        if self.is_causal:
            allowed &= idx.view(L, 1) >= kdx.view(1, S)
        if self.sliding_window is not None:
            # HF / flash-attention semantics: attend to |q-k| < window (exclusive)
            allowed &= (idx.view(L, 1) - kdx.view(1, S)).abs() < self.sliding_window
        if not self.add_alibi:
            return None if bool(allowed.all()) else allowed.view(1, 1, L, S)
        mask = torch.full((1, 1, L, S), float("-inf"), device=device, dtype=dtype)
        mask.masked_fill_(allowed.view(1, 1, L, S), 0.0)
        return mask + self.cache.get_alibi_bias(L, S, self.nheads, device, dtype)

    def _varlen(self, q, k, v, query_input, key_input):
        """q,k,v packed: (Tq,H,D) / (Tk,H,D). One SDPA per sequence over views.

        Sequence boundaries come from cu_seqlens, so document separation needs no mask:
        attention can never cross a segment.
        """
        cu_q = self.cache.get_cu_seqlens_cpu(query_input.cu_seqlens)
        same_pack = key_input is query_input or key_input.cu_seqlens is query_input.cu_seqlens
        cu_k = cu_q if same_pack else self.cache.get_cu_seqlens_cpu(key_input.cu_seqlens)
        out = torch.zeros_like(q)
        for i in range(len(cu_q) - 1):
            q0, q1 = cu_q[i], cu_q[i + 1]
            if q1 == q0:
                continue
            k0, k1 = (q0, q1) if same_pack else (cu_k[i], cu_k[i + 1])
            qs, ks, vs = (t.transpose(-3, -2).unsqueeze(0) for t in (q[q0:q1], k[k0:k1], v[k0:k1]))
            mask = self._segment_mask(qs.shape[-2], ks.shape[-2], q.device, q.dtype)
            out[q0:q1] = self._attention(qs, ks, vs, mask).squeeze(0).transpose(-3, -2)
        return out

    # ---------------- padded / plain batch path ----------------
    def _build_attention_mask(self, q, k, padding_mask, document_mask):
        B, nheads, L, _ = q.shape
        _, _, S, _ = k.shape
        device, dtype = q.device, q.dtype
        has_padding = padding_mask is not None
        if not self.add_alibi and not has_padding and document_mask is None and self.sliding_window is None:  # Fast path for NormalTensor
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

    def _softcap_attention(self, q, k, v, attn_mask):
        """tanh softcapping (Gemma2-style) needs materialized scores: plain math attention."""
        cap = self.attn_logit_softcap
        scores = torch.einsum('bhqd,bhkd->bhqk', q, k) / (q.shape[-1] ** 0.5)
        scores = cap * torch.tanh(scores / cap)
        if attn_mask is None and self.is_causal:   # fast path returned nothing: causal only
            L, S = q.shape[-2], k.shape[-2]
            attn_mask = torch.ones(L, S, dtype=torch.bool, device=q.device).tril()
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float('-inf'))
        elif attn_mask is not None:
            scores = scores + attn_mask
        return scores.softmax(dim=-1) @ v

    def forward(self, qkv=None, q=None, k=None, v=None, query_input=None, key_input=None,
                document_mask=None, **kwargs):
        if qkv is not None:
            q, k, v = qkv.unbind(dim=-3)      # (...,S,3,H,D) -> three views
        if key_input is None:
            key_input = query_input
        # Inputs are (...,S,H,D) (this kernel's tensor_order_input); SDPA wants (...,H,S,D).
        if getattr(query_input, 'isUnpadded', False):
            return self._varlen(q, k, v, query_input, key_input)
        q, k, v = (t.transpose(-3, -2) for t in (q, k, v))
        padding_mask = key_input.padding_mask if isinstance(key_input, PaddedTensor) else None
        attn_mask = self._build_attention_mask(q, k, padding_mask, document_mask)
        out = self._attention(q, k, v, attn_mask)
        return out.transpose(-3, -2)          # (B,H,L,D) -> (B,L,H,D), this kernel's output order
