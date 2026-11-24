"""FlashAttention incremental decoding: `flash_attn_with_kvcache` over the shared paged KV cache.

This is the fast path for inference on a machine with flash-attn but without flashinfer: the new
keys and values are written into the cache and the attention reads the prefix back in a single
kernel, so a decode step costs one token of work instead of the whole sequence.

Paged flash-attn wants pages that are a multiple of 256 tokens, which the KV cache is sized with
(see the kv_page_multiple metadata and GenerationMixin.paged_cache). RoPE stays outside the kernel,
applied by the mixer at the absolute position the state carries; ALiBi and sliding windows are
handed to flash-attn directly.
"""
import torch
import torch.nn as nn

from matformer.matformer_registry import registry

@registry.register(
    'attention', 'normal', 'flash_kv',
    requires=['flash_attn'],
    priority=15,
    constraints={'is_training': False},
    metadata={'tensor_order_input': '?SHD', 'tensor_order_output': '?SHD',
              'supports_unpadding': True, 'supports_packed_qkv': False,
              'supports_sliding_window': True, 'supports_alibi': True, 'supports_gqa': True,
              'needs_kv_cache': True, 'kv_page_multiple': 256}
)
class FlashAttentionKVCacheKernel(nn.Module):
    @staticmethod
    def _test():
        """One real incremental call: the paged entry point is a different code path from the
        training one, so it has to be proven, not assumed."""
        from flash_attn import flash_attn_with_kvcache
        if not torch.cuda.is_available():
            raise RuntimeError("flash-attn requires a CUDA device, none is available")
        q, k, v = (torch.zeros(1, 1, 1, 16, dtype=torch.float16, device='cuda') for _ in range(3))
        cache = torch.zeros(256, 256, 1, 16, dtype=torch.float16, device='cuda')
        flash_attn_with_kvcache(
            q, cache, cache.clone(), k, v,
            cache_seqlens=torch.ones(1, dtype=torch.int32, device='cuda'),
            block_table=torch.arange(256, dtype=torch.int32, device='cuda').view(1, -1),
            causal=True)
        return True

    def __init__(self, nheads, head_dim, is_causal, sliding_window, positional_encoding, cache,
                 device, layer_idx=None, num_key_value_heads=None, attn_logit_softcap=None, **kwargs):
        super().__init__()
        self.nheads, self.num_kv_heads = nheads, num_key_value_heads or nheads
        self.head_dim, self.is_causal = head_dim, is_causal
        self.sliding_window, self.attn_logit_softcap = sliding_window, attn_logit_softcap
        self.layer_idx = 0 if layer_idx is None else layer_idx
        self.alibi_slopes = (cache.get_alibi_slopes(nheads, device=torch.device(device),
                                                    dtype=torch.float32)
                             if 'alibi' in positional_encoding else None)

    def forward(self, q=None, k=None, v=None, query_input=None, key_input=None, **kwargs):
        from flash_attn import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
        state = (getattr(query_input, 'attributes', None) or {}).get('kv_state')
        # (-1, -1) is "no local window": anything else puts flash-attn in local mode, where the two
        # entry points align the window differently. Only a real sliding window belongs here.
        window = (-1, -1) if self.sliding_window is None else (self.sliding_window - 1, 0)
        if state is None:                       # no cache asked for: the ordinary flash calls
            if q.dim() == 3:
                # Packed varlen — masking, evaluation, any unpadded batch: one sequence after the
                # other, boundaries in cu_seqlens. flash_attn_func reads four dimensions and dies on
                # these with "Dimension out of range (expected [-3,2], got 3)"; the varlen entry
                # point is the one that understands them, exactly as plain flash uses it.
                key_input = key_input if key_input is not None else query_input
                return flash_attn_varlen_func(
                    q, k, v,
                    cu_seqlens_q=query_input.cu_seqlens_device(q.device, torch.int32),
                    cu_seqlens_k=key_input.cu_seqlens_device(q.device, torch.int32),
                    max_seqlen_q=query_input.max_seq_len, max_seqlen_k=key_input.max_seq_len,
                    softmax_scale=None, causal=self.is_causal, window_size=window,
                    softcap=self.attn_logit_softcap or 0.0, alibi_slopes=self.alibi_slopes)
            return flash_attn_func(q, k, v, causal=self.is_causal, softmax_scale=None,
                                   softcap=self.attn_logit_softcap or 0.0,
                                   window_size=window, alibi_slopes=self.alibi_slopes)
        keys, values = state.cache.layer(self.layer_idx)
        return flash_attn_with_kvcache(
            q, keys, values, k, v,
            cache_seqlens=state.cache_seqlens,
            block_table=state.page_table,
            softmax_scale=None,
            causal=self.is_causal,
            window_size=window,
            softcap=self.attn_logit_softcap or 0.0,
            alibi_slopes=self.alibi_slopes,
        )
