"""FlashInfer attention kernel: paged keys and values, so inference appends one token at a time
instead of recomputing the whole prefix.

It is an inference-only implementation (constraints={'is_training': False}): training keeps
flash/sdpa. It is also the only kernel that needs a KV batch state, so it is never selected by
priority -- ask for it with MATFORMER_ATTENTION_NORMAL=flashinfer or run_task.py --attention.

The state (matformer.utils.kv_cache.KVBatchState) travels inside the batch's attributes and says
everything the kernel needs: which pages back each sequence, how long it is, and where in the page
table the tokens of this call must be written. Positions are absolute, which is what makes a
decode step append after the prefix rather than overwrite it; RoPE stays outside the kernel (the
mixer applies it with the same offsets), so ALiBi is what flashinfer is asked for here.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from matformer.matformer_registry import registry
from matformer.utils.kv_cache import PAGED_KV_CACHE_KEY


@registry.register(
    'attention', 'normal', 'flashinfer',
    requires=['flashinfer'],
    priority=0,          # needs a KV state on its input: ask for it, it is never taken by default
    constraints={'is_training': False},
    metadata={
        'tensor_order_input': '?SHD',
        'tensor_order_output': '?SHD',
        'supports_unpadding': True,
        'supports_packed_qkv': False,
        'supports_sliding_window': True,
        'supports_alibi': True,
        'supports_gqa': True,
        'needs_kv_cache': True,
    }
)
class FlashInferAttentionKernel(nn.Module):
    WORKSPACE_BYTES = 64 << 20

    @staticmethod
    def _test():
        """One real paged prefill on the smallest possible problem: flashinfer is importable and
        still useless without a CUDA device, and its kernels are JIT-compiled on first use, so
        this also pays that cost once instead of halfway through a generation."""
        import flashinfer
        if not torch.cuda.is_available():
            raise RuntimeError("flashinfer requires a CUDA device, none is available")
        q, k, v = (torch.zeros(1, 1, 1, 64, dtype=torch.float16, device='cuda') for _ in range(3))
        pages = torch.zeros(1, 1, 1, 64, dtype=torch.float16, device='cuda'), k.clone()
        indptr = torch.tensor([0, 1], dtype=torch.int32, device='cuda')
        one = torch.ones(1, dtype=torch.int32, device='cuda')
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            torch.zeros(8 << 20, dtype=torch.uint8, device='cuda'), "NHD")
        wrapper.plan(indptr, indptr, one, one, 1, 1, 64, 1, q_data_type=torch.float16)
        wrapper.run(q.reshape(1, 1, 64), pages)
        return True

    def __init__(self, nheads, head_dim, is_causal, sliding_window, positional_encoding, cache,
                 device, layer_idx=None, num_key_value_heads=None, attn_logit_softcap=None, **kwargs):
        super().__init__()
        self.nheads, self.num_kv_heads = nheads, num_key_value_heads or nheads
        self.head_dim, self.is_causal, self.cache = head_dim, is_causal, cache
        self.sliding_window, self.attn_logit_softcap = sliding_window, attn_logit_softcap
        self.layer_idx = 0 if layer_idx is None else layer_idx
        self.alibi = 'alibi' in (positional_encoding or [])
        self.workspace = None
        self.runtimes = {}

    def _plan_args(self):
        return dict(pos_encoding_mode='ALIBI' if self.alibi else 'NONE',
                    window_left=-1 if self.sliding_window is None else self.sliding_window - 1,
                    logits_soft_cap=self.attn_logit_softcap or 0.0)

    def _runtime(self, state, q):
        """One pair of flashinfer wrappers per (shape, dtype), reused for every step: planning is
        per batch, allocating a wrapper per call is not free."""
        signature = (q.dtype, state.cache.dtype)
        runtime = self.runtimes.get(signature)
        if runtime is None:
            import flashinfer
            if self.workspace is None:
                self.workspace = torch.zeros(self.WORKSPACE_BYTES, dtype=torch.uint8,
                                             device=state.cache.device)
            runtime = {'prefill': flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.workspace, "NHD"),
                       'decode': flashinfer.BatchDecodeWithPagedKVCacheWrapper(self.workspace, "NHD"),
                       'planned': {}}
            self.runtimes[signature] = runtime
        return runtime

    def _plan(self, state, runtime, q, kv):
        """Plan once per step: every layer of this kernel shares the batch structure, only its
        pages differ, and the pages are handed to run(), not to plan()."""
        if runtime['planned'].get(state.phase) == state.version:
            return
        heads = dict(num_qo_heads=self.nheads, num_kv_heads=self.num_kv_heads,
                     head_dim=self.head_dim, page_size=kv.page_size)
        if state.phase == 'decode':
            runtime['decode'].plan(state.kv_indptr, state.kv_indices, state.kv_last_page_len,
                                   **heads, **self._plan_args(),
                                   q_data_type=q.dtype, kv_data_type=kv.dtype)
        else:
            runtime['prefill'].plan(state.qo_indptr, state.kv_indptr, state.kv_indices,
                                    state.kv_last_page_len, causal=self.is_causal,
                                    seq_lens=state.seq_lens,
                                    **heads, **self._plan_args(),
                                    q_data_type=q.dtype, kv_data_type=kv.dtype)
        runtime['planned'][state.phase] = state.version

    def forward(self, q=None, k=None, v=None, query_input=None, key_input=None, **kwargs):
        import flashinfer.page
        state = (query_input.attributes or {}).get('kv_state') if hasattr(query_input, 'attributes') else None
        if state is None:
            raise RuntimeError("The flashinfer kernel needs a KV batch state on the input: generate "
                               "with use_cache=True, or do not select this attention implementation.")
        kv = state.cache
        shape = q.shape
        flat = lambda t, heads: t.reshape(-1, heads, self.head_dim).contiguous()
        q, k, v = flat(q, self.nheads), flat(k, self.num_kv_heads), flat(v, self.num_kv_heads)
        flashinfer.page.append_paged_kv_cache(
            k, v, state.batch_indices, state.positions, kv.layer(self.layer_idx),
            state.kv_indices, state.kv_indptr, state.kv_last_page_len, kv_layout="NHD")
        runtime = self._runtime(state, q)
        self._plan(state, runtime, q, kv)
        return runtime[state.phase].run(q, kv.layer(self.layer_idx)).reshape(shape)
