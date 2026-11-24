"""Paged KV cache: backend-independent physical storage for the keys and values the model
computes, so an inference kernel can append to and attend over a sequence without rebuilding it.

Two pieces: PagedKVCache owns the pages and the per-layer tensors, allocated lazily so a model
that never generates pays nothing; KVBatchState is one running batch, tracking how much each of
its sequences holds and which pages back it. The state is what travels inside the batch's
attributes while generating, and it outlives a single generate() call when a conversation does:
that reuse is the whole point, so the cache never resizes itself while a state is alive.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch

PAGED_KV_CACHE_KEY = "paged_kv"


class PagedKVCache:
    def __init__(self, num_layers, num_kv_heads, head_dim, dtype, device,
                 page_size: int = 16, max_tokens: int = 32768):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device or 'cpu')
        self.page_size = page_size
        self.max_pages = max(1, -(-max_tokens // page_size))
        self._free = list(range(self.max_pages))
        self._layers: List[Optional[tuple]] = [None] * num_layers

    def allocate(self, pages: int) -> list:
        if pages > len(self._free):
            raise RuntimeError(f"PagedKVCache out of pages: {pages} asked, {len(self._free)} free "
                               f"({self.max_pages} of {self.page_size} tokens total).")
        return [self._free.pop() for _ in range(pages)]

    def release(self, pages) -> None:
        self._free.extend(pages)

    def layer(self, layer_idx: int, dtype=None, device=None) -> tuple:
        """The (k, v) tensors of one layer, (max_pages, page_size, kv_heads, head_dim) each.
        Created on demand with the dtype and device of the activations that arrive, so a model
        moved or cast after it was built never mismatches its own cache."""
        if self._layers[layer_idx] is None:
            shape = (self.max_pages, self.page_size, self.num_kv_heads, self.head_dim)
            dtype, device = dtype or self.dtype, device or self.device
            self._layers[layer_idx] = (torch.empty(shape, dtype=dtype, device=device),
                                       torch.empty(shape, dtype=dtype, device=device))
        return self._layers[layer_idx]

    def new_state(self, batch_size: int) -> "KVBatchState":
        return KVBatchState(self, batch_size)

    def clear(self) -> None:
        """Forget every page and release the layer tensors: the memory is genuinely given back."""
        self._free = list(range(self.max_pages))
        self._layers = [None] * self.num_layers

    @property
    def token_bytes(self) -> int:
        """What one token of conversation costs: every layer, key and value."""
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * self.dtype.itemsize

    @property
    def capacity_tokens(self) -> int:
        return self.max_pages * self.page_size

    @property
    def used_tokens(self) -> int:
        return (self.max_pages - len(self._free)) * self.page_size

    @property
    def bytes(self) -> int:
        """Memory of the pages built so far. It depends on the capacity, not on how much was
        generated: a reservation is a reservation."""
        live = sum(1 for layer in self._layers if layer is not None)
        return live * 2 * self.capacity_tokens * self.num_kv_heads * self.head_dim \
            * self.dtype.itemsize

    def report(self) -> str:
        return (f"{self.bytes / 2**20:.0f} MiB over {self.max_pages} pages "
                f"[{self.used_tokens}/{self.capacity_tokens} tokens, "
                f"{self.token_bytes / 1024:.1f} KiB/token]")


@dataclass
class KVBatchState:
    """One batch in flight: the pages of every sequence, its length, and the index tensors the
    kernels are given. `prepare` is called once per forward step with the number of new tokens
    each sequence is about to consume."""
    cache: PagedKVCache
    page_ids: List[list]
    lengths: List[int]
    phase: str = "prefill"
    backend_state: dict = field(default_factory=dict)

    def __init__(self, cache: PagedKVCache, batch_size: int):
        self.cache, self.page_ids, self.lengths = cache, [[] for _ in range(batch_size)], [0] * batch_size
        self.q_lens, self.version = [0] * batch_size, 0

    @property
    def total_tokens(self) -> int:
        return sum(self.lengths)

    def prepare(self, q_lens, phase: Optional[str] = None) -> "KVBatchState":
        q_lens = [int(n) for n in q_lens]
        cache, old = self.cache, self.lengths
        for i, (length, n) in enumerate(zip(old, q_lens)):
            missing = (length + n + cache.page_size - 1) // cache.page_size - len(self.page_ids[i])
            if missing > 0:
                self.page_ids[i].extend(cache.allocate(missing))
        self.lengths = [length + n for length, n in zip(old, q_lens)]
        self.phase = phase or ("decode" if all(n == 1 for n in q_lens) else "prefill")
        self.prefix_lens = old
        self._indexes(q_lens, old)
        self.version += 1
        return self

    def _indexes(self, q_lens, old) -> None:
        device, page_size = self.cache.device, self.cache.page_size
        q = torch.tensor(q_lens, dtype=torch.int32, device=device)
        self.qo_indptr = torch.cat((q.new_zeros(1), q.cumsum(0)))
        counts = torch.tensor([len(pages) for pages in self.page_ids], dtype=torch.int32, device=device)
        self.kv_indptr = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        self.kv_indices = torch.tensor([page for pages in self.page_ids for page in pages],
                                       dtype=torch.int32, device=device)
        self.kv_last_page_len = torch.tensor([(n - 1) % page_size + 1 for n in self.lengths],
                                             dtype=torch.int32, device=device)
        self.batch_indices = torch.arange(len(q_lens), dtype=torch.int32, device=device).repeat_interleave(q)
        # Positions in the page table are absolute: a decode step appends at the length the
        # sequence already had, otherwise it would overwrite the token before it.
        local = (torch.arange(int(q.sum()), dtype=torch.int32, device=device)
                 - self.qo_indptr[:-1].repeat_interleave(q))
        self.positions = torch.tensor(old, dtype=torch.int32, device=device).repeat_interleave(q) + local
        self.seq_lens = self.kv_last_page_len.new_tensor(self.lengths, dtype=torch.int32)
        # flash-attn takes the length the sequence had *before* this call and appends on top of it,
        # while flashinfer is told the total it must reach: both are exposed, neither recomputed.
        self.cache_seqlens = self.seq_lens - q.to(self.seq_lens.device)
        # Where each token of this call lands: its page and its offset inside it. Cheap to compute
        # (one pass over the new tokens, once per step) and shared by every paged kernel.
        self.write_pages = torch.tensor([self.page_ids[int(b)][int(p) // page_size]
                                         for b, p in zip(self.batch_indices.tolist(),
                                                         self.positions.tolist())],
                                        dtype=torch.long, device=device)
        self.write_offsets = (self.positions % page_size).long()
        width = max(len(pages) for pages in self.page_ids)
        self.page_table = torch.tensor(
            [pages + [pages[-1]] * (width - len(pages)) for pages in self.page_ids],
            dtype=torch.int32, device=device)

    def write(self, layer_idx, k, v):
        """Scatter the new keys and values into their pages; returns the layer's two tensors."""
        keys, values = self.cache.layer(layer_idx, k.dtype, k.device)
        keys[self.write_pages, self.write_offsets] = k.reshape(-1, *k.shape[-2:]).to(keys.dtype)
        values[self.write_pages, self.write_offsets] = v.reshape(-1, *v.shape[-2:]).to(values.dtype)
        return keys, values

    def gather(self, pages):
        """The whole prefix of every sequence, read back from the page table: (B, length, heads, dim)."""
        length = int(self.seq_lens.max())
        positions = torch.arange(length, device=pages.device)
        selected = self.page_table[:, (positions // self.cache.page_size).long()]
        return pages[selected, (positions % self.cache.page_size).long()]

    def release(self) -> None:
        """Give the pages back without dropping the tensors: a new conversation, same cache."""
        self.cache.release(page for pages in self.page_ids for page in pages)
        self.page_ids = [[] for _ in self.page_ids]
        self.lengths = [0] * len(self.lengths)
