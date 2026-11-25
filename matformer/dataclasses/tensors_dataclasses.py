#matformer/dataclasses/tensors_dataclasses.py

from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, replace, field
from typing import ClassVar, Tuple, Optional
import torch
import torch.nn.functional as F
from einops import rearrange


@dataclass(frozen=False, kw_only=True)
class TensorDC:
    tensor: torch.Tensor
    tensor_order: Optional[str] = None  # Useful to keep track of what is represented in the tensor (ex. heads in MHA)    
    following_attributes: dict = field(default_factory=dict)      # same (B,S,...) leading dims as tensor; padded/unpadded together with it
    attributes: dict = field(default_factory=dict)  # anything else: different-shaped tensors, metadata, etc. and carried unchanged
    isUnpadded: ClassVar[bool] = False
    isPadded: ClassVar[bool] = False
    isNormal: ClassVar[bool] = True
    def has(self, key: str) -> bool:
        """Return True if `key` is present in either attributes dict."""
        return key in self.following_attributes or key in self.attributes

    @property
    def shape(self) -> Tuple[int, ...]: return self.tensor.shape
    @property
    def nested(self) -> bool: return self.tensor.nested  # NB: pre-existing; likely should be `.is_nested` — left untouched, not in scope
    @property
    def dtype(self) -> torch.dtype: return self.tensor.dtype
    @property
    def device(self) -> torch.device: return self.tensor.device

    def to(self, *args, **kwargs) -> "TensorDC":
        """Move tensor and all tensor-valued attributes to a device/dtype, in place."""
        self.tensor = self.tensor.to(*args, **kwargs)
        self.following_attributes = {k: v.to(*args, **kwargs) for k, v in self.following_attributes.items()}
        self.attributes = {
            k: (v.to(*args, **kwargs) if isinstance(v, torch.Tensor) else v)
            for k, v in self.attributes.items()
        }
        return self

    def __add__(self, addendum) -> "TensorDC":
        """Add a tensor (or another TensorDC's tensor) to this one; attributes are untouched. Used e.g. for residual connections."""
        if isinstance(addendum, TensorDC):
            return replace(self, tensor=self.tensor + addendum.tensor)
        elif isinstance(addendum, torch.Tensor):
            return replace(self, tensor=self.tensor + addendum)
        return NotImplemented

    def __mul__(self, factor) -> "TensorDC":
        """Elementwise product with a tensor (or another TensorDC's tensor); attributes are untouched. Used e.g. in gated MLPs."""
        if isinstance(factor, TensorDC):
            return replace(self, tensor=self.tensor * factor.tensor)
        elif isinstance(factor, torch.Tensor):
            return replace(self, tensor=self.tensor * factor)
        return NotImplemented

    def apply(self, module, *args, **kwargs) -> "TensorDC":
        """Syntactic sugar for replace(self, tensor=module(self.tensor, *args, **kwargs)).
        The idiomatic way to pass a TensorDC through any plain-tensor module: the wrapper
        (padding mask, tensor_order, attributes) rides along untouched."""
        return replace(self, tensor=module(self.tensor, *args, **kwargs))

    def unpad(self) -> "UnpaddedTensor":
        """Convert to an UnpaddedTensor. Implemented only on PaddedTensor."""
        raise NotImplementedError(f"Cannot unpad a tensor of type {type(self).__name__}")

    def pad(self, seq_len: Optional[int] = None) -> "PaddedTensor":
        """Convert to a PaddedTensor. Implemented only on UnpaddedTensor."""
        raise NotImplementedError(f"Cannot pad a tensor of type {type(self).__name__}")

    def detach(self) -> list:
        """Return a list of per-sample tensors with padding removed."""
        raise NotImplementedError(f"Cannot detach a tensor of type {type(self).__name__}")

    def stats(self, include_padding: bool = False, **kwargs) -> dict:
        """Statistics of the tensor, padding excluded by default.

        Delegates to :func:`matformer.utils.tensor_stats.compute_tensor_stats`, the one definition
        of a tensor's statistics. ``PaddedTensor`` drops its padding positions first, so every
        per-token statistic describes real tokens only; pass ``include_padding=True`` to count
        them too, which is useful when diagnosing the padding path itself.
        """
        from matformer.utils.tensor_stats import compute_tensor_stats
        tensor = self.tensor
        mask = getattr(self, "padding_mask", None)
        if not include_padding and mask is not None and tensor.dim() >= 2:
            tensor = tensor[~mask]
        return compute_tensor_stats(tensor, **kwargs)

    def with_attr(self, key: str, value) -> "TensorDC":
        """Return a copy with one non_following_attributes entry replaced."""
        return replace(self, non_following_attributes={**self.non_following_attributes, key: value})
    def detach_attributes(self) -> list:
        raise NotImplementedError
@dataclass(frozen=False, kw_only=True)
class NormalTensor(TensorDC):
    def pad(self) -> "NormalTensor":
        """No-op: already in unpadded/unbatched semantics."""
        return self
    def detach(self) -> list:
        """Return the wrapped tensor as a single-element list."""
        return [self.tensor]
    def reverse(self) -> "NormalTensor":
        """ flip the single sequence dim (dim=1, i.e. S in B,S,...)."""
        raise NotImplementedError
        # Fix if dimension includes batch or not
        return replace(self, tensor=self.tensor.flip(dims=[1]))
    def __len__(self):
        return len(self.tensor)
@dataclass(frozen=False, kw_only=True)
class PaddedTensor(TensorDC):
    isPadded: ClassVar[bool] = True
    padding_mask: torch.Tensor  # True = padding position, False = real data

    def pad(self) -> "PaddedTensor":
        return self

    def unpad(self) -> "UnpaddedTensor":
        """Strip padding, flattening (batch, seq) into a packed sequence indexed by cu_seqlens."""
        mask = ~self.padding_mask
        indices = torch.nonzero(mask.flatten()).flatten()
        unpadded_x = rearrange(self.tensor, 'b s ... -> (b s) ...')[indices]

        seqlens = mask.sum(dim=1, dtype=torch.int32)
        cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
        # Keep max_seq_len a python int: a 0-d GPU tensor would force an implicit
        # GPU->CPU sync every time it is passed to flash-attn / triton kernels (per layer).
        max_seq_len = int(seqlens.max())

        def _unpad(v):
            return v.reshape(-1, *v.shape[2:])[indices]

        return UnpaddedTensor(
            tensor=unpadded_x,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seq_len=max_seq_len,
            original_seq_len=self.padding_mask.shape[1],
            batch_size=self.padding_mask.shape[0],
            following_attributes={k: _unpad(v) for k, v in self.following_attributes.items()},
            attributes=self.attributes,
        )
    def __len__(self):
        return self.padding_mask.shape[0]
    def detach(self) -> list:
        """Return a list of raw tensors, one per batch element, with padding removed."""
        return [self.tensor[t][self.padding_mask[t] == False] for t in range(self.padding_mask.shape[0])]

    def pad_right(self) -> "PaddedTensor":
        """Left-pad so real tokens are right-aligned — required for autoregressive generation.

        Args: none. Reads real lengths from padding_mask.
        Returns: new PaddedTensor with real tokens flush-right, pads on the left.
        """
        real_lens = (~self.padding_mask).sum(dim=1)
        B, L = self.tensor.shape[0], self.tensor.shape[1]
        out = self.tensor.new_zeros(B, L)
        new_mask = torch.ones(B, L, dtype=torch.bool, device=self.tensor.device)
        for i, n in enumerate(real_lens.tolist()):
            n = int(n)
            out[i, L-n:] = self.tensor[i, :n]
            new_mask[i, L-n:] = False
        return replace(self, tensor=out, padding_mask=new_mask)
    @classmethod
    def from_rows(cls, rows: list, pad_value=0) -> "PaddedTensor":
        """Inverse of detach(): rebuild a PaddedTensor from variable-length per-sequence
        tensors, batching to the max length among them and deriving padding_mask fresh."""
        lengths = [r.shape[0] for r in rows]
        B, L = len(rows), max(lengths)
        out = rows[0].new_full((B, L, *rows[0].shape[1:]), pad_value)
        mask = torch.ones(B, L, dtype=torch.bool, device=rows[0].device)
        for i, r in enumerate(rows):
            out[i, :len(r)] = r
            mask[i, :len(r)] = False
        return cls(tensor=out, padding_mask=mask)

@dataclass
class PackedLayout:
    """Index helpers for one packed layout, shared by every layer and by both directions of a
    bidirectional mixer instead of being rebuilt on each call.

    Keyed by the identity of the cu_seqlens tensor (see packed_layout), so a batch pays for
    these index vectors once per step whatever the number of layers. They also survive
    dataclasses.replace, which is what the reverse pass needs: a mixer hands back a brand new
    UnpaddedTensor around the same cu_seqlens, and the reversal index must not be rebuilt there.
    """
    cu_seqlens: torch.Tensor
    n: int
    segment_ids: torch.Tensor = field(init=False)
    reverse_index: torch.Tensor = field(init=False)
    cu_pair: torch.Tensor = field(init=False)

    def __post_init__(self):
        pos = torch.arange(self.n, device=self.cu_seqlens.device)
        self.segment_ids = torch.searchsorted(self.cu_seqlens, pos, right=True) - 1
        starts, ends = self.cu_seqlens[:-1], self.cu_seqlens[1:]
        self.reverse_index = starts[self.segment_ids] + (ends[self.segment_ids] - 1 - pos)
        # Reversing each segment in place leaves lengths and segment order untouched, so the pack
        # followed by its reversed twin is described by these same boundaries shifted by n.
        self.cu_pair = torch.cat([self.cu_seqlens, self.cu_seqlens[1:] + self.n])


_LAYOUT_CACHE: OrderedDict = OrderedDict()
_LAYOUT_CACHE_SIZE = 16


def packed_layout(cu_seqlens: torch.Tensor, n: int) -> PackedLayout:
    key = (id(cu_seqlens), n)
    layout = _LAYOUT_CACHE.get(key)
    if layout is not None:
        _LAYOUT_CACHE.move_to_end(key)
        return layout
    layout = PackedLayout(cu_seqlens, n)
    # The cache holds cu_seqlens itself, so an id can never be recycled under a live entry.
    _LAYOUT_CACHE[key] = layout
    while len(_LAYOUT_CACHE) > _LAYOUT_CACHE_SIZE:
        _LAYOUT_CACHE.popitem(last=False)
    return layout


_CU_ON_DEVICE_CACHE: OrderedDict = OrderedDict()
_CU_ON_DEVICE_CACHE_SIZE = 16


def cu_seqlens_on_device(cu_seqlens: torch.Tensor, device, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """cu_seqlens on `device`, with at most one transfer per source tensor.

    The dataloader builds cu_seqlens on the CPU (host-side slicing stays free there), but
    varlen kernels (flash-attn, fla) insist on a device copy: a bare .to(device) inside a
    kernel re-runs the H2D hop once per layer, per direction. Keyed by the identity of the
    source tensor, exactly like packed_layout, so a batch pays the transfer once per step
    however many layers consume it. dtype=None keeps the source dtype.
    """
    target = torch.device(device)
    if cu_seqlens.device == target and (dtype is None or cu_seqlens.dtype == dtype):
        return cu_seqlens
    key = (id(cu_seqlens), str(target), dtype)
    cached = _CU_ON_DEVICE_CACHE.get(key)
    if cached is not None:
        _CU_ON_DEVICE_CACHE.move_to_end(key)
        return cached[1]
    moved = cu_seqlens.to(device=target, dtype=dtype)
    # The cache holds cu_seqlens itself, so an id can never be recycled under a live entry.
    _CU_ON_DEVICE_CACHE[key] = (cu_seqlens, moved)
    while len(_CU_ON_DEVICE_CACHE) > _CU_ON_DEVICE_CACHE_SIZE:
        _CU_ON_DEVICE_CACHE.popitem(last=False)
    return moved


@dataclass(frozen=False, kw_only=True)
class UnpaddedTensor(TensorDC):
    indices: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seq_len: int
    original_seq_len: int
    batch_size: int
    isUnpadded: ClassVar[bool] = True

    def unpad(self) -> "UnpaddedTensor":
        return self

    def pad(self, seq_len: Optional[int] = None, pad_token=0) -> "PaddedTensor":
        """Scatter packed tokens back into a (batch, seq_len) padded layout. `pad_token` fills the main tensor's pad slots; attributes are always zero-filled. A target narrower than original_seq_len recomputes the scatter positions from the packed lengths, so a short batch is not padded to the model window."""
        target_seq_len = seq_len if seq_len is not None else self.original_seq_len
        if target_seq_len == self.original_seq_len:
            indices = self.indices
        else:
            lengths = (self.cu_seqlens[1:] - self.cu_seqlens[:-1]).tolist()
            step = target_seq_len
            indices = torch.cat([
                torch.arange(row * step, row * step + length, dtype=torch.long,
                             device=self.indices.device)
                for row, length in enumerate(lengths)]) if lengths else self.indices
        n = self.batch_size * target_seq_len

        def _scatter(v, fill=0):
            out = v.new_full((n, *v.shape[1:]), fill)
            out[indices] = v
            return rearrange(out, '(b s) ... -> b s ...', b=self.batch_size, s=target_seq_len)

        padding_mask_flat = torch.ones(n, dtype=torch.bool, device=self.indices.device)
        padding_mask_flat[indices] = False

        return PaddedTensor(
            tensor=_scatter(self.tensor, pad_token),
            padding_mask=rearrange(padding_mask_flat, '(b s) -> b s', b=self.batch_size, s=target_seq_len),
            following_attributes={k: _scatter(v) for k, v in self.following_attributes.items()},
            attributes=self.attributes,
        )
    def layout(self) -> PackedLayout:
        return packed_layout(self.cu_seqlens, self.tensor.shape[0])

    def cu_seqlens_device(self, device=None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """cu_seqlens on the compute device (default: the one holding the tokens), transferred
        at most once per batch — see cu_seqlens_on_device."""
        return cu_seqlens_on_device(self.cu_seqlens,
                                    self.tensor.device if device is None else device, dtype)

    def reverse_index(self) -> torch.Tensor:
        return self.layout().reverse_index

    def segment_ids(self) -> torch.Tensor:
        """Which packed segment each token belongs to, for masking taps that would reach into
        the neighbouring document."""
        return self.layout().segment_ids

    def reverse(self) -> "UnpaddedTensor":
        """Flip within each packed segment; involution, and the index is shared, never rebuilt."""
        idx = self.reverse_index()
        return replace(self, tensor=self.tensor[idx],
                       following_attributes={k: v[idx] for k, v in self.following_attributes.items()})

    def __len__(self):
        return self.batch_size

def apply_kernel(module, x, *args, **kwargs):
    """Run a kernel on `x`, which is either a TensorDC (its padding/unpadding wrapper rides along)
    or a plain tensor. Registered kernels must accept both: nested inside another kernel - a linear
    used by an MLP, for instance - there is no TensorDC left to carry the wrapper."""
    return x.apply(module, *args, **kwargs) if isinstance(x, TensorDC) else module(x, *args, **kwargs)
