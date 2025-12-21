from __future__ import annotations
from dataclasses import dataclass, replace, field
from typing import ClassVar, Tuple, Optional, Dict, Any, Callable
import torch
import torch.nn.functional as F

@dataclass(frozen=False, kw_only=True)
class TensorDC:
    tensor: torch.Tensor
    cloze_mask: Optional[torch.Tensor] = None
    document_mask: Optional[torch.Tensor] = None
    recurrence_mask: Optional[torch.Tensor] = None
    tensor_order: Optional[str] = None
    extra_attributes: dict = field(default_factory=dict)
    extra_follow_keys: list = field(default_factory=list)
    isUnpadded: ClassVar[bool] = False
    isPadded: ClassVar[bool] = False
    isNormal: ClassVar[bool] = True

    @property
    def has_cloze_mask(self) -> bool: return self.cloze_mask is not None
    @property
    def has_document_mask(self) -> bool: return self.document_mask is not None
    @property
    def shape(self) -> Tuple[int, ...]: return self.tensor.shape
    @property
    def dtype(self) -> torch.dtype: return self.tensor.dtype
    @property
    def device(self) -> torch.device: return self.tensor.device

    def to(self, *args, **kwargs):
        self.tensor = self.tensor.to(*args, **kwargs)
        if self.cloze_mask is not None: self.cloze_mask = self.cloze_mask.to(*args, **kwargs)
        if self.document_mask is not None: self.document_mask = self.document_mask.to(*args, **kwargs)
        if self.recurrence_mask is not None: self.recurrence_mask = self.recurrence_mask.to(*args, **kwargs)
        for k in self.extra_attributes:
            if isinstance(self.extra_attributes[k], torch.Tensor):
                self.extra_attributes[k] = self.extra_attributes[k].to(*args, **kwargs)
        return self

    def __add__(self, addendum) -> "TensorDC":
        val = addendum.tensor if isinstance(addendum, TensorDC) else addendum
        return replace(self, tensor=self.tensor + val)

    def _map_tensors(self, func: Callable[[torch.Tensor], torch.Tensor]) -> Dict[str, Any]:
        """Helper to apply a transformation function to all relevant tensors DRYly."""
        res = {
            "tensor": func(self.tensor),
            "cloze_mask": func(self.cloze_mask) if self.has_cloze_mask else None,
            "document_mask": func(self.document_mask) if self.has_document_mask else None,
            "extra_attributes": {}
        }
        for k, v in self.extra_attributes.items():
            if k in self.extra_follow_keys and isinstance(v, torch.Tensor):
                res["extra_attributes"][k] = func(v)
            else:
                res["extra_attributes"][k] = v
        return res

@dataclass(frozen=False, kw_only=True)
class PaddedTensor(TensorDC):
    isPadded: ClassVar[bool] = True
    padding_mask: torch.Tensor

    def pad(self): return self

    def unpad(self) -> UnpaddedTensor:
        # 1. Setup indices (Sync point, do it once)
        flat_mask = (~self.padding_mask).view(-1)
        indices = torch.nonzero(flat_mask).squeeze(1)
        
        # 2. Sequence length metadata
        seqlens = (~self.padding_mask).sum(dim=1, dtype=torch.int32)
        cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))

        # 3. Apply unpadding to all tracked tensors
        def _unpad_op(t: torch.Tensor) -> torch.Tensor:
            # Reshape to [B*S, ...] and index
            return t.view(-1, *t.shape[2:])[indices] if t.dim() > 2 else t.view(-1)[indices]

        processed = self._map_tensors(_unpad_op)

        return UnpaddedTensor(
            indices=indices, cu_seqlens=cu_seqlens,
            max_seq_len=int(seqlens.max()), 
            original_seq_len=self.padding_mask.shape[1],
            batch_size=self.padding_mask.shape[0],
            recurrence_mask=self.recurrence_mask,
            extra_follow_keys=self.extra_follow_keys,
            **processed
        )

@dataclass(frozen=False, kw_only=True)
class UnpaddedTensor(TensorDC):
    indices: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seq_len: int
    original_seq_len: int
    batch_size: int
    isUnpadded: ClassVar[bool] = True

    def unpad(self): return self

    def pad(self, seq_len: Optional[int] = None, pad_token=0) -> PaddedTensor:
        target_s = seq_len if seq_len is not None else self.original_seq_len
        total_elements = self.batch_size * target_s
        
        def _pad_op(t: torch.Tensor) -> torch.Tensor:
            # Create output buffer
            out_shape = (total_elements, *t.shape[1:])
            out = t.new_full(out_shape, pad_token) if pad_token != 0 else t.new_zeros(out_shape)
            # Scatter unpadded data back
            out[self.indices] = t
            # Reshape back to [B, S, ...]
            return out.view(self.batch_size, target_s, *t.shape[1:])

        processed = self._map_tensors(_pad_op)
        
        # Build padding mask
        mask_flat = torch.ones(total_elements, dtype=torch.bool, device=self.device)
        mask_flat[self.indices] = False
        
        return PaddedTensor(
            padding_mask=mask_flat.view(self.batch_size, target_s),
            recurrence_mask=self.recurrence_mask,
            extra_follow_keys=self.extra_follow_keys,
            **processed
        )
class ModuleWrapper(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x, *args, **kwargs):
        if isinstance(x, TensorDC):
            out_tensor = self.module(x.tensor, *args, **kwargs)
            return replace(x, tensor=out_tensor)
        else:
            return self.module(x, *args, **kwargs)
