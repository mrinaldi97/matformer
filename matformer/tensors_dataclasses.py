from __future__ import annotations
from dataclasses import dataclass, replace
from typing import ClassVar, Tuple, Optional
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

@dataclass(frozen=False,kw_only=True)
class TensorDC:
    tensor: torch.Tensor
    cloze_mask: Optional[torch.Tensor] = None # To be used for MLM objectives 
    document_mask: Optional[torch.Tensor] = None # Useful for BLT, Multimodal transformers...
    
        
    isUnpadded: ClassVar[bool] = False
    isPadded: ClassVar[bool] = False
    isNormal: ClassVar[bool] = True

    @property
    def has_cloze_mask(self) -> bool:
        return self.cloze_mask is not None

    @property
    def has_document_mask(self) -> bool:
        return self.document_mask is not None

    @property
    def shape(self) -> Tuple[int, ...]: return self.tensor.shape
    @property
    def nested(self) -> bool: return self.tensor.nested   
    @property
    def dtype(self) -> torch.dtype: return self.tensor.dtype
    @property
    def device(self) -> torch.device: return self.tensor.device
    def to(self,*args, **kwargs):
        self.tensor=self.tensor.to(*args, **kwargs)
        return self
        
        
    def __add__(self, addendum) -> "TensorDC":
        # The additions only add tensors, no masks. Fine for residual connections.
        if isinstance(addendum, TensorDC):
            return replace(self, tensor=self.tensor + addendum.tensor)
        elif isinstance(addendum, torch.Tensor):
            return replace(self, tensor=self.tensor + addendum)
        return NotImplemented

    def unpad(self) -> "UnpaddedTensor":
        raise NotImplementedError(f"Cannot unpad a tensor of type {type(self).__name__}")

    def pad(self, seq_len: Optional[int] = None) -> "PaddedTensor":
        raise NotImplementedError(f"Cannot pad a tensor of type {type(self).__name__}")

@dataclass(frozen=False,kw_only=True)
class NormalTensor(TensorDC):
    pass


@dataclass(frozen=False,kw_only=True)
class PaddedTensor(TensorDC):
    isPadded: ClassVar[bool] = True
    padding_mask: torch.Tensor
    def pad(self):
        return self
    def unpad(self) -> UnpaddedTensor:
        mask = ~self.padding_mask
        indices = torch.nonzero(mask.flatten()).flatten()
        unpadded_x = rearrange(self.tensor, 'b s ... -> (b s) ...')[indices]

        seqlens = mask.sum(dim=1, dtype=torch.int32)
        cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))

        unpadded_cloze_mask = None
        if self.has_cloze_mask:
            assert self.cloze_mask.shape == self.padding_mask.shape, "Cloze mask must have the same shape as padding mask"
            unpadded_cloze_mask = self.cloze_mask.flatten()[indices]

        unpadded_doc_mask = None
        if self.has_document_mask:
            assert self.document_mask.shape == self.padding_mask.shape, "Document mask must have the same shape as padding mask"
            unpadded_doc_mask = self.document_mask.flatten()[indices]

        return UnpaddedTensor(
            tensor=unpadded_x,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seq_len=int(seqlens.max()),
            original_seq_len=self.padding_mask.shape[1],
            batch_size=self.padding_mask.shape[0],
            cloze_mask=unpadded_cloze_mask,
            document_mask=unpadded_doc_mask
        )


@dataclass(frozen=False,kw_only=True)
class UnpaddedTensor(TensorDC):
    indices: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seq_len: int
    original_seq_len: int
    batch_size: int
    isUnpadded: ClassVar[bool] = True
    def unpad(self):
        return self
    def pad(self, seq_len: Optional[int] = None, pad_token=0) -> PaddedTensor:
        target_seq_len = seq_len if seq_len is not None else self.original_seq_len
        if pad_token==0:
            out = self.tensor.new_zeros(self.batch_size * target_seq_len, *self.tensor.shape[1:])
        else:
            out = self.tensor.new_ones(self.batch_size * target_seq_len, *self.tensor.shape[1:])*pad_token
        out[self.indices] = self.tensor
        
        padding_mask_flat = torch.ones(size=(self.batch_size * target_seq_len,), dtype=torch.bool, device=self.indices.device)
        padding_mask_flat[self.indices] = False

        padded_cloze_mask = None
        if self.has_cloze_mask:
            cloze_mask_flat = self.cloze_mask.new_zeros(self.batch_size * target_seq_len, dtype=self.cloze_mask.dtype)
            cloze_mask_flat[self.indices] = self.cloze_mask
            padded_cloze_mask = rearrange(cloze_mask_flat, '(b s) -> b s', b=self.batch_size, s=target_seq_len)

        padded_doc_mask = None
        if self.has_document_mask:
            doc_mask_flat = self.document_mask.new_zeros(self.batch_size * target_seq_len, dtype=self.document_mask.dtype)
            doc_mask_flat[self.indices] = self.document_mask
            padded_doc_mask = rearrange(doc_mask_flat, '(b s) -> b s', b=self.batch_size, s=target_seq_len)
            
        return PaddedTensor(
            tensor=rearrange(out, '(b s) ... -> b s ...', b=self.batch_size, s=target_seq_len),
            padding_mask=rearrange(padding_mask_flat, '(b s) -> b s', b=self.batch_size, s=target_seq_len),
            cloze_mask=padded_cloze_mask,
            document_mask=padded_doc_mask
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
