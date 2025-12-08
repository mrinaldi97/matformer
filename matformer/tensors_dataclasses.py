from __future__ import annotations
from dataclasses import dataclass, replace, field
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
    tensor_order: Optional[str] = None # Useful to keep track fof what is represented in the tensor (ex. heads in MHA)
    extra_attributes: dict = field(default_factory=dict) # A free dict, but that can work in tandem with extra_follow_keys in case of tensors
    extra_follow_keys: list = field(default_factory=list) # The extra_attributes that will follow padding/unpadding destiny 
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
    def to(self, *args, **kwargs):
        self.tensor = self.tensor.to(*args, **kwargs)
        if self.cloze_mask is not None:
            self.cloze_mask = self.cloze_mask.to(*args, **kwargs)
        if self.document_mask is not None:
            self.document_mask = self.document_mask.to(*args, **kwargs)
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
		unpadded_extra = {}
		for k, v in self.extra_attributes.items():
			if k in self.extra_follow_keys and isinstance(v, torch.Tensor):
				# v is shaped [b, s, ...] or [b, s]
				flat = v.reshape(-1, *v.shape[2:]) if v.dim() > 2 else v.flatten()
				unpadded_extra[k] = flat[indices]
			else:
				# keep unchanged
				unpadded_extra[k] = v
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
			document_mask=unpadded_doc_mask,
			extra_attributes=unpadded_extra,
			extra_follow_keys=self.extra_follow_keys
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
		padded_extra = {}
		for k, v in self.extra_attributes.items():
			if k in self.extra_follow_keys and isinstance(v, torch.Tensor):
				# v is indexed by the same flatten indices
				flat = v
				padded = flat.new_zeros(self.batch_size * target_seq_len, *flat.shape[1:])
				padded[self.indices] = flat
				padded_extra[k] = rearrange(padded, '(b s) ... -> b s ...',
											b=self.batch_size, s=target_seq_len)
			else:
				padded_extra[k] = v
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
            document_mask=padded_doc_mask,
            extra_attributes=padded_extra,
            extra_follow_keys=self.extra_follow_keys
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
