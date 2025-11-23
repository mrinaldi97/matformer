from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
from matformer.matformer_registry import registry


@registry.register(
    'positional_encoding', 'rope', 'flash',
    requires=['flash_attn'],
    priority=10,
    metadata={
        'tensor_order_input': 'BSHD',
        'tensor_order_output': 'BSHD',
        'tensor_order_qkv_packed_input': 'BS3HD',
        'tensor_order_qkv_packed_input': 'BS3HD',
        'supports_unpadding': True,
        'supports_packed_qkv': False,  # But NOT simultaneously with unpadding
    }
)
class FlashRotaryEmbedding(nn.Module):
    """
    RoPE implementation using Flash Attention's optimized CUDA kernels.
    
    Note: Flash Attention's apply_rotary_emb_qkv_ does NOT support variable-length
    sequences (unpadding). When unpadding is needed, we fall back to applying
    rotary embeddings separately to Q and K using apply_rotary_emb.
    """
    
    @staticmethod
    def is_available():
        try:
            from flash_attn.layers.rotary import apply_rotary_emb, apply_rotary_emb_qkv_
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False
    
    def __init__(self, interleaved: bool = False, **kwargs):
        """
        Args:
            interleaved: If True, rotate pairs of even and odd dimensions (GPT-J style).
                        If False, rotate 1st half and 2nd half (GPT-NeoX style, default).
        """
        super().__init__()
        self.interleaved = interleaved
    
    def _halve_cos_sin(self, cos: Tensor, sin: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Convert cos/sin from shape (seqlen, dim) to (seqlen, dim/2).
        
        The cache computes cos/sin with the full head dimension by concatenating
        [freqs, freqs], but Flash Attention expects only (seqlen, rotary_dim/2).
        """
        rotary_dim_half = cos.shape[-1] // 2
        return cos[..., :rotary_dim_half], sin[..., :rotary_dim_half]
    
    def forward(
        self,
        qkv: Optional[Tensor] = None,
        q: Optional[Tensor] = None,
        k: Optional[Tensor] = None,
        cos: Optional[Tensor] = None,
        sin: Optional[Tensor] = None,
        cu_seqlens: Optional[Tensor] = None,
        max_seq_len: Optional[int] = None,
        seqlen_offsets: Union[int, Tensor] = 0,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """
        Apply rotary embeddings to Q and K.
        
        Args:
            qkv: Packed QKV tensor of shape (batch, seqlen, 3, nheads, headdim)
                 or (total_seqlen, 3, nheads, headdim) for unpadded.
                 Mutually exclusive with separate q, k inputs.
            q: Query tensor of shape (batch, seqlen, nheads, headdim)
               or (total_seqlen, nheads, headdim) for unpadded.
            k: Key tensor of shape (batch, seqlen, nheads, headdim)
               or (total_seqlen, nheads, headdim) for unpadded.
            cos: Cosine values, shape (seqlen, dim). Will be halved internally.
            sin: Sine values, shape (seqlen, dim). Will be halved internally.
            cu_seqlens: Cumulative sequence lengths for variable-length batching.
                       Shape (batch + 1,). If provided, uses unpadded path.
            max_seq_len: Maximum sequence length (required if cu_seqlens provided).
            seqlen_offsets: Position offset(s) for KV cache inference.
        
        Returns:
            Tuple of (qkv, q, k) where:
            - If qkv input was provided: (rotated_qkv, None, None)
            - If q, k inputs were provided: (None, rotated_q, rotated_k)
        """
        from flash_attn.layers.rotary import apply_rotary_emb, apply_rotary_emb_qkv_
        
        # Halve cos/sin to match Flash Attention's expected format
        cos_half, sin_half = self._halve_cos_sin(cos, sin)
        
        is_unpadded = cu_seqlens is not None
        
        if qkv is not None:
            # Packed QKV path
            if is_unpadded:
                # Flash Attention's apply_rotary_emb_qkv_ does NOT support unpadding.
                # We must unpack, apply separately, and repack.
                # qkv shape: (total_seqlen, 3, nheads, headdim)
                q_rot = apply_rotary_emb(
                    qkv[:, 0],  # (total_seqlen, nheads, headdim)
                    cos_half,
                    sin_half,
                    interleaved=self.interleaved,
                    inplace=False,
                    seqlen_offsets=seqlen_offsets,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seq_len,
                )
                k_rot = apply_rotary_emb(
                    qkv[:, 1],  # (total_seqlen, nheads, headdim)
                    cos_half,
                    sin_half,
                    interleaved=self.interleaved,
                    inplace=False,
                    seqlen_offsets=seqlen_offsets,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seq_len,
                )
                # Repack: stack along dim=1 to get (total_seqlen, 3, nheads, headdim)
                qkv_out = torch.stack([q_rot, k_rot, qkv[:, 2]], dim=1)
                return qkv_out, None, None
            else:
                # Padded path: use the optimized packed function
                # qkv shape: (batch, seqlen, 3, nheads, headdim)
                qkv_rot = apply_rotary_emb_qkv_(
                    qkv,
                    cos_half,
                    sin_half,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offsets,
                )
                return qkv_rot, None, None
        else:
            # Separate Q, K path
            assert q is not None and k is not None, "Must provide either qkv or both q and k"
            
            q_rot = apply_rotary_emb(
                q,
                cos_half,
                sin_half,
                interleaved=self.interleaved,
                inplace=False,
                seqlen_offsets=seqlen_offsets,
                cu_seqlens=cu_seqlens if is_unpadded else None,
                max_seqlen=max_seq_len if is_unpadded else None,
            )
            k_rot = apply_rotary_emb(
                k,
                cos_half,
                sin_half,
                interleaved=self.interleaved,
                inplace=False,
                seqlen_offsets=seqlen_offsets,
                cu_seqlens=cu_seqlens if is_unpadded else None,
                max_seqlen=max_seq_len if is_unpadded else None,
            )
            return None, q_rot, k_rot
            
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor


@registry.register(
    'positional_encoding', 'rope', 'torch',
    requires=['torch'],
    priority=1,
    metadata={
        'tensor_order_input': 'BSHD',
        'supports_unpadding': False,
        'supports_packed_qkv': False,
    }
)
class TorchRotaryEmbedding(nn.Module):
    """
    Pure PyTorch RoPE implementation (fallback when Flash Attention unavailable).
    
    Applies rotary positional embeddings by rotating pairs of features.
    Features i and i+d/2 are treated as 2D coordinates and rotated by
    position-dependent angles.
    """
    
    @staticmethod
    def is_available():
        return True
    
    def __init__(self, **kwargs):
        super().__init__()
    
    def _neg_half(self, x: Tensor) -> Tensor:
        """
        Rearrange for rotation: [-x[..., d/2:], x[..., :d/2]]
        
        This creates the "rotated" version needed for the rotation formula:
        x_rot = x * cos + neg_half(x) * sin
        """
        d_2 = x.shape[-1] // 2
        return torch.cat([-x[..., d_2:], x[..., :d_2]], dim=-1)
    
    def _apply_rotary(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """
        Apply rotary embedding to tensor x.
        
        Args:
            x: Input tensor of shape (batch, seq_len, nheads, headdim)
            cos: Cosine values of shape (seq_len, headdim)
            sin: Sine values of shape (seq_len, headdim)
        
        Returns:
            Rotated tensor of same shape as x
        """
        seq_len = x.shape[1]
        
        # Reshape cos/sin for broadcasting: (seq_len, dim) -> (1, seq_len, 1, dim)
        cos = cos[:seq_len].unsqueeze(0).unsqueeze(2)
        sin = sin[:seq_len].unsqueeze(0).unsqueeze(2)
        
        # Apply rotation: x_rot = x * cos + neg_half(x) * sin
        return x * cos + self._neg_half(x) * sin
    
    def forward(
        self,
        qkv: Optional[Tensor] = None,
        q: Optional[Tensor] = None,
        k: Optional[Tensor] = None,
        cos: Optional[Tensor] = None,
        sin: Optional[Tensor] = None,
        cu_seqlens: Optional[Tensor] = None,
        max_seq_len: Optional[int] = None,
        seqlen_offsets: Union[int, Tensor] = 0,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """
        Apply rotary embeddings to Q and K.
        
        Args:
            qkv: Not supported in this implementation, must be None.
            q: Query tensor of shape (batch, seq_len, nheads, headdim)
            k: Key tensor of shape (batch, seq_len, nheads, headdim)
            cos: Cosine values, shape (seq_len, headdim)
            sin: Sine values, shape (seq_len, headdim)
            cu_seqlens: Not supported, must be None.
            max_seq_len: Ignored.
            seqlen_offsets: Position offset for KV cache (int only).
        
        Returns:
            Tuple of (None, rotated_q, rotated_k)
        """
        assert qkv is None, "TorchRotaryEmbedding does not support packed QKV"
        assert cu_seqlens is None, "TorchRotaryEmbedding does not support unpadding"
        assert q is not None and k is not None, "Must provide both q and k"
        
        # Handle sequence offset (for KV cache during inference)
        if isinstance(seqlen_offsets, int) and seqlen_offsets > 0:
            seq_len = q.shape[1]
            cos = cos[seqlen_offsets:seqlen_offsets + seq_len]
            sin = sin[seqlen_offsets:seqlen_offsets + seq_len]
        
        q_rot = self._apply_rotary(q, cos, sin)
        k_rot = self._apply_rotary(k, cos, sin)
        
        return None, q_rot, k_rot
