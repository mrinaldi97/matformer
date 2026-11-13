from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
from matformer.matformer_registry import registry


@registry.register(
    'positional_encoding', 'rope', 'flash',
    requires=['flash_attn'],
    priority=10, #Check against new FA3/FA4!!!
    metadata={
        'tensor_order_input': 'BSHD',
        'tensor_order_output': 'BSHD',
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
    def _test():
        """flash-attn's rotary kernels need CUDA and are installed only as a side effect of
        flash-attn itself: try one real rotation on the smallest possible tensor."""
        q = torch.randn(1, 2, 1, 16)
        FlashRotaryEmbedding()(q=q, k=q.clone(), cos=torch.ones(2, 16), sin=torch.zeros(2, 16))
        return True
    
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


@registry.register(
    'positional_encoding', 'rope', 'torch',
    requires=['torch'],
    priority=1,
    metadata={
        'tensor_order_input': 'BSHD',
        'tensor_order_output': 'BSHD',
        'tensor_order_qkv_packed_input': None,  # packed qkv not accepted; the loader unpacks it
        'supports_unpadding': True,
        'supports_packed_qkv': False,
        }
)
class TorchRotaryEmbedding(nn.Module):
    """
    Pure PyTorch RoPE implementation (fallback when Flash Attention unavailable).

    Supports both padded ((B, S, H, D) with position-indexed cos/sin) and
    varlen-packed ((T, H, D) with cu_seqlens) inputs. Positions for a pack are
    computed with a vectorized searchsorted (no python loops, no GPU->CPU
    syncs), so this implementation never forces a pad/unpad round trip.
    """
    
    @staticmethod
    def _test():
        q = torch.randn(1, 2, 1, 16)
        TorchRotaryEmbedding()(q=q, k=q.clone(), cos=torch.ones(2, 16), sin=torch.zeros(2, 16))
        return True
    
    def __init__(self, interleaved: bool = False, **kwargs):
        super().__init__()
        self.interleaved = interleaved
    
    def _neg_half(self, x: Tensor) -> Tensor:
        """
        Rearrange for rotation: [-x[..., d/2:], x[..., :d/2]]
        
        This creates the "rotated" version needed for the rotation formula:
        x_rot = x * cos + neg_half(x) * sin
        """
        if self.interleaved:

            # GPT-J style: rotate the (even, odd) pairs - x[..., 2i] pairs with x[..., 2i+1]

            return torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).flatten(-2)

        d_2 = x.shape[-1] // 2

        return torch.cat([-x[..., d_2:], x[..., :d_2]], dim=-1)

    @staticmethod
    def _positions(cu_seqlens: Tensor, total: int) -> Tensor:
        """Per-token positions within each packed segment, (T,), all vectorized."""
        ar = torch.arange(total, device=cu_seqlens.device)
        seg = torch.searchsorted(cu_seqlens, ar, right=True) - 1
        return ar - cu_seqlens[seg]

    def _rotate_3d(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """x: (T, H, D); cos/sin: (T, dim) already indexed per token."""
        rot_dim = cos.shape[-1]
        x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
        c, s = cos.unsqueeze(1), sin.unsqueeze(1)  # broadcast over heads
        return torch.cat([x_rot * c + self._neg_half(x_rot) * s, x_pass], dim=-1)
    
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
        rot_dim = cos.shape[-1]
        x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]

        cos = cos[:seq_len].unsqueeze(0).unsqueeze(2)
        sin = sin[:seq_len].unsqueeze(0).unsqueeze(2)

        rotated = self._neg_half(x_rot)   # note: _neg_half must operate on rot_dim width now
        x_rotated = x_rot * cos + rotated * sin
        return torch.cat([x_rotated, x_pass], dim=-1)
    
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
            q: Query tensor of shape (batch, seq_len, nheads, headdim) or (T, nheads, headdim)
            k: Key tensor of shape (batch, seq_len, nheads, headdim) or (T, nheads, headdim)
            cos: Cosine values, shape (seq_len, headdim)
            sin: Sine values, shape (seq_len, headdim)
            cu_seqlens: cumulative sequence lengths of the pack (T path).
            max_seq_len: ignored (kept for interface compatibility).
            seqlen_offsets: Position offset for KV cache (int only).
        
        Returns:
            Tuple of (None, rotated_q, rotated_k)
        """
        assert qkv is None, "TorchRotaryEmbedding does not support packed QKV"
        assert q is not None and k is not None, "Must provide both q and k"
        offset = seqlen_offsets if isinstance(seqlen_offsets, int) else 0

        if cu_seqlens is not None:
            # Varlen: x may arrive either as a pack (T, H, D) or as a padded-grid
            # (B, S, H, D) laid out so that each segment is a contiguous span of
            # grid rows (the layout _transpose_for_kernel produces from cu_seqlens).
            # Flatten to (T, H, D) and index cos/sin per token with vectorized positions.
            x = q.reshape(-1, *q.shape[2:]) if q.dim() == 4 else q
            # k keeps its own head count: grouped-query keys have fewer heads than queries,
            # and cos/sin are per token, so they broadcast over heads either way.
            k_x = k.reshape(-1, *k.shape[2:]) if k.dim() == 4 else k
            positions = self._positions(cu_seqlens, x.shape[0]) + offset
            return None, self._rotate_3d(x, cos[positions], sin[positions]), self._rotate_3d(k_x, cos[positions], sin[positions])

        # Padded path
        if offset > 0:
            seq_len = q.shape[1]
            cos = cos[offset:offset + seq_len]
            sin = sin[offset:offset + seq_len]
        
        q_rot = self._apply_rotary(q, cos, sin)
        k_rot = self._apply_rotary(k, cos, sin)
        
        return None, q_rot, k_rot
