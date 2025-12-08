import torch
import math
from typing import Optional, Tuple, Any, Dict, Callable
from torch.nn.attention.flex_attention import flex_attention, create_block_mask,create_nested_block_mask


class CachedStuff:
    """
    This class will store elements that, for efficiency reason, can be cached.
    It is better to initialize this class at the highest possible level of the Matformer model.
    It caches:
    - ALiBi slopes and bias matrices
    - Attention masks (SDPA-style and block masks) 
    - Rotary embeddings
    - Sinusoidal embeddings (WIP)
    
    Example:
        >>> cache = CachedStuff()
        >>> alibi_bias = cache.get_alibi_bias(L=512, S=512, nheads=8, device='cuda', dtype=torch.float16)
        >>> mask = cache.get_attention_mask(query, kv, nheads=8, causal=True)
        >>> cache.clear_cache()  # Manual memory management
    """
    
    def __init__(self):
        self.slope_cache = {}
        self.bias_cache = {}
        self.mask_cache = {}
        self.rotary_emb_cache = {}
        self.storage={}
    # --------- ALiBi---------
    
    def _get_slopes(self, n):
        """Original ALiBi slopes function from the paper."""
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)                   # In the paper, we only train models that have 2^a heads for some a. This function has
        else:                                                 # some good properties that only occur when the input is a power of 2. To maintain that even
            closest_power_of_2 = 2**math.floor(math.log2(n)) # when the number of heads is not a power of 2, we use this workaround.
            return get_slopes_power_of_2(closest_power_of_2) + self._get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]

    def _build_alibi(self, L: int, S: int, num_heads: int, slopes: torch.Tensor, device=None):
        """ALiBi bias function from the github repo, compatible also with bert-style models."""
        # positions
        context_position = torch.arange(L, device=device)[:, None]   # (L, 1)
        memory_position  = torch.arange(S, device=device)[None, :]   # (1, S)
        relative_position = torch.abs(memory_position - context_position)  # (L, S)
        # slopes: tensor of shape (num_heads,), e.g. from get_slopes(num_heads)
        slopes = slopes.to(device) * -1  # negative slopes
        # shape (H, L, S)
        alibi = slopes[:, None, None] * relative_position[None, :, :]
        # final shape (1, H, L, S)
        return alibi.unsqueeze(0)

    def get_alibi_slopes(self, nheads: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Get ALiBi slopes from cache or put in cache."""
        key = (nheads, device, dtype)
        if key not in self.slope_cache:
            slopes = self._get_slopes(nheads)
            self.slope_cache[key] = torch.tensor(slopes, device=device, dtype=dtype)
        return self.slope_cache[key]

    def get_alibi_bias(self, L: int, S: int, nheads: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Get cached ALiBi bias matrix of shape (1, nheads, L, S)."""
        key = (nheads, L, S, device, dtype)
        if key not in self.bias_cache:
            slopes = self.get_alibi_slopes(nheads, device, dtype)
            self.bias_cache[key] = self._build_alibi(L, S, nheads, slopes, device).to(dtype)
        return self.bias_cache[key]

    # --------- Attention Masks ---------
    
    def _build_attention_mask(self, B: int, L: int, S: int, nheads: int, device: torch.device, dtype: torch.dtype,
                             causal: bool, sliding_window: Optional[int], document_mask: Optional[torch.Tensor],
                             add_alibi: bool) -> torch.Tensor:
        # Build boolean mask
        q_idx = torch.arange(L, device=device).view(L, 1)
        k_idx = torch.arange(S, device=device).view(1, S)
        mask_bool = torch.ones(B, L, S, dtype=torch.bool, device=device)
        
        if causal:
            mask_bool &= (q_idx >= k_idx)
        if sliding_window is not None:
            mask_bool &= (torch.abs(q_idx - k_idx) <= sliding_window)
        if document_mask is not None:
            doc_mask = document_mask[:, :, None] == document_mask[:, None, :]
            mask_bool &= doc_mask

        # Convert to float mask
        float_mask = torch.full((B, nheads, L, S), float("-inf"), device=device, dtype=dtype)
        float_mask.masked_fill_(mask_bool.unsqueeze(1), 0.0)

        # Add ALiBi if requested
        if add_alibi:
            bias = self.get_alibi_bias(L, S, nheads, device, dtype)
            float_mask = float_mask + bias

        return float_mask
    
    def get_attention_mask(self, query: torch.Tensor, kv: torch.Tensor, nheads: int, *,
                          causal: bool = False, sliding_window: Optional[int] = None,
                          document_mask: Optional[torch.Tensor] = None,
                          add_alibi: bool = False) -> torch.Tensor:
        """
        Get cached SDPA-style attention mask of shape (B, nheads, L, S).
        
        Args:
            query: Query tensor for shape/device/dtype info
            kv: Key/value tensor for shape info
            nheads: Number of attention heads
            causal: Apply causal masking
            sliding_window: Optional sliding window size
            document_mask: Optional per-token document IDs (not cached, computed on-the-fly)
            add_alibi: Whether to add ALiBi positional bias
            
        Returns:
            Float mask where -inf = masked, 0.0 = unmasked (+ ALiBi if requested)
        """
        B, L, S = query.shape[0], query.shape[-2], kv.shape[-2]
        device, dtype = query.device, query.dtype

        # If document_mask is provided, don't cache - compute on-the-fly
        if document_mask is not None:
            return self._build_attention_mask(B, L, S, nheads, device, dtype, causal, sliding_window, document_mask, add_alibi)

        # Cache only when no document_mask
        cache_key = ("attn", nheads, L, S, causal, sliding_window, add_alibi, device, dtype)
        
        if cache_key not in self.mask_cache:
            self.mask_cache[cache_key] = self._build_attention_mask(B, L, S, nheads, device, dtype, causal, sliding_window, None, add_alibi)
            
        return self.mask_cache[cache_key]

    def _build_block_mask(self, query: torch.Tensor, kv: torch.Tensor, H: int,
                         causal: bool,  sliding_window: Optional[int], document_mask: Optional[torch.Tensor],
                         nested: bool, B:Optional[int] = None):
        """Build block mask with all specified constraints."""
        device = query.device
        
        def mask_fn(b, h, q, k):
            m = True
            if causal:
                m &= (q >= k)
            if sliding_window is not None:
                m &= (abs(q - k) <= sliding_window)
            if document_mask is not None:
                m &= (document_mask[b, q] == document_mask[b, k])
            return m

        if nested:
            return create_nested_block_mask(mask_mod=mask_fn, q_nt=query, kv_nt=kv, B=B, H=H, device=device)
        else:
            L, S = query.shape[-2], kv.shape[-2]
            return create_block_mask(mask_mod=mask_fn, Q_LEN=L, KV_LEN=S, B=B, H=H, device=device)

    def get_flex_blockmask(self, query: torch.Tensor, kv: torch.Tensor, nheads: int, *,
                          causal: bool = False, sliding_window: Optional[int] = None,
                          document_mask: Optional[torch.Tensor] = None, nested: bool = False, B:Optional[int] = None):
        """Get cached block mask for custom attention kernels."""
        L, S = query.shape[-2], kv.shape[-2]
        device = query.device
        # If we are using nested tensors, we cannot cache (Right? Could someone check?)
        if nested:
            return self._build_block_mask(query, kv, B, nheads, causal, sliding_window, document_mask, nested, create_block_mask, create_nested_block_mask)          
        # If document_mask is provided, don't cache - compute on-the-fly
        if document_mask is not None:
            return self._build_block_mask(query, kv, B, nheads, causal, sliding_window, document_mask, nested, create_block_mask, create_nested_block_mask)

        # Cache only when no document_mask
        cache_key = ("flex", H, L, S, causal, sliding_window, nested, device)
        
        if cache_key not in self.mask_cache:
            self.mask_cache[cache_key] = self._build_block_mask(query, kv, B, nheads, causal, sliding_window, None, nested, create_block_mask, create_nested_block_mask)
            
        return self.mask_cache[cache_key]

    # --------- Rotary Embeddings ---------
    
    def _compute_inv_freq(self, dim: int, theta: float, device: torch.device) -> torch.Tensor:
        """Compute inverse frequencies in fp32 for precision."""
        return 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))

    def get_rotary_cos_sin(self, seq_len: int, dim: int, device: torch.device, dtype: torch.dtype,
                          theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cached cos/sin for RoPE of shape (seq_len, dim).
        
        Computation done in fp32 for precision (critical under AMP), output cast to dtype.
        Uses torch.outer instead of einsum to avoid AMP precision loss.
        """
        key = (dim, theta, device)
        
        # Check if cache exists and is large enough
        if key in self.rotary_emb_cache:
            cos, sin, cached_len = self.rotary_emb_cache[key]
            if cached_len >= seq_len and cos.dtype == dtype:
                return cos[:seq_len], sin[:seq_len]
        
        # Compute in fp32 for precision
        inv_freq = self._compute_inv_freq(dim, theta, device)
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # (seq_len, dim//2)
        freqs = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim) - standard RoPE interleave
        
        cos, sin = freqs.cos().to(dtype), freqs.sin().to(dtype)
        self.rotary_emb_cache[key] = (cos, sin, seq_len)
        
        return cos, sin

    def get_rotary_emb_old(self, dim: int, theta: int = 10000):
        """Get cached rotary embedding instance."""
        key = (dim, theta)
        if key not in self.rotary_emb_cache:
            from rotary_embedding_torch import RotaryEmbedding
            self.rotary_emb_cache[key] = RotaryEmbedding(dim=dim, theta=theta)
        return self.rotary_emb_cache[key]

    # --------- Cache Management ---------
    
    def clear_cache(self, cache_name: Optional[str] = None) -> None:
        """Clear specific cache or all caches."""
        if cache_name is None:
            self.slope_cache.clear()
            self.bias_cache.clear()
            self.mask_cache.clear()
            self.rotary_emb_cache.clear()
        elif cache_name == 'slopes':
            self.slope_cache.clear()
        elif cache_name == 'bias':
            self.bias_cache.clear()
        elif cache_name == 'masks':
            self.mask_cache.clear()
        elif cache_name == 'rotary':
            self.rotary_emb_cache.clear()
        else:
            raise ValueError(f"Unknown cache name: {cache_name}")

    def get_cache_size(self) -> Dict[str, int]:
        """Get number of entries in each cache."""
        return {
            'slopes': len(self.slope_cache),
            'bias': len(self.bias_cache),
            'masks': len(self.mask_cache),
            'rotary': len(self.rotary_emb_cache)
        }
