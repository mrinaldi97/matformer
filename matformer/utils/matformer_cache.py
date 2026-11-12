import torch
import math
from typing import Optional, Tuple, Any, Dict, Callable
from torch.nn.attention.flex_attention import flex_attention, create_block_mask#,create_nested_block_mask
from warnings import warn

class LayerStateCache:
    """
    Generic per-layer sequential state store, indexed by layer_idx. Satisfies the
    minimal contract expected by fla's get_layer_cache/update_layer_cache (len,
    __getitem__, .update(layer_idx=..., **kwargs)), but is intentionally generic —
    not tied to any one state shape — so the same object can later hold KV-cache
    entries or other per-layer recurrent state, not just FLA's recurrent/conv state.
    """
    def __init__(self):
        self._states = {}

    def __len__(self):
        return (max(self._states.keys()) + 1) if self._states else 0

    def __getitem__(self, layer_idx):
        return self._states[layer_idx]

    def __contains__(self, layer_idx):
        return layer_idx in self._states

    def update(self, layer_idx, **kwargs):
        self._states[layer_idx] = kwargs
        return self._states[layer_idx]

    def get(self, layer_idx, default=None):
        return self._states.get(layer_idx, default)

    def clear(self):
        self._states = {}

def compute_yarn_inv_freq(dim, base, factor, original_max_position_embeddings,
                           beta_fast, beta_slow, attention_factor, device):
    """Returns (inv_freq, mscale). Standard YaRN NTK-by-parts interpolation."""
    def find_correction_dim(num_rot, dim, base, max_pos):
        return (dim * math.log(max_pos / (num_rot * 2 * math.pi))) / (2 * math.log(base))

    def find_correction_range(lo, hi, dim, base, max_pos):
        l = math.floor(find_correction_dim(lo, dim, base, max_pos))
        h = math.ceil(find_correction_dim(hi, dim, base, max_pos))
        return max(l, 0), min(h, dim - 1)

    def linear_ramp(mn, mx, n, device):
        if mn == mx:
            mx += 0.001
        v = (torch.arange(n, dtype=torch.float32, device=device) - mn) / (mx - mn)
        return torch.clamp(v, 0, 1)

    pos_freqs = base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    inv_extrap = 1.0 / pos_freqs
    inv_interp = 1.0 / (factor * pos_freqs)

    low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_max_position_embeddings)
    extrap_factor = 1 - linear_ramp(low, high, dim // 2, device)
    inv_freq = inv_interp * (1 - extrap_factor) + inv_extrap * extrap_factor

    mscale = attention_factor if attention_factor is not None else (0.1 * math.log(factor) + 1.0)
    return inv_freq, mscale
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
        self.cu_seqlens_cache = {}
        self.storage={}
        self.additional_logs={}
        self.layer_states = LayerStateCache()
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

    # --------- Sequence lengths ---------

    def get_cu_seqlens_cpu(self, cu_seqlens: torch.Tensor) -> list:
        """cu_seqlens as a python list, with at most one device sync per tensor.

        A packed batch carries the same cu_seqlens object through every layer, while varlen
        kernels need the boundaries on the host to slice sequences: without this a model
        would pay a sync per layer. The tensor itself is kept in the cache, which is what
        makes data_ptr a safe key -- while an entry is alive its memory cannot be reused.
        """
        key = (cu_seqlens.data_ptr(), cu_seqlens.numel(), str(cu_seqlens.device))
        cached = self.cu_seqlens_cache.get(key)
        if cached is not None and cached[0] is cu_seqlens:
            return cached[1]
        values = cu_seqlens.detach().to(device='cpu', dtype=torch.int64).tolist()
        self.cu_seqlens_cache[key] = (cu_seqlens, values)
        while len(self.cu_seqlens_cache) > 8:
            self.cu_seqlens_cache.pop(next(iter(self.cu_seqlens_cache)))
        return values

    # --------- Attention Masks ---------

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
    
    def _compute_inv_freq(self, dim, theta, device, scaling_type="default", factor=1.0,
                        original_max_position_embeddings=None, beta_fast=32, beta_slow=1,
                        attention_factor=None, low_freq_factor=1.0, high_freq_factor=4.0):
        """dim here is the rotary_dim (already reduced by partial_rotary_factor upstream)."""
        if scaling_type == "linear":
            inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
            return inv_freq / factor, 1.0
        if scaling_type == "dynamic":
            # caller must pass seq_len-adjusted theta 
            inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
            return inv_freq, 1.0
        if scaling_type == "yarn":
            from matformer.utils.yarn import compute_yarn_inv_freq  # new small helper, see below
            return compute_yarn_inv_freq(dim, theta, factor, original_max_position_embeddings,
                                        beta_fast, beta_slow, attention_factor, device)
        if scaling_type == "llama3":
            # HF LlamaRotaryEmbedding default_rope_function: wavelengths below
            # high_freq_wavelen stay unscaled, above low_freq_wavelen are divided by
            # `factor`, in between smoothly blended (Llama 3.1-style rope_scaling).
            inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
            wavelen = 2 * torch.pi / inv_freq
            high_freq_wavelen = original_max_position_embeddings / high_freq_factor
            low_freq_wavelen = original_max_position_embeddings / low_freq_factor
            blend = torch.clamp((original_max_position_embeddings / wavelen - low_freq_factor)
                                / (high_freq_factor - low_freq_factor), 0.0, 1.0)
            inv_freq = (1 - blend) * inv_freq / factor + blend * inv_freq
            return inv_freq, 1.0
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        return inv_freq, 1.0

    def get_rotary_cos_sin(self, seq_len, dim, device, dtype, theta=10000.0,
                            partial_rotary_factor=1.0, scaling_type="default", scaling_factor=1.0,
                            original_max_position_embeddings=None, beta_fast=32, beta_slow=1,
                            attention_factor=None, low_freq_factor=1.0, high_freq_factor=4.0):
        rotary_dim = int(dim * partial_rotary_factor)
        key = (rotary_dim, theta, device, scaling_type, scaling_factor, low_freq_factor, high_freq_factor)
        if key in self.rotary_emb_cache:
            cos, sin, cached_len = self.rotary_emb_cache[key]
            if cached_len >= seq_len and cos.dtype == dtype:
                return cos[:seq_len], sin[:seq_len]

        inv_freq, mscale = self._compute_inv_freq(rotary_dim, theta, device, scaling_type,
                                                scaling_factor, original_max_position_embeddings,
                                                beta_fast, beta_slow, attention_factor,
                                                low_freq_factor, high_freq_factor)
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        freqs = torch.cat([freqs, freqs], dim=-1)
        cos, sin = (freqs.cos() * mscale).to(dtype), (freqs.sin() * mscale).to(dtype)
        self.rotary_emb_cache[key] = (cos, sin, seq_len)
        return cos, sin



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

def ensure_cache(cache):
    """Ensure cache and registry initialization"""
    if cache is None:
        warn("Cache not provided, initializing new cache. Consider passing cache for efficiency.")
        cache = CachedStuff()
    return cache


