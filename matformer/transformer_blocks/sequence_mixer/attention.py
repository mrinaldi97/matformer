"""
File: matformer/transformer_blocks/attention.py
"""
#Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
#Matformer
from matformer.matformer_module import MatformerModule
from matformer.dataclasses.tensors_dataclasses import TensorDC, NormalTensor, PaddedTensor, UnpaddedTensor
from matformer.dataclasses.matformer_config import MatformerConfig
from matformer.utils.matformer_cache import ensure_cache, CachedStuff
from matformer.matformer_registry import registry
#Other
from dataclasses import replace
from warnings import warn
from typing import Optional, Literal, Union, List, Dict, Any
import einops

from dataclasses import dataclass, field
@registry.register_dataclass("sequence_mixer", "attention", default=True)
@dataclass
class AttentionConfig(MatformerConfig):
    q_dim: Optional[int] = None
    k_dim: Optional[int] = None
    v_dim: Optional[int] = None
    head_dim: Optional[int] = None
    is_cross_attention: bool = False
    sliding_window_size: Optional[int] = None
    positional_encoding: Optional[
        Union[
            List[Literal["alibi", "rope", "sinusoidal", "nope", "learnable"]],
            Literal["alibi", "rope", "sinusoidal", "nope", "learnable"],
        ]
    ] = "alibi"
    qkv_bias: Optional[bool] = None
    attn_output_bias: Optional[bool] = None
    q_norm: Optional[bool] = False
    k_norm: Optional[bool] = False
    qk_norm_zero_centered: Optional[bool] = False
    qk_norm_per_head: Optional[bool] = False
    qk_norm_full_width: Optional[bool] = False  # QK-norm over the flattened (H*D) projection instead of per-head (OLMo-2)
    qk_norm_eps: Optional[float] = None  # explicit eps for the QK-norms (LFM2 ships norm_eps=1e-5 while the norm kernels default to 1e-6)
    gated_attention: Optional[bool] = False
    attn_logit_softcap: Optional[float] = None  # Gemma2-style tanh capping of attention logits
    qkv_clamp: Optional[float] = None  # clamp q/k/v projections to +-qkv_clamp (HF clip_qkv, OLMo-1)
    num_attention_heads: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    interpretability_friendly: Optional[bool] = False
    hooks: Dict[str, Union[str, Dict[str, Any]]] = field(default_factory=dict)
    # ROPE STUFF
    rope_theta: float = 10000.0
    rope_interleaved: Optional[bool] = False  # GPT-J-style interleaved pairs (NeoBERT) instead of half-split
    rope_scaling_type: Optional[Literal["default", "linear", "dynamic", "yarn", "llama3"]] = "default"
    rope_scaling_factor: Optional[float] = 1.0
    rope_scaling_original_max_position_embeddings: Optional[int] = None
    rope_scaling_low_freq_factor: Optional[float] = 1.0   # llama3 scaling only
    rope_scaling_high_freq_factor: Optional[float] = 4.0  # llama3 scaling only
    rope_scaling_beta_fast: Optional[float] = 32
    rope_scaling_beta_slow: Optional[float] = 1
    rope_scaling_attention_factor: Optional[float] = None
    partial_rotary_factor: Optional[float] = 1.0

@registry.register("sequence_mixer", "attention", "default")  
class MultiHeadAttention(MatformerModule):
    # Stable parameter names
    packed_proj: "param_name:qkv_proj"      # Packed Q+K+V projection (when applicable)
    q_proj: "param_name:q_proj"             # Separate query projection (cross-attention)
    k_proj: "param_name:k_proj"             # Separate key projection (cross-attention)
    v_proj: "param_name:v_proj"             # Separate value projection (cross-attention)
    out_proj: "param_name:o_proj"           # Output projection
    attn_kernel: "param_name:attn_kernel"   # Attention implementation kernel
    available_sockets = ['q', 'k', 'v', 'attn_scores', 'attn_pattern', 'z', 'attn_out', 'gate'] #Warning: hooks are available only in the Mechanistic Interpretability branch

    def __init__(
        self,
        config: 'AttentionConfig',
        is_causal: bool = True,
        is_hybrid=False, #Initialize two kernels, one for is_causal = True, another one for is_causal = False; supports giving is_causal as a forward argument
        device: str = 'cuda',
        layer_idx=0,
        cache: Optional['CachedStuff'] = None,
        hidden_size: Optional[int] = None #Not used
    ):
        super().__init__()
        self.config = config
        if isinstance(self.config.positional_encoding, str):
            self.config.positional_encoding = [self.config.positional_encoding]
        # Assertions


        if self.config.num_key_value_heads is None:
            self.config.num_key_value_heads = self.config.num_attention_heads  # default: MHA
        assert self.config.num_attention_heads % self.config.num_key_value_heads == 0, "nheads must be divisible by num_key_value_heads"
        if self.config.is_cross_attention:
            assert self.config.k_dim is not None, "You asked for a cross attention, but you haven't provided keys dim"
            assert self.config.v_dim is not None, "You asked for a cross attention, but you haven't provided values dim"
        else:
            self.config.k_dim = self.config.q_dim
            self.config.v_dim = self.config.q_dim

        # Initialization
        if cache is None:
            warn("Cache not provided to MultiHeadAttention, initializing new cache. Consider passing cache for efficiency.")
            cache = CachedStuff()
        self.cache = cache
        self.force_separed_qkv = self.config.q_norm or self.config.k_norm  # QK-Norm needs unpacked Q/K, so skip the fused qkv_proj path
        self.n_groups = self.config.num_attention_heads // self.config.num_key_value_heads  #1 for MHA, >1 for GQA
        self.is_causal = is_causal
        self.qkv_samedim = self.config.q_dim == self.config.k_dim and self.config.q_dim == self.config.v_dim
        self.head_dim=self.config.head_dim       
        if not self.config.head_dim:
            assert self.config.q_dim % self.config.num_attention_heads == 0, "q_dim is not divisible by nheads"
            self.head_dim = self.config.q_dim // self.config.num_attention_heads
        self.q_inner_dim = self.config.num_attention_heads * self.head_dim   # 4096 for Qwen3-4B        
        self.layer_idx=layer_idx if layer_idx is not None else 0
        self.is_hybrid = is_hybrid
        qk_norm_variant = 'rmsnorm_zc' if (self.config.qk_norm_per_head or self.config.qk_norm_zero_centered) else 'rmsnorm'
        zc_kwargs = lambda heads: {'groups': heads if self.config.qk_norm_per_head else None, 'zero_centered': self.config.qk_norm_zero_centered} if qk_norm_variant == 'rmsnorm_zc' else {}
        qk_eps = {} if self.config.qk_norm_eps is None else {'eps': self.config.qk_norm_eps}
        if self.config.qk_norm_full_width:
            # OLMo-2 style: one RMSNorm over the whole flattened q (or k) projection
            self.q_norm = registry.create('norm', 'rmsnorm', normalized_shape=self.q_inner_dim, **qk_eps) if self.config.q_norm else None
            self.k_norm = registry.create('norm', 'rmsnorm', normalized_shape=self.head_dim * self.config.num_key_value_heads, **qk_eps) if self.config.k_norm else None
        else:
            self.q_norm = registry.create('norm', qk_norm_variant, normalized_shape=self.head_dim, **zc_kwargs(self.config.num_attention_heads), **qk_eps) if self.config.q_norm else None
            self.k_norm = registry.create('norm', qk_norm_variant, normalized_shape=self.head_dim, **zc_kwargs(self.config.num_key_value_heads), **qk_eps) if self.config.k_norm else None
        self.gate_proj = registry.create('linear','linear',in_features=self.config.q_dim,out_features=self.q_inner_dim,bias=False) if self.config.gated_attention else None


        if self.config.interpretability_friendly:
            try:
                import einops
            except ImportError:
                print("Einops is required to do Mechanistic Interpretability!")
            self._init_hooks(self.config.hooks)
        # RoPE initialization
        if 'rope' in self.config.positional_encoding:
            #self.rotary_embedding = self.cache.get_rotary_emb(self.head_dim)
            self.rotary_embedding = registry.create('positional_encoding','rope',
                                                    interleaved=self.config.rope_interleaved)
            self.rotary_embedding_meta=self.rotary_embedding._matformer_metadata
        if not self.config.interpretability_friendly:
            # Initialize fast attention kernel
            if is_hybrid: #Initialize two kernels
                self.attn_kernel_causal= registry.create(
                    'attention',
                    'normal',
                    nheads=self.config.num_attention_heads,
                    head_dim=self.head_dim,
                    num_key_value_heads=self.config.num_key_value_heads,
                    is_causal=True,
                    sliding_window=self.config.sliding_window_size,
                    positional_encoding=self.config.positional_encoding,
                    attn_logit_softcap=self.config.attn_logit_softcap,
                    cache=cache,
                    device=device
                )
                self.attn_kernel_noncausal= registry.create(
                    'attention',
                    'normal',
                    nheads=self.config.num_attention_heads,
                    head_dim=self.head_dim,
                    num_key_value_heads=self.config.num_key_value_heads,
                    is_causal=False,
                    sliding_window=self.config.sliding_window_size,
                    positional_encoding=self.config.positional_encoding,
                    attn_logit_softcap=self.config.attn_logit_softcap,
                    cache=cache,
                    device=device
                )
                self.causal_kernel_meta=self.attn_kernel_causal._matformer_metadata
                self.noncausal_kernel_meta=self.attn_kernel_noncausal._matformer_metadata
            else: #Normal branch
                self.attn_kernel = registry.create(
                    'attention',
                    'normal',
                    nheads=self.config.num_attention_heads,
                    head_dim=self.head_dim,
                    num_key_value_heads=self.config.num_key_value_heads,
                    is_causal=is_causal,
                    sliding_window=self.config.sliding_window_size,
                    positional_encoding=self.config.positional_encoding,
                    attn_logit_softcap=self.config.attn_logit_softcap,
                    cache=cache,
                    device=device
                )
                # Get kernel metadata
                self.kernel_meta = self.attn_kernel._matformer_metadata

        if not self.force_separed_qkv and (not self.config.is_cross_attention or self.qkv_samedim) and self.n_groups == 1: # Packed qkv projection for efficiency
            self.packed_proj=registry.create('linear','linear',in_features=self.config.q_dim, out_features=3*self.q_inner_dim,bias=self.config.qkv_bias)
        else:
            self.q_proj = registry.create('linear','linear', in_features=self.config.q_dim, out_features=self.q_inner_dim, bias=self.config.qkv_bias)
            self.k_proj=registry.create('linear','linear',in_features=self.config.k_dim, out_features=self.head_dim * self.config.num_key_value_heads,bias=self.config.qkv_bias)
            self.v_proj=registry.create('linear','linear',in_features=self.config.v_dim, out_features=self.head_dim * self.config.num_key_value_heads,bias=self.config.qkv_bias)

        self.out_proj = registry.create('linear','linear', in_features=self.q_inner_dim, out_features=self.config.q_dim, bias=self.config.attn_output_bias)

    def _qk_norm(self, q, k):
        """q, k: raw tensors shaped (..., H, D). Applies whichever of q_norm/k_norm is configured."""
        if self.config.qk_norm_full_width and (self.q_norm is not None or self.k_norm is not None):
            # full-width norms expect the flattened (..., H*D) layout (OLMo-2)
            flat = lambda x: x.flatten(-2)
            unflat = lambda x, h: x.unflatten(-1, (h, x.shape[-1] // h))
            if self.q_norm is not None:
                q = unflat(self.q_norm(NormalTensor(tensor=flat(q))).tensor, q.shape[-2])
            if self.k_norm is not None:
                k = unflat(self.k_norm(NormalTensor(tensor=flat(k))).tensor, k.shape[-2])
            return q, k
        if self.q_norm is not None: q = self.q_norm(NormalTensor(tensor=q)).tensor
        if self.k_norm is not None: k = self.k_norm(NormalTensor(tensor=k)).tensor
        return q, k

    def _gate(self, attn_out, query_tensor):
        """attn_out: (..., H, D). query_tensor: (..., q_dim), the same input that produced Q."""
        if self.gate_proj is None:
            return attn_out
        gate = torch.sigmoid(self.gate_proj(NormalTensor(tensor=query_tensor)).tensor).unflatten(-1, attn_out.shape[-2:])
        return attn_out * gate
    @staticmethod
    def _pack_qkv(q, k, v):
        normalize = lambda s: s.translate(str.maketrans('', '', '?B'))
        assert normalize(q.tensor_order) == normalize(k.tensor_order) == normalize(v.tensor_order), "QKV must have same tensor order"
        assert q.tensor.shape == k.tensor.shape == v.tensor.shape, "_pack_qkv requires Q, K, V to have identical shapes (MHA only). Use separate projections for GQA."
        current_order_norm = normalize(q.tensor_order)

        if current_order_norm == "HSD":
            # [?, H, S, D] => [?, S, H, D]
            q_t = q.tensor.transpose(1, 2)
            k_t = k.tensor.transpose(1, 2)
            v_t = v.tensor.transpose(1, 2)
            #  [?, S, 3, H, D]
            packed_tensor = torch.stack([q_t, k_t, v_t], dim=-3)
            new_order = '?S3HD'
        elif current_order_norm == "SHD":
            # [?, S, 3, H, D]
            packed_tensor = torch.stack([q.tensor, k.tensor, v.tensor], dim=-3)
            new_order = '?S3HD'
        else:
            raise ValueError(f"Unsupported tensor order: {q.tensor_order}")

        return replace(q, tensor=packed_tensor, tensor_order=new_order)

    @staticmethod
    def _unpack_qkv(qkv_packed):
            # S3HD => ?SHD
            q_t, k_t, v_t = qkv_packed.tensor.unbind(dim=-3)
            order = qkv_packed.tensor_order.replace('3', '')
            return (replace(qkv_packed, tensor=q_t, tensor_order=order),
                    replace(qkv_packed, tensor=k_t, tensor_order=order),
                    replace(qkv_packed, tensor=v_t, tensor_order=order))

    @staticmethod
    def _seq_geometry(tensor):
        """(max_seq_len, cu_seqlens) of an input: the pack boundaries when unpadded,
        otherwise the sequence length of its padded/plain grid."""
        if isinstance(tensor, UnpaddedTensor):
            return tensor.max_seq_len, tensor.cu_seqlens
        shape = tensor.shape
        return (shape[0] if len(shape) == 2 else shape[1]), None

    @staticmethod
    def _transpose_for_kernel(tensor, wanted_from_kernel):
        """ Currently this helper function only supports conversion from and to flash style, sdpa style
            BHSD
            BSHD
            that is, when transposing dimensions 1 and 2
            Every other conversion wil raise an error
            Letters 'B' and '?' are trated as if they are the same, it represents equivalence of batching/unbatching in Matformer's jargon.
            The tensorDC must have a 'tensor_order' attribute, if not the conversion will fail too
        """
        if not isinstance(tensor, TensorDC) or not hasattr(tensor, 'tensor_order'):
            raise Exception
        normalize = lambda s: s.translate(str.maketrans('', '', '?B'))
        wanted = normalize(wanted_from_kernel)
        current = normalize(tensor.tensor_order)
        if current == wanted:
            return tensor  # No change required, great!
        if (current=='SHD' and wanted=='HSD') or (current=='HSD' and wanted == 'SHD'):
            # It can be transposed
            return replace(tensor, tensor=tensor.tensor.transpose(1,2), tensor_order='?' + wanted)
        print(f"Current: {current} Wanted: {wanted}")
        raise Exception

    """
    The properties defined below serves for Mechanistic Interpretability purposes: they provide
    diverse views over the weight that facilitates MI. Can be ignored for a training model or in general
    if MI is not at stake. The MI part can be recognized by the suffix "_MI"
    """
    @property
    def W_Q_MI(self):
        if hasattr(self, 'packed_proj'):
            return self.packed_proj.inner.weight.view(3, self.config.num_attention_heads, self.head_dim, -1)[0].transpose(-1, -2)
        return self.q_proj.inner.weight.view(self.config.num_attention_heads, self.head_dim, -1).transpose(-1, -2)

    @property
    def W_K_MI(self):
        if hasattr(self, 'packed_proj'):
            return self.packed_proj.inner.weight.view(3, self.config.num_attention_heads, self.head_dim, -1)[1].transpose(-1, -2)
        return self.k_proj.inner.weight.view(self.config.num_key_value_heads, self.head_dim, -1).transpose(-1, -2)

    @property
    def W_V_MI(self):
        if hasattr(self, 'packed_proj'):
            return self.packed_proj.inner.weight.view(3, self.config.num_attention_heads, self.head_dim, -1)[2].transpose(-1, -2)
        return self.v_proj.inner.weight.view(self.config.num_key_value_heads, self.head_dim, -1).transpose(-1, -2)

    @property
    def W_O_MI(self):
        return self.out_proj.inner.weight.view(-1, self.config.num_attention_heads, self.head_dim).permute(1, 2, 0)
    @property
    def B_Q_MI(self):
        if not self.config.qkv_bias: return 0
        if hasattr(self, 'packed_proj'):
            return self.packed_proj.inner.bias.view(3, self.config.num_attention_heads, self.head_dim)[0]
        return self.q_proj.inner.bias.view(self.config.num_attention_heads, self.head_dim)

    @property
    def B_K_MI(self):
        if not self.config.qkv_bias: return 0
        if hasattr(self, 'packed_proj'):
            return self.packed_proj.inner.bias.view(3, self.config.num_attention_heads, self.head_dim)[1]
        return self.k_proj.inner.bias.view(self.config.num_key_value_heads, self.head_dim)

    @property
    def B_V_MI(self):
        if not self.config.qkv_bias: return 0
        if hasattr(self, 'packed_proj'):
            return self.packed_proj.inner.bias.view(3, self.config.num_attention_heads, self.head_dim)[2]
        return self.v_proj.inner.bias.view(self.config.num_key_value_heads, self.head_dim)

    @property
    def B_O_MI(self):
        if not self.config.attn_output_bias: return 0
        return self.out_proj.bias  # shape: [d_model]

    def forward(self, query_input, original_x=None, key_input=None, value_input=None, document_mask=None, is_causal=None, interpretability=False):
        if is_causal is None:
            is_causal=self.is_causal
        # 1. Extract max seq len and cu_seqlens (if appliable), convert to Normal Tensor in case a pytorch tensor is passed.
        cu_seqlens=None
        if isinstance(query_input, UnpaddedTensor):
            max_seq_len = query_input.max_seq_len # This variable can be used, for example, to get the correct RoPe cos and sin from cache, but in general is useful
            cu_seqlens=query_input.cu_seqlens_device() # Used for unpadding
        elif isinstance(query_input, PaddedTensor):
            max_seq_len = query_input.shape[1]
        else: #normal tensor
            shape=query_input.shape
            if len(shape)==2:
                max_seq_len = query_input.shape[0]
            else:
                max_seq_len=query_input.shape[1]

        if key_input is None:
            key_input = query_input
        if value_input is None:
            value_input = query_input

        if self.config.interpretability_friendly or interpretability:
            return self.interpretability_friendly_forward(query_input,original_x,key_input,value_input,document_mask,cu_seqlens,max_seq_len, is_causal)
        else:
            return self.normal_forward(query_input,original_x,key_input,value_input,document_mask,cu_seqlens,max_seq_len, is_causal)
    def interpretability_friendly_forward(self, query_input, original_x=None, key_input=None, value_input=None, document_mask=None, cu_seqlens=None, max_seq_len=None, is_causal=False):
            """
            A modified forward that is optimal for working on mechanistic interpretability.
            """
            # 1. For now, the MI version works only with PaddedTensors
            assert isinstance(query_input, TensorDC)
            wasUnpadded = True if isinstance(query_input, UnpaddedTensor) else False
            q_pad, k_pad, v_pad = query_input.pad(), key_input.pad(), value_input.pad()
            q, k, v = q_pad.tensor, k_pad.tensor, v_pad.tensor
            padding_mask = k_pad.padding_mask

            # 2. Query, key and value vectors using the MI-friendly  weights
            q = einops.einsum(q, self.W_Q_MI, "batch position d_model, n_heads d_model d_head -> batch position n_heads d_head") + self.B_Q_MI
            k = einops.einsum(k, self.W_K_MI, "batch position d_model, kv_heads d_model d_head -> batch position kv_heads d_head") + self.B_K_MI
            v = einops.einsum(v, self.W_V_MI, "batch position d_model, kv_heads d_model d_head -> batch position kv_heads d_head") + self.B_V_MI
            q, k, v = self._run_socket('q', q), self._run_socket('k', k), self._run_socket('v', v)
            q, k = self._qk_norm(q, k)
            if 'rope' in self.config.positional_encoding:
                #cos, sin = self.cache.get_rotary_cos_sin(q.shape[1], self.head_dim, q.device, q.dtype)
                cos, sin = self.cache.get_rotary_cos_sin(
                    max_seq_len, self.head_dim, query_input.device, query_input.dtype,
                    theta=self.config.rope_theta,  # if not already threaded through, add this too
                    partial_rotary_factor=self.config.partial_rotary_factor,
                    scaling_type=self.config.rope_scaling_type,
                    scaling_factor=self.config.rope_scaling_factor,
                    original_max_position_embeddings=self.config.rope_scaling_original_max_position_embeddings,
                    low_freq_factor=self.config.rope_scaling_low_freq_factor,
                    high_freq_factor=self.config.rope_scaling_high_freq_factor,
                    beta_fast=self.config.rope_scaling_beta_fast,
                    beta_slow=self.config.rope_scaling_beta_slow,
                    attention_factor=self.config.rope_scaling_attention_factor,
                )
                cos, sin = cos[None, :, None, :], sin[None, :, None, :]
                if self.config.rope_interleaved:
                    rotate = lambda x: torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).flatten(-2)
                else:
                    rotate = lambda x: torch.cat([-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]], dim=-1)
                q, k = q * cos + rotate(q) * sin, k * cos + rotate(k) * sin
            if self.n_groups > 1: # Expanding GQA to MHA
                k = k.repeat_interleave(self.n_groups, dim=2)
                v = v.repeat_interleave(self.n_groups, dim=2)
            # 3. Computation of attention scores
            attn_scores = einops.einsum(q, k, "batch posQ n_heads d_head, batch posK n_heads d_head -> batch n_heads posQ posK")
            attn_scores = (attn_scores / self.head_dim**0.5) # Standard sqrt(d) scaling
            if self.config.attn_logit_softcap:
                attn_scores = self.config.attn_logit_softcap * torch.tanh(
                    attn_scores / self.config.attn_logit_softcap)
            if 'alibi' in self.config.positional_encoding:
                attn_scores = attn_scores + self.cache.get_alibi_bias(q.shape[1], k.shape[1], self.config.num_attention_heads, q.device, q.dtype)
            # Padding / causal / sliding-window / document masking (mirrors SDPAKernel._build_attention_mask)
            if padding_mask is not None or is_causal or self.config.sliding_window_size is not None or document_mask is not None:
                L, S = q.shape[1], k.shape[1]
                mask_bool = torch.ones(q.shape[0], L, S, dtype=torch.bool, device=q.device)
                if padding_mask is not None:
                    mask_bool &= ~padding_mask.unsqueeze(1)
                if is_causal or self.config.sliding_window_size is not None:
                    q_idx, k_idx = torch.arange(L, device=q.device).view(L, 1), torch.arange(S, device=q.device).view(1, S)
                    if is_causal:
                        mask_bool &= (q_idx >= k_idx)
                    if self.config.sliding_window_size is not None:
                        # HF / flash-attention semantics: attend to |q-k| < window (exclusive)
                        mask_bool &= (torch.abs(q_idx - k_idx) < self.config.sliding_window_size)
                if document_mask is not None:
                    mask_bool &= (document_mask[:, :L, None] == document_mask[:, None, :S])
                attn_scores = attn_scores.masked_fill(~mask_bool.unsqueeze(1), float('-inf'))
            attn_scores = self._run_socket('attn_scores', attn_scores)

            # 4. Softmax
            attn_pattern = attn_scores.softmax(-1)
            attn_pattern = self._run_socket('attn_pattern', attn_pattern)

            # 5. Weighed sums of values according to attention pattern
            z = einops.einsum(v, attn_pattern, "batch posK n_heads d_head, batch n_heads posQ posK -> batch posQ n_heads d_head")
            z = self._run_socket('z', z)
            z = self._gate(z, q_pad.tensor)
            z = self._run_socket('gate', z)
            # 6. Output calculation
            attn_out = einops.einsum(z, self.W_O_MI, "batch posQ n_heads d_head, n_heads d_head d_model -> batch posQ d_model") + self.B_O_MI
            attn_out = self._run_socket('attn_out', attn_out)

            # 7. Reinserting into tensorDC
            attn_out = replace(query_input, tensor=attn_out)

            # (optional) repadding
            if wasUnpadded:
                attn_out = attn_out.unpad()
            return attn_out
    def normal_forward(self, query_input, original_x=None, key_input=None, value_input=None, document_mask=None, cu_seqlens=None,max_seq_len=None, is_causal=False):
        """
        Input Tensors:
        query_input: (Batch, Seqlen, Qdim)
        key_input: (Batch, Seqlen, Kdim) [If omitted, self-attention]
        value_input: (Batch, Seqlen, Vdim) [If omitted, self-attention]
        Output:
        (Batch, Seqlen, Qdim) # Qdim is the query dimension as well as the output.
        """
        if not self.is_hybrid:
            kernel_meta=self.kernel_meta
            attn_kernel=self.attn_kernel
        else:
            if is_causal:
                kernel_meta=self.causal_kernel_meta
                attn_kernel=self.attn_kernel_causal
            else:
                kernel_meta=self.noncausal_kernel_meta
                attn_kernel=self.attn_kernel_noncausal

        supports_unpadding = kernel_meta.get('supports_unpadding', False)
        if 'rope' in self.config.positional_encoding:
            supports_unpadding = supports_unpadding and self.rotary_embedding_meta.get('supports_unpadding', False) # If either the RoPe or the attn kernel do not support unpadding, sequence is repadded
        supports_packed_qkv = kernel_meta.get('supports_packed_qkv', False)
        rope_supports_packed = self.rotary_embedding_meta.get('supports_packed_qkv', False) if  'rope' in self.config.positional_encoding else True

        # Set defaults for self-attention

        if key_input is None:
            key_input = query_input
        if value_input is None:
            value_input = query_input
        # Decided before any repadding: pad() creates distinct objects even when both
        # inputs describe the very same tokens.
        key_is_query = key_input is query_input

        # If the attention implementation does not support unpadding, eventually unpadded sequences must be padded
        repadded = False
        auto_unpadded = False
        if not supports_unpadding:
            if isinstance(query_input, UnpaddedTensor):
                repadded = True
                query_input = query_input.pad()
                key_input = key_input.pad()
                value_input = value_input.pad()
        elif (isinstance(query_input, PaddedTensor) and query_input.padding_mask is not None
              and not kernel_meta.get('supports_padding_mask', False)):
            # Kernel supports unpadding but, in its padded path, has no way to apply an explicit
            # padding mask (e.g. flash-attn's padded API): padding must be removed. Kernels that
            # do mask padding themselves (sdpa) keep the batch padded and skip the round trip.
            auto_unpadded = True
            # Self-attention: key/value ARE query (same object) -> unpad once, not three times
            # (each unpad() runs nonzero + cumsum + gathers over the full hidden states)
            query_input = query_input.unpad()
            if key_input is query_input and value_input is query_input:
                key_input = value_input = query_input
            else:
                key_input = key_input.unpad()
                value_input = value_input.unpad()
            cu_seqlens = query_input.cu_seqlens_device()
            max_seq_len = query_input.max_seq_len



        # Projecting (eventually, packed projection)
        clamp = self.config.qkv_clamp
        if self.qkv_samedim and not self.config.is_cross_attention and not self.force_separed_qkv and self.n_groups == 1:
            qkv_projected = NormalTensor(tensor=self.packed_proj(query_input).tensor, tensor_order='?S(3*D)')
            if clamp:  # elementwise, so clamping the packed tensor == clamping each of q/k/v
                qkv_projected = replace(qkv_projected, tensor=qkv_projected.tensor.clamp(-clamp, clamp))
            q, k, v = None, None, None
        elif self.qkv_samedim and self.config.is_cross_attention and not self.force_separed_qkv and self.n_groups == 1:
            qkv_projected = None
            w = torch.chunk(self.packed_proj.inner.weight, 3, dim=0)
            b = torch.chunk(self.packed_proj.inner.bias, 3, dim=0) if self.config.qkv_bias else (None, None, None)
            q = NormalTensor(tensor=F.linear(query_input.tensor, w[0], b[0]),tensor_order='?SD')
            k = NormalTensor(tensor=F.linear(key_input.tensor, w[1], b[1]),tensor_order='?SD')
            v = NormalTensor(tensor=F.linear(value_input.tensor, w[2], b[2]),tensor_order='?SD')
        else:
            qkv_projected = None
            q = NormalTensor(tensor=self.q_proj(query_input).tensor,tensor_order='?SD')
            k = NormalTensor(tensor=self.k_proj(key_input).tensor,tensor_order='?SD')
            v = NormalTensor(tensor=self.v_proj(value_input).tensor,tensor_order='?SD')
            if clamp:
                q, k, v = (replace(t, tensor=t.tensor.clamp(-clamp, clamp)) for t in (q, k, v))

        # 3. Heads creation
        if qkv_projected is not None:
            qkv_projected = replace(qkv_projected, tensor=qkv_projected.tensor.unflatten(-1, [3, self.config.num_attention_heads, self.head_dim]), tensor_order='?S3HD')
        else:
            q = replace(q, tensor=q.tensor.unflatten(-1, [self.config.num_attention_heads, self.head_dim]),tensor_order='?SHD')
            k = replace(k, tensor=k.tensor.unflatten(-1, [self.config.num_key_value_heads, self.head_dim]),tensor_order='?SHD')
            v = replace(v, tensor=v.tensor.unflatten(-1, [self.config.num_key_value_heads, self.head_dim]),tensor_order='?SHD')
            q_t, k_t = self._qk_norm(q.tensor, k.tensor)
            q, k = replace(q, tensor=q_t), replace(k, tensor=k_t)
        # 3b. (facultative) Apply RoPe
        if 'rope' in self.config.positional_encoding:
            repack_after_rope = False
            if qkv_projected is not None and (not rope_supports_packed or not key_is_query): # Unpack if RoPe doesn't support packed qkv, or if keys need their own positions
                q, k, v = self._unpack_qkv(qkv_projected)
                qkv_projected = None
                repack_after_rope = supports_packed_qkv  #Repack only if supported by the attention kernel
            elif qkv_projected is not None and rope_supports_packed:
                assert qkv_projected.tensor_order == self.rotary_embedding_meta['tensor_order_qkv_packed_input']

            if qkv_projected is None:  #non-packed branch, adapt to requested tensor order
                q = self._transpose_for_kernel(q, self.rotary_embedding_meta['tensor_order_input'])
                k = self._transpose_for_kernel(k, self.rotary_embedding_meta['tensor_order_input'])

            # 1. Get sin and cos from cache
            rope_kwargs = dict(
                theta=self.config.rope_theta,
                partial_rotary_factor=self.config.partial_rotary_factor,
                scaling_type=self.config.rope_scaling_type,
                scaling_factor=self.config.rope_scaling_factor,
                original_max_position_embeddings=self.config.rope_scaling_original_max_position_embeddings,
                low_freq_factor=self.config.rope_scaling_low_freq_factor,
                high_freq_factor=self.config.rope_scaling_high_freq_factor,
                beta_fast=self.config.rope_scaling_beta_fast,
                beta_slow=self.config.rope_scaling_beta_slow,
                attention_factor=self.config.rope_scaling_attention_factor,
            )
            cos, sin = self.cache.get_rotary_cos_sin(
                max_seq_len, self.head_dim, query_input.device, query_input.dtype, **rope_kwargs
            )
            # 2. Rotate query and keys
            # cu_seqlens refers to the packed layout; if the sequence was repadded
            # (kernel without unpadding support), positions come from the padded grid
            # instead — passing the stale cu_seqlens would rotate in padding tokens.
            rope_cu_seqlens = None if repadded else cu_seqlens
            if key_is_query:
                qkv_t, q_t, k_t = self.rotary_embedding(
                    qkv=qkv_projected.tensor if qkv_projected is not None else None,
                    q=q.tensor if q is not None else None,
                    k=k.tensor if k is not None else None,
                    cos=cos, sin=sin,
                    cu_seqlens=rope_cu_seqlens,
                    max_seq_len=None if repadded else max_seq_len
                )
            else:
                # Cross-attention: keys sit on their own grid (own length, own segment
                # boundaries), so they are rotated with their own cos/sin and positions.
                key_max_seq_len, key_cu_seqlens = self._seq_geometry(key_input)
                key_cos, key_sin = self.cache.get_rotary_cos_sin(
                    key_max_seq_len, self.head_dim, key_input.device, key_input.dtype, **rope_kwargs
                )
                _, q_t, _ = self.rotary_embedding(
                    q=q.tensor, k=q.tensor, cos=cos, sin=sin,
                    cu_seqlens=rope_cu_seqlens,
                    max_seq_len=None if repadded else max_seq_len
                )
                _, _, k_t = self.rotary_embedding(
                    q=k.tensor, k=k.tensor, cos=key_cos, sin=key_sin,
                    cu_seqlens=None if repadded else key_cu_seqlens,
                    max_seq_len=key_max_seq_len
                )
                qkv_t = None
            if qkv_projected is not None:
                qkv_projected = replace(qkv_projected, tensor=qkv_t, tensor_order=self.rotary_embedding_meta['tensor_order_qkv_packed_output'])
            else:
                q = replace(q, tensor=q_t, tensor_order=self.rotary_embedding_meta['tensor_order_output'])
                k = replace(k, tensor=k_t, tensor_order=self.rotary_embedding_meta['tensor_order_output'])


            # Repack if needed for attention kernel
            if repack_after_rope:
                qkv_projected = self._pack_qkv(q, k, v)
                q, k, v = None, None, None
        kernel_input_order = kernel_meta.get('tensor_order_input', '?SHD')
        tensor_order_qkv_packed = kernel_meta.get('tensor_order_qkv_packed_input', None)

        if qkv_projected is not None and not supports_packed_qkv:
            q, k, v = self._unpack_qkv(qkv_projected)
            qkv_projected = None

        if qkv_projected is not None:
            qkv_projected = self._transpose_for_kernel(qkv_projected, tensor_order_qkv_packed if tensor_order_qkv_packed is not None else kernel_input_order)
        else:
            q = self._transpose_for_kernel(q, kernel_input_order)
            k = self._transpose_for_kernel(k, kernel_input_order)
            v = self._transpose_for_kernel(v, kernel_input_order)

        if self.n_groups > 1 and not kernel_meta.get('supports_gqa', False):
            # Kernel doesn't support GQA natively: expand K/V to full nheads
            dim = 1 if k.tensor_order.replace('?', 'B') == 'BHSD' else 2
            k = replace(k, tensor=k.tensor.repeat_interleave(self.n_groups, dim=dim))
            v = replace(v, tensor=v.tensor.repeat_interleave(self.n_groups, dim=dim))
        # Attention computation
        attn_output = attn_kernel(
            qkv=qkv_projected.tensor if qkv_projected is not None else None,
            q=q.tensor if q is not None else None,
            k=k.tensor if k is not None else None,
            v=v.tensor if v is not None else None,
            query_input=query_input, key_input=key_input,
            original_x=original_x
        )

        # Transpose from kernel's output format to expected format (B, S, H, Hd)
        kernel_output_order = kernel_meta.get('tensor_order_output', '?SHD')
        attn_output = NormalTensor(tensor=attn_output, tensor_order=kernel_output_order)
        attn_output = self._transpose_for_kernel(attn_output, '?SHD')

        # Post-attention stuff
        attn_output = self._gate(attn_output.tensor, query_input.tensor)
        attn_output = attn_output.flatten(-2)  # (B, S, H, Hd) -> (B, S, D)
        output_tensor = self.out_proj(NormalTensor(tensor=attn_output)).tensor

        # Handle output based on input type
        if hasattr(query_input, 'tensor'):
            query_input = replace(query_input, tensor=output_tensor)
        else:
            query_input = output_tensor

        if repadded:
            query_input = query_input.unpad()
        elif auto_unpadded:
            query_input = query_input.pad()

        return query_input
