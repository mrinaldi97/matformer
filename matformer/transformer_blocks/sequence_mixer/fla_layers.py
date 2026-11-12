from matformer.matformer_registry import registry
from dataclasses import dataclass, fields, asdict, field, replace
from typing import Optional, Literal, Union, Any, Dict, List
from matformer.dataclasses.matformer_config import MatformerConfig
from matformer.dataclasses.tensors_dataclasses import TensorDC, NormalTensor, PaddedTensor, UnpaddedTensor
from torch import nn
import torch
import torch.nn.functional as F
from fla.layers.gated_deltanet import GatedDeltaNet
from fla.layers.kda import KimiDeltaAttention
from fla.layers.gdn2 import GatedDeltaNet2
from fla.layers.mamba3 import Mamba3
from fla.layers.gla import GatedLinearAttention
from fla.layers.rwkv7 import RWKV7Attention
from fla.layers.delta_net import DeltaNet
from fla.layers.hgrn2 import HGRN2Attention
from matformer.matformer_module import MatformerModule
from copy import deepcopy
from warnings import warn
import functools
import inspect

@dataclass
class BidirectionalConfig:
    """Anticausal (future-context) branch knobs.

    Second-class by design: bidirectionality is an experiment, so every knob lives here and the
    feature can be removed by deleting this class plus the bwd branch of FLASequenceMixer,
    without touching the causal path or the fla wrappers.
    """
    allow_bidirectionalize: bool = True
    bidirectional_type: Literal["cnn", "fla"] = "fla"
    bidirectional_merge_type: Literal["sum", "gated_sum", "concat_proj", "gated_concat_proj"] = "concat_proj"
    # The reverse branch is a second full scan, so on the mixers whose per-token cost is set by
    # the projection width rather than by the head split (Mamba-3: intermediate = expand *
    # hidden_size, num_heads derived) head_dim/num_heads alone shrink nothing. The next three
    # knobs are inherited (None) unless set, and are skipped on mixers that lack the field.
    bidirectional_head_dim: int = 64
    bidirectional_num_heads: int = 6
    bidirectional_expand: Optional[int] = None
    bidirectional_state_size: Optional[int] = None
    bidirectional_is_mimo: Optional[bool] = None
    # Scan both directions in a single varlen call instead of two sequential ones. Only applies
    # to the mirrored branch (reverse_mixer/cnn keep their own path).
    bidirectional_batched: bool = False
    # Borrow the anticausal branch from the registry: any sequence_mixer variant (a cheap
    # gla/hgrn2/delta_net/rwkv7, or even "attention") run causal on the reversed sequence, which
    # is exactly "reads the future". Wins over bidirectional_type=cnn when the future context
    # needed is longer than a conv receptive field, and over the mirrored branch when the main
    # mixer is too expensive to scan twice.
    reverse_mixer: Optional[str] = None
    reverse_mixer_config: Dict[str, Any] = field(default_factory=dict)
    cnn_n_layers: int = 8
    cnn_kernel_size: int = 3
    cnn_dilation_growth: Literal["exponential", "linear", "none"] = "exponential"


@dataclass
class FLAConfig(MatformerConfig, BidirectionalConfig):
    positional_encoding: Optional[
        Union[
            List[Literal["alibi", "rope", "sinusoidal", "nope", "learnable"]],
            Literal["alibi", "rope", "sinusoidal", "nope", "learnable"],
        ]
    ] = "alibi"
    is_causal: bool = True


@registry.register_dataclass("sequence_mixer", "gated_deltanet", default=True)
@dataclass
class GatedDeltaNetConfig(FLAConfig):
    expand_v: float = 2.0
    head_dim: int = 256
    num_heads: int = 6
    num_v_heads: Optional[int] = None
    mode: Literal["chunk", "fused_recurrent"] = "chunk"
    use_gate: bool = True
    use_short_conv: bool = True
    allow_neg_eigval: bool = False
    conv_size: int = 4
    conv_bias: bool = False
    norm_eps: float = 1e-5


@registry.register_dataclass("sequence_mixer", "kda")
@dataclass
class KDAConfig(FLAConfig):
    expand_v: float = 1.0
    head_dim: int = 128
    num_heads: int = 16
    num_v_heads: Optional[int] = None
    mode: Literal["chunk", "fused_recurrent"] = "chunk"
    use_short_conv: bool = True
    allow_neg_eigval: bool = False
    safe_gate: bool = False
    lower_bound: Optional[float] = None
    conv_size: int = 4
    conv_bias: bool = False


class DilatedConvStack(nn.Module):
    """Anticausal depthwise dilated conv stack over a packed (N, D) sequence.

    A Conv1d(k, dilation=d) fed with right-only padding computes out[i] = sum_j w_j * h[i + j*d]:
    the current position plus (k-1)*d purely future taps. The nn.Conv1d modules are kept, so
    checkpoints still load, but the sum is evaluated as masked gathers, which fixes two things
    and one trap: taps no longer reach into the neighbouring packed document (the padded conv
    did that silently), no F.pad is allocated per layer, and the whole thing is gather/add on a
    dynamic length, so torch.compile stops recompiling every time the pack length changes.
    RF = 1 + (k-1) * sum(dilations).
    """
    COMPILE = True

    def __init__(self, hidden_size, n_layers=8, kernel_size=3, dilation_growth="exponential", residual=True):
        super().__init__()
        dilations = {
            "exponential": [2 ** i for i in range(n_layers)],
            "linear": list(range(1, n_layers + 1)),
            "none": [1] * n_layers,
        }[dilation_growth]

        self.dilations = dilations
        self.kernel_size = kernel_size
        self.residual = residual
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_size, hidden_size, kernel_size, dilation=d, groups=hidden_size, bias=False)
            for d in dilations
        ])
        if self.COMPILE:
            # dynamic: pack lengths change every batch, and a shape guard per length is what made
            # this branch recompile once per layer per step.
            self.forward = torch.compile(self.forward, dynamic=True)

    @property
    def receptive_field(self):
        return 1 + (self.kernel_size - 1) * sum(self.dilations)

    def forward(self, x, segment_ids):
        n = x.shape[0]
        pos = torch.arange(n, device=x.device)
        h = x
        for conv, d in zip(self.convs, self.dilations):
            w = conv.weight.reshape(x.shape[-1], self.kernel_size)
            y = w[:, 0] * h
            for j in range(1, self.kernel_size):
                reach = pos + j * d
                tapped = reach.clamp(max=n - 1)
                same_doc = (reach < n) & segment_ids[tapped].eq(segment_ids)
                y = y + w[:, j] * h[tapped].where(same_doc.unsqueeze(-1), 0.)
            h = h + y if self.residual else y
        return h


class BackwardBranch(nn.Module):
    def run(self, x, run_fn): raise NotImplementedError


class ReverseBranch(BackwardBranch):
    def __init__(self, fla_cls, **kwargs):
        super().__init__()
        self.mixer = fla_cls(**kwargs)

    def run(self, x, run_fn):
        # Reverse inside each real sequence, run the mixer, reverse the result back. The
        # merge in FLASequenceMixer.forward happens in the caller's layout, so a padded
        # input must come back padded (the kernel only ever sees the packed form).
        packed = x if getattr(x, 'isUnpadded', False) else x.unpad()
        out = run_fn(self.mixer, packed.reverse(), False).reverse()
        return out if getattr(x, 'isUnpadded', False) else out.pad()


class RegistryReverseBranch(BackwardBranch):
    """Anticausal branch built from any registry sequence_mixer, run causal on the reversed
    sequence. The sub-mixer is created with allow_bidirectionalize off, which is what keeps the
    recursion finite; it also handles its own packing, so it only ever sees the packed form."""

    def __init__(self, mixer):
        super().__init__()
        self.mixer = mixer

    def run(self, x, run_fn):
        packed = x if getattr(x, 'isUnpadded', False) else x.unpad()
        out = self.mixer(packed.reverse())
        return out if getattr(x, 'isUnpadded', False) else out.pad()


class CNNAnticausalBranch(BackwardBranch):
    def __init__(self, hidden_size, n_layers, kernel_size, dilation_growth):
        super().__init__()
        self.stack = DilatedConvStack(hidden_size, n_layers=n_layers, kernel_size=kernel_size,
                                       dilation_growth=dilation_growth)

    def run(self, x, run_fn):
        u = x if getattr(x, 'isUnpadded', False) else x.unpad()
        return replace(u, tensor=self.stack(u.tensor, u.segment_ids()))


class FLASequenceMixer(MatformerModule):
    inner: "transparent"
    fla_cls = None

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _accepted_params(fla_cls):
        # fla layers are inconsistent: most swallow unknown kwargs, but some (Mamba-3) have a
        # closed signature. Returning None means "pass everything through".
        params = inspect.signature(fla_cls.__init__).parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return None
        return frozenset(params)

    @classmethod
    def _layer_kwargs(cls, config, **runtime):
        kwargs = {**{k: v for k, v in config.asdict().items() if k != "name"}, **runtime}
        accepted = cls._accepted_params(cls.fla_cls)
        return {k: v for k, v in kwargs.items() if accepted is None or k in accepted}

    @staticmethod
    def _override(config, name, value):
        # Not every mixer exposes every knob (Mamba-3 derives num_heads from expand *
        # hidden_size, so it has no num_heads field); a missing field is simply skipped.
        if any(f.name == name for f in fields(config)):
            setattr(config, name, value)

    @classmethod
    def _branch_config(cls, config, hidden_size):
        """Config for a mixer borrowed from the registry: keep every field the target class also
        declares, patch reverse_mixer_config on top, and disable nesting."""
        target = registry.get_definition("sequence_mixer", config.reverse_mixer)
        if target is None:
            raise ValueError(f"sequence_mixer '{config.reverse_mixer}' has no config dataclass, "
                             f"so it cannot be used as reverse_mixer")
        valid = {f.name for f in fields(target)}
        sub = target.from_dict({**{k: v for k, v in config.asdict().items() if k in valid},
                                **config.reverse_mixer_config})
        cls._override(sub, "allow_bidirectionalize", False)
        # The attention family reads its width from q/k/v_dim, normally stamped by the model.
        for dim in ("q_dim", "k_dim", "v_dim"):
            if any(f.name == dim for f in fields(sub)) and getattr(sub, dim) is None:
                setattr(sub, dim, hidden_size)
        return sub

    def _build_backward_branch(self, config, hidden_size, layer_idx):
        if config.bidirectional_type == "cnn":
            return CNNAnticausalBranch(hidden_size, config.cnn_n_layers,
                                       config.cnn_kernel_size, config.cnn_dilation_growth)
        if config.reverse_mixer:
            return RegistryReverseBranch(registry.create(
                "sequence_mixer", config.reverse_mixer,
                config=self._branch_config(config, hidden_size), hidden_size=hidden_size,
                layer_idx=layer_idx, is_causal=True))
        bwd_config = deepcopy(config)
        self._override(bwd_config, "head_dim", config.bidirectional_head_dim)
        self._override(bwd_config, "num_heads", config.bidirectional_num_heads)
        for field_name, value in (
            ("expand", config.bidirectional_expand),
            ("state_size", config.bidirectional_state_size),
            ("is_mimo", config.bidirectional_is_mimo),
        ):
            if value is not None:
                self._override(bwd_config, field_name, value)
        return ReverseBranch(self.fla_cls, **self._layer_kwargs(
            bwd_config, hidden_size=hidden_size, layer_idx=layer_idx))

    def __init__(self, config, hidden_size, device=None, layer_idx=None, cache=None,
                 is_causal=False, is_hybrid=False):
        super().__init__()
        self.inner = self.fla_cls(**self._layer_kwargs(
            config, hidden_size=hidden_size, layer_idx=layer_idx))
        self.merge = config.bidirectional_merge_type
        self.bidirectional = config.allow_bidirectionalize and ((not is_causal) or is_hybrid)
        self.batched = False

        if self.bidirectional:
            # Batching the two directions means one kernel call, hence one set of weights serving
            # both: it ties the branch to the main mixer, so no shrunken/cnn/borrowed branch can
            # take part, and self.bwd is not built at all (the mirrored half costs nothing extra).
            batchable = config.bidirectional_type == "fla" and not config.reverse_mixer
            if config.bidirectional_batched and not batchable:
                warn("bidirectional_batched runs both directions through one shared kernel call, "
                     f"so it needs the mirrored branch: falling back to the unbatched path for "
                     f"bidirectional_type={config.bidirectional_type} reverse_mixer={config.reverse_mixer}")
            self.batched = config.bidirectional_batched and batchable
            self.bwd = None if self.batched else self._build_backward_branch(
                config, hidden_size, layer_idx)

            if self.merge in ("concat_proj", "gated_concat_proj"):
                self.merge_proj = registry.create(
                    "linear", "linear", in_features=2 * hidden_size, out_features=hidden_size, bias=False)
            if self.merge in ("gated_sum", "gated_concat_proj"):
                self.gate_proj = registry.create(
                    "linear", "linear", in_features=hidden_size, out_features=hidden_size, bias=True)

    def _run(self, mixer, x, use_cache):
        # fla kernels take either a packed (1, T, H) sequence with cu_seqlens, or a plain
        # (B, T, H) batch with a 0-1 padding mask. NormalTensor is already a plain batch
        # (it carries no padding), so it must not be asked to unpad.
        if getattr(x, 'isUnpadded', False):
            hidden, cu_seqlens, attn_mask = x.tensor.unsqueeze(0), x.cu_seqlens_device(), None
        elif getattr(x, 'isPadded', False) and x.padding_mask is not None:
            hidden, cu_seqlens, attn_mask = x.tensor, None, ~x.padding_mask
        else:
            hidden, cu_seqlens, attn_mask = x.tensor if hasattr(x, 'tensor') else x, None, None

        # fla layers disagree about what they hand back (three-tuples mostly, but RWKV-7 appends
        # v_first), so only the first slot is contractual.
        out = mixer(hidden, attention_mask=attn_mask, cu_seqlens=cu_seqlens,
                    past_key_values=None, use_cache=False)[0]

        if getattr(x, 'isUnpadded', False):
            out = out.squeeze(0)
        return replace(x, tensor=out) if hasattr(x, 'tensor') else out

    def _run_paired(self, x):
        """Both directions in a single varlen call: [packed ; reversed packed] is one sequence,
        described by PackedLayout.cu_pair, because reversing each segment in place leaves lengths
        and segment order alone. One set of prep kernels, one launch chain, twice the tiles per
        wave; the reversal index comes from the shared layout, so nothing is rebuilt here."""
        n = x.tensor.shape[0]
        hidden = torch.cat([x.tensor, x.reverse().tensor], dim=0).unsqueeze(0)
        out = self.inner(hidden, attention_mask=None, cu_seqlens=x.layout().cu_pair,
                         past_key_values=None, use_cache=False)[0].squeeze(0)
        return (replace(x, tensor=out[:n]),
                replace(x, tensor=out[n:][x.reverse_index()]))

    def _merge(self, fwd_out, bwd_out):
        if self.merge == "sum":
            return replace(fwd_out, tensor=fwd_out.tensor + bwd_out.tensor)
        if self.merge == "concat_proj":
            return self.merge_proj(replace(fwd_out, tensor=torch.cat([fwd_out.tensor, bwd_out.tensor], dim=-1)))
        if self.merge == "gated_sum":
            gate = torch.sigmoid(self.gate_proj(fwd_out).tensor)
            return replace(fwd_out, tensor=gate * fwd_out.tensor + (1 - gate) * bwd_out.tensor)
        if self.merge == "gated_concat_proj":
            gate = torch.sigmoid(self.gate_proj(fwd_out).tensor)
            cat = torch.cat([gate * fwd_out.tensor, (1 - gate) * bwd_out.tensor], dim=-1)
            return self.merge_proj(replace(fwd_out, tensor=cat))
        raise ValueError(f"Unknown bidirectional_merge_type {self.merge}")

    def forward(self, x, original_x=None, is_causal=None, interpretability=False,
                document_mask=None, use_cache=True, **kwargs):
        if self.bidirectional and self.batched:
            packed = x if getattr(x, 'isUnpadded', False) else x.unpad()
            fwd_out, bwd_out = self._run_paired(packed)
            merged = self._merge(fwd_out, bwd_out)
            return merged if getattr(x, 'isUnpadded', False) else merged.pad()

        fwd_out = self._run(self.inner, x, False)
        if not self.bidirectional:
            return fwd_out
        return self._merge(fwd_out, self.bwd.run(x, self._run))


@registry.register("sequence_mixer", "gated_deltanet", "fla", requires=["fla"], priority=10)
class GatedDeltaNetMixer(FLASequenceMixer):
    fla_cls = GatedDeltaNet


@registry.register("sequence_mixer", "kda", "fla", requires=["fla"], priority=10)
class KDAMixer(FLASequenceMixer):
    fla_cls = KimiDeltaAttention


@registry.register_dataclass("sequence_mixer", "gated_deltanet2")
@dataclass
class GatedDeltaNet2Config(FLAConfig):
    expand_v: float = 1.0
    head_dim: int = 128
    num_heads: int = 16
    num_v_heads: Optional[int] = None
    mode: Literal["chunk", "fused_recurrent"] = "chunk"
    use_short_conv: bool = True
    allow_neg_eigval: bool = False
    conv_size: int = 4
    conv_bias: bool = False


@registry.register("sequence_mixer", "gated_deltanet2", "fla", requires=["fla"], priority=10)
class GatedDeltaNet2Mixer(FLASequenceMixer):
    fla_cls = GatedDeltaNet2


@registry.register_dataclass("sequence_mixer", "mamba3")
@dataclass
class Mamba3Config(FLAConfig):
    # num_heads is not a knob here: Mamba-3 derives it as expand * hidden_size // head_dim.
    state_size: int = 128
    expand: int = 2
    head_dim: int = 64
    n_groups: int = 1
    rope_fraction: Literal[0.5, 1.0] = 0.5
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    A_floor: float = 1e-4
    is_outproj_norm: bool = False
    is_mimo: bool = False
    mimo_rank: int = 4
    chunk_size: int = 64
    use_bias: bool = False
    norm_eps: float = 1e-5


@registry.register("sequence_mixer", "mamba3", "fla", requires=["fla"], priority=10)
class Mamba3Mixer(FLASequenceMixer):
    fla_cls = Mamba3


@registry.register_dataclass("sequence_mixer", "gla")
@dataclass
class GLAConfig(FLAConfig):
    # No head_dim: GLA splits the projected width by num_heads/num_kv_heads.
    mode: Literal["chunk", "fused_recurrent", "recurrent"] = "chunk"
    expand_k: float = 0.5
    expand_v: float = 1.0
    num_heads: int = 4
    num_kv_heads: Optional[int] = None
    feature_map: Optional[str] = None
    use_short_conv: bool = False
    conv_size: int = 4
    conv_bias: bool = False
    use_output_gate: bool = True
    gate_fn: Literal["swish", "relu", "sigmoid"] = "swish"
    elementwise_affine: Optional[bool] = True
    norm_eps: float = 1e-5
    gate_logit_normalizer: int = 16
    gate_low_rank_dim: int = 16
    clamp_min: Optional[float] = None
    fuse_norm: bool = True


@registry.register("sequence_mixer", "gla", "fla", requires=["fla"], priority=10)
class GLAMixer(FLASequenceMixer):
    fla_cls = GatedLinearAttention


@registry.register_dataclass("sequence_mixer", "rwkv7")
@dataclass
class RWKV7Config(FLAConfig):
    mode: Literal["chunk", "fused_recurrent", "recurrent"] = "chunk"
    head_dim: Optional[int] = 64
    num_heads: Optional[int] = None
    decay_low_rank_dim: Optional[int] = None
    gate_low_rank_dim: Optional[int] = None
    a_low_rank_dim: Optional[int] = None
    v_low_rank_dim: Optional[int] = None
    value_dim: Optional[int] = None
    # Only steers RWKV-7's weight init (time_decay goes 0 -> 1 across depth); keep it equal to
    # the number of layers of the model, otherwise layers get the schedule of a deeper/shallower net.
    num_hidden_layers: Optional[int] = 12
    elementwise_affine: Optional[bool] = True
    norm_eps: float = 1e-5
    fuse_norm: bool = False


@registry.register("sequence_mixer", "rwkv7", "fla", requires=["fla"], priority=10)
class RWKV7Mixer(FLASequenceMixer):
    fla_cls = RWKV7Attention


@registry.register_dataclass("sequence_mixer", "delta_net")
@dataclass
class DeltaNetConfig(FLAConfig):
    mode: Literal["chunk", "fused_recurrent", "recurrent"] = "chunk"
    expand_k: float = 1.0
    expand_v: float = 1.0
    num_heads: int = 4
    use_beta: bool = True
    use_gate: bool = False
    use_short_conv: bool = True
    conv_size: int = 4
    conv_bias: bool = False
    allow_neg_eigval: bool = False
    qk_activation: Literal["silu", "relu"] = "silu"
    qk_norm: Literal["l2", "softmax", "none"] = "l2"
    norm_eps: float = 1e-5


@registry.register("sequence_mixer", "delta_net", "fla", requires=["fla"], priority=10)
class DeltaNetMixer(FLASequenceMixer):
    fla_cls = DeltaNet


@registry.register_dataclass("sequence_mixer", "hgrn2")
@dataclass
class HGRN2Config(FLAConfig):
    mode: Literal["chunk", "fused_recurrent", "recurrent"] = "chunk"
    num_heads: Optional[int] = None
    expand_ratio: Optional[int] = 128
    use_short_conv: bool = False
    conv_size: int = 4
    conv_bias: bool = False
    elementwise_affine: Optional[bool] = True
    norm_eps: float = 1e-5


@registry.register("sequence_mixer", "hgrn2", "fla", requires=["fla"], priority=10)
class HGRN2Mixer(FLASequenceMixer):
    fla_cls = HGRN2Attention