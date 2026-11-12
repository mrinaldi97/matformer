# matformer/dataclasses/model_dataclasses.py
"""
--- Custom layers ---
`ModelConfig.default_layer` is the LayerConfig every layer starts from. `custom_layers` is a
dict of *patches* on top of it, keyed by either an exact layer index or a pattern string:

    model:
      default_layer:
        attn_impl: flash
      custom_layers:
        even:                       # pattern: applies to every even layer
          sliding_window_size: 128
        12:                         # exact index: applies to layer 12 only
          hooks: {pre_attn: probe}

Supported pattern keys:
    "every_N"                  every Nth layer (layer_idx % N == 0)
    "every_N_offset_K"         every Nth layer, shifted by K (layer_idx % N == K % N) --
                                e.g. "every_4_offset_3" matches 3, 7, 11, ...; offset defaults
                                to 0 (i.e. "every_N" and "every_N_offset_0" are equivalent)
    "odd" / "even"             all odd / even layers
    "odd_start_S_end_E"        odd/even layers restricted to [S, E)
    "S_E"  (e.g. "4_8")        layers in [S, E); "range_S_E" is an accepted alias

get_layer_config(idx) resolves the effective LayerConfig by cascading, in order:
    default_layer  ->  every matching pattern (declaration order)  ->  an exact-index entry
Each step deep-merges onto the previous one -- nested dicts (like 'hooks') merge key-wise
rather than being replaced wholesale. So a custom_layers entry only needs to state the fields
it's actually changing: adding {12: {hooks: {pre_attn: probe}}} on top of an "even" pattern that
already set sliding_window_size for layer 12 keeps that sliding_window_size, it doesn't reset it.
This is also exactly how you add a hook to one specific layer: a hook is just a field
(`hooks: {position: definition}`) inside a LayerConfig, so it goes through the same custom_layers
mechanism as any other per-layer override -- see HooksMixin for the hook definition format
itself. The one place deep-merge doesn't help: if two patches set a *list* of hooks at the same
position, the later one replaces the list outright rather than concatenating -- restate the full
list if you want both.
"""
from dataclasses import dataclass, fields, asdict, field
from typing import Optional, Literal, Union, List, Tuple, Dict, Any, ClassVar
from matformer.dataclasses.matformer_config import MatformerConfig, MatformerMultiConfig, deep_merge
from matformer.matformer_registry import registry

@registry.register_dataclass("layer", "default", default=True)
@dataclass
class LayerConfig(MatformerConfig):
    config_keywoard: ClassVar[str] = "layer"
    _merge_fields: ClassVar[frozenset] = frozenset({"hooks"})
    normalization: Optional[Literal["layernorm", "rmsnorm", "rmsnorm_zc"]] = "rmsnorm"
    normalization_position: Optional[Literal["pre", "post"]] = "post"
    ffn_activation: Optional[Literal["gelu", "gelu_tanh", "geglu", "geglu_tanh", "swiglu"]] = "swiglu"
    interpretability_friendly: Optional[bool] = False
    bias: Optional[bool] = False
    mlp_bias: Optional[bool] = None
    sequence_mixer: Optional[str] = "attention"
    sequence_mixer_config: Dict[str, Any] = field(default_factory=dict)
    # position -> hook definition(s); see HooksMixin. A definition is a registry key string
    # ("probe") or a dict with a mandatory 'type' plus constructor kwargs ({"type": "probe", "scale": 0.1}).
    hooks: Dict[str, Union[str, Dict[str, Any]]] = field(default_factory=dict)
    stream_type: Optional[str] = "residual_stream"
    def __post_init__(self):
        if self.mlp_bias is None:
            self.mlp_bias = self.bias
        self.sequence_mixer_config=registry.get_definition("sequence_mixer",self.sequence_mixer).from_dict(self.sequence_mixer_config)

@registry.register_dataclass("tokenizer", "huggingface", default=True)
@dataclass
class TokenizerConfig(MatformerConfig):
    pretrained_name: Optional[str] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    mask_token_id: Optional[int] = None


@registry.register_dataclass("masking", "default", default=True)
@dataclass
class MaskingConfig(MatformerConfig):
    masked_substitution_rate: Optional[Union[float, Tuple[float, float], List[float]]] = None
    cloze_probability: Optional[float] = None
    random_probability: Optional[float] = None
    same_probability: Optional[float] = None
    hybrid_mlm_prob: Optional[float] = None
    hybrid_curriculum: Optional[bool] = None
    hybrid_mlm_start: Optional[float] = None
    hybrid_mlm_end: Optional[float] = None
    hybrid_equal_final_step: Optional[int] = None


@registry.register_dataclass("model", "default", default=True)
@dataclass
class ModelConfig(MatformerConfig):
    _merge_fields: ClassVar[frozenset] = frozenset({"hooks", "custom_layers"})
    _linked_fields: ClassVar[list] = ["tokenizer", "plugins"]
    name: Optional[str] = None
    model_class: Optional[str] = None

    # Core dimensions
    hidden_size: Optional[int] = None
    ffn_factor: Optional[float] = None
    num_hidden_layers: Optional[int] = None
    stream_type: Optional[str] = "residual_stream"
    layer_schedule: Optional[List[int]] = None
    # Vocabulary & embeddings
    vocab_size: Optional[int] = None
    tie_word_embeddings: Optional[bool] = False
    lm_head_bias: Optional[bool] = True    # HF causal LMs ship no lm_head bias -> set false in their adapters
    lm_head_transform: Optional[bool] = False  # BERT-style dense->gelu->LayerNorm before the output projection
    lm_head_transform_bias: Optional[bool] = True  # bias on the transform's dense layer (HF classifier_bias)
    final_logit_softcap: Optional[float] = None  # tanh capping of lm_head logits, Gemma2-style (HF final_logit_softcapping)
    type_vocab_size: Optional[int] = None # segment ("token type") embedding table; BERT-style only
    embedding_norm: Optional[bool] = False  # norm applied after word+position+type embeddings (BERT-style)
    embedding_scale: Optional[float] = None  # multiplier on token embeddings, Gemma-style (sqrt(hidden_size))
    position_embedding_offset: Optional[int] = 0  # learnable pos-emb row offset (RoBERTa: padding_idx + 1)

    # Normalization
    rms_norm_eps: Optional[float] = None
    norm_bias: Optional[bool] = True  # bias on every layernorm (HF norm_bias; false e.g. ModernBERT)
    norm_affine: Optional[bool] = True  # learnable scale on norms; false for OLMo-1's non-parametric LayerNorm
    final_norm: Optional[bool] = True    # HF BERT-style models have no final norm -> set false in their adapters

    # Attention
    attention_type: Optional[List[str]] = field(default_factory=list)
    sliding_type: Optional[Literal["full", "disabled", "partial"]] = "disabled"
    max_position_embeddings: Optional[int] = None
    block_size_for_attention: Optional[int] = None
    # Compilation, loss, behaviour
    compile_flexattn: Optional[bool] = False
    bias: Optional[bool] = False
    loss_type: str = "cross_entropy_loss"
    training_objective: Optional[str] = None
    is_causal: Optional[bool] = None

    hooks: Dict[str, Union[str, Dict[str, Any]]] = field(default_factory=dict)

    # Nested configs. custom_layers values are *patches* (partial LayerConfig dicts), resolved
    # lazily and cascaded on top of default_layer by get_layer_config() -- see module docstring.
    default_layer: LayerConfig = field(default_factory=LayerConfig)
    custom_layers: Dict[Union[int, str], Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.default_layer, dict):
            self.default_layer = LayerConfig.from_dict(self.default_layer)
        self.custom_layers = {
            k: (asdict(v) if isinstance(v, LayerConfig) else dict(v)) for k, v in self.custom_layers.items()
        }
        self._layer_cache: Dict[int, LayerConfig] = {}

    def get_layer_config(self, layer_idx: int) -> LayerConfig:
        """Resolve the effective LayerConfig for `layer_idx` (cached). See module docstring for
        the cascade/merge rules."""
        if layer_idx not in self._layer_cache:
            merged = asdict(self.default_layer)
            for key, patch in self.custom_layers.items():          # patterns, declaration order
                if not self._is_exact(key) and self._matches_pattern(layer_idx, key):
                    merged = deep_merge(merged, patch)
            for key in (layer_idx, str(layer_idx)):                 # exact match, applied last
                if key in self.custom_layers:
                    merged = deep_merge(merged, self.custom_layers[key])
                    break
            layer_config = LayerConfig.from_dict(merged)
            if layer_config.sequence_mixer == "attention":
                layer_config.sequence_mixer_config.q_dim = self.hidden_size
                layer_config.sequence_mixer_config.k_dim = self.hidden_size
                layer_config.sequence_mixer_config.v_dim = self.hidden_size
                layer_config.sequence_mixer_config.is_cross_attention = False
                layer_config.sequence_mixer_config.hooks = layer_config.hooks
            self._layer_cache[layer_idx] = layer_config
        return self._layer_cache[layer_idx]

    @staticmethod
    def _is_exact(key: Union[int, str]) -> bool:
        """True if `key` names one specific layer (an int, or a digit string) rather than a pattern."""
        return isinstance(key, int) or (isinstance(key, str) and key.isdigit())

    def _matches_pattern(self, layer_idx: int, pattern: str) -> bool:
        """Match layer_idx against a custom_layers pattern key. See module docstring for the
        supported forms. Exact indices never reach this function (handled in get_layer_config)."""
        parts = pattern.split("_")
        if parts[0] == "range":
            parts = parts[1:]  # "range_S_E" and "S_E" are equivalent
        kw = lambda name, default: next((int(parts[i + 1]) for i, p in enumerate(parts) if p == name), default)
        if parts[0] == "every":
            n = int(parts[1])
            offset = kw("offset", 0)
            return layer_idx % n == offset % n
        if parts[0] in ("odd", "even"):
            start, end = kw("start", 0), kw("end", self.num_hidden_layers or 10**9)
            return start <= layer_idx < end and (layer_idx % 2 == 1) == (parts[0] == "odd")
        if len(parts) == 2 and parts[0].lstrip("-").isdigit() and parts[1].isdigit():
            return int(parts[0]) <= layer_idx < int(parts[1])
        return False



