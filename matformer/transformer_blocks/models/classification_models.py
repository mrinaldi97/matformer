from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from matformer.matformer_registry import registry               
from matformer.dataclasses.model_dataclasses import ModelConfig
from matformer.plugins.plugin_classes import Plugin, PluginConfig
from matformer.utils.matformer_cache import ensure_cache, CachedStuff
from matformer.transformer_blocks import TransformerWithEmbeddingHead, TransformerWithLMHead, NakedTransformer
from matformer.matformer_module import MatformerModule
from matformer.dataclasses.tensors_dataclasses import NormalTensor

@registry.register_dataclass("plugin", "classification", default=True)
@dataclass
class ClassificationPluginConfig(PluginConfig):
    scope: Literal["sequence", "token"] = "sequence"
    pooling_type: Literal["cls", "mean", "last", "max", "attention"] = "cls"
    num_classes: Optional[int] = None
    pre_head: Optional[str] = None                 # Another model before the head ("transformer")
    additional_transformer_layers: Optional[int] = 1
    head_type: Literal["linear", "mlp"] = "linear"
    head_config: Dict[str, Any] = field(default_factory=dict)
    classifier_dropout_p: float = 0.1
    classifier_dropout_inplace: bool = False
    # attention pooling (pooling_type="attention"); unrelated to the transformer's own attention heads
    attention_pool_heads: int = 8                  # number of learned pooling queries
    attention_pool_use_tanh: bool = False          # tanh weights instead of softmax
    attention_pool_dropout_p: float = 0.0          # drops pooling keys during training
    # if True, the plugin stores its logits in the cache instead of returning them
    # (used for probes; keeps the plugin "passive")
    as_probe: bool = False
    do_rms_norm: bool = False
    rms_elementwise_affine: bool = False
    exclude_end_from_norm: bool = True #Don't compute RMS norm if plugin is mounted on "final_norm"
    in_features: Optional[int] = None #If None, models' hidden size
    l1_penalty: float = 0.0
    random_proj_dim: Optional[int] = None
    random_proj_seed: int = 0

@registry.register("plugin", "classification", "default")
class ClassificationPlugin(Plugin):
    name = "classification"
    classification_head: "param_name:classifier"
    dropout: "param_name:dropout"
    attn_q: "param_name:attn_q"

    def __init__(self, config, model_config, cache):
        super().__init__(config, model_config, ensure_cache(cache))
        self.cache = ensure_cache(cache)
        self.config = config
        in_features=config.in_features if config.in_features else model_config.hidden_size
        self.dropout = registry.create(
            "dropout", "dropout",
            p=config.classifier_dropout_p,
            inplace=config.classifier_dropout_inplace,
        )

        self.classification_transformer = None # It is easy to add new transformer layers before the classification head
        if config.pre_head == 'transformer':
            pre_cfg = deepcopy(model_config)
            pre_cfg.num_hidden_layers = config.additional_transformer_layers
            pre_cfg.layer_schedule = None            # otherwise the main model's schedule is inherited
            pre_cfg._layer_cache = {}                # so per-layer configs are recomputed
            self.classification_transformer = NakedTransformer(pre_cfg, cache=self.cache)
        if config.do_rms_norm:
           norm_kwargs = {"normalized_shape": in_features, "elementwise_affine":config.rms_elementwise_affine}
           self.norm = registry.create("norm", "rmsnorm", **norm_kwargs)
        self.attn_q = None
        if config.scope == 'sequence' and config.pooling_type == 'attention':
            self._init_attn_pool()
            in_features *= config.attention_pool_heads   # head sees concat of the pooled vectors
        Q = None
        if config.random_proj_dim:
            g = torch.Generator().manual_seed(config.random_proj_seed)
            Q = torch.linalg.qr(torch.randn(in_features, config.random_proj_dim, generator=g)).Q  # (D, k)
            in_features = config.random_proj_dim
        self.register_buffer("random_proj", Q, persistent=False)
        self.classification_head = self._build_head(in_features, config.num_classes or 2)
        if config.l1_penalty:
            self.l1("classifier.weight", config.l1_penalty)
    def _build_head(self, in_features, out_features):
        if self.config.head_type == 'linear':
            return registry.create(
                "linear", "linear",
                in_features=in_features, out_features=out_features,
            )
        return registry.create(
            "classification_head", self.config.head_type,
            in_features=in_features, out_features=out_features,
            **{k: v for k, v in self.config.head_config.items()
               if k != 'transformer_layers'},
        )

    def _init_attn_pool(self):
        c, d = self.config, self.model_config.hidden_size
        self.attn_q = registry.create(
            "linear", "linear", in_features=d, out_features=c.attention_pool_heads, bias=False)
        self.attn_q.inner.weight.data.zero_()                       # start from uniform attention
        self.attn_position_weight = nn.Parameter(torch.zeros(c.attention_pool_heads))  # per-head bias on position
        self.attn_hook = nn.Identity()                               # register_forward_hook to read the weights

    def _attn_pooled(self, tensor, padding_mask):
        """tensor (B,T,D), padding_mask (B,T) True = padding  ->  (B, heads*D)"""
        c = self.config
        mask = (~padding_mask).float()                               # (B,T), 1 = real token
        position = torch.arange(tensor.shape[1], device=tensor.device, dtype=torch.float32)

        k = self.attn_q(NormalTensor(tensor=tensor)).tensor.float() + position[None, :, None] * self.attn_position_weight.float()  # (B,T,H)
        if self.training:
            k = torch.where(torch.rand_like(k) < c.attention_pool_dropout_p, -1e9, k)
        k = k - (1 - mask)[..., None] * 1e9                          # padding always below everything else

        if c.attention_pool_use_tanh:
            p = torch.tanh(k) * mask[..., None]                      # tanh(-1e9) = -1, so zero padding explicitly
        else:
            p = torch.softmax(k, dim=-2)                             # over T; padded weights are exactly 0
        self.attn_hook(p)                                            # (B,T,H)
        p = p.to(tensor.dtype)
        return torch.einsum('bth,btd->bhd', p, tensor).flatten(1)    # pooled_h = Σ_t p[t,h] x_t, head-major

    def _pooled(self, hidden_states):
        tensor = hidden_states.tensor
        padding_mask = getattr(
            hidden_states, 'padding_mask',
            torch.zeros(tensor.shape[:2], dtype=torch.bool, device=tensor.device),
        ).to(device=tensor.device, dtype=torch.bool)
        pt = self.config.pooling_type
        if pt == 'cls':
            return tensor[:, 0, :]
        if pt == 'attention':
            return self._attn_pooled(tensor, padding_mask)
        keep = (~padding_mask).unsqueeze(-1)
        if pt == 'mean':
            return (tensor * keep).sum(1) / keep.sum(1).clamp(min=1e-9)
        if pt == 'max':
            return tensor.masked_fill(
                padding_mask.unsqueeze(-1), float('-inf')
            ).max(1).values
        if pt == 'last':
            last = (~padding_mask).sum(1).clamp(min=1) - 1
            return tensor.gather(
                1, last[:, None, None].expand(-1, 1, tensor.shape[-1])
            ).squeeze(1)
        raise ValueError(f"Pooling type '{pt}' not in ['cls', 'mean', 'max', 'last', 'attention']")

    def _compute(self, x):
        if self.classification_transformer:
            x = self.classification_transformer(x)
        if self.config.do_rms_norm:
            if self.config.exclude_end_from_norm and "final_norm" in self.id:
                pass
            else:
                x=self.norm(x)
        if self.config.scope == 'token':
            t = x.tensor.to(self.classification_head.inner.weight.dtype)
            return self.classification_head(self.dropout(NormalTensor(tensor=t))).tensor
        if self.config.do_rms_norm:
            F.rms_norm(x.tensor.float(), x.tensor.shape[-1:])
        pooled = self._pooled(x.pad()) # Padding before pooling
        if self.random_proj is not None:
            pooled = pooled.float() @ self.random_proj
        pooled = pooled.to(self.classification_head.inner.weight.dtype)
        return self.classification_head(self.dropout(NormalTensor(tensor=pooled))).tensor

    def forward(self, x, position=None, **kwargs):
        logits = self._compute(x)
        if self.config.as_probe:
            # stored under the socket address; "@n" marks the n-th pass in looped models
            key = self.address if not self.visit else f"{self.address}@{self.visit}"
            self.cache.storage.setdefault('probe_outputs', {})[key] = logits
            return None
        return logits


def classification_plugin_config(model_config, **overrides):
    """Plugin config built from the same-named fields of a model config."""
    probe_fields = registry.get_definition_defaults("plugin", "classification").keys()
    values = {k: getattr(model_config, k) for k in probe_fields if hasattr(model_config, k)}
    return registry.get_definition("plugin", "classification").from_dict({**values, **overrides})


# ClassificationModel

@registry.register_dataclass("model","classification")
@dataclass
class ClassificationConfig(ModelConfig):
    """Base configuration for classification tasks"""
    scope: Literal["sequence", "token"] = "sequence"
    pooling_type: Literal["cls", "mean", "last", "max", "attention"] = "cls"
    num_classes: Optional[int] = None
    head_type: Literal["linear", "mlp"] = "linear"
    head_config: Dict[str, Any] = field(default_factory=dict)
    pre_head: Optional[str] = None #Another model before the head
    additional_transformer_layers: Optional[int] = 1
    classifier_dropout_p: float = 0.1
    classifier_dropout_inplace: bool = False
    # attention pooling (pooling_type="attention"); unrelated to the transformer's own attention heads
    attention_pool_heads: int = 8                  # number of learned pooling queries
    attention_pool_use_tanh: bool = False          # tanh weights instead of softmax
    attention_pool_dropout_p: float = 0.0          # drops pooling keys during training
    freeze_base_model: bool = False
    base_model: Optional[str] = None #The path (or repo name) of the base model to take config and weights from
    base_model_hf: bool = False #wether the base model is from huggingface or not
    do_rms_norm: bool = False 
    rms_elementwise_affine: bool = False
    in_features: Optional[int] = None 
@registry.register("model", "classification", "default")
class TransformerWithClassificationHead(MatformerModule):
    encoder: "transparent"

    def __init__(self, config, cache=None, **kwargs):
        super().__init__()
        self.cache = ensure_cache(cache)
        self.config = config
        self.encoder = TransformerWithLMHead(config=config, cache=self.cache)

    def default_plugins(self):
        return [("*.final_norm", classification_plugin_config(self.config, as_probe=False))]
    def forward(self, x, attention_mask=None, **kwargs):
        return self.encoder(x, return_type='hidden', **kwargs)


# MultiProbeConfig / MultiProbeClassifier

@registry.register_dataclass("model", "multiprobe")
@dataclass
class MultiProbeConfig(ClassificationConfig):
    probe_targets: List[str] = field(default_factory=lambda: ["*.layers.*.layer_output","*final_norm"])
    base_model: Optional[str] = None
    base_model_hf: bool = False
    base_model: Optional[str] = None #The path (or repo name) of the base model to take config and weights from
    base_model_hf: bool = False #wether the base model is from huggingface or not
    do_rms_norm: bool = False
    rms_elementwise_affine: bool = False
    l1_penalty: float = 0.0
    random_proj_dim: Optional[int] = None
    random_proj_seed: int = 0
    in_features: Optional[int] = None
@registry.register("model", "multiprobe", "default")
class MultiProbeClassifier(MatformerModule):
    def __init__(self, config, cache=None, **kwargs):
        super().__init__()
        self.cache = ensure_cache(cache)
        self.config = config
        self.encoder = TransformerWithEmbeddingHead(config=config, cache=self.cache)

    def default_plugins(self):
        targets = self.config.probe_targets
        if isinstance(targets, str):
            targets = [targets]
        cfg = classification_plugin_config(self.config, as_probe=True)
        return [(target, cfg) for target in targets]

    def forward(self, x, **kwargs):
        self.cache.storage['probe_outputs'] = {}
        self.encoder(x, **kwargs)
        return self.cache.storage['probe_outputs']   # {address: logits}
