from dataclasses import dataclass, field
from typing import List, Optional, Literal, Union

@dataclass
class BaseSubModelConfig:
    # core dims
    hidden_size: Optional[int] = None
    ffn_factor: Optional[float] = None
    num_hidden_layers: Optional[int] = None
    num_attention_heads: Optional[int] = None

    # vocabulary & embeddings
    vocab_size: Optional[int] = None
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    tie_word_embeddings: Optional[bool] = None

    # normalization
    rms_norm_eps: Optional[float] = None

    # attention & masking
    attention_type: Optional[List[str]] = field(default_factory=list)
    sliding_window_size: Optional[int] = None
    sliding_layers: Optional[List[int]] = field(default_factory=list)
    sliding_type: Optional[Literal['full','disabled','partial']] = None
    max_position_embeddings: Optional[int] = None
    block_size_for_attention: Optional[int] = None
    # stuff for masked modeling:
    mask_token_id: Optional[int] = None
    masked_substitution_rate:Optional[float] = None
    # compilation & bias
    compile_flexattn: Optional[bool] = None
    bias: Optional[bool] = None

    # behavior
    training_objective: Optional[str] = None
    is_causal: Optional[bool] = None
    alibi: Optional[bool] = None
    attn_impl: Optional[str] = None

    has_text_autoencoder: Optional[bool] = None
@dataclass
class SubModelConfig(BaseSubModelConfig):
    name: Optional[str] = None

@dataclass
class EntropyConfig:
    entropy_model_path: Optional[str] = None
    entropy_smoothing: Optional[float] = None

@dataclass
class ModelConfig(BaseSubModelConfig):
    # top-level model info
    name: Optional[str] = None
    model_class: Optional[str] = None
    has_text_autoencoder: Optional[bool] = None
    has_entropy_model: Optional[bool] = None

    # optional subconfigs - now accept both dict and dataclass
    encoder: Optional[Union[SubModelConfig, dict]] = None
    decoder: Optional[Union[SubModelConfig, dict]] = None
    entropy: Optional[Union[EntropyConfig, dict]] = None

    def __post_init__(self):
        # Convert encoder dict to SubModelConfig if needed
        if self.encoder is not None and isinstance(self.encoder, dict):
            self.encoder = SubModelConfig(**self.encoder)
        
        # Convert decoder dict to SubModelConfig if needed
        if self.decoder is not None and isinstance(self.decoder, dict):
            self.decoder = SubModelConfig(**self.decoder)
        
        # Convert entropy dict to EntropyConfig if needed
        if self.entropy is not None and isinstance(self.entropy, dict):
            self.entropy = EntropyConfig(**self.entropy)

@dataclass
class RNNConfig:
    max_epochs: int
    vocab_size: int
    input_dim: int
    hidden_dim: int
    output_dim: int
    num_layers: int
    rnn_type: str
    bidirectional: bool
    eos_token_id: int
    pad_token_id: int
    bos_token_id: int  
    lr: float
    max_target_len: int
    beta: float  
    
    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)


