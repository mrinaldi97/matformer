from dataclasses import dataclass, field
from typing import List, Optional, Literal, Union, Dict, Callable, Tuple, Any
import torch.nn as nn
import importlib

@dataclass
class LayerConfig:
    """Configuration for a single transformer layer"""
    attn_impl: Optional[Literal['flash','sdpa','flex','xformers','wersa']] = "flash"
    sliding_window_size: Optional[int] = None
    positional_encoding: Optional[Union[List[Literal['alibi','rope','sinusoidal','nope', 'learnable']], Literal['alibi','rope','sinusoidal','nope', 'learnable']]] = 'alibi'    
    normalization: Optional[Literal['layernorm','rmsnorm']] = 'rmsnorm'
    normalization_position: Optional[Literal['pre','post']] = 'post'
    ffn_activation: Optional[Literal['gelu','swiglu']] = 'swiglu'
    
    hooks: Dict[str, Union[str, Dict[str, Any]]] = field(default_factory=dict)
    def __getitem__(self, key):
        """Allow dict-like access to dataclass fields"""
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        """Allow dict-like setting of dataclass fields"""
        setattr(self, key, value)
    def get(self, key, default=None):
        """Dict-like get method with default value"""
        return getattr(self, key, default)        
def resolve_hook(hook_spec: Union[str, nn.Module, Callable], config: 'ModelConfig') -> Union[nn.Module, Callable]:
    """Resolve hook specification to actual callable/module"""
    if isinstance(hook_spec, (nn.Module, type(lambda: None))):
        return hook_spec
    
    if isinstance(hook_spec, str):
        if '.' in hook_spec:
            # Import from module: "custom_functions.my_hook"
            module_name, func_name = hook_spec.rsplit('.', 1)
            module = importlib.import_module(module_name)
            return getattr(module, func_name)
        else:
            # Class name - instantiate with config
            module = importlib.import_module('custom_functions')  # default module
            cls = getattr(module, hook_spec)
            return cls(config) if issubclass(cls, nn.Module) else cls
    
    raise ValueError(f"Invalid hook specification: {hook_spec}")

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
    sliding_type: Optional[Literal['full','disabled','partial']] = None
    max_position_embeddings: Optional[int] = None
    block_size_for_attention: Optional[int] = None
    rope_theta: float = 10000.0
    # stuff for masked modeling:
    mask_token_id: Optional[int] = None
    masked_substitution_rate: Optional[Union[float, Tuple[float, float], List[float]]] = None #fixed rate (float)/ rate range (Tuple/List[float, float] lower_bound,upper_bound )
    cloze_probability: Optional[float] = None
    random_probability: Optional[float] = None
    same_probability: Optional[float] = None
    # compilation & bias
    compile_flexattn: Optional[bool] = None
    bias: Optional[bool] = None
    loss_type: str = 'normal' #Can be normal or fused
    # behavior
    training_objective: Optional[str] = None
    is_causal: Optional[bool] = None
    has_text_autoencoder: Optional[bool] = None
    # tokenizer
    tokenizer_type: Optional[str] = None #ex. HuggingFace
    tokenizer_name: Optional[str] = None # Ex mrinaldi/Gettone

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
    
    # Layer configuration system
    default_layer: LayerConfig = field(default_factory=LayerConfig)
    #custom_layers: Dict[Union[int, str], LayerConfig] = field(default_factory=LayerConfig)
    custom_layers: Dict[Union[int, str], LayerConfig] = field(default_factory=dict)

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
            
        # Convert custom_layers dict entries to LayerConfig if needed
        for key, config in self.custom_layers.items():
            if isinstance(config, dict):
                self.custom_layers[key] = LayerConfig(**config)
    
    def get_layer_config(self, layer_idx: int) -> LayerConfig:
        """Get configuration for a specific layer"""
        if layer_idx in self.custom_layers:
            return self.custom_layers[layer_idx]
        
        for pattern, config in self.custom_layers.items():
            if isinstance(pattern, str) and self._matches_pattern(layer_idx, pattern):
                return config
                
        return self.default_layer
    
    def _matches_pattern(self, layer_idx: int, pattern: str) -> bool:
        """Compact pattern matcher for layer groups"""
        parts = pattern.split()
        
        if pattern.startswith("every_"):
            step = int(pattern[6:])
            return layer_idx % step == 0
            
        elif pattern.startswith(("odd", "even")):
            is_odd = pattern.startswith("odd")
            start = next((int(parts[i+1]) for i, p in enumerate(parts) if p == "start"), 0)
            end = next((int(parts[i+1]) for i, p in enumerate(parts) if p == "end"), self.num_hidden_layers or 32)
            
            return (start <= layer_idx < end and 
                    (layer_idx % 2 == 1) == is_odd)
        
        elif "_" in pattern and pattern.replace("_", "").replace("range", "").isdigit():
            #  "_8_16" -> layers 8,9,10...15
            parts = pattern.split("_")
            if len(parts) >= 3:
                start, end = int(parts[-2]), int(parts[-1])
                return start <= layer_idx < end
        elif pattern.isdigit():
            return int(pattern)==int(layer_idx)
        return False

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

@dataclass
class ClassificationConfig(ModelConfig):
    """Config for sequence classification tasks."""
    
    # Classification-specific params
    num_labels: int = 2
    pooling_type: Literal['cls', 'mean'] = 'cls'
    classifier_dropout_p: float = 0.1
    classifier_dropout_inplace: bool = False
    
    # Fine-tuning specific
    freeze_encoder: bool = False
    pretrained_checkpoint: Optional[str] = None

@dataclass
class TokenClassificationConfig(ModelConfig):
    """Config for token-level classification (NER, POS tagging)."""
    
    num_labels: int = 2
    classifier_dropout_p: float = 0.1
    classifier_dropout_inplace: bool = False
    freeze_encoder: bool = False
    pretrained_checkpoint: Optional[str] = None

def load_and_validate_classification_config_from_dict(config_dict: dict) -> ClassificationConfig:
    """Parse and validate config dict into ClassificationConfig."""
    
    # Validate required fields
    required = ['hidden_size', 'num_hidden_layers', 'num_attention_heads', 
                'vocab_size', 'default_layer', 'save_dir', 'data', 'training']
    missing = [f for f in required if f not in config_dict]
    
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
      
    required_data = ["train_file", "text_label", "target_label"]
    missing += [f for f in required_data if f not in config_dict["data"]]
    
    if missing:
        raise ValueError(f"Missing required fields in the data section: {missing}")
    
    # Convert default_layer if dict
    if isinstance(config_dict['default_layer'], dict):
        config_dict['default_layer'] = LayerConfig(**config_dict['default_layer'])
    
    # Set defaults
    config_dict.setdefault('custom_layers', {})
    
    # Separate training/data config from model config
    extra_fields = {k: config_dict.pop(k) for k in ['training', 'data', 'save_dir', 
                    'wandb_project', 'wandb_run_name'] if k in config_dict}
    
    # Create config and attach extras
    config = ClassificationConfig(**config_dict)
    for k, v in extra_fields.items():
        setattr(config, k, v)
    
    print(f"Loaded: {config.num_hidden_layers}L x {config.hidden_size}d × {config.num_attention_heads}h, "
          f"{config.num_labels} classes, {config.pooling_type} pooling")
    
    return config