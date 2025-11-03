import os
import torch
import torch.nn as nn
from dataclasses import asdict, fields, is_dataclass
from typing import Optional, Tuple, Union
from transformers import PretrainedConfig, PreTrainedModel, AutoTokenizer
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutput
from transformers.utils import logging
import re
from matformer.model_config import ModelConfig
from matformer.transformer_blocks import TransformerWithEmbeddingHead, TransformerWithLMHead, TransformerWithClassificationHead
from matformer.matformer_tokenizers import MatformerTokenizer
from matformer.tensors_dataclasses import NormalTensor, PaddedTensor

logger = logging.get_logger(__name__)

# ============================================================================
# Config conversion utilities
# ============================================================================

def safe_asdict(obj):
    if not is_dataclass(obj):
        return obj
    result = {}
    for f in fields(obj):
        v = getattr(obj, f.name)
        if is_dataclass(v):
            result[f.name] = safe_asdict(v)
        elif isinstance(v, dict):
            result[f.name] = {k: safe_asdict(val) if is_dataclass(val) else val for k, val in v.items()}
        else:
            result[f.name] = v
    return result

def safe_from_dict(data: dict, target_class):
    if not is_dataclass(target_class):
        return data
    kwargs = {}
    field_types = {f.name: f.type for f in fields(target_class)}
    for k, v in data.items():
        if k not in field_types:
            continue
        ftype = field_types[k]
        if is_dataclass(ftype) and isinstance(v, dict):
            kwargs[k] = safe_from_dict(v, ftype)
        else:
            kwargs[k] = v
    return target_class(**kwargs)

# ============================================================================
# HuggingFace-compatible config
# ============================================================================

class MatformerConfig(PretrainedConfig):
    model_type = "matformer"

    def __init__(self, **kwargs):
        self._matformer_config_dict = kwargs.pop('_matformer_config_dict', None)
        self.use_cache = kwargs.pop('use_cache', False)
        self.tie_word_embeddings = kwargs.pop('tie_word_embeddings', False)

        self.hidden_size = kwargs.get('hidden_size', 768)
        self.num_hidden_layers = kwargs.get('num_hidden_layers', 12)
        self.num_attention_heads = kwargs.get('num_attention_heads', 12)
        self.vocab_size = kwargs.get('vocab_size', 50257)
        self.max_position_embeddings = kwargs.get('max_position_embeddings', 2048)
        self.pad_token_id = kwargs.get('pad_token_id', 0)
        self.bos_token_id = kwargs.get('bos_token_id', 1)
        self.eos_token_id = kwargs.get('eos_token_id', 2)

        if self._matformer_config_dict is None:
            self._matformer_config_dict = kwargs.copy()

        super().__init__(**kwargs)

    def to_matformer_config(self):
        if self._matformer_config_dict:
            config_dict = self._matformer_config_dict.copy()
        else:
            config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        return ModelConfig(**config_dict)

    @classmethod
    def from_matformer_config(cls, mat_config, **overrides):
        config_dict = safe_asdict(mat_config)
        config_dict.update(overrides)
        config_dict['_matformer_config_dict'] = config_dict.copy()
        return cls(**config_dict)

# ============================================================================
# Base Model
# ============================================================================

class MatformerPreTrainedModel(PreTrainedModel):
    config_class = MatformerConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TransformerBlock"]
    _skip_keys_device_placement = ["past_key_values"]

# ============================================================================
# Model with embeddings
# ============================================================================

class MatformerModel(MatformerPreTrainedModel):
    def __init__(self, config: MatformerConfig):
        super().__init__(config)
        mat_config = config.to_matformer_config()
        self.transformer = TransformerWithEmbeddingHead(mat_config)
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.embed_tokens.module

    def set_input_embeddings(self, value):
        self.transformer.embed_tokens.module = value

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, return_dict=None, **kwargs):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Must provide input_ids or inputs_embeds")
        model_input = input_ids if input_ids is not None else inputs_embeds
        if not isinstance(model_input, NormalTensor):
            model_input = NormalTensor(tensor=model_input)
        hidden_states = self.transformer(model_input, **kwargs)
        if hasattr(hidden_states, 'tensor'):
            hidden_states = hidden_states.tensor
        if not return_dict:
            return (hidden_states,)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, hidden_states=None, attentions=None)

# ============================================================================
# Causal LM
# ============================================================================

class MatformerForCausalLM(MatformerPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.module.weight"]

    def __init__(self, config: MatformerConfig):
        super().__init__(config)
        mat_config = config.to_matformer_config()
        device = self._get_device()
        self.transformer = TransformerWithLMHead(config=mat_config, tokenizer=None, device=device)
        self.move_to_cuda()
        self.main_input_name = "input_ids"
        self.post_init()

    def _get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    def move_to_cuda(self):
        import torch
        self.transformer=self.transformer.to('cuda').to(torch.bfloat16)
        for module in self.transformer.modules():
            if hasattr(module, "alibi_slopes") and module.alibi_slopes is not None:
                module.alibi_slopes = module.alibi_slopes.to(dtype=torch.float32)
        self.transformer.eval()
    def get_input_embeddings(self):
        return self.transformer.transformer.embed_tokens.module

    def set_input_embeddings(self, value):
        self.transformer.transformer.embed_tokens.module = value

    def get_output_embeddings(self):
        return self.transformer.lm_head.module

    def set_output_embeddings(self, new_embeddings):
        self.transformer.lm_head.module = new_embeddings

    def forward(self, input_ids=None, attention_mask=None, labels=None, inputs_embeds=None, return_dict=None, **kwargs):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Must provide input_ids or inputs_embeds")
        model_input = input_ids if input_ids is not None else inputs_embeds
        if not isinstance(model_input, NormalTensor):
            model_input = NormalTensor(tensor=model_input)
        logits = self.transformer(model_input, **kwargs)
        if hasattr(logits, 'tensor'):
            logits = logits.tensor
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None, hidden_states=None, attentions=None)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "inputs_embeds": inputs_embeds}
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        return past_key_values

# ============================================================================
# Sequence classification
# ============================================================================

class MatformerForSequenceClassification(MatformerPreTrainedModel):
    def __init__(self, config: MatformerConfig):
        super().__init__(config)
        mat_config = config.to_matformer_config()
        self.num_labels = getattr(config, 'num_labels', 2)
        pooling_type = getattr(config, 'pooling_type', 'cls')
        self.classifier = TransformerWithClassificationHead(config=mat_config, tokenizer=None, pooling_type=pooling_type, num_features=self.num_labels)
        self.post_init()

    def get_input_embeddings(self):
        return self.classifier.transformer.embed_tokens.module

    def set_input_embeddings(self, value):
        self.classifier.transformer.embed_tokens.module = value

    def forward(self, input_ids=None, attention_mask=None, labels=None, inputs_embeds=None, return_dict=None, **kwargs):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Must provide input_ids or inputs_embeds")
        model_input = input_ids if input_ids is not None else inputs_embeds
        if not isinstance(model_input, NormalTensor):
            model_input = NormalTensor(tensor=model_input)
        logits = self.classifier(model_input, attention_mask=attention_mask)
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and labels.dtype in (torch.long, torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze() if self.num_labels == 1 else logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=None, attentions=None)

# ============================================================================
# Checkpoint loader
# ============================================================================

def rename_state_dict_keys(state_dict: dict) -> dict:
    """
    Robustly rename old Matformer checkpoint keys to match the current model.
    
    - Strips 'model.' prefix from Lightning checkpoints.
    - Adds 'transformer.' prefix to lm_head and embedding keys if missing.
    - Fixes nested transformer layers including self_attn, mlp, and norm modules.
    """

    renamed = {}

    for key, value in state_dict.items():
        new_key = key

        # ---- 1. Strip top-level 'model.' prefix if present ----
        if new_key.startswith("model."):
            new_key = new_key[len("model."):]

        # ---- 2. Top-level lm_head or embed_tokens missing 'transformer.' prefix ----
        #if re.match(r'^(lm_head|transformer\.embed_tokens)', new_key) and not new_key.startswith("transformer."):
        #    new_key = f"transformer.{new_key}"

        # ---- 3. Nested transformer layers ----
        # old keys: transformer.layers.{i}.*
        layer_match = re.match(r'^transformer\.layers\.(\d+)\.(.+)', new_key)
        if layer_match:
            layer_idx, rest = layer_match.groups()
            new_key = f"transformer.transformer.transformer.layers.{layer_idx}.{rest}"

        # ---- 4. Add 'module' to submodules if missing ----
        submodule_match = re.match(
            r'^(transformer\.transformer\.transformer\.layers\.\d+\.(?:self_attn|mlp|attn_norm|mlp_norm|transformer))\.(\w+)$',
            new_key
        )
        if submodule_match:
            prefix, suffix = submodule_match.groups()
            if not suffix.startswith("module"):
                new_key = f"{prefix}.module.{suffix}"

        renamed['transformer.'+new_key] = value

    return renamed
def load_matformer_checkpoint(checkpoint_path: str, model_cls=None, device: str = None, **config_overrides):
    # ---- device ----
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu"

    # ---- load checkpoint ----
    if checkpoint_path.endswith(".ckpt"):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        mat_config = ckpt["hyper_parameters"]["config"]
        state_dict = ckpt["state_dict"]
    elif os.path.isdir(checkpoint_path):
        config_path = os.path.join(checkpoint_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing config.json in {checkpoint_path}")
        from matformer.model_config import ModelConfig
        mat_config = ModelConfig.from_json(config_path)
        weights_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Missing pytorch_model.bin in {checkpoint_path}")
        state_dict = torch.load(weights_path, map_location="cpu")
    else:
        raise ValueError(f"Invalid checkpoint_path: {checkpoint_path}")

    # ---- convert to HF config ----
    hf_config = MatformerConfig.from_matformer_config(mat_config)

    # ---- initialize model ----
    model = model_cls(hf_config) if model_cls else MatformerForCausalLM(hf_config)

    # ---- rename keys if necessary ----
    state_dict = rename_state_dict_keys(state_dict)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        logger.warning(f"[Matformer] Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
        for i in missing_keys:
            print(f"Missing {i}")
        for i in unexpected_keys:
            print(f"Unexpected {i}")
    # ---- attach tokenizer ----
    #try:
        #tokenizer = MatformerTokenizer.from_pretrained(checkpoint_path)
    #except Exception:
    if True:
        from transformers import AutoTokenizer
        logger.warning("No tokenizer found, using fallback.")
        hf_tok = AutoTokenizer.from_pretrained("mrinaldi/Gettone-TEST")
        #tokenizer = MatformerTokenizer(tokenizer=hf_tok, config=mat_config)
    model.tokenizer = hf_tok

    # ---- move model to device ----
    model.to(device).to(torch.bfloat16)
    for module in model.modules():
        if hasattr(module, "alibi_slopes") and module.alibi_slopes is not None:
            module.alibi_slopes = module.alibi_slopes.to(dtype=torch.float32)
    model.eval()
    return model

# -----------------------------------------------------------------------------
# Payload: automatically include a helper script when pushing to the Hub
# -----------------------------------------------------------------------------

payload_script = '''# modeling_matformer.py
# Auto-generated by Matformer integration for Hugging Face Hub compatibility

import os
import sys

matformer_root = os.getenv("MATFORMER_ROOT")
if matformer_root:
    matformer_root = os.path.abspath(os.path.expanduser(matformer_root))
    if matformer_root not in sys.path:
        sys.path.insert(0, matformer_root)

try:
    from matformer.modelling_matformer import (
        MatformerForCausalLM,
        MatformerForSequenceClassification,
        MatformerModel,
        MatformerConfig,
        register_matformer
    )
    register_matformer()
except ImportError as e:
    raise ImportError(
        "To use this model, please install the Matformer library: "
        "`pip install matformer`\\n"
        "Alternatively, set the `MATFORMER_ROOT` environment variable to a local clone:\\n"
        "  export MATFORMER_ROOT=/path/to/matformer"
    ) from e
'''

readme = """---
tags:
- matformer
- custom-model
library_name: matformer
---

# Matformer Model

This model was trained using the [Matformer](https://github.com/...) library.

## Usage

First, install the required package:

```bash
pip install matformer

Then load the model with trust_remote_code=True:

from transformers import AutoModelForCausalLM

# Load model
hf_model = AutoModelForCausalLM.from_pretrained(
    "YOUR_HF_ORG/YOUR_MODEL_NAME",
    trust_remote_code=True
)

# Access the tokenizer
tokenizer = hf_model.tokenizer

# Simple generation example
import torch

prompt = "The transformer model is a"
inputs = tokenizer.encode(prompt, add_bos=True, add_eos=False)
inputs = torch.tensor([inputs], device=hf_model.device)

with torch.no_grad():
    outputs = hf_model.generate(inputs, max_new_tokens=50)

decoded = tokenizer.decode(outputs[0].tolist())
print(decoded)

    Replace YOUR_HF_ORG/YOUR_MODEL_NAME with the actual model ID.
    """

def _write_modeling_script(repo_path: str):
    """
    Writes the modeling_matformer.py file to the repo so that
    trust_remote_code=True works seamlessly on the Hub.
    """
    script_path = os.path.join(repo_path, "modeling_matformer.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(payload_script)
    logger.info(f"Added modeling_matformer.py to {repo_path}")


def _patched_push_to_hub(self, *args, **kwargs):
    # Save original method and temporarily restore it to avoid recursion
    original_push = PreTrainedModel.push_to_hub
    if hasattr(original_push, '_matformer_patched'):
        # Temporarily revert to the real original
        PreTrainedModel.push_to_hub = getattr(original_push, '__wrapped__', original_push)
    
    try:
        repo_path = kwargs.get("repo_path", ".")
        _write_modeling_script(repo_path)
        result = original_push(self, *args, **kwargs)
    except Exception as e:
        logger.warning(f"Could not write modeling script: {e}")
        result = original_push(self, *args, **kwargs)
    finally:
        # Re-patch after the call
        PreTrainedModel.push_to_hub = _patched_push_to_hub
        PreTrainedModel.push_to_hub._matformer_patched = True

    return result


if not hasattr(PreTrainedModel.push_to_hub, '_matformer_patched'):
    _patched_push_to_hub.__wrapped__ = PreTrainedModel.push_to_hub
    PreTrainedModel.push_to_hub = _patched_push_to_hub
    PreTrainedModel.push_to_hub._matformer_patched = True
