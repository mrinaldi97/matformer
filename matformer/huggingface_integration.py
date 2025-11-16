import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import asdict, fields, is_dataclass
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast, SequenceClassifierOutput, MaskedLMOutput
from transformers.generation import GenerationMixin
import json
import os
from pathlib import Path

from matformer.models import PL_ModelWrapper
from matformer.transformer_blocks import Autoregressive_Model, BERTModel
from matformer.model_config import ModelConfig
#from matformer.tensors_dataclasses import NormalTensor


def to_dict(obj):
    if not is_dataclass(obj):
        return obj
    result = {}
    for f in fields(obj):
        v = getattr(obj, f.name)
        if is_dataclass(v):
            result[f.name] = to_dict(v)
        elif isinstance(v, dict):
            result[f.name] = {k: to_dict(val) if is_dataclass(val) else val for k, val in v.items()}
        else:
            result[f.name] = v
    return result


def from_dict(data: dict, target_class):
    if not is_dataclass(target_class):
        return data
    kwargs = {}
    field_types = {f.name: f.type for f in fields(target_class)}
    for k, v in data.items():
        if k not in field_types:
            continue
        ftype = field_types[k]
        if is_dataclass(ftype) and isinstance(v, dict):
            kwargs[k] = from_dict(v, ftype)
        else:
            kwargs[k] = v
    return target_class(**kwargs)


class MatformerConfig(PretrainedConfig):
    model_type = "matformer"

    def __init__(self, **kwargs):
        self._matformer_config_dict = kwargs.pop('_matformer_config_dict', None)
        self._checkpoint_path = kwargs.pop('_checkpoint_path', None)
        self._model_class = kwargs.pop('_model_class', None)
        self._tokenizer_name = kwargs.pop('_tokenizer_name', None)
        
        self.use_cache = kwargs.pop('use_cache', True)
        self.tie_word_embeddings = kwargs.pop('tie_word_embeddings', False)
        self.hidden_size = kwargs.get('hidden_size', 768)
        self.num_hidden_layers = kwargs.get('num_hidden_layers', 12)
        self.num_attention_heads = kwargs.get('num_attention_heads', 12)
        self.vocab_size = kwargs.get('vocab_size', 50257)
        self.max_position_embeddings = kwargs.get('max_position_embeddings', 2048)
        self.pad_token_id = kwargs.get('pad_token_id', 0)
        self.bos_token_id = kwargs.get('bos_token_id', 1)
        self.eos_token_id = kwargs.get('eos_token_id', 2)
        #self.num_labels = kwargs.get('num_labels', 2)

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
        config_dict = to_dict(mat_config)
        config_dict.update(overrides)
        config_dict['_matformer_config_dict'] = config_dict.copy()
        return cls(**config_dict)


class MatformerPreTrainedModel(PreTrainedModel):
    config_class = MatformerConfig
    base_model_prefix = "matformer"
    supports_gradient_checkpointing = False
    _no_split_modules = ["TransformerBlock"]

    def _init_weights(self, module):
        pass
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        
        if config is None:
            config_path = Path(pretrained_model_name_or_path) / "config.json"
            if config_path.exists():
                config = MatformerConfig.from_pretrained(pretrained_model_name_or_path)
            else:
                from huggingface_hub import hf_hub_download
                config_file = hf_hub_download(repo_id=pretrained_model_name_or_path, filename="config.json")
                config = MatformerConfig.from_pretrained(pretrained_model_name_or_path)
        
        checkpoint_path = Path(pretrained_model_name_or_path) / "checkpoint.ckpt"
        if not checkpoint_path.exists():
            from huggingface_hub import hf_hub_download
            checkpoint_path = hf_hub_download(repo_id=pretrained_model_name_or_path, filename="checkpoint.ckpt")
        
        map_location = kwargs.pop("device_map", "cuda" if torch.cuda.is_available() else "cpu")
        if map_location == "auto":
            map_location = "cuda" if torch.cuda.is_available() else "cpu"
            
        return cls._load_from_checkpoint(checkpoint_path, config, map_location)


class MatformerModel(MatformerPreTrainedModel):
    def __init__(self, config: MatformerConfig):
        super().__init__(config)
        self.matformer_model = None
        
    @classmethod
    def _load_from_checkpoint(cls, checkpoint_path, config, map_location):
        instance = cls(config)
        
        ModelClass = eval(config._model_class) if config._model_class else Autoregressive_Model
        
        model, _ = PL_ModelWrapper.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            ModelClass=ModelClass,
            map_location=map_location,
            tokenizer=config._tokenizer_name
        )
        
        model = model.to(map_location).to(torch.bfloat16).eval()
        
        for module in model.modules():
            if hasattr(module, "alibi_slopes") and module.alibi_slopes is not None:
                module.alibi_slopes = module.alibi_slopes.to(dtype=torch.float32)
        
        instance.matformer_model = model
        instance.post_init()
        
        return instance
        
    def forward(self, input_ids, attention_mask=None, **kwargs):
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
        #input_ids = NormalTensor(tensor=input_ids)
        return self.matformer_model(input_ids)


class MatformerForCausalLM(MatformerPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.module.inner.weight", "lm_head.module.inner.bias"]
    
    def __init__(self, config: MatformerConfig):
        super().__init__(config)
        self.matformer_model = None
        
    @classmethod
    def _load_from_checkpoint(cls, checkpoint_path, config, map_location):
        instance = cls(config)
        
        model, _ = PL_ModelWrapper.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            ModelClass=Autoregressive_Model,
            map_location=map_location,
            tokenizer=config._tokenizer_name
        )
        
        model = model.to(map_location).to(torch.bfloat16).eval()
        
        for module in model.modules():
            if hasattr(module, "alibi_slopes") and module.alibi_slopes is not None:
                module.alibi_slopes = module.alibi_slopes.to(dtype=torch.float32)
        
        instance.matformer_model = model
        instance.post_init()
        
        return instance
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
        
        #input_ids = NormalTensor(tensor=input_ids)
        hidden_states = self.matformer_model(input_ids,return_type='hidden')
        logits = self.matformer_model.model.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return CausalLMOutputWithPast(loss=loss, logits=logits)
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}


class MatformerForMaskedLM(MatformerPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight", "lm_head.bias"]
    
    def __init__(self, config: MatformerConfig):
        super().__init__(config)
        self.matformer_model = None
        
    @classmethod
    def _load_from_checkpoint(cls, checkpoint_path, config, map_location):
        instance = cls(config)
        
        model, _ = PL_ModelWrapper.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            ModelClass=BERTModel,
            map_location=map_location,
            tokenizer=config._tokenizer_name
        )
        
        model = model.to(map_location).to(torch.bfloat16).eval()
        
        for module in model.modules():
            if hasattr(module, "alibi_slopes") and module.alibi_slopes is not None:
                module.alibi_slopes = module.alibi_slopes.to(dtype=torch.float32)
        
        instance.matformer_model = model
        instance.post_init()
        
        return instance
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
        
        #input_ids = NormalTensor(tensor=input_ids)
        hidden_states = self.matformer_model(input_ids,return_type='hidden')
        logits = self.matformer_model.model.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
        
        return MaskedLMOutput(loss=loss, logits=logits)


class MatformerForSequenceClassification(MatformerPreTrainedModel):
    def __init__(self, config: MatformerConfig):
        super().__init__(config)
        self.matformer_model = None
        self.num_labels = config.num_labels
        
    @classmethod
    def _load_from_checkpoint(cls, checkpoint_path, config, map_location):
        instance = cls(config)
        
        model, _ = PL_ModelWrapper.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            ModelClass=BERTModel,
            map_location=map_location,
            tokenizer=config._tokenizer_name
        )
        
        model = model.to(map_location).to(torch.bfloat16).eval()
        
        for module in model.modules():
            if hasattr(module, "alibi_slopes") and module.alibi_slopes is not None:
                module.alibi_slopes = module.alibi_slopes.to(dtype=torch.float32)
        
        if not hasattr(model.model, 'classification_head'):
            model.model.init_classification_head(config.num_labels)
        
        instance.matformer_model = model
        instance.post_init()
        
        return instance
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
        
        #input_ids = NormalTensor(tensor=input_ids)
        logits = self.matformer_model.forward_classification(input_ids, attention_mask)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        
        return SequenceClassifierOutput(loss=loss, logits=logits)


def register_matformer():
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSequenceClassification
    
    AutoConfig.register("matformer", MatformerConfig)
    AutoModel.register(MatformerConfig, MatformerModel)
    AutoModelForCausalLM.register(MatformerConfig, MatformerForCausalLM)
    AutoModelForMaskedLM.register(MatformerConfig, MatformerForMaskedLM)
    AutoModelForSequenceClassification.register(MatformerConfig, MatformerForSequenceClassification)


def load_matformer_model(checkpoint_path, config_dict, model_type='auto', **kwargs):
    map_location = kwargs.get('map_location', 'cuda')
    
    if model_type == 'auto':
        model_class_name = config_dict.get('model_class', 'Autoregressive_Model')
        if 'Autoregressive' in model_class_name:
            model_type = 'causal'
        elif 'BERT' in model_class_name:
            model_type = 'masked'
        else:
            raise ValueError(f"Cannot auto-detect model type from {model_class_name}")
    
    matformer_config = ModelConfig(**config_dict['model_config'])
    hf_config = MatformerConfig.from_matformer_config(
        matformer_config,
        _checkpoint_path=checkpoint_path,
        _model_class=config_dict.get('model_class'),
        _tokenizer_name=config_dict['tokenizer']['pretrained_name'],
        num_labels=kwargs.get('num_labels', 2)
    )
    
    if model_type == 'causal':
        return MatformerForCausalLM._load_from_checkpoint(checkpoint_path, hf_config, map_location)
    elif model_type == 'masked':
        return MatformerForMaskedLM._load_from_checkpoint(checkpoint_path, hf_config, map_location)
    elif model_type == 'classification':
        return MatformerForSequenceClassification._load_from_checkpoint(checkpoint_path, hf_config, map_location)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def create_modeling_file():
    return '''# modeling_matformer.py
import os
import sys

matformer_root = os.getenv("MATFORMER_ROOT")
if matformer_root:
    matformer_root = os.path.abspath(os.path.expanduser(matformer_root))
    if matformer_root not in sys.path:
        sys.path.insert(0, matformer_root)

try:
    from matformer.huggingface_integration import (
        MatformerForCausalLM,
        MatformerForMaskedLM,
        MatformerForSequenceClassification,
        MatformerModel,
        MatformerConfig,
        register_matformer
    )
    register_matformer()
except ImportError as e:
    import subprocess
    import tempfile
    
    print("Installing Matformer from GitHub...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "git+https://github.com/mrinaldi97/matformer.git"
        ])
        
        from matformer.huggingface_integration import (
            MatformerForCausalLM,
            MatformerForMaskedLM,
            MatformerForSequenceClassification,
            MatformerModel,
            MatformerConfig,
            register_matformer
        )
        register_matformer()
        
    except Exception as install_error:
        raise ImportError(
            "Failed to install Matformer. Install manually:\\n"
            "  pip install git+https://github.com/mrinaldi97/matformer.git\\n"
            "Or set MATFORMER_ROOT environment variable"
        ) from install_error
'''


def create_readme(model_name, model_type):
    code_example = ""
    if model_type == "causal":
        code_example = f'''from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "{model_name}",
    trust_remote_code=True
)
tokenizer=AutoTokenizer.from_pretrained(model.config._tokenizer_name)
text = "The transformer model is a"
inputs = tokenizer(text,return_tensors='pt')['input_ids'].to(model.device)

with torch.no_grad():
    outputs = model.generate(inputs, max_new_tokens=50)
    
generated = model.matformer_model.tokenizer.decode(outputs[0].tolist())

print(generated)'''
    elif model_type == "masked":
        code_example = f'''from transformers import AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained(
    "{model_name}",
    trust_remote_code=True
)'''
    else:
        code_example = f'''from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "{model_name}",
    trust_remote_code=True
)'''
    
    return f'''---
tags:
- matformer
- custom-model
library_name: transformers
---

# Matformer Model

Trained using [Matformer](https://github.com/mrinaldi97/matformer).

## Installation

```bash
pip install git+https://github.com/mrinaldi97/matformer.git
```

## Usage

```python
import torch
{code_example}
```
'''


def push_to_hub(model, config_dict, repo_id, token=None, model_type='auto'):
    from huggingface_hub import HfApi, create_repo
    import shutil
    
    if model_type == 'auto':
        model_class_name = config_dict.get('model_class', 'Autoregressive_Model')
        if 'Autoregressive' in model_class_name:
            model_type = 'causal'
        elif 'BERT' in model_class_name:
            model_type = 'masked'
    
    api = HfApi(token=token)
    create_repo(repo_id, token=token, exist_ok=True)
    
    auto_map = {
        "AutoConfig": "modeling_matformer.MatformerConfig",
        "AutoModel": "modeling_matformer.MatformerModel",
    }
    
    if model_type == 'causal':
        auto_map["AutoModelForCausalLM"] = "modeling_matformer.MatformerForCausalLM"
    elif model_type == 'masked':
        auto_map["AutoModelForMaskedLM"] = "modeling_matformer.MatformerForMaskedLM"
    elif model_type == 'classification':
        auto_map["AutoModelForSequenceClassification"] = "modeling_matformer.MatformerForSequenceClassification"
    
    model.config.auto_map = auto_map
    
    temp_dir = Path("./temp_hf_upload")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        model.config.save_pretrained(temp_dir)
        
        modeling_content = create_modeling_file()
        with open(temp_dir / "modeling_matformer.py", "w") as f:
            f.write(modeling_content)
        
        readme_content = create_readme(repo_id, model_type)
        with open(temp_dir / "README.md", "w") as f:
            f.write(readme_content)
        
        with open(temp_dir / "matformer_config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        
        checkpoint_source = model.config._checkpoint_path
        if checkpoint_source and Path(checkpoint_source).exists():
            shutil.copy(checkpoint_source, temp_dir / "checkpoint.ckpt")
        
        api.upload_folder(
            folder_path=str(temp_dir),
            repo_id=repo_id,
            token=token,
        )
        
        print(f"Model pushed to {repo_id}")
        
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--hf_name", required=True)
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--model_type", default="auto", choices=["auto", "causal", "masked", "classification"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_labels", type=int, default=2)
    
    args = parser.parse_args()
    
    with open(args.config) as f:
        config_dict = json.load(f)
    
    model = load_matformer_model(
        args.checkpoint, 
        config_dict, 
        model_type=args.model_type,
        map_location=args.device,
        num_labels=args.num_labels
    )
    
    push_to_hub(
        model, 
        config_dict, 
        args.hf_name, 
        token=args.hf_token,
        model_type=args.model_type
    )


if __name__ == "__main__":
    main()
