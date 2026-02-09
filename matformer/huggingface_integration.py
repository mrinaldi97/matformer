import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import asdict, fields, is_dataclass
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast, SequenceClassifierOutput, MaskedLMOutput,TokenClassifierOutput
from transformers.generation import GenerationMixin
import json
import os
from pathlib import Path
from matformer.models import PL_ModelWrapper
from matformer.transformer_blocks import Autoregressive_Model, BERTModel, TransformerWithClassificationHead, TransformerWithTokenClassificationHead
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

    def __init__(self, seed=42, **kwargs):
        self.seed = seed
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
    base_model_prefix = "matformer_model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["TransformerBlock"]

    def _init_weights(self, module):
        pass
        #try:
        #    from matformer.initialization import init_transformer_weights_
        #    init_transformer_weights_(module)
        #except:
        #    print("Weights' Initialization failed!") 
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        
        if config is None:
            config_path = Path(pretrained_model_name_or_path) / "config.json"
            if config_path.exists():
                config = MatformerConfig.from_pretrained(pretrained_model_name_or_path)
            else:
                # Try to find config.json in parent directory
                parent_config_path = Path(pretrained_model_name_or_path).parent / "config.json"
                if parent_config_path.exists():
                    config = MatformerConfig.from_pretrained(str(Path(pretrained_model_name_or_path).parent))
                else:
                    # Fall back to downloading from HuggingFace Hub
                    from huggingface_hub import hf_hub_download
                    config_file = hf_hub_download(repo_id=pretrained_model_name_or_path, filename="config.json")
                    config = MatformerConfig.from_pretrained(pretrained_model_name_or_path)
        
        checkpoint_path = Path(pretrained_model_name_or_path) / "checkpoint.ckpt"
        if not checkpoint_path.exists():
            base_path = Path(pretrained_model_name_or_path)
            if base_path.is_dir():
                ckpt_files = list(base_path.glob("*.ckpt"))
                if ckpt_files:
                    checkpoint_path = ckpt_files[0]
            else:
                from huggingface_hub import hf_hub_download
                checkpoint_path = hf_hub_download(repo_id=pretrained_model_name_or_path, filename="checkpoint.ckpt")
        
        map_location = kwargs.pop("device_map", "cuda" if torch.cuda.is_available() else "cpu")
        if map_location == "auto":
            map_location = "cuda" if torch.cuda.is_available() else "cpu"
            
        return cls._load_from_checkpoint(checkpoint_path, config, map_location)
    @classmethod
    def _load_from_checkpoint(cls, checkpoint_path, config, map_location):
        #set_seed(config.seed)

        instance = cls(config)
        
        model_class_map = {
            'MatformerForCausalLM': Autoregressive_Model,
            'MatformerForMaskedLM': BERTModel,
            'MatformerForSequenceClassification': BERTModel,
            'MatformerForTokenClassification': BERTModel,
            'MatformerModel': eval(config._model_class) if config._model_class else Autoregressive_Model
        }
        
        ModelClass = model_class_map.get(cls.__name__, Autoregressive_Model)
        
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
    

class MatformerModel(MatformerPreTrainedModel):
    def __init__(self, config: MatformerConfig):
        super().__init__(config)
        self.matformer_model = None
        self.config=config

    def forward(self, input_ids, attention_mask=None, **kwargs):
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
        return self.matformer_model(input_ids)


class MatformerForCausalLM(MatformerPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.module.inner.weight", "lm_head.module.inner.bias"]
    
    def __init__(self, config: MatformerConfig):
        super().__init__(config)
        self.matformer_model = None
        self.config=config
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        if isinstance(input_ids, list):
            input_ids = self.matformer_model.model.collate_generation_inputs(
                input_ids, 
                pad_token_id=self.config.pad_token_id
            )
        elif isinstance(input_ids, torch.Tensor) and input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        hidden_states = self.matformer_model(input_ids, return_type='hidden')
        logits = self.matformer_model.model.lm_head(hidden_states).tensor
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),ignore_index=self.config.pad_token_id
            )
        
        return CausalLMOutputWithPast(loss=loss, logits=logits)
    def generate(
        self,
        inputs=None,
        input_ids=None,
        max_length=None,
        max_new_tokens=None,
        min_length=0,
        min_new_tokens=None,
        temperature=1.0,
        top_k=0,
        top_p=0.9,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        do_sample=True,
        num_return_sequences=1,
        num_beams=1,
        pad_token_id=None,
        eos_token_id=None,
        bos_token_id=None,
        stopping_criteria=None,
        output_scores=False,
        return_dict_in_generate=False,
        use_cache=None,
        length_penalty=None,
        early_stopping=None,
        **kwargs
    ):
        if num_beams > 1:
            warn("Beam search not supported, ignoring num_beams parameter")
        if use_cache is not None:
            warn("use_cache not supported, ignoring parameter")
        if length_penalty is not None:
            warn("length_penalty not supported (requires beam search), ignoring parameter")
        if early_stopping is not None:
            warn("early_stopping not supported (requires beam search), ignoring parameter")
        if return_dict_in_generate:
            warn("return_dict_in_generate not fully supported, will return tensors only")
        
        if input_ids is None:
            input_ids = inputs
        
        if max_new_tokens is not None:
            if input_ids is not None:
                input_length = input_ids.shape[-1] if isinstance(input_ids, torch.Tensor) else len(input_ids)
                effective_max_length = input_length + max_new_tokens
            else:
                effective_max_length = max_new_tokens
        elif max_length is not None:
            effective_max_length = max_length
        else:
            effective_max_length = 100
        
        if min_new_tokens is not None:
            if input_ids is not None:
                input_length = input_ids.shape[-1] if isinstance(input_ids, torch.Tensor) else len(input_ids)
                effective_min_length = input_length + min_new_tokens
            else:
                effective_min_length = min_new_tokens
        else:
            effective_min_length = min_length
        
        if not do_sample:
            temperature = 1e-7
        
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        if bos_token_id is None:
            bos_token_id = self.config.bos_token_id
        
        result = self.matformer_model.model.generate(
            prompt=None,
            input_ids=input_ids,
            max_length=effective_max_length,
            min_length=effective_min_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            num_return_sequences=num_return_sequences,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            stopping_criteria=stopping_criteria,
            output_scores=output_scores,
            return_type='pt'
        )
        
        if output_scores:
            output_ids, scores = result
            if return_dict_in_generate:
                return {"sequences": output_ids, "scores": scores}
            return output_ids
        
        return result
    
  
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}


class MatformerForMaskedLM(MatformerPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight", "lm_head.bias"]
    
    def __init__(self, config: MatformerConfig):
        super().__init__(config)
        self.matformer_model = None
        self.config=config
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        if isinstance(input_ids, list):
            input_ids = self.matformer_model.model.collate_generation_inputs(
                input_ids,
                pad_token_id=self.config.pad_token_id
            )
        elif isinstance(input_ids, torch.Tensor) and input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        hidden_states = self.matformer_model(input_ids, return_type='hidden')
        logits = self.matformer_model.model.lm_head(hidden_states).tensor
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1), ignore_index=self.config.pad_token_id
            )
        
        return MaskedLMOutput(loss=loss, logits=logits)


class MatformerForSequenceClassification(MatformerPreTrainedModel):

    _tied_weights_keys = [
        "matformer_model.model.lm_head.weight",
        "matformer_model.model.encoder.embed_tokens.module.inner.weight",
    ]
     
    def __init__(self, config: MatformerConfig, pooling_type='cls', dropout_rate=0.1):
        super().__init__(config)
        self.matformer_model = None
        self.num_labels = config.num_labels
        self.classifier = None
        self.pooling_type = pooling_type
        self.dropout = nn.Dropout(dropout_rate)
    @classmethod
    def _load_from_checkpoint(cls, checkpoint_path, config, map_location):
        
        instance = super()._load_from_checkpoint(checkpoint_path, config, map_location)
        #set_seed(config.seed)
        #instance.matformer_model.model.change_num_labels(instance.num_labels)
        if hasattr(instance.matformer_model.model, 'hidden_size'):
            hidden_size = instance.matformer_model.model.hidden_size
        elif hasattr(instance.config, 'hidden_size'):
            hidden_size = instance.config.hidden_size
        else:
            # Default fallback
            hidden_size = 768
        
        # Create classification head: dropout + linear layer
        instance.classifier = nn.Sequential(
            # nn.Dropout(0.1), #there are two dropouts in the original code, one in the model and one in the head, we keep both for now
            nn.Linear(hidden_size, config.num_labels)
        ).to(map_location).to(torch.bfloat16)

        return instance   
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
        
        if labels is not None:
            if self.num_labels > 1 and (labels.dtype not in (torch.long, torch.int)):
                problem_type = "multi_label_classification"
            else:
                problem_type = "single_label_classification"
        
        # if not hasattr(self.matformer_model, "classification_head"):
            #we actually do not directly call the classification head but we still require it as a 'type check'
            # raise Exception("The underneath model of a MatformerForSequenceClassification needs to have an attr 'classification_head'")
        
        hidden_states = self.matformer_model(input_ids, return_type='hidden')
        if self.pooling_type == 'cls':
            # [CLS] in pos. 0
            pooled_output = hidden_states.tensor[:, 0, :]
        elif self.pooling_type == 'mean':
            #TODO: check if the mask works
            if attention_mask is None:
                pooled_output = hidden_states.tensor.mean(dim=1)
            else:
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.tensor.size()).to(hidden_states.dtype)
                sum_hidden = torch.sum(hidden_states.tensor * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled_output = sum_hidden / sum_mask
        else:
            raise ValueError(f"{self.pooling_type} not in 'cls','mean'")

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            if problem_type == "single_label_classification":
                loss = F.cross_entropy(logits, labels)
            elif problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
            else:
                raise ValueError("Unknown problem type for classification")
        
        return SequenceClassifierOutput(loss=loss, logits=logits)



class MatformerForTokenClassification(MatformerPreTrainedModel):

    _tied_weights_keys = [
        "matformer_model.model.lm_head.weight",
        "matformer_model.model.encoder.embed_tokens.module.inner.weight",
    ]
     
     
    def __init__(self, config: MatformerConfig, dropout_rate=0.1, **kwargs):
        super().__init__(config)
        self.matformer_model = None
        self.num_labels = config.num_labels
        self.classifier = None
        self.dropout = nn.Dropout(dropout_rate)
        
    @classmethod
    def _load_from_checkpoint(cls, checkpoint_path, config, map_location):
        
        instance = super()._load_from_checkpoint(checkpoint_path, config, map_location)
        instance.classifier = nn.Sequential(
            # nn.Dropout(0.1),
            nn.Linear(config.hidden_size, config.num_labels)
        ).to(map_location).to(torch.bfloat16)

        #set_seed(config.seed)
        #instance.matformer_model.model.change_num_labels(instance.num_labels)
        return instance
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
        
        # if not hasattr(self.matformer_model, "classification_head"):
            #we actually do not directly call the classification head but we still require it as a 'type check'
            # raise Exception("The underneath model of a MatformerForSequenceClassification needs to have an attr 'classification_head'")
        
        hidden_states = self.matformer_model(input_ids, return_type='hidden')
        pooled_output = self.dropout(hidden_states.tensor)
        logits = self.classifier(pooled_output)
        
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
        
        return TokenClassifierOutput(loss=loss, logits=logits)


def register_matformer():
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoModelForTokenClassification
    
    AutoConfig.register("matformer", MatformerConfig)
    AutoModel.register(MatformerConfig, MatformerModel)
    AutoModelForCausalLM.register(MatformerConfig, MatformerForCausalLM)
    AutoModelForMaskedLM.register(MatformerConfig, MatformerForMaskedLM)
    AutoModelForSequenceClassification.register(MatformerConfig, MatformerForSequenceClassification)
    AutoModelForTokenClassification.register(MatformerConfig, MatformerForTokenClassification)


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


def push_to_hub(model, config_dict, repo_id, token=None, model_type='auto', clean_checkpoint=True):
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
        if clean_checkpoint:
            ckpt=torch.load(model.config._checkpoint_path,weights_only=False,map_location='cpu')
            ckpt_clean={
                "state_dict":ckpt["state_dict"],
                "hyper_parameters":ckpt["hyper_parameters"]
            }
            checkpoint_source="checkpoint_to_upload.ckpt"
            torch.save(ckpt_clean,checkpoint_source)            
            print(f"Saved temporary {checkpoint_source}")
        else:
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


def upload_to_huggingface(checkpoint,config,hf_name,hf_token=None,weights_only=True,model_type='auto',device='cpu',num_labels=2):
    with open(config) as f:
        config_dict = json.load(f)
    model = load_matformer_model(
        checkpoint, 
        config_dict, 
        model_type=model_type,
        map_location=device,
        num_labels=num_labels,
        load_mode='publication' if weights_only else 'full'
    )
    
    push_to_hub(
        model, 
        config_dict, 
        hf_name, 
        token=hf_token,
        model_type=model_type,
        clean_checkpoint=weights_only
    )	

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--hf_name", required=True)
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--weights_only",default=True, help="Upload only models'state dict and hyperparameters, ditch the rest (opt. state dict, dataloader, schedule)")
    parser.add_argument("--model_type", default="auto", choices=["auto", "causal", "masked", "classification"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num_labels", type=int, default=2)
    
    args = parser.parse_args()
    upload_to_huggingface(
        args.checkpoint,
        args.config,
        args.hf_name,
        hf_token=args.hf_token,
        weights_only=args.weights_only,
        model_type=args.model_type,
        device=args.device,
        num_labels=args.num_labels
    )
    

    



if __name__ == "__main__":
    main()
