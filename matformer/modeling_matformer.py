import torch
from dataclasses import asdict, fields
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutput, 
    MaskedLMOutput, 
    SequenceClassifierOutput,
    CausalLMOutput
)
import torch.nn as nn
import sys
sys.path.append('../') #temporary
from matformer.model_config import ModelConfig
from matformer.utils import dataclass_from_dict
from matformer.transformer_blocks import (
    NakedTransformer,
    TransformerWithEmbeddingHead, 
    TransformerWithLMHead,
    TransformerWithClassificationHead
)

class HF_Matformer(PretrainedConfig):
    model_type = "matformer"
    
    def __init__(self, **kwargs):
        # Convert dataclass fields to individual config attributes
        matformer_config = dataclass_from_dict(kwargs, ModelConfig)
        config_dict = asdict(matformer_config)
        kwargs.update(config_dict)
        super().__init__(**kwargs)
    
    def to_matformer_config(self) -> ModelConfig:
       
        # Recreate the ModelConfig from the HF model's dictionary if the fields are in the ModelConfig required fields
        return dataclass_from_dict({k: v for k, v in self.__dict__.items() if k in {f.name for f in fields(ModelConfig)}},ModelConfig)

    
    def to_dict(self) -> dict:
        config_dict = super().to_dict()
        
        model_config_fields = {f.name for f in fields(ModelConfig)}
        for field_name in model_config_fields:
            if hasattr(self, field_name):
                config_dict[field_name] = getattr(self, field_name)
        
        return config_dict
    
    def _diff_dict(self, config_dict: dict) -> dict:
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: dict, **kwargs) -> "HF_Matformer":
        return cls(**config_dict, **kwargs)
    
    @classmethod
    def from_matformer_config(
        cls, mat_conf: ModelConfig, **overrides
    ) -> "HF_Matformer":
        params = asdict(mat_conf)
        params.update(overrides)
        return cls(**params)

class MatformerModel(PreTrainedModel):
    config_class = HF_Matformer
    base_model_prefix = "transformer"
    
    def __init__(self, config: HF_Matformer):
        super().__init__(config)
        matformer_config = config.to_matformer_config()
      
        
        self.post_init()
    
    def forward(self, 
                hidden_states,
                output_hidden_states=None,
                output_attentions=None,
                return_dict=None,
                **kwargs):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        hidden_states = self.transformer(hidden_states, **kwargs)
        
        if not return_dict:
            return (hidden_states,)
        
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=None,  # TODO
            attentions=None,     # TODO
        )

class MatformerWithEmbeddings(PreTrainedModel):
    config_class = HF_Matformer
    base_model_prefix = "model"
    
    def __init__(self, config: HF_Matformer):
        super().__init__(config)
        matformer_config = config.to_matformer_config()

        self.post_init()
    
    def forward(self,
                input_ids=None,
                inputs_embeds=None,
                output_hidden_states=None,
                output_attentions=None,
                return_dict=None,
                **kwargs):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if input_ids is not None:
            model_input = input_ids
        elif inputs_embeds is not None:
            raise NotImplementedError("inputs_embeds not yet supported - requires embedding layer bypass")
        else:
            raise ValueError("input_ids must be provided")
        
        hidden_states = self.model(model_input, **kwargs)
        
        if not return_dict:
            return (hidden_states,)
        
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=None,
            attentions=None,
        )
    
    def get_input_embeddings(self):
        return self.model.embed_tokens.module
    
    def set_input_embeddings(self, value):
        self.model.embed_tokens.module = value
    
    def resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)
        self.config.vocab_size = new_num_tokens
        return self.get_input_embeddings()

class MatformerForCausalLM(PreTrainedModel):
    config_class = HF_Matformer
    base_model_prefix = "model"
    
    def __init__(self, config: HF_Matformer):
        super().__init__(config)
        matformer_config = config.to_matformer_config()

        self.post_init()
    
    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)
    
    def get_input_embeddings(self):
        return self.model.transformer.embed_tokens.module
    
    def set_input_embeddings(self, value):
        self.model.transformer.embed_tokens.module = value

class MatformerForMaskedLM(PreTrainedModel):
    config_class = HF_Matformer
    base_model_prefix = "model"
    
    def __init__(self, config: HF_Matformer):
        super().__init__(config)
        matformer_config = config.to_matformer_config()
        self.post_init()
    
    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                inputs_embeds=None,
                output_hidden_states=None,
                output_attentions=None,
                return_dict=None,
                **kwargs):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if input_ids is not None:
            model_input = input_ids
        elif inputs_embeds is not None:
            raise NotImplementedError("inputs_embeds not yet supported")
        else:
            raise ValueError("input_ids must be provided")
        
        logits = self.model(model_input, **kwargs)
        
        loss = None
        if labels is not None:
            # Calculate masked LM loss (only on masked positions)
            loss_fct = nn.CrossEntropyLoss()
            
            if attention_mask is not None:
                # Apply attention mask and only calculate loss on non-padded tokens
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.config.vocab_size)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output
        
        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
    
    def get_input_embeddings(self):
        return self.model.transformer.embed_tokens.module
    
    def set_input_embeddings(self, value):
        self.model.transformer.embed_tokens.module = value

class MatformerForSequenceClassification(PreTrainedModel):
    config_class = HF_Matformer
    base_model_prefix = "model"
    
    def __init__(self, config: HF_Matformer):
        super().__init__(config)
        

        if not hasattr(config, 'num_labels'):
            raise ValueError("config must have 'num_labels' attribute for TransformerWithClassificationHead")
        if not hasattr(config, 'pooling_type'):
            config.pooling_type = 'cls'  # Default to CLS pooling
        
        matformer_config = config.to_matformer_config()
        

        self.model = TransformerWithClassificationHead(
            config=matformer_config,
            tokenizer=getattr(config, 'tokenizer', None),
            pooling_type=config.pooling_type,
            num_features=config.num_labels
        )
        self.post_init()
    
    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                inputs_embeds=None,
                output_hidden_states=None,
                output_attentions=None,
                return_dict=None,
                **kwargs):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if input_ids is not None:
            model_input = input_ids
        elif inputs_embeds is not None:
            model_input = inputs_embeds
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")
        
        logits = self.model(model_input, attention_mask=attention_mask)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
    
    def get_input_embeddings(self):
        return self.model.embed_tokens.module
    
    def set_input_embeddings(self, value):
        self.model.embed_tokens.module = value

# Utility function to load checkpoints
def load_checkpoint_file(
    checkpoint_path: str,
    model_cls=MatformerForCausalLM,
    config_cls=HF_Matformer,
    device: str = "cpu",
    **override_configs
) -> PreTrainedModel:
    """Load a PyTorch Lightning checkpoint into a HuggingFace model"""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    mat_conf: ModelConfig = ckpt["hyper_parameters"]["config"]
    hf_cfg = config_cls.from_matformer_config(mat_conf, **override_configs)
    model = model_cls(hf_cfg)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model

# TODO: Model registration for AutoModel classes
# from transformers import (
#     AutoConfig,
#     AutoModel, 
#     AutoModelForCausalLM,
#     AutoModelForMaskedLM,
#     AutoModelForSequenceClassification
# )
# 
# AutoConfig.register("matformer", HF_Matformer)
# AutoModel.register(HF_Matformer, MatformerWithEmbeddings)
# AutoModelForCausalLM.register(HF_Matformer, MatformerForCausalLM)
# AutoModelForMaskedLM.register(HF_Matformer, MatformerForMaskedLM)
# AutoModelForSequenceClassification.register(HF_Matformer, MatformerForSequenceClassification)
