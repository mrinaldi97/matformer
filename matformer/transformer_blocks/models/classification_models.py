"""
File: matformer/transformer_blocks/models/classification_models.py
"""
import sys
sys.path.append('../')
#Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
#Matformer
from matformer.matformer_module import MatformerModule
from matformer.tensors_dataclasses import TensorDC, NormalTensor, PaddedTensor, UnpaddedTensor, ModuleWrapper
from matformer.model_config import ModelConfig  
from matformer.utils.matformer_cache import ensure_cache_and_registry, CachedStuff
from matformer.matformer_registry import registry
from matformer.transformer_blocks import TransformerWithLMHead, TransformerWithEmbeddingHead
#Other
from dataclasses import replace
from copy import deepcopy
from warnings import warn
from typing import Optional, List, Literal, Any


class _ClassificationModel(MatformerModule):
    def __init__(self, config: ModelConfig, tokenizer=None, pooling_type='cls',
                 num_features=None, head_type='linear', head_config=None,
                 pre_head=None, layers_to_remove=0, device=None, cache=None):
        super().__init__()
        self.cache = ensure_cache_and_registry(cache)
        self.config = config
        self.tokenizer = tokenizer
        self.pooling_type = pooling_type
        self.head_type  = head_type  or getattr(config, 'head_type',  'linear')
        self.head_config = head_config if head_config is not None else getattr(config, 'head_config', {})
        self.pre_head   = pre_head   or getattr(config, 'pre_head',   None)
        self.layers_to_remove = layers_to_remove or getattr(config, 'layers_to_remove', 0)
        self.num_features = (
            num_features or
            getattr(config, 'num_features', None) or
            2
        )
        self.classifier_dropout_p = getattr(config, "classifier_dropout_p", 0.1)
        self.classifier_dropout_inplace = getattr(config, "classifier_dropout_inplace", False)

        self.dropout = ModuleWrapper(
            self.cache.registry.create(
                "dropout", "dropout",
                p=self.classifier_dropout_p,
                inplace=self.classifier_dropout_inplace
            )
        )

        encoder_config = config
        if self.layers_to_remove > 0: # In Matformer it's easy to attach the classification head to a layer that is not the last
            encoder_config = deepcopy(config)
            encoder_config.num_hidden_layers -= self.layers_to_remove
        self.encoder = TransformerWithLMHead(config=encoder_config, cache=self.cache) # TODO: EmbeddingHead not working (probably bad loading)

        self.classification_transformer = None # Similarly, it is easy to add new transformer layers before the classification head
        if self.pre_head == 'transformer':
            pre_cfg = deepcopy(config)
            pre_cfg.num_hidden_layers = self.head_config.get('transformer_layers', 1)
            self.classification_transformer = NakedTransformer(pre_cfg, cache=self.cache)

        self.classification_head = self._build_head(config.hidden_size, self.num_features)
    def _build_head(self, in_features, out_features):
        if self.head_type == 'linear':
            return ModuleWrapper(
                self.cache.registry.create(
                    "linear", "linear",
                    in_features=in_features,
                    out_features=out_features
                )
            )
        return ModuleWrapper(
            self.cache.registry.create(
                "classification_head", self.head_type,
                in_features=in_features,
                out_features=out_features,
                **{k: v for k, v in self.head_config.items() if k != 'transformer_layers'}
            )
        )

    def freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def change_num_labels(self, new_num_labels):
        """Change the number of output features of the classification head."""
        self.num_features = new_num_labels
        self.classification_head = self._build_head(self.config.hidden_size, new_num_labels)
        reference_param = next(self.encoder.parameters())
        # Move to the same device and dtype as the model
        self.classification_head = self.classification_head.to(
            device=reference_param.device,
            dtype=reference_param.dtype
        )
        print(f"Number of labels changed to {new_num_labels}")
    

class TransformerWithClassificationHead(_ClassificationModel):
    classification_head: "param_name:classifier"
    dropout: "param_name:dropout"
    classification_transformer: "param_name:classification_transformer"
    encoder: "transparent" 
    def _pool(self, hidden_states, attention_mask=None):
        """Pool hidden states (PaddedTensor) into a single vector for classification."""
        if self.pooling_type == 'cls':
            # [CLS] in pos. 0
            return hidden_states.tensor[:, 0, :]
        elif self.pooling_type == 'mean':
            #TODO: check if the mask works
            if attention_mask is None:
                return hidden_states.tensor.mean(dim=1)
            mask_expanded = attention_mask.unsqueeze(-1).expand(
                hidden_states.tensor.size()
            ).to(hidden_states.dtype)
            sum_hidden = torch.sum(hidden_states.tensor * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_hidden / sum_mask
        else:
            raise ValueError(f"{self.pooling_type} not in 'cls','mean'")
    def forward(self, x, attention_mask=None, **kwargs):
        """
        Args:
            x: Input tensor or PaddedTensor
            attention_mask: Optional explicit mask. If None, extracted from PaddedTensor
        """

        #print(f"[FORWARD] x type: {type(x)}")
        #if isinstance(x, PaddedTensor):
        #    print(f"[FORWARD] x.tensor.shape: {x.tensor.shape}")
        #    print(f"[FORWARD] x.padding_mask.shape: {x.padding_mask.shape}")

        # Remove return_type before passing to encoder
        # TODO: perchè?
        kwargs.pop('return_type', None)
        if self.classification_transformer is not None:
            hidden_states = self.classification_transformer(self.encoder(x, return_type='hidden', **kwargs)).pad()
        else:
            hidden_states = self.encoder(x, return_type='hidden', **kwargs).pad() # (B,S,D)
        # Warning: hidden states are now a PaddedTensor so that pooling can be accomplished
        pooled_output = self._pool(hidden_states, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classification_head(pooled_output)
        return logits            
            

class TransformerWithTokenClassificationHead(MatformerModule):
    classification_head: "param_name:classifier"
    dropout: "param_name:dropout"
    classification_transformer: "param_name:classification_transformer"
    encoder: "transparent"
    def forward(self, x, attention_mask=None, **kwargs):
        kwargs.pop('return_type', None)
        if self.classification_transformer is not None:
            hidden_states = self.classification_transformer(self.encoder(x, **kwargs)).tensor
        else:
            hidden_states = self.encoder(x, **kwargs).tensor
        hidden_states = self.dropout(hidden_states)
        logits = self.classification_head(hidden_states)
        return logits

