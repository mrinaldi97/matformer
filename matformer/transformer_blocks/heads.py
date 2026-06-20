"""
File: matformer/transformer_blocks/heads.py
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
from matformer.matformer_tokenizers import ByteLevelTokenizer,MatformerTokenizer
from matformer.transformer_blocks import NakedTransformer
#Other
from dataclasses import replace
from warnings import warn
from typing import Optional, List, Literal, Any



class TransformerWithEmbeddingHead(MatformerModule):
    
    # Stable parameter names
    embed_tokens: "param_name:embed_tokens"
    embed_positions: "param_name:embed_positions"
    blocks: "param_name:blocks"
    available_hooks = ['pre_embed', 'post_embed']

    def __init__(self, config: ModelConfig, cache=None):
        super().__init__()
        self.config = config
        self.cache = ensure_cache_and_registry(cache)
        
        # Token embeddings
        self.embed_tokens = ModuleWrapper(
            self.cache.registry.create(
                "embedding", "embedding",
                num_embeddings= config.vocab_size,
                embedding_dim=config.hidden_size,
                padding_idx=config.pad_token_id
            )
        )
        
        # Learnable positional embeddings (if configured)
        self.embed_positions = (
            ModuleWrapper(
                self.cache.registry.create(
                    "embedding", "embedding",
                    num_embeddings=config.max_position_embeddings,
                    embedding_dim=config.hidden_size
                )
            )
            if  "learnable" in config.default_layer['positional_encoding']
            else None
        )
        
        # Core transformer
        self.blocks = NakedTransformer(config, cache=self.cache)
        self.hooks = self._init_hooks(config.hooks or {})
    def forward(self, x, **kwargs):
        """Embed tokens (+ positions) then process through transformer."""
        x = self._apply_hook('pre_embed', x)
        embeddings = self.embed_tokens(x) 
        embeddings = self._apply_hook('post_embed', embeddings)
        # Add learnable positional embeddings if configured
        if self.embed_positions is not None:
            batch_size, seq_len = x.tensor.shape
            position_ids = torch.arange(seq_len, dtype=torch.long, device=x.tensor.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            position_ids_wrapped = replace(x, tensor=position_ids)
            position_embeds = self.embed_positions(position_ids_wrapped)
            embeddings = replace(embeddings, tensor=embeddings.tensor + position_embeds.tensor)
        
        return self.blocks(embeddings, **kwargs)
    def get_cls_token(self, x, **kwargs):
        """
        A debug function. Probably will be removed in the future.
        Return the cls token (token in first position)
        
        """
        return self.forward(x,**kwargs).tensor[:, 0, :]
        

class TransformerWithLMHead(MatformerModule):
    """
    Complete language model with embeddings and language modeling head.
    Suitable for autoregressive (GPT-like) or masked (BERT-like) language models.
    Supports weight tying between embeddings and output projection.
    """
    
    # Stable parameter names
    lm_head: "param_name:lm_head"
    encoder: "param_name:encoder"
    
    def __init__(self, config: ModelConfig, tokenizer=None, device=None, cache=None):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.cache = ensure_cache_and_registry(cache)
        
        # Language modeling head (output projection)
        self.lm_head = ModuleWrapper(
            self.cache.registry.create(
                "linear", "linear",
                in_features=config.hidden_size,
                out_features=config.vocab_size
            )
        )
        
        # Transformer with embeddings
        self.encoder = TransformerWithEmbeddingHead(config, cache=self.cache)
        
        ################## WARNING !!! ##################
        # Due to a bug discovered too late, currently weight tying is allowed only
        # with the environment variable ALLOW_WEIGHT_TYING set to TRUE
        # This is extremely temporary: we have several checkpoints trained with
        # weight tying set up in the config but that actually didn't worked
        # until we fix the configurations file present in those checkpoints
        # for easiness the weight tying is disabled
        # Be aware of this if you are going to train a model with Matformer!
        ################## WARNING !!! ##################
        # Weight tying: share embeddings with output projection
        import os 
        allow_tying = os.environ.get("ALLOW_WEIGHT_TYING", "false").lower() == "true"

        if config.tie_word_embeddings and allow_tying:
            self.lm_head.module.inner.weight = self.encoder.embed_tokens.module.inner.weight
            
    def forward(self, x, return_type='logits', **kwargs):
        """Forward pass with optional return type.
        
        Args:
            x: Input token IDs
            return_type: 'logits' for language modeling head output,
                        'hidden' for transformer hidden states
        """
        hidden_states = self.encoder(x, **kwargs)
        
        if return_type == 'logits':
            return self.lm_head(hidden_states)
        else:
            return hidden_states
    def collate_generation_inputs(batch, pad_token_id, max_seq_len=None, varlen_strategy='unpadding'):
        if batch is None or len(batch) == 0:
            warn("Empty batch received in collate_generation_inputs")
            return None
        
        sequences = []
        for item in batch:
            if isinstance(item, dict):
                seq = item.get("input_ids", item.get("object", item))
            elif isinstance(item, torch.Tensor):
                seq = item.tolist() if item.dim() > 0 else [item.item()]
            elif isinstance(item, list):
                seq = item
            else:
                seq = [item]
            sequences.append(seq)
        
        if varlen_strategy == 'nested':
            return torch.nested.nested_tensor([torch.tensor(s) for s in sequences], layout=torch.jagged)
        
        if max_seq_len is None:
            max_seq_len = max(len(s) for s in sequences)
        
        padded_ids = []
        for seq in sequences:
            if len(seq) < max_seq_len:
                padded_ids.append(seq + [pad_token_id] * (max_seq_len - len(seq)))
            else:
                padded_ids.append(seq[:max_seq_len])
        
        tensors = torch.tensor(padded_ids, dtype=torch.long)
        padding_masks = (tensors == pad_token_id)
        
        sequence = PaddedTensor(tensor=tensors, padding_mask=padding_masks)
        
        if varlen_strategy == 'unpadding':
            sequence = sequence.unpad()
        
        return sequence
