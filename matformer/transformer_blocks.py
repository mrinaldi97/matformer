"""
File: matformer/transformer_blocks.py
"""
import torch
import torch.nn as nn
from matformer.transformer_functions import MultiHeadAttention, PackedSwiGLUFFN
from matformer.tensors_dataclasses import TensorDC, NormalTensor, PaddedTensor, UnpaddedTensor, ModuleWrapper
from torch.nn import RMSNorm
from matformer.model_config import ModelConfig  
from matformer.utils import LpPooling, MaskBuilder
from functools import partial, reduce
from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    create_block_mask,
    create_nested_block_mask,
    create_mask,
    and_masks,
    or_masks,
    noop_mask
)


#flex_attention = torch.compile(flex_attention) # Chiarire questione compilazione (dove? di che tipo? migliora o peggiora? in che casi farla?)

class TransformerBlock(nn.Module):
    """ A transformer self-attention block
        It applies a pre layernorm 
        A self-attention layer
        A SwiGLU Mlp Layer
        A post layer norm
        It takes all the necessary configuration from the ModelConfig object
        The block_mask for the attention can be passed either at the init or during the forward
    """
    
    def __init__(self, config: ModelConfig, block_mask=None):
        super().__init__()
        self.input_layernorm = ModuleWrapper(RMSNorm(normalized_shape=config.hidden_dim,eps=config.rms_norm_eps,elementwise_affine=True))
        self.self_attn = MultiHeadAttention(bias=config.bias, q_dim=config.hidden_dim, k_dim=config.hidden_dim, v_dim=config.hidden_dim, tot_dim=config.hidden_dim, nheads=config.n_heads, block_mask=block_mask, attn_impl=config.attn_impl, alibi=config.alibi, is_causal=config.is_causal)      
        self.post_attention_layernorm = ModuleWrapper(RMSNorm(normalized_shape=config.hidden_dim,eps=config.rms_norm_eps,elementwise_affine=True))
        self.mlp = PackedSwiGLUFFN(config)
    def forward(self, x, block_mask=None):
        x = self.input_layernorm(x)
        x = x + self.self_attn(query_input=x, key_input=x, value_input=x, block_mask=block_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x





class NakedTransformer(nn.Module):
    """
    This transformer implementation purposely misses the embedding
    as well as the "unembedding" layer.
    The reason is that is a Transformer meant to run only on "patches".
    It applies n transformer blocks as defined in the ModelConfig
    Still needs some revisions:
        1) High VRAM consumption with Flex Attention and in particular if nested tensors are used;
        2) A decision should be made about where to compute block masks
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        #If attention is not flash, it's a good idea to cache the block mask:
        if config.attention_type != 'flash':
            self.mask_builder = MaskBuilder(config)
            self.block_mask=None
            self.sliding_mask=None          
        self.norm = ModuleWrapper(RMSNorm(normalized_shape=config.hidden_dim, eps=config.rms_norm_eps, elementwise_affine=True))
        self.layers = nn.ModuleList()
        for _ in range(config.n_layers):
            self.layers.append(TransformerBlock(config=config)) 

        self.config.max_seqlen=self.config.max_seqlen -1 # Da ricordarsi perchè e dove serviva, trovare in caso soluzione più pulita             
        """
        # Generate mask templates in __init__ for "simple" cases (batch invariant cases, no document)
        THIS IS DISABLED AFTER SWITCHING TO NESTED TENSOR LAYOUT, BECAUSE THEY NEED TO BE RECOMPUTED EVERY TIME
        HOWEVER, THIS COULD SPEED UP THE MODEL, SO LET'S RETHINK ABOUT THIS AND EVENTUALLY PUT SOME CONDITION
        if 'causal' in self.config.attention_type:
            self.block_mask=self.mask_builder.build_mask_tensor(self.config.max_seqlen, self.config.max_seqlen, attention_types=['causal'],device=self.device)
        if 'sliding' in self.config.attention_type:
            if 'causal' in self.config.attention_type:
                self.block_mask=self.mask_builder.build_mask_tensor(self.config.max_seqlen, self.config.max_seqlen, attention_types=['sliding','causal'], is_sliding=True,device=self.device)
            else:
                self.block_mask=self.mask_builder.build_mask_tensor(self.config.max_seqlen, self.config.max_seqlen, attention_types=['sliding'], is_sliding=True,device=self.device)    
        """
    def forward(self, x, y_cross=None, document_mask=None, cloze_mask=None, inference_fix=False):
        
        q_len=x.original_seq_len if isinstance(x,UnpaddedTensor) else x.shape[1]
        kv_len = y_cross.shape[1] if y_cross is not None else q_len  # If we are not in cross-attention settings, we take x for both query and kv
        """
         We have to decide in the forward if employing the masks generated during the __init__ (much faster)
         or if we need to generate new masks in cases such as:
         1) Document attention type and doc.mask passed;
         2) The model is used at an higher seq_len
        """     
        assert self.config.sliding_type in ['full','disabled','partial'], "Invalid sliding type config."
        if False: #THIS IS DISABLED AFTER SWITCHING TO NESTED TENSOR LAYOUT, BECAUSE THEY NEED TO BE RECOMPUTED EVERY TIME
        #if document_mask is not None or cloze_mask is not None or q_len>self.config.max_seqlen or inference_fix==True:
            # In these cases I have to regenerate the masks
            #with torch.no_grad():
            #    dummy_query=self.self_attn.dummy_get_query(x=x)
            if self.config.sliding_type != 'disabled':
                sliding_mask=self.mask_builder.build_mask_tensor(query=x, kv=y_cross, attention_types=self.config.attention_type, is_sliding=True,nested=x.tensor.is_nested)
            if self.config.sliding_type != 'full':
                block_mask=self.mask_builder.build_mask_tensor(query=x, kv=y_cross, attention_types=self.config.attention_type, is_sliding=False,nested=x.tensor.is_nested)             
        else:
            block_mask=self.block_mask
            sliding_mask=self.block_mask
            
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx in self.config.sliding_layers or self.config.sliding_type=='full' :
                x = layer(x, block_mask=sliding_mask)
            else:
                x = layer(x, block_mask=block_mask)
        x = self.norm(x)

        return x
        
class TransformerWithEmbeddingHead(nn.Module):
    """
    Adding an embedding layer at the beginning
    """
    def __init__(self,config: ModelConfig):
        super().__init__()
        self.embed_tokens = ModuleWrapper(nn.Embedding(num_embeddings=config.vocab_size,embedding_dim=config.hidden_dim,padding_idx=config.pad_id))
        self.transformer = NakedTransformer(config)
    def forward(self,x, **kwargs): 
        embeddings=self.embed_tokens(x)
        return self.transformer(embeddings,**kwargs)
    
class TransformerWithLMHead(nn.Module):
    """
    Adding an LM Head to TransformerWithEmbeddingHead. This is enough for Bert-like/GPT-like models.
    """
    def __init__(self,config: ModelConfig):
        super().__init__()      
        self.lm_head = ModuleWrapper(nn.Linear(config.hidden_dim, config.vocab_size))
        self.transformer = TransformerWithEmbeddingHead(config)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.transformer.embed_tokens.weight

    def forward(self,x,**kwargs):

        x=self.transformer(x,**kwargs)
        x= self.lm_head(x)
        return x
