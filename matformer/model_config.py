from dataclasses import dataclass, field
from typing import List, Literal

@dataclass
class ModelConfig:
    # core dims
    hidden_size:             int                 
    ffn_factor:             float               # mlp dim = hidden_size * ffn_factor
    num_hidden_layers:               int                 
    num_attention_heads:                int                 
    # vocabulary & embeddings
    vocab_size:             int                 
    pad_token_id:                 int     
    bos_token_id: int
    eos_token_id: int           
    tie_word_embeddings:    bool                
    # normalization
    rms_norm_eps:           float               
    # attention & masking
    attention_type:         List[str]           # supported: ['sliding','causal','cloze','document']
    sliding_window_size:    int                 
    sliding_layers:         List[int]           # if sliding is partial, list of layers where to slide
    sliding_type:           Literal['full','disabled','partial']                                          
    max_position_embeddings:             int                 
    block_size_for_attention:int               
    # compilation 
    compile_flexattn:       bool                # if True, wrap MultiHeadAttention in torch.compile()
    # projection bias
    bias:                   bool                # should Linear projections include bias? 
    # 
    name:                   str             # model's name
    #
    training_objective:     str                # 'autoregressive' or 'masked'
    #
    is_causal: str
    alibi: str
    attn_impl: str # 'flash', 'flex', 'sdpa'
    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)
