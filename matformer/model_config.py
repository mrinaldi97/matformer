from dataclasses import dataclass, field
from typing import List, Literal

@dataclass
class ModelConfig:
    # core dims
    hidden_dim:             int                 
    ffn_factor:             float               # mlp dim = hidden_dim * ffn_factor
    n_layers:               int                 
    n_heads:                int                 
    # vocabulary & embeddings
    vocab_size:             int                 
    pad_id:                 int     
    bos_id: int
    eos_id: int           
    tie_word_embeddings:    bool                
    # normalization
    rms_norm_eps:           float               
    # attention & masking
    attention_type:         List[str]           # supported: ['sliding','causal','cloze','document']
    sliding_window_size:    int                 
    sliding_layers:         List[int]           # if sliding is partial, list of layers where to slide
    sliding_type:           Literal['full','disabled','partial']                                          
    max_seqlen:             int                 
    block_size_for_attention:int               
    # compilation 
    compile_flexattn:       bool                # if True, wrap MultiHeadAttention in torch.compile()
    # projection bias
    bias:                   bool                # should Linear projections include bias? 
    # 
    name:                   str             # model's name
    #
    training_objective:     str                # 'autoregressive' or 'masked'
    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)
