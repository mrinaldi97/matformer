"""
File 
transformer_functions.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from typing import Optional, List
from matformer.model_config import ModelConfig
#flex_attention = torch.compile(flex_attention,dynamic=True)
"""

Matformer implementation of self-attention
We need to support:
    * Attention score modification:
        * Alibi
        * Normal-causal mask (GPT-Like)
        * No mask (Bert-Like)
        * Sliding window attention
        * Custom (ex. attention on all the previous text, image also after)
    * Cross attention

"""
class MultiHeadAttention(nn.Module):
    def __init__(
                self,
                q_dim: int,
                k_dim: int,
                v_dim: int,
                tot_dim: int,
                nheads: int,
                bias: bool,
                #dropout: float=0.0, //Not supported by FlexAttention yet
                block_mask=None
                ):
        super().__init__()
        self.nheads=nheads
        assert tot_dim % self.nheads == 0, "Embedding dim is not divisible by nheads"
        self.tot_dim=tot_dim
        """
        The block mask can either be defined during the inizialization, in cases such as vanilla
        causal or sliding window, or directly passed during the forward, in cases such as document
        mask. In any case, this part of the implementation requires a review for efficiency
        """
        if block_mask is not None:
            self.set_mask=1
            self.block_mask=block_mask
        else:
            self.set_mask=0
        #self.dropout=dropout
        self.bias=bias
        self.qkv_samedim = q_dim==k_dim and q_dim==v_dim
        self.residual_dim=q_dim #Residual dim is used also as output dim
        
        if self.qkv_samedim:
            # If query, key and values have the same dimension, pack for better efficiency
            self.packed_proj=nn.Linear(self.residual_dim,3*tot_dim,bias=bias)
        else:
            # Three distinct projections
            self.q_proj=nn.Linear(q_dim,tot_dim,bias=bias)
            self.k_proj=nn.Linear(k_dim,tot_dim,bias=bias)
            self.v_proj=nn.Linear(v_dim,tot_dim,bias=bias)
            
        self.out_proj=nn.Linear(tot_dim,self.residual_dim,bias=bias)
        self.head_dim=tot_dim//self.nheads

    def forward(
                self,
                query_input: torch.Tensor,
                key_input: torch.Tensor,
                value_input: torch.Tensor,
                block_mask = None
            ) -> torch.Tensor:
        """
        Input Tensors:
        query_input: (Batch, Seqlen, Qdim)
        key_input: (Batch, Seqlen, Kdim)
        value_input: (Batch, Seqlen, Vdim)
        block_mask: BlockMask object
        Output:
        (Batch, Seqlen, Qdim) //Qdim is the query dimension as well as the
        "residual stream" dimension.

        Modifications:
        This forward functions allow for several modification compared to
        the standard Attention implementation.
        It can work in:
            * Causal-attention mode => for tasks like chat generation, decoder-only style
            * Bidirectional-attention mode => like BERT
            * Mixed attention mode => Like causal attention, but in some cases, for example
                for images generation and comprehension, the attention is allowed 
                to see al text behind (causal) but the entire image patch (bidirectional)

        Moreover, it supports:
            * ALiBi
            * Sliding Window Attention
            * Self-Attention
            * Cross-Attention

        """
        if self.set_mask==1:
            block_mask=self.block_mask
        qlen=query_input.shape[1]
        klen=key_input.shape[1]
        if self.qkv_samedim:
            if query_input is key_input and key_input is value_input:
                # We are in self-attention mode
                # Use of packed projections for efficiency
                result=self.packed_proj(query_input)
                query,key,value=torch.chunk(result,3,dim=-1)
            else:
                # We are in cross-attention mode, but same embedding dimensions
                # Extracting the weights from packed projections
                q_weight,k_weight,v_weight = torch.chunk(
                        self.packed_proj.weight,3,dim=0)
                if self.bias:
                    q_bias,k_bias,v_bias = torch.chunk(
                            self.packed_proj.bias,3,dim=0)
                else:
                    q_bias,k_bias,v_bias=None,None,None
                query=F.linear(query_input,q_weight,q_bias)
                key=F.linear(key_input,k_weight,k_bias)
                value=F.linear(value_input,v_weight,v_bias)
        else:
            # We are in cross-attention, different embedding dimensions
            query=self.q_proj(query_input)
            key=self.k_proj(key_input)
            value=self.v_proj(value_input)
        # Splitting for each head
        # (Batch, seqlen, tot_dim) => (Batch, seqlen, nhead, head_dim) => (Batch, nhead, seqlen, head_dim)
        query=query.unflatten(-1,[self.nheads, self.head_dim]).transpose(1,2)
        key=key.unflatten(-1,[self.nheads, self.head_dim]).transpose(1,2)
        value=value.unflatten(-1,[self.nheads, self.head_dim]).transpose(1,2)
        
        
        ### FLEX ATTENTION ###

        
        def generate_alibi_bias(H):
            #WARNING: THIS IS INEFFICIENT AND SHOULD BE FIXED, 
            # Alibi Bias
            # From https://github.com/pytorch-labs/attention-gym/examples/flex_attn.ipynb
            alibi_bias = []
            for h in range(H):
                alibi_bias.append(-((h + 1) * 8.0 / H))
            alibi_bias = torch.tensor(alibi_bias).to(query.device)
            alibi_bias = torch.exp2(alibi_bias).to(query.device)
            return alibi_bias.to(query.device)
            
        self.alibi_bias = generate_alibi_bias(self.nheads)
        def _alibi_score_mod(score, b, h, q_idx, kv_idx):
            return (score + self.alibi_bias[h] * (kv_idx - q_idx)).to(query.device)
        """
            * If attn_type == bidirectional, no mask
            * If attn_type == causal, causal mask
            * If attn_type == hybrid <= Causal attention, but allowed to see the next multimedia tokens
                                    up to the next non-multimedia token
            * Sliding window support
        """
        attn_output = flex_attention(query, key, value, score_mod=_alibi_score_mod,block_mask=block_mask)
        attn_output=attn_output.transpose(1,2).flatten(-2)
        return self.out_proj(attn_output)
     
class PackedSwiGLUFFN(nn.Module):
    #Adapted from https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html
    def __init__(
        self,
        config: ModelConfig
    ):
        super().__init__()
        dim = config.hidden_dim
        hidden_dim = int(config.hidden_dim * config.ffn_factor)  # Direct scaling
        self.w13 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        
    def forward(self, x):
        x1, x3 = torch.chunk(self.w13(x), 2, dim=-1)
        return self.w2(F.silu(x1) * x3)

"""     
class RMSNorm(nn.Module):
    #From https://github.com/Emericen/tiny-qwen
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_dim))
        self.variance_epsilon = config.rms_norm_eps
    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)
"""


