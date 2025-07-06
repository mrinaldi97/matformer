"""
File transformer_functions.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from typing import Optional, List
from matformer.model_config import ModelConfig
from matformer.tensors_dataclasses import TensorDC, NormalTensor, PaddedTensor, UnpaddedTensor
from dataclasses import replace

from torch.nn.functional import scaled_dot_product_attention
import gc
try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func, flash_attn_varlen_func
    from flash_attn.modules.mha import get_alibi_slopes
    _is_flash_attn_available=True
except:
    _is_flash_attn_available=False
def printmem(text):
    device = torch.device('cuda:0')
    free, total = torch.cuda.mem_get_info(device)
    mem_used_MB = (total - free) / 1024 ** 2
    print(f"Memory at {text}: {mem_used_MB}")

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
                block_mask=None,
                attn_impl='flex',
                alibi=True
                ):
        super().__init__()
        self.nheads=nheads
        self.attn_impl=attn_impl
        self.alibi=alibi
        if _is_flash_attn_available:
              self.alibi_slopes = torch.tensor(get_alibi_slopes(nheads), device='cuda', dtype=torch.float32) #Precomputing alibi slopes
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
        """
        It would be much better to generate the alibi bias here. However, it was creating problems so it's temporarily in the forward.
        def generate_alibi_bias(nheads):
            # Alibi Bias
            # From https://github.com/pytorch-labs/attention-gym/examples/flex_attn.ipynb
            alibi_bias = []
            for h in range(nheads):
                alibi_bias.append(-((h + 1) * 8.0 / nheads))
            alibi_bias = torch.tensor(alibi_bias).to('cuda')
            self.alibi_bias = torch.exp2(alibi_bias).to('cuda')
            return alibi_bias.to('cuda')

        self.alibi_bias = generate_alibi_bias(self.nheads)
        """
    def dummy_get_query(self,x): # Temporary, debug function
        result=self.packed_proj(x)
        query,_,_=torch.chunk(result,3,dim=-1)
        return query

    def forward(
                self,
                query_input,
                key_input,
                value_input,
                block_mask = None
            ):
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
        #printmem("Beginning of attention")
        #TODO: INVESTIGATE WHY .detach().requires_grad_() is required with nested tensors! With normal padded tensors, it works also without
        if self.set_mask==1: # Check if the block mask is set in the __init__()
            block_mask=self.block_mask
            
        # Extracting the tensors from the TensorDC dataclass:
        
        query_tensor=query_input.tensor
        key_tensor=key_input.tensor
        value_tensor=value_input.tensor

        if self.attn_impl=='flex' or self.attn_impl=='sdpa':
            if isinstance(query_input,UnpaddedTensor):
                query_padded=query_input.pad()
                key_padded=key_input.pad()
                value_padded=value_input.pad()
                query_tensor=query_padded.tensor
                key_tensor=key_padded.tensor
                value_tensor=value_padded.tensor 
        qlen=query_tensor.shape[1]
        klen=key_tensor.shape[1]
        if self.qkv_samedim:
            if query_tensor is key_tensor and key_tensor is value_tensor:
                self_attn=True
                # We are in self-attention mode
                # Use of packed projections for efficiency
                result=self.packed_proj(query_tensor)
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
                query=F.linear(query_tensor,q_weight,q_bias)
                key=F.linear(key_tensor,k_weight,k_bias)
                value=F.linear(value_tensor,v_weight,v_bias)

        else:
            # We are in cross-attention, different embedding dimensions
            query=self.q_proj(query_tensor)
            key=self.k_proj(key_tensor)
            value=self.v_proj(value_tensor)
        # Splitting for each head
        # (Batch, seqlen, tot_dim) => (Batch, seqlen, nhead, head_dim) => (Batch, nhead, seqlen, head_dim)
        query=query.unflatten(-1,[self.nheads, self.head_dim]).transpose(1,2)
        key=key.unflatten(-1,[self.nheads, self.head_dim]).transpose(1,2)
        value=value.unflatten(-1,[self.nheads, self.head_dim]).transpose(1,2)

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


        if self.attn_impl=='flex':
                
            if self.alibi:
                attn_output = flex_attention(query, key, value, score_mod=_alibi_score_mod,block_mask=block_mask)
            else:
                attn_output = flex_attention(query, key, value, block_mask=block_mask)
                
        elif self.attn_impl=='sdpa':
            if query.is_nested:
                print("WARNING: Nested tensors doesn't support attn_mask in sdpa! DON'T TRUST THIS EXECUTION")
                attn_output=scaled_dot_product_attention(query,key,value)
            else:
                if self.alibi:
                    L, S = query.shape[-2], key.shape[-2]
                    pos_bias = self.alibi_bias.view(-1, 1, 1) * (torch.arange(S, device=query.device) - torch.arange(L, device=query.device).view(-1, 1))
                    if block_mask is not None:
                        attn_mask = torch.where(block_mask.unsqueeze(0), pos_bias, float('-inf'))
                    else:
                        attn_mask=pos_bias
                    attn_output = scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, is_causal=False)
                else:
                    attn_output=scaled_dot_product_attention(query, key, value, attn_mask=block_mask, is_causal=False)
        elif self.attn_impl=='flash' and _is_flash_attn_available:
                """
                Funzione sperimentale
                1) Mettere anche la versione packed, facile ma farlo in modo pulito
                    qkv: (batch_size, seqlen, 3, nheads, headdim)
                    flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False,
                                  window_size=(-1, -1), alibi_slopes=None, deterministic=False):
                2) Evitare la doppia trasposizione 2,1 ma allo stesso tempo evitare di impiastricciare il codice sopra con un if
                
                3) Capire cosa supporta... ok alibi (ma diversamente da come fatto fin'ora), ok causal, ok sliding window.
                   Non gli si può però passare una attention mask personalizzata
                    flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                        window_size=(-1, -1), alibi_slopes=None, deterministic=False):
                """           
                query=query.transpose(1,2)
                key=key.transpose(1,2)
                value=value.transpose(1,2)
                if isinstance(query_input,UnpaddedTensor):
                    attn_output=flash_attn_varlen_func(query,key,value,
                        cu_seqlens_q=query_input.cu_seqlens.to('cuda'), cu_seqlens_k=key_input.cu_seqlens.to('cuda'),
                        max_seqlen_q=query_input.max_seq_len, max_seqlen_k=key_input.max_seq_len,alibi_slopes=self.alibi_slopes,causal=True).transpose(1,2)
                else:
                    attn_output=flash_attn_func(query,key,value,alibi_slopes=self.alibi_slopes,causal=True).transpose(1,2)
        else:
            print("Implementazione non supportata. Disponibilità Flash attention: ", _is_flash_attn_available)
            
        
        attn_output=attn_output.transpose(1,2).flatten(-2)
        
        if isinstance(query_input,UnpaddedTensor) and self.attn_impl != 'flash':
            return replace(query_padded,tensor=self.out_proj(attn_output)).unpad()
        else:
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

    def forward(self, _input):
        x=_input.tensor
        x1, x3 = torch.chunk(self.w13(x), 2, dim=-1)
        _input.tensor=self.w2(F.silu(x1) * x3)
        return _input

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


