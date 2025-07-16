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
try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func, flash_attn_varlen_func
    from flash_attn.modules.mha import get_alibi_slopes
    _is_flash_attn_available=True
except:
    _is_flash_attn_available=False

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
                attn_impl='flash',
                alibi=True,
                causal=True,
                device='cuda'
                ):
        super().__init__()
        self.nheads=nheads
        self.device=device
        self.attn_impl=attn_impl
        self.alibi=alibi
        self.is_causal=causal
        if attn_impl=='flash' and _is_flash_attn_available and alibi:
              self.alibi_slopes = torch.tensor(get_alibi_slopes(nheads), device=self.device, dtype=torch.float32) #Precomputing alibi slopes
        assert tot_dim % self.nheads == 0, "Embedding dim is not divisible by nheads"
        self.tot_dim=tot_dim
        """
        The block mask can either be defined during the inizialization, in cases such as vanilla
        causal or sliding window, or directly passed during the forward, in cases such as document
        mask. In any case, this part of the implementation requires a review for efficiency
        """
        self.set_mask = int(block_mask is not None)
        self.block_mask = block_mask
        #self.dropout=dropout
        self.bias=bias
        self.qkv_samedim = q_dim==k_dim and q_dim==v_dim
        self.residual_dim=q_dim #Residual dim is used also as output dim
        self.head_dim=tot_dim//self.nheads

        if self.qkv_samedim:
            self.packed_proj=nn.Linear(self.residual_dim,3*tot_dim,bias=bias)
        else:
            self.q_proj=nn.Linear(q_dim,tot_dim,bias=bias)
            self.k_proj=nn.Linear(k_dim,tot_dim,bias=bias)
            self.v_proj=nn.Linear(v_dim,tot_dim,bias=bias)

        self.out_proj=nn.Linear(tot_dim,self.residual_dim,bias=bias)

    def _project(self, q, k, v):
        if self.qkv_samedim:
            if q is k and k is v:
                result = self.packed_proj(q)
                return torch.chunk(result, 3, dim=-1)
            else:
                w = torch.chunk(self.packed_proj.weight, 3, dim=0)
                b = torch.chunk(self.packed_proj.bias, 3, dim=0) if self.bias else (None, None, None)
                return (F.linear(q, w[0], b[0]), F.linear(k, w[1], b[1]), F.linear(v, w[2], b[2]))
        else:
            return self.q_proj(q), self.k_proj(k), self.v_proj(v)

    def _reshape_heads(self, x): # (B, S, D) -> (B, H, S, Hd)
        return x.unflatten(-1, [self.nheads, self.head_dim]).transpose(1, 2)

    def _maybe_pad(self, x): return x.pad() if isinstance(x, UnpaddedTensor) else x
    def _maybe_unpad(self, x, orig): return replace(orig, tensor=x).unpad() if isinstance(orig, UnpaddedTensor) else x

    def _flash_forward(self, q, k, v, query_input, key_input):
        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))  # B, S, H, Hd
        if isinstance(query_input, UnpaddedTensor):
            return flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=query_input.cu_seqlens.to(self.device),
                cu_seqlens_k=key_input.cu_seqlens.to(self.device),
                max_seqlen_q=query_input.max_seq_len,
                max_seqlen_k=key_input.max_seq_len,
                alibi_slopes=self.alibi_slopes,
                causal=self.is_causal
            ).transpose(1, 2)
        else:
            return flash_attn_func(q, k, v, alibi_slopes=self.alibi_slopes, causal=self.is_causal).transpose(1, 2)

    def _flex_forward(self, q, k, v, block_mask):
        if self.alibi:
            def generate_alibi_bias(H):
                #WARNING: THIS IS INEFFICIENT AND SHOULD BE FIXED,
                # Alibi Bias
                # From https://github.com/pytorch-labs/attention-gym/examples/flex_attn.ipynb
                alibi_bias = []
                for h in range(H):
                    alibi_bias.append(-((h + 1) * 8.0 / H))
                alibi_bias = torch.tensor(alibi_bias).to(q.device)
                alibi_bias = torch.exp2(alibi_bias).to(q.device)
                return alibi_bias.to(q.device)

            self.alibi_bias = generate_alibi_bias(self.nheads)

            def _alibi_score_mod(score, b, h, q_idx, kv_idx):
                return (score + self.alibi_bias[h] * (kv_idx - q_idx)).to(q.device)

            return flex_attention(q, k, v, score_mod=_alibi_score_mod, block_mask=block_mask)
        else:
            return flex_attention(q, k, v, block_mask=block_mask)

    def _sdpa_forward(self, q, k, v, block_mask):
        if self.alibi:
            L, S = q.shape[-2], k.shape[-2]
            pos_bias = self.alibi_bias.view(-1, 1, 1) * (
                torch.arange(S, device=q.device) - torch.arange(L, device=q.device).view(-1, 1)
            )
            attn_mask = torch.where(block_mask.unsqueeze(0), pos_bias, float('-inf')) if block_mask is not None else pos_bias
            return scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        else:
            return scaled_dot_product_attention(q, k, v, attn_mask=block_mask, is_causal=self.is_causal)

    def forward(self, query_input, key_input, value_input, block_mask = None):
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
        # Common parts
        query_tensor=query_input.tensor
        key_tensor=key_input.tensor
        value_tensor=value_input.tensor

        query, key, value = self._project(query_tensor, key_tensor, value_tensor)
        query, key, value = map(self._reshape_heads, (query, key, value)) # B, H, S, Hd

        if self.attn_impl=='flash':
            attn_output = self._flash_forward(query, key, value, query_input, key_input)

        else:
            if self.set_mask == 1:
                block_mask = self.block_mask

            if isinstance(query_input, UnpaddedTensor):
                query_input = self._maybe_pad(query_input)
                key_input = self._maybe_pad(key_input)
                value_input = self._maybe_pad(value_input)
                query_tensor, key_tensor, value_tensor = query_input.tensor, key_input.tensor, value_input.tensor
                query, key, value = self._project(query_tensor, key_tensor, value_tensor)
                query, key, value = map(self._reshape_heads, (query, key, value))

            if self.attn_impl == 'flex':
                attn_output = self._flex_forward(query, key, value, block_mask)
            else:  # SDPA
                attn_output = self._sdpa_forward(query, key, value, block_mask)

            attn_output = attn_output.transpose(1, 2)  # back to (B, S, H, Hd)

        attn_output = attn_output.flatten(-2)
        return self._maybe_unpad(self.out_proj(attn_output), query_input)

      
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


