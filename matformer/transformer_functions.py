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
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func, flash_attn_varlen_func, flash_attn_varlen_qkvpacked_func
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
                hidden_size: int,
                nheads: int,
                bias: bool,
                #dropout: float=0.0, //Not supported by FlexAttention yet
                block_mask=None,
                attn_impl='flash',
                alibi=True,
                is_causal=True,
                device='cuda'  
                ):
        super().__init__()
        self.nheads=nheads
        self.attn_impl=attn_impl
        self.alibi=alibi
        self.is_causal=is_causal
        assert hidden_size % self.nheads == 0, "Embedding dim is not divisible by nheads"
        self.hidden_size=hidden_size
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
        self.q_dim=q_dim
        self.head_dim=hidden_size//self.nheads
        
        if attn_impl=='flash' and _is_flash_attn_available and alibi:
            #For multi gpu training, register alibi_slopes as buffer instead of creating tensor 
            alibi_slopes = torch.tensor(get_alibi_slopes(nheads), dtype=torch.float32)
            self.register_buffer('alibi_slopes', alibi_slopes)
        else:
            self.register_buffer('alibi_slopes', None)  
            
        if self.qkv_samedim:
            self.packed_proj=nn.Linear(self.q_dim,3*hidden_size,bias=bias)
        else:
            self.q_proj=nn.Linear(q_dim,hidden_size,bias=bias)
            self.k_proj=nn.Linear(k_dim,hidden_size,bias=bias)
            self.v_proj=nn.Linear(v_dim,hidden_size,bias=bias)

        self.out_proj=nn.Linear(hidden_size,self.q_dim,bias=bias)
    def _project_packed(self,qkv):
        return self.packed_proj(qkv)
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
    def _maybe_unpad(self, x, orig):
        if isinstance(orig, UnpaddedTensor) and not isinstance(x, UnpaddedTensor):
            return replace(orig, tensor=x)
        return x

    def _flash_forward(self, q=None, k=None, v=None, qkv=None, query_input=None, key_input=None, packed=False, sliding=False, sliding_window_size=None):
        # Inputs: (B,H,S,Hd)
        if sliding:
            sliding_window=(-sliding_window_size,-sliding_window_size)
        else:
            sliding_window=(-1,-1) #From flash attn docs, this disables sliding window
        if not packed:
            q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))  # [B], S, H, Hd
        if isinstance(query_input, UnpaddedTensor):
            #(S, H, Hd).
            if packed:
                return flash_attn_varlen_qkvpacked_func(
                    qkv,
                    cu_seqlens=query_input.cu_seqlens.type_as(qkv),  
                    max_seqlen=query_input.max_seq_len,
                    alibi_slopes=None,
                    causal=self.is_causal,
                    window_size=sliding_window
                )               
            else:
                return flash_attn_varlen_func(
                    q, k, v,
                    cu_seqlens_q=query_input.cu_seqlens.type_as(q),  
                    cu_seqlens_k=key_input.cu_seqlens.type_as(k),    
                    max_seqlen_q=query_input.max_seq_len,
                    max_seqlen_k=key_input.max_seq_len,
                    alibi_slopes=self.alibi_slopes,
                    causal=self.is_causal,
                    window_size=sliding_window
                )
        else:
            #Inputs: (B,S,H,Hd)
            # Outputs: out: (B,S,H,Hd).
            if packed:
                return flash_attn_qkvpacked_func(qkv, alibi_slopes=self.alibi_slopes, causal=self.is_causal, window_size=sliding_window)
            else:
                return flash_attn_func(q, k, v, alibi_slopes=self.alibi_slopes, causal=self.is_causal, window_size=sliding_window)

    def _flex_forward(self, q, k, v, block_mask):
        if self.alibi:
            def generate_alibi_bias(H):
                #WARNING: THIS IS INEFFICIENT AND SHOULD BE FIXED,
                # Alibi Bias
                # From https://github.com/pytorch-labs/attention-gym/examples/flex_attn.ipynb
                alibi_bias = []
                for h in range(H):
                    alibi_bias.append(-((h + 1) * 8.0 / H))
                alibi_bias = torch.tensor(alibi_bias)  
                alibi_bias = alibi_bias.type_as(q)     
                alibi_bias = torch.exp2(alibi_bias).type_as(q)  
                return alibi_bias

            self.alibi_bias = generate_alibi_bias(self.nheads)

            def _alibi_score_mod(score, b, h, q_idx, kv_idx):
                return (score + self.alibi_bias[h] * (kv_idx - q_idx))  

            return flex_attention(q, k, v, score_mod=_alibi_score_mod, block_mask=block_mask)
        else:
            return flex_attention(q, k, v, block_mask=block_mask)

    def _sdpa_forward(self, q, k, v, block_mask):
        if self.alibi:
            L, S = q.shape[-2], k.shape[-2]
            pos_bias = self.alibi_bias.view(-1, 1, 1) * (
                torch.arange(S).type_as(q) - torch.arange(L).type_as(q).view(-1, 1)  
            )
            attn_mask = torch.where(block_mask.unsqueeze(0), pos_bias, float('-inf')) if block_mask is not None else pos_bias
            return scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        else:
            return scaled_dot_product_attention(q, k, v, attn_mask=block_mask, is_causal=self.is_causal)

    def forward(self, query_input, key_input, value_input, block_mask = None,sliding=False,sliding_window_size=None):
        # The sliding window info are required only by flash attn
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
        # We are using TensorDC dataclasses, thus tensors should be extracted using .tensor
        query_tensor=query_input.tensor #(B,S,D)
        key_tensor=key_input.tensor #(B,S,D)
        value_tensor=value_input.tensor #(B,S,D)
        #Packed flash attention is temporarily disabled for debug (checking if projection is done correctly + checking alibi slopes in the packed case)
        if False and self.attn_impl=='flash' and self.qkv_samedim and query_tensor is key_tensor and key_tensor is value_tensor:
            # I can use flash attention with packed projection for efficiency
            qkv=self._project_packed(query_tensor) # (B, S, 3*H)
            qkv=qkv.unflatten(-1, [3, self.nheads, self.head_dim])  # (B, S, 3, H, Hd)
            packed=True
        else:
            packed=False
            query, key, value = self._project(query_tensor, key_tensor, value_tensor) #(B,S,D)
            query, key, value = map(self._reshape_heads, (query, key, value)) #(B,H,S,Hd)

        # Up to this part, shapes are in the form: B,H,S,Hd
        # After the execution of the attention implementation, B,S,H,Hd is expected
        # Batch has to be considered to be omitted when we use unpadding + flash attention
        if self.attn_impl=='flash':
            if packed:
                attn_output = self._flash_forward(qkv=qkv, query_input=query_input, key_input=key_input, packed=True, sliding=sliding, sliding_window_size=sliding_window_size) # (B, S, H, Hd) [B not present in case of unpadding]              
            else:
                attn_output = self._flash_forward(q=query, k=key, v=value, query_input=query_input, key_input=key_input, packed=False, sliding=sliding, sliding_window_size=sliding_window_size) # (B, S, H, Hd) [B not present in case of unpadding]

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

            attn_output = attn_output.transpose(1, 2)  # Transpose to match: (B, S, H, Hd)
        

        attn_output = attn_output.flatten(-2) #The flatten expects: (B, S, H, Hd)
        return replace(query_input, tensor=self.out_proj(attn_output))        

class PackedSwiGLUFFN(nn.Module):
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        hidden_size: Optional[int] = None,
        ffn_factor: Optional[float] = None
    ):
        super().__init__()

        if config is not None:
            hidden_size = config.hidden_size
            ffn_factor = config.ffn_factor

        # Fail hard if still None
        hidden_size = int(hidden_size * ffn_factor)

        self.w13 = nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        self.w2 = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, _input: TensorDC) -> TensorDC:
        x1, x3 = torch.chunk(self.w13(_input.tensor), 2, -1)
        out = self.w2(F.silu(x1) * x3)
        return replace(_input, tensor=out)


        

"""
class RMSNorm(nn.Module):
    #From https://github.com/Emericen/tiny-qwen
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size))
        self.variance_epsilon = config.rms_norm_eps
    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)
"""
