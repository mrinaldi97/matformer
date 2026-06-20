"""
File: matformer/transformer_blocks/attention.py
"""
#Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
#Matformer
from matformer.matformer_module import MatformerModule
from matformer.tensors_dataclasses import TensorDC, NormalTensor, PaddedTensor, UnpaddedTensor, ModuleWrapper 
from matformer.utils.matformer_cache import ensure_cache_and_registry, CachedStuff
from matformer.matformer_registry import registry
#Other
from dataclasses import replace
from warnings import warn
from typing import Optional
import einops

class MultiHeadAttention(MatformerModule):
    # Stable parameter names
    packed_proj: "param_name:qkv_proj"      # Packed Q+K+V projection (when applicable)
    q_proj: "param_name:q_proj"             # Separate query projection (cross-attention)
    k_proj: "param_name:k_proj"             # Separate key projection (cross-attention)
    v_proj: "param_name:v_proj"             # Separate value projection (cross-attention)
    out_proj: "param_name:o_proj"           # Output projection
    attn_kernel: "param_name:attn_kernel"   # Attention implementation kernel
    available_hooks = ['q', 'k', 'v', 'attn_scores', 'attn_pattern', 'z', 'attn_out'] #Warning: hooks are available only in the Mechanistic Interpretability branch

    def __init__(
        self,
        q_dim: int,               
        k_dim: Optional[int] = None,
        v_dim: Optional[int] = None,
        is_cross_attention: bool = False,
        nheads: int = 8,
        num_key_value_heads: int = None, #If not none, Grouped Query Attention
        bias: bool = False,
        positional_encoding = ['rope'],  # 'alibi', 'rope', 'nope', 'sinusoidal', 'learnable'
        #dropout: float = 0.0, # Not supported by FlexAttention yet
        cache: Optional['CachedStuff'] = None,
        attn_variant: str = 'normal',  # 'normal', 'wersa', 'linear', etc.
        attn_impl: str = 'flash',  # Preferred implementation within variant
        is_causal: bool = True,
        sliding_window: Optional[int] = None,
        device: str = 'cuda',
        interpretability_friendly=False,
        hooks={},
        is_hybrid=False, #Initialize two kernels, one for is_causal = True, another one for is_causal = False; supports giving is_causal as a forward argument
        layer_idx=0
    ):
        super().__init__()
        if isinstance(positional_encoding, str):
            positional_encoding = [positional_encoding]        
        # Assertions

        assert q_dim % nheads == 0, "q_dim is not divisible by nheads"
        if num_key_value_heads is None:
            num_key_value_heads = nheads  # default: MHA
        assert nheads % num_key_value_heads == 0, "nheads must be divisible by num_key_value_heads"        
        if is_cross_attention:
            assert k_dim is not None, "You asked for a cross attention, but you haven't provided keys dim"
            assert v_dim is not None, "You asked for a cross attention, but you haven't provided values dim"
        else:
            k_dim=q_dim
            v_dim=q_dim 

        # Initialization
        if cache is None:     
            warn("Cache not provided to MultiHeadAttention, initializing new cache. Consider passing cache for efficiency.")
            cache = CachedStuff()
        self.cache = cache
        
        # Initialize registry if not present
        if not hasattr(self.cache, 'registry'):
            warn("Registry not found in cache, initializing. Consider initializing registry in cache.")
            self.cache.registry = registry
        self.force_separed_qkv=False
        self.nheads = nheads
        self.kv_nheads = num_key_value_heads  
        self.n_groups = nheads // self.kv_nheads  #1 for MHA, >1 for GQA
        self.attn_variant = attn_variant
        self.positional_encoding = positional_encoding
        self.is_causal = is_causal
        self.is_cross_attention = is_cross_attention
        self.sliding_window = sliding_window
        self.bias = bias
        self.qkv_samedim = q_dim == k_dim and q_dim == v_dim
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.head_dim = q_dim // nheads
        self.layer_idx=layer_idx if layer_idx is not None else 0
        self.mechanistic_interpretability=interpretability_friendly
        self.is_hybrid = is_hybrid
        if interpretability_friendly:
            try:
                import einops
            except ImportError:
                print("Einops is required to do Mechanistic Interpretability!")
            self._init_hooks(hooks)
        # RoPE initialization
        if 'rope' in self.positional_encoding:
            #self.rotary_embedding = self.cache.get_rotary_emb(self.head_dim)
            self.rotary_embedding = self.cache.registry.create('positional_encoding','rope')
            self.rotary_embedding_meta=self.rotary_embedding._matformer_metadata     
        if not interpretability_friendly:
            # Initialize fast attention kernel
            if is_hybrid: #Initialize two kernels
                self.attn_kernel_causal= self.cache.registry.create(
                    'attention', 
                    attn_variant,
                    preferred=attn_impl,
                    nheads=nheads,
                    head_dim=self.head_dim,
                    num_key_value_heads=num_key_value_heads,
                    is_causal=True,
                    sliding_window=sliding_window,
                    positional_encoding=positional_encoding,
                    cache=cache,
                    device=device
                )
                self.attn_kernel_noncausal= self.cache.registry.create(
                    'attention', 
                    attn_variant,
                    preferred=attn_impl,
                    nheads=nheads,
                    head_dim=self.head_dim,
                    num_key_value_heads=num_key_value_heads,
                    is_causal=False,
                    sliding_window=sliding_window,
                    positional_encoding=positional_encoding,
                    cache=cache,
                    device=device
                )   
                self.causal_kernel_meta=self.attn_kernel_causal._matformer_metadata
                self.noncausal_kernel_meta=self.attn_kernel_noncausal._matformer_metadata      
            else: #Normal branch
                self.attn_kernel = self.cache.registry.create(
                    'attention', 
                    attn_variant,
                    preferred=attn_impl,
                    nheads=nheads,
                    head_dim=self.head_dim,
                    num_key_value_heads=num_key_value_heads,
                    is_causal=is_causal,
                    sliding_window=sliding_window,
                    positional_encoding=positional_encoding,
                    cache=cache,
                    device=device
                )   
                # Get kernel metadata
                self.kernel_meta = self.attn_kernel._matformer_metadata
            
        if not self.force_separed_qkv and (not is_cross_attention or self.qkv_samedim) and self.n_groups == 1: # Packed qkv projection for efficiency  
            self.packed_proj=self.cache.registry.create('linear','linear',in_features=self.q_dim, out_features=3*q_dim,bias=bias)
        else:
            self.q_proj=self.cache.registry.create('linear','linear',in_features=q_dim, out_features=q_dim,bias=bias)
            self.k_proj=self.cache.registry.create('linear','linear',in_features=k_dim, out_features=self.head_dim * self.kv_nheads,bias=bias)
            self.v_proj=self.cache.registry.create('linear','linear',in_features=v_dim, out_features=self.head_dim * self.kv_nheads,bias=bias)

        self.out_proj=self.cache.registry.create('linear','linear',in_features=q_dim,out_features=self.q_dim,bias=bias) # Out projection

    @staticmethod
    def _pack_qkv(q, k, v):
        normalize = lambda s: s.translate(str.maketrans('', '', '?B'))
        assert normalize(q.tensor_order) == normalize(k.tensor_order) == normalize(v.tensor_order), "QKV must have same tensor order"
        assert q.tensor.shape == k.tensor.shape == v.tensor.shape, "_pack_qkv requires Q, K, V to have identical shapes (MHA only). Use separate projections for GQA."
        current_order_norm = normalize(q.tensor_order)
        
        if current_order_norm == "HSD":
            # [?, H, S, D] => [?, S, H, D]
            q_t = q.tensor.transpose(1, 2)
            k_t = k.tensor.transpose(1, 2)
            v_t = v.tensor.transpose(1, 2)
            #  [?, S, 3, H, D]
            packed_tensor = torch.stack([q_t, k_t, v_t], dim=-3)
            new_order = '?S3HD'
        elif current_order_norm == "SHD":
            # [?, S, 3, H, D]
            packed_tensor = torch.stack([q.tensor, k.tensor, v.tensor], dim=-3)
            new_order = '?S3HD'
        else:
            raise ValueError(f"Unsupported tensor order: {q.tensor_order}")
        
        return replace(q, tensor=packed_tensor, tensor_order=new_order)
    
    @staticmethod
    def _unpack_qkv(qkv_packed):
            # S3HD => ?SHD
            q_t, k_t, v_t = qkv_packed.tensor.unbind(dim=-3)
            order = qkv_packed.tensor_order.replace('3', '')
            return (replace(qkv_packed, tensor=q_t, tensor_order=order),
                    replace(qkv_packed, tensor=k_t, tensor_order=order),
                    replace(qkv_packed, tensor=v_t, tensor_order=order))
        
    @staticmethod
    def _transpose_for_kernel(tensor, wanted_from_kernel):
        """ Currently this helper function only supports conversion from and to flash style, sdpa style 
            BHSD
            BSHD
            that is, when transposing dimensions 1 and 2 
            Every other conversion wil raise an error
            Letters 'B' and '?' are trated as if they are the same, it represents equivalence of batching/unbatching in Matformer's jargon.
            The tensorDC must have a 'tensor_order' attribute, if not the conversion will fail too
        """
        if not isinstance(tensor, TensorDC) or not hasattr(tensor, 'tensor_order'):
            raise Exception       
        normalize = lambda s: s.translate(str.maketrans('', '', '?B'))
        wanted = normalize(wanted_from_kernel)
        current = normalize(tensor.tensor_order)        
        if current == wanted:
            return tensor  # No change required, great!  
        if (current=='SHD' and wanted=='HSD') or (current=='HSD' and wanted == 'SHD'):  
            # It can be transposed
            return replace(tensor, tensor=tensor.tensor.transpose(1,2), tensor_order='?' + wanted)   
        print(f"Current: {current} Wanted: {wanted}")
        raise Exception

    """
    The properties defined below serves for Mechanistic Interpretability purposes: they provide
    diverse views over the weight that facilitates MI. Can be ignored for a training model or in general
    if MI is not at stake. The MI part can be recognized by the suffix "_MI"
    """
    @property
    def W_Q_MI(self):
        if hasattr(self, 'packed_proj'):
            return self.packed_proj.inner.weight.view(3, self.nheads, self.head_dim, -1)[0].transpose(-1, -2)
        return self.q_proj.inner.weight.view(self.nheads, self.head_dim, -1).transpose(-1, -2)

    @property
    def W_K_MI(self):
        if hasattr(self, 'packed_proj'):
            return self.packed_proj.inner.weight.view(3, self.nheads, self.head_dim, -1)[1].transpose(-1, -2)
        return self.k_proj.inner.weight.view(self.kv_nheads, self.head_dim, -1).transpose(-1, -2)

    @property
    def W_V_MI(self):
        if hasattr(self, 'packed_proj'):
            return self.packed_proj.inner.weight.view(3, self.nheads, self.head_dim, -1)[2].transpose(-1, -2)
        return self.v_proj.inner.weight.view(self.kv_nheads, self.head_dim, -1).transpose(-1, -2)

    @property
    def W_O_MI(self):
        return self.out_proj.inner.weight.view(-1, self.nheads, self.head_dim).permute(1, 2, 0)
    @property
    def B_Q_MI(self):
        if not self.bias: return 0
        if hasattr(self, 'packed_proj'):
            return self.packed_proj.inner.bias.view(3, self.nheads, self.head_dim)[0]
        return self.q_proj.inner.bias.view(self.nheads, self.head_dim)

    @property
    def B_K_MI(self):
        if not self.bias: return 0
        if hasattr(self, 'packed_proj'):
            return self.packed_proj.inner.bias.view(3, self.nheads, self.head_dim)[1]
        return self.k_proj.inner.bias.view(self.kv_nheads, self.head_dim)

    @property
    def B_V_MI(self):
        if not self.bias: return 0
        if hasattr(self, 'packed_proj'):
            return self.packed_proj.inner.bias.view(3, self.nheads, self.head_dim)[2]
        return self.v_proj.inner.bias.view(self.kv_nheads, self.head_dim)

    @property
    def B_O_MI(self):
        if not self.bias: return 0
        return self.out_proj.bias  # shape: [d_model] 
    
    def forward(self, query_input, original_x=None, key_input=None, value_input=None, document_mask=None, is_causal=None, interpretability=False):
        if is_causal is None:
            is_causal=self.is_causal
        # 1. Extract max seq len and cu_seqlens (if appliable), convert to Normal Tensor in case a pytorch tensor is passed.
        cu_seqlens=None
        if isinstance(query_input, UnpaddedTensor):
            max_seq_len = query_input.max_seq_len # This variable can be used, for example, to get the correct RoPe cos and sin from cache, but in general is useful
            cu_seqlens=query_input.cu_seqlens.to(query_input.device) # Used for unpadding
        elif isinstance(query_input, PaddedTensor):
            max_seq_len = query_input.shape[1]
        else: #normal tensor
            shape=query_input.shape
            if len(shape)==2:
                max_seq_len = query_input.shape[0]
            else:
                max_seq_len=query_input.shape[1]
        
        if key_input is None:
            key_input = query_input
        if value_input is None:
            value_input = query_input  
             
        if self.mechanistic_interpretability or interpretability:
            return self.interpretability_friendly_forward(query_input,original_x,key_input,value_input,document_mask,cu_seqlens,max_seq_len, is_causal)            
        else:
            return self.normal_forward(query_input,original_x,key_input,value_input,document_mask,cu_seqlens,max_seq_len, is_causal)
    def interpretability_friendly_forward(self, query_input, original_x=None, key_input=None, value_input=None, document_mask=None, cu_seqlens=None, max_seq_len=None, is_causal=False):
        """
        A modified forward that is optimal for working on mechanistic interpretability.
        """
        # 1. For now, the MI version works only with PaddedTensors
        assert isinstance(query_input, TensorDC)
        wasUnpadded = True if isinstance(query_input, UnpaddedTensor) else False
        q = query_input.pad().tensor
        k = key_input.pad().tensor
        v = value_input.pad().tensor

        # 2. Query, key and value vectors using the MI-friendly  weights
        q = einops.einsum(q, self.W_Q_MI, "batch position d_model, n_heads d_model d_head -> batch position n_heads d_head") + self.B_Q_MI
        k = einops.einsum(k, self.W_K_MI, "batch position d_model, kv_heads d_model d_head -> batch position kv_heads d_head") + self.B_K_MI
        v = einops.einsum(v, self.W_V_MI, "batch position d_model, kv_heads d_model d_head -> batch position kv_heads d_head") + self.B_V_MI
        q, k, v = self._apply_hook('q', q), self._apply_hook('k', k), self._apply_hook('v', v)

        if 'rope' in self.positional_encoding:
            cos, sin = self.cache.get_rotary_cos_sin(q.shape[1], self.head_dim, q.device, q.dtype)
            cos, sin = cos[None, :, None, :], sin[None, :, None, :]
            rotate = lambda x: torch.cat([-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]], dim=-1)
            q, k = q * cos + rotate(q) * sin, k * cos + rotate(k) * sin
        if self.n_groups > 1: # Expanding GQA to MHA
            k = k.repeat_interleave(self.n_groups, dim=2)
            v = v.repeat_interleave(self.n_groups, dim=2)
            # 3. Computation of attention scores    
        attn_scores = einops.einsum(q, k, "batch posQ n_heads d_head, batch posK n_heads d_head -> batch n_heads posQ posK")
        attn_scores = (attn_scores / self.head_dim**0.5) # Standard sqrt(d) scaling
        if 'alibi' in self.positional_encoding:
            attn_scores = attn_scores + self.cache.get_alibi_bias(q.shape[1], k.shape[1], self.nheads, q.device, q.dtype)
        if is_causal:
            mask = torch.ones(*attn_scores.shape[-2:], device=attn_scores.device, dtype=torch.bool).tril()
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        attn_scores = self._apply_hook('attn_scores', attn_scores)

        # 4. Softmax
        attn_pattern = attn_scores.softmax(-1)
        attn_pattern = self._apply_hook('attn_pattern', attn_pattern)

        # 5. Weighed sums of values according to attention pattern
        z = einops.einsum(v, attn_pattern, "batch posK n_heads d_head, batch n_heads posQ posK -> batch posQ n_heads d_head")
        z = self._apply_hook('z', z)

        # 6. Output calculation
        attn_out = einops.einsum(z, self.W_O_MI, "batch posQ n_heads d_head, n_heads d_head d_model -> batch posQ d_model") + self.B_O_MI
        attn_out = self._apply_hook('attn_out', attn_out)

        # 7. Reinserting into tensorDC
        attn_out = replace(query_input, tensor=attn_out)

        # (optional) repadding
        if wasUnpadded:
            attn_out = attn_out.unpad()
        return attn_out
    def normal_forward(self, query_input, original_x=None, key_input=None, value_input=None, document_mask=None, cu_seqlens=None,max_seq_len=None, is_causal=False):
        """
        Input Tensors:
        query_input: (Batch, Seqlen, Qdim)
        key_input: (Batch, Seqlen, Kdim) [If omitted, self-attention]
        value_input: (Batch, Seqlen, Vdim) [If omitted, self-attention]
        Output:
        (Batch, Seqlen, Qdim) # Qdim is the query dimension as well as the output.
        """
        if not self.is_hybrid:
            kernel_meta=self.kernel_meta
            attn_kernel=self.attn_kernel
        else:
            if is_causal:
                kernel_meta=self.causal_kernel_meta
                attn_kernel=self.attn_kernel_causal
            else:
                kernel_meta=self.noncausal_kernel_meta
                attn_kernel=self.attn_kernel_noncausal
        
        supports_unpadding = kernel_meta.get('supports_unpadding', False)
        if 'rope' in self.positional_encoding:
            supports_unpadding = supports_unpadding and self.rotary_embedding_meta.get('supports_unpadding', False) # If either the RoPe or the attn kernel do not support unpadding, sequence is repadded
        supports_packed_qkv = kernel_meta.get('supports_packed_qkv', False)
        rope_supports_packed = self.rotary_embedding_meta.get('supports_packed_qkv', False) if  'rope' in self.positional_encoding else True  
        
        # Set defaults for self-attention
        
        if key_input is None:
            key_input = query_input
        if value_input is None:
            value_input = query_input
                
        # If the attention implementation does not support unpadding, eventually unpadded sequences must be padded
        repadded = False
        if not supports_unpadding:
            if isinstance(query_input, UnpaddedTensor):
                repadded = True  # A flag: if original inputs were padded from unpadded tensors, they will be unpadded again at the end
                #warn('The selected attention implementation does not support unpadding. Sequence is automatically repadded. This can lead to a loss in performances')
                query_input = query_input.pad()
                key_input = key_input.pad()
                value_input = value_input.pad()


        
        # Projecting (eventually, packed projection)
        if self.qkv_samedim and not self.is_cross_attention and not self.force_separed_qkv and self.n_groups == 1:
            qkv_projected = NormalTensor(tensor=self.packed_proj(query_input.tensor), tensor_order='?S(3*D)')
            q, k, v = None, None, None
        elif self.qkv_samedim and self.is_cross_attention and not self.force_separed_qkv and self.n_groups == 1:
            qkv_projected = None
            w = torch.chunk(self.packed_proj.inner.weight, 3, dim=0)
            b = torch.chunk(self.packed_proj.inner.bias, 3, dim=0) if self.bias else (None, None, None)
            q = NormalTensor(tensor=F.linear(query_input.tensor, w[0], b[0]),tensor_order='?SD')
            k = NormalTensor(tensor=F.linear(key_input.tensor, w[1], b[1]),tensor_order='?SD')
            v = NormalTensor(tensor=F.linear(value_input.tensor, w[2], b[2]),tensor_order='?SD')
        else:
            qkv_projected = None
            q = NormalTensor(tensor=self.q_proj(query_input.tensor),tensor_order='?SD')
            k = NormalTensor(tensor=self.k_proj(key_input.tensor),tensor_order='?SD')
            v = NormalTensor(tensor=self.v_proj(value_input.tensor),tensor_order='?SD')
            
        # 3. Heads creation  
        if qkv_projected is not None:
            qkv_projected = replace(qkv_projected, tensor=qkv_projected.tensor.unflatten(-1, [3, self.nheads, self.head_dim]), tensor_order='?S3HD')
        else:
            q = replace(q, tensor=q.tensor.unflatten(-1, [self.nheads, self.head_dim]),tensor_order='?SHD')
            k = replace(k, tensor=k.tensor.unflatten(-1, [self.kv_nheads, self.head_dim]),tensor_order='?SHD')
            v = replace(v, tensor=v.tensor.unflatten(-1, [self.kv_nheads, self.head_dim]),tensor_order='?SHD')

        # 3b. (facultative) Apply RoPe
        if 'rope' in self.positional_encoding:
            repack_after_rope = False  
            if qkv_projected is not None and not rope_supports_packed: # Unpack if RoPe doesn't support packed qkv  
                q, k, v = self._unpack_qkv(qkv_projected)
                qkv_projected = None
                repack_after_rope = supports_packed_qkv  #Repack only if supported by the attention kernel
            elif qkv_projected is not None and rope_supports_packed:
                assert qkv_projected.tensor_order == self.rotary_embedding_meta['tensor_order_qkv_packed_input']
            
            if qkv_projected is None:  #non-packed branch, adapt to requested tensor order 
                q = self._transpose_for_kernel(q, self.rotary_embedding_meta['tensor_order_input'])
                k = self._transpose_for_kernel(k, self.rotary_embedding_meta['tensor_order_input'])

            # 1. Get sin and cos from cache
            cos, sin = self.cache.get_rotary_cos_sin(max_seq_len, self.head_dim, device=query_input.device, dtype=query_input.dtype)  

            # 2. Rotate query and keys
            qkv_t, q_t, k_t = self.rotary_embedding(
                qkv=qkv_projected.tensor if qkv_projected is not None else None,
                q=q.tensor if q is not None else None,
                k=k.tensor if k is not None else None,
                cos=cos, sin=sin,
                cu_seqlens=cu_seqlens, max_seq_len=max_seq_len
            )
            if qkv_projected is not None:
                qkv_projected = replace(qkv_projected, tensor=qkv_t, tensor_order=self.rotary_embedding_meta['tensor_order_qkv_packed_output'])
            else:
                q = replace(q, tensor=q_t, tensor_order=self.rotary_embedding_meta['tensor_order_output'])
                k = replace(k, tensor=k_t, tensor_order=self.rotary_embedding_meta['tensor_order_output'])


            # Repack if needed for attention kernel 
            if repack_after_rope:
                qkv_projected = self._pack_qkv(q, k, v)
                q, k, v = None, None, None
        kernel_input_order = kernel_meta.get('tensor_order_input', '?SHD')
        tensor_order_qkv_packed = kernel_meta.get('tensor_order_qkv_packed_input', None)
        
        if qkv_projected is not None and not supports_packed_qkv:
            q, k, v = self._unpack_qkv(qkv_projected)
            qkv_projected = None

        if qkv_projected is not None:
            qkv_projected = self._transpose_for_kernel(qkv_projected, tensor_order_qkv_packed if tensor_order_qkv_packed is not None else kernel_input_order)
        else:
            q = self._transpose_for_kernel(q, kernel_input_order)
            k = self._transpose_for_kernel(k, kernel_input_order)
            v = self._transpose_for_kernel(v, kernel_input_order)

        if self.n_groups > 1 and not kernel_meta.get('supports_gqa', False):
            # Kernel doesn't support GQA natively: expand K/V to full nheads
            dim = 1 if k.tensor_order.replace('?', 'B') == 'BHSD' else 2
            k = replace(k, tensor=k.tensor.repeat_interleave(self.n_groups, dim=dim))
            v = replace(v, tensor=v.tensor.repeat_interleave(self.n_groups, dim=dim))
        # Attention computation
        attn_output = attn_kernel(
            qkv=qkv_projected.tensor if qkv_projected is not None else None,
            q=q.tensor if q is not None else None,
            k=k.tensor if k is not None else None,
            v=v.tensor if v is not None else None,
            query_input=query_input, key_input=key_input,
            original_x=original_x
        )

        # Transpose from kernel's output format to expected format (B, S, H, Hd)
        kernel_output_order = kernel_meta.get('tensor_order_output', '?SHD')
        attn_output = NormalTensor(tensor=attn_output, tensor_order=kernel_output_order)
        attn_output = self._transpose_for_kernel(attn_output, '?SHD')

        # Post-attention stuff
        attn_output = attn_output.tensor.flatten(-2)  # (B, S, H, Hd) -> (B, S, D)
        output_tensor = self.out_proj(attn_output)

        # Handle output based on input type
        if hasattr(query_input, 'tensor'):
            query_input = replace(query_input, tensor=output_tensor)
        else:
            query_input = output_tensor

        if repadded:
            query_input = query_input.unpad()

        return query_input 
