class MultiHeadAttention_WorkInProgress(MatformerModule):
    # Stable parameter names
    packed_proj: "param_name:qkv_proj"      # Packed Q+K+V projection (when applicable)
    q_proj: "param_name:q_proj"             # Separate query projection (cross-attention)
    k_proj: "param_name:k_proj"             # Separate key projection (cross-attention)
    v_proj: "param_name:v_proj"             # Separate value projection (cross-attention)
    out_proj: "param_name:o_proj"           # Output projection
    attn_kernel: "param_name:attn_kernel"   # Attention implementation kernel
    
    def __init__(
        self,
        q_dim: int,               
        k_dim: Optional[int] = None,
        v_dim: Optional[int] = None,
        is_cross_attention: bool = False,
        nheads: int = 8,
        bias: bool = False,
        positional_encoding: str = 'rope',  # 'alibi', 'rope', 'nope', 'sinusoidal', 'learnable'
        #dropout: float = 0.0, # Not supported by FlexAttention yet
        cache: Optional['CachedStuff'] = None,
        attn_variant: str = 'normal',  # 'normal', 'wersa', 'linear', etc.
        attn_impl: str = 'flash',  # Preferred implementation within variant
        is_causal: bool = True,
        sliding_window: Optional[int] = None,
        device: str = 'cuda'  
    ):
        super().__init__()

        # Assertions
        assert q_dim % nheads == 0, "q_dim is not divisible by nheads"
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
        
        # RoPE initialization
        if self.positional_encoding == 'rope':
            #self.rotary_embedding = self.cache.get_rotary_emb(self.head_dim)
            self.rotary_embedding = self.cache.registry.create('positional_encoding','rope')
            self.rotary_embedding_meta=self.rotary_embedding._matformer_metadata
        # Initialize attention kernel
        self.attn_kernel = self.cache.registry.create(
            'attention', 
            attn_variant,
            preferred=attn_impl,
            nheads=nheads,
            head_dim=self.head_dim,
            is_causal=is_causal,
            sliding_window=sliding_window,
            positional_encoding=positional_encoding,
            cache=cache,
            device=device
        )
        
        # Get kernel metadata
        self.kernel_meta = self.attn_kernel._matformer_metadata
        
        if not is_cross_attention or self.qkv_samedim: # Packed qkv projection for efficiency  
            self.packed_proj=self.cache.registry.create('linear','linear',in_features=self.q_dim, out_features=3*q_dim,bias=bias)
        else:
            self.q_proj=self.cache.registry.create('linear','linear',in_features=q_dim, out_features=q_dim,bias=bias)
            self.k_proj=self.cache.registry.create('linear','linear',in_features=k_dim, out_features=q_dim,bias=bias)
            self.v_proj=self.cache.registry.create('linear','linear',in_features=v_dim, out_features=q_dim,bias=bias)

        self.out_proj=self.cache.registry.create('linear','linear',in_features=q_dim,out_features=self.q_dim,bias=bias) # Out projection

    @staticmethod
    def _pack_qkv(q, k, v):
        assert q.tensor_order == k.tensor_order == v.tensor_order, "QKV must have same tensor order"
        packed_tensor = torch.stack([q.tensor, k.tensor, v.tensor], dim=2)
        order = q.tensor_order
        new_order = order[:2] + '3' + order[2:] if len(order) > 2 else order + '3'
        return replace(q, tensor=packed_tensor, tensor_order=new_order)

    @staticmethod
    def _unpack_qkv(qkv_packed):
        q_t, k_t, v_t = qkv_packed.tensor.unbind(dim=2)
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
        if current == wanted[0] + wanted[2] + wanted[1] + wanted[3:]:
            # It can be transposed
            return tensor.replace(tensor=tensor.tensor.transpose(1, 2), tensor_order='?' + wanted)     
        raise Exception

    def forward(self, query_input, original_x=None, key_input=None, value_input=None, document_mask=None):
        """
        Input Tensors:
        query_input: (Batch, Seqlen, Qdim)
        key_input: (Batch, Seqlen, Kdim) [If omitted, self-attention]
        value_input: (Batch, Seqlen, Vdim) [If omitted, self-attention]
        Output:
        (Batch, Seqlen, Qdim) # Qdim is the query dimension as well as the output.
        """
        
        # 1. Extract max seq len and cu_seqlens (if appliable), convert to Normal Tensor in case a pytorch tensor is passed.
        cu_seqlens=None
        if isinstance(query_input, UnpaddedTensor):
            max_seq_len = query_input.max_seq_len # This variable can be used, for example, to get the correct RoPe cos and sin from cache, but in general is useful
            cu_seqlens=query_input.cu_seqlens # Used for unpadding
        elif isinstance(query_input, PaddedTensor):
            max_seq_len = query_input.shape[1]
        else: #normal tensor
            shape=query_input.shape
            if len(shape)==2:
                max_seq_len = query_input.shape[0]
            else:
                max_seq_len=query_input.shape[1]
                
                
        #self.device=query_input.device
        #self.dtype=query_input.dtype
        
        # 2. 
        
        supports_unpadding = self.kernel_meta.get('supports_unpadding', False)
        if self.positional_encoding == 'rope':
            supports_unpadding = supports_unpadding and self.rotary_embedding_meta.get('supports_unpadding', False) # If either the RoPe or the attn kernel do not support unpadding, sequence is repadded
        supports_packed_qkv = self.kernel_meta.get('supports_packed_qkv', False)
        rope_supports_packed = self.rotary_embedding_meta.get('supports_packed_qkv', False) if self.positional_encoding == 'rope' else True  
        
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
        if self.qkv_samedim and not self.is_cross_attention and not self.force_separed_qkv:
            qkv_projected = NormalTensor(tensor=self.packed_proj(query_input.tensor), tensor_order='?S(3*D)')
            q, k, v = None, None, None
        elif self.qkv_samedim and self.is_cross_attention and not self.force_separed_qkv:
            qkv_projected = None
            w = torch.chunk(self.packed_proj.weight, 3, dim=0)
            b = torch.chunk(self.packed_proj.bias, 3, dim=0) if self.bias else (None, None, None)
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
            k = replace(k, tensor=k.tensor.unflatten(-1, [self.nheads, self.head_dim]),tensor_order='?SHD')
            v = replace(v, tensor=v.tensor.unflatten(-1, [self.nheads, self.head_dim]),tensor_order='?SHD')

        # 3b. (facultative) Apply RoPe
        if self.positional_encoding == 'rope':
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

        kernel_input_order = self.kernel_meta.get('tensor_order_input', '?SHD')
        tensor_order_qkv_packed = self.kernel_meta.get('tensor_order_qkv_packed_input', None)
        
        if qkv_projected is not None and not supports_packed_qkv:
            q, k, v = self._unpack_qkv(qkv_projected)
            qkv_projected = None

        if qkv_projected is not None:
            qkv_projected = self._transpose_for_kernel(qkv_projected, tensor_order_qkv_packed if tensor_order_qkv_packed is not None else kernel_input_order)
        else:
            q = self._transpose_for_kernel(q, kernel_input_order)
            k = self._transpose_for_kernel(k, kernel_input_order)
            v = self._transpose_for_kernel(v, kernel_input_order)

        # Attention computation
        attn_output = self.attn_kernel(
            qkv=qkv_projected.tensor if qkv_projected is not None else None,
            q=q.tensor if q is not None else None,
            k=k.tensor if k is not None else None,
            v=v.tensor if v is not None else None,
            query_input=query_input, key_input=key_input,
            original_x=original_x
        )

        # Transpose from kernel's output format to expected format (B, S, H, Hd)
        kernel_output_order = self.kernel_meta.get('tensor_order_output', '?SHD')
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
