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
                        
        # Packed qkv projection for efficiency  
        if not is_cross_attention or self.qkv_samedim:
            self.packed_proj = nn.Linear(self.q_dim, 3 * q_dim, bias=bias)
        else:
            self.q_proj = nn.Linear(q_dim, q_dim, bias=bias)
            self.k_proj = nn.Linear(k_dim, q_dim, bias=bias)
            self.v_proj = nn.Linear(v_dim, q_dim, bias=bias)
            
        self.out_proj = nn.Linear(q_dim, self.q_dim, bias=bias)  # Out projection
    def _transpose_for_kernel(self, tensor_dc, target_order):
        """
        Helper to transpose tensors based on metadata.
        Supports: BHSD <-> BSHD conversions
        Tracks current order in tensor_dc.extra_attributes['tensor_order']
        """
        current_order = tensor_dc.extra_attributes.get('tensor_order', 'HSD')
        if current_order == target_order:
            return tensor_dc

        if (current_order == 'HSD' and target_order == 'SHD') or (current_order == 'SHD' and target_order == 'HSD'):
            new_extra = tensor_dc.extra_attributes.copy()
            new_extra['tensor_order'] = target_order
            return NormalTensor(tensor=tensor_dc.tensor.transpose(1, 2), extra_attributes=new_extra)

        # Packed layouts: BS3HD and S3HD are only allowed if already matching the kernel's target layout.
        # This helper does not perform conversion for packed layouts; callers must convert or reject.
        if current_order in ('S3HD', 'S3HD'):
            raise NotImplementedError(
                f"Packed QKV layout conversion not supported in _transpose_for_kernel: {current_order} -> {target_order}"
            )

        raise NotImplementedError(f"Unsupported tensor order conversion: {current_order} -> {target_order}")

    def _unpack_qkv(self, qkv_projected):
        """Unpack qkv into separate q, k, v NormalTensors, preserving extra_attributes."""
        ea = qkv_projected.extra_attributes
        order = ea.get('tensor_order', None)
        t = qkv_projected.tensor

        # Packed padded: (B, S, 3, H, Hd)
        if order == 'S3HD':
            q = t[:, :, 0]  # (B, S, H, Hd)
            k = t[:, :, 1]
            v = t[:, :, 2]
            new_ea = ea.copy()
            new_ea['tensor_order'] = 'HSD'
            return (
                NormalTensor(tensor=q, extra_attributes=new_ea.copy()),
                NormalTensor(tensor=k, extra_attributes=new_ea.copy()),
                NormalTensor(tensor=v, extra_attributes=new_ea.copy()),
            )

        # Packed unpadded: (S, 3, H, Hd)
        if order == 'S3HD':
            q = t[:, 0]  # (S, H, Hd)
            k = t[:, 1]
            v = t[:, 2]
            new_ea = ea.copy()
            new_ea['tensor_order'] = 'HSD'
            return (
                NormalTensor(tensor=q, extra_attributes=new_ea.copy()),
                NormalTensor(tensor=k, extra_attributes=new_ea.copy()),
                NormalTensor(tensor=v, extra_attributes=new_ea.copy()),
            )

        # Fallback: classic BSD flat packing
        if order in (None, 'SD'):
            q_t, k_t, v_t = torch.chunk(t, 3, dim=-1)
            return (
                NormalTensor(tensor=q_t, extra_attributes=ea.copy()),
                NormalTensor(tensor=k_t, extra_attributes=ea.copy()),
                NormalTensor(tensor=v_t, extra_attributes=ea.copy()),
            )

        raise NotImplementedError(f"Unpacking not supported for layout {order}")

    def _pack_qkv(self, q, k, v):
        """Pack separate q, k, v into single qkv NormalTensor."""
        return NormalTensor(tensor=torch.cat([q.tensor, k.tensor, v.tensor], dim=-1), extra_attributes=q.extra_attributes.copy())

    def forward(self, query_input, original_x=None, key_input=None, value_input=None, document_mask=None):
        """
        Input Tensors:
        query_input: (Batch, Seqlen, Qdim)
        key_input: (Batch, Seqlen, Kdim) [If omitted, self-attention]
        value_input: (Batch, Seqlen, Vdim) [If omitted, self-attention]
        Output:
        (Batch, Seqlen, Qdim) # Qdim is the query dimension as well as the output.
        """
        if isinstance(query_input, UnpaddedTensor):
            max_seq_len = query_input.max_seq_len # This variable can be used, for example, to get the correct RoPe cos and sin from cache, but in general is useful
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
        
        supports_unpadding = self.kernel_meta.get('supports_unpadding', False)
        if self.positional_encoding == 'rope':
            supports_unpadding = supports_unpadding and self.rotary_embedding_meta.get('supports_unpadding', False)
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
                query_input = query_input.pad()
                key_input = key_input.pad()
                value_input = value_input.pad()

        # Extract tensors (handle both regular tensors and Matformer's TensorDC)
        q_tensor = query_input.tensor if hasattr(query_input, 'tensor') else query_input
        k_tensor = key_input.tensor if hasattr(key_input, 'tensor') else key_input
        v_tensor = value_input.tensor if hasattr(value_input, 'tensor') else value_input
        
        # Projecting (eventually, packed projection)
        if self.qkv_samedim and not self.is_cross_attention and not self.force_separed_qkv:
            qkv_projected = NormalTensor(tensor=self.packed_proj(q_tensor), extra_attributes={'tensor_order': 'SD'})
            q, k, v = None, None, None
            is_qkv_packed=True
        elif self.qkv_samedim and self.is_cross_attention and not self.force_separed_qkv:
            qkv_projected = None
            w = torch.chunk(self.packed_proj.weight, 3, dim=0)
            b = torch.chunk(self.packed_proj.bias, 3, dim=0) if self.bias else (None, None, None)
            q = NormalTensor(tensor=F.linear(q_tensor, w[0], b[0]), extra_attributes={'tensor_order': 'SD'})
            k = NormalTensor(tensor=F.linear(k_tensor, w[1], b[1]), extra_attributes={'tensor_order': 'SD'})
            v = NormalTensor(tensor=F.linear(v_tensor, w[2], b[2]), extra_attributes={'tensor_order': 'SD'})
            is_qkv_packed=False
        else:
            qkv_projected = None
            q = NormalTensor(tensor=self.q_proj(q_tensor), extra_attributes={'tensor_order': 'SD'})
            k = NormalTensor(tensor=self.k_proj(k_tensor), extra_attributes={'tensor_order': 'SD'})
            v = NormalTensor(tensor=self.v_proj(v_tensor), extra_attributes={'tensor_order': 'SD'})
            is_qkv_packed=False
        # Creating the heads (B, S, D) -> (B, H, S, Hd) or (B, S, 3*D) -> (B, S, 3, H, Hd) for packed 
        if qkv_projected is not None:
            # Packed: (B, S, 3*D) -> (B, S, 3, H, Hd)
            t = qkv_projected.tensor.unflatten(-1, [3, self.nheads, self.head_dim])
            # Detect whether batch dim exists (padded / normal) or not (unpadded represented as (S, 3, H, Hd))
            if t.dim() == 5:
                order = 'S3HD'
            elif t.dim() == 4:
                order = 'S3HD'
            else:
                raise NotImplementedError(f"Unsupported packed qkv tensor shape: {tuple(t.shape)}")
            qkv_projected = NormalTensor(tensor=t, extra_attributes={'tensor_order': order})
        else:
            q = NormalTensor(tensor=q.tensor.unflatten(-1, [self.nheads, self.head_dim]).transpose(1, 2), extra_attributes={'tensor_order': 'HSD'})
            k = NormalTensor(tensor=k.tensor.unflatten(-1, [self.nheads, self.head_dim]).transpose(1, 2), extra_attributes={'tensor_order': 'HSD'})
            v = NormalTensor(tensor=v.tensor.unflatten(-1, [self.nheads, self.head_dim]).transpose(1, 2), extra_attributes={'tensor_order': 'HSD'})

        # Apply RoPe
        if self.positional_encoding == 'rope':
            rope_input_order = self.rotary_embedding_meta.get('tensor_order_input', 'HSD')

            # Unpack if RoPe doesn't support packed qkv  
            repack_after_rope = False  
            if qkv_projected is not None and not rope_supports_packed:
                # (3, B, H, S, Hd) -> separate q, k, v 
                q = NormalTensor(tensor=qkv_projected.tensor[0], extra_attributes={'tensor_order': 'SHD'})  
                k = NormalTensor(tensor=qkv_projected.tensor[1], extra_attributes={'tensor_order': 'SHD'})  
                v = NormalTensor(tensor=qkv_projected.tensor[2], extra_attributes={'tensor_order': 'SHD'})
                qkv_projected = None  
                repack_after_rope = supports_packed_qkv  #Repack only if supported by the attention kernel
            elif qkv_projected is not None and rope_supports_packed:
                rope_qkv_packed_order = self.rotary_embedding_meta.get('tensor_order_qkv_packed_input', None)
                current_order = qkv_projected.extra_attributes.get('tensor_order', None)
                if rope_qkv_packed_order is not None and current_order != rope_qkv_packed_order:
                    raise NotImplementedError(
                        f"RoPE kernel expects packed layout {rope_qkv_packed_order} but current is {current_order}"
                    )

            # Transpose to RoPe's expected format 
            if qkv_projected is not None:
                qkv_projected = self._transpose_for_kernel(qkv_projected, rope_input_order)  
            else: 
                q = self._transpose_for_kernel(q, rope_input_order)
                k = self._transpose_for_kernel(k, rope_input_order)

            # 1. Get sin and cos from cache
            cos, sin = self.cache.get_rotary_cos_sin(max_seq_len, self.head_dim, device=query_input.device, dtype=self.dtype)  

            cu_seqlens = query_input.cu_seqlens if isinstance(query_input, UnpaddedTensor) else None
            max_seq_len_rope = query_input.max_seq_len if isinstance(query_input, UnpaddedTensor) else None

            # 2. Rotate query and keys
            qkv_t, q_t, k_t = self.rotary_embedding(
                qkv=qkv_projected.tensor if qkv_projected is not None else None,
                q=q.tensor if q is not None else None,
                k=k.tensor if k is not None else None,
                cos=cos, sin=sin,
                cu_seqlens=cu_seqlens, max_seq_len=max_seq_len_rope
            )
            if qkv_projected is not None:
                qkv_projected = NormalTensor(tensor=qkv_t, extra_attributes=qkv_projected.extra_attributes)
            else:
                q = NormalTensor(tensor=q_t, extra_attributes=q.extra_attributes)
                k = NormalTensor(tensor=k_t, extra_attributes=k.extra_attributes)

            # Repack if needed for attention kernel 
            if repack_after_rope:
                qkv_projected = NormalTensor(  
                    tensor=torch.stack([q.tensor, k.tensor, v.tensor], dim=2) if q.tensor.dim() == 4 else torch.stack([q.tensor, k.tensor, v.tensor], dim=1), 
                    extra_attributes=q.extra_attributes.copy()  
                )  
                # When repacked above, ensure extra_attributes tensor_order is set correctly
                # prefer BS3HD for batched, S3HD for unbatched
                if qkv_projected.tensor.dim() == 5:
                    qkv_projected.extra_attributes['tensor_order'] = 'S3HD'
                elif qkv_projected.tensor.dim() == 4:
                    qkv_projected.extra_attributes['tensor_order'] = 'S3HD'
                q, k, v = None, None, None

        kernel_input_order = self.kernel_meta.get('tensor_order_input', 'HSD')

        tensor_order_qkv_packed = self.kernel_meta.get('tensor_order_qkv_packed_input', None)
        if qkv_projected is not None:
            if supports_packed_qkv:
                current_order = qkv_projected.extra_attributes.get('tensor_order', None)
                if tensor_order_qkv_packed is not None and current_order != tensor_order_qkv_packed:
                    raise NotImplementedError(
                        f"Kernel expects packed layout {tensor_order_qkv_packed} but current is {current_order}"
                    )
            else:
                q,k,v=self._unpack_qkv(qkv_projected)
                qkv_projected=None

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
        kernel_output_order = self.kernel_meta.get('tensor_order_output', 'HSD')
        attn_output = NormalTensor(tensor=attn_output, extra_attributes={'tensor_order': kernel_output_order})
        attn_output = self._transpose_for_kernel(attn_output, 'SHD')

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
