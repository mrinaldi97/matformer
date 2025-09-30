# This code regards BLT implementation, that is still in WIP phase  
"""
        The encoder has two streams:
        h => The stream of bytes embeddings. 
        p => The stream of patch embeddings, that will be the input of the Global Transformer
        Beginning:
        1) Obtaining byte embeddings for each byte
        2) Pooling (avg or max) the byte embeddings of each patch into a "starting patch embedding"
        Layers:
        For each layer, we have:
            a) Normal transformer block with self-attention (masking strategy=sliding window), input bytes embeddings, output bytes embeddings
            b) Cross attention block (masking strategy=each patch-query can attend only to bytes key and values pertinent to the bytes of its patch boundaries)
        Output:
            a) hl => The byte embeddings after the transformer, to be used for the Decoder
            b) pl => The final patches to be inserted into the Global Decoder
            
        Attention masks:
            * The self-attention part of the encoder, attending to byte embeddings, can attend to the preceding bytes even if they trespass
                the path boundary, but within a specific sliding window;
            * In the cross-attention blocks, between patches and bytes, each patch can only attend to the bytes pertaining to its boundaries
                
"""    

class TextEncoder(nn.Module):
   def __init__(self, configs_dict, device='cuda'): 
       super().__init__()
       self.device = device
       self.text_config = configs_dict['text_encoder']
       self.entropy_config = configs_dict['entropy_model']
       # The query (patches) in principle can be of different dimension, but in this implementation we will use the same hidden_size        
       qkvdim = self.text_config.hidden_size
               
       self.byte_embeddings = nn.Embedding(self.entropy_config.vocab_size, qkvdim)
       self.pooling = LpPooling()   
       self.bytes_layers = nn.ModuleList([NakedTransformer(self.text_config, device) for _ in range(self.text_config.num_hidden_layers)])
       # Same things for the cross-attention number of heads. In this implementation, it will be the same as bytes' self attention number of heads, but it could be changed        
       self.patches_layers = nn.ModuleList([MultiHeadAttention(bias=self.text_config.bias, q_dim=qkvdim, k_dim=qkvdim, v_dim=qkvdim, tot_dim=qkvdim, nheads=self.text_config.num_attention_heads) for _ in range(self.text_config.num_hidden_layers)])
       self.norm = RMSNorm(normalized_shape=qkvdim, eps=self.text_config.rms_norm_eps, elementwise_affine=True)
       self.mask_builder = MaskBuilder(self.text_config)
   def debug_raw_input(self, input_tensor, cutting_points):
        #Print on a text file to check if the cut of patches is done correctly
        with open('debug_raw_input.txt', 'a') as f:
            B, S = input_tensor.shape  
            
            for b in range(B):
                f.write(f"\n=== BATCH {b} ===\n")    
                f.write(f"Original text: ")
                original_text=''.join([chr(b-2) if 0 <= b -2 <= 255 else '?' for b in input_tensor[b].cpu().tolist()])
                f.write(original_text)
                f.write('\n PATCHES:')           
                last_tuple=cutting_points[b][-1]
                new_last_tuple=(last_tuple[0],S)
                cutting_points[b][-1]=new_last_tuple
                cuts = [end - start for start, end in cutting_points[b]]
                cut_tensor = input_tensor[b].split(cuts, dim=0)
                for single_cut in cut_tensor:
                    patch_bytes = single_cut.cpu().tolist()
                    patch_text = ''.join([chr(b - 2) if 0 <= b - 2 <= 255 else '?' for b in patch_bytes])
                    f.write(f"{patch_text}\n")

   def create_patches(self, h, cutting_points):
        ### Da rifare! Cambiare logica
        B, S, D = h.shape
        assert B == len(cutting_points), "Size mismatch: batch dim and cutting points length are different"
        
        all_patches = []
        max_patches = max(len(cp) for cp in cutting_points)  
        
        for b in range(B):
            last_tuple=cutting_points[b][-1]
            new_last_tuple=(last_tuple[0],S)
            cutting_points[b][-1]=new_last_tuple
            cuts = [end - start for start, end in cutting_points[b]]
            cut_tensor = h[b].split(cuts, dim=0)
            pooled_patches = []
            for patch in cut_tensor:
                pooled_patch = self.pooling(patch, dim=0)
                pooled_patches.append(pooled_patch)
            
            while len(pooled_patches) < max_patches:
                pooled_patches.append(torch.zeros_like(pooled_patches[0]))
            
            batch_result = torch.stack(pooled_patches, dim=0)
            all_patches.append(batch_result)
        
        #return torch.stack(all_patches, dim=0)  
        return torch.nested.nested_tensor(all_patches, layout=torch.jagged)
   def forward(self, bytes_seq, patches=None, bytegroups=None, cutting_points=None):
        #print("\tStep di debug encoder.")
        #self.debug_raw_input(bytes_seq,cutting_points)
        time_start=27
        h = self.byte_embeddings(bytes_seq)  # 1) Embed bytes
        B, bytes_seqlen, _ = h.shape
        time2=27
        #print(f"\tByte embeddings calcolati in {time2-time_start}")
        # 2) either use given patches, or build them
        if patches is None:
            patches = self.create_patches(h, cutting_points)
        time3=27
        #print(f"\tPatches create in {time3-time2}")
        # 3) The blockmask is computed during the forward pass. Each patch's query can attend only to the bytes forming that patch. For this reason, we will use document_id masking scheme.        
        # (this was for padded) patches_block_mask = self.mask_builder.build_mask_tensor(q_len=patches.shape[1], kv_len=bytes_seqlen, attention_types=['document'], device=self.device, document_mask=bytegroups, batch_size=B)
        patches_block_mask = self.mask_builder.build_mask_tensor(query=patches, kv=h, attention_types=['document'], document_mask=bytegroups, batch_size=B)

        time4=27
        #print(f"\tBlock mask ha richiesto {time4-time3}")
        # 4) stack of local transformer + cross-attn

        for byte_layer, cross_attn in zip(self.bytes_layers, self.patches_layers):
            time_a=27
            h = byte_layer(h)
            time_b=27
            patches = cross_attn(query_input=patches, key_input=h, value_input=h, block_mask=patches_block_mask)
            time_c=27
            #print(f"  Encoder layer: {time_b-time_a} per la self (solo bytes) e {time_c-time_b} per la cross (kv bytes, query patche)")
        time5=27
        #print(f"\tStack transformer loops ha richiesto {time5-time4}")
        # 5) final norms            
        return self.norm(h), self.norm(patches)

class TextDecoder(nn.Module):
   def __init__(self, configs_dict, device='cuda'):
       super().__init__()
       self.device = device
       self.text_config = configs_dict['text_decoder']
       qkvdim = self.text_config.hidden_size
       self.xattn = nn.ModuleList([MultiHeadAttention(bias=self.text_config.bias, q_dim=qkvdim, k_dim=qkvdim, v_dim=qkvdim, tot_dim=qkvdim, nheads=self.text_config.num_attention_heads) for _ in range(self.text_config.num_hidden_layers)])
       self.block = nn.ModuleList([NakedTransformer(self.text_config, device) for _ in range(self.text_config.num_hidden_layers)])
       self.norm = RMSNorm(qkvdim, eps=self.text_config.rms_norm_eps)
       self.output = nn.Linear(qkvdim, self.text_config.vocab_size, bias=False)
       self.mask_builder = MaskBuilder(self.text_config)

   def forward(self, h, patches, bytegroups=None):  
       time1=27
       if bytegroups is not None:
           B, bytes_seqlen, _ = h.shape
           cross_mask = self.mask_builder.build_mask_tensor(
               query=h, 
               kv=patches, 
               attention_types=['document'], 
               nested=True, 
               document_mask=bytegroups,
               batch_size=B
           )
       else:
           cross_mask = None
       time2=27
       #print(f" Decoder: creazione block mask: {time2-time1}") 
       for xattn, block in zip(self.xattn, self.block):
           time_a=27
           h = h + xattn(query_input=h, key_input=patches, value_input=patches, block_mask=cross_mask)
           time_b=27
           h = block(h)
           time_c=27
           #print(f"   Decoder layer: {time_b-time_a} per la cross (kv patch, bytes query) e {time_c-time_b} per la self (bytes)")
       time3=27
       #print(f" Decoder: esecuzione blocchi: {time3-time2}")
       output=self.output(self.norm(h))
       return output
     

class BLTTransfomer(nn.Module):
   """
   This is an implementation of a BLT Transformer but with some modification from the Meta paper.
   
   """
   def __init__(self,entropy_config,text_encoder_config,global_config,text_decoder_config,entropymodel, smoothing=None, device='cuda'):
       super().__init__()
       #Creating a dictionary with the config for each model
       configs=dict()
       configs['entropy_model']=entropy_config
       configs['text_encoder']=text_encoder_config
       configs['text_decoder']=text_decoder_config
       configs['global_transformer']=global_config
       self.configs=configs
       self.device=device
       self.entropymodel = entropymodel
       if smoothing is None:
           self.entropy_smoothing=None
       elif smoothing == -1:   
           self.entropy_smoothing=nn.Parameter(0.5)
       else:
           self.entropy_smoothing=smoothing        
       self.textencoder=TextEncoder(configs, device=device)
       self.textdecoder=TextDecoder(configs,device=device)
       self.latenttransformer=NakedTransformer(configs['global_transformer'], device=device)

   def forward(self,text_tokens,text,smoothing=None):
       #print("Step di debug.")
       time_start=27
       if smoothing is None:
           smoothing=self.entropy_smoothing
       cutting_points, _, bytegroups = self.entropymodel.monotonicity_breakpoints(prompt=text, smoothing=smoothing)
       time2=27
       #print(f"Byte groups calcolati in {time2-time_start}")
       bytegroups=bytegroups.to(self.device)
       h,p=self.textencoder(text_tokens, bytegroups=bytegroups, cutting_points=cutting_points)
       time3=27
       #print(f"Testo codificato in {time3-time2}")
       p=self.latenttransformer(p)
       time4=27
       #print(f"Global transformer ha richiesto {time4-time3}")
       output=self.textdecoder(h,p, bytegroups=bytegroups)
       time5=27
       #print(f"E il decoder: {time3-time2}")
       #print("Uscita dalla forward:")
       ##print(output)
       return output
