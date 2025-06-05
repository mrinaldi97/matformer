"""
File: matformer/transformer_blocks.py
"""
import torch
import torch.nn as nn
from matformer.transformer_functions import MultiHeadAttention, PackedSwiGLUFFN
from torch.nn import RMSNorm
from matformer.model_config import ModelConfig  
from matformer.utils import LpPooling
from functools import partial, reduce
from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    create_block_mask,
    create_nested_block_mask,
    create_mask,
    and_masks,
    or_masks,
    noop_mask
)
import gc
#import time
def printmem(text):
    # Una funzioncina rozza per controllare l'uso della memoria, da togliere una volta che il codice è completo
    device = torch.device('cuda:0')
    free, total = torch.cuda.mem_get_info(device)
    mem_used_MB = (total - free) / 1024 ** 2
    print(f"Memory at {text}: {mem_used_MB}")
#flex_attention = torch.compile(flex_attention) # Chiarire questione compilazione (dove? di che tipo? migliora o peggiora? in che casi farla?)

class TransformerBlock(nn.Module):
    """ A transformer self-attention block
        It applies a pre layernorm 
        A self-attention layer
        A SwiGLU Mlp Layer
        A post layer norm
        It takes all the necessary configuration from the ModelConfig object
        The block_mask for the attention can be passed either at the init or during the forward
    """
    
    def __init__(self, config: ModelConfig, block_mask=None, attn_impl='flex'):
        super().__init__()
        self.input_layernorm = RMSNorm(normalized_shape=config.hidden_dim,eps=config.rms_norm_eps,elementwise_affine=True)
        qkvdim=int(config.hidden_dim)
        self.self_attn = MultiHeadAttention(bias=config.bias, q_dim=qkvdim, k_dim=qkvdim, v_dim=qkvdim, tot_dim=qkvdim, nheads=config.n_heads, block_mask=block_mask, attn_impl=attn_impl)      
        self.post_attention_layernorm = RMSNorm(normalized_shape=config.hidden_dim,eps=config.rms_norm_eps,elementwise_affine=True)
        self.mlp = PackedSwiGLUFFN(config)
    def forward(self, x, block_mask=None):
        qkv_input=self.input_layernorm(x)
        x = x + self.self_attn(query_input=qkv_input, key_input=qkv_input, value_input=qkv_input, block_mask=block_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x



class MaskBuilder:
    def __init__(self, config, attn_impl='flex'):
        self.attn_impl=attn_impl
        self.config = config

    def _get_masks(self, attention_types, L, S, B, device, **kwargs):
        q_idx = torch.arange(L, device=device).view(L, 1)
        k_idx = torch.arange(S, device=device).view(1, S)
        
        and_masks = []
        or_masks = []
        
        if 'causal' in attention_types:
            and_masks.append(q_idx >= k_idx)
        
        if kwargs.get('is_sliding'):
            and_masks.append(torch.abs(q_idx - k_idx) <= self.config.sliding_window_size)
        
        if 'cloze' in attention_types and kwargs.get('cloze_mask') is not None:
            and_masks.append(kwargs['cloze_mask'][:, :, None].expand(B, L, S))
        
        if 'document' in attention_types and kwargs.get('document_mask') is not None:
            doc = kwargs['document_mask']
            or_masks.append(doc[:, :, None] == doc[:, None, :])
        
        return and_masks, or_masks

    def _mask_fn(self, b, h, q_idx, kv_idx, attention_types, **kwargs):
        and_result = torch.ones_like(q_idx, dtype=torch.bool)
        or_result = torch.zeros_like(q_idx, dtype=torch.bool)
        
        if 'causal' in attention_types:
            and_result = and_result & (q_idx >= kv_idx)
        
        if kwargs.get('is_sliding'):
            and_result = and_result & (torch.abs(q_idx - kv_idx) <= self.config.sliding_window_size)
        
        if 'cloze' in attention_types and kwargs.get('cloze_mask') is not None:
            and_result = and_result & kwargs['cloze_mask'][b, q_idx]
        
        if 'document' in attention_types and kwargs.get('document_mask') is not None:
            or_result = or_result | (kwargs['document_mask'][b, q_idx] == kwargs['document_mask'][b, kv_idx])
        
        return and_result | or_result

    def build_mask_tensor(self, attention_types, query, kv=None, batch_size=None, num_heads=None, is_sliding=False, document_mask=None, cloze_mask=None, nested=False, **kwargs):
        kv = kv or query        
        B, L, S = query.shape[0], query.shape[-2], kv.shape[-2]
        if self.attn_impl == 'sdpa':
            if kwargs.get('nested'):
                print("WARNING: Attention mask not supported in SDPA with nested tensors.")
                return None
            
            and_masks, or_masks = self._get_masks(attention_types, L, S, B, query.device, **kwargs)
            
            final_mask = torch.ones(B, L, S, dtype=torch.bool, device=query.device)
            if and_masks:
                final_mask = reduce(torch.logical_and, and_masks)
            if or_masks:
                final_mask = final_mask | reduce(torch.logical_or, or_masks)
            
            return final_mask
        elif self.attn_impl == 'flex':
            mask_fn = lambda b, h, q, k: self._mask_fn(b, h, q, k, attention_types, **kwargs)
            if kwargs.get('nested'):
                return create_nested_block_mask(mask_mod=mask_fn,q_nt=query,kv_nt=kv,B=batch_size, H=num_heads, device=query.device)
            else:
                return create_block_mask(mask_mod=mask_fn, Q_LEN=L, KV_LEN=L, B=batch_size, H=num_heads, device=query.device)
        else:
            print("Unsupported attention implementation!")
            return None




class NakedTransformer(nn.Module):
    """
    This transformer implementation purposely misses the embedding
    as well as the "unembedding" layer.
    The reason is that is a Transformer meant to run only on "patches". O
    It applies n transformer blocks as defined in the ModelConfig
    Still needs some revisions:
        1) High VRAM consumption with Flex Attention and in particular if nested tensors are used;
        2) A decision should be made about where to compute block masks
    """
    def __init__(self, config: ModelConfig, device, attn_impl='sdpa'):
        super().__init__()
        self.device=device
        self.config = config
        self.mask_builder = MaskBuilder(config, attn_impl=attn_impl)
        self.norm = RMSNorm(normalized_shape=config.hidden_dim, eps=config.rms_norm_eps, elementwise_affine=True)
        self.layers = nn.ModuleList()
        for _ in range(config.n_layers):
            self.layers.append(TransformerBlock(config=config, attn_impl=attn_impl)) 
        self.block_mask=None
        self.sliding_mask=None
        self.config.max_seqlen=self.config.max_seqlen -1 # Da ricordarsi perchè e dove serviva, trovare in caso soluzione più pulita
        #self.self_attn = MultiHeadAttention(bias=config.bias, q_dim=config.hidden_dim, k_dim=config.hidden_dim, v_dim=config.hidden_dim, tot_dim=config.hidden_dim, nheads=config.n_heads, block_mask=self.block_mask)              
        """
        # Generate mask templates in __init__ for "simple" cases (batch invariant cases, no document/cloze)
        THIS IS DISABLED AFTER SWITCHING TO NESTED TENSOR LAYOUT, BECAUSE THEY NEED TO BE RECOMPUTED EVERY TIME
        HOWEVER, THIS COULD SPEED UP THE MODEL, SO LET'S RETHINK ABOUT THIS AND EVENTUALLY PUT SOME CONDITION
        if 'causal' in self.config.attention_type:
            self.block_mask=self.mask_builder.build_mask_tensor(self.config.max_seqlen, self.config.max_seqlen, attention_types=['causal'],device=self.device)
        if 'sliding' in self.config.attention_type:
            if 'causal' in self.config.attention_type:
                self.block_mask=self.mask_builder.build_mask_tensor(self.config.max_seqlen, self.config.max_seqlen, attention_types=['sliding','causal'], is_sliding=True,device=self.device)
            else:
                self.block_mask=self.mask_builder.build_mask_tensor(self.config.max_seqlen, self.config.max_seqlen, attention_types=['sliding'], is_sliding=True,device=self.device)    
        """
    def forward(self, x, y_cross=None, document_mask=None, cloze_mask=None, inference_fix=False):
        #Some part of the functions behaves different if x is a nested tensor, so first of all let's figure it out.
        if x.is_nested:
            nested=True
        else:
            nested=False
        
        batch_size, q_len, _ = x.shape
        kv_len = y_cross.shape[1] if y_cross is not None else q_len  # If we are not in cross-attention settings, we take x for both query and kv
        """
         We have to decide in the forward if employing the masks generated during the __init__ (much faster)
         or if we need to generate new masks in cases such as:
         1) Document attention type and doc.mask passed;
         2) Cloze attention type and cloze mask passed;
         3) The model is used at an higher seq_len
        """     
        assert self.config.sliding_type in ['full','disabled','partial'], "Invalid sliding type config."
        device=x.device
        #gc.collect()
        #torch.cuda.empty_cache()
        #printmem("After having collected (first)")        
        #printmem("Before creating the masks")
        if True: #THIS IS DISABLED AFTER SWITCHING TO NESTED TENSOR LAYOUT, BECAUSE THEY NEED TO BE RECOMPUTED EVERY TIME
        #if document_mask is not None or cloze_mask is not None or q_len>self.config.max_seqlen or inference_fix==True:
            # In these cases I have to regenerate the masks
            #with torch.no_grad():
            #    dummy_query=self.self_attn.dummy_get_query(x=x)
            if self.config.sliding_type != 'disabled':
                sliding_mask=self.mask_builder.build_mask_tensor(query=x, kv=y_cross, attention_types=self.config.attention_type, is_sliding=True,nested=nested)
            if self.config.sliding_type != 'full':
                block_mask=self.mask_builder.build_mask_tensor(query=x, kv=y_cross, attention_types=self.config.attention_type, is_sliding=False,nested=nested)             
        else:
            block_mask=self.block_mask
            sliding_mask=self.block_mask
        # --- Layers ---
  
        #printmem("After having created the masks")
        #gc.collect()
        #torch.cuda.empty_cache()
        #printmem("After having collected (before the blocks)")
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx in self.config.sliding_layers or self.config.sliding_type=='full' :
                x = layer(x, block_mask=sliding_mask)
            else:
                x = layer(x, block_mask=block_mask)
        x = self.norm(x)
        #printmem("After having executed the forward")
        #gc.collect()
        #torch.cuda.empty_cache()
        #printmem("After having collected (after the blocks)")
        return x
class TransformerWithEmbeddingHead(nn.Module):
    """
    Adding an embedding layer at the beginning
    """
    def __init__(self,config: ModelConfig,device):
        super().__init__()
        self.embed_tokens = nn.Embedding(num_embeddings=config.vocab_size,embedding_dim=config.hidden_dim,padding_idx=config.pad_id)
        self.transformer = NakedTransformer(config,device)
    def forward(self,x, **kwargs):
        embeddings=self.embed_tokens(x)
        return self.transformer(embeddings,**kwargs)
    
class TransformerWithLMHead(nn.Module):
    """
    Adding an LM Head to TransformerWithEmbeddingHead. This is sufficient for Bert-like/GPT-like models.
    """
    def __init__(self,config: ModelConfig,device):
        super().__init__()      
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size)      
        self.transformer = TransformerWithEmbeddingHead(config,device)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.transformer.embed_tokens.weight

    def forward(self,x, **kwargs):
        return self.lm_head(self.transformer(x, **kwargs))  
    
  
  
# From now on, the "core" it's over. The code below regards BLT implementation, that is still in WIP phase  


      
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

#The BLT-Specific classes should be removed from here and moved in another file
class TextEncoder(nn.Module):

   def __init__(self, configs_dict, device='cuda'): 
       super().__init__()
       self.device = device
       self.text_config = configs_dict['text_encoder']
       self.entropy_config = configs_dict['entropy_model']
       # The query (patches) in principle can be of different dimension, but in this implementation we will use the same hidden_dim        
       qkvdim = self.text_config.hidden_dim
               
       self.byte_embeddings = nn.Embedding(self.entropy_config.vocab_size, qkvdim)
       self.pooling = LpPooling()   
       self.bytes_layers = nn.ModuleList([NakedTransformer(self.text_config, device) for _ in range(self.text_config.n_layers)])
       # Same things for the cross-attention number of heads. In this implementation, it will be the same as bytes' self attention number of heads, but it could be changed        
       self.patches_layers = nn.ModuleList([MultiHeadAttention(bias=self.text_config.bias, q_dim=qkvdim, k_dim=qkvdim, v_dim=qkvdim, tot_dim=qkvdim, nheads=self.text_config.n_heads) for _ in range(self.text_config.n_layers)])
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
       qkvdim = self.text_config.hidden_dim
       self.xattn = nn.ModuleList([MultiHeadAttention(bias=self.text_config.bias, q_dim=qkvdim, k_dim=qkvdim, v_dim=qkvdim, tot_dim=qkvdim, nheads=self.text_config.n_heads) for _ in range(self.text_config.n_layers)])
       self.block = nn.ModuleList([NakedTransformer(self.text_config, device) for _ in range(self.text_config.n_layers)])
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
