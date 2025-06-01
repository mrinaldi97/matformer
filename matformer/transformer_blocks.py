"""
File: matformer/transformer_blocks.py
"""
import torch
import torch.nn as nn
from matformer.transformer_functions import MultiHeadAttention, PackedSwiGLUFFN
from torch.nn import RMSNorm
from matformer.model_config import ModelConfig  
from matformer.utils import LpPooling
from functools import partial
from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    create_block_mask,
    create_mask,
    and_masks,
    or_masks,
    noop_mask
)
#flex_attention = torch.compile(flex_attention)
class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, block_mask=None):
        super().__init__()
        self.input_layernorm = RMSNorm(normalized_shape=config.hidden_dim,eps=config.rms_norm_eps,elementwise_affine=True)
        totdim=config.hidden_dim
        qkvdim=int(totdim)
        
        if config.compile_flexattn==True:
            self.self_attn = torch.compile(MultiHeadAttention(bias=config.bias, q_dim=qkvdim, k_dim=qkvdim, v_dim=qkvdim, tot_dim=totdim, nheads=config.n_heads, block_mask=block_mask))
        else:
            self.self_attn = MultiHeadAttention(bias=config.bias, q_dim=qkvdim, k_dim=qkvdim, v_dim=qkvdim, tot_dim=totdim, nheads=config.n_heads, block_mask=block_mask)      
        self.post_attention_layernorm = RMSNorm(normalized_shape=config.hidden_dim,eps=config.rms_norm_eps,elementwise_affine=True)
        self.mlp = PackedSwiGLUFFN(config)
    def forward(self, x, block_mask=None):
        qkv_input=self.input_layernorm(x)
        x = x + self.self_attn(query_input=qkv_input, key_input=qkv_input, value_input=qkv_input, block_mask=block_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x
 
class MaskBuilder:
    """
    Handles block mask creation logic
    """
    def __init__(self, config):
        self.config = config

    def _mask_fn_causal(self, b, h, q_idx, kv_idx, **kwargs):
        return q_idx >= kv_idx

    def _mask_fn_sliding(self, b, h, q_idx, kv_idx,window_size, **kwargs):
        return kv_idx > q_idx - window_size

    def _mask_fn_document(self, b, h, q_idx, kv_idx, document_mask, **kwargs):
        if document_mask is None:
            print("WARNING: Document block-mask function called without document_mask data.")
            return True # Permissive
        return document_mask[b][q_idx] == document_mask[b][kv_idx]

    def _mask_fn_cloze(self, b, h, q_idx, kv_idx, cloze_mask, **kwargs): 
        if cloze_mask is None:
            print("WARNING: Cloze block-mask function called without cloze_mask.") 
            return True # Permissive
        return cloze_mask[b][q_idx]

    def build_mask_tensor(self, q_len, kv_len,attention_types,device, batch_size=None, num_heads=None,                             
                             is_sliding=False,
                             document_mask=None,
                             cloze_mask=None):
        """
        Builds the list of active mask functions and generates the final mask tensor.
        They can be appended to the final mask in "and" modality or "or" modality.
        """
        and_fns = []
        or_fns = []

        if 'causal' in attention_types:
            and_fns.append(self._mask_fn_causal)

        if is_sliding:
            bound_sliding_fn = partial(self._mask_fn_sliding, window_size=self.config.sliding_window_size)
            and_fns.append(bound_sliding_fn)
        
        if 'document' in attention_types:
            bound_doc_fn = partial(self._mask_fn_document, document_mask=document_mask)
            or_fns.append(bound_doc_fn)

        if 'cloze' in attention_types:
            bound_cloze_fn = partial(self._mask_fn_cloze, cloze_mask=cloze_mask)
            and_fns.append(bound_cloze_fn)

        combined_and_conditions = and_masks(*and_fns) if and_fns else None
        
        final_mask_logic = combined_and_conditions
        if or_fns:
            combined_or_conditions = or_masks(*or_fns)
            if final_mask_logic is not None and combined_or_conditions is not None:
                final_mask_logic = or_masks(final_mask_logic, combined_or_conditions)
            elif combined_or_conditions is not None :
                final_mask_logic = combined_or_conditions
        
        if not and_fns and not or_fns and final_mask_logic is None: 
             pass 

        return create_block_mask(
            mask_mod=final_mask_logic,
            B=batch_size, H=num_heads,
            Q_LEN=q_len, KV_LEN=kv_len,
            BLOCK_SIZE=self.config.block_size_for_attention,
            device=device
        )


class NakedTransformer(nn.Module):
    """
    This transformer implementation purposely misses the embedding
    as well as the "unembedding" layer.
    The reason is that is a Transformer meant to run only on "patches"
    """
    def __init__(self, config: ModelConfig, device):
        super().__init__()
        self.device=device
        self.config = config
        self.mask_builder = MaskBuilder(config)
        self.norm = RMSNorm(normalized_shape=config.hidden_dim, eps=config.rms_norm_eps, elementwise_affine=True)
        self.layers = nn.ModuleList()
        for _ in range(config.n_layers):
            self.layers.append(TransformerBlock(config=config)) 
        self.block_mask=None
        self.sliding_mask=None
        self.config.max_seqlen=self.config.max_seqlen -1 #Una cosa merdosa
        # Generate mask templates in __init__ for "simple" cases (batch invariant cases, no document/cloze)
        if 'causal' in self.config.attention_type:
            self.block_mask=self.mask_builder.build_mask_tensor(self.config.max_seqlen, self.config.max_seqlen, attention_types=['causal'],device=self.device)
        if 'sliding' in self.config.attention_type:
            if 'causal' in self.config.attention_type:
                self.block_mask=self.mask_builder.build_mask_tensor(self.config.max_seqlen, self.config.max_seqlen, attention_types=['sliding','causal'], is_sliding=True,device=self.device)
            else:
                self.block_mask=self.mask_builder.build_mask_tensor(self.config.max_seqlen, self.config.max_seqlen, attention_types=['sliding'], is_sliding=True,device=self.device)    

    def forward(self, x, y_cross=None, document_mask=None, cloze_mask=None, inference_fix=False):
        batch_size, q_len, _ = x.shape
        kv_len = y_cross.shape[1] if y_cross is not None else q_len 
        """
         We have to decide in the forward if employing the masks generated during the __init__ (much faster)
         or if we need to generate new masks in cases such as:
         1) Document attention type and doc.mask passed;
         2) Cloze attention type and cloze mask passed;
         3) The model is used at an higher seq_len
        """     
        assert self.config.sliding_type in ['full','disabled','partial'], "Invalid sliding type config."
        device=x.device
        #if True: #Check why resize doesn't work, important for inference
        if document_mask is not None or cloze_mask is not None or q_len>self.config.max_seqlen or inference_fix==True:
            # In these cases I have to regenerate the masks
            if self.config.sliding_type != 'disabled':
                sliding_mask=self.mask_builder.build_mask_tensor(q_len, kv_len, attention_types=self.config.attention_type, is_sliding=True,device=device)
            if self.config.sliding_type != 'full':
                block_mask=self.mask_builder.build_mask_tensor(q_len, kv_len, attention_types=self.config.attention_type, is_sliding=False,device=device)             
        else:
            block_mask=self.block_mask
            sliding_mask=self.block_mask
        # --- Layers ---
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx in self.config.sliding_layers or self.config.sliding_type=='full' :
                x = layer(x, block_mask=sliding_mask)
            else:
                x = layer(x, block_mask=block_mask)
        x = self.norm(x)
        return x
class TransformerWithEmbeddingHead(nn.Module):
    """
    
    """
    def __init__(self,config: ModelConfig,device):
        super().__init__()
        self.embed_tokens = nn.Embedding(num_embeddings=config.vocab_size,embedding_dim=config.hidden_dim,padding_idx=config.pad_id)
        self.transformer = NakedTransformer(config,device)
    def forward(self,x, **kwargs):
        embeddings=self.embed_tokens(x)
        return self.transformer(embeddings,**kwargs)
    
class TransformerWithLMHead(nn.Module):
    def __init__(self,config: ModelConfig,device):
        super().__init__()      
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size)      
        self.transformer = TransformerWithEmbeddingHead(config,device)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.transformer.embed_tokens.weight

    def forward(self,x, **kwargs):
        return self.lm_head(self.transformer(x, **kwargs))  
    
        
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
       self.patches_layers = nn.ModuleList([MultiHeadAttention(q_dim=qkvdim, k_dim=qkvdim, v_dim=qkvdim, tot_dim=qkvdim, nheads=self.text_config.n_heads) for _ in range(self.text_config.n_layers)])
       self.norm = RMSNorm(normalized_shape=qkvdim, eps=self.text_config.rms_norm_eps, elementwise_affine=True)
       self.mask_builder = MaskBuilder(self.text_config)

   def create_patches(self, h, bytegroups):
       B, _, D = h.shape
       P = bytegroups.max().item()
       patches = h.new_zeros(B, P, D)
       for group in range(1, P + 1):
           mask = (bytegroups == group)
           patches[:, group-1] = self.pooling(h[:, mask], dim=1)
       return patches

   def forward(self, bytes_seq, patches=None, bytegroups=None):  
       h = self.byte_embeddings(bytes_seq)  # 1) Embed bytes
       B, bytes_seqlen, _ = h.shape
       # 2) either use given patches, or build them
       if patches is None:
           patches = self.create_patches(h, bytegroups)

       # 3) The blockmask is computed during the forward pass. Each patch's query can attend only to the bytes forming that patch. For this reason, we will use document_id masking scheme.        
       block_mask = self.mask_builder.build_mask_tensor(q_len=bytes_seqlen, kv_len=bytes_seqlen, attention_types=['document'], device=self.device, document_mask=bytegroups)
       # 4) stack of local transformer + cross-attn
       
       for byte_layer, cross_attn in zip(self.bytes_layers, self.patches_layers):
           h = byte_layer(h)
           patches = cross_attn(query_input=patches, key_input=h, value_input=h, block_mask=block_mask)
       # 5) final norms            
       return self.norm(h), self.norm(patches)

class TextDecoder(nn.Module):
   def __init__(self, configs_dict, device='cuda'):
       super().__init__()
       self.device = device
       self.text_config = configs_dict['text_decoder']
       qkvdim = self.text_config.hidden_dim
       self.xattn = nn.ModuleList([MultiHeadAttention(q_dim=qkvdim, k_dim=qkvdim, v_dim=qkvdim, tot_dim=qkvdim, nheads=self.text_config.n_heads) for _ in range(self.text_config.n_layers)])
       self.block = nn.ModuleList([NakedTransformer(self.text_config, device) for _ in range(self.text_config.n_layers)])
       self.norm = RMSNorm(qkvdim, eps=self.text_config.rms_norm_eps)
       self.output = nn.Linear(qkvdim, self.text_config.vocab_size, bias=False)
       self.mask_builder = MaskBuilder(self.text_config)

   def forward(self, h, patches, bytegroups=None):  
       if bytegroups is not None:
           B, bytes_seqlen, _ = h.shape
           cross_mask = self.mask_builder.build_mask_tensor(
               q_len=bytes_seqlen, 
               kv_len=patches.shape[1], 
               attention_types=['document'], 
               device=self.device, 
               document_mask=bytegroups,
               batch_size=B
           )
       else:
           cross_mask = None
           
       for xattn, block in zip(self.xattn, self.block):
           h = h + xattn(query_input=h, key_input=patches, value_input=patches, block_mask=cross_mask)
           h = block(h)
       return self.output(self.norm(h))


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
       self.textdecoder=TextDecoder(configs,device=self.device)
       self.latenttransformer=NakedTransformer(configs['global_transformer'])

   def forward(self,text,smoothing=None):
       if smoothing is None:
           smoothing=self.entropy_smoothing
       _, _, bytegroups = self.entropymodel.monotonicity_breakpoints(prompt=text, smoothing=smoothing)
       h,p=self.textencoder(text, bytegroups=bytegroups)
       p=self.latenttransformer(p)
       output=self.textdecoder(h,p, bytegroups=bytegroups)
       return output
