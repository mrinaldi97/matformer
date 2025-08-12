"""
File: matformer/transformer_blocks.py
"""
import torch
import torch.nn as nn
from matformer.transformer_functions import MultiHeadAttention, PackedSwiGLUFFN
from matformer.tensors_dataclasses import TensorDC, NormalTensor, PaddedTensor, UnpaddedTensor, ModuleWrapper
from torch.nn import RMSNorm
from matformer.model_config import ModelConfig  
from matformer.utils import LpPooling, MaskBuilder
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
from matformer.masked_models import maskerator
from matformer.tokenizers import ByteLevelTokenizer,MatformerTokenizer
import torch.nn.functional as F
from tqdm import tqdm

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
    
    def __init__(self, config: ModelConfig, block_mask=None):
        super().__init__()
        self.input_layernorm = ModuleWrapper(RMSNorm(normalized_shape=config.hidden_size,eps=config.rms_norm_eps,elementwise_affine=True))
        self.self_attn = MultiHeadAttention(bias=config.bias, q_dim=config.hidden_size, k_dim=config.hidden_size, v_dim=config.hidden_size, hidden_size=config.hidden_size, nheads=config.num_attention_heads, block_mask=block_mask, attn_impl=config.attn_impl, alibi=config.alibi, is_causal=config.is_causal)      
        self.post_attention_layernorm = ModuleWrapper(RMSNorm(normalized_shape=config.hidden_size,eps=config.rms_norm_eps,elementwise_affine=True))
        self.mlp = PackedSwiGLUFFN(config)
        self.config=config
    def forward(self, x, block_mask=None, sliding=False):
        x = self.input_layernorm(x)
        x = x + self.self_attn(query_input=x, key_input=x, value_input=x, block_mask=block_mask, sliding=sliding, sliding_window_size=self.config.sliding_window_size)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x
    def debug_forward(self, x, block_mask=None, sliding=False):
        x0 = x  
        x1 = self.input_layernorm(x0)
        x1.tensor.register_hook(lambda g: print("grad @ input_layernorm:", g.norm().item()))
        
        a = self.self_attn(query_input=x1, key_input=x1, value_input=x1, block_mask=block_mask, sliding=sliding, sliding_window_size=self.config.sliding_window_size)
        a.tensor.register_hook(lambda g: print("grad @ attn_out:", g.norm().item()))

        x2 = x1 + a
        m = self.post_attention_layernorm(x2)
        m.tensor.register_hook(lambda g: print("grad @ post_ln:", g.norm().item()))
        
        f = self.mlp(m)  
        f.tensor.register_hook(lambda g: print("grad @ mlp_out:", g.norm().item()))
        
        x3 = x2 + f
        return x3




class NakedTransformer(nn.Module):
    """
    This transformer implementation purposely misses the embedding
    as well as the "unembedding" layer.
    The reason is that is a Transformer meant to run only on "patches".
    It applies n transformer blocks as defined in the ModelConfig
    Still needs some revisions:
        1) High VRAM consumption with Flex Attention and in particular if nested tensors are used;
        2) A decision should be made about where to compute block masks
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        #If attention is not flash, it's a good idea to cache the block mask:
        if config.attention_type != 'flash':
            self.mask_builder = MaskBuilder(config)
            self.block_mask=None
            self.sliding_mask=None          
        self.norm = ModuleWrapper(RMSNorm(normalized_shape=config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True))
        self.layers = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.layers.append(TransformerBlock(config=config)) 

        self.config.max_position_embeddings=self.config.max_position_embeddings -1 # Da ricordarsi perchè e dove serviva, trovare in caso soluzione più pulita             
        """
        # Generate mask templates in __init__ for "simple" cases (batch invariant cases, no document)
        THIS IS DISABLED AFTER SWITCHING TO NESTED TENSOR LAYOUT, BECAUSE THEY NEED TO BE RECOMPUTED EVERY TIME
        HOWEVER, THIS COULD SPEED UP THE MODEL, SO LET'S RETHINK ABOUT THIS AND EVENTUALLY PUT SOME CONDITION
        if 'causal' in self.config.attention_type:
            self.block_mask=self.mask_builder.build_mask_tensor(self.config.max_position_embeddings, self.config.max_position_embeddings, attention_types=['causal'],device=self.device)
        if 'sliding' in self.config.attention_type:
            if 'causal' in self.config.attention_type:
                self.block_mask=self.mask_builder.build_mask_tensor(self.config.max_position_embeddings, self.config.max_position_embeddings, attention_types=['sliding','causal'], is_sliding=True,device=self.device)
            else:
                self.block_mask=self.mask_builder.build_mask_tensor(self.config.max_position_embeddings, self.config.max_position_embeddings, attention_types=['sliding'], is_sliding=True,device=self.device)    
        """
    def forward(self, x, y_cross=None, document_mask=None, cloze_mask=None, inference_fix=False):
        
        q_len=x.original_seq_len if isinstance(x,UnpaddedTensor) else x.shape[1]
        kv_len = y_cross.shape[1] if y_cross is not None else q_len  # If we are not in cross-attention settings, we take x for both query and kv
        """
         We have to decide in the forward if employing the masks generated during the __init__ (much faster)
         or if we need to generate new masks in cases such as:
         1) Document attention type and doc.mask passed;
         2) The model is used at an higher seq_len
        """     
        assert self.config.sliding_type in ['full','disabled','partial'], "Invalid sliding type config."
        if False: #THIS IS DISABLED AFTER SWITCHING TO NESTED TENSOR LAYOUT, BECAUSE THEY NEED TO BE RECOMPUTED EVERY TIME
        #if document_mask is not None or cloze_mask is not None or q_len>self.config.max_position_embeddings or inference_fix==True:
            # In these cases I have to regenerate the masks
            #with torch.no_grad():
            #    dummy_query=self.self_attn.dummy_get_query(x=x)
            if self.config.sliding_type != 'disabled':
                sliding_mask=self.mask_builder.build_mask_tensor(query=x, kv=y_cross, attention_types=self.config.attention_type, is_sliding=True,nested=x.tensor.is_nested)
            if self.config.sliding_type != 'full':
                block_mask=self.mask_builder.build_mask_tensor(query=x, kv=y_cross, attention_types=self.config.attention_type, is_sliding=False,nested=x.tensor.is_nested)             
        else:
            block_mask=self.block_mask
            sliding_mask=self.block_mask
            
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx in self.config.sliding_layers or self.config.sliding_type=='full' :
                x = layer(x, block_mask=sliding_mask, sliding=True)
            else:
                x = layer(x, block_mask=block_mask, sliding=False)
        x = self.norm(x)

        return x
        
class TransformerWithEmbeddingHead(nn.Module):
    """
    Adding an embedding layer at the beginning
    """
    def __init__(self,config: ModelConfig):
        super().__init__()
        self.embed_tokens = ModuleWrapper(nn.Embedding(num_embeddings=config.vocab_size,embedding_dim=config.hidden_size,padding_idx=config.pad_token_id))
        self.transformer = NakedTransformer(config)
    def forward(self,x, **kwargs): 
        embeddings=self.embed_tokens(x)
        return self.transformer(embeddings,**kwargs)



		 
	

class TransformerWithLMHead(nn.Module):
    """
    Adding an LM Head to TransformerWithEmbeddingHead. This is enough for Bert-like/GPT-like models.
    """
    def __init__(self,config: ModelConfig,tokenizer=None,device=None):
        super().__init__()      
        self.lm_head = ModuleWrapper(nn.Linear(config.hidden_size, config.vocab_size))
        self.transformer = TransformerWithEmbeddingHead(config)
        self.device=device #This is used only for inference!
        if config.tie_word_embeddings:
            self.lm_head.weight = self.transformer.embed_tokens.weight
        self.config=config
        self.tokenizer=tokenizer
        
    def forward(self,x,**kwargs):

        x=self.transformer(x,**kwargs)
        x= self.lm_head(x)
        return x

class TransformerWithClassificationHead(TransformerWithEmbeddingHead):
    def __init__(self, config: ModelConfig, tokenizer=None, pooling_type='mean', num_features=2):
        super().__init__(config)
        self.classification_head = ModuleWrapper(nn.Linear(config.hidden_size, num_features))
        self.config = config
        self.tokenizer = tokenizer
        self.pooling_type = pooling_type
        self.num_features = num_features

    def forward(self, x, attention_mask=None):
        outputs = self.transformer(x) # (B,S,D)

        if self.pooling_type == 'cls':
            # [CLS] in pos. 0
            pooled = outputs[:, 0, :]
        elif self.pooling_type == 'mean':
            print("NOT IMPLEMENTED")
            raise ValueError(f"Non ho ancora implementato la media")
        else:
            raise ValueError(f"{self.pooling_type} not in 'cls','mean'")

        logits = self.classification_head(pooled)
        return logits
        
        

class BERTModel(TransformerWithLMHead):
    def inference_testing(self, input_text, masking_ratio=0.25,datatype=torch.bfloat16):
        #Copiata qui al volo per far girare i modelli già addestrati, da togliere. 
        sequence = self.tokenizer.encode(input_text)
        sequence = torch.tensor(sequence).unsqueeze(0).to(self.device)
        masked_list, cloze_list = maskerator(sequence, MASK_TOKEN=0, substitution_rate=masking_ratio)
        masked_list.to(self.device)
        masked_sequence = NormalTensor(tensor=masked_list)
        model_input=deepcopy(masked_sequence)
        with torch.no_grad():
            logits = self(model_input)
        predictions = torch.argmax(logits.tensor, dim=-1)
        targets = sequence.squeeze()
        mask = cloze_list.squeeze().bool()
        correct = (predictions.squeeze()[mask] == targets[mask]).sum().item()
        total = mask.sum().item()
        accuracy = correct / total if total > 0 else 0.0
        out_tokens=list()
        for i,token in enumerate(masked_sequence.tensor.squeeze().tolist()):
            if token != 0:
                print(token)
                out_tokens.append(self.tokenizer.decode(token))
            else:
                out_tokens.append(f"[ {self.tokenizer.decode(predictions.squeeze()[i])} ]")
        
        return accuracy, out_tokens
class Autoregressive_Model(TransformerWithLMHead):
    def generate(self, prompt=None, max_length=100, temperature=1.0, top_k=0, top_p=0.9):
        """
        Generate a sequence starting from an optional prompt

        Args:
            prompt: Optional starting prompt as bytes or None for empty start
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature (1.0=normal, <1.0=more conservative)
            top_k: Limit sampling to top k tokens (0 for no limit)
            top_p: Nucleus sampling probability threshold

        Returns:
            ByteTensor of generated sequence
        """
        self.eval()  

        if prompt is None:
            current_ids = torch.tensor([[self.config.bos_token_id]], device=self.device)
        else:
            # Tokenize the prompt if it's provided as bytes
            assert isinstance(prompt, str), "Prompt expected as string"
            tokenizer = self.tokenizer
            prompt_ids = torch.tensor(tokenizer.encode(prompt), device=self.device)
            current_ids = torch.tensor(prompt_ids.unsqueeze(0), device=self.device)
            # The forward expects: [batch_size, seq_len, vocab_size]
            print(f"Prompt_ids shape: {prompt_ids.shape}")
            print(f"Current_ids shape: {current_ids.shape}")

        for _ in tqdm(range(max_length)):
            with torch.no_grad():
                outputs = self(NormalTensor(tensor=current_ids)).tensor
                
            next_token_logits = outputs[:, -1, :]  # Get logits for the last position

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                top_k_values, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(1, top_k_indices, top_k_values)

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, float('-inf'))

            # Convert logits to probabilities
            probs = F.softmax(next_token_logits, dim=-1)

            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            #print(f"Generated: {next_token.item()} Char: {tokenizer.decode(next_token.item())}")
            # Append the new token to the sequence
            current_ids = torch.cat([current_ids, next_token], dim=1)

            # Stop if we generated an EOS token
            if next_token.item() == self.config.eos_token_id:
                break
        return self.tokenizer.decode(current_ids.squeeze().tolist())
class EntropyModel(Autoregressive_Model):
    def compute_entropy(self, prompts):
        """
        Return a tensor of size sequence length containing the entropy for each received text
        """
        self.eval()
        if isinstance(prompts, str):
            prompts = [prompts]

        prompt_ids = self.tokenizer.batch_encode(prompts)  # [B, seq_len]
        prompt_ids = prompt_ids.to(self.device)

        with torch.no_grad():
            logits = self(prompt_ids)  # [B, seq_len, vocab_size]

        epsilon = 1e-10
        probs = torch.nn.functional.softmax(logits.tensor, dim=-1)
        logprobs = torch.log(probs + epsilon)
        entropy = -torch.sum(probs * logprobs, dim=-1)  # [B, seq_len]

        return entropy[:, 1:] 

    def monotonicity_breakpoints(self, prompt=None, entropy=None, smoothing=None):
        # Da cambiare: il taglio delle patch provenienti dal text encoder potrebbe effettuarsi qui, sfruttando i nested tensor
        # Al momento l'implementazione che prevede il ciclo sui batch nel text encoder funziona ma è esageratamente inefficiente a livello di tempi di calcolo
        if smoothing is None:
            smoothing = 0
            print("WARNING: You are running the entropy model without a smoothing set.")

        if prompt is not None:
            entropy = self.compute_entropy(prompt)  # Expected shape: [B, seq_len]
        elif entropy is None:
            raise ValueError("Either provide `prompt` or `entropy`.")

        if entropy.dim() == 1:
            entropy = entropy.unsqueeze(0)  # Make it batched [1, seq_len]

        B, seq_len = entropy.shape
        cutting_masks = torch.zeros_like(entropy, device='cpu')
        group_masks = torch.zeros_like(entropy, dtype=torch.long, device='cpu')
        cutting_points_all = []

        for b in range(B):
            ent = entropy[b].cpu()
            cutting_points = []
            cutting_mask = torch.zeros(seq_len, device='cpu')
            prev_entr = float('inf')
            start_point = 0
            for i in range(seq_len):
                if ent[i] > prev_entr + smoothing:
                    cutting_points.append((start_point, i+1))
                    cutting_mask[i] = 1
                    start_point = i+1
                prev_entr = ent[i]
            group_mask = torch.cumsum(cutting_mask, dim=0)
            cutting_masks[b] = cutting_mask
            group_masks[b] = group_mask
            cutting_points_all.append(cutting_points)
        #print("------------- ENTROPY DEBUG ------------------")
        #print("----CUTTING MASK:")
        #print(cutting_masks)
        #print("----- GROUP MASKs")
        #print(group_masks)
        #print("CUTTING POINTS")
        #print(cutting_points_all)
        return cutting_points_all, cutting_masks, group_masks

    def cut_text(self,text,cutting_points=None,smoothing=None):
        """
        Cut a text according to cutting points
        """
        if cutting_points is None:
            cutting_points,_=self.monotonicity_breakpoints(prompt=text,smoothing=smoothing)
        text_chunks=[text[i:j] for i,j in cutting_points]
        return text_chunks
