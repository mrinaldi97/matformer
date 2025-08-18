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
from datetime import datetime
import sys
sys.path.append('../')
from dataclasses import replace

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

        #self.config.max_position_embeddings=self.config.max_position_embeddings -1 # Da ricordarsi perchè e dove serviva, trovare in caso soluzione più pulita             
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



         
    
class TransformerWithCharAutoencoder(nn.Module):
    def __init__(self,config:ModelConfig,device=None, tokenizer=None):
        super().__init__()
        from char_autoencoder.autoencoders import TransCharAutoencoder, TransCharAutoencoder_Encoder, TransCharAutoencoder_Decoder
        self.encoder=TransCharAutoencoder_Encoder(config=config.encoder)
        self.decoder=TransCharAutoencoder_Decoder(config=config.decoder)
        self.transformer=NakedTransformer(config)
        self.projection_in=ModuleWrapper(nn.Linear(config.encoder.hidden_size,config.hidden_size))
        self.projection_out=ModuleWrapper(nn.Linear(config.hidden_size,config.decoder.hidden_size))
    def forward(self, x, **kwargs):
        """
        Input:
            x: TensorDC with fields:
               - tensor: LongTensor of token ids, shape [B, P, S]  (or [P, S] if UnpaddedTensor)
               - padding_mask: BoolTensor, shape [B, P, S]
                   False = real token, True = padded token *inside* that patch S.
                   Additionally, whole "fake" patches have all True across S.

            B = batch size (number of sequences)
            P = max number of patches per sequence (global big-transformer length)
            S = tokens per patch (chars per patch)

        Goal:
            1) Skip compute for fake patches in encoder and decoder.
            2) Keep big transformer input [B, P, D] with fixed P (no inter-sequence mixing).
            3) Preserve padding_mask; return it broadcast to decoder length.
        """

        # 1) Creating padding masks
        if isinstance(x, UnpaddedTensor):
            
            B = 1 # Unpadding is as if batch size is 1
            P, S = x.tensor.shape
            padding_mask = torch.zeros((B, P, S), dtype=torch.bool, device=x.tensor.device)
        else:
            B, P, S = x.tensor.shape
            padding_mask = x.padding_mask  # [B, P, S], True = padded token
            assert padding_mask.shape == (B, P, S)

        # To align with big transformer sequence length, there are "Fake patch" where all S tokens are padded
        # patch_mask[b, p] == True  means a "fake patch"
        patch_mask = padding_mask.all(dim=-1)     #[B,P] Bool => This indicates the patches where each element is True, thus patches that are entirely padding and should be skipped     
        keep_idx = (~patch_mask).reshape(B * P) #Patches to be encoded, that contains actual charachters      # [B*P], Bool
        num_real = keep_idx.sum().item()              # N_real

        # 2) Encoder (only on real patches to save compute)
        #  [B, P, S] => [B*P, S], only "real patches" (keep_idx), final shape => [N_real, S].
        flat_x = replace(
            x,
            tensor=x.tensor.view(B * P, S)[keep_idx],                  # [N_real, S]
            padding_mask=padding_mask.view(B * P, S)[keep_idx]         # [N_real, S]
        )


        flat_z = self.encoder(flat_x, lengths=None)                    # [N_real, 1, D]
        D = flat_z.tensor.shape[-1]
        z_real = replace(flat_z, tensor=flat_z.tensor.view(-1, D))     # [N_real, 1, D] => [N_real, D]

        # 3) Go back to [B, P, D] for the big transformer, reinserting fake patches
        z_full_tensor = z_real.tensor.new_zeros(B, P, D)               # [B, P, D], zeros
        z_full_tensor.view(B * P, D)[keep_idx] = z_real.tensor         # Fill the zeros with data where there are real patches
        # In the TensorDC for the transformer, reuse the original token-level padding_mask
        z_full = replace(z_real, tensor=z_full_tensor, padding_mask=padding_mask)  # [B, P, D]

        # 4) Big transformer (fixed-length [B, P, D], eventually unpadded to save compute)
        z_t = self.projection_in(z_full)                               # [B, P, hidden_in]

        # ---- elegant patch-level unpad: build a patch-unpadded tensor using patch_mask ----
        # We create a temporary UnpaddedTensor that represents PACKED real patches per sequence,
        # where each row is one patch embedding (we ignore token-level padding inside real patches).
        # This way the transformer will receive packed rows = N_real and cu_seqlens tracking counts PER SEQUENCE.
        device = z_t.tensor.device
        H = z_t.tensor.shape[-1]  # hidden dim

        # patch_keep: True for real (keep) patches, False for fake patches
        patch_keep = (~patch_mask)  # [B, P] bool, True = keep

        # indices: flat indices into view(B*P, H) selecting rows to pack (order: seq0 patches, seq1 patches, ...)
        indices = torch.nonzero(patch_keep.view(B * P), as_tuple=False).flatten().to(device=device).long()  # [N_real]

        # packed: [N_real, H] in the same order that cu_seqlens will expect
        packed = z_t.tensor.view(B * P, H)[indices]  # [N_real, H]

        # keep_counts: number of kept patches per sequence (length B)
        #keep_counts = patch_keep.sum(dim=1, dtype=torch.long, device=device)  # [B]
        keep_counts = patch_keep.sum(dim=1)
        keep_counts=keep_counts.to(torch.long).to(device)
        # cu_seqlens: prefix-sum with initial zero, shape [B+1], dtype long (int64)
        cu_seqlens = torch.cat([torch.tensor([0], device=device, dtype=torch.long),
                                torch.cumsum(keep_counts, dim=0)])  # [B+1]

        # Build a temporary UnpaddedTensor describing the packed patch rows.
        # Note: we set original_seq_len = P so pad() will put rows back into B x P layout.
        z_packed_input = UnpaddedTensor(
            tensor=packed,            # [N_real, H]
            indices=indices,          # linear indices into (B*P) layout
            cu_seqlens=cu_seqlens,    # [B+1]
            max_seq_len=int(keep_counts.max().item()) if keep_counts.numel() > 0 else 0,
            original_seq_len=P,
            batch_size=B
        )

        # Now call transformer in packed/unpadded mode (your transformer should detect UnpaddedTensor and use flash-attn)
        z_packed_out = self.transformer(z_packed_input, **kwargs)  # returns UnpaddedTensor (packed rows) [N_real, H_out]

        # Pad the packed output back to full [B, P, H_out] (uses the indices/cu_seqlens inside UnpaddedTensor)
        z_t = z_packed_out.pad(seq_len=P)      # [B, P, H_dec]

        # ---- end of patch-level unpad/pad trick ----------------------------------------------------

        z_t = self.projection_out(z_t)                                 # [B, P, H_dec]
        # z_t=z_t.pad()  # already padded

        # 5) Decoder (only real patches to save compute)
        # [B, P, H_dec] => [N_real, H_dec]
        z_real_post = replace(
            z_t,
            tensor=z_t.tensor.view(B * P, -1)[keep_idx],               # [N_real, H_dec]
            padding_mask=None
        )

        z_real_post = replace(z_real_post, tensor=z_real_post.tensor.view(-1, 1, z_real_post.tensor.shape[-1]))  # [N_real, 1, H_dec]
        x_real_logits, _ = self.decoder(z_real_post, lengths=None)     # [N_real, S_out, V]
        S_out, vocab_size = x_real_logits.tensor.shape[1], x_real_logits.tensor.shape[2]

        # ---- 5) SCATTER decoder outputs back into [B, P, S_out, V] -----------------------------------
        full_logits = x_real_logits.tensor.new_zeros(B, P, S_out, vocab_size)   # [B, P, S_out, V]
        full_logits.view(B * P, S_out, vocab_size)[keep_idx] = x_real_logits.tensor

        # ---- 6) Output padding mask (broadcast to S_out) ---------------------------------------------
        # We keep the original token-level mask for real patches INSIDE S (to mask inner 259 in loss),
        # but for the output we also need the global fake-patch mask at the decoder length S_out.
        # The safest default is to broadcast the patch-level padding to S_out (True on fake patches).
        out_padding_mask = patch_mask.unsqueeze(-1).expand(B, P, S_out)  # [B, P, S_out], True = fake patch positions

        # Return logits as a TensorDC; we preserve the "global" padding via out_padding_mask.
        return replace(x_real_logits, tensor=full_logits, padding_mask=out_padding_mask)


                   
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
        
        # VECTORIZED VERSION - replaces the batch loop
        # Move entropy to CPU once for all operations
        ent_cpu = entropy.cpu()
        
        # Create padding for comparison: add infinity at the beginning of each sequence
        # This ensures the first element is never considered a breaking point
        inf_pad = torch.full((B, 1), float('inf'), device='cpu')
        padded_entropy = torch.cat([inf_pad, ent_cpu], dim=1)  # Shape: [B, seq_len+1]
        
        # Vectorized monotonicity violation detection
        # Compare each position with the previous one across all batches simultaneously
        # breaking_points[b, i] = True if entropy[b, i] > entropy[b, i-1] + smoothing
        breaking_points = ent_cpu > padded_entropy[:, :-1] + smoothing  # Shape: [B, seq_len]
        
        # Convert breaking points to cutting masks (same logic as before, but vectorized)
        cutting_masks_vectorized = breaking_points.float()
        
        # Generate group masks using cumulative sum (vectorized version of the previous logic)
        group_masks_vectorized = torch.cumsum(breaking_points.long(), dim=1)
        
        # Extract cutting points for each batch item
        # This part still requires some iteration, but only over breaking points, not all positions
        cutting_points_all_vectorized = []
        for b in range(B):
            # Find positions where breaking points occur
            break_positions = torch.where(breaking_points[b])[0]
            cutting_points = []
            
            if len(break_positions) > 0:
                start_point = 0
                for pos in break_positions:
                    cutting_points.append((start_point, pos.item()))  # Note: removed +1 here
                    start_point = pos.item()
                
                # Add the final segment from the last breaking point to the end
                if start_point < seq_len:
                    cutting_points.append((start_point, seq_len))
            else:
                # If no breaking points, the entire sequence is one chunk
                cutting_points.append((0, seq_len))
            
            cutting_points_all_vectorized.append(cutting_points)
        # Update the output tensors with vectorized results
        cutting_masks[:] = cutting_masks_vectorized
        group_masks[:] = group_masks_vectorized
        cutting_points_all = cutting_points_all_vectorized
        
        #print("------------- ENTROPY DEBUG ------------------")
        #print("----CUTTING MASK:")
        #print(cutting_masks)
        #print("----- GROUP MASKs")
        #print(group_masks)
        #print("CUTTING POINTS")
        #print(cutting_points_all)
        return cutting_points_all, cutting_masks, group_masks
    def old_monotonicity_breakpoints(self, prompt=None, entropy=None, smoothing=None):
        # Da cambiare: il taglio delle patch provenienti dal text encoder potrebbe effettuarsi qui, sfruttando i nested tensor
        # Al momento l'implementazione che prevede il ciclo sui batch nel text encoder funziona ma è esageratamente inefficiente a livello di tempi di calcolo
        if smoothing is None:
            smoothing = 0
            print("WARNING: You are running the entropy model without a smoothing set.")

        if prompt is not None:
            #start=datetime.now()
            entropy = self.compute_entropy(prompt)  # Expected shape: [B, seq_len]
            #end=datetime.now()
            #print(f"Esecuzione del modello {end-start}")
        elif entropy is None:
            raise ValueError("Either provide `prompt` or `entropy`.")
        #start=datetime.now()

        if entropy.dim() == 1:
            entropy = entropy.unsqueeze(0)  # Make it batched [1, seq_len]

        B, seq_len = entropy.shape
        cutting_masks = torch.zeros_like(entropy, device='cpu')
        group_masks = torch.zeros_like(entropy, dtype=torch.long, device='cpu')
        cutting_points_all = []
        #end=datetime.now()
        #print(f"Creazione delle robette {end-start}")  
        #start=datetime.now()    
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
        #end=datetime.now()
        #print(f"Ciclo sui batch: {end-start}")
        return cutting_points_all, cutting_masks, group_masks

    def old_cut_text(self,text,cutting_points=None,smoothing=None):
        """
        Cut a text according to cutting points
        """
        if cutting_points is None:
            cutting_points,_=self.monotonicity_breakpoints(prompt=text,smoothing=smoothing)
        text_chunks=[text[i:j] for i,j in cutting_points]
        return text_chunks
    def cut_text(self, text, cutting_points=None, smoothing=None, hard_limit=None):
            """
            Cut a text or batch of texts according to cutting points.
            
            Args:
                text: str or list of str - single text or batch of texts
                cutting_points: list of tuples or list of list of tuples - cutting points for each text
                smoothing: float - smoothing parameter if cutting_points is None
                hard_limit: int - maximum character length for chunks, splits if exceeded
                
            Returns:
                list of str chunks (if single text) or list of list of str chunks (if batch)
            """
            # Handle single text input (backwards compatibility)
            hard_limit_violations=0
            if isinstance(text, str):
                if cutting_points is None:
                    cutting_points, _, _ = self.monotonicity_breakpoints(prompt=text, smoothing=smoothing)
                # Avoid error if cutting points is empty (ex. for a single word)
                if len(cutting_points)==0:
                    cutting_points=[0,len(text)] 
                # cutting_points is a list of lists, take the first (and only) one
                elif isinstance(cutting_points[0], list):
                    cutting_points = cutting_points[0]
                    
                text_chunks = [text[i:j] for i, j in cutting_points]
                # Apply hard limit splitting
                if hard_limit is not None:
                    final_chunks = []
                    for chunk in text_chunks:
                        if len(chunk) <= hard_limit:
                            final_chunks.append(chunk)
                        else:
                            final_chunks.extend([chunk[k:k+hard_limit] for k in range(0, len(chunk), hard_limit)])
                            hard_limit_violations+=1
                    text_chunks = final_chunks
                return text_chunks
            
            # Handle batch input
            if isinstance(text, list):
                if cutting_points is None:
                    cutting_points, _, _ = self.monotonicity_breakpoints(prompt=text, smoothing=smoothing)
                
                # Vectorized batch processing - process all texts simultaneously
                all_chunks = []
                for text_item, cutting_item in zip(text, cutting_points):
                    text_chunks = [text_item[i:j] for i, j in cutting_item]
                    # Apply hard limit splitting
                    if hard_limit is not None:
                        final_chunks = []
                        for chunk in text_chunks:
                            if len(chunk) <= hard_limit:
                                final_chunks.append(chunk)
                            else:
                                final_chunks.extend([chunk[k:k+hard_limit] for k in range(0, len(chunk), hard_limit)])
                                hard_limit_violations+=1
                        text_chunks = final_chunks
                    all_chunks.append(text_chunks)
                
                return all_chunks,hard_limit_violations
            
            else:
                raise ValueError("text must be either a string or a list of strings")
