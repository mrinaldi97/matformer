"""
File: matformer/transformer_blocks/models/other_models.py
"""
#Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
#Matformer
from matformer.matformer_module import MatformerModule
from matformer.tensors_dataclasses import TensorDC, NormalTensor, PaddedTensor, UnpaddedTensor, ModuleWrapper
from matformer.model_config import ModelConfig  
from matformer.utils.matformer_cache import ensure_cache_and_registry, CachedStuff
from matformer.matformer_registry import registry
from matformer.utils.masked_models import Maskerator
from matformer.matformer_tokenizers import ByteLevelTokenizer,MatformerTokenizer
from .language_models import Autoregressive_Model
from ..transformer import NakedTransformer
#Other
from functools import partial, reduce
from tqdm import tqdm
from datetime import datetime
from dataclasses import replace
from copy import deepcopy
from warnings import warn
from typing import Optional, List, Literal, Any

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
class TransformerWithCharAutoencoder(MatformerModule):
    def __init__(self, config, device=None, tokenizer=None, log=None):
        super().__init__()
        from char_autoencoder.autoencoders import TransCharAutoencoder_Encoder, TransCharAutoencoder_Decoder
        self.log=log
        self.encoder = TransCharAutoencoder_Encoder(config=config.encoder)
        self.decoder = TransCharAutoencoder_Decoder(config=config.decoder)
        self.transformer = NakedTransformer(config)
        self.projection_in = ModuleWrapper(nn.Linear(config.encoder.hidden_size, config.hidden_size))
        self.projection_out = ModuleWrapper(nn.Linear(config.hidden_size, config.decoder.hidden_size))
        self.config = config
        
        # Training phase state
        self.training_phase = 'autoencoder-training'

    def forward(self, x, skip_transformer=False, skip_decoder=False):
        """Phase 1: Train autoencoder only (bypass transformer)
        Input could be [B,S] or [B,P,S]
        If [B,S], we expect a PaddedTensor or a NormalTensor (for inference), "padding" refers to "internal" padding (per each sequence)
        If [B,P,S], we expect a PaddedTensor or an Unpadded Tensor (or NormalTensor for inference), where "padding" refers to "external" padding (per each batch). Thus, to reconstruct the internal padding, we need the internal padding mask
        
        If input is of shape B,P,S, flatten the patch with the batch => B,P,S => B*P,S but skip computation on "external" padding
        """
        """
        I expect as input a dictionary composed of
        {
        "tensor": see comment above
        "padding_masks_external": True if the entire patch is to be padded, False otherwise [B,P,S]
        "padding_masks_internal": Contains information about padding of each sequence inside patches (ready to be flattened) [B,P,S]
        }
        """
        tensor = x["tensor"]  # [B,P,S]
        padding_masks_external = x["padding_masks_external"]  # [B,P,S]
        padding_masks_internal = x["padding_masks_internal"]  # [B,P,S]
        
        # Step 1: Encoder - Filter valid patches
        valid_patch_mask = ~padding_masks_external.all(dim=-1)  # [B,P]
        valid_tensor = tensor[valid_patch_mask]  # [B*P_valid, S]
        valid_internal_masks = padding_masks_internal[valid_patch_mask]  # [B*P_valid, S]
        
        padded_tensor = PaddedTensor(
            tensor=valid_tensor,
            padding_mask=valid_internal_masks
        )
        z = self.encoder(padded_tensor)  # [B*P_valid, S] -> [B*P_valid, D]
        
        if not skip_transformer:
            # Step 2: Reshape for transformer
            B, P = valid_patch_mask.shape
            D = z.tensor.size(-1)
            z_full = torch.zeros(B, D, device=z.device, dtype=z.dtype)
            z_full[valid_patch_mask] = z.tensor  # [B, P]
            
            external_padding_mask = ~valid_patch_mask  # [B, P]
            z_padded = PaddedTensor(tensor=z_full, padding_mask=external_padding_mask)
            
            # Step 3: Transformer
            z = self.projection_in(z_padded)
            z = self.transformer(z)  # [B, P, D]
            z = self.projection_out(z)
            
            # Step 4: Back to decoder format
            z_valid = z.tensor[valid_patch_mask]  # [B*P_valid, D]
            z_valid_wrapped = replace(z, tensor=z_valid)
        else:
            # Skip transformer - encoder directly connected to decoder
            z_valid_wrapped = z
        
        # Early return for encoder-only phase
        if skip_decoder:
            return z_valid_wrapped  # [B*P_valid, D]
        
        # Step 5: Decoder
        logits, _ = self.decoder(z_valid_wrapped)  # [B*P_valid, D] -> [B*P_valid, S, vocab_size]
        
        # Step 6: Restore final output
        B, P = valid_patch_mask.shape
        S = tensor.size(-1)
        vocab_size = logits.tensor.size(-1)
        output_logits = torch.zeros(B, P, S, vocab_size, device=tensor.device, dtype=logits.tensor.dtype)
        output_logits[valid_patch_mask] = logits.tensor
        
        return output_logits

    def set_training_phase(self, phase: str):
        """Set training phase and configure module gradients"""
        self.training_phase = phase
        
        phase_configs = {
            'autoencoder-training': {'encoder': True, 'transformer': False, 'decoder': True},
            'patch-training': {'encoder': True, 'transformer': True, 'decoder': False},
            'autoencoder-final-annealing': {'encoder': False, 'transformer': True, 'decoder': True}
        }
        
        if phase not in phase_configs:
            raise ValueError(f"Unknown phase: {phase}")
        
        config = phase_configs[phase]
        self.encoder.requires_grad_(config['encoder'])
        self.transformer.requires_grad_(config['transformer'])
        self.decoder.requires_grad_(config['decoder'])

    def training_step(self, batch, phase=None, log=None):
        """Simplified training step dispatcher"""
        phase = phase or self.training_phase
        
        if phase == 'autoencoder-training':
            return self._autoencoder_loss(batch, log=log)
        elif phase == 'patch-training':
            return self._patch_loss(batch)
        elif phase == 'autoencoder-final-annealing':
            return self._fine_tuning_loss(batch)
        else:
            raise ValueError(f"Unknown training phase: {phase}")

    def _autoencoder_loss(self, batch, log):
        """Phase 1: Autoencoder reconstruction loss"""
        x = batch['input_ids'] 
        
        # Forward with autoencoder only (skip transformer)
        logits = self.forward(x, skip_transformer=True)  # [B, P, S, vocab_size]
        
        # Get target tensor
        targets = x["tensor"]  # [B, P, S]
        valid_mask = ~x["padding_masks_external"].all(dim=-1)  # [B, P]
        
        # Compute loss only on valid patches
        loss = 0
        count = 0
        for b in range(targets.size(0)):
            for p in range(targets.size(1)):
                if valid_mask[b, p]:
                    patch_logits = logits[b, p]  # [S, vocab_size]
                    patch_targets = targets[b, p]  # [S]
                    
                    # Mask internal padding
                    internal_mask = ~x["padding_masks_internal"][b, p]
                    if internal_mask.any():
                        valid_logits = patch_logits[internal_mask]
                        valid_targets = patch_targets[internal_mask]
                        loss += F.cross_entropy(valid_logits, valid_targets)
                        count += 1
        loss=loss / max(count, 1)
        log('train/autoencoder_loss', loss, prog_bar=True)
        return loss

    def _patch_loss(self, batch):
        """Phase 2: Patch-level autoregressive prediction"""
        x = batch['input_ids']
        
        # Get patch embeddings (skip decoder)
        z = self.forward(x, skip_decoder=True)  # [B*P_valid, D]
        
        # Reshape to get batch structure back
        valid_mask = ~x["padding_masks_external"].all(dim=-1)  # [B, P]
        B, P = valid_mask.shape
        D = z.tensor.size(-1)
        
        z_full = torch.zeros(B, P, D, device=z.tensor.device, dtype=z.tensor.dtype)
        z_full[valid_mask] = z.tensor
        
        # Autoregressive loss: predict next patch from previous patches
        if P <= 1:
            return torch.tensor(0.0, device=z.tensor.device, requires_grad=True)
        
        # Input: patches 0 to P-2, Target: patches 1 to P-1
        input_patches = z_full[:, :-1]  # [B, P-1, D]
        target_patches = z_full[:, 1:]  # [B, P-1, D]
        
        # Only compute loss where both input and target are valid
        input_valid = valid_mask[:, :-1]
        target_valid = valid_mask[:, 1:]
        both_valid = input_valid & target_valid
        
        if both_valid.any():
            return F.mse_loss(input_patches[both_valid], target_patches[both_valid])
        else:
            return torch.tensor(0.0, device=z.tensor.device, requires_grad=True)

    def _fine_tuning_loss(self, batch):
        """Phase 3"""
        x = batch['input_ids']
        
        # Full forward pass
        logits = self.forward(x)  # [B, P, S, vocab_size]
        targets = x["tensor"]  # [B, P, S]
        
        # Compute loss on all valid positions
        loss = 0
        count = 0
        
        valid_patch_mask = ~x["padding_masks_external"].all(dim=-1)
        
        for b in range(targets.size(0)):
            for p in range(targets.size(1)):
                if valid_patch_mask[b, p]:
                    patch_logits = logits[b, p]  # [S, vocab_size]
                    patch_targets = targets[b, p]  # [S]
                    
                    # Use internal padding mask
                    internal_mask = ~x["padding_masks_internal"][b, p]
                    if internal_mask.any():
                        valid_logits = patch_logits[internal_mask]
                        valid_targets = patch_targets[internal_mask]
                        loss += F.cross_entropy(valid_logits, valid_targets)
                        count += 1
        
        return loss / max(count, 1)
