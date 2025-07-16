import argparse
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tqdm import tqdm
from matformer.tokenizers import ByteLevelTokenizer
from matformer.metrics import BitsPerByte  
from matformer.training_functions import MatformerDataModule
from matformer.model_config import ModelConfig  
from matformer.transformer_blocks import TransformerWithLMHead
from matformer.tensors_dataclasses import PaddedTensor,UnpaddedTensor
from matformer.masked_models import maskerator

from dataclasses import replace
from copy import deepcopy

class EntropyModel(pl.LightningModule):
    def __init__(self, config, device, train_config=None,inference_fix=False,nested=False,attn_impl='flash'):
        super().__init__()
        self.save_hyperparameters()  
        self.inference_fix=inference_fix
        config['attn_imlp']=attn_impl
        self.model = TransformerWithLMHead(config)
        self.tokenizer=ByteLevelTokenizer(config)
        self.pad_id = config.pad_id
        self.bos_id=config.bos_id
        self.eos_id=config.eos_id
        self.nested=nested
        if train_config is not None:
            self.total_steps = train_config['max_steps']
            #self.warmup_steps = train_config['warmup_steps']
            self.lr = train_config['lr']
            #self.clip_grad = train_config['clip_grad']

    def forward(self, _input):
        return self.model(_input.to(self.device),inference_fix=self.inference_fix)
        
    def training_step(self, batch, batch_idx):
        if self.nested:
            #Not implemented yet! Wrong code below (won't work with tok.zers different than Bytes)
            sequence = self.tokenizer.batch_encode(batch['text'], nested=True)
        else:
            sequence = batch['input_ids']
        masked=False
        input_sequence=sequence
        if masked:
            if self.nested:
                masked_sequences, cloze_masks = zip(*[maskerator(seq, MASK_TOKEN=0, substitution_rate=0.25) 
                                                      for seq in sequence.unbind()])
                sequence = torch.stack(masked_sequences)
                cloze_mask = torch.stack(cloze_masks)
            else:
                masked_list, cloze_list = maskerator(sequence.tensor, MASK_TOKEN=0, substitution_rate=0.25)
                masked_sequence=deepcopy(sequence)
                masked_sequence = replace(masked_sequence,tensor=masked_list)
                cloze_mask = cloze_list
                input_sequence=masked_sequence
        
        logits = self(deepcopy(input_sequence))
        
        if self.nested:
            logits_flat = torch.cat(logits.unbind())
            targets_flat = torch.cat(sequence.unbind()).to(logits_flat.device)
            base_mask = torch.ones_like(targets_flat, dtype=torch.bool, device=targets_flat.device)
            cloze_mask_flat = torch.cat(cloze_mask.unbind()).to(logits_flat.device) if masked else None
        elif logits.isPadded:
            _,_, vocab_size = logits.shape
            logits_flat = logits.tensor.reshape(-1, vocab_size)
            targets_flat = sequence.tensor.reshape(-1)
            base_mask = (targets_flat != self.pad_id)
            cloze_mask_flat = cloze_mask.reshape(-1) if masked else None
        else:
            logits_flat = logits.tensor
            targets_flat = sequence.tensor
            base_mask = torch.ones_like(targets_flat, dtype=torch.bool, device=targets_flat.device)
            cloze_mask_flat = cloze_mask if masked else None
        
        if masked:
            mask = cloze_mask_flat & base_mask
            loss = F.cross_entropy(logits_flat[mask], targets_flat[mask])
            
            """
            import json
            with open("debug_masked.jsonl","a") as f:
                data={
                "input_logits":input_sequence.tensor.tolist(),
                "targets":targets_flat.tolist(),
                "mask":mask.tolist()
                }
                f.write(f"{data}\n")
            """
        else:
            mask = base_mask[1:]
            loss = F.cross_entropy(logits_flat[:-1][mask], targets_flat[1:][mask])
        """with torch.no_grad():
            preds = logits_flat[mask].argmax(-1)
            acc = (preds == targets_flat[mask]).float().mean()
        self.log('train/mask_acc', acc, prog_bar=True)    
        """    
        self.log('train/loss', loss, prog_bar=True)
        return loss
    def configure_optimizers(self):
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
            return optimizer
    

    def compute_entropy(self, prompts):
        """
        Return a tensor of size sequence length containing the entropy for each received text
        """
        self.eval()
        if isinstance(prompts, str):
            prompts = [prompts]

        prompt_ids = self.tokenizer.batch_encode(prompts, padding=True, truncation=True)  # [B, seq_len]
        prompt_ids = prompt_ids.to(self.device)

        with torch.no_grad():
            logits = self(prompt_ids)  # [B, seq_len, vocab_size]

        epsilon = 1e-10
        probs = torch.nn.functional.softmax(logits, dim=-1)
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
            current_ids = torch.tensor([[self.model.config.bos_id]], device=self.device)
        else:
            # Tokenize the prompt if it's provided as bytes
            assert isinstance(prompt, str), "Prompt expected as string"
            tokenizer = self.tokenizer
            prompt_ids = tokenizer.encode(prompt)
            current_ids = torch.tensor(prompt_ids.unsqueeze(0), device=self.device)
            # The forward expects: [batch_size, seq_len, vocab_size]
            print(f"Prompt_ids shape: {prompt_ids.shape}")
            print(f"Current_ids shape: {current_ids.shape}")
        
        for _ in tqdm(range(max_length)):
            with torch.no_grad():
                outputs = self(current_ids)
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
            print(f"Generated: {next_token.item()} Char: {tokenizer.single_decode(next_token.item())}")
            # Append the new token to the sequence
            current_ids = torch.cat([current_ids, next_token], dim=1)
            
            # Stop if we generated an EOS token
            if next_token.item() == self.eos_id:
                break
        
        return current_ids
        
    @staticmethod
    def load_from_checkpoint(checkpoint_path, config=None, map_location=None, inference_fix=False):
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        
        if config is None:
            if 'hyper_parameters' in checkpoint and 'config' in checkpoint['hyper_parameters']:
                config = checkpoint['hyper_parameters']['config']
                print("Found this config:")
                print(config)
                
            else:
                raise ValueError("Config not found in checkpoint and not provided. Please provide a config.")    
        model = EntropyModel(config, device=map_location, inference_fix=inference_fix, train_config=None)  
        model.load_state_dict(checkpoint['state_dict'])
        return model,config
