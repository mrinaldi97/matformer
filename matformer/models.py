import argparse
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tqdm import tqdm
from matformer.tokenizers import ByteLevelTokenizer,MatformerTokenizer
from matformer.metrics import BitsPerByte  
from matformer.training_functions import MatformerDataModule,Muon
from matformer.model_config import ModelConfig  
from matformer.masked_models import maskerator
from matformer.initialization import init_transformer_weights_
#import matformer.transformer_blocks
from matformer.tensors_dataclasses import PaddedTensor,UnpaddedTensor,NormalTensor
from matformer.debug_methods import train_debug_print
from torch.optim import AdamW
from transformers import get_scheduler
import math
import torch.distributed as dist

try:
    from transformers import AutoTokenizer
except:
    print("Hugginface's tokenizer is not available on this system. Please pip install transformers")
from dataclasses import replace
from copy import deepcopy


class PL_ModelWrapper(pl.LightningModule):
    def __init__(self,ModelClass,config,tokenizer,device,batch_size=None,train_config=None,inference_fix=False,nested=False):
        super().__init__()
        self.config=config
        self.train_config=train_config
        self.nested=nested # Poco elegante, temporaneo ma in genere rivedere l'implementazione Nested, per ora WIP
        self.save_hyperparameters()  
        self.inference_fix=inference_fix #Temporaneo
        self.model = ModelClass(config,tokenizer=tokenizer) 
        self.model.apply(init_transformer_weights_)
        self.tokenizer=tokenizer #Un MatformerTokenizer
        self.batch_size=batch_size # Utile per il learning rate scheduling
    def forward(self, _input):
        return self.model(_input.to(self.device),inference_fix=self.inference_fix)
    def training_step(self, batch, batch_idx):
        if self.nested:
            #Not implemented yet! Wrong code below (won't work with tok.zers different than Bytes)
            sequence = self.tokenizer.batch_encode(batch['text'], nested=True)
        else:
            sequence = batch['input_ids'] # Arriva la sequenza già tokenizzata dal MatformerDataModule
        masked=True if self.config['training_objective']=='masked' else False

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

        ### Input al modello ###
        model_input=deepcopy(input_sequence)
        logits = self(deepcopy(model_input))

        if self.nested:
            logits_flat = torch.cat(logits.unbind())
            targets_flat = torch.cat(sequence.unbind()).to(logits_flat.device)
            base_mask = torch.ones_like(targets_flat, dtype=torch.bool, device=targets_flat.device)
            cloze_mask_flat = torch.cat(cloze_mask.unbind()).to(logits_flat.device) if masked else None
        elif logits.isPadded:
            _,_, vocab_size = logits.shape
            logits_flat = logits.tensor.reshape(-1, vocab_size)
            targets_flat = sequence.tensor.reshape(-1)
            base_mask = (targets_flat != self.config.pad_token_id)
            cloze_mask_flat = cloze_mask.reshape(-1) if masked else None
        else:
            logits_flat = logits.tensor
            targets_flat = sequence.tensor
            base_mask = torch.ones_like(targets_flat, dtype=torch.bool, device=targets_flat.device)
            cloze_mask_flat = cloze_mask if masked else None

        if masked:
            mask = cloze_mask_flat & base_mask
            #train_debug_print(_input=model_input.tensor, output=targets_flat[mask], model_cfg=self.config, tokenizer=self.tokenizer, varlen_strategy='unpadding')            
            loss = F.cross_entropy(logits_flat[mask], targets_flat[mask])

        else:
            mask = base_mask[1:]
            #train_debug_print(_input=model_input.tensor, output=targets_flat[1:][mask], model_cfg=self.config, tokenizer=self.tokenizer, varlen_strategy='unpadding')            
            loss = F.cross_entropy(logits_flat[:-1][mask], targets_flat[1:][mask])
        if masked: #Logging also the accuracy
            preds = logits_flat[mask].argmax(dim=-1)
            targets = targets_flat[mask]
            acc = (preds == targets).float().mean()
            self.log("train/accuracy", acc, prog_bar=True, on_step=True, on_epoch=True,batch_size=self.batch_size)
         
        current_lr = self.lr_schedulers().get_last_lr()[0]
        self.log("lr", current_lr, prog_bar=True, on_step=True, on_epoch=False,batch_size=self.batch_size)
        self.log('train/loss', loss, prog_bar=True,batch_size=self.batch_size)
        return loss
    def configure_optimizers(self):
        use_muon = self.train_config["optimizer"].lower() == "muon"
        if use_muon:
            muon_params = []
            adamw_params = []
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                if (
                    "lm_head" in name
                    or "embed_tokens" in name
                    or param.ndim < 2
                ):
                    adamw_params.append(param)
                    print(f"{name} in AdamW (ndim={param.ndim})")
                else:
                    muon_params.append(param)
                    print(f"{name} in Muon")

            optimizer = Muon(
                lr=self.train_config["lr"],
                wd=self.train_config.get("weight_decay", 0.01),
                muon_params=muon_params,
                momentum=self.train_config.get("muon_momentum", 0.95),
                nesterov=self.train_config.get("muon_nesterov", True),
                ns_steps=self.train_config.get("muon_ns_steps", 5),
                adamw_params=adamw_params,
                adamw_betas=self.train_config.get("betas", (0.9, 0.95)),
                adamw_eps=self.train_config.get("eps", 1e-10),
            )
        else:
            optimizer = AdamW(
                self.parameters(),
                lr=self.train_config["lr"],
                weight_decay=self.train_config.get("weight_decay", 0.01),
            )

        # === Scheduler ===
        if not self.train_config.get("lr_scheduling", False):
            return optimizer

        total_steps = math.ceil(
            (self.train_config["num_batches"] // self.train_config.get("accumulate_grad_batches", 1))
            * self.train_config.get("max_epochs", 1)
        )
        self.total_training_steps = total_steps

        if self.train_config.get("scheduler") == "custom":
            warmup   = self.train_config.get("warmup_steps", int(0.05 * total_steps))
            hold     = self.train_config.get("hold_steps", int(0.10 * total_steps))
            target   = self.train_config.get("final_lr", 0.0)
            base_lr  = optimizer.param_groups[0]["lr"]
            factor   = target / base_lr if base_lr > 0 else 0.0

            def lr_schedule(step):
                if step < warmup: return step / max(1, warmup)
                if step < warmup + hold: return 1.0
                prog = (step - warmup - hold) / max(1, total_steps - warmup - hold)
                return factor + (1 - factor) * 0.5 * (1 + math.cos(math.pi * prog))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule, last_epoch=-1)
        else:
            warmup = self.train_config.get("warmup_steps", int(0.05 * total_steps))
            scheduler = get_scheduler(
                name=self.train_config.get("scheduler", "linear"),
                optimizer=optimizer,
                num_warmup_steps=warmup,
                num_training_steps=total_steps
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }


    def debug_on_after_backward(self):
        # Log dei gradienti per debug
        for name, param in self.named_parameters():
            if param.requires_grad and param.grad is None:
                self.logger.experiment.log({"broken_grad/" + name: 1})
            elif param.grad is not None:
                grad_norm = param.grad.norm().item()
                self.logger.experiment.log({f"grad/{name}": grad_norm})
                self.log(f"grad_max/{name}", param.grad.abs().max().item(), on_step=True)
                self.log(f"grad_min/{name}", param.grad.abs().min().item(), on_step=True)
    @staticmethod
    def load_from_checkpoint(checkpoint_path, ModelClass, config=None, map_location=None, inference_fix=False, tokenizer=None, varlen_strategy='padding'):
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

        if config is None:
            if 'hyper_parameters' in checkpoint and 'config' in checkpoint['hyper_parameters']:
                config = checkpoint['hyper_parameters']['config']
                print("Found this config:")
                print(config)

            else:
                raise ValueError("Config not found in checkpoint and not provided. Please provide a config.")    
        tokenizer = (
            AutoTokenizer.from_pretrained(tokenizer) 
            if tokenizer != 'bytes' else 'bytes'
        )
        tokenizer = MatformerTokenizer(
            config,
            tokenizer=tokenizer,
            varlen_strategy=varlen_strategy
        )     

        model = PL_ModelWrapper(ModelClass=ModelClass, config=config, tokenizer=tokenizer, device=map_location, inference_fix=inference_fix, train_config=None)  
        model.load_state_dict(checkpoint['state_dict'])
        return model,config   
        
        
             
###### From here: OLD CODE #######
"""
class OLD_PL_TransformerWithLMHead(pl.LightningModule):
    def __init__(self, config, tokenizer, device, train_config=None,inference_fix=False,nested=False):
        super().__init__()
        self.config=config    
        self.train_config=train_config  
        self.nested=nested # Poco elegante, temporaneo ma in genere rivedere l'implementazione Nested, per ora WIP
        self.save_hyperparameters()  
        self.inference_fix=inference_fix #Temporaneo
        self.model = TransformerWithLMHead(config) 
        self.tokenizer=tokenizer

    def forward(self, _input):
        return self.model(_input.to(self.device),inference_fix=self.inference_fix)
       
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
    def generate(self, prompt=None, max_length=100, temperature=1.0, top_k=0, top_p=0.9):
"""
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
    def training_step(self, batch, batch_idx):
        if self.nested:
            #Not implemented yet! Wrong code below (won't work with tok.zers different than Bytes)
            sequence = self.tokenizer.batch_encode(batch['text'], nested=True)
        else:
            sequence = batch['input_ids'] # Arriva la sequenza già tokenizzata dal MatformerDataModule

        masked=True if self.config['training_objective']=='masked' else False

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

        ### Input al modello ###
        model_input=deepcopy(input_sequence)
        logits = self(deepcopy(model_input))

        if self.nested:
            logits_flat = torch.cat(logits.unbind())
            targets_flat = torch.cat(sequence.unbind()).to(logits_flat.device)
            base_mask = torch.ones_like(targets_flat, dtype=torch.bool, device=targets_flat.device)
            cloze_mask_flat = torch.cat(cloze_mask.unbind()).to(logits_flat.device) if masked else None
        elif logits.isPadded:
            _,_, vocab_size = logits.shape
            logits_flat = logits.tensor.reshape(-1, vocab_size)
            targets_flat = sequence.tensor.reshape(-1)
            base_mask = (targets_flat != self.config.pad_token_id)
            cloze_mask_flat = cloze_mask.reshape(-1) if masked else None
        else:
            logits_flat = logits.tensor
            targets_flat = sequence.tensor
            base_mask = torch.ones_like(targets_flat, dtype=torch.bool, device=targets_flat.device)
            cloze_mask_flat = cloze_mask if masked else None

        if masked:
            mask = cloze_mask_flat & base_mask
            #train_debug_print(_input=model_input.tensor, output=targets_flat[mask], model_cfg=self.config, tokenizer=self.tokenizer, varlen_strategy='unpadding')            
            loss = F.cross_entropy(logits_flat[mask], targets_flat[mask])

        else:
            mask = base_mask[1:]
            #train_debug_print(_input=model_input.tensor, output=targets_flat[1:][mask], model_cfg=self.config, tokenizer=self.tokenizer, varlen_strategy='unpadding')            
            loss = F.cross_entropy(logits_flat[:-1][mask], targets_flat[1:][mask])

        self.log('train/loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        self.train_config["optimizer"]=='muon':
            use_muon = True
        else:
            use_muon = False

        if use_muon:
            from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
            import torch.distributed as dist

            hidden_weights = []
            adamw_params = []

            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                if (
                    'lm_head' in name
                    or 'embed_tokens' in name
                    or param.ndim < 2
                ):
                    adamw_params.append(param)
                    print(f"{name} in Adam (ndim= {param.ndim})")
                else:
                    hidden_weights.append(param)
                    print(f"{name} in Muon")

            param_groups = [
                dict(
                    params=hidden_weights,
                    use_muon=True,
                    lr=self.train_config["muon_lr"],
                    weight_decay=0.01,
                    momentum=0.95
                ),
                dict(
                    params=adamw_params,
                    use_muon=False,
                    lr=self.train_config["lr"],
                    betas=(0.9, 0.95),
                    eps=1e-10,
                    weight_decay=0.01,
                ),
            ]
            if dist.is_available() and dist.is_initialized():
                optimizer = MuonWithAuxAdam(param_groups)
            else:
                optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.train_config["lr"])

        return optimizer


    @staticmethod
    def load_from_checkpoint(checkpoint_path, config=None, map_location=None, inference_fix=False, tokenizer=None, varlen_strategy='padding'):
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

        if config is None:
            if 'hyper_parameters' in checkpoint and 'config' in checkpoint['hyper_parameters']:
                config = checkpoint['hyper_parameters']['config']
                print("Found this config:")
                print(config)

            else:
                raise ValueError("Config not found in checkpoint and not provided. Please provide a config.")    
        tokenizer = (
            AutoTokenizer.from_pretrained(tokenizer) 
            if tokenizer != 'bytes' else 'bytes'
        )
        tokenizer = MatformerTokenizer(
            config,
            tokenizer=tokenizer,
            varlen_strategy=varlen_strategy
        )                
        model = PL_TransformerWithLMHead(config, device=map_location, inference_fix=inference_fix, train_config=None, tokenizer=tokenizer)  
        model.load_state_dict(checkpoint['state_dict'])
        return model,config

class old_Autoregressive_Model(PL_TransformerWithLMHead):
    def generate(self, prompt=None, max_length=100, temperature=1.0, top_k=0, top_p=0.9):
"""
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
"""
        self.eval()  

        if prompt is None:
            current_ids = torch.tensor([[self.config.bos_token_id]], device=self.device)
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
            if next_token.item() == self.config.eos_token_id:
                break

        return current_ids

class old_EntropyModel(Autoregressive_Model):
    def compute_entropy(self, prompts):
"""
        #Return a tensor of size sequence length containing the entropy for each received text
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
        #Cut a text according to cutting points
"""
        if cutting_points is None:
            cutting_points,_=self.monotonicity_breakpoints(prompt=text,smoothing=smoothing)
        text_chunks=[text[i:j] for i,j in cutting_points]
        return text_chunks

class old_BERTModel(PL_TransformerWithLMHead):
    def inference_testing(self, input_text, masking_ratio=0.25):
        sequence = self.tokenizer.encode(input_text)
        sequence = sequence.unsqueeze(0).to(self.device)
        masked_list, cloze_list = maskerator(sequence, MASK_TOKEN=0, substitution_rate=masking_ratio)
        masked_sequence = replace(PaddedTensor(tensor=sequence), tensor=masked_list)
        with torch.no_grad():
            logits = self(masked_sequence)
        predictions = torch.argmax(logits.tensor, dim=-1)
        targets = sequence.squeeze()
        mask = cloze_list.squeeze().bool()
        correct = (predictions.squeeze()[mask] == targets[mask]).sum().item()
        total = mask.sum().item()
        accuracy = correct / total if total > 0 else 0.0
        predicted_tokens = [self.tokenizer.single_decode(tok.item()) for tok in predictions.squeeze()[mask]]
        return accuracy, predicted_tokens
"""
