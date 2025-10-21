import argparse
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
from matformer.matformer_tokenizers import ByteLevelTokenizer,MatformerTokenizer
from matformer.extra.muon import Muon
from matformer.model_config import ModelConfig  
from matformer.masked_models import Maskerator
from matformer.initialization import init_transformer_weights_
#import matformer.transformer_blocks
from matformer.tensors_dataclasses import PaddedTensor,UnpaddedTensor,NormalTensor
#from matformer.debug_methods import train_debug_print
from torch.optim import AdamW, Adam
from transformers import get_scheduler
import math
import torch.distributed as dist
import numpy as np
from dataclasses import replace
from copy import deepcopy
from transformers import AutoTokenizer


class PL_ModelWrapper(pl.LightningModule):
    def __init__(self,ModelClass,config,tokenizer,device,batch_size=None,train_config=None,inference=False):
        super().__init__()
        self.config=config
        self.train_config=train_config
        self.save_hyperparameters()  
        self.model = ModelClass(config,tokenizer=tokenizer,device=device)    
        self.nested = None    
        if not getattr(self, "_restored_from_ckpt", False): 
            self.model.apply(init_transformer_weights_)
            
        self.batch_size=batch_size # Utile per il learning rate scheduling
        self.maskerator=Maskerator(mask_token=self.config.mask_token_id,substitution_rate=self.config.masked_substitution_rate)
    def forward(self, _input):
        return self.model(_input.to(self.device))
    def on_load_checkpoint(self, checkpoint):
        self._restored_from_ckpt = True        
    def training_step(self, batch, batch_idx):
        sequence = batch # Arriva la sequenza già tokenizzata dal MatformerDataModule
        masked=True if self.config.training_objective=='masked' else False
        input_sequence=sequence
        if masked:
            masked_tokens,cloze_mask=self.maskerator(sequence.tensor)
            masked_sequence=deepcopy(sequence)
            masked_sequence=replace(masked_sequence,tensor=masked_tokens)
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
            vocab_size = logits.shape[-1]
            logits_flat = logits.tensor.reshape(-1, vocab_size)
            targets_flat = sequence.tensor.reshape(-1)
            #print(f"Logits flat shape: {logits_flat.shape}")
            #print(f"Targets flat shape: {targets_flat.shape}")
            base_mask = (targets_flat != self.config.pad_token_id)
            """
            autoencoders_experimental=False
            if autoencoders_experimental:
                base_mask = (targets_flat.to(logits.tensor.device) != 259)   
                fake_mask = ~logits.padding_mask.reshape(-1).to(logits.tensor.device )
                base_mask = base_mask & fake_mask               
            """
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
            # So this was wrong? Jerik controlla se puoi:
            loss = F.cross_entropy(logits_flat[:-1][mask], targets_flat[1:][mask])
            #loss=F.cross_entropy(logits_flat[:-1][mask[:-1]], targets_flat[1:][mask[:-1]])

            
        if masked: #Logging also the accuracy
            preds = logits_flat[mask].argmax(dim=-1)
            targets = targets_flat[mask]
            acc = (preds == targets).float().mean()
            self.log("train/accuracy", acc, prog_bar=True, on_step=True, on_epoch=True,batch_size=self.batch_size)
         
        current_lr = self.lr_schedulers().get_last_lr()[0]
        try:
            self.log("lr", current_lr, prog_bar=True, on_step=True, on_epoch=False,batch_size=self.batch_size)
            self.log('train/loss', loss, prog_bar=True,batch_size=self.batch_size)
        except:
            pass
        additional_metrics=True
        if additional_metrics:
            if self.global_step % 100 == 0:
               grad_norms, param_norms, grad_param_ratios = {}, {}, {}
               all_grads = []
               for name, p in self.named_parameters():
                   if p.grad is not None:
                       grad_norm = p.grad.detach().norm(2).item()
                       param_norm = p.detach().norm(2).item()
                       
                       grad_norms[f"grad_norm/{name}"] = grad_norm
                       param_norms[f"param_norm/{name}"] = param_norm
                       
                       if param_norm > 1e-8:
                           grad_param_ratios[f"grad_param_ratio/{name}"] = grad_norm / param_norm
                       
                       all_grads.append(p.grad.detach().flatten())
                       
                       if hasattr(self, '_prev_params') and name in self._prev_params:
                           update_norm = (p.detach() - self._prev_params[name]).norm(2).item()
                           self.log(f"param_update/{name}", update_norm, on_step=True, batch_size=self.batch_size)
               
               if all_grads:
                   total_grad_norm = torch.norm(torch.cat(all_grads)).item()
                   self.log("diagnostics/total_grad_norm", total_grad_norm, on_step=True, batch_size=self.batch_size)
               
               for metrics in [grad_norms, param_norms, grad_param_ratios]:
                   for k, v in metrics.items():
                       self.log(k, v, on_step=True, on_epoch=False, batch_size=self.batch_size)
               
               self._prev_params = {name: p.detach().clone() for name, p in self.named_parameters()}
               
               max_param = max(param_norms.values()) if param_norms else 0
               min_param = min(param_norms.values()) if param_norms else 0
               self.log("diagnostics/param_norm_spread", max_param - min_param, on_step=True, batch_size=self.batch_size)            
        return loss
        
    def on_before_optimizer_step(self, optimizer):
        # This runs after gradient clipping but before optimizer step
        if self.global_step % 100 == 0:
            clipped_grads = []
            for p in self.parameters():
                if p.grad is not None:
                    clipped_grads.append(p.grad.detach().flatten())
            if clipped_grads:
                post_clip_norm = torch.norm(torch.cat(clipped_grads)).item()
                self.log("diagnostics/post_clip_grad_norm", post_clip_norm, on_step=True, batch_size=self.batch_size)
                
    def configure_optimizers(self):
        #use_muon = self.train_config["optimizer"].lower() == "muon"
        
        if self.train_config["optimizer"] == "muon":
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
            
        elif self.train_config["optimizer"] == "adamw":
            optimizer = AdamW(
                self.parameters(),
                lr=self.train_config["lr"],
                weight_decay=self.train_config.get("weight_decay", 0.01),
            )
        elif self.train_config["optimizer"] == "adam":
            optimizer = Adam(
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
            warmup = self.train_config.get("warmup_steps", int(0.05 * total_steps))
            hold = self.train_config.get("hold_steps", int(0.10 * total_steps))
            target = self.train_config.get("final_lr", 0.0)
            base_lr = optimizer.param_groups[0]["lr"]
            factor = target / base_lr if base_lr > 0 else 0.0

            def lr_schedule(step):
                if step < warmup:
                    return step / max(1, warmup)
                if step < warmup + hold:
                    return 1.0
                prog = (step - warmup - hold) / max(1, total_steps - warmup - hold)
                return factor + (1 - factor) * 0.5 * (1 + math.cos(math.pi * prog))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule, last_epoch=-1)

        elif self.train_config.get("scheduler") == "cosine_decay":
            from transformers import get_cosine_schedule_with_warmup
            
            warmup = self.train_config.get("warmup_steps", int(0.05 * total_steps))
            final_lr = self.train_config.get("final_lr", 0.0)
            
            scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=warmup,
                num_training_steps=total_steps            )

        elif self.train_config.get("scheduler") == "linear_decay":
            from transformers import get_linear_schedule_with_warmup
            
            warmup = self.train_config.get("warmup_steps", int(0.05 * total_steps))
            
            scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=warmup,
                num_training_steps=total_steps
            )


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
    def load_from_checkpoint(checkpoint_path, ModelClass, config=None, map_location=None, tokenizer=None, varlen_strategy='padding'):
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

        model = PL_ModelWrapper(ModelClass=ModelClass, config=config, tokenizer=tokenizer, device=map_location, train_config=None)  
        model.load_state_dict(checkpoint['state_dict'])
        return model,config   
        

