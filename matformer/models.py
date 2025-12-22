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
#from copy import deepcopy Provo a rimuovere
from transformers import AutoTokenizer
from matformer.matformer_registry import registry
from matformer.cached_stuff import CachedStuff
from copy import deepcopy
from matformer.matformer_module import MatformerModule

class PL_ModelWrapper(MatformerModule):
    def __init__(self,ModelClass,config,tokenizer,device,batch_size=None,train_config=None,inference=False):
        super().__init__()
        self.config=config
        self.train_config=train_config
        self.save_hyperparameters()  
        # Initialize cache and set registry
        self.cache = CachedStuff()
        self.cache.registry = registry
        self.model = ModelClass(config,tokenizer=tokenizer,device=device,cache=self.cache)    
        self.nested = None  
        if self.config.loss_type=='fused':
            self.cross_entropy_loss = self.cache.registry.create("loss", "cross_entropy_loss_fused", *[], **{"ignore_index":config.pad_token_id})
        else: #22785MiB
            self.cross_entropy_loss = self.cache.registry.create("loss", "cross_entropy_loss", *[], **{"ignore_index":config.pad_token_id})

        if not getattr(self, "_restored_from_ckpt", False): 
            self.model.apply(init_transformer_weights_)
            
        self.batch_size=batch_size # Utile per il learning rate scheduling
        
        # Maskerator setup
        self.maskerator=Maskerator(mask_token=self.config.mask_token_id,
                                       substitution_rate=self.config.masked_substitution_rate,
                                       pad_token_id=self.config.pad_token_id,
                                       cloze_prob=self.config.cloze_probability,
                                       random_prob=self.config.random_probability,
                                       same_prob=self.config.same_probability,
                                       vocab_size=self.config.vocab_size)


        
    def forward(self, _input,*args,**kwargs):
        if isinstance(_input,torch.Tensor):
            _input=NormalTensor(tensor=_input)
        return self.model(_input.to(self.device),*args,**kwargs)
    def on_load_checkpoint(self, checkpoint):
        self._restored_from_ckpt = True       
        
     
    def training_step(self, batch, batch_idx=None):
        input_sequence = batch['sequence'] # Arriva la sequenza già tokenizzata dal MatformerDataModule
        if batch['worker_has_finished']:
            zero_loss = sum(p.sum() for p in self.parameters()) * 0.0 #Questa roba è da riguardare attentamente!!!
            return zero_loss
        masked=True if self.config.training_objective=='masked' else False
        if self.config.training_objective == 'crazy':
            self.crazy_previous_state = not getattr(self, 'crazy_previous_state', False)
            masked = self.crazy_previous_state
            for m in self.model.modules():
                if hasattr(m, 'is_causal'):
                    m.is_causal = not masked
                if hasattr(m, 'attn_kernel') and hasattr(m.attn_kernel, 'is_causal'):
                    m.attn_kernel.is_causal = not masked

        #input_sequence=sequence
        if masked:
            # If masking rate is variable and variable rate is per document, we need to be sure that the tensor has batch dimension
            #if isinstance(sequence,UnpaddedTensor):
            #    repad=True
            #    sequence=sequence.pad()
            #else:
            #    repad=False
            #masked_tokens,cloze_mask,masking_ratio=self.maskerator(sequence.tensor)
            #input_sequence=replace(sequence,tensor=masked_tokens,cloze_mask=cloze_mask)  
            #if repad:
            #    input_sequence=input_sequence.unpad()  
            input_sequence,masking_ratio=self.maskerator(input_sequence) 
            original_sequence=batch['sequence'] 
        if self.config.loss_type=='fused':
            model_return_type = 'hidden'
            flattening_dimension = self.config.hidden_size
            loss_kwargs = {"lm_head_weight": self.model.lm_head.module.inner.weight}
            if self.config.bias:
                loss_kwargs["lm_head_bias"] = self.model.lm_head.module.inner.bias #TODO: Better way to access inner attributes of wrapped modules
        else: #Normal loss
            model_return_type='logits'
            flattening_dimension=self.config.vocab_size
            loss_kwargs={}
        
        
        ### Input al modello ###
        model_output = self(input_sequence, return_type=model_return_type) #Return type can be 'logits' or 'hidden' (required for fused loss)   
        is_unpadded = isinstance(model_output, UnpaddedTensor)

        if is_unpadded:
            model_output_flat = model_output.tensor
            targets_flat = original_sequence.tensor 
            # If already unpadded, all tokens are valid
            base_mask = torch.ones_like(targets_flat, dtype=torch.bool)
            cloze_mask_flat = input_sequence.cloze_mask if masked else None
        else:
            # (B, S, H) -> 2D (B*S, H)
            model_output_flat = model_output.tensor.view(-1, model_output.tensor.size(-1))
            # (B, S) -> 1D (B*S)
            targets_flat = sequence.tensor.view(-1)
            base_mask = (targets_flat != self.config.pad_token_id)
            cloze_mask_flat = input_sequence.cloze_mask.view(-1) if masked else None

        # 2. Setting the training objective
        if masked:
            mask = cloze_mask_flat & base_mask
            inputs = model_output_flat[mask]
            targets = targets_flat[mask]
        else: # Autoregressive
            mask = base_mask[1:]
            inputs = model_output_flat[:-1][mask]
            targets = targets_flat[1:][mask]
        # 4. Getting the loss
        loss = self.cross_entropy_loss(inputs, targets, **loss_kwargs)  
        self.log('train/loss', loss, prog_bar=True,batch_size=self.batch_size)      
        if 'aux_losses' in self.cache.storage:
                    aux_losses = self.cache.storage['aux_losses']
                    if aux_losses:
                        total_aux_loss = torch.stack(aux_losses).sum()
                        aux_weight = 0.1 
                        loss += aux_weight * total_aux_loss
                        self.log("train/aux_memory_loss", total_aux_loss.item(), on_step=True)
                        self.log("train/total_loss",loss,batch_size=self.batch_size)
                    self.cache.storage['aux_losses'] = []        
        

        
        if self.config.training_objective == 'crazy':
            self.log(f'train/loss_masked_{str(masked)}',loss, prog_bar=True,batch_size=self.batch_size)
        try:
            current_lr = self.lr_schedulers().get_last_lr()[0]
            self.log("lr", current_lr, prog_bar=True, on_step=True, on_epoch=False,batch_size=self.batch_size)
        except:
            pass                 
        if masked: #Logging also the accuracy
            preds = model_output_flat[mask].argmax(dim=-1)
            targets = targets_flat[mask]
            acc = (preds == targets).float().mean()
            self.log("train/accuracy", acc, prog_bar=True, on_step=True, on_epoch=True,batch_size=self.batch_size)
            #self.log("train/masking_rate",masking_ratio,prog_bar=False,on_step=True,on_epoch=False,batch_size=self.batch_size)
        """ TODO Currently disabled
        if self.nested:
            logits_flat = torch.cat(logits.unbind())
            targets_flat = torch.cat(sequence.unbind()).to(logits_flat.device)
            base_mask = torch.ones_like(targets_flat, dtype=torch.bool, device=targets_flat.device)
            cloze_mask_flat = torch.cat(cloze_mask.unbind()).to(logits_flat.device) if masked else None
        """
        if len(self.cache.additional_logs.items())>0:
            #Additional logs
            for k in self.cache.additional_logs.keys():
                self.log(k,self.cache.additional_logs[k],on_step=True,batch_size=self.batch_size)
        
        #Trying to simplify the logic,now it by defaults pad everything again.It could be less efficient,we need some tiny test and benchmark
        #1) See if the loss makes sense directly with unpadding
        #2) See if repadding causes a significant performances loss
        """
        elif logits.isPadded:
            vocab_size = logits.shape[-1]
            logits_flat = logits.tensor.reshape(-1, vocab_size)
            targets_flat = sequence.tensor.reshape(-1)
            #print(f"Logits flat shape: {logits_flat.shape}")
            #print(f"Targets flat shape: {targets_flat.shape}")
            base_mask = (targets_flat != self.config.pad_token_id)
        
        
            autoencoders_experimental=False
            if autoencoders_experimental:
                base_mask = (targets_flat.to(logits.tensor.device) != 259)   
                fake_mask = ~logits.padding_mask.reshape(-1).to(logits.tensor.device )
                base_mask = base_mask & fake_mask  
                     
            
            cloze_mask_flat = cloze_mask.reshape(-1) if masked else None
        else:
            logits_flat = logits.tensor
            targets_flat = sequence.tensor
            base_mask = torch.ones_like(targets_flat, dtype=torch.bool, device=targets_flat.device)
            cloze_mask_flat = cloze_mask if masked else None

        if masked:
            mask = cloze_mask_flat & base_mask
            loss = self.cross_entropy_loss(logits_flat[mask], targets_flat[mask])
        else:
            mask = base_mask[1:]
            loss = self.cross_entropy_loss(logits_flat[:-1][mask], targets_flat[1:][mask])
        """          

         

        # TODO: this part has to be revised and cleaned
        additional_metrics=True
        if additional_metrics:

            if self.global_step % 100 == 0:
               grad_norms, param_norms, grad_param_ratios = {}, {}, {}
               all_grads = []
               for name, p in self.named_parameters():
                   if 'gate' in name and p.numel() == 1:
                        self.log(f"gates/{name}_value", p.item(), on_step=True, batch_size=self.batch_size)
                        self.log(f"gates/{name}_opening", torch.tanh(p).item(), on_step=True, batch_size=self.batch_size)
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
        if self.train_config["optimizer"] == "muonclip":
            from muon import MuonClip, MuonConfig
            base_lr = self.train_config["lr"]
            
            from types import SimpleNamespace

            model_config = SimpleNamespace(
                num_key_value_heads=self.config.num_attention_heads,
                num_attention_heads=self.config.num_attention_heads,
                head_dim=self.config.hidden_size // self.config.num_attention_heads
            )
            
            muon_lr = base_lr * 0.2 * math.sqrt(max(
                p.shape[0] for p in self.parameters() if p.ndim >= 2 and p.requires_grad
            ))
            
            muon_config = MuonConfig(
                lr=base_lr, ###To check!!!
                muon_beta=self.train_config.get("muon_momentum", 0.95),
                muon_decay=self.train_config.get("weight_decay", 0.01),
                ns_steps=self.train_config.get("muon_ns_steps", 5),
                adam_betas=self.train_config.get("betas", (0.9, 0.95)),
                adam_decay=self.train_config.get("weight_decay", 0.01),
                adam_eps=self.train_config.get("eps", 1e-10),
                enable_clipping=True,
                clipping_layers_mapping={"q_proj": "packed_proj", "k_proj": "packed_proj"},
                clipping_threshold=self.train_config.get("clip_threshold", 50.0),
                clipping_alpha=self.train_config.get("clip_alpha", 0.5),
                log_max_logits=False,
                cans_ortho=False,
                estimate_lower_bound=False
            )
            
            optimizer = MuonClip(self, model_config, muon_config)
            
        elif self.train_config["optimizer"] == "muonflash":
            muon_params = []
            adamw_params = []
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                if "conv" in name:
                    adamw_params.append(param)
                    print(f"{name} (Convolutional) in AdamW (ndim={param.ndim})")                          
                elif "lm_head" in name or "embed_tokens" in name or param.ndim < 2:
                    adamw_params.append(param)
                    print(f"{name} in AdamW (ndim={param.ndim})")
                else:
                    muon_params.append(param)
                    print(f"{name} in Muon")
            
            base_lr = self.train_config["lr"]
            
            from flash_muon import Muon as FlashMuon
            muon_lr = base_lr * 0.2 * math.sqrt(max(muon_params[0].shape[:2])) if muon_params else base_lr
            optimizer = [
                FlashMuon(muon_params, lr=muon_lr, 
                         momentum=self.train_config.get("muon_momentum", 0.95), rank=0, world_size=1),
                torch.optim.AdamW(adamw_params, lr=base_lr, 
                                betas=self.train_config.get("betas", (0.9, 0.95)),
                                weight_decay=self.train_config.get("weight_decay", 0.01))
            ]
            
        elif self.train_config["optimizer"] == "muon":
            muon_params = []
            adamw_params = []
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                if "conv" in name:
                    adamw_params.append(param)
                    print(f"{name} (Convolutional) in AdamW (ndim={param.ndim})")                          
                elif "lm_head" in name or "embed_tokens" in name or param.ndim < 2:
                    adamw_params.append(param)
                    print(f"{name} in AdamW (ndim={param.ndim})")
                else:
                    muon_params.append(param)
                    print(f"{name} in Muon")
            
            base_lr = self.train_config["lr"]
            
            optimizer = Muon(
                lr=base_lr,
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


        total_steps = self.train_config.get("total_steps") 
        
        self.total_training_steps = total_steps
        
        def create_scheduler(opt):
            if self.train_config.get("scheduler") == "custom":
                warmup = int(self.train_config.get("warmup_steps", 0.05) * total_steps) 
                hold = int(self.train_config.get("hold_steps", 0.10) * total_steps)
                target = self.train_config.get("final_lr", 0.0)
                base_lr = opt.param_groups[0]["lr"]
                factor = target / base_lr if base_lr > 0 else 0.0

                def lr_schedule(step):
                    if step < warmup:
                        return step / max(1, warmup)
                    if step < warmup + hold:
                        return 1.0
                    prog = (step - warmup - hold) / max(1, total_steps - warmup - hold)
                    return factor + (1 - factor) * 0.5 * (1 + math.cos(math.pi * prog))

                return torch.optim.lr_scheduler.LambdaLR(opt, lr_schedule, last_epoch=-1)

            elif self.train_config.get("scheduler") == "cosine_decay":
                from transformers import get_cosine_schedule_with_warmup
                warmup = int(self.train_config.get("warmup_steps", 0.05) * total_steps)
                return get_cosine_schedule_with_warmup(
                    optimizer=opt,
                    num_warmup_steps=warmup,
                    num_training_steps=total_steps
                )

            elif self.train_config.get("scheduler") == "linear_decay":
                from transformers import get_linear_schedule_with_warmup
                warmup = int(self.train_config.get("warmup_steps", 0.05) * total_steps)
                return get_linear_schedule_with_warmup(
                    optimizer=opt,
                    num_warmup_steps=warmup,
                    num_training_steps=total_steps
                )

            else:
                warmup = int(self.train_config.get("warmup_steps", 0.05) * total_steps)
                return get_scheduler(
                    name=self.train_config.get("scheduler", "linear"),
                    optimizer=opt,
                    num_warmup_steps=warmup,
                    num_training_steps=total_steps
                )
        
        if isinstance(optimizer, list):
            scheduler = [create_scheduler(opt) for opt in optimizer]
        else:
            scheduler = create_scheduler(optimizer)

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
    def load_from_checkpoint(checkpoint_path, ModelClass, config=None, map_location=None, tokenizer=None, overrides=None,varlen_strategy='padding'):
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

        if config is None:
            if 'hyper_parameters' in checkpoint and 'config' in checkpoint['hyper_parameters']:
                config = checkpoint['hyper_parameters']['config']
            else:
                raise ValueError("Config not found in checkpoint and not provided. Please provide a config.")  
                
        """  
        tokenizer = (
            AutoTokenizer.from_pretrained(tokenizer) 
            if tokenizer != 'bytes' else 'bytes'
        )
        """
        if overrides is not None:
            for k,v in overrides.items():
                setattr(config,k,v)
        tokenizer = MatformerTokenizer(
            config=config,
            tokenizer_type='huggingface',
            tokenizer_name=tokenizer,
            varlen_strategy=varlen_strategy
        )     

        model = PL_ModelWrapper(ModelClass=ModelClass, config=config, tokenizer=tokenizer, device=map_location, train_config=None)  
        #model.load_state_dict(checkpoint['state_dict'])
        model.load_stable_state_dict(checkpoint['state_dict'], strict=False)
        print("Found this config:")
        print(config)
        return model,config   
        

