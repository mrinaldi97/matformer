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
    model: "transparent"
    def __init__(self,ModelClass,config,tokenizer,device,batch_size=None,train_config=None,inference=False,load_mode="full", skip_init=False, training_step_type='pretraining'):
        super().__init__()
        self.config=config
        self.train_config=train_config
        self.save_hyperparameters()  
        self.load_mode = load_mode
        # Initialize cache and set registry
        self.cache = CachedStuff()
        self.cache.registry = registry
        self.model = ModelClass(config,tokenizer=tokenizer,device=device,cache=self.cache)  
        self.training_step_type=training_step_type # Provvisorio, trovare un modo più elegante
        self.nested = None  
        if self.config.loss_type=='normal':
            self.config.loss_type='cross_entropy_loss' # For legacy models
        print(f"Initializing {self.config.loss_type} as loss function.")
        self.loss_function=self.cache.registry.create("loss",self.config.loss_type)
        #if self.config.loss_type=='fused':
        #    self.loss_function = self.cache.registry.create("loss", "cross_entropy_loss_fused", *[], **{"ignore_index":config.pad_token_id})
        #else:
        #    self.loss_function = self.cache.registry.create("loss", "cross_entropy_loss", *[], **{"ignore_index":config.pad_token_id})
        if skip_init==False: # Provvisorio, trovare un modo più elegante
            if not getattr(self, "_restored_from_ckpt", False): 
                print("Initializing transformer weights...")
                self.model.apply(init_transformer_weights_)
            
        self.batch_size=batch_size # Utile per il learning rate scheduling
        self.tokenizer=tokenizer
        # Maskerator setup
        self.maskerator=Maskerator(mask_token=self.config.mask_token_id,
                                       substitution_rate=self.config.masked_substitution_rate,
                                       pad_token_id=self.config.pad_token_id,
                                       cloze_prob=self.config.cloze_probability,
                                       random_prob=self.config.random_probability,
                                       same_prob=self.config.same_probability,
                                       vocab_size=self.config.vocab_size)


        if self.training_step_type=='classification':
            # Initializing the F1 metric (unless torchmetrics is not available)
            try: 
                from torchmetrics.classification import MulticlassF1Score, BinaryF1Score
                if self.config.num_classes == 2:
                    self.train_f1 = BinaryF1Score()
                    self.val_f1 = BinaryF1Score()
                else:
                    self.train_f1 = MulticlassF1Score(num_classes=self.config.num_classes, average='macro')
                    self.val_f1 = MulticlassF1Score(num_classes=self.config.num_classes, average='macro')
            except:
                print("Install torchmetrics if you want F1 scores. (pip install torchmetrics)")
                self.train_f1=None
                self.val_f1=None
        
    def forward(self, _input,*args,**kwargs):
        if isinstance(_input,torch.Tensor):
            _input=NormalTensor(tensor=_input)
        return self.model(_input.to(self.device),*args,**kwargs)
    def on_load_checkpoint(self, checkpoint):
        self._restored_from_ckpt = True       
        if self.load_mode in ["weights_only", "weights_and_optimizer"]:
           checkpoint["lr_schedulers"] = [] 
           checkpoint["epoch"] = 0
           checkpoint["global_step"] = 0
           checkpoint.pop("loops", None)
           checkpoint.pop("MatformerDataModule",None)
           checkpoint.pop("callbacks", None)
        if self.load_mode == "weights_only":
            checkpoint["optimizer_states"] = []
        if self.load_mode == 'publication':
            new_checkpoint=checkpoint['state_dict']
            new_checkpoint=checkpoint['hyper_parameters']
    def validation_step(self, batch, batch_idx=None):
        if self.training_step_type == 'pretraining':
            raise NotImplementedError
        else:
            input_ids = batch["input_ids"].unpad()
            logits = self(input_ids)
            loss = self.loss_function(logits, batch["labels"])
            preds = logits.argmax(dim=-1)
            if self.val_f1 is not None:
                self.val_f1.update(preds, batch["labels"])
            acc = (preds == batch["labels"]).float().mean()
            self.log_dict({'val/loss': loss, 'val/accuracy': acc}, batch_size=self.batch_size, sync_dist=True)
            return loss
    def on_validation_epoch_end(self):
        if self.val_f1 is not None:
            self.log('val/f1', self.val_f1.compute(), prog_bar=True, on_epoch=True, batch_size=self.batch_size, sync_dist=True)
            self.val_f1.reset()
    def on_train_epoch_end(self):
        if self.train_f1 is not None:
            self.log('train/f1', self.train_f1.compute(), prog_bar=True, on_epoch=True, batch_size=self.batch_size, sync_dist=True)
            self.train_f1.reset()           
    def training_step(self, batch, batch_idx=None):
        if self.training_step_type=='pretraining':
            return self.pretraining_step(batch,batch_idx)
        elif self.training_step_type=='classification':
            return self.classification_step(batch, batch_idx)
        else:
            raise ValueError        
    def classification_step(self, batch, batch_idx=None):
        input_ids = batch["input_ids"]
        targets = batch["labels"]
        input_ids=input_ids.unpad() # Unpadding to avoid attention attending pad tokens
        logits = self(input_ids)
        #logits=logits.pad() # Restore padding for the loss function to match targets
        #print(f"Debug: input_ids => {input_ids.shape}, targets=> {targets.shape}, logits=> {logits.shape}")        
        loss_kwargs={} # Todo
        loss = self.loss_function(logits, targets, **loss_kwargs)
        self.log('train/loss', loss, prog_bar=True,batch_size=self.batch_size)  
        preds = logits.argmax(dim=-1)
        if self.train_f1 is not None:
            self.train_f1.update(preds, targets)
        acc = (preds == targets).float().mean()  
        self.log("train/accuracy", acc, prog_bar=True, on_step=True, on_epoch=True,batch_size=self.batch_size) 
        return loss
        
    def pretraining_step(self, batch, batch_idx=None):
        try:
            input_sequence = batch['sequence'] # Arriva la sequenza già tokenizzata dal MatformerDataModule
            batch['sequence'] = input_sequence
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
            if self.config.loss_type=='cross_entropy_loss_fused':
                model_return_type = 'hidden'
                flattening_dimension = self.config.hidden_size
                loss_kwargs = {"lm_head_weight": self.model.lm_head.module.inner.weight}
                if hasattr(self.model.lm_head.module.inner, "bias"):
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
                targets_flat = original_sequence.tensor.view(-1)
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
            loss = self.loss_function(inputs, targets, **loss_kwargs)  
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
            #if len(self.cache.additional_logs.items()) > 0:
            #            for k, v in self.cache.additional_logs.items():
            #                self.log(k, v, on_step=True, batch_size=self.batch_size)
            #            self.cache.additional_logs.clear()
                        
                   
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
                loss = self.loss_function(logits_flat[mask], targets_flat[mask])
            else:
                mask = base_mask[1:]
                loss = self.loss_function(logits_flat[:-1][mask], targets_flat[1:][mask])
            """          

             

            # TODO: this part has to be revised and cleaned
            additional_metrics=False
            if additional_metrics:
                if self.global_step % 100 == 0:         
                    with torch.no_grad():
                        if self.cache.additional_logs:
                             keys = list(self.cache.additional_logs.keys())
                             vals = [v if torch.is_tensor(v) else torch.tensor(v, device=self.device) for v in self.cache.additional_logs.values()]
                             vals_cpu = torch.stack(vals).detach().cpu().float().tolist()
                             log_dict = dict(zip(keys, vals_cpu))
                             self.log_dict(log_dict, on_step=True, batch_size=self.batch_size)
                             self.cache.additional_logs.clear()                         
                        # 'inputs' is already flattened and filtered for the loss (works for GPT/BERT/Unpadded)
                        # We sample the first 1024 tokens to keep the compute cost negligible
                        sample_size = min(inputs.size(0), 1024)
                        diag_data = inputs[:sample_size]

                        if self.config.loss_type != 'cross_entropy_loss_fused':
                            # 1. Logit Statistics (Saturation check)
                            self.log("health/logits_max", diag_data.max(), on_step=True, batch_size=self.batch_size)
                            self.log("health/logits_std", diag_data.std(), on_step=True, batch_size=self.batch_size)
                            
                            # 2. Entropy (Confidence check)
                            probs = torch.softmax(diag_data, dim=-1)
                            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean()
                            self.log("health/entropy", entropy, on_step=True, batch_size=self.batch_size)
                        else:
                            # If fused, inputs are actually hidden states
                            self.log("health/hidden_max", diag_data.max(), on_step=True, batch_size=self.batch_size)
                            self.log("health/hidden_std", diag_data.std(), on_step=True, batch_size=self.batch_size)

                        # 3. Gating Health (if the model has gates)
                        for name, p in self.named_parameters():
                            if 'gate' in name and p.numel() == 1:
                                self.log(f"gates/{name}_val", p.item(), on_step=True, batch_size=self.batch_size)          
            return loss
            """
            This exception must be only for extreme cases. To avoid breaking a large training if only a minor issue occurs,
            ex a particularly long batch, the batch is skipped. The event will be logged. Be sure that this number is limited
            to a very small amount of steps.
            """
        except Exception as e:
            print("------------------  ALARM ------------------")
            if not any(s in str(e).lower() for s in ("out of memory", "cuda out of memory", "cublas")):
                print("CAUGHT EXCEPTION: ")
                print(e)
                print("Training will try to continue but this may be unreliable. Careful check what is happening")
                self.log("train/EXCEPTIONS",1,on_step=True,on_epoch=False,batch_size=self.batch_size)
            else:
                print("OUT OF MEMORY ERROR")
                print("Training is continuing, but be sure that this happens very seldom or reduce the batch size.")
                self.log(
                    "train/skipped_oom", 1,
                    on_step=True, on_epoch=False, batch_size=self.batch_size
                )
            print("All the gradients from the current batch and all the previous accumulated gradients were discarded.")
            try:
                opts = self.optimizers()
                for o in opts if isinstance(opts, list) else (opts,):
                    o.zero_grad(set_to_none=True)
            except Exception:
                pass

            if hasattr(self.cache, "storage"):
                self.cache.storage.clear()

            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            return sum(p.sum() for p in self.parameters()) * 0.0
        
    def on_before_optimizer_step(self, optimizer):
        additional_metrics=False
        if additional_metrics:
            if self.global_step % 100 == 0:
                grad_norm_sq = 0.0
                weight_norm_sq = 0.0
                
                for p in self.parameters():
                    if p.grad is not None:
                        # Use .item() to move scalars to CPU immediately, preventing GPU sync overhead later
                        grad_norm_sq += p.grad.detach().norm(2).item() ** 2
                        weight_norm_sq += p.detach().norm(2).item() ** 2
                
                total_grad_norm = grad_norm_sq ** 0.5
                total_weight_norm = weight_norm_sq ** 0.5
                
                self.log("health/total_grad_norm", total_grad_norm, on_step=True, batch_size=self.batch_size)

                # 4. Update-to-Weight Ratio (The "Karpathy Metric")
                try:
                    lr = optimizer.param_groups[0]['lr']
                    if total_weight_norm > 1e-8:
                        ratio = (lr * total_grad_norm) / total_weight_norm
                        self.log("health/update_weight_ratio", ratio, on_step=True, batch_size=self.batch_size)
                except Exception:
                    pass
    def configure_optimizers(self):
        if self.train_config.get("no_decay_for_embedding", False) and self.train_config["optimizer"] != "muon":
            raise ValueError("no_decay_for_embedding for optimizers different than Muon not implemented yet! (Altough it's easy). Please remove it ")

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
                    #print(f"{name} (Convolutional) in AdamW (ndim={param.ndim})")                          
                elif "lm_head" in name or "embed_tokens" in name or param.ndim < 2:
                    adamw_params.append(param)
                    #print(f"{name} in AdamW (ndim={param.ndim})")
                else:
                    muon_params.append(param)
                    #print(f"{name} in Muon")
            
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
            no_decay_params = []
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                if "conv" in name:
                    adamw_params.append(param)
                    #print(f"{name} (Convolutional) in AdamW (ndim={param.ndim})")    
                elif "lm_head" in name or "embed_tokens" in name or param.ndim < 2:
                    adamw_params.append(param)
                    #print(f"{name} in AdamW (ndim={param.ndim})")
                else:
                    muon_params.append(param)
                    #print(f"{name} in Muon")
                if self.train_config.get("no_decay_for_embedding", False) and ("lm_head" in name or "embed_tokens" in name):
                    no_decay_params.append(param)
                    #print(f"{name} has weight decay disabled.")
                              
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
                no_decay_params=no_decay_params
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

    @staticmethod
    def load_from_checkpoint(checkpoint_path, ModelClass, config=None, train_config=None, map_location=None, tokenizer=None, overrides=None,varlen_strategy='padding', external_mapping=None, training_step_type='pretraining', skip_init=True):
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
        if type(tokenizer) != MatformerTokenizer:
          tokenizer = MatformerTokenizer(
              config=config,
              tokenizer_type='huggingface',
              tokenizer=tokenizer,
              tokenizer_name=tokenizer,
              varlen_strategy=varlen_strategy,
          )                
   

        model = PL_ModelWrapper(ModelClass=ModelClass, config=config, train_config=train_config, tokenizer=tokenizer, device=map_location, skip_init=skip_init, training_step_type=training_step_type)  
        #model.load_state_dict(checkpoint['state_dict'])
        problems=model.load_stable_state_dict(checkpoint['state_dict'], strict=False, external_mapping=external_mapping)
        #print("Found this config:")
        #print(config)
        print(problems)
        return model,config   
        

