#matformer/training/optimizers.py

import torch
from matformer.extra.muon import Muon
from torch.optim import AdamW, Adam

def configure_optimizers(model,train_config):
    if train_config.get("no_decay_for_embedding", False) and train_config["optimizer"] != "muon":
        raise ValueError("no_decay_for_embedding for optimizers different than Muon not implemented yet! (Altough it's easy). Please remove it ")

    if train_config["optimizer"] == "muonclip":
        from muon import MuonClip, MuonConfig
        base_lr = train_config["lr"]
        
        from types import SimpleNamespace

        model_config = SimpleNamespace(
            num_key_value_heads=model.config.num_attention_heads,
            num_attention_heads=model.config.num_attention_heads,
            head_dim=model.config.hidden_size // model.config.num_attention_heads
        )
        
        muon_lr = base_lr * 0.2 * math.sqrt(max(
            p.shape[0] for p in model.parameters() if p.ndim >= 2 and p.requires_grad
        ))
        
        muon_config = MuonConfig(
            lr=base_lr, ###To check!!!
            muon_beta=train_config.get("muon_momentum", 0.95),
            muon_decay=train_config.get("weight_decay", 0.01),
            ns_steps=train_config.get("muon_ns_steps", 5),
            adam_betas=train_config.get("betas", (0.9, 0.95)),
            adam_decay=train_config.get("weight_decay", 0.01),
            adam_eps=train_config.get("eps", 1e-10),
            enable_clipping=True,
            clipping_layers_mapping={"q_proj": "packed_proj", "k_proj": "packed_proj"},
            clipping_threshold=train_config.get("clip_threshold", 50.0),
            clipping_alpha=train_config.get("clip_alpha", 0.5),
            log_max_logits=False,
            cans_ortho=False,
            estimate_lower_bound=False
        )
        
        optimizer = MuonClip(self, model_config, muon_config)
        
    elif train_config["optimizer"] == "muonflash":
        muon_params = []
        adamw_params = []
        for name, param in model.named_parameters():
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
        
        base_lr = train_config["lr"]
        
        from flash_muon import Muon as FlashMuon
        muon_lr = base_lr * 0.2 * math.sqrt(max(muon_params[0].shape[:2])) if muon_params else base_lr
        optimizer = [
            FlashMuon(muon_params, lr=muon_lr, 
                     momentum=train_config.get("muon_momentum", 0.95), rank=0, world_size=1),
            torch.optim.AdamW(adamw_params, lr=base_lr, 
                            betas=train_config.get("betas", (0.9, 0.95)),
                            weight_decay=train_config.get("weight_decay", 0.01))
        ]
        
    elif train_config["optimizer"] == "muon":
        muon_params = []
        adamw_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
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
            if train_config.get("no_decay_for_embedding", False) and ("lm_head" in name or "embed_tokens" in name):
                no_decay_params.append(param)
                #print(f"{name} has weight decay disabled.")
                          
        base_lr = train_config["lr"]
        
        optimizer = Muon(
            lr=base_lr,
            wd=train_config.get("weight_decay", 0.01),
            muon_params=muon_params,
            momentum=train_config.get("muon_momentum", 0.95),
            nesterov=train_config.get("muon_nesterov", True),
            ns_steps=train_config.get("muon_ns_steps", 5),
            adamw_params=adamw_params,
            adamw_betas=train_config.get("betas", (0.9, 0.95)),
            adamw_eps=train_config.get("eps", 1e-10),
            no_decay_params=no_decay_params
        )
        
    elif train_config["optimizer"] == "adamw":
        optimizer = AdamW(
            model.parameters(),
            lr=train_config["lr"],
            weight_decay=train_config.get("weight_decay", 0.01),
        )
    elif train_config["optimizer"] == "adam":
        optimizer = Adam(
            model.parameters(),
            lr=train_config["lr"],
            weight_decay=train_config.get("weight_decay", 0.01),
        )
    if not isinstance(optimizer,list):
        optimizer=[optimizer]
    return optimizer
