#matformer/training/lr_scheduler.py
import torch
import math
def configure_lr_schedulers(train_config, optimizers):
    if not isinstance(optimizers,list):
        print("Warning: configure_lr_schedulers expects optimizers to be a list")
        optimizers=[optimizers]
    if not train_config.get("lr_scheduling", False):
        return None

    total_steps  = train_config["total_steps"]
    warmup_steps = int(train_config.get("warmup_steps", 0.05) * total_steps)
    hold_steps   = int(train_config.get("hold_steps",   0.0)  * total_steps)
    decay_steps  = max(1, total_steps - warmup_steps - hold_steps)
    anneal_step  = train_config.get("annealing_start_step")
     
    DECAY_FNS = {
        "custom":     lambda p, s, e: e + (s - e) * 0.5 * (1 + math.cos(math.pi * p)), #deprecated, is the same as 'cosine'
        "cosine":     lambda p, s, e: e + (s - e) * 0.5 * (1 + math.cos(math.pi * p)),
        "linear":     lambda p, s, e: s + (e - s) * p,
        "polynomial": lambda p, s, e: e + (s - e) * (1 - p) ** train_config.get("poly_power", 2),
        "constant":   lambda p, s, e: s,
    }
    decay = DECAY_FNS[train_config.get("scheduler", "cosine")]
     
    def make_schedule(opt):
        lr           = opt.param_groups[0]["lr"]
        end_factor   = train_config.get("final_lr", 0.0) / lr
        anneal_factor = train_config.get("annealing_lr", lr) / lr if anneal_step else None
     
        def lr_schedule(step):
            if anneal_step and step >= anneal_step:
                prog = (step - anneal_step) / max(1, total_steps - anneal_step)
                return decay(min(prog, 1.0), anneal_factor, end_factor)
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            if step < warmup_steps + hold_steps:
                return 1.0
            prog = (step - warmup_steps - hold_steps) / decay_steps
            return decay(min(prog, 1.0), 1.0, end_factor)
     
        return torch.optim.lr_scheduler.LambdaLR(opt, lr_schedule)
     
    
    return [make_schedule(optimizer) for optimizer in optimizers] 
     
