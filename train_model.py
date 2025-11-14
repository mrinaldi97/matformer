"""
A function to train a generic model using the matformer library
A JSON config file must be provided
But each argument can be overridden from the CLI
"""

import argparse, json, torch, pytorch_lightning as pl, wandb
from pathlib import Path
from importlib import import_module
from transformers import AutoTokenizer
from matformer.matformer_tokenizers import MatformerTokenizer
from matformer.data_module import MatformerDataModule
from matformer.model_config import ModelConfig
from matformer.models import PL_ModelWrapper
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
import math, os
from datetime import datetime

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def apply_overrides(cfg, overrides):
    for key, val in overrides.items():
        keys = key.split('.')
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        try:
            d[keys[-1]] = eval(val)
        except:
            d[keys[-1]] = val
    return cfg

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--override', nargs='*', default=[])
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--checkpoint', type=str, default=None) 
    parser.add_argument('--start-from-scratch', action='store_true') 

    args = parser.parse_args()
    overrides = dict(kv.split('=') for kv in args.override)
    return args.config, overrides, args.gpu, args.checkpoint, args.start_from_scratch

def get_model_class(model_class: str):
    module = import_module("matformer.transformer_blocks")
    return getattr(module, model_class)

def main():
    config_path, overrides, device_count, ckpt_arg, start_scratch = parse_args()
    cfg = apply_overrides(load_config(config_path), overrides)
    
    #--------------------------------------#
    # Removed training_objective and is_causal. 
    # Brutto da vedere, credo si possa fare di meglio
    model_class = cfg['model_class']
    training_objective = "autoregressive" if model_class == "Autoregressive_Model" else "masked"
    is_causal = True if model_class == "Autoregressive_Model" else False
    
    cfg['model_config']['training_objective'] = training_objective
    cfg['model_config']['is_causal'] = is_causal
    #--------------------------------------#
    
    model_cfg = ModelConfig(**cfg['model_config'])
    train_cfg = cfg['training']
    data_cfg = cfg['data']
    tok_cfg = cfg['tokenizer']
    save_dir = cfg.get('save_dir', './checkpoints')
    
    
    
    pl.seed_everything(train_cfg.get('seed', 27))
    
    # Detect device
    if torch.cuda.is_available():
        accelerator = 'gpu'
        precision = '16-mixed'
        device_string = 'cuda'
    elif torch.backends.mps.is_available():
        accelerator = device_string = 'mps'
        precision = '32'
    else:
        accelerator = device_string = 'cpu'
        precision = '32'
    """
    tokenizer = (
        AutoTokenizer.from_pretrained(tok_cfg['pretrained_name'])
        if tok_cfg['type'] == 'huggingface' else str(tok_cfg['type'])
    )

    # If we are in masked mode, be sure that there is a mask token and config is consistent
    if model_cfg.training_objective == 'masked':
        print("Addestramento di un modello MLM")
        if tokenizer.mask_token is None:
            tokenizer.add_special_tokens({"mask_token": "<mask>"})
            print(f"Added <mask> token to {tok_cfg['pretrained_name']} at id={tokenizer.mask_token_id}")

        mask_id = tokenizer.mask_token_id

        if getattr(model_cfg, "mask_token_id", None) is None:
            model_cfg.mask_token_id = mask_id
        else:
            if model_cfg.mask_token_id != mask_id:
                raise ValueError(
                    f"Inconsistent mask_token_id: model_cfg={model_cfg.mask_token_id}, tokenizer={mask_id}"
                )

        if model_cfg.vocab_size < len(tokenizer):
            print(f"Expanding vocab_size from {model_cfg.vocab_size} -> {len(tokenizer)}")
            model_cfg.vocab_size = len(tokenizer)

    tokenizer = MatformerTokenizer(
        config=model_cfg,
        tokenizer=tokenizer,
        varlen_strategy=tok_cfg['varlen_strategy']
    )
    """
    # Create data module with MDAT dataset
    data = MatformerDataModule(
        mdat_path=data_cfg['data_root'],
        iteration_modality='chunked_tokens', 
        pad_token_id=model_cfg.pad_token_id,
        varlen_strategy=tok_cfg['varlen_strategy'],
        mdat_strategy=data_cfg['mdat_strategy'],
        mdat_view=data_cfg['mdat_view'],
        with_meta=False,
        max_seq_len=model_cfg.max_position_embeddings,
        batch_size=data_cfg['batch_size']
    )
    data.setup()
    # Calculate training steps if dataset length is available
    max_epochs = train_cfg.get("max_epochs", 1)
    if hasattr(data, '__len__') and len(data) > 0: #Nel caso di più GPU viene già divisa per numero di GPU (es. /4)
        num_batches = math.ceil(len(data) / data_cfg["batch_size"])
        accumulate_grad_batches = train_cfg.get("accumulate_grad_batches", 1)
        total_steps = (num_batches // accumulate_grad_batches) * max_epochs
        train_cfg["total_steps"] = total_steps
        train_cfg["num_batches"] = num_batches
    else:
        print("The Datamodule is not returning the length. Thus, LR scheduling is disabled")
        train_cfg["lr_scheduling"] = False
    
    # Initialize model
    ModelClass = get_model_class(cfg['model_class'])
    
    model = PL_ModelWrapper(
        ModelClass, 
        config=model_cfg, 
        tokenizer=None, 
        train_config=train_cfg, 
        device=device_string, 
        batch_size=data_cfg['batch_size']
    )
    compile=False
    if compile:
       model=torch.compile(model)
    
    # Create timestamped checkpoint filename to avoid name clashes
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    #checkpoint_name = f"{train_cfg.get('checkpoint_name', 'model')}_{timestamp}"    
    checkpoint_name = train_cfg.get('checkpoint_name', 'model') # jerik
    run_name = cfg.get('wandb_run_name', 'training-run') # jerik
    
    # Setup logging
    wandb_logger = WandbLogger(
        name=f"{run_name}_{timestamp}",
        #name=cfg.get('wandb_run_name', 'training-run'),
        project=cfg.get('wandb_project', 'matformer'),
        config=cfg
    )
    
    checkpoint = ModelCheckpoint(
        dirpath=save_dir,
        filename=checkpoint_name,
        save_top_k=1,
        save_last=True,
        every_n_train_steps=train_cfg.get("save_every_n_steps", None)  
    )

    torch.set_float32_matmul_precision('high')

    # Handle checkpoint loading
    ckpt_path = None
    if not start_scratch:
        if ckpt_arg and os.path.exists(ckpt_arg):
            ckpt_path = ckpt_arg
        else:
            # last_ckpt = Path(save_dir) / f"{checkpoint_name}_last.ckpt"
            # if last_ckpt.exists():
            #     ckpt_path = str(last_ckpt)
            
            # jerik
            last_ckpt = Path(save_dir) / "last.ckpt"
            if last_ckpt.exists():
                print(f"Resuming training from {last_ckpt}")
                ckpt_path = str(last_ckpt)
            else:
                print("No checkpoint found, starting from scratch.")
            


    # Create trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint],
        precision=precision,
        gradient_clip_val=train_cfg.get('gradient_clip_val', 1),
        accelerator=accelerator,
        devices=device_count,
        log_every_n_steps=10,
        accumulate_grad_batches=train_cfg.get('accumulate_grad_batches', 1),
        default_root_dir=save_dir,
        max_epochs=max_epochs
    )

    try:
        trainer.fit(model, data, ckpt_path=ckpt_path)
    except KeyboardInterrupt:
        response = input("\nTraining interrupted. Save model? (y/n): ").strip().lower()
        if response == 'y':
            interrupted_checkpoint = f"{save_dir}/interrupted_{timestamp}.ckpt"
            trainer.save_checkpoint(interrupted_checkpoint)
            print(f"Checkpoint saved as {interrupted_checkpoint}")
        else:
            print("Checkpoint not saved.")


if __name__ == '__main__':
    main()

"""
CLI overrides via --override
train_model.py --config CONFIG.json --override key1=val1 key2.nested=val2 
• Each override is a single KEY=VALUE pair.
• Keys are dotted paths that mirror the JSON hierarchy.
• Values are parsed with eval first; if that fails they are kept as strings.
Examples: 

# Run a quick debug with smaller model
--override model_config.hidden_size=128 model_config.num_hidden_layers=2

# Switch to cosine scheduler and longer warmup
--override training.scheduler="cosine" training.warmup_steps=5000

# Change batch-size, learning-rate and run name
--override data.batch_size=32 training.lr=1e-4 wandb_run_name=quick_try
"""
