"""
A function to train a generic model using the matformer library
A JSON config file must be provided
But each argument can be overridden from the CLI
"""

import argparse, json, torch, pytorch_lightning as pl, wandb
from pathlib import Path
from importlib import import_module
from transformers import AutoTokenizer
from matformer.tokenizers import MatformerTokenizer
from matformer.training_functions import MatformerDataModule
from matformer.model_config import ModelConfig
from matformer.models import PL_ModelWrapper
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
import math, os

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
    # Autoencoder experiment arguments
    parser.add_argument('--autoencoder-experiment', action='store_true')
    parser.add_argument('--batch-size-phase1', type=int, default=None)
    parser.add_argument('--batch-size-phase2', type=int, default=None)
    parser.add_argument('--batch-size-phase3', type=int, default=None)

    args = parser.parse_args()
    overrides = dict(kv.split('=') for kv in args.override)
    return args.config, overrides, args.gpu, args.checkpoint, args.start_from_scratch,args

def get_model_class(model_class: str):
    module = import_module("matformer.transformer_blocks")
    return getattr(module, model_class)
"""

Il dataloader prende il documento, usa il modello di entropia per 
dividere il documento in patch. Può però far vedere al modello di entropia solo 1024 caratteri alla volta.

Il dataloader ritorna al chiamante le patch, in formato testuale, ad esempio:
["ciao ","come ","stai?"]

Il chiamante esegue la codifica delle patch testuali
in patch unicode ed esegue il padding delle patch NON CON IL TOKEN DI PAD ma con l'EOS dell'encoder. È ancora un bug da risolvere.
["ciao <EOS><EOS>...","come <EOS><EOS>...","stai?<EOS><EOS>..."]

Il modello riceve in ingresso il batch delle patch.
La loss viene calcolata sul testo paddato con l'EOS

1) Caricare il modello di entropia nello script di training
2) Passarlo al dataloader


"""
def main():
    config_path, overrides, device_count, ckpt_arg, start_scratch, args = parse_args()
    cfg = apply_overrides(load_config(config_path), overrides)

    model_cfg = ModelConfig(**cfg['model_config'])
    train_cfg = cfg['training']
    data_cfg = cfg['data']
    tok_cfg = cfg['tokenizer']
    save_dir = cfg.get('save_dir', './checkpoints')
    pl.seed_everything(train_cfg.get('seed', 27))
    tokenizer = (
        AutoTokenizer.from_pretrained(tok_cfg['pretrained_name'])
        if tok_cfg['type'] == 'huggingface' else str(tok_cfg['type'])
    )

    # If we are in masked mode, be sure that there is a mask token and config is consistent
    if model_cfg.training_objective=='masked':
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
        model_cfg,
        tokenizer=tokenizer,
        varlen_strategy=tok_cfg['varlen_strategy']
    )
    
    data = MatformerDataModule(
        data_root=data_cfg['data_root'],
        batch_size=data_cfg['batch_size'],
        num_workers=data_cfg.get('num_workers', 0),
        config=model_cfg,
        tokenizer=tokenizer
    )
    max_epochs = train_cfg.get("max_epochs", 1)
    if data.__len__() is not None:
        num_batches = math.ceil(len(data)/data_cfg["batch_size"])
        accumulate_grad_batches = train_cfg.get("accumulate_grad_batches", 1)
        total_steps = (num_batches // accumulate_grad_batches) * max_epochs
        train_cfg["total_steps"] = total_steps
        train_cfg["num_batches"] = num_batches
    else:
        print("The Datamodule is not returning the length. Thus, LR scheduling is disabled")
        train_cfg["lr_scheduling"]=False
    ModelClass = get_model_class(cfg['model_class'])
    model = PL_ModelWrapper(ModelClass,config=model_cfg, tokenizer=tokenizer, train_config=train_cfg, device='cuda', batch_size=data_cfg['batch_size'])

    wandb_logger = WandbLogger(
        name=cfg.get('wandb_run_name', 'training-run'),
        project=cfg.get('wandb_project', 'matformer'),
        config=cfg
    )
    checkpoint = ModelCheckpoint(
        dirpath=save_dir,
        filename=train_cfg.get('checkpoint_name', 'model'),
        save_top_k=1,
        save_last=True,
        every_n_train_steps=train_cfg.get("save_every_n_steps", None)  
    )

     #overfit_batches=1 => Useful for debug
     #max_steps=train_cfg['max_steps'],
    torch.set_float32_matmul_precision('high')

    ckpt_path = None
    if not start_scratch:
        if ckpt_arg and os.path.exists(ckpt_arg):
            ckpt_path = ckpt_arg
        else:
            last_ckpt = Path(save_dir) / "last.ckpt"
            if last_ckpt.exists():
                ckpt_path = str(last_ckpt)

    if getattr(args, 'autoencoder_experiment', False):
        phases = ['autoencoder-training', 'autoencoder-patch-train', 'autoencoder-final-annealing']
        batch_sizes = [
            getattr(args, 'batch_size_phase1', None) or data_cfg['batch_size_phase_1'],
            getattr(args, 'batch_size_phase2', None) or data_cfg['batch_size_phase_2'],
            getattr(args, 'batch_size_phase3', None) or data_cfg['batch_size_phase_3']
        ]
        
        print("Running autoencoder experiment with 3 phases")
        
        for i, (phase, batch_size) in enumerate(zip(phases, batch_sizes)):

            print(f"\nStarting Phase {i+1}: {phase} with batch size {batch_size}")
            # Update data module if batch size changed
            if batch_size != data_cfg['batch_size']:
                print(f"Reinitializing data module with batch size {batch_size}")
                data = MatformerDataModule(
                    data_root=data_cfg['data_root'],
                    batch_size=batch_size,
                    num_workers=data_cfg.get('num_workers', 0),
                    config=model_cfg,
                    tokenizer=tokenizer,
                    autoencoder_experiment=True
                )
                
            # Set training phase (this handles freezing/unfreezing)
            model.model.set_training_phase(phase)
            
            # Create new trainer for this phase
            trainer = pl.Trainer(
                logger=wandb_logger,
                callbacks=[checkpoint],
                precision='16-mixed',
                gradient_clip_val=train_cfg.get('gradient_clip_val', 1),
                accelerator='gpu',
                devices=device_count,
                log_every_n_steps=10,
                accumulate_grad_batches=train_cfg.get('accumulate_grad_batches', 1),
                default_root_dir=save_dir,
                max_epochs=1
            )
            
            # Train for one epoch using the same checkpoint
            trainer.fit(model, data, ckpt_path=ckpt_path)
            
        print("Completed all autoencoder training phases")
        return

    # Standard training
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint],
        precision='16-mixed',
        gradient_clip_val=train_cfg.get('gradient_clip_val', 1),
        accelerator='gpu',
        devices=device_count,
        log_every_n_steps=10,
        accumulate_grad_batches=train_cfg.get('accumulate_grad_batches', 1),
        default_root_dir=save_dir,
        max_epochs=train_cfg.get("max_epochs", 1)
    )

    try:
        trainer.fit(model, data, ckpt_path=ckpt_path)
    except KeyboardInterrupt:
        response = input("\nTraining interrupted. Save model? (y/n): ").strip().lower()
        if response == 'y':
            trainer.save_checkpoint(f"{save_dir}/interrupted.ckpt")
            print("Checkpoint saved.")
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
