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
import math
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
    args = parser.parse_args()
    overrides = dict(kv.split('=') for kv in args.override)
    return args.config, overrides, args.gpu

def get_model_class(model_class: str):
    module = import_module("matformer.transformer_blocks")
    return getattr(module, model_class)

def main():
    config_path, overrides, device_count = parse_args()
    cfg = apply_overrides(load_config(config_path), overrides)

    model_cfg = ModelConfig(**cfg['model_config'])
    train_cfg = cfg['training']
    data_cfg = cfg['data']
    tok_cfg = cfg['tokenizer']
    save_dir = cfg.get('save_dir', './checkpoints')

    pl.seed_everything(train_cfg.get('seed', 27))

    tokenizer = (
        AutoTokenizer.from_pretrained(tok_cfg['pretrained_name']) 
        if tok_cfg['type'] == 'huggingface' else 'bytes'
    )
    
    tokenizer = MatformerTokenizer(
        model_cfg,
        tokenizer=tokenizer,
        varlen_strategy=tok_cfg['varlen_strategy']
    )

    data = MatformerDataModule(
        data_root=data_cfg['data_root'],
        batch_size=data_cfg['batch_size'],
        num_workers=data_cfg.get('num_workers', 2),
        config=model_cfg,
        tokenizer=tokenizer
    )
    num_batches = math.ceil(len(data)/data_cfg["batch_size"])
    accumulate_grad_batches = train_cfg.get("accumulate_grad_batches", 1)
    max_epochs = train_cfg.get("max_epochs", 1)

    total_steps = (num_batches // accumulate_grad_batches) * max_epochs
    train_cfg["total_steps"] = total_steps
    train_cfg["num_batches"] = num_batches
    ModelClass = get_model_class(cfg['model_class'])
    model = PL_ModelWrapper(ModelClass, config=model_cfg, tokenizer=tokenizer, train_config=train_cfg, device='cuda', batch_size=data_cfg['batch_size'])

    wandb_logger = WandbLogger(
        name=cfg.get('wandb_run_name', 'training-run'),
        project=cfg.get('wandb_project', 'matformer'),
        config=cfg
    )
    checkpoint = ModelCheckpoint(
        dirpath=save_dir,
        filename=train_cfg.get('checkpoint_name', 'model'),
        save_top_k=1,
        save_last=True
    )
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint],
        precision='16-mixed',
        accelerator='gpu',
        devices=device_count,
        strategy='ddp',
        log_every_n_steps=10,
        accumulate_grad_batches=train_cfg.get('accumulate_grad_batches', 1),
        default_root_dir=save_dir,
        max_epochs=1
    )
     #overfit_batches=1 => Useful for debug
     #max_steps=train_cfg['max_steps'],
    torch.set_float32_matmul_precision('high')
    try:
        trainer.fit(model, data)
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
