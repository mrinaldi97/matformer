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

import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Matformer model')
    parser.add_argument('--config', type=str, help="Path to single combined config file")
    parser.add_argument('--model_config', type=str, help="Path to model_config.json")
    parser.add_argument('--training_config', type=str, help="Path to train_config.json")
    parser.add_argument('--data_config', type=str, help="Path to data_config.json")
    parser.add_argument('--tokenizer_config', type=str, help="Path to tokenizer_config.json")
    parser.add_argument('--override', nargs='*', default=[], help="Override config parameters as key=value pairs")
    parser.add_argument('--gpu', type=int, default=1, help="GPU device ID")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to checkpoint file")
    parser.add_argument('--start-from-scratch', action='store_true', help="Start training from scratch")
    parser.add_argument('--simulate', action='store_true', help="Instantiate model and print state_dict shapes, then exit")
    parser.add_argument('--dump-json', type=str, default=None, help="Path to dump JSON state dict shapes")
    parser.add_argument('--debug-steps', type=int, default=None, help="If you choose this, train for one epoch on this number of steps")
    parser.add_argument('--compile', action='store_true', help="Torch.compile the whole model")
    args = parser.parse_args()
    
    separate_configs = {
        'model_config': args.model_config,
        'training_config': args.training_config,
        'data_config': args.data_config,
        'tokenizer_config': args.tokenizer_config
    }
    separate_count = sum(1 for v in separate_configs.values() if v is not None)
    
    # Enforce exclusive requirement
    if args.config is not None and separate_count > 0:
        parser.error("Cannot specify both --config and individual config files. Choose ONE approach.")
    
    if args.config is None and separate_count == 0:
        parser.error("Must specify either --config OR all four individual config files (--model_config, --training_config, --data_config, --tokenizer_config)")
    
    if args.config is None and separate_count != 4:
        missing = [k for k, v in separate_configs.items() if v is None]
        parser.error(f"Missing {len(missing)} individual config file(s): {', '.join(missing)}")
    
    # Build config_paths dictionary
    if args.config is not None:
        config_paths = args.config
    else:
        config_paths = {
            "model_config": args.model_config,
            "training": args.training_config,
            "data": args.data_config,
            "tokenizer": args.tokenizer_config
        }
    
    overrides = {}
    for item in args.override:
        try:
            k, v = item.split('=', 1)  # Split on first '=' only
            overrides[k] = v
        except ValueError:
            parser.error(f"Override '{item}' must be in key=value format")
    
    return config_paths, overrides, args.gpu, args.checkpoint, args.start_from_scratch,args.simulate,args.dump_json,args.debug_steps,args.compile

def get_model_class(model_class: str):
    module = import_module("matformer.transformer_blocks")
    return getattr(module, model_class)

def extract_state_dict_shapes(state_dict):
    """
    Extracts a dict of param_name -> shape string
    """
    shapes = {}
    for k, v in state_dict.items():
        shapes[k] = "x".join(str(d) for d in v.shape)
    return shapes

def save_state_dict_json(state_dict, path):
    shapes = extract_state_dict_shapes(state_dict)
    with open(path, 'w') as f:
        json.dump(shapes, f, indent=2)

def load_and_prepare_configs(config_paths, overrides):
    """
    Loads multiple separate JSON configs, merges them, applies overrides,
    and derives any dependent configuration properties (like is_causal).
    """
    if isinstance(config_paths,dict): # Multiple config files
        model_cfg_dict = load_config(config_paths["model_config"])
        train_cfg_dict = load_config(config_paths["training"])
        data_cfg_dict = load_config(config_paths["data"])
        tok_cfg_dict = load_config(config_paths["tokenizer"])
        cfg = {
        "model_class": model_cfg_dict.pop("model_class", None),
        "save_dir": model_cfg_dict.pop("save_dir", "./checkpoints"),
        "wandb_project": model_cfg_dict.pop("wandb_project", "default_project"),
        "wandb_run_name": model_cfg_dict.pop("wandb_run_name", "default_run"),
        "model_config": model_cfg_dict, 
        "training": train_cfg_dict,
        "data": data_cfg_dict,
        "tokenizer": tok_cfg_dict
        }

    else: #Single config file 
        json_config=load_config(config_paths)
        cfg = {
        "model_class": json_config['model_class'],
        "save_dir": json_config.pop("save_dir", "./checkpoints"),
        "wandb_project": json_config.pop("wandb_project", "default_project"),
        "wandb_run_name": json_config.pop("wandb_run_name", "default_run"),
        "model_config": json_config['model_config'], 
        "training": json_config['training'],
        "data": json_config['data'],
        "tokenizer": json_config['tokenizer'],
        "training_objective":json_config.pop("training_objective",None),
        "is_causal":json_config.pop("is_causal",None)        
        }       


    cfg = apply_overrides(cfg, overrides)

    model_class = cfg['model_class'] 
    if getattr(cfg['model_config'],'training_objective',None) is None:
        cfg['model_config']['training_objective'] = "autoregressive" if model_class == "Autoregressive_Model" else "masked"
    if getattr(cfg['model_config'],'is_causal',None) is None:
        cfg['model_config']['is_causal'] = True if model_class == "Autoregressive_Model" else False
    cfg['model_config']['tokenizer_type']=cfg['tokenizer']['type']
    cfg['model_config']['tokenizer_name']=cfg['tokenizer']['pretrained_name']
    if 'wanted_from_strategy' not it cfg['data'].keys():;
         cfg['data']['wanted_from_strategy']='chunked_tokens'
    model_config_dict_clean = cfg['model_config']
    train_config_dict = cfg['training']
    data_config_dict = cfg['data']
    tok_config_dict = cfg['tokenizer']
    
    return model_config_dict_clean, train_config_dict, data_config_dict, tok_config_dict, cfg


def main():
    #config_path, overrides, device_count, ckpt_arg, start_scratch, simulate, dump_json = parse_args()
    #cfg = apply_overrides(load_config(config_path), overrides)
    
    config_paths, overrides, device_count, ckpt_arg, start_scratch, simulate, dump_json, debug_steps,_compile = parse_args()
    model_config_dict, train_cfg, data_cfg, tok_cfg, cfg = load_and_prepare_configs(config_paths, overrides)
    
    #model_cfg = ModelConfig(**cfg['model_config'])
    model_cfg = ModelConfig(**model_config_dict)
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
    
    # Create data module with MDAT dataset
    data = MatformerDataModule(
        mdat_path=data_cfg['data_root'],
        iteration_modality=data_cfg['wanted_from_strategy'], 
        pad_token_id=model_cfg.pad_token_id,
        varlen_strategy=tok_cfg['varlen_strategy'],
        mdat_strategy=data_cfg['mdat_strategy'],
        mdat_view=data_cfg['mdat_view'],
        with_meta=False,
        max_seq_len=model_cfg.max_position_embeddings,
        batch_size=data_cfg['batch_size'],
        num_devices=device_count
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
    print("Len debug")
    print(data.__len__())
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
    
    if simulate:
        print("=== SIMULATION MODE ===")
        print("Stable state_dict parameter names and shapes:")
        shapes = extract_state_dict_shapes(model.parameters_state_dict())
        for k, v in shapes.items():
            print(f"{k}: {v}")
        if dump_json:
            save_state_dict_json(model.parameters_state_dict(), dump_json)
            print(f"State dict shapes saved to {dump_json}")
        return
    
    # Handle checkpoint loading
    ckpt_path = None
    if not start_scratch:
        if ckpt_arg and os.path.exists(ckpt_arg):
            ckpt_path = ckpt_arg
        else:
            last_ckpt = Path(save_dir) / "last.ckpt"
            if last_ckpt.exists():
                print(f"Resuming training from {last_ckpt}")
                ckpt_path = str(last_ckpt)
            else:
                print("No checkpoint found, starting from scratch.")
    
    # Create timestamped checkpoint filename to avoid name clashes
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = train_cfg.get('checkpoint_name', 'model') # jerik
    run_name = cfg.get('wandb_run_name', 'training-run') # jerik
    if not debug_steps:
        checkpoint_name=f"{checkpoint_name}_{timestamp}"
    else:
        checkpoint_name=f"{checkpoint_name}_DEBUG_{debug_steps}_{timestamp}"
    # Setup logging
    wandb_logger = WandbLogger(
        name=f"{run_name}_{timestamp}",
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
    if debug_steps is not None:
        max_epochs=None
        max_steps=debug_steps
    else:
        max_steps=-1
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
        max_epochs=max_epochs,
        max_steps=max_steps
    )
    if _compile:
        try:
            if tok_cfg['varlen_strategy']:
                print("Trying compilation: dynamic sequence length, normal")
                try:
                    model=torch.compile(model, dynamic=True)
                except:
                    print("Trying compilation: dynamic sequence length, normal autotune")
                    model=torch.compile(model, dynamic=True)
            else:
                print("Trying compilation: fixed sequence length, max autotune")
                try:
                    model=torch.compile(model, mode="max-autotune")
                except:
                    print("Trying compilation: fixed sequence length, normale autotune")
                    model=torch.compile(model)
        except:
            print("Compilation failed! Running non-compiled model")

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
    
    #Rename last.ckpt with a better name
    try:
        os.rename(os.path.join(save_dir,'last.ckpt'), os.path.join(save_dir,f'{checkpoint_name}_last.ckpt'))
    except:
        print("Last.ckpt non trovato, probabilmente già salvato con nome corretto.")
    
if __name__ == '__main__':
    main()

