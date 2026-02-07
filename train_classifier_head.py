"""
A function to train the classifier head using the matformer library
A JSON config file must be provided
But each argument can be overridden from the CLI

objective: making it the most general and easiest at the same time 
the target are linguists and researchers that should be able to use
their own data
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
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.profilers import AdvancedProfiler

# from pytorch_lightning.plugins import DDPPlugin
import math, os
from datetime import datetime

from inference import load_inference_model as load_from_checkpoint
from matformer.transformer_blocks import BERTModel
from matformer.classification_training_data_loader import ClassificationTrainingDataLoader
from matformer.classification_data_module import ClassificationDataset, ClassificationDataModule

def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def apply_overrides(cfg, overrides):
    for key, val in overrides.items():
        keys = key.split(".")
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
    parser = argparse.ArgumentParser(description="Train a Matformer model")
    parser.add_argument(
        "--config", type=str, help="Path to single combined config file"
    )
    parser.add_argument("--model_config", type=str, help="Path to model_config.json")
    parser.add_argument("--training_config", type=str, help="Path to train_config.json")
    parser.add_argument("--data_config", type=str, help="Path to data_config.json")
    parser.add_argument(
        "--tokenizer_config", type=str, help="Path to tokenizer_config.json"
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Override config parameters as key=value pairs",
    )
    parser.add_argument("--gpu", type=int, default=1, help="Number of GPU(s)")
    parser.add_argument("--nodes", type=int, default=1, help="Number of Node(s)")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint file"
    )
    parser.add_argument(
        "--start-from-scratch", action="store_true", help="Start training from scratch"
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Instantiate model and print state_dict shapes, then exit",
    )
    parser.add_argument(
        "--dump-json",
        type=str,
        default=None,
        help="Path to dump JSON state dict shapes",
    )
    parser.add_argument(
        "--debug-steps",
        type=int,
        default=None,
        help="If you choose this, train for one epoch on this number of steps",
    )
    parser.add_argument(
        "--compile", action="store_true", help="Torch.compile the whole model"
    )
    parser.add_argument(
        "--load-mode",
        type=str,
        choices=["full", "weights_only", "weights_and_optimizer"],
        default="full",
        help="Checkpoint loading strategy",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=[
            "16-mixed",
            "bf16-mixed",
            "32",
            "16",
            "bf16",
            "32-true",
            "bf16-true",
            "16-true",
            "64-true",
            "transformer-engine",
        ],
        default="bf16-mixed",
        help="precision",
    )
    args = parser.parse_args()

    separate_configs = {
        "model_config": args.model_config,
        "training_config": args.training_config,
        "data_config": args.data_config,
        "tokenizer_config": args.tokenizer_config,
    }
    separate_count = sum(1 for v in separate_configs.values() if v is not None)

    # Enforce exclusive requirement
    if args.config is not None and separate_count > 0:
        parser.error(
            "Cannot specify both --config and individual config files. Choose ONE approach."
        )

    if args.config is None and separate_count == 0:
        parser.error(
            "Must specify either --config OR all four individual config files (--model_config, --training_config, --data_config, --tokenizer_config)"
        )

    if args.config is None and separate_count != 4:
        missing = [k for k, v in separate_configs.items() if v is None]
        parser.error(
            f"Missing {len(missing)} individual config file(s): {', '.join(missing)}"
        )

    # Build config_paths dictionary
    if args.config is not None:
        config_paths = args.config
    else:
        config_paths = {
            "model_config": args.model_config,
            "training": args.training_config,
            "data": args.data_config,
            "tokenizer": args.tokenizer_config,
        }

    overrides = {}
    for item in args.override:
        try:
            k, v = item.split("=", 1)  # Split on first '=' only
            overrides[k] = v
        except ValueError:
            parser.error(f"Override '{item}' must be in key=value format")

    return (
        config_paths,
        overrides,
        args.gpu,
        args.checkpoint,
        args.start_from_scratch,
        args.simulate,
        args.dump_json,
        args.debug_steps,
        args.compile,
        args.nodes,
        args.load_mode,
        args.precision,
    )


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
    with open(path, "w") as f:
        json.dump(shapes, f, indent=2)


def load_and_prepare_configs(config_paths, overrides):
    """
    Loads multiple separate JSON configs, merges them, applies overrides,
    and derives any dependent configuration properties (like is_causal).
    """
    if isinstance(config_paths, dict):  # Multiple config files
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
            "tokenizer": tok_cfg_dict,
        }

    else:  # Single config file
        json_config = load_config(config_paths)
        cfg = {
            "model_class": json_config["model_class"],
            "save_dir": json_config.pop("save_dir", "./checkpoints"),
            "wandb_project": json_config.pop("wandb_project", "default_project"),
            "wandb_run_name": json_config.pop("wandb_run_name", "default_run"),
            "model_config": json_config["model_config"],
            "training": json_config["training"],
            "data": json_config["data"],
            "tokenizer": json_config["tokenizer"],
            "training_objective": json_config.pop("training_objective", None),
            "is_causal": json_config.pop("is_causal", None),
        }

    cfg = apply_overrides(cfg, overrides)

    model_class = cfg["model_class"]
    if getattr(cfg["model_config"], "training_objective", None) is None:
        cfg["model_config"]["training_objective"] = (
            "autoregressive" if model_class == "Autoregressive_Model" else "masked"
        )
    if getattr(cfg["model_config"], "is_causal", None) is None:
        cfg["model_config"]["is_causal"] = (
            True if model_class == "Autoregressive_Model" else False
        )
    cfg["model_config"]["tokenizer_type"] = cfg["tokenizer"]["type"]
    cfg["model_config"]["tokenizer_name"] = cfg["tokenizer"]["pretrained_name"]
    if "wanted_from_strategy" not in cfg["data"].keys():
        cfg["data"]["wanted_from_strategy"] = "chunked_tokens"
    model_config_dict_clean = cfg["model_config"]
    train_config_dict = cfg["training"]
    data_config_dict = cfg["data"]
    tok_config_dict = cfg["tokenizer"]

    return (
        model_config_dict_clean,
        train_config_dict,
        data_config_dict,
        tok_config_dict,
        cfg,
    )


def main():
    print("hi!")
    # load modello
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok_arg = 'bytes'
    ModelClass = BERTModel
    checkpoint_path = "/mnt/llmdata/data/FINALE_32768.ckpt"
    model, cfg = load_from_checkpoint(checkpoint_path=checkpoint_path, ModelClass=ModelClass, map_location=device, tokenizer=tok_arg)
    print("loaded")
    
    train_loader = ClassificationTrainingDataLoader(filepath="utils/data.csv", text_column="text", label_column="is_profane")

    dm = ClassificationDataModule(
        data_loader=train_loader,
        tokenizer=model.tokenizer,
        max_seq_len=1024, #cfg.max_seq_len,
        pad_token_id=model.tokenizer.pad_token_id, 
        batch_size=32,
        num_devices=1
    )
        
    print("data loader ready!")
    dm.setup()
    print(dm.__len__())
    print(dm.params())


if __name__ == "__main__":
    main()
