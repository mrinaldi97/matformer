"""
A function to train the classifier head using the matformer library
A JSON config file must be provided
But each argument can be overridden from the CLI

objective: making it the most general and easiest at the same time 
the target are linguists and researchers that should be able to use
their own data
"""
from typing import Literal
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

from matformer.transformer_blocks import BERTModel, TransformerWithEmbeddingHead, TransformerWithClassificationHead, TransformerWithTokenClassificationHead
from matformer.classification_training_data_loader import ClassificationTrainingDataLoader
from matformer.classification_data_module import ClassificationDataset, ClassificationDataModule
from matformer.tensors_dataclasses import PaddedTensor, UnpaddedTensor
from matformer.matformer_tokenizers import ByteLevelTokenizer,MatformerTokenizer
import torch.serialization as serialization

serialization.add_safe_globals([BERTModel, TransformerWithEmbeddingHead,TransformerWithClassificationHead, TransformerWithTokenClassificationHead, ModelConfig])

def extract_config_from_checkpoint(checkpoint_path):
  checkpoint = torch.load(checkpoint_path, weights_only=False)
  return checkpoint['hyper_parameters']['config']

# Build checkpoint key mapping
def build_checkpoint_mapping(num_layers=28):
    mapping = {
        'encoder.embed_tokens.weight': 'embed_tokens.weight',
        'encoder.blocks.norm.weight': 'blocks.norm.weight',
    }
    
    for i in range(num_layers):
        layer_maps = {
            f'encoder.blocks.layers.{i}.attn_norm.weight': f'blocks.layers.{i}.attn_norm.weight',
            f'encoder.blocks.layers.{i}.mlp_norm.weight': f'blocks.layers.{i}.mlp_norm.weight',
            f'encoder.blocks.layers.{i}.attn.qkv_proj.weight': f'blocks.layers.{i}.self_attn.packed_proj.weight',
            f'encoder.blocks.layers.{i}.attn.o_proj.weight': f'blocks.layers.{i}.self_attn.out_proj.weight',
            f'encoder.blocks.layers.{i}.mlp.gate_proj.weight': f'blocks.layers.{i}.mlp.gate_proj.weight',
            f'encoder.blocks.layers.{i}.mlp.up_proj.weight': f'blocks.layers.{i}.mlp.up_proj.weight',
            f'encoder.blocks.layers.{i}.mlp.down_proj.weight': f'blocks.layers.{i}.mlp.down_proj.weight',
        }
        mapping.update(layer_maps)
    
    return mapping
  
  
def load_model_from_checkpoint(checkpoint_path, config, num_classes, task, map_location='cpu', tokenizer=None):
    """
    Load classification model with pretrained encoder weights using PL_ModelWrapper.
    
    Args:
        checkpoint_path: Path to pretrained checkpoint
        config: ModelConfig object (or path to config JSON)
        num_classes: Number of output classes
        task: "sentence-level" or "token-level"
        map_location: Device to load model on
        tokenizer: Tokenizer name or instance
    """
    # Add classification-specific config
    config.num_labels = num_classes
    config.classifier_dropout_p = 0.1
    config.classifier_dropout_inplace = False
    
    # Select model class based on task
    if task == "sentence-level":
        ModelClass = TransformerWithClassificationHead
        config.pooling_type = 'cls'  # or 'mean'
    elif task == "token-level":
        ModelClass = TransformerWithTokenClassificationHead
    else:
        raise ValueError(f"task must be 'sentence-level' or 'token-level', got {task}")
    
    # Build checkpoint key mapping for pretrained weights
    checkpoint_mapping = build_checkpoint_mapping(num_layers=config.num_hidden_layers)
    
    # Load using PL_ModelWrapper
    model, config = PL_ModelWrapper.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        ModelClass=ModelClass,
        config=config,
        map_location=map_location,
        tokenizer=tokenizer,
        varlen_strategy='padding',
        external_mapping=checkpoint_mapping  # Handle key name mismatch
    )
    
    print(f"Loaded pretrained encoder from {checkpoint_path}")
    print(f"Model: {config.name}, {config.num_hidden_layers} layers")
    print(f"Task: {task}, {num_classes} classes")
    
    # === VERIFICATION ===
    print(f"\n--- Loading Verification ---")
    
    # 1. Check encoder has non-random weights
    first_layer_weight = model.model.encoder.blocks.layers[0].self_attn.packed_proj.inner.weight
    weight_std = first_layer_weight.std().item()
    weight_mean = first_layer_weight.abs().mean().item()
    
    print(f"First layer weight stats: mean={weight_mean:.4f}, std={weight_std:.4f}")
    assert weight_std > 0.01, "Encoder weights look uninitialized (std too low)"
    
    # 2. Verify forward pass works
    dummy_input = torch.randint(0, config.vocab_size, (2, 64)).to(map_location)
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Forward pass output shape: {output.shape}")
        expected_shape = (2, num_classes) if task == "sentence-level" else (2, 64, num_classes)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    print(f"Loading verified successfully\n")
    
    return model
  
# taken from models.py
def load_tokenizer(config=None, tokenizer="bytes", varlen_strategy=None):
  tokenizer = MatformerTokenizer(
            config=config,
            tokenizer=tokenizer,
            tokenizer_name=tokenizer,
            varlen_strategy=varlen_strategy
        )     
  return tokenizer
            
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
    
    train_loader = ClassificationTrainingDataLoader(filepath="utils/data.csv", text_column="text", label_column="is_profane")

    checkpoint_path = "/mnt/llmdata/data/FINALE_32768.ckpt"
    ### config
    config = extract_config_from_checkpoint(checkpoint_path)
    # Add classification-specific fields
    config.classifier_dropout_p = 0.1
    config.classifier_dropout_inplace = False
    
    tokenizer = load_tokenizer(config=config)
    
    model = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        config=config,
        num_classes=train_loader.get_num_labels(),
        task="sentence-level",
        map_location="cuda",
        tokenizer=tokenizer
    )
    
    print("model loaded")

    dm = ClassificationDataModule(
        data_loader=train_loader,
        tokenizer=tokenizer,
        max_seq_len=1024, #cfg.max_seq_len,
        pad_token_id=config.pad_token_id , 
        batch_size=32,
        num_devices=1
    )
        
    print("data loader ready!")
    
    
    
    ### Test dataloader
    ##dataloader = dm.train_dataloader()
    ##print(f"Dataloader created, batches: {len(dataloader)}")
    ##
    ### Get one batch
    ##batch = next(iter(dataloader))
    ##print(f"\nBatch keys: {batch.keys()}")
    ##print(f"Input IDs shape: {batch['input_ids'].shape}")
    ##print(f"Attention mask shape: {batch['attention_mask'].shape}")
    ##print(f"Labels shape: {batch['labels'].shape}")
    ##print(f"Labels unique values: {batch['labels'].unique()}")
    ##
    ### Test model forward pass
    ##print("\nTesting model forward pass...")
    ##model = model.to(device).eval()
    ##
    ##with torch.no_grad():
    ##    # Create PaddedTensor from batch
    ##    sequence = PaddedTensor(
    ##        tensor=batch['input_ids'].to(device),
    ##        padding_mask=(batch['attention_mask'] == 0).to(device)
    ##    )
    ##    
    ##    # Test hidden states output
    ##    hidden_states = model(sequence, return_type='hidden')
    ##    print(f"Hidden states type: {type(hidden_states)}")
    ##    if isinstance(hidden_states, UnpaddedTensor):
    ##        print(f"Hidden states tensor shape: {hidden_states.tensor.shape}")
    ##    else:
    ##        print(f"Hidden states shape: {hidden_states.tensor.shape}")
    ##    
    ##    # Extract [CLS] representations
    ##    if isinstance(hidden_states, UnpaddedTensor):
    ##        cls_hidden = []
    ##        cu_seqlens = hidden_states.cu_seqlens
    ##        for i in range(len(cu_seqlens) - 1):
    ##            start_idx = cu_seqlens[i]
    ##            cls_hidden.append(hidden_states.tensor[start_idx])
    ##        cls_hidden = torch.stack(cls_hidden)
    ##    else:
    ##        cls_hidden = hidden_states.tensor[:, 0, :]
    ##    
    ##    print(f"CLS hidden shape: {cls_hidden.shape}")
    ##    print(f"Expected: [batch_size={batch['input_ids'].shape[0]}, hidden_size={cfg.hidden_size}]")
    ##
    ##print("\nAll tests passed!")


if __name__ == "__main__":
    main()
