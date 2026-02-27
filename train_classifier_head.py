"""
A function to train the classifier head using the matformer library
A JSON config file must be provided
But each argument can be overridden from the CLI

objective: making it the most general and easiest at the same time 
the target are linguists and researchers that should be able to use
their own data
"""
from typing import Literal
import sys
import argparse, json, torch, pytorch_lightning as pl, wandb
from pathlib import Path
from importlib import import_module
from transformers import AutoTokenizer
from matformer.matformer_tokenizers import MatformerTokenizer
from matformer.data_module import MatformerDataModule
from matformer.model_config import ModelConfig, LayerConfig, ClassificationConfig, TokenClassificationConfig, load_and_validate_classification_config_from_dict
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

serialization.add_safe_globals([
    BERTModel, TransformerWithEmbeddingHead, TransformerWithClassificationHead, 
    TransformerWithTokenClassificationHead, ModelConfig, ClassificationConfig, 
    TokenClassificationConfig, LayerConfig
])

def load_model_from_checkpoint(checkpoint_path, config, train_config, 
                               task, freeze_base_model=True, map_location='cpu', tokenizer=None):
    """Load classification model with pretrained encoder weights."""
    
    if task == "sentence-level":
        ModelClass = TransformerWithClassificationHead
    elif task == "token-level":
        ModelClass = TransformerWithTokenClassificationHead
    else:
        raise ValueError(f"task must be 'sentence-level' or 'token-level', got {task}")
    
    model, config = PL_ModelWrapper.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        ModelClass=ModelClass,
        config=config,
        train_config=train_config,
        map_location=map_location,
        tokenizer=tokenizer,
        varlen_strategy='unpadding',
        external_mapping=None,
        training_step_type='classification'
    )
    
    print(f"Loaded pretrained encoder from {checkpoint_path}")
    print(f"Model: {config.name}, {config.num_hidden_layers} layers")
    #print(f"Task: {task}, {num_features} classes")

    if freeze_base_model:
        print("\n--- Freezing encoder ---")
        model.model.freeze_encoder()

        # Verify freezing
        encoder_params = sum(p.numel() for p in model.model.encoder.parameters())
        encoder_trainable = sum(p.numel() for p in model.model.encoder.parameters() if p.requires_grad)
        head_trainable = sum(p.numel() for p in model.model.classification_head.parameters() if p.requires_grad)
        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Encoder params: {encoder_params:,} (trainable: {encoder_trainable:,})")
        print(f"Head params: {head_trainable:,}")
        print(f"Total trainable: {total_trainable:,}")

        assert encoder_trainable == 0, "Encoder should have 0 trainable params"
        assert head_trainable > 0, "Classification head should be trainable"
    
    return model
  

            
def load_classification_config(
    task_config_path: str,
    inherit_from_checkpoint: bool = True
) -> ClassificationConfig:
    """
    Load classification config with optional checkpoint inheritance.
    
    Args:
        task_config_path: Path to task config JSON
        inherit_from_checkpoint: If True, fill missing fields from checkpoint config
    
    Returns:
        ClassificationConfig with merged settings
    """
    # Load task config
    if not Path(task_config_path).exists():
        raise FileNotFoundError(f"Config file not found: {task_config_path}")
    
    with open(task_config_path, 'r') as f:
        task_dict = json.load(f)
        
    checkpoint_path = task_dict.get("pretrained_checkpoint", None)
    
    if checkpoint_path is None:
        raise ValueError("checkpoint_path must be provided in config as 'pretrained_checkpoint'")
    
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint config if inheriting
    checkpoint_config = None
    if inherit_from_checkpoint:
        print(f"Loading checkpoint config from: {checkpoint_path}")
        checkpoint_config = extract_config_from_checkpoint(checkpoint_path)
    
    # Build final config dict
    if inherit_from_checkpoint and checkpoint_config:
        # Start with checkpoint config as base
        final_dict = vars(checkpoint_config).copy()
        
        # Track what gets overridden
        inherited_fields = set(final_dict.keys())
        overridden_fields = set(task_dict.keys()) & inherited_fields
        new_fields = set(task_dict.keys()) - inherited_fields

        # Override with task config (task config takes priority)
        final_dict.update(task_dict)

        if inherited_fields:
            print(f"Inherited {len(inherited_fields)} fields from checkpoint\n")
        if overridden_fields:
            print(f"Overridden {len(overridden_fields)} fields: {sorted(overridden_fields)}\n")
        if new_fields:
            print(f"Added {len(new_fields)} new fields: {sorted(new_fields)}\n")
    else:
        # Use only task config
        final_dict = task_dict.copy()
    
    final_dict['pretrained_checkpoint'] = checkpoint_path
    config = load_and_validate_classification_config_from_dict(final_dict)
    return config


def extract_config_from_checkpoint(checkpoint_path):
  checkpoint = torch.load(checkpoint_path, weights_only=False)
  return checkpoint['hyper_parameters']['config']  

def save_classification_model(model, trainer, config, save_dir, name="final_model"):
    """Save classification model after training."""
    save_path = Path(save_dir) / name
    save_path.mkdir(parents=True, exist_ok=True)
 
    full_path = save_path / "final.ckpt"
    trainer.save_checkpoint(full_path)
    print(f"Saved Lightning checkpoint: {full_path}")
    
    # State dict
    state_dict_path = save_path / "model_state.pt"
    torch.save({
        'model_state_dict': model.model.state_dict(),
        'config': config,
        'num_labels': model.model.num_features
    }, state_dict_path)
    print(f"Saved state dict: {state_dict_path}")
    
    # Classification head only (for transfer to other encoders)
    head_path = save_path / "classification_head.pt"
    torch.save({
        'head_state_dict': model.model.classification_head.state_dict(),
        'num_labels': model.model.num_features,
        'pooling_type': model.model.pooling_type,
        'dropout_p': model.model.classifier_dropout_p
    }, head_path)
    print(f"Saved classification head: {head_path}")
    
    return save_path

def run_training(config_path, start_scratch=True, num_gpus=1, num_nodes=1, checkpoint_path=None):  
    print("\n --- Config ---")
    config = load_classification_config(config_path)
    print("\n")    
    
    save_dir = getattr(config,'save_dir')
    pl.seed_everything(getattr(config,'seed', 27))
    
    # Detect device
    if torch.cuda.is_available():
        accelerator = 'gpu'
        device_string = 'cuda'
    elif torch.backends.mps.is_available():
        accelerator = device_string = 'mps'
    else:
        accelerator = device_string = 'cpu'
    
    train_loader = ClassificationTrainingDataLoader(
      filepath=getattr(config,"data")["train_file"],
      text_column=getattr(config,"data")["text_label"],
      label_column=getattr(config,"data")["target_label"]
    )
    val_loader = (
        ClassificationTrainingDataLoader(
            filepath=getattr(config,"data")["val_file"],
            text_column=getattr(config,"data")["text_label"],
            label_column=getattr(config,"data")["target_label"]
        )
        if "val_file" in getattr(config, "data", {})
        else None
    )
    tokenizer = MatformerTokenizer(
                config=config,
                tokenizer_type=config.tokenizer_type,
                tokenizer_name=config.tokenizer_name,
                varlen_strategy="unpadding"   
            )     
    print("\n--- Labels distribution ---")
    print(train_loader.get_label_distribution())
    if config.training.get('loss', {}).get('class_weights', False) == "auto":
      class_weights = train_loader.get_class_weights(strategy='inverse_frequency')
      config.training['loss']['class_weights'] = class_weights
      print(f"Class weights: {class_weights}")
      print()
    
    freeze_base_model = getattr(config, 'freeze_base_model', True)
    #if freeze_base_model and :
    #  config["batch_size"] = 32
    train_config=config.training
    print("Debug 1 train_config")
    print(train_config)
    print("\nLoading model..")    
    if checkpoint_path is not None:
        print(f"Using {checkpoint_path} (from argument) instead of {getattr(config,'pretrained_checkpoint')} (from config)")
    else:
        checkpoint_path=getattr(config,'pretrained_checkpoint')
    model = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        config=config,
        train_config=train_config,
        task="sentence-level",
        map_location="cuda",
        tokenizer=tokenizer,
        freeze_base_model = freeze_base_model
    )
    
    print("\nLoading data loader..")
    dm = ClassificationDataModule(
        data_loader=train_loader,
        val_data_loader = val_loader,
        tokenizer=tokenizer,
        max_seq_len=1024, #cfg.max_seq_len,
        pad_token_id=config.pad_token_id , 
        batch_size=getattr(config,"training")["batch_size"],
        num_workers= getattr(config,"data")["num_workers"],
        varlen_strategy = "unpadding"
    )   
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = getattr(config, 'name', 'name')
    run_name = getattr(config, 'wandb_run_name', 'training-run')
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    #don't add timestamp if run_name already has run identifier
    if '_run' in run_name or 'run' in run_name.lower():
        final_run_name = run_name
    else:
        final_run_name = f"{run_name}_{timestamp}"
    
    # Setup logging
    wandb_logger = WandbLogger(
        name = final_run_name,
        project = getattr(config, 'wandb_project', 'matformer'),
        config = config
    )
    
    checkpoint = ModelCheckpoint(
        dirpath = save_dir,
        filename = checkpoint_name,
        save_top_k = 1,
        save_last = True,
        every_n_train_steps = getattr(config, "save_every_n_steps", None),
        enable_version_counter = True,
        save_on_train_epoch_end = getattr(config, "save_on_train_epoch_end", False)
    )
    torch.set_float32_matmul_precision('high')
    
    strategy = DDPStrategy(gradient_as_bucket_view=True,static_graph=True,find_unused_parameters=False)
    trainer = pl.Trainer(
        logger = wandb_logger,
        callbacks = [checkpoint],
        precision = getattr(config, 'precision', 'bf16-mixed'),
        gradient_clip_val = getattr(config, 'training')["gradient_clip_val"],
        accelerator = accelerator,
        devices = num_gpus,
        log_every_n_steps = 10,
        accumulate_grad_batches = getattr(config, 'accumulate_grad_batches', 1),
        default_root_dir = save_dir,
        max_epochs = getattr(config, 'training')["max_epochs"],
        max_steps = getattr(config, 'max_steps',-1),
        strategy = strategy,
        num_nodes = num_nodes
    )
    
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

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print("\n--- Starting trainer.fit() ---")
    trainer.fit(model, dm, ckpt_path=ckpt_path)
    
    wandb.finish() 
    
    print("\n--- Saving final model ---")
    final_save_path = save_classification_model(
        model=model,
        trainer=trainer,  # Pass trainer
        config=config,
        save_dir=save_dir,
        name=f"{checkpoint_name}_final"
    )
    print(f"\nModel saved to: {final_save_path}")

import argparse

def main():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('config', type=str, help='Path to config file')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to checkpoint (override the config)')
    parser.add_argument('--gpu', type=int, default=1, help='Number of GPUs (default: 1)')
    parser.add_argument('--nodes', type=int, default=1, help='Number of nodes (default: 1)')
    
    args = parser.parse_args()
    
    start_scratch = args.checkpoint_path is None
    run_training(args.config, start_scratch, args.gpu, args.nodes, checkpoint_path=args.checkpoint_path)

if __name__ == "__main__":
    main()
