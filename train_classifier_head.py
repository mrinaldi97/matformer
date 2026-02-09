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

serialization.add_safe_globals([BERTModel, TransformerWithEmbeddingHead,TransformerWithClassificationHead, TransformerWithTokenClassificationHead, ModelConfig])

def load_model_from_checkpoint(checkpoint_path, config, train_config, num_features, 
                               task, map_location='cpu', tokenizer=None):
    """Load classification model with pretrained encoder weights."""
    
    # Add classification-specific config
    config.classifier_dropout_p = 0.1
    config.classifier_dropout_inplace = False
    
    if task == "sentence-level":
        ModelClass = TransformerWithClassificationHead
    elif task == "token-level":
        ModelClass = TransformerWithTokenClassificationHead
    else:
        raise ValueError(f"task must be 'sentence-level' or 'token-level', got {task}")
    
    # Load model
    model, config = PL_ModelWrapper.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        ModelClass=ModelClass,
        config=config,
        train_config=train_config,
        map_location=map_location,
        tokenizer=tokenizer,
        varlen_strategy='padding',
        external_mapping=None,
        num_features=num_features
    )
    
    print(f"Loaded pretrained encoder from {checkpoint_path}")
    print(f"Model: {config.name}, {config.num_hidden_layers} layers")
    print(f"Task: {task}, {num_features} classes")

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
        expected_shape = (2, num_features) if task == "sentence-level" else (2, 64, num_features)
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
            
def load_classification_config(
    task_config_path: str,
    checkpoint_path: str = None,
    inherit_from_checkpoint: bool = True
) -> ClassificationConfig:
    """
    Load classification config with optional checkpoint inheritance.
    
    Args:
        task_config_path: Path to task config JSON
        checkpoint_path: Path to pretrained checkpoint (optional if in task config)
        inherit_from_checkpoint: If True, fill missing fields from checkpoint config
        validate_compatibility: If True, validate config compatibility
    
    Returns:
        ClassificationConfig with merged settings
    """
    # Load task config
    if not Path(task_config_path).exists():
        raise FileNotFoundError(f"Config file not found: {task_config_path}")
    
    with open(task_config_path, 'r') as f:
        task_dict = json.load(f)
    
    # Determine checkpoint path
    if checkpoint_path is None:
        checkpoint_path = task_dict.get('pretrained_checkpoint')
    
    if not checkpoint_path:
        raise ValueError("checkpoint_path must be provided or in config as 'pretrained_checkpoint'")
    
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

        # Print inheritance info
        if inherited_fields:
            print(f"Inherited {len(inherited_fields)} fields from checkpoint\n")
        if overridden_fields:
            print(f"Overridden {len(overridden_fields)} fields: {sorted(overridden_fields)}\n")
        if new_fields:
            print(f"Added {len(new_fields)} new fields: {sorted(new_fields)}\n")
    else:
        # Use only task config
        final_dict = task_dict.copy()
    
    # Ensure pretrained_checkpoint is set
    final_dict['pretrained_checkpoint'] = checkpoint_path
    
    # Parse into ClassificationConfig
    config = load_and_validate_classification_config_from_dict(final_dict)
    
    return config


def extract_config_from_checkpoint(checkpoint_path):
  checkpoint = torch.load(checkpoint_path, weights_only=False)
  return checkpoint['hyper_parameters']['config']  

def main():
  
    print("\nINIZIO")
    print(torch.cuda.memory_allocated() / 1e9, "GB allocated")
    print(torch.cuda.memory_reserved() / 1e9, "GB reserved")

    checkpoint_path = "/mnt/llmdata/data/FINALE_32768.ckpt"
    config_path = "configs/classification_head/config.json"
    start_scratch = True
  
    # config
    print("\n --- Config ---")
    config = load_classification_config(config_path, checkpoint_path)
    print("\n"+ "-"*40+"\n")
    
    save_dir = getattr(config, 'save_dir', './checkpoints')
    pl.seed_everything(getattr(config, 'seed', 27))
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
    tokenizer = load_tokenizer(config=config)

    print("\nLoading model..")    
    model = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        config=config,
        train_config=getattr(config,'training'),
        num_features=train_loader.get_num_labels(),
        task="sentence-level",
        map_location="cuda",
        tokenizer=tokenizer
    )
    
    print("\nLoading data loader..")
    dm = ClassificationDataModule(
        data_loader=train_loader,
        val_data_loader = val_loader,
        tokenizer=tokenizer,
        max_seq_len=1024, #cfg.max_seq_len,
        pad_token_id=config.pad_token_id , 
        batch_size=getattr(config,"training")["batch_size"],
        num_devices=1
    )   
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = getattr(config, 'name', 'name')
    run_name = getattr(config, 'wandb_run_name', 'training-run')
    # Setup logging
    wandb_logger = WandbLogger(
        name=f"{run_name}_{timestamp}",
        project=getattr(config, 'wandb_project', 'matformer'),
        config=config
    )
    
    checkpoint = ModelCheckpoint(
        dirpath=save_dir,
        filename=checkpoint_name,
        save_top_k=1,
        save_last=True,
        every_n_train_steps=getattr(config, "save_every_n_steps", None),
        enable_version_counter=True,
        save_on_train_epoch_end=True
    )
    torch.set_float32_matmul_precision('high')
    
    strategy=DDPStrategy(gradient_as_bucket_view=True,static_graph=True,find_unused_parameters=False)
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint],
        precision=getattr(config, 'precision', 'bf16-mixed'),
        gradient_clip_val=getattr(config, 'gradient_clip_val', 1),
        accelerator=accelerator,
        devices=1,
        log_every_n_steps=10,
        accumulate_grad_batches=getattr(config, 'accumulate_grad_batches', 1),
        default_root_dir=save_dir,
        max_epochs=getattr(config, 'max_epochs',5),
        max_steps=getattr(config, 'max_steps',-1),
        strategy=strategy,
        num_nodes=1
    )
    
    print("\nTRAINER")
    print(torch.cuda.memory_allocated() / 1e9, "GB allocated")
    print(torch.cuda.memory_reserved() / 1e9, "GB reserved")
    
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
                
    # After dm initialization, before trainer.fit():
    print("\n=== DATA BATCH CHECK ===")
    dm.setup()
    sample_batch = next(iter(dm.train_dataloader()))
    print(f"Batch input_ids shape: {sample_batch['input_ids'].shape}")
    print(f"Batch labels shape: {sample_batch['labels'].shape}")
    print(f"Memory after batch: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Test forward pass manually
    print("\n=== MANUAL FORWARD TEST ===")
    model = model.cuda()
    print(f"Memory after model.cuda(): {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    with torch.no_grad():
        sample_batch = move_batch_to_device(sample_batch, 'cuda')  # FIX HERE
        print(f"Memory after batch.cuda(): {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        output = model(sample_batch['input_ids'])
        print(f"Output shape: {output.shape}")
        print(f"Memory after forward: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print("\n=== Starting trainer.fit() ===")
    
    trainer.fit(model, dm, ckpt_path=ckpt_path)
    
def move_batch_to_device(batch, device):
    moved = {}
    for k, v in batch.items():
        if isinstance(v, PaddedTensor):
            moved[k] = PaddedTensor(
                tensor=v.tensor.to(device),
                padding_mask=v.padding_mask.to(device)
            )
        else:
            moved[k] = v.to(device)
    return moved    


if __name__ == "__main__":
    main()
