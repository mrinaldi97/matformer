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
class CheckpointResultsCallback(Callback):
    def __init__(self, csv_path, run_name, base_model_path):
        self.csv_path = Path(csv_path)
        self.run_name = run_name
        self.base_model_path = base_model_path
        self.epoch_metrics = []

    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        self.epoch_metrics.append({
            'epoch': trainer.current_epoch,
            'val_loss':     m.get('val/loss',     float('nan')),
            'val_accuracy': m.get('val/accuracy', float('nan')),
            'val_f1':       m.get('val/f1',       float('nan')),
            'train_loss':   m.get('train/loss',   float('nan')),
            'train_accuracy': m.get('train/accuracy', float('nan')),
            'train_f1':     m.get('train/f1',     float('nan')),
        })

    def on_fit_end(self, trainer, pl_module):
        import csv, json
        # Scalar-ify any remaining tensors
        clean = [{k: (v.item() if hasattr(v, 'item') else v) for k, v in e.items()} 
                 for e in self.epoch_metrics]
        last = clean[-1] if clean else {}
        row = {
            'run_name':       self.run_name,
            'base_model':     self.base_model_path,
            'timestamp':      datetime.now().isoformat(),
            'num_epochs':     trainer.current_epoch + 1,
            'per_epoch_metrics': json.dumps(clean),
            **{f'final_{k}': v for k, v in last.items() if k != 'epoch'}
        }
        write_header = not self.csv_path.exists()
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

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



def run_training(config_path, num_gpus=1, num_nodes=1, base_model_path=None, run_name=None, base_model_config=None, skip_save=False, results_csv_path=None):
    
    # 1. Load the config from the JSON
    with open(config_path,"r") as f:
        task_dict=json.load(f)
    
    # Base model path overrides the config
    if base_model_path is not None:
        print(f"Using {base_model_path} instead of {task_dict.get('pretrained_checkpoint')} (present in config)")
        task_dict['pretrained_checkpoint']=base_model_path
    else:
        base_model_path=task_dict['pretrained_checkpoint']
    # Detect device
    if torch.cuda.is_available():
        accelerator = 'gpu'
        device_string = 'cuda'
    elif torch.backends.mps.is_available():
        accelerator = device_string = 'mps'
    else:
        accelerator = device_string = 'cpu'    
        
    # Loading state dict and config (if present in checkpoint)
    checkpoint=torch.load(base_model_path, weights_only=False)
    
    if base_model_config is None:
        try:
            base_model_config = checkpoint['hyper_parameters']['config']
        except:
            raise Exception
  
    base_model_state_dict=checkpoint['state_dict']
    final_dict = vars(base_model_config).copy()
    inherited_fields = set(final_dict.keys())
    overridden_fields = set(task_dict.keys()) & inherited_fields
    new_fields = set(task_dict.keys()) - inherited_fields
    final_dict.update(task_dict)
    if True:
        if inherited_fields:
            print(f"Inherited {len(inherited_fields)} fields from checkpoint\n")
        if overridden_fields:
            print(f"Overridden {len(overridden_fields)} fields: {sorted(overridden_fields)}\n")
        if new_fields:
            print(f"Added {len(new_fields)} new fields: {sorted(new_fields)}\n")  

    config = load_and_validate_classification_config_from_dict(final_dict)
    
    task="sentence-level"
    
    
    if task == "sentence-level":
        ModelClass = TransformerWithClassificationHead
    elif task == "token-level":
        ModelClass = TransformerWithTokenClassificationHead
    else:
        raise ValueError(f"task must be 'sentence-level' or 'token-level', got {task}")
    tokenizer = MatformerTokenizer(
                config=config,
                tokenizer_type=config.tokenizer_type,
                tokenizer_name=config.tokenizer_name,
                varlen_strategy="unpadding"   
            )    
     # Instantiate the model   
    model = PL_ModelWrapper(
        ModelClass, 
        config=config, 
        tokenizer=tokenizer, 
        train_config=config.training, 
        device=device_string, 
        batch_size=config.training['batch_size'],
        training_step_type='classification'
        )    
    
    # Load the base model weights into the classification model
    missing,unexpected=model._load_stable_state_dict(base_model_state_dict)
    print(f"Missing: (it's normal)\n{missing}\n\nUnexpected (it's normal): {unexpected}")
    
    print(f"Loaded pretrained encoder from {base_model_path}")
    print(f"Model: {config.name}, {config.num_hidden_layers} layers")
    #print(f"Task: {task}, {num_features} classes")

    if config.freeze_base_model:
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
     

    save_dir = getattr(config,'save_dir')
    pl.seed_everything(getattr(config,'seed', 27))
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
  
    print("\n--- Labels distribution ---")
    print(train_loader.get_label_distribution())
    config.loss_type=config.training['loss']['type'] #Sporco... sovrascrivo il config del modello con la loss x classificazione
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
    if run_name is None:
        run_name = getattr(config, 'wandb_run_name', 'training-run')
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    if '_run' in run_name or 'run' in run_name.lower():
        final_run_name = run_name
    else:
        final_run_name = f"{run_name}_{timestamp}"
    
    wandb_logger = WandbLogger(
        name = final_run_name,
        project = getattr(config, 'wandb_project', 'matformer'),
        config = config
    )
    
    checkpoint = ModelCheckpoint(
        dirpath = save_dir,
        filename = checkpoint_name,
        save_top_k = 0,
        save_last = False,
        every_n_train_steps = getattr(config, "save_every_n_steps", None),
        enable_version_counter = True,
        save_on_train_epoch_end = getattr(config, "save_on_train_epoch_end", False)
    )
    results_callback = (
    CheckpointResultsCallback(csv_path=results_csv_path, run_name=final_run_name, base_model_path=base_model_path)
    if results_csv_path is not None else None
    )
    callbacks = [checkpoint] + ([results_callback] if results_callback else [])

    torch.set_float32_matmul_precision('high')
    
    strategy = DDPStrategy(gradient_as_bucket_view=True,static_graph=True,find_unused_parameters=False)
    trainer = pl.Trainer(
        logger = wandb_logger,
        callbacks = callbacks,
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
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print("\n--- Starting trainer.fit() ---")
    trainer.fit(model, dm)
    
    wandb.finish() 
    if not skip_save:
        print("\n--- Saving final model ---")
        final_save_path = save_classification_model(
            model=model,
            trainer=trainer,  
            config=config,
            save_dir=save_dir,
            name=f"{checkpoint_name}_final"
        )
        print(f"\nModel saved to: {final_save_path}")

def checkpoint_mode(config_path, checkpoints_dir, num_gpus=1, num_nodes=1, run_name_prefix=None, skip_save=False):
    """Iterate over all .ckpt files in checkpoints_dir, running fine-tuning for each
    and appending results to a shared CSV in the same directory.
    The CSV is named after the config file with a progressive suffix, and the config
    is copied alongside it — so each distinct run is fully reproducible and traceable.
    Already-processed checkpoints are skipped, making it safe to interrupt and resume."""
    import csv, shutil

    ckpt_files = sorted(Path(checkpoints_dir).glob("*.ckpt"))
    if not ckpt_files:
        print(f"No .ckpt files found in {checkpoints_dir}")
        return

    config_stem = Path(config_path).stem  # e.g. "sentiment_task"

    # Find the next free progressive index for this config name in this directory
    # e.g. sentiment_task_1.csv, sentiment_task_2.csv, ...
    # If an existing CSV already has a copy of this exact config, reuse it (resume case)
    existing_csvs = sorted(Path(checkpoints_dir).glob(f"{config_stem}_*.csv"))
    csv_path = None
    for candidate in existing_csvs:
        companion_config = candidate.with_suffix('.json')
        if companion_config.exists() and companion_config.read_bytes() == Path(config_path).read_bytes():
            csv_path = candidate
            print(f"Resuming existing run: {candidate.name}")
            break

    if csv_path is None:
        next_idx = len(existing_csvs) + 1
        csv_path = Path(checkpoints_dir) / f"{config_stem}_{next_idx}.csv"
        companion_config = csv_path.with_suffix('.json')
        shutil.copy(config_path, companion_config)
        print(f"New run: {csv_path.name} (config saved as {companion_config.name})")

    # Build set of already-processed checkpoint paths from existing CSV
    already_processed = set()
    if csv_path.exists():
        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                already_processed.add(row['base_model'])
        print(f"{len(already_processed)} checkpoints already processed, skipping.")

    print(f"Found {len(ckpt_files)} checkpoints: {[f.name for f in ckpt_files]}")

    for ckpt in ckpt_files:
        if str(ckpt) in already_processed:
            print(f"Skipping {ckpt.name} (already processed)")
            continue

        run_name = f"{run_name_prefix}_{ckpt.stem}" if run_name_prefix else ckpt.stem
        print(f"\n{'='*60}\nRunning checkpoint: {ckpt.name}  |  run: {run_name}\n{'='*60}")
        try:
            run_training(
            config_path=config_path,
            num_gpus=num_gpus,
            num_nodes=num_nodes,
            base_model_path=str(ckpt),
            run_name=run_name,
            results_csv_path=csv_path,
            skip_save=skip_save
        )
        except Exception as e:
            print(f"ERROR. {e}")

def main():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('config', type=str, help='Path to config file')
    parser.add_argument('--base_model_path', type=str, default=None, help='Path to base model (override the config)')
    parser.add_argument('--gpu', type=int, default=1, help='Number of GPUs (default: 1)')
    parser.add_argument('--nodes', type=int, default=1, help='Number of nodes (default: 1)')
    parser.add_argument('--run_name', type=str, default=None, help="Name of the run for logging")
    parser.add_argument('--skip_save',action='store_true', help="Don't save the model at the end of the run")
    parser.add_argument('--checkpoint_mode', action='store_true')
    parser.add_argument('--checkpoints_dir', type=str, default=None)
    parser.add_argument('--run_name_prefix', type=str, default=None)
    args = parser.parse_args()

    if args.checkpoint_mode:
        checkpoint_mode(args.config, args.checkpoints_dir, args.gpu, args.nodes, args.run_name_prefix, skip_save=args.skip_save)
    else:
        run_training(config_path=args.config, num_gpus=args.gpu, num_nodes=args.nodes, base_model_path=args.base_model_path, run_name=args.run_name, skip_save=args.skip_save)
if __name__ == "__main__":
    main()
