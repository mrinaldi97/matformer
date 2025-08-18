import sys
sys.path.append('../') # Da sostituire con pyproject.toml eccetera ok per ora
import sys, json, argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.optim import AdamW
from transformers import get_scheduler
from functools import partial
import random
from matformer.model_config import ModelConfig
from autoencoders import TransCharAutoencoder
from datasets import JSONLDataset, RandomDataset, generate_random_batch, collate_fn

class TransCharAutoencoderLightning(pl.LightningModule):
    def __init__(self, encoder_config, decoder_config, train_config):
        super().__init__()
        self.save_hyperparameters()
        self.model = TransCharAutoencoder(encoder_config, decoder_config, device='cuda')
        
        ## Inizio roba che di base non serve più con il training senza curriculum
        self.curriculumStep = train_config.get("curriculum_start", 0)
        self.patience = 0
        self.saveit = False
        self.curriculumTarget = train_config.get("curriculum_end", encoder_config.max_position_embeddings)
        self.max_curriculum_steps = train_config.get("curriculum_max_steps", 1_000_000)
        self.curriculum_patience = train_config.get("curriculum_patience", 50)
        self.until_convergence = train_config.get("until_convergence", None)
        self.target_loss = train_config.get("target_loss", 0.001)
        self.post_convergence_steps = train_config.get("post_convergence_steps", 500)
        self.post_convergence_counter = 0
        self.converged = False
        ## Fine roba
        
        self.encoder_config = encoder_config
        self.train_config = train_config
        self._acc_len_bins = {"patches": {}, "rand": {}}

    def forward(self, input_ids, lengths):
        return self.model(input_ids, lengths, real_mode=False)

    def _compute_accuracy(self, logits, targets, pad):
        with torch.no_grad():
            #print(f" Logits shape: {logits.shape}, Targets shape: {targets.shape}")
            try:
                pred = logits.argmax(-1)
                mask = targets != pad
                correct = (pred[mask] == targets[mask]).sum().float()
                total = mask.sum().clamp(min=1).float()
                return (correct / total).detach()
            except:
                return 0
    def training_step(self, batch, batch_idx):
        if self.train_config.get("curriculum_mode", False):
            return self.training_step_curriculum(batch, batch_idx)
        
        input_ids, sequence_lengths = batch
        char_logits, seqlen_logits = self(input_ids, sequence_lengths)
        
        #seqlen_targets = (sequence_lengths.tensor.squeeze(-1)-1).long()
        #seqlen_loss = nn.functional.cross_entropy(seqlen_logits.tensor, seqlen_targets)
        
        char_targets = input_ids.tensor.view(-1).long()
        char_logits_flat = char_logits.tensor.view(-1, char_logits.tensor.size(-1))
        char_loss = nn.functional.cross_entropy(char_logits_flat, char_targets, ignore_index=self.encoder_config.pad_token_id)
        
        #loss = 2 * seqlen_loss + char_loss
        loss=char_loss
        acc = self._compute_accuracy(char_logits.tensor, input_ids.tensor, self.encoder_config.pad_token_id)
        #self.log("train/seqlen_loss",seqlen_loss)
        #self.log("train/char_loss",char_loss)        
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", acc, prog_bar=True)
        return loss
    def _validation_step(self, batch, batch_idx, dataloader_idx=0):
        pass
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        ds = "patches" if dataloader_idx == 0 else "rand"
        input_ids, sequence_lengths = batch
        char_logits, seqlen_logits = self(input_ids, sequence_lengths)
        
        #seqlen_targets = (sequence_lengths.tensor.squeeze(-1)-1).long()
        #seqlen_loss = nn.functional.cross_entropy(seqlen_logits.tensor, seqlen_targets)
        
        char_targets = input_ids.tensor.view(-1).long()
        char_logits_flat = char_logits.tensor.view(-1, char_logits.tensor.size(-1))
        char_loss = nn.functional.cross_entropy(char_logits_flat, char_targets, ignore_index=self.encoder_config.pad_token_id)
        
        #loss = 2 * seqlen_loss + char_loss
        loss=char_loss
        acc = self._compute_accuracy(char_logits.tensor, input_ids.tensor, self.encoder_config.pad_token_id)
        
        self.log(f"val/{ds}/loss", loss, prog_bar=True, add_dataloader_idx=False)
        self.log(f"val/{ds}/acc", acc, prog_bar=True, add_dataloader_idx=False)
        
        # per-length acc
        with torch.no_grad():
            true_len = sequence_lengths.tensor.squeeze(-1).tolist()
            pred = char_logits.tensor.argmax(-1)
            for i, L in enumerate(true_len):
                L = max(1, int(L))
                t, p = input_ids.tensor[i, :L], pred[i, :L]
                hit = (p == t).float().mean().item()
                if L not in self._acc_len_bins[ds]:
                    self._acc_len_bins[ds][L] = [0.0, 0]
                self._acc_len_bins[ds][L][0] += hit
                self._acc_len_bins[ds][L][1] += 1
        
        return {"loss": loss.detach(), "acc": acc.detach()}

    def on_validation_epoch_start(self):
        self._acc_len_bins = {"patches": {}, "rand": {}}

    def _on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        for metric, key in [("loss", "l"), ("acc", "a")]:
            p, r = metrics.get(f"val/patches/{metric}"), metrics.get(f"val/rand/{metric}")
            if p is not None and r is not None:
                self.log(f"val/avg/{metric}", 0.5 * (p + r))
        
        for ds in ("patches", "rand"): #Calcolo dell'accuracy per tutte le sequence length
            for L, (s, n) in sorted(self._acc_len_bins[ds].items()):
                v = s / max(n, 1)
                self.log(f"accuracy[{ds}][{L}]", torch.tensor(v))
                self.log(f"val/{ds}/acc_by_len/{L}", torch.tensor(v))

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.train_config['lr'], weight_decay=self.train_config['weight_decay'])
        sched = get_scheduler("cosine", opt, 1000, 50_000)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1}}

    def train_dataloader(self):
        dataset = JSONLDataset(self.train_config['train_path'])
        max_len = min(self.train_config.get('max_len', self.encoder_config.max_position_embeddings), 
                      self.encoder_config.max_position_embeddings)
        #dataset=RandomDataset(max_len=max_len)
        return torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.train_config['batch_size'],
            shuffle=False,
            num_workers=self.train_config.get('num_workers', 2),
            collate_fn=partial(collate_fn, max_len=max_len, pad_token=self.encoder_config.pad_token_id, eos_token=self.encoder_config.eos_token_id)
        )

    def val_dataloader(self, limit_val_examples: bool = True):
        max_len = min(
            self.train_config.get('max_len', self.encoder_config.max_position_embeddings),
            self.encoder_config.max_position_embeddings
        )
        val_batch_size = self.train_config.get('val_batch_size', self.train_config['batch_size'])
        
        loaders = []
        for path in [self.train_config['patches_path'], self.train_config['rand_path']]:
            dataset = JSONLDataset(path)
            if limit_val_examples:
                dataset = torch.utils.data.Subset(dataset, range(min(300*val_batch_size, len(dataset))))
            
            loaders.append(torch.utils.data.DataLoader(
                dataset,
                batch_size=val_batch_size,
                shuffle=False,
                num_workers=self.train_config.get('num_workers', 2),
                collate_fn=partial(collate_fn, max_len=max_len, pad_token=self.encoder_config.pad_token_id, eos_token=self.encoder_config.eos_token_id)
            ))
        return loaders

    def on_train_end(self):
        full_val_loaders = self.val_dataloader(limit_val_examples=False)

        results = self.trainer.validate(self, dataloaders=full_val_loaders)

        if isinstance(results, list):
            for i, res in enumerate(results):
                for k, v in res.items():
                    self.log(f"val_full/dataloader_{i}/{k}", v, prog_bar=False)
        else:
            for k, v in results.items():
                self.log(f"val_full/{k}", v, prog_bar=False)

class CurriculumCheckpointCallback(pl.Callback):
    def __init__(self, dirpath="./checkpoints_trans_autoencoder/"):
        self.dirpath = dirpath
        self.last_curriculum_step = -1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if hasattr(pl_module, 'saveit') and pl_module.saveit:
            checkpoint_path = f"{self.dirpath}/curr_step_{pl_module.curriculumStep}_step{trainer.global_step}.ckpt"
            trainer.save_checkpoint(checkpoint_path)
            self.last_curriculum_step = pl_module.curriculumStep

def main():
    torch.set_float32_matmul_precision('high')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/tiny.json")
    parser.add_argument("--curriculum_mode", action="store_true")
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--curriculum_start", type=int, default=None)
    parser.add_argument("--curriculum_end", type=int, default=None)
    parser.add_argument("--curriculum_patience", type=int, default=None)
    parser.add_argument("--curriculum_max_steps", type=int, default=None)
    parser.add_argument("--save_every_n_steps", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--until_convergence", type=int, default=None)
    parser.add_argument("--target_loss", type=float, default=None)
    parser.add_argument("--post_convergence_steps", type=int, default=None)
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--patches_path", type=str, default=None)
    parser.add_argument("--rand_path", type=str, default=None)
    parser.add_argument("--val_every_n_steps", type=int, default=None)
    parser.add_argument("--val_batch_size", type=int, default=2048)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    train_config = config['training'].copy()
    for key in ['max_len', 'batch_size', 'lr', 'max_epochs', 'num_workers', 'curriculum_start',
                'curriculum_end', 'curriculum_patience', 'curriculum_max_steps', 'save_every_n_steps',
                'until_convergence', 'target_loss', 'post_convergence_steps', 'train_path',
                'patches_path', 'rand_path', 'val_every_n_steps', 'val_batch_size']:
        if getattr(args, key, None) is not None:
            train_config[key] = getattr(args, key)
    
    train_config['curriculum_mode'] = args.curriculum_mode
    if train_config.get('val_batch_size') is None:
        train_config['val_batch_size'] = train_config['batch_size']
    
    encoder_config = ModelConfig(**config['encoder'])
    decoder_config = ModelConfig(**config['decoder'])
    
    model = TransCharAutoencoderLightning(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        train_config=train_config
    )
    """
    if args.resume_from_checkpoint is not None:
        ckpt = torch.load(args.resume_from_checkpoint, map_location='cuda', weights_only=False)
        sd = ckpt.get('state_dict', ckpt)
        missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
    """
    wandb_logger = WandbLogger(
        name='training-run',
        project='trans_char_autoencoder',
        config=None
    )
    
    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator="gpu",
        devices=1,
        max_epochs=train_config.get('max_epochs', 10),
        precision="16",
        log_every_n_steps=1,
        gradient_clip_val=0.5,
        val_check_interval=train_config.get('val_every_n_steps', 2000),
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath="./checkpoints/",
                filename="autoencoder-{step:08d}",
                every_n_train_steps=train_config.get('save_every_n_steps', 1000),
                save_top_k=-1
            ),
            CurriculumCheckpointCallback(dirpath="./checkpoints/")
        ]
    )
    
    trainer.fit(model, ckpt_path=args.resume_from_checkpoint)
    #trainer.fit(model)
if __name__ == "__main__":
    main()
