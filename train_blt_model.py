"""
Non so se funziona, codice di appoggio non ancora testato
"""

import argparse
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from matformer.tokenizers import ByteLevelTokenizer
from matformer.training_functions import MatformerDataModule
from matformer.model_config import ModelConfig
from matformer.models import EntropyModel
from matformer.transformer_blocks import BLTTransfomer

class BLTLightningModule(pl.LightningModule):
    def __init__(self, blt_model, train_config, tokenizer_config):
        super().__init__()
        self.blt_model = blt_model
        self.train_config = train_config
        self.tokenizer_config = tokenizer_config
        self.save_hyperparameters(ignore=['blt_model'])

    def forward(self, text_tokens, smoothing=None):
        return self.blt_model(text_tokens, smoothing=smoothing)

    def _common_step(self, batch, batch_idx):
        input_ids = batch['input_ids'] if isinstance(batch, dict) else batch
        targets = input_ids[:, 1:].contiguous().view(-1)
        inputs = input_ids[:, :-1].contiguous()
        logits = self(inputs)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets, ignore_index=self.tokenizer_config.pad_id)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.train_config["lr"])
        if self.train_config.get("warmup_steps", 0) > 0 and self.train_config.get("max_steps"):
            def lr_lambda(current_step):
                if current_step < self.train_config["warmup_steps"]:
                    return float(current_step) / float(max(1, self.train_config["warmup_steps"]))
                return max(0.0, float(self.train_config["max_steps"] - current_step) / float(max(1, self.train_config["max_steps"] - self.train_config["warmup_steps"])))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}
        return optimizer

def main():
    parser = argparse.ArgumentParser(description='Train BLT model (Compact Version)')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--entropy_model_checkpoint', type=str, required=True)
    parser.add_argument('--dump_dir', type=str, default='./blt_checkpoints_compact')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_steps', type=int, default=100000)
    parser.add_argument('--max_epochs', type=int, default=None)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--precision', type=str, default='16-mixed')
    parser.add_argument('--compile_model', action='store_true')
    parser.add_argument('--smoothing', type=float, default=None)
    args = parser.parse_args()

    train_hyperparams = {"lr": args.learning_rate, "warmup_steps": args.warmup_steps, "max_steps": args.max_steps if args.max_steps > 0 else None}
    pl.seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpus > 0 else "cpu")

    try:
        loaded_entropy_model, entropy_cfg_obj = EntropyModel.load_from_checkpoint(args.entropy_model_checkpoint, map_location=device)
    except TypeError:
        loaded_entropy_model = EntropyModel.load_from_checkpoint(args.entropy_model_checkpoint, map_location=device)
        if hasattr(loaded_entropy_model, 'hparams') and isinstance(loaded_entropy_model.hparams.get('config'), ModelConfig): entropy_cfg_obj = loaded_entropy_model.hparams.config
        elif hasattr(loaded_entropy_model, 'config') and isinstance(loaded_entropy_model.config, ModelConfig): entropy_cfg_obj = loaded_entropy_model.config
        else: raise ValueError("Could not retrieve ModelConfig from loaded EntropyModel.")
    loaded_entropy_model.eval().to(device)

    tokenizer = ByteLevelTokenizer(config=entropy_cfg_obj)
    
    common_params = {'vocab_size': entropy_cfg_obj.vocab_size, 'pad_id': entropy_cfg_obj.pad_id, 'max_seqlen': entropy_cfg_obj.max_seqlen}

    text_encoder_config = ModelConfig(name='BLT Text Encoder', hidden_dim=768, ffn_factor=1.0, n_layers=6, n_heads=16, tie_word_embeddings=False, rms_norm_eps=1e-6, attention_type=['causal', 'sliding'], sliding_window_size=512, sliding_type='full', block_size_for_attention=128, compile_flexattn=False, bias=False, **common_params)
    text_decoder_config = ModelConfig(name='BLT Text Decoder', hidden_dim=768, ffn_factor=1.0, n_layers=6, n_heads=16, tie_word_embeddings=False, rms_norm_eps=1e-6, attention_type=['causal', 'sliding'], sliding_window_size=512, sliding_type='full', block_size_for_attention=128, compile_flexattn=False, bias=False, **common_params)
    global_model_config = ModelConfig(name='BLT Global Model', hidden_dim=768, ffn_factor=1.0, n_layers=12, n_heads=16, tie_word_embeddings=False, rms_norm_eps=1e-6, attention_type=['causal', 'sliding'], sliding_window_size=512, sliding_type='full', block_size_for_attention=128, compile_flexattn=False, bias=False, **common_params)

    data_module = MatformerDataModule(data_root=args.data_root, batch_size=args.batch_size, num_workers=args.num_workers, config=entropy_cfg_obj, tokenizer=tokenizer)
    
    blt_model_instance = BLTTransfomer(entropy_config=entropy_cfg_obj, text_encoder_config=text_encoder_config, global_config=global_model_config, text_decoder_config=text_decoder_config, entropymodel=loaded_entropy_model, smoothing=args.smoothing, device=str(device))
    
    lightning_blt_model = BLTLightningModule(blt_model=blt_model_instance, train_config=train_hyperparams, tokenizer_config=entropy_cfg_obj)

    callbacks = [pl.callbacks.ModelCheckpoint(dirpath=args.dump_dir, filename='blt_compact-{epoch:02d}-{val_loss:.2f}', save_top_k=2, save_last=True, monitor='val_loss'), pl.callbacks.LearningRateMonitor(logging_interval='step')]
    
    trainer_params = {"default_root_dir": args.dump_dir, "precision": args.precision, "log_every_n_steps": 100, "accumulate_grad_batches": 1, "accelerator": "gpu" if args.gpus > 0 else "cpu", "callbacks": callbacks}
    if args.gpus > 0: trainer_params["devices"] = args.gpus
    if args.gpus > 1: trainer_params["strategy"] = "ddp" 
    if args.max_steps and args.max_steps > 0: trainer_params["max_steps"] = args.max_steps
    elif args.max_epochs: trainer_params["max_epochs"] = args.max_epochs
    else: trainer_params["max_epochs"] = 1
    trainer = pl.Trainer(**trainer_params)

    if args.compile_model: lightning_blt_model = torch.compile(lightning_blt_model)
    torch.set_float32_matmul_precision("high")
    trainer.fit(lightning_blt_model, datamodule=data_module)

if __name__ == '__main__':
    main() 
