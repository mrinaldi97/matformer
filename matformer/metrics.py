"""
File: matformer/metrics.py

"""
import torch
import pytorch_lightning as pl
import math
import torchmetrics
from matformer.model_config import ModelConfig  


class BitsPerByte(torchmetrics.Metric):
    def __init__(self, config):
        super().__init__()
        self.add_state("loss_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("byte_count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.config=config
    def update(self, loss, batch):
        # Calculate total bytes in batch (excluding special tokens)
        # batch shape: [batch_size, seq_len]
        mask = (batch != config.eos_token_id) & (batch != config.pad_token_id) & (batch != config.bos_token_id)  # Exclude BOS, EOS, PAD
        byte_count = mask.sum().item()
        
        # Accumulate loss and byte count
        self.loss_sum += loss.item() * batch.size(0)  # Multiply by batch size as loss is mean
        self.byte_count += byte_count
    
    def compute(self):
        # BPB = CrossEntropyLoss / ln(2) * num_bytes
        return self.loss_sum / (math.log(2) * self.byte_count) if self.byte_count > 0 else torch.tensor(float('inf'))

   
