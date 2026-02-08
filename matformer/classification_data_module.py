import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import json
import csv

class ClassificationDataset(Dataset):
    def __init__(self, examples):
        """
        Args:
            examples: List of (input_ids, label) tuples
        """
        self.examples = examples
    
    def __getitem__(self, idx):
        return self.examples[idx]
    
    def __len__(self):
        return len(self.examples)


class ClassificationDataModule(pl.LightningDataModule):
    def __init__(self, data_loader, tokenizer, max_seq_len, batch_size, 
                 pad_token_id, num_devices=1, val_data_loader=None):
        """
        Args:
            data_loader: class that handles data loading (default works on CSV/TSV/JSON files)
            tokenizer: Tokenizer instance with encode() method
            max_seq_len: Maximum sequence length
            batch_size: Batch size
            pad_token_id: Token ID for padding
            num_devices: Number of GPUs/devices
            val_data_loader: Optional loader
        """
        super().__init__()
        self.data_loader = data_loader
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.pad_token_id = pad_token_id
        self.num_devices = num_devices
        self.val_data_loader = val_data_loader
        self.num_labels = data_loader.get_num_labels()
    
    def setup(self, stage=None):
        
        texts, labels, ids = self.data_loader.get_data()
        
        # Tokenize and create examples
        train_examples = []
        for text, label in zip(texts, labels):
            input_ids = self.tokenizer.encode(text)
            if len(input_ids) > self.max_seq_len:
                input_ids = input_ids[:self.max_seq_len]
            train_examples.append((input_ids, int(label)))
        
        self.train_dataset = ClassificationDataset(train_examples)
        
        # Same for validation if exists
        if self.val_data_loader is not None:
            val_texts, val_labels, _ = self.val_data_loader.get_data()
            val_examples = []
            for text, label in zip(val_texts, val_labels):
                input_ids = self.tokenizer.encode(text)
                if len(input_ids) > self.max_seq_len:
                    input_ids = input_ids[:self.max_seq_len]
                val_examples.append((input_ids, int(label)))
            self.val_dataset = ClassificationDataset(val_examples)
    
    def collate_fn(self, batch):
        """
        Collate function to pad sequences and create batches
        
        Args:
            batch: List of (input_ids, label) tuples
        
        Returns:
            Dict with input_ids, attention_mask, and labels tensors
        """
        input_ids_list = []
        labels_list = []
        
        for input_ids, label in batch:
            # Pad to max_seq_len
            padded = input_ids + [self.pad_token_id] * (self.max_seq_len - len(input_ids))
            input_ids_list.append(padded)
            labels_list.append(label)
        
        # Convert to tensors
        input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long)
        labels_tensor = torch.tensor(labels_list, dtype=torch.long)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids_tensor != self.pad_token_id).long()
        
        return {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask,
            "labels": labels_tensor
        }
    
    def _create_dataloader(self, dataset, shuffle):
    """Helper method to create dataloader with optional distributed sampling"""
    is_distributed = dist.is_available() and dist.is_initialized()
    
    if is_distributed:
        sampler = DistributedSampler(
            dataset,
            shuffle=shuffle
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=self.collate_fn,
            num_workers=0
        )
    else:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            num_workers=0
        )

def train_dataloader(self):
    """Create training dataloader with optional distributed sampling"""
    return self._create_dataloader(self.train_dataset, shuffle=True)

def val_dataloader(self):
    """Create validation dataloader if validation data exists"""
    if self.val_dataset is None:
        return None
    
    return self._create_dataloader(self.val_dataset, shuffle=False)
    
    def __len__(self):
        """Return dataset length"""
        if self.train_dataset is not None:
            return len(self.train_dataset)
        return 0
      
    def params(self):
      return {
          "num_labels": self.num_labels,
          "max_seq_len": self.max_seq_len,
          "batch_size": self.batch_size,
          "pad_token_id": self.pad_token_id}
      