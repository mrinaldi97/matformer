import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import json
import csv
from matformer.tensors_dataclasses import PaddedTensor

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
                 pad_token_id, task_type='sequence', num_workers=1,
                 val_data_loader=None, varlen_strategy = "Padding"):
        """
        Args:
            data_loader: class that handles data loading (default works on CSV/TSV/JSON files)
            tokenizer: Tokenizer instance with encode() method
            max_seq_len: Maximum sequence length
            batch_size: Batch size
            pad_token_id: Token ID for padding
            num_workers: Number of workers
            val_data_loader: Optional loader
        """
        super().__init__()
        self.data_loader = data_loader
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.pad_token_id = pad_token_id
        self.num_workers = num_workers
        self.val_data_loader = val_data_loader
        self.num_labels = data_loader.get_num_labels()
        self.varlen_strategy = varlen_strategy
        self.task_type = task_type
    
    
    def setup(self, stage=None):
      train_truncation = 0
      val_truncation = 0
        
      train_samples , train_truncation = self._sampler(self.data_loader)
      self.train_dataset = ClassificationDataset(train_samples)
        
      # Same for validation if exists
      if self.val_data_loader is not None:
        val_samples , val_truncation = self._sampler(self.val_data_loader)
        self.val_dataset = ClassificationDataset(val_samples)
            
      if train_truncation or val_truncation:
        print(f"\n--- Truncation of long samples ---")
        print(f"Currently the max length of samples is set to {self.max_seq_len}")
        print(f"This caused the truncation of {train_truncation} in the train set and {val_truncation} in the validation set")
        print("If this is unwanted behaviour, consider modifying the value in the config file\n" + "-" * 30)
    
    
    def _sampler(self, data_loader):
      truncation = 0
      texts, labels, ids = data_loader.get_data()
      samples = []
      
      for text, label in zip(texts, labels):
        input_ids = self.tokenizer.encode(text)
        input_ids, label, trunc = self._truncate_sample(input_ids, label)
        truncation += trunc
        samples.append((input_ids, int(label)))
        
      return samples, truncation
    
    def _truncate_sample(self, input_ids, label):
        """Truncate input and label if needed. Returns (input_ids, label, was_truncated)"""
        if len(input_ids) <= self.max_seq_len:
            return input_ids, label, 0
        
        input_ids = input_ids[:self.max_seq_len]
        if self.task_type == 'token':
            label = label[:self.max_seq_len]
        return input_ids, label, 1
    
    
    """
    La collate_fn restituisce un dizionario input_ids, attention_mask, labels. Sta quindi passando degli oggetti torch.Tensor; È possibile con molta facilità integrare in questo punto la logica padding/unpadding di matformer in modo da armonizzarlo con il codice usato in addestramento;
    """
    def collate_fn(self, batch):
      
        input_ids_list = []
        labels_list = []
        
        for input_ids, label in batch:
            padded = input_ids + [self.pad_token_id] * (self.max_seq_len - len(input_ids))
            input_ids_list.append(padded)
            labels_list.append(label)
        
        input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long)
        labels_tensor = torch.tensor(labels_list, dtype=torch.long)
        padding_mask = (input_ids_tensor == self.pad_token_id)
        padded_sequence = PaddedTensor(tensor=input_ids_tensor, padding_mask=padding_mask)
        
        if self.varlen_strategy=='unpad':
          padded_sequence.unpad()
        
        return {
            "input_ids": padded_sequence,
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
                num_workers=self.num_workers,
                drop_last=True
            )
        else:
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                collate_fn=self.collate_fn,
                num_workers=self.num_workers,
                drop_last=True
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
      