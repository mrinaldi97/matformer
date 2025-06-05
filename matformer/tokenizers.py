"""
File: matformer/tokenizers.py
"""
import torch
from typing import List
from matformer.model_config import ModelConfig  

class ByteLevelTokenizer:
    def __init__(self, config):
        self.seq_len = config.max_seqlen
        self.bos_id = config.bos_id
        self.eos_id = config.eos_id
        self.offset = 0
        self.pad_id = config.pad_id
        
    def encode(self, text, padding=False, truncation=False):
        """Convert a single text string to token ids"""
        byte_ids = [self.bos_id] + [b + self.offset for b in text.encode('utf-8', errors='ignore')] + [self.eos_id]
        if truncation:
            byte_ids = byte_ids[:self.seq_len]
        if padding:
            byte_ids = byte_ids + [self.pad_id] * (self.seq_len - len(byte_ids))
        return torch.tensor(byte_ids, dtype=torch.long)
    def single_decode(self,_id):
        """Decode a single id to its byte"""
        return chr(_id-self.offset)
    def decode(self, ids):
        """Convert token ids back to text, removing special tokens"""
        bytes_data = bytearray([max(0, id - self.offset) for id in ids])
        return bytes_data.decode('utf-8', errors='ignore')
    
    def batch_encode(self, batch, padding=False, truncation=False, nested=False):
        """Convert a batch of text strings to token ids"""
        if nested:
            encoded=[self.encode(s,padding,truncation) for s in batch]
            inputs=torch.nested.nested_tensor([s[:-1] for s in encoded], layout=torch.jagged)
            targets=torch.nested.nested_tensor([s[1:] for s in encoded], layout=torch.jagged)
            sequence=torch.nested.nested_tensor([s for s in encoded], layout=torch.jagged)
            return inputs,targets,sequence
        else:
            return torch.stack([self.encode(s, padding, truncation) for s in batch])
    
    def batch_decode(self, batch_ids):
        """Convert a batch of token ids back to text"""
        return [self.decode(ids) for ids in batch_ids]
    
    def __call__(self, batch):
        """Make the tokenizer callable for use as collate_fn"""
        return self.batch_encode(batch)
