"""
File: matformer/tokenizers.py
"""
import torch
from typing import List
from matformer.model_config import ModelConfig  
from matformer.tensors_dataclasses import PaddedTensor,UnpaddedTensor
class ByteLevelTokenizer:
    def __init__(self, config, varlen_strategy=None):
        self.seq_len = config.max_seqlen
        self.bos_id = config.bos_id
        self.eos_id = config.eos_id
        self.offset = 0
        self.pad_id = config.pad_id
        
        """
        Note about varlen_strategy: this is very important!
        It decides the type of tensor coming from the tokenizer:
            => 'padding': PaddedTensor with tensor of seqlen padded with 0 and a padding mask
            => 'unpadding': UnpaddedTensor
            => 'nested': NormalTensor with a nested tensor inside
        """
        self.varlen_strategy=varlen_strategy
    def encode(self, text, padding=False, truncation=False):
        """Convert a single text string to token ids"""
        byte_ids = [self.bos_id] + [b + self.offset for b in text.encode('utf-8', errors='ignore')] + [self.eos_id]
        if truncation:
            byte_ids = byte_ids[:self.seq_len]
        if padding:
            byte_ids = byte_ids + [self.pad_id] * (self.seq_len - len(byte_ids))
        if self.varlen_strategy in ('unpadding', 'padding'):
            padding_mask = torch.cat([torch.zeros(len(byte_ids), dtype=torch.bool), torch.ones(self.seq_len - len(byte_ids), dtype=torch.bool)])
            return torch.tensor(byte_ids,dtype=torch.long),padding_mask
        else:
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
        if nested or self.varlen_strategy=='nested':
            encoded=[self.encode(s,padding,truncation) for s in batch]
            inputs=torch.nested.nested_tensor([s[:-1] for s in encoded], layout=torch.jagged)
            targets=torch.nested.nested_tensor([s[1:] for s in encoded], layout=torch.jagged)
            sequence=torch.nested.nested_tensor([s for s in encoded], layout=torch.jagged)
            return inputs,targets,sequence
        
        if self.varlen_strategy is None:
            return torch.stack([self.encode(s, padding, truncation) for s in batch])
            
        tensors, padding_masks = zip(*[self.encode(s, padding, truncation) for s in batch])
        
        tensors = torch.stack(tensors)
        padding_masks = torch.stack(padding_masks)
        
        inputs = PaddedTensor(tensor=tensors[:, :-1], padding_mask=padding_masks[:, :-1])
        targets = PaddedTensor(tensor=tensors[:, 1:], padding_mask=padding_masks[:, 1:])
        sequence = PaddedTensor(tensor=tensors, padding_mask=padding_masks)
        
        if self.varlen_strategy=='padding':
            return inputs, targets, sequence
        elif self.varlen_strategy=='unpadding':
            return inputs.unpad(), targets.unpad(), sequence.unpad()

        
        
    
    def batch_decode(self, batch_ids):
        """Convert a batch of token ids back to text"""
        return [self.decode(ids) for ids in batch_ids]
    
    def __call__(self, batch):
        """Make the tokenizer callable for use as collate_fn"""
        return self.batch_encode(batch)
