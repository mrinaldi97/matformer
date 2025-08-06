"""
File: matformer/tokenizers.py
"""
import torch
from typing import List
from matformer.model_config import ModelConfig
from matformer.tensors_dataclasses import PaddedTensor, UnpaddedTensor
from dataclasses import dataclass, replace

class MatformerTokenizer:
    """
    This is an abstract tokenizer that will wrap other tokenizers (such as a ByteLevelTokenizer or an huggingface-compatible tokenizer)
    to adapt it to the logic of the Matformer architecture. In particular, it will take care of the delicate batch encoding, working with
    all the possibilities compatible with Matformers: PaddedTensors, UnpaddedTensors and Pytorch's Nested Tensors.
    """
    def __init__(self, config, tokenizer, varlen_strategy):
        self.config = config
        if tokenizer == 'bytes':
            self.tokenizer = ByteLevelTokenizer(config)
        else: #Directly pass an HuggingFace tokenizer
            self.tokenizer = tokenizer

        self.seq_len = config.max_position_embeddings
        self.pad_token_id = config.pad_token_id
        self.varlen_strategy = varlen_strategy
        """
        Note about self.config.varlen_strategy: this is very important!
        It decides the type of tensor coming from the tokenizer:
            => 'padding': PaddedTensor with tensor of seqlen padded with 0 and a padding mask
            => 'unpadding': UnpaddedTensor
            => 'nested': NormalTensor with a nested tensor inside

        The parameter self.config['training_objective'] can be 'masked' for a bert-like model, or 'autoregressive' for a gpt-like model.
        In case of masked, the tokenizer expects self.config.masked.substitution_rate (ex 0.2 to say that 20% of tokens are substituted),
        self.config.masked.masking_percentage (how may tokens to substitute with [MASK], ex. 0.8 for 80%), then self.config.masked.random_percentage
        and self.config.masked.same_percentage for the other possible masking strategies amounts.
        """
    def process_pretokenized_batch(token_sequences, config, varlen_strategy, pad_token_id):
        """
        Helper function to process pre-tokenized sequences from MatformerDataset
        into the format expected by the model training code.
        
        Args:
            token_sequences: List of token lists from MatformerDataset
            config: ModelConfig
            varlen_strategy: 'padding', 'unpadding', or 'nested'
            pad_token_id: Padding token ID
        
        Returns:
            Processed sequence in the format expected by PL_ModelWrapper
        """
        seq_len = config.max_position_embeddings
        
        # Truncate sequences to max length 
        token_sequences = [seq[:seq_len] for seq in token_sequences]
        
        if varlen_strategy == 'nested':
            sequence = torch.nested.nested_tensor(token_sequences, layout=torch.jagged)
            return sequence
        
        # Pad sequences
        padded_ids = [ids + [pad_token_id] * (seq_len - len(ids)) for ids in token_sequences]
        tensors = torch.tensor(padded_ids, dtype=torch.long)
        padding_masks = (tensors == pad_token_id)
        sequence = PaddedTensor(tensor=tensors, padding_mask=padding_masks)
        
        if varlen_strategy == 'unpadding':
            sequence = sequence.unpad()
        
        return sequence
    def encode(self, text: str, truncation=True):
        input_ids = self.tokenizer(text)['input_ids']
        if truncation:
            input_ids = input_ids[:self.seq_len]
        return input_ids

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

    def batch_decode(self, batch_ids: List[List[int]]) -> List[str]:
        return [self.decode(ids) for ids in batch_ids]

    def batch_encode(self, batch: List[str]):
        
        all_input_ids = []

        for s in batch:
            input_ids = self.encode(s, truncation=True)
            all_input_ids.append(input_ids)

        if self.varlen_strategy == 'nested':
            sequence = torch.nested.nested_tensor(all_input_ids, layout=torch.jagged)
            inputs = torch.nested.nested_tensor([ids[:-1] for ids in all_input_ids], layout=torch.jagged)
            targets = torch.nested.nested_tensor([ids[1:] for ids in all_input_ids], layout=torch.jagged)
            return inputs, targets, sequence

        padded_ids = [ids + [self.pad_token_id] * (self.seq_len - len(ids)) for ids in all_input_ids]
        tensors = torch.tensor(padded_ids, dtype=torch.long)
        padding_masks = (tensors == self.pad_token_id)
        sequence = PaddedTensor(tensor=tensors, padding_mask=padding_masks)
        if self.varlen_strategy == 'unpadding':
                sequence = sequence.unpad()
        return sequence

class ByteTokenizer:
    def __init__(self, eos_token, pad_token):
        self.bos_token_id = 257
        self.eos_token_id = 258
        self.pad_token_id = 259
        self.eos_token=258
        self.bos_token=257 
        self.pad_token=259
        self.offset = 0

    def __call__(self, text: str) -> dict:
        byte_ids = [b + self.offset for b in text.encode('utf-8', errors='ignore')] + [self.eos_token_id]
        return byte_ids

    def decode(self, ids: List[int]) -> str:
        special_tokens = {self.bos_token_id, self.eos_token_id, self.pad_token_id}
        bytes_data = bytearray([max(0, i - self.offset) for i in ids if i not in special_tokens])
        return bytes_data.decode('utf-8', errors='replace')

class ByteLevelTokenizer:
    def __init__(self, config):
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.pad_token_id = config.pad_token_id
        self.offset = 0

    def __call__(self, text: str) -> dict:
        byte_ids = [self.bos_token_id] + [b + self.offset for b in text.encode('utf-8', errors='ignore')] + [self.eos_token_id]
        return {'input_ids': byte_ids}

    def decode(self, ids: List[int]) -> str:
        special_tokens = {self.bos_token_id, self.eos_token_id, self.pad_token_id}
        bytes_data = bytearray([max(0, i - self.offset) for i in ids if i not in special_tokens])
        return bytes_data.decode('utf-8', errors='ignore')
