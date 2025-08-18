"""
File: matformer/tokenizers.py
"""
import torch
from typing import List, Union
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
            self.tokenizer_modality='bytes'
        elif tokenizer =='ae_bytes':
            self.tokenizer=AutoencoderByteTokenizer(encoder_seq_len=config.encoder.max_position_embeddings, eos_token_id=config.encoder.eos_token_id)
            self.tokenizer_modality='ae_bytes'
        else: #Directly pass an HuggingFace tokenizer
            self.tokenizer = tokenizer
            self.tokenizer_modality='huggingface'
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
    def process_pretokenized_batch(self,token_sequences, config, varlen_strategy, pad_token_id):
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
        
        if token_sequences is not None:
            new_sequences = []
            for seq in token_sequences:
                if seq is None:
                    print(f"WARNING: A sequence is None")
                    new_sequences.append([])
                    continue

                if len(seq) > seq_len:
                    print(f"WARNING: A sequence is longer than max length ({len(seq)} > {seq_len}) and it's truncated")
                    seq = seq[:seq_len]

                new_sequences.append(seq)
            
            token_sequences = new_sequences
        else:
            print("WARNING: GOT A None TOKEN SEQUENCES FROM THE DATALOADER!")
            token_sequences = []

        
        
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
        
    def batch_encode(self, batch: List[Union[str, List[str]]]):
        if len(batch) > 0 and isinstance(batch[0], list):
            ae_tok = self.tokenizer
            per_seq = [ae_tok.batch_encode(patches) for patches in batch]  # each [P, S]
            B = len(per_seq)
            S = per_seq[0].shape[1]
            P_pad = self.seq_len  # pad/truncate patches to big-transformer length

            # init with "empty patch" (all EOS)
            tensors = torch.full((B, P_pad, S), 259, dtype=torch.long)
            padding_masks = torch.ones((B, P_pad, S), dtype=torch.bool)  # True = padded patch
            for b, t in enumerate(per_seq):
                P = min(t.shape[0], P_pad)
                if P > 0:
                    tensors[b, :P, :] = t[:P]
                    padding_masks[b, :P, :] = False  
            #print(f"Shape of tensors: {tensors.shape}")
        else:
            #print("DEBUG: SOno nel ramo normale")
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

    def _batch_encode(self, batch: List[str]):
        
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


        

class AutoencoderByteTokenizer:
    """
        This tokenizer receives batch of text chunks, for example:
        ['ciao ','come ','stai?'],['bene,',' tu?']
        It will return a structure:
        B,P,S where B is the external batch, the one seen by the big transormer, and here we will apply usual Padding or Unpadding
        P is the internal batch, here we will apply internal padding when we tokenize
        def encode_texts(texts, max_len, pad_token, eos_token):
    """
    def __init__(self, encoder_seq_len: int, eos_token_id: int):
        self.encoder_seq_len = int(encoder_seq_len)
        self.eos = int(eos_token_id)
    def encode(self, text: str) -> torch.Tensor:
        ids = list(text.encode("utf-8"))
        ids += [self.eos, self.eos]
        ids = ids[: self.encoder_seq_len]
        if len(ids) < self.encoder_seq_len:
            ids += [259] * (self.encoder_seq_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def batch_encode(self, texts: List[str]) -> torch.Tensor:
        return torch.stack([self.encode(t) for t in texts], dim=0)  # [B, S]        
        

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
