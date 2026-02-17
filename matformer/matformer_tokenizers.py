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
    def __init__(self, tokenizer=None,tokenizer_type=None, tokenizer_name=None, tokenizer_args=None, varlen_strategy=None, config=None):
        self.config = config
        tokenizer=tokenizer_type #[kept for compatibility, deprecated]
        if tokenizer == 'bytes':
            self.tokenizer = ByteLevelTokenizer(config)
            self.tokenizer_modality='bytes'
            self.vocab_size=255
            self.return_type=int
        elif tokenizer =='ae_bytes':
            self.tokenizer=AutoencoderByteTokenizer(encoder_seq_len=config.encoder.max_position_embeddings, eos_token_id=config.encoder.eos_token_id)
            self.tokenizer_modality='ae_bytes'
            self.vocab:size=255
            self.return_type=int
        elif tokenizer=='huggingface':
            from transformers import AutoTokenizer
            #self.tokenizer=AutoTokenizer.from_pretrained(tokenizer_name,**tokenizer_args)
            self.tokenizer=AutoTokenizer.from_pretrained(tokenizer_name)            
            self.vocab_size=self.tokenizer.vocab_size
            self.mask_token_id=self.tokenizer.mask_token_id
            self.bos_token_id=self.tokenizer.bos_token_id
            self.eos_token_id=self.tokenizer.eos_token_id
            self.return_type=int
        else: #Directly pass an HuggingFace tokenizer [kept for compatibility, deprecated]
            self.tokenizer = tokenizer
            self.tokenizer_modality='huggingface'
            self.vocab_size=tokenizer.vocab_size
            self.return_type=int

        self.seq_len = config.max_position_embeddings if config else None
        self.pad_token_id = config.pad_token_id if config else None
        self.varlen_strategy = varlen_strategy if varlen_strategy else None
        
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
    def encode(
        self,
        text: str,
        truncation: bool = False,
        add_eos: bool = False,
        add_bos: bool = False,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False
    ):
        # ByteLevelTokenizer already handles bos/eos tokens in __call__
        if isinstance(self.tokenizer, ByteLevelTokenizer):
            encoded = self.tokenizer(text)
            input_ids = encoded["input_ids"]
        else:
            # HuggingFace tokenizers
            encoded = self.tokenizer(
                text,
                add_special_tokens=add_special_tokens,
                return_offsets_mapping=return_offsets_mapping
            )
            input_ids = encoded["input_ids"]

            if add_bos and getattr(self.tokenizer, "bos_token_id", None) is not None:
                input_ids = [self.tokenizer.bos_token_id] + input_ids

            if add_eos and getattr(self.tokenizer, "eos_token_id", None) is not None:
                input_ids = input_ids + [self.tokenizer.eos_token_id]

        if truncation and hasattr(self, "seq_len"):
            input_ids = input_ids[:self.seq_len]

        if return_offsets_mapping:
            return {
                "input_ids": input_ids,
                "offset_mapping": encoded.get("offset_mapping", None)
            }

        return input_ids


    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

    def batch_decode(self, batch_ids: List[List[int]]) -> List[str]:
        return [self.decode(ids) for ids in batch_ids]
        
    def batch_encode(self, batch: List[Union[str, List[str]]]):
        if len(batch) > 0 and isinstance(batch[0], list):
            #### This branch is intended to be used only for the char autoencoder experiment ###
            ae_tok = self.tokenizer
            
            # 1. Get tensors and masks from the nested batches
            per_seq = [ae_tok.batch_encode(patches) for patches in batch]
            per_seq_tensors, per_seq_masks = zip(*per_seq)
            
            B = len(per_seq_tensors)
            S = self.config.encoder.max_position_embeddings
            P_pad = self.seq_len

            # 2. Initializing "empty patches" filled with only PAD ids
            tensors = torch.full((B, P_pad, S), self.config.encoder.pad_token_id, dtype=torch.long)
            padding_masks_external = torch.ones((B, P_pad, S), dtype=torch.bool)
            padding_masks_internal=torch.ones((B, P_pad, S), dtype=torch.bool)
            # 3. Fill the tensors and masks
            for b in range(B):
                t_batch = per_seq_tensors[b]
                m_batch = per_seq_masks[b]
                P = min(t_batch.shape[0], P_pad)
                
                if P > 0:
                    tensors[b, :P, :] = t_batch[:P]  # Fill with token IDs from the inner batch
                    padding_masks_external[b, :P, :] = False #These boolean value refers only to fully padded patches, to be used to perform unpadding on the big transformer
                    padding_masks_internal[b, :P, :] = m_batch[:P]    # Fill with the actual padding mask from the tokenizer, to be used in the autoencoder
                    return {'tensor':tensors,'padding_masks_external':padding_masks_external,'padding_masks_internal':padding_masks_internal,'varlen_strategy':self.varlen_strategy}
        else:
            ### This is the normal Matformer branch ###
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
        The encode and batch_encode functions will also return "internal" padding masks
        def encode_texts(texts, max_len, pad_token, eos_token):
    """
    def __init__(self, encoder_seq_len: int, eos_token_id: int):
        self.encoder_seq_len = int(encoder_seq_len)
        self.eos = int(eos_token_id)
        self.pad=int(259)
    def encode(self, text: str) -> torch.Tensor:
        ids = list(text.encode("utf-8"))
        ids += [self.eos]
        ids = ids[: self.encoder_seq_len]
        if len(ids) < self.encoder_seq_len:
            ids += [self.pad] * (self.encoder_seq_len - len(ids))
        tensor=torch.tensor(ids, dtype=torch.long)
        padding_mask=tensor==self.pad
        return tensor,padding_mask

    def batch_encode(self, texts: List[str]) -> tuple[torch.Tensor, torch.Tensor]:
        encoded_pairs = [self.encode(t) for t in texts]
        tensors, padding_masks = zip(*encoded_pairs)
        return torch.stack(tensors, dim=0), torch.stack(padding_masks, dim=0) 
        

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
