"""
File: matformer/tokenizers.py
"""
import torch
from typing import List, Optional, Union
from matformer.dataclasses.model_dataclasses import TokenizerConfig
from matformer.matformer_registry import registry

@registry.register("tokenizer", "huggingface", "default")
class MatformerTokenizer:
    """
    This is an abstract tokenizer that will wrap other tokenizers (such as a ByteLevelTokenizer or an huggingface-compatible tokenizer)
    to adapt it to the logic of the Matformer architecture.
    The best way to initialize is to provide a TokenizerConfig: the module will take care to initialize the tokenizer 
    Alternatively, it is also possible to directly pass a tokenizer as argument. It will work with HF tokenizers, but to be used with other tokenizers, it is 
    user's responsability to do the check.

    Args:
        config (TokenizerConfig): tokenizer configuration dataclass
        tokenizer (optional): any tokenizer module (HF-compatible)
        reload_hf_cfg (bool): instead of the special tokens id and vocab size provided in the config, reload them from the HF tokenizer
        max_seq_len (optional, int): if set, truncates the sequence at max_seq_len
        return_tensors (bool or str): if True/"pt", encode returns HF-style padded torch tensors; if "padded_tensor" or
            "unpadded_tensor", encode returns a matformer PaddedTensor/UnpaddedTensor instead (padding_mask is derived
            from the attention_mask, never from comparing token ids, so it stays correct even when pad_token_id == eos_token_id)
    """

    TENSOR_TYPE = "pt"

    def __init__(
        self,
        config: Optional[TokenizerConfig] = None,
        tokenizer=None,
        reload_hf_cfg: bool = False,
        max_seq_len: Optional[int] = None,
        return_tensors: Union[bool, str] = False,
    ):
        """Initialize tokenizer from a config or a raw tokenizer instance."""
        self.config = config
        self.max_seq_len = max_seq_len

        if isinstance(return_tensors, bool):
            self.return_tensors = self.TENSOR_TYPE if return_tensors else None
        else:
            assert return_tensors in (self.TENSOR_TYPE, "padded_tensor", "unpadded_tensor"), (
                f"return_tensors must be a bool, '{self.TENSOR_TYPE}', 'padded_tensor' or 'unpadded_tensor', "
                f"got {return_tensors!r}"
            )
            self.return_tensors = return_tensors

        if config:
            from transformers import AutoTokenizer
            if config.pretrained_name is None:
                self.tokenizer=None
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_name,trust_remote_code=True)
                if reload_hf_cfg:
                    self._reload_huggingface_config()

        else:
            assert tokenizer is not None
            self.tokenizer = tokenizer
            self._reload_huggingface_config()

    def _reload_huggingface_config(self) -> None:
        """Reload special token ids and vocab size directly from the attached HF tokenizer."""
        self.config = TokenizerConfig.from_dict(
            {
                "type": "huggingface",
                "pretrained_name": self.tokenizer.name_or_path,
                "bos_token_id": self.tokenizer.bos_token_id or 0,
                "eos_token_id": self.tokenizer.eos_token_id or 0,
                "pad_token_id": self.tokenizer.pad_token_id or 0,
                "mask_token_id": self.tokenizer.mask_token_id or 0,
                "vocab_size": self.tokenizer.vocab_size,
            }
        )
    @classmethod
    def from_huggingface(cls, repo) -> "MatformerTokenizer":
        """Instantiate directly from a HuggingFace repo name or path."""
        from transformers import AutoTokenizer
        instance = cls(tokenizer=AutoTokenizer.from_pretrained(repo))
        return instance

    def _add_special_tokens(self, seq: List[int], add_bos: bool, add_eos: bool) -> List[int]:
        """Prepend BOS and/or append EOS token id to a sequence list."""
        if add_bos:
            seq = [self.config.bos_token_id] + seq
        if add_eos:
            seq = seq + [self.config.eos_token_id]
        return seq

    def encode(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
        add_eos: bool = False,
        add_bos: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict:
        """
        Encode text into token ids.

        Args:
            text (str or List[str]): input text or batch of texts to encode
            max_length (int, optional): maximum sequence length for this call; overrides self.max_seq_len
            add_bos (bool): prepend BOS token id to each sequence
            add_eos (bool): append EOS token id to each sequence
            return_offsets_mapping (bool): whether to return character-to-token offset mapping

        Returns:
            dict with keys: input_ids, offset_mapping, was_truncated, attention_mask
            (if self.return_tensors is "padded_tensor"/"unpadded_tensor", input_ids is a TensorDC and attention_mask is None,
            since the mask is embedded in the returned PaddedTensor/UnpaddedTensor instead)
        """
        if isinstance(text, str):
            text = [text]

        max_len = max_length or self.max_seq_len

        if self.return_tensors:
            assert max_len is not None, "max_seq_len must be set at initialization when return_tensors=True"

        # Any tensor-returning mode (incl. padded_tensor/unpadded_tensor) needs real "pt" tensors from HF first.
        kwargs = dict(
            return_offsets_mapping=return_offsets_mapping,
            return_tensors=(self.TENSOR_TYPE if self.return_tensors else None),
        )
        if self.return_tensors:
            reserved = int(add_bos) + int(add_eos)
            effective_max_len = max_len - reserved if (add_bos or add_eos) else max_len
            kwargs.update(padding="max_length", max_length=effective_max_len, truncation=True)

        encoded = self.tokenizer(text, add_special_tokens=False, **kwargs)
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask", None)

        if self.return_tensors:
            B = input_ids.shape[0]
            if attention_mask is None:
                # Some tokenizers don't return one; every position is real data.
                attention_mask = torch.ones_like(input_ids)
            if add_bos:
                input_ids = torch.cat([torch.full((B, 1), self.config.bos_token_id, dtype=input_ids.dtype), input_ids], dim=1)
                attention_mask = torch.cat([torch.ones(B, 1, dtype=attention_mask.dtype), attention_mask], dim=1)
            if add_eos:
                input_ids = torch.cat([input_ids, torch.full((B, 1), self.config.eos_token_id, dtype=input_ids.dtype)], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones(B, 1, dtype=attention_mask.dtype)], dim=1)
            if add_bos or add_eos:
                input_ids = input_ids[:, :max_len]
                attention_mask = attention_mask[:, :max_len]

            if self.return_tensors in ("padded_tensor", "unpadded_tensor"):
                from matformer.dataclasses.tensors_dataclasses import PaddedTensor

                # padding_mask is derived positionally from attention_mask, never from comparing token ids
                # against pad_token_id: this keeps a real appended EOS distinguishable from padding even
                # when pad_token_id == eos_token_id (a common HF setup for tokenizers without a dedicated pad token).
                padded = PaddedTensor(tensor=input_ids, padding_mask=~attention_mask.bool())
                result = padded.unpad() if self.return_tensors == "unpadded_tensor" else padded

                return {
                    "input_ids": result,
                    "offset_mapping": encoded.get("offset_mapping", None),
                    "was_truncated": max_len is not None,
                    "attention_mask": None,
                }
        else:
            for i, seq in enumerate(input_ids):
                input_ids[i] = self._add_special_tokens(seq, add_bos, add_eos)
                if max_len:
                    input_ids[i] = input_ids[i][:max_len]

        return {
            "input_ids": input_ids,
            "offset_mapping": encoded.get("offset_mapping", None),
            "was_truncated": max_len is not None,
            "attention_mask": attention_mask,
        }

    def decode(self, ids: Union[List[int], List[List[int]], torch.Tensor]) -> Union[str, List[str]]:
        """Decode token ids back to text. Supports single sequences and batches."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids[0], list):
            return self.tokenizer.batch_decode(ids)
        return self.tokenizer.decode(ids)






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
        self.bos_token_id = config.tokenizer.bos_token_id
        self.eos_token_id = config.tokenizer.eos_token_id
        self.pad_token_id = config.tokenizer.pad_token_id
        self.offset = 0

    def __call__(self, text: str) -> dict:
        byte_ids = [self.bos_token_id] + [b + self.offset for b in text.encode('utf-8', errors='ignore')] + [self.eos_token_id]
        return {'input_ids': byte_ids}

    def decode(self, ids: List[int]) -> str:
        special_tokens = {self.bos_token_id, self.eos_token_id, self.pad_token_id}
        bytes_data = bytearray([max(0, i - self.offset) for i in ids if i not in special_tokens])
        return bytes_data.decode('utf-8', errors='ignore')
