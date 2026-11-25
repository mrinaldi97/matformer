"""
File: matformer/tokenizers.py
"""
import json
import warnings
from pathlib import Path

import torch
from typing import List, Optional, Union
from matformer.dataclasses.model_dataclasses import TokenizerConfig
from matformer.matformer_registry import registry

@registry.register("tokenizer", "huggingface", "default", priority=0)
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
    SPECIAL_TOKENS = ("bos_token", "eos_token", "pad_token", "mask_token")
    chat_template = None   # the model's Jinja ChatTemplate, attached by the wrapper at load time

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
            if config.pretrained_name is None:
                self.tokenizer=None
            else:
                self.tokenizer = self._load_backend(config.pretrained_name)
                if reload_hf_cfg:
                    self._reload_huggingface_config()
                else:
                    self._sync_from_backend()

        else:
            assert tokenizer is not None
            self.tokenizer = self._load_backend(tokenizer)
            self._reload_huggingface_config()

    def _sync_from_backend(self) -> None:
        """Fill the special-token ids the config left unset with the ones of the loaded backend,
        so a checkpoint declaring only `pretrained_name` is enough to encode with. Backends that
        keep the special tokens out of the object they expose (gigatoken's facade) are resolved
        through the tokenizer_config.json of their source instead."""
        missing = [name for name in self.SPECIAL_TOKENS if getattr(self.config, f"{name}_id", None) is None]
        if not missing:
            return
        names = dict(getattr(self.tokenizer, 'special_tokens_map', None) or {})
        names.update({name: getattr(self.tokenizer, name, None) for name in self.SPECIAL_TOKENS})
        if not any(names.get(name) for name in missing):
            names.update(self._tokenizer_config())
        for name in missing:
            token = names.get(name)
            token = next((t for t in token if isinstance(t, str)), None) if isinstance(token, list) else token
            if token:
                setattr(self.config, f"{name}_id", self.token_id(token))

    def _tokenizer_config(self) -> dict:
        """The tokenizer_config.json of the backend's source, as a plain dict."""
        source = getattr(self.tokenizer, 'name_or_path', None)
        if not source:
            return {}
        path = Path(source)
        if not path.is_dir():
            from huggingface_hub import hf_hub_download
            path = Path(hf_hub_download(source, 'tokenizer_config.json')).parent
        path = path / 'tokenizer_config.json'
        return json.loads(path.read_text()) if path.exists() else {}

    def token_id(self, token: str) -> Optional[int]:
        convert = getattr(self.tokenizer, 'convert_tokens_to_ids', None)
        return convert(token) if convert is not None else None

    def _load_backend(self, source):
        """Instantiate the underlying tokenizer from a pretrained name; a ready-made instance
        is passed through untouched. Subclasses override this to plug other backends."""
        if isinstance(source, str):
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(source, trust_remote_code=True)
        return source

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
        """Prepend BOS and/or append EOS token id to a sequence list. A tokenizer without one of
        them (bos_token_id is None for e.g. every Qwen) simply leaves that side untouched."""
        if add_bos and self.config.bos_token_id is not None:
            seq = [self.config.bos_token_id] + seq
        if add_eos and self.config.eos_token_id is not None:
            seq = seq + [self.config.eos_token_id]
        return seq

    def chat_specials(self) -> dict:
        """{name: token-string} for the special tokens a chat template may reference (bos, eos,
        pad, mask). Resolved through the backend so a template that writes the eos token by name,
        or splices bos/eos itself, sees the real strings. Names left unset in the config are dropped."""
        specials = {}
        for name in self.SPECIAL_TOKENS:
            token_id = getattr(self.config, f"{name}_id", None)
            if token_id is None:
                continue
            surface = getattr(self.tokenizer, "convert_ids_to_tokens", None)
            specials[name] = surface(token_id) if surface is not None else self.decode([token_id])
        return specials

    def encode_tokens(self, text: str) -> List[int]:
        """Bare token ids of one string, no special tokens added: the callable a ChatTemplate uses
        to tokenize content and template segments. The template owns bos/eos, never this."""
        return self.encode(text, add_bos=False, add_eos=False)["input_ids"][0]

    def encode(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
        add_eos: bool = False,
        add_bos: bool = False,
        return_offsets_mapping: bool = False,
        is_split_into_words: bool = False,
        return_word_ids: bool = False,
    ) -> dict:
        """
        Encode text into token ids.

        Args:
            text (str or List[str]): input text or batch of texts to encode. With
                is_split_into_words it is a pre-tokenized sequence (a list of words), and the
                tokenizer keeps the word boundaries instead of re-splitting the text
            max_length (int, optional): maximum sequence length for this call; overrides self.max_seq_len
            add_bos (bool): prepend BOS token id to each sequence
            add_eos (bool): append EOS token id to each sequence
            return_offsets_mapping (bool): whether to return character-to-token offset mapping
            is_split_into_words (bool): treat 'text' as already split into words
            return_word_ids (bool): also return, per sequence, the word each subword token comes
                from (None for special tokens); needed to align word-level labels on subwords

        Returns:
            dict with keys: input_ids, offset_mapping, was_truncated, attention_mask
            ('word_ids' is added when return_word_ids is set).
            (if self.return_tensors is "padded_tensor"/"unpadded_tensor", input_ids is a TensorDC and attention_mask is None,
            since the mask is embedded in the returned PaddedTensor/UnpaddedTensor instead)
        """
        if isinstance(text, str):
            text = [text]
        elif is_split_into_words and text and isinstance(text[0], str):
            # A list of words is ONE pre-tokenized sequence for HuggingFace, which would return it
            # unbatched: wrapping it keeps this encode() batch-shaped on every input, as the
            # '[0]'-style indexing done by the callers (and the word_ids below) assumes.
            text = [text]

        max_len = max_length or self.max_seq_len

        if self.return_tensors:
            assert max_len is not None, "max_seq_len must be set at initialization when return_tensors=True"
        if return_word_ids and self.return_tensors:
            raise ValueError(
                "return_word_ids is only supported in list mode: padding/unpadding would move the "
                "tokens away from the word they were aligned to"
            )

        # Any tensor-returning mode (incl. padded_tensor/unpadded_tensor) needs real "pt" tensors from HF first.
        kwargs = dict(
            return_offsets_mapping=return_offsets_mapping,
            return_tensors=(self.TENSOR_TYPE if self.return_tensors else None),
        )
        if is_split_into_words:
            kwargs["is_split_into_words"] = True
        if self.return_tensors:
            reserved = int(add_bos) + int(add_eos)
            effective_max_len = max_len - reserved if (add_bos or add_eos) else max_len
            kwargs.update(padding="max_length", max_length=effective_max_len, truncation=True)

        encoded = self.tokenizer(text, add_special_tokens=False, **kwargs)
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask", None)
        word_ids = None
        if return_word_ids:
            if not hasattr(encoded, "word_ids"):
                raise NotImplementedError(
                    f"{type(self.tokenizer).__name__} does not expose word_ids; word-level label "
                    f"alignment needs a fast HuggingFace tokenizer"
                )
            word_ids = [encoded.word_ids(index) for index in range(len(input_ids))]
        if torch.is_tensor(input_ids) and input_ids.dtype != torch.long:
            input_ids = input_ids.long()  # gigatoken's "pt" buffers are int32
        if torch.is_tensor(attention_mask) and attention_mask.dtype != torch.long:
            attention_mask = attention_mask.long()

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
            if word_ids is not None:
                # Special tokens are added outside the tokenizer, so they have no word of origin:
                # they become None, exactly like a subword that belongs to no word. The slice
                # mirrors the one above, so ids and word_ids stay position by position aligned.
                word_ids = [([None] * int(add_bos) + wids + [None] * int(add_eos))[:max_len]
                            for wids in word_ids]

        return {
            "input_ids": input_ids,
            "offset_mapping": encoded.get("offset_mapping", None),
            "was_truncated": max_len is not None,
            "attention_mask": attention_mask,
            "word_ids": word_ids,
        }

    def decode(self, ids: Union[List[int], List[List[int]], torch.Tensor]) -> Union[str, List[str]]:
        """Decode token ids back to text. Supports single sequences and batches."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids[0], list):
            return self.tokenizer.batch_decode(ids)
        return self.tokenizer.decode(ids)

    def tokens(self, ids: Union[List[int], torch.Tensor]) -> List[str]:
        """Surface form of every id, one string per token (subword pieces included), so that a
        prediction can be shown next to the token it was made for."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        convert = getattr(self.tokenizer, 'convert_ids_to_tokens', None)
        if convert is not None:
            return list(convert(ids))
        return [self.decode([one]) for one in ids]


@registry.register("tokenizer", "huggingface", "gigatoken", priority=10,
                   requires=["gigatoken"], constraints={"offset_mapping": False})
class GigatokenTokenizer(MatformerTokenizer):
    """
    gigatoken backend (https://github.com/marcelroed/gigatoken)
    """

    def _load_backend(self, source):
        """Wrap the model with gigatoken's HF-compatible facade; raises (after a warning) if
        gigatoken cannot load it, so the registry moves on to the next implementation."""
        import gigatoken
        try:
            backend = gigatoken.Tokenizer(source).as_hf()
        except Exception as error:
            warnings.warn(f"gigatoken cannot load {source!r} ({error}); "
                          "falling back to the default huggingface tokenizer implementation.")
            raise
        backend.name_or_path = source if isinstance(source, str) else getattr(source, "name_or_path", "")
        return backend

    def encode(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
        add_eos: bool = False,
        add_bos: bool = False,
        return_offsets_mapping: bool = False,
        is_split_into_words: bool = False,
        return_word_ids: bool = False,
    ) -> dict:
        if return_offsets_mapping:
            raise NotImplementedError(
                "the gigatoken implementation has no offset mapping; create the tokenizer with "
                "registry.create('tokenizer', 'huggingface', constraints=['offset_mapping'], ...) "
                "to get the huggingface implementation")
        if return_word_ids:
            raise NotImplementedError(
                "the gigatoken implementation has no word mapping; create the tokenizer with "
                "registry.create('tokenizer', 'huggingface', ...) to get the huggingface one")
        return super().encode(text, max_length=max_length, add_eos=add_eos, add_bos=add_bos,
                              is_split_into_words=is_split_into_words)





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
