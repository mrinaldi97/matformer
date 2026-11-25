---
title: Add a tokenizer
sources:
  - matformer/matformer_tokenizers.py
  - matformer/dataclasses/model_dataclasses.py
---
# Add a tokenizer

A tokenizer backend is a registry entry in the `tokenizer` category. For a backend compatible with a HuggingFace tokenizer, subclass `MatformerTokenizer` and override `_load_backend`.

## Class

```python
from matformer.matformer_registry import registry
from matformer.matformer_tokenizers import MatformerTokenizer

@registry.register("tokenizer", "huggingface", "my_backend", priority=100,
                   requires=["my_backend"])
class MyTokenizer(MatformerTokenizer):
    def _load_backend(self, source):
        import my_backend
        return my_backend.load(str(source))
```

`requires` is checked before the implementation is considered. If `my_backend` is not installed, the registry uses the next implementation. `priority=100` wins when the package is available. The variant is `huggingface` because the config section already uses that variant; another family would need a new config dataclass.

## Selection

The registry reads `MATFORMER_<CATEGORY>_<VARIANT>`:

```bash
MATFORMER_TOKENIZER_HUGGINGFACE=my_backend python train_model.py --cfg my.yaml
```

or programmatically:

```python
model = load_model(configs=configs, preferred="my_backend")
```

The config names the variant (`tokenizer: {type: huggingface, ...}`). The registry selects the implementation.

## Backends without the HuggingFace interface

Implement the methods called by the rest of the framework:

```python
class MyPureTokenizer:
    def encode(self, text, add_bos=False, add_eos=False, ...): ...
    def decode(self, ids): ...
    def tokens(self, ids): ...
```

The object also needs a `config` attribute with `bos_token_id`, `eos_token_id`, `pad_token_id`, `mask_token_id` and `vocab_size`.

## Test

```python
from matformer.matformer_registry import registry
tok = registry.create("tokenizer", "huggingface", config=configs.tokenizer)
ids = tok.encode("Ciao mondo", add_bos=True, add_eos=True)["input_ids"]
print(tok.decode(ids), tok.tokens(ids))
```

Check the round trip and the special-token ids. The data module uses `pad_token_id` for padding. The masked learning signal uses `mask_token_id`.

## Preprocessing

Text normalization before tokenization belongs to the dataset. The lazy dataset reads `text_column`; a custom dataset or a splitter function can normalize, strip, lowercase or split the text. The tokenizer maps text to ids.
