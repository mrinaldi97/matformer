---
title: Tokenizers
sources:
  - matformer/matformer_tokenizers.py
  - matformer/dataclasses/model_dataclasses.py
category_docs: [tokenizer]
---
# Tokenizers

Matformer does not depend on a particular tokenizer library. It wraps a backend in `MatformerTokenizer`, which is what the rest of the framework expects: a small object with `encode`, `decode`, `tokens`, the special-token ids and a `TokenizerConfig`.

The default backend is a HuggingFace tokenizer, loaded from `pretrained_name`. The same class also accepts a ready-made tokenizer instance, which is how the byte-level tokenizers in the same file bypass the `transformers` dependency. A `gigatoken` backend is registered as a separate implementation with higher priority and `requires: [gigatoken]`. When the package is installed it becomes the default; force the HuggingFace backend with `MATFORMER_TOKENIZER_HUGGINGFACE=default` if you need the old one.

## Configuration

```yaml
tokenizer:
  type: huggingface
  pretrained_name: sapienzanlp/Minerva-350M-base-v1.0
  bos_token_id: 1
  eos_token_id: 2
  pad_token_id: 0
  mask_token_id: 32768
```

The four special-token ids are optional. When they are missing, `MatformerTokenizer` fills them from the loaded backend; for backends whose facade hides the special tokens, it reads `tokenizer_config.json` from the source. This means a checkpoint that declares only `pretrained_name` is still enough to encode and decode.

The tokenizer config is available as `configs.tokenizer`, and `configs.tokenizer.instantiate()` returns the wrapper. A model built by the wrapper gets `model.tokenizer` and `model.tokenizer.config` attached, which is what tasks use.

## Encoding

`encode` accepts text or a list of texts and returns a dict, by default with `input_ids` and `attention_mask`. The `return_tensors` option controls the rest:

* `True` or `"pt"`: a HuggingFace-style padded `torch` tensor;
* `"padded_tensor"`: a [[components/tensors | PaddedTensor]], with `padding_mask` derived from the attention mask, so it stays correct when `pad_token_id == eos_token_id`;
* `"unpadded_tensor"`: an `UnpaddedTensor`, ready for the packed path;
* `False`: plain Python lists.

`max_seq_len` truncates. `add_bos` and `add_eos` control the special-token decoration. `tokens(ids)` returns the string pieces, which is what the task consoles use to show a token next to its prediction, and `chat(messages)` applies the backend's chat template when one exists.

## Adding a tokenizer

A custom tokenizer is a class registered in the `tokenizer` category. Subclassing `MatformerTokenizer` and overriding `_load_backend` is enough if the backend behaves like a HuggingFace tokenizer. If it does not, implement `encode`, `decode` and `tokens` directly; the rest of the framework only calls those.

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

The registry key for the default is `tokenizer.huggingface.default`; a priority above zero wins when the required package is importable. See the [[tutorials/custom-tokenizer | custom tokenizer tutorial]] for the full walkthrough.

## MDAT tokenizers

The MDAT pretokenization strategies use tokenizers from the data layer; see [[data/mdat | MDAT]]. The byte-level tokenizers in `matformer_tokenizers.py` are minimal helpers for tokenizer-free experiments.
