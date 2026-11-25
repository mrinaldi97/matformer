---
title: Use a Matformer model from HuggingFace
sources:
  - matformer/loading/huggingface_integration.py
  - matformer/loading/huggingface_holder.py
---
# Use a Matformer model from HuggingFace

## Load a Matformer checkpoint

If the repository contains a `matformer.yaml` (this is what `save_pretrained` writes), the HF shell finds it:

```python
from matformer.loading.huggingface_integration import MatformerForCausalLM

model = MatformerForCausalLM.from_pretrained("org/my-matformer-checkpoint", device="cuda")
text = model.generate(input_ids=model.get_tokenizer()("Hello")["input_ids"], max_new_tokens=32)
```

`from_pretrained` accepts `num_labels` and `pooling_type` for the classification classes. `device_map` is accepted for compatibility but ignored: Matformer does not shard a model across devices. Use `device="cuda"` or call `.to()`.

## Load a plain HuggingFace model into Matformer

```python
from matformer.matformer_wrapper import load_model

result = load_model(path="hf://sapienzanlp/Minerva-350M-base-v1.0",
                    map_location="cuda", inference=True)
model, configs = result["model"], result["configs"]
```

The `hf://` prefix selects the HF checkpoint holder. A local folder works without it. The returned model is a normal Matformer model, with `model(tokens)`, `model.generate(...)`, plugins, sockets and the stable state dict.

## Fine-tune from a HF checkpoint

```yaml
model:
  type: classification
  base_model: "hf://sapienzanlp/Minerva-350M-base-v1.0"
  base_model_hf: true
  num_classes: 3
  pooling_type: mean

training:
  data: train
  optimizer: adamw
  lr: 2e-5
  max_epochs: 3
  freeze: ["all"]
  unfreeze: ["classification"]
```

`base_model` tells the trainer to inherit the model and tokenizer sections from the checkpoint. The `freeze`/`unfreeze` pair keeps the backbone fixed and trains only the classification plugin. Remove them to fine-tune the whole model.

## Publish it back

```python
model.save_pretrained("out/my-model")
```

The folder contains `model.safetensors`, `matformer.yaml`, `config.json`, `modeling_matformer.py` and the tokenizer. It loads with `MatformerForCausalLM.from_pretrained` and, through `auto_map`, with the `transformers` API.

If the model uses a custom module, copy the module into `MATFORMER_ASSETS/extra_plugins` before publishing. `modeling_matformer.py` points at the installed package, so the custom module must be installed or on that path.
