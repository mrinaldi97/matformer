---
title: Train a model from a YAML file
sources:
  - train_model.py
  - run_task.py
  - matformer/training/launch.py
---
# Train a model from a YAML file

The two entry points are `train_model.py` and `run_task.py`. They do the same first step, loading a config, and then either train a model or run a task on a model.

## The smallest config

```yaml
model:
  type: language_model
  hidden_size: 256
  num_hidden_layers: 4
  ffn_factor: 4
  num_attention_heads: 8
  num_key_value_heads: 8
  vocab_size: 32000
  max_position_embeddings: 512
  is_causal: true
  default_layer:
    normalization: rmsnorm
    ffn_activation: swiglu
    sequence_mixer: attention

tokenizer:
  type: huggingface
  pretrained_name: sapienzanlp/Minerva-350M-base-v1.0

data:
  - name: train
    type: lazy
    data_root: data/train.jsonl
    file_format: jsonl
    batch_size: 8
    max_seq_len: 512
    task: lm

training:
  data: train
  learning_signals: [autoregressive]
  optimizer: adamw
  lr: 5e-4
  max_epochs: 1
  save_dir: checkpoints
  checkpoint_name: tiny
  logger: [sqlite, tqdm]

validation:
  - name: quick
    ref_task: test
    every_n_steps: 500

task:
  - name: test
    type: evaluation
    ref_data: train
    learning_signal: autoregressive
```

Run it:

```bash
python train_model.py --cfg tiny.yaml
```

`--sweeps` prints the plan without running it; any value containing a top-level `|` is a sweep. `--override training.lr=1e-3` patches one field, and `--override 'training.lr=5e-4|1e-4'` declares a sweep from the command line.

## What happens

1. `MatformerConfigs` loads the YAML, resolves `includes`, applies overrides and validates references.
2. `MatformerTrainer` builds fabric, decides whether to resume, loads or creates the model, builds the data module and the optimizers.
3. The loop runs, logging one row per optimizer step, validating when the `validation` section says so, and saving when the training config says so.
4. At the end, the model (and optionally the plugins) is saved in the checkpoint folder.

## Run a task on the result

```bash
python run_task.py --cfg tiny.yaml --model checkpoints/tiny_v_1_s_1000.ckpt
```

or, with the checkpoint declared in the config:

```yaml
checkpoint:
  - name: finale
    path: checkpoints/tiny_v_1_s_1000.ckpt
```

then:

```bash
python run_task.py --cfg tiny.yaml
```

`run_task.py` opens an interactive console. `task test`, `run`, `show test`, `stats` are the commands you will use first. The console is described in [[tasks/interactive-session | The interactive console]].
