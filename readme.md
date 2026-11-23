# Matformer

Matformer is a modular PyTorch library for language-model research. Its autoregressive pretraining and masked language model (MLM) annealing workflows share configurable transformer blocks.

## Autoregressive pretraining

The autoregressive runner uses Lightning Fabric and reads pretokenized MDAT data. Start with an example configuration and set the data root, tokenizer path, vocabulary and special-token IDs to match your corpus. The standard example applies each block once. The looped example reuses the same four blocks three times per forward pass. [Standard configuration](configs/pretraining/italian.json) · [Looped configuration](configs/pretraining/italian_looped.json)

```bash
python train_model.py --config configs/pretraining/italian.json --device cuda
```

The configuration groups model architecture under `model_config`, optimizer and schedule under `training`, and input settings under `tokenizer` and `data`. The pretraining guide covers installation, distributed execution, exact token budgets and checkpoint selection. [Pretraining guide](docs/pretraining.md)

## Annealing

MLM annealing continues training a pretrained model on a prepared corpus. A fresh run loads the model weights and MLM head from the starting checkpoint and initializes a new optimizer. To compare training recipes, use the same validation documents and masks across runs.

Supply the starting weights, prepared data directory, passed preflight receipt, output directory, run ID and training mask rate. The current runner checks the starting checkpoint against a pinned SHA-256 digest. Using weights from another training step requires updating that check and validating the checkpoint.

From the repository root, inspect the supported options with `python -m scripts.pretraining.anneal_mlm --help`. The following is the per-task command for the current four-GPU distributed runner. Set the variables to your experiment's inputs before launch.

```bash
python -m scripts.pretraining.anneal_mlm \
  --checkpoint "$START_CHECKPOINT" \
  --data "$DATA_DIR" \
  --preflight "$PREFLIGHT_RECEIPT" \
  --output "$OUTPUT_DIR" \
  --run-id "$RUN_ID" \
  --rate "$MASK_RATE"
```

Choose a fresh output directory for each experiment. Follow training and validation in `logs/<run-id>/metrics.jsonl` under that directory. To resume after an interruption, keep the output directory and logs, repeat the original arguments, and add `--resume "$OUTPUT_DIR/last.pt"`. This restores the model, optimizer and per-rank random states. [Annealing runner](scripts/pretraining/anneal_mlm.py) · [Data preparation](scripts/data/prepare_mlm_annealing.py)

## Data loading with MDAT

An MDAT corpus stores named **domains** of source documents. A **strategy** defines tokenization and chunking, and a **view** identifies a set of documents such as the training split. A selection lets you reuse that view with a different domain mixture or token budget.

For example, this `data.mdat_selection` in an autoregressive config selects up to 7 million cached tokens from `wiki` and 3 million from `books`. Use domain names present in your view. Selection keeps whole documents, so the achieved counts can fall below either cap.

```json
{
  "domain_tokens": {"wiki": 7000000, "books": 3000000},
  "seed": 27
}
```

Inspect the dataset's `selection_summary` for actual document and token counts. The data guide also shows how to open a view directly from Python. [MDAT data-loading guide](matformer/data/README.md)

## Architecture and extensions

`TransformerBlock` combines attention, normalization and a feed-forward layer. `NakedTransformer` stacks these blocks, and `TransformerWithLMHead` adds the token embedding and vocabulary projection. Configure a shared layer layout with `default_layer` and override individual layers with `custom_layers`. [Transformer blocks](matformer/transformer_blocks/transformer.py) · [Model configuration](matformer/model_config.py)

The registry selects an implementation for each component, such as an attention kernel or a SwiGLU feed-forward layer. Custom implementations declare their dependencies, priority and parameter-name mappings with `registry.register`. Hooks such as `pre_attn` and `pre_mlp` let you attach experimental modules at specific points in a block. [Registry implementation](matformer/utils/matformer_registry_implementation.py) · [Hook guide](docs/hooks.md)
