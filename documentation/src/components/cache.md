---
title: The cache
sources:
  - matformer/utils/matformer_cache.py
  - matformer/utils/kv_cache.py
category_docs: [activation_store]
---
# The cache

`CachedStuff` is the object every model and plugin receives as `cache`. It stores precomputed masks, ALiBi biases and rotary tables, so they are not rebuilt at every layer and every step.

It is better to create one at the highest possible level of the model. The wrapper creates one in `MatformerModelWrapper.__init__`, the trainer passes it through, and `ensure_cache(cache)` builds one if a caller forgot. When a model is constructed without a cache you get a warning, not an error.

## What it holds

| Attribute | Contents |
|-----------|----------|
| `slope_cache` | ALiBi slopes, keyed by head count, device and dtype |
| `bias_cache` | ALiBi bias matrices, keyed by length, head count, device and dtype |
| `mask_cache` | FlexAttention block masks, keyed by shape and mask options |
| `rotary_emb_cache` | `cos`/`sin` tables for RoPE, including YaRN and Llama-3 scaling |
| `cu_seqlens_cache` | `cu_seqlens` as a Python list, with at most one device sync per tensor |
| `storage` | free-form dictionary shared by plugins and learning signals |
| `layer_states` | per-layer sequential state, for recurrent mixers |
| `kv_layers` | what each attention layer says a cached step needs |

The methods are the stable interface: `get_alibi_bias`, `get_alibi_slopes`, `get_flex_blockmask`, `get_rotary_cos_sin`, `get_cu_seqlens_cpu`, `clear_cache`, `get_cache_size`. They are called by the registry implementations; you normally do not call them directly.

## The storage dictionary

`cache.storage` is a shared dictionary for plugins and learning signals. A probe writes under `probe_outputs`; the training step reads `aux_losses` (for example a MoE router loss). There is no schema.

Two keys are special to the framework:

* `aux_losses`: a list of tensors. The wrapper sums them into the total training loss and logs them separately as `train/aux_loss`.
* `probe_outputs`: a dictionary from socket address (with `@n` for the n-th visit in a looped model) to the probe's logits. `MultiProbeClassifier` and `VirtualModel` return it.

## The KV cache

The paged KV cache used by generation is also owned by `CachedStuff`. A mixer registers what it needs with `register_kv_layer(layer_idx, num_kv_heads, head_dim, *metadata)` when it is built. `kv_cache(tokens, ...)` then creates one cache for the whole model, or raises if a layer has no cache-aware kernel.

The cache does not resize itself. A conversation that outgrows it must be cleared with `clear` in the task console or with `reset_kv_cache()`. Growing the cache during a live conversation would invalidate the stored state and increase memory at every turn.

## Clearing

`clear_cache()` drops the shape-keyed caches (slopes, biases, masks, rotary). The trainer's memory probe calls it between attempts, because a new batch size would otherwise leave another mask matrix in memory. `clear_cache('masks')` clears one family.
