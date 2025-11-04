# Matformer
## ⚙️ Configuration Guide

### Configuration File Structure

The config JSON has 5 main sections:

#### 1. Model Configuration (`model_config`)

**Basic Architecture:**
```json
{
  "hidden_size": 512,           // Model dimension
  "num_hidden_layers": 4,       // Number of transformer layers
  "num_attention_heads": 8,     // Attention heads (must divide hidden_size)
  "ffn_factor": 1.0,           // FFN size = hidden_size * ffn_factor
  "max_position_embeddings": 1024  // Maximum sequence length
}
```

**Vocabulary & Tokens:**
```json
{
  "vocab_size": 32769,          // Size of vocabulary
  "pad_token_id": 0,           // Padding token
  "bos_token_id": 1,           // Beginning of sequence
  "eos_token_id": 2,           // End of sequence
  "mask_token_id": 32768       // Mask token (for MLM)
}
```

**Layer Configuration (`default_layer`):**
```json
{
  "attn_impl": "sdpa",                    // Attention: "flash", "sdpa", "flex", "wersa"
  "positional_encoding": "learnable",     // Position: "learnable", "rope", "alibi", "sinusoidal", "nope"
  "normalization": "layernorm",           // Norm type: "layernorm", "rmsnorm"
  "normalization_position": "pre",        // Norm position: "pre", "post"
  "ffn_activation": "gelu"               // Activation: "gelu", "swiglu"
}
```

**Training Objective:**
```json
{
  "training_objective": "masked",        // "masked" for BERT-style MLM, "causal" for GPT-style
  "is_causal": false,                   // false for BERT, true for GPT
  "masked_substitution_rate": 0.25      // Percentage of tokens to mask (MLM only)
}
```

#### 2. Training Configuration (`training`)
```json
{
  "optimizer": "adam",                   // Optimizer type: "adam", "adamw", "muon"
  "lr": 5e-4,                           // Learning rate
  "final_lr": 2e-5,                     // Final LR for scheduler
  "lr_scheduling": true,                 // Enable LR scheduling
  "scheduler": "custom",                 // Scheduler type
  "warmup_steps": 750,                  // Warmup steps
  "hold_steps": 300,                    // Steps to hold at max LR
  "weight_decay": 0.01,                 // Weight decay
  "gradient_clip_val": 1.0,             // Gradient clipping
  "max_epochs": 1,                      // Number of epochs
  "accumulate_grad_batches": 5,         // Gradient accumulation
  "save_every_n_steps": 5000,           // Checkpoint frequency
  "seed": 27                            // Random seed
}
```

#### 3. Tokenizer Configuration (`tokenizer`)
```json
{
  "type": "huggingface",                                    // Tokenizer type
  "pretrained_name": "sapienzanlp/Minerva-350M-base-v1.0", // HF model name
  "varlen_strategy": "padding"                              // "padding" or other strategies
}
```

#### 4. Data Configuration (`data`)
```json
{
  "data_root": "./datasets/mini_dataset_1000",  // Path to MDAT dataset
  "batch_size": 4,                              // Batch size
  "num_workers": 1,                             // DataLoader workers
  "mdat_strategy": "Minerva1024",               // Pretokenization strategy
  "mdat_view": null                             // Dataset view (optional)
}
```

#### 5. Logging & Checkpoints
```json
{
  "save_dir": "./checkpoints",         // Checkpoint directory
  "wandb_project": "Test",            // Weights & Biases project
  "wandb_run_name": "Test-Model"      // W&B run name
}
```