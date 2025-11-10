# Matformer
Matformer is a library designed to train and execute state-of-the-art transformer models. 

Unlike common implementations, such as HuggingFace’s transformers or Nvidia’s Megatron, it is designed from scratch with simplicity and modularity in mind but without renouncing to all the possible improvements to make transformers’ training more efficient.

As the research on the subject goes on, several novelties are introduced in the libraries. However, this often leads to an explosion of libraries’ complexity, where new functions are introduced abruptely in the codebase, for example by introducing many conditional branches or by repeating many times entire part of the code, breaking the DRY principles.

The goal of Matformer is to produce a very compact and clear code, with the smallest possible amount of repetition, in line with principles of software design and easily extensible with additional modules that allows important modification to the models’ definitions without altering the core parts of the library.
Matformers’ core is so easy to read that it is ready to be used also for didactical purposes: teachers are given with the possibility to present to their student an easy to understand codebase and guide them in the understanding of how a transformer model works. 
Matformer’s main focus is on elegancy and readability of the code, an aspect often ignored in the world of codebases for deep-learning models’ training.

This allows a twofold use of Matformer: on the one hand, it is a library ready for the development of models with many different possible configurations ecompassing the most common scenario adopted in 2025 for training models; on the other hand, it is a library designed from the Academia to the Academia: this means that it gives to researchers from all over the world an easily hackable transformer implementation ready to embrace many possible experimental scenarios. If, let’s say, a particular custom module has to be added or modifed with respect to the base Transformer design, it is not necessary to alter even a single row of code of the library itself, but the module can be easily imported and used through the configuration files. Matformer will also take care to pack the custom code in the models’repository ready to be uploaded in platforms such as HuggingFace. 

This doesn’t come at expenses of speed: it is very important to exploit as much as possible the capabilities of the hardware when training complex models such as transformers. That’s why Matformer already contains the code necessary to maximize training’s performances, such as FlashAttention, sequence packing and high-performance fused kernels.

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
