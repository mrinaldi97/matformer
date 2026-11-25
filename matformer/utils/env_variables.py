# matformer/utils/env_variabless.py
import os


def _bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    return default if val is None else val.strip().lower() not in ("0", "false", "no", "")


# --- Training ---
# TORCH_MATMUL_PRECISION: torch's float32 matmul precision ("highest", "high", "medium"); defaults to "high"
matmul_precision = os.environ.get("TORCH_MATMUL_PRECISION", "high")

# --- Logging ---
# LOG_PROGRESS_BAR_STEPS: how often the tqdm progress bar refreshes, in optimizer steps
progress_bar_steps = int(os.environ.get("LOG_PROGRESS_BAR_STEPS", 100))
# LOG_WANDB: enable the wandb backend when the training config asks for it (default: true)
use_wandb = _bool("LOG_WANDB", True)
# LOG_TQDM: enable the tqdm progress bar (default: true)
use_tqdm = _bool("LOG_TQDM", True)

# --- DDP ---
# DDP_GRADIENT_AS_BUCKET: pass gradient_as_bucket_view to the DDP strategy (default: false)
ddp_gradient_as_bucket = _bool("DDP_GRADIENT_AS_BUCKET", False)
# DDP_FIND_UNUSED_PARAMETERS: pass find_unused_parameters to the DDP strategy (default: true)
ddp_find_unused_parameters = _bool("DDP_FIND_UNUSED_PARAMETERS", True)

# --- Registry ---
# MATFORMER_REGISTRY_VERBOSE: print the registry's selection notices (default: true)
registry_verbose =  _bool("MATFORMER_REGISTRY_VERBOSE", True)
# MATFORMER_<CATEGORY>_<VARIANT>: force one implementation of a registry variant, e.g. MATFORMER_ATTENTION_NORMAL=sdpa;
# the category and variant are upper-cased and non-alphanumeric characters become underscores
# MATFORMER_SERVER: start the optional matformer server when the package is imported (default: true; set 0 to disable)
# MATFORMER_SERVER_ADDRESS: host:port the CLI looks for the server at (default: 127.0.0.1:8765)
# MATFORMER_SERVER_PORT: port the optional server binds when the server is enabled
# MATFORMER_API_KEY: key sent with server requests, when the server requires one

# --- Paths and assets ---
# MATFORMER_ASSETS: absolute path of the shared folder holding configs, checkpoints, datasets, logs, venvs, experiments and extra_plugins;
# it wins over the `assets:` entry of configuration.yaml in the repo root
# MATFORMER_MDAT_ROOT: root used to resolve portable, relative MDAT paths
# MATFORMER_ROOT: root of the matformer checkout, used by the publishing helpers
# HF_HOME: standard HuggingFace cache; passed through to launched jobs

# --- Model components ---
# ALLOW_WEIGHT_TYING: allow tying the embedding and LM-head weights (default: true; set false to keep them separate)
# LEGACY_TOKENIZER: load the legacy tokenizer format instead of the current one
# USE_ZLIB: compress the LMDB stores with zlib instead of the default backend (default: false)

# --- Control and jobs ---
# MATFORMER_CONTROL: path of the control mailbox the trainer polls (see matformer/training/control.py)
# MATFORMER_STATUS: path of the heartbeat JSON the trainer writes
# MATFORMER_JOB: job name recorded in the heartbeat
# SLURM_PARTITION: default partition read by the Slurm helper when it is set
# SLURM_PROCID: rank read by the Slurm helper and recorded in the heartbeat
# CUDA_VISIBLE_DEVICES: standard CUDA device selection; also set by the launcher for a specified GPU list
