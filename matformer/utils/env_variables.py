# matformer/utils/env_variabless.py
import os


def _bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    return default if val is None else val.strip().lower() not in ("0", "false", "no", "")


# Training
matmul_precision = os.environ.get("TORCH_MATMUL_PRECISION", "high")

# Logging
progress_bar_steps = int(os.environ.get("LOG_PROGRESS_BAR_STEPS", 100))
use_wandb = _bool("LOG_WANDB", True)
use_tqdm = _bool("LOG_TQDM", True)

# DDP
ddp_gradient_as_bucket = _bool("DDP_GRADIENT_AS_BUCKET", False)
ddp_find_unused_parameters = _bool("DDP_FIND_UNUSED_PARAMETERS", True)

# Registry
registry_verbose =  _bool("MATFORMER_REGISTRY_VERBOSE", True)

