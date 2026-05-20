import os

# Training

matmul_precision=os.environ.get("TORCH_MATMUL_PRECISION",'high')
### Logging
progress_bar_steps = os.environ.get("LOG_PROGRESS_BAR_STEPS", 100)
use_wandb = os.environ.get("LOG_WANDB",True)
use_tqdm = os.environ.get("LOG_TQDM",True)

### DDP
gradient_as_bucket_view=os.environ.get("DDP_GRADIENT_AS_BUCKET", False)
find_unused_parameters=os.environ.get("DDP_FIND_UNUSED_PARAMETERS", False)

