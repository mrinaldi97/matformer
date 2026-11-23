#!/bin/bash
# Source after environment activation. Keep application temporary files on WORK,
# never on Leonardo's job-local /tmp or login-node /scratch_local.
matformer_storage_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
case "$matformer_storage_root" in
    /leonardo_work/*) ;;
    *) echo 'Leonardo launchers require a checkout under /leonardo_work.' >&2; return 1 ;;
esac
matformer_runtime_dir="$matformer_storage_root/.runtime/${SLURM_JOB_ID:-local}"
export TMPDIR="$matformer_runtime_dir/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR" SQLITE_TMPDIR="$TMPDIR"
export JOBLIB_TEMP_FOLDER="$TMPDIR"
export XDG_CACHE_HOME="$matformer_runtime_dir/cache"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export WANDB_CACHE_DIR="$XDG_CACHE_HOME/wandb"
export WANDB_CONFIG_DIR="$matformer_runtime_dir/wandb-config"
export WANDB_DATA_DIR="$matformer_runtime_dir/wandb-data"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" \
    "$HF_HOME" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR"
printf 'Application temporary files and caches: %s\n' "$matformer_runtime_dir"
