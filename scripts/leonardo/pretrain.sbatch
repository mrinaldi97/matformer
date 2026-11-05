#!/bin/bash
#SBATCH --job-name=italian-pretrain
#SBATCH --account=IscrC_ALBERTIN
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --signal=USR1@900
#SBATCH --output=logs/pretrain-%j.out
#SBATCH --error=logs/pretrain-%j.out

# Submit from the repository root. Create logs/ before sbatch.
# One srun task per GPU; Fabric reads SLURM_PROCID/LOCALID/NODEID.
# Do not nest torchrun or another process launcher inside this srun.
set -eo pipefail
matformer_wandb_mode="${WANDB_MODE:-offline}"
if [[ -n "${MATFORMER_ACTIVATE:-}" ]]; then
    source "$MATFORMER_ACTIVATE"
fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="$matformer_wandb_mode"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

config="${1:-configs/pretraining/italian.json}"
if [[ $# -gt 0 ]]; then shift; fi
gpus_per_node="${SLURM_NTASKS_PER_NODE:?Set one task per GPU}"
if [[ ! "$gpus_per_node" =~ ^[0-9]+$ ]]; then
    echo 'Use a uniform integer --ntasks-per-node matching GPUs per node.' >&2
    exit 2
fi
srun --unbuffered --gpu-bind=none python -u train_model.py \
    --config "$config" --gpu "$gpus_per_node" --nodes "$SLURM_NNODES" \
    --device cuda --precision bf16-mixed "$@"
