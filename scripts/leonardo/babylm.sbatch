#!/bin/bash
#SBATCH --job-name=babylm-smoke
#SBATCH --account=IscrC_ALBERTIN
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --signal=USR1@180
#SBATCH --chdir=/leonardo_work/IscrC_ALBERTIN/matformer
#SBATCH --output=logs/babylm-%j.out
#SBATCH --error=logs/babylm-%j.out

set -eo pipefail
matformer_wandb_mode="${WANDB_MODE:-offline}"
source /leonardo_work/IscrC_ALBERTIN/cscozzar/activate_matformer.sh
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_ENTITY=team-jerik
export WANDB_PROJECT=UniTO-LLM
export WANDB_MODE="$matformer_wandb_mode"

mode="${1:-smoke}"
config=configs/babyLM/babyLM_piccolo_1epoch_leonardo.json
args=(--config "$config" --gpu 1 --nodes 1 --device cuda --precision bf16-mixed)
case "$mode" in
  smoke)
    args+=(--start-from-scratch --max-steps 2 --override
      training.checkpoint_name=babylm_piccolo_smoke
      save_dir=./checkpoints_babylm_smoke
      wandb_run_name=BabyLM-piccolo-smoke
      training.save_every_n_steps=1
      training.validate_every_n_steps=1
      training.validation_max_batches=2)
    ;;
  full)
    # The existing entry point resumes the latest checkpoint with this name,
    # or starts from scratch when none exists.
    ;;
  *) echo 'Usage: sbatch [Slurm options] scripts/leonardo/babylm.sbatch smoke|full' >&2; exit 2 ;;
esac
printf 'Mode: %s; job: %s; W&B: %s/%s (%s)\n' "$mode" "$SLURM_JOB_ID" "$WANDB_ENTITY" "$WANDB_PROJECT" "$WANDB_MODE"
nvidia-smi -L
srun --unbuffered python -u train_model.py "${args[@]}"
