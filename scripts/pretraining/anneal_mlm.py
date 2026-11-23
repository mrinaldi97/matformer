"""Run one independent AlBERTone101 annealing experiment in a four-GPU allocation."""
import argparse
import os
from pathlib import Path
import torch
from matformer.training.mlm_annealing import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--rate', type=float, choices=[.15, .28, .40], required=True)
    parser.add_argument('--checkpoint', type=Path, default=Path('checkpoints/retrieval/sources/AlBERTone101/checkpoint.ckpt'))
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=271118)
    parser.add_argument('--micro-tokens', type=int, default=16384)
    parser.add_argument('--accumulation', type=int, default=32)
    parser.add_argument('--validate-every', type=int, default=100)
    parser.add_argument('--save-every', type=int, default=250)
    parser.add_argument('--max-seconds', type=int, default=27900)
    parser.add_argument('--smoke', type=int, default=0)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Training requires a Slurm allocation')
    if not args.smoke and args.preflight is None:
        parser.error('Production requires a passed preflight receipt')
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', 8)))
    train(args)


if __name__ == '__main__':
    main()
