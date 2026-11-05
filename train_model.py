"""Train or resume autoregressive pretraining from a JSON configuration."""
import argparse
import torch
from matformer.training.prepare_training import instantiate_training
from matformer.training.training_loop import launch_training_loop
from matformer.training.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--override', nargs='*', default=[], help='Dotted key=JSON_value overrides')
    parser.add_argument('--gpu', type=int, default=1, help='GPUs per node (CPU processes for tests)')
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--checkpoint', help='Explicit trusted checkpoint; otherwise resume latest')
    parser.add_argument('--weights-only', action='store_true', help='Start a new experiment from weights')
    parser.add_argument('--next-data-stage', action='store_true', help='Continue a completed stage on new data')
    parser.add_argument('--start-from-scratch', action='store_true', help='Start a new checkpoint version')
    parser.add_argument('--max-steps', type=int, help='Absolute optimizer-step stopping limit')
    parser.add_argument('--ckpt-version', type=int)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--precision', choices=['bf16-mixed', '16-mixed', '32-true'], default='bf16-mixed')
    args = parser.parse_args()
    if min(args.gpu, args.nodes) < 1:
        parser.error('--gpu and --nodes must be positive')
    if args.start_from_scratch and args.checkpoint:
        parser.error('Choose either --start-from-scratch or --checkpoint')
    overrides = {}
    for item in args.override:
        if '=' not in item:
            parser.error(f'Override must be key=value: {item}')
        key, value = item.split('=', 1)
        overrides[key] = value
    return args, overrides


def main():
    torch.set_float32_matmul_precision('high')
    args, overrides = parse_args()
    configs = load_config(args.config, overrides)
    fabric, state = instantiate_training(*configs, device=args.device, precision=args.precision,
        num_nodes=args.nodes, device_count=args.gpu, checkpoint=args.checkpoint,
        start_from_scratch=args.start_from_scratch, max_steps=args.max_steps,
        version=args.ckpt_version, _compile=args.compile,
        weights_only=args.weights_only, next_data_stage=args.next_data_stage)
    launch_training_loop(fabric, state)
    fabric.print('Training complete. Bye.')


if __name__ == '__main__':
    main()
