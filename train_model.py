"""
A function to train a generic model using the matformer library
A JSON config file must be provided
But each argument can be overridden from the CLI

Extended with CSV-based iteration:
  --iter-cfg FILE         : iterate over config files listed in CSV (first column)
  --iter-override FIELD FILE : iterate over values for FIELD from CSV (first column)
                             multiple --iter-override combine via Cartesian product
"""
# train_model.py
import argparse
import csv
import gc
import itertools
import torch
from matformer.training.matformer_trainer import MatformerTrainer
from matformer.training.utils import detect_device
from matformer.matformer_configs import MatformerConfigs
from matformer.utils import env_variables

# CLI dest -> TrainingConfig field name, wherever they differ (gpu/nodes/ckpt_version/reset
# are CLI-friendly names; the config fields have their own established names).
CLI_TO_TRAINING_FIELD = {
    'gpu': 'device_count', 'nodes': 'num_nodes', 'reset': 'what_to_reset',
    'ckpt_version': 'version', 'checkpoint': 'checkpoint',
    'start_from_scratch': 'start_from_scratch', 'max_steps': 'max_steps', 'compile': 'compile',
}

def parse_args():
    """
    Specific for train_model.py, returns CLI arguments and overrides dict to be used in main() function.
    """
    parser = argparse.ArgumentParser(description='Train a Matformer model')
    parser.add_argument("--cfg", action="append", required=False, metavar="YAML", help="Pass one or more YAMLs via --cfg; they merge left-to-right.")
    parser.add_argument("--override", action="append", required=False, help="Pass in-line overrides (ex. data.batch_size=48) they merge left-to-right.")
    parser.add_argument('--gpu', type=int, default=1, help="Number of GPU(s)")
    parser.add_argument('--nodes', type=int, default=1, help="Number of Node(s)")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to checkpoint file (use reset to decide if to do a partial or full resume)")
    parser.add_argument('--reset', type=str, default=None, help="What do you want to reset from the checkpoint, use comma separated values: '['lr_scheduler', 'optimizer', 'datamodule', 'steps']'")
    parser.add_argument('--start-from-scratch', action='store_true', help="Start training from scratch, advance version if checkpoint already exists")
    parser.add_argument('--max-steps', type=int, default=None, help="If you choose this, train for one epoch on this number of steps")
    parser.add_argument('--ckpt-version', type=int, default=None, help="If restarting a checkpoint, decide to restart from a specific version")
    parser.add_argument('--compile', action='store_true', help="Torch.compile the whole model")
    parser.add_argument('--device', type=str, default='auto', help="Device. 'auto' is default (choose the best), supported 'cuda','mps','cpu', 'tpu'")
    parser.add_argument('--no-save-model', action="store_true", help="Skip model saving")
    parser.add_argument('--no-save-plugins', action="store_true", help="Skip plugins saving")

    # Arguments for CSV-based iteration
    parser.add_argument('--iter-cfg', type=str, default=None,
                        help='CSV file with list of config YAML paths (one per row). '
                             'If provided, iterate over these configs, merging with --cfg if given.')
    parser.add_argument('--iter-override', action='append', nargs=2, metavar=('FIELD', 'FILE'), default=None,
                        help='CSV file with list of values for a given field. Can be specified multiple times. '
                             'The values are combined via Cartesian product.')

    args = parser.parse_args()
    if args.reset is not None:
        for a in args.reset.split(','):
            assert a.strip() in ['lr_scheduler', 'optimizer', 'datamodule', 'steps']
    assert args.device in ['auto','cuda','cpu','mps', 'tpu']
    return args, parser


def read_csv_values(filepath):
    """Read the first column of a CSV file, skip empty lines, return stripped strings."""
    values = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip():
                values.append(row[0].strip())
    return values


def build_configs(args, parser, config_files, overrides):
    """Create the configs and apply the CLI training flags that differ from their defaults."""
    configs = MatformerConfigs.create_configs(config_files, overrides=overrides)
    training_overrides = {
        field: getattr(args, dest) for dest, field in CLI_TO_TRAINING_FIELD.items()
        if getattr(args, dest) != parser.get_default(dest)
    }
    if training_overrides:
        configs.load_config(stdin={"training": training_overrides})
    return configs


def main():
    torch.set_float32_matmul_precision(env_variables.matmul_precision)
    # 1. Parse the arguments and overrides
    args, parser = parse_args()
    if args.device == 'auto':
        device = detect_device()
    else:
        device = args.device
    if device == 'cuda':
        assert torch.cuda.is_available()

    # 2. Determine if we are iterating
    is_iterating = args.iter_cfg is not None or args.iter_override is not None

    if not is_iterating:
        # Single run 
        configs = build_configs(args, parser, args.cfg, args.override)
        trainer = MatformerTrainer(configs, device=device)
        trainer.run(save_at_end=True,save_model=not(args.no_save_model), save_plugins=not(args.no_save_plugins))
        print("Training complete.  Bye.")
        return

    # Iteration mode 
    # Prepare the config rows
    if args.iter_cfg:
        cfg_rows = read_csv_values(args.iter_cfg)
        if not cfg_rows:
            print("Warning: --iter-cfg file is empty. No experiments will be run.")
            return
    else:
        cfg_rows = [None]  # one run with base config only

    # Prepare the override combinations
    override_fields = []
    override_values_lists = []
    if args.iter_override:
        for field, filepath in args.iter_override:
            vals = read_csv_values(filepath)
            if not vals:
                print(f"Warning: --iter-override file {filepath} is empty. Skipping this field.")
                continue
            override_fields.append(field)
            override_values_lists.append(vals)

    if override_fields:
        # Cartesian product of all value lists
        override_combinations = []
        for combo in itertools.product(*override_values_lists):
            override_combinations.append(dict(zip(override_fields, combo)))
    else:
        override_combinations = [{}]  # one combination with no overrides

    # Total number of experiments
    total_experiments = len(cfg_rows) * len(override_combinations)
    experiment_index = 0

    for cfg_row in cfg_rows:
        for override_comb in override_combinations:
            experiment_index += 1
            print(f"\n=== Experiment {experiment_index}/{total_experiments} ===")
            if cfg_row is not None:
                print(f"  Additional config: {cfg_row}")
            if override_comb:
                print(f"  Overrides: {override_comb}")

            # Build config files: base --cfg + optional iter-cfg row
            config_files = args.cfg.copy() if args.cfg else []
            if cfg_row is not None:
                config_files.append(cfg_row)   # row merges after base

            # Build overrides: base --override + iter-override values
            base_overrides = args.override.copy() if args.override else []
            iter_override_strs = [f"{field}={value}" for field, value in override_comb.items()]
            all_overrides = base_overrides + iter_override_strs

            configs = build_configs(args, parser, config_files, all_overrides)

            trainer = MatformerTrainer(configs, device=device)
            trainer.run(save_at_end=True)

            # Clean up to free resources before next experiment
            del trainer, configs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\nAll experiments completed. Bye.")


if __name__ == '__main__':
    main()