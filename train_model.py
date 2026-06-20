"""
A function to train a generic model using the matformer library
A JSON config file must be provided
But each argument can be overridden from the CLI
"""
#train_model.py
import argparse
import json
import torch
from datetime import datetime
import argparse
import sys
from lightning.fabric import Fabric
from lightning.pytorch.loggers import WandbLogger
from lightning.fabric.strategies import DDPStrategy
from matformer.training.prepare_training import instantiate_training
from matformer.training.training_loop import launch_training_loop
from matformer.training.utils import load_config,detect_device,compile_model,get_model_class
from matformer.training.ckpt_manager import CheckpointDirectoryManager

def parse_args():
    """
    Specific for train_model.py, returns CLI arguments and overrides dict to be used in main() function.
    """
    parser = argparse.ArgumentParser(description='Train a Matformer model')
    parser.add_argument('--config', type=str, help="Path to single combined config file")
    parser.add_argument('--override', nargs='*', default=[], help="Override config parameters as key=value pairs")
    parser.add_argument('--gpu', type=int, default=1, help="Number of GPU(s)")
    parser.add_argument('--nodes', type=int, default=1, help="Number of Node(s)")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to checkpoint file (use reset to decide if to do a partial or full resume)")
    parser.add_argument('--reset', type=str, default=None,help="What do you want to reset from the checkpoint, use comma separated values: '['lr_scheduler', 'optimizer', 'datamodule', 'steps']'") #, choices=['lr_scheduler', 'optimizer', 'datamodule', 'steps']
    parser.add_argument('--start-from-scratch', action='store_true', help="Start training from scratch, advance version if checkpoint already exists")
    parser.add_argument('--max-steps', type=int, default=None, help="If you choose this, train for one epoch on this number of steps")
    parser.add_argument('--ckpt-version',type=int, default=None, help="If restarting a checkpoint, decide to restart from a specific version")
    parser.add_argument('--compile', action='store_true', help="Torch.compile the whole model")
    parser.add_argument('--device',type=str,default='auto',help="Device. 'auto' is default (choose the best), supported 'cuda','mps','cpu'")
    parser.add_argument('--precision', type=str, choices=['16-mixed', 'bf16-mixed', '32','16','bf16','32-true','bf16-true','16-true','64-true','transformer-engine'], 
                        default='bf16-mixed', help="precision")
    args = parser.parse_args()
    if args.reset is not None:
        for a in args.reset.split(','):
            assert a in ['lr_scheduler', 'optimizer', 'datamodule', 'steps']
    assert args.device in ['auto','cuda','cpu','mps']
    overrides = {}
    for item in args.override:
        try:
            k, v = item.split('=', 1)  # Split on first '=' only
            overrides[k] = v
        except ValueError:
            parser.error(f"Override '{item}' must be in key=value format")
    
    return args,overrides
    


def main():

    

    torch.set_float32_matmul_precision('high') # Should be a Environment Variable   
    # 1. Parse the arguments and overrides
    args, overrides = parse_args()  

        
    # 2. Load the model config
    model_config, train_config, data_config, tokenizer_config, raw_config = load_config(args.config, overrides)


    # 4. Prepare training
    fabric, training_dict = instantiate_training(
        model_config, 
        train_config, data_config, tokenizer_config, raw_config,
        device=args.device, precision=args.precision,
        num_nodes=args.nodes, device_count=args.gpu,
        checkpoint=args.checkpoint, start_from_scratch=args.start_from_scratch,
        what_to_reset=args.reset, max_steps=args.max_steps, version=args.ckpt_version, _compile=args.compile
    )
    
    # 5. Launch training
    launch_training_loop(fabric,training_dict,save_at_end=True)
    # 6. Training has ended
    print("Training complete.  Bye.")

if __name__ == '__main__':
    main()

