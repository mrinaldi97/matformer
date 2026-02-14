import json
import itertools
from pathlib import Path
from typing import Any, Dict, List
from copy import deepcopy
from datetime import datetime
import sys
from train_classifier_head import run_training

def expand_grid_config(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand config with list values into grid of individual configs."""
    grid_values = {}
    
    # Check training.lr
    if 'training' in config and 'lr' in config['training']:
        lr = config['training']['lr']
        if isinstance(lr, list):
            grid_values['training.lr'] = lr
    
    # Check freeze_base_model
    if 'freeze_base_model' in config:
        freeze = config['freeze_base_model']
        if isinstance(freeze, list):
            grid_values['freeze_base_model'] = freeze
    
    # If no grid params, return single config
    if not grid_values:
        return [config]
    
    # Generate all combinations
    param_names = list(grid_values.keys())
    param_values = [grid_values[name] for name in param_names]
    combinations = list(itertools.product(*param_values))
    
    # Create config for each combination
    configs = []
    for combo in combinations:
        new_config = deepcopy(config)
        
        for param_name, value in zip(param_names, combo):
            if param_name == 'training.lr':
                new_config['training']['lr'] = value
            else:
                new_config[param_name] = value
        
        configs.append(new_config)
    
    return configs
  
def create_run_identifier(config: Dict[str, Any], run_idx: int) -> str:
    """Create unique identifier for this run."""
    parts = [f"run{run_idx}"]
    
    if 'training' in config and 'lr' in config['training']:
        lr = config['training']['lr']
        parts.append(f"lr{lr:.0e}".replace('-', ''))
    
    if 'freeze_base_model' in config:
        freeze = config['freeze_base_model']
        parts.append(f"freeze{freeze}")
    
    return "_".join(parts)
  
def main():
    if len(sys.argv) < 2:
      config_path = "configs/classification_head/config.json"
    else:
      config_path = sys.argv[1]
    
    if not Path(config_path).exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    with open(config_path, 'r') as f:
        base_config = json.load(f)
    
    configs = expand_grid_config(base_config)
    
    print(f"\n{'='*60}")
    print(f"GRID SEARCH: {len(configs)} configurations")
    print(f"{'='*60}\n")
    
    # Print grid summary
    for i, cfg in enumerate(configs, 1):
        print(f"Config {i}:")
        if 'training' in cfg and 'lr' in cfg['training']:
            print(f"  lr: {cfg['training']['lr']}")
        if 'freeze_base_model' in cfg:
            print(f"  freeze_base_model: {cfg['freeze_base_model']}")
        print()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = Path("./temp_grid_configs")
    temp_dir.mkdir(exist_ok=True)
    
    base_save_dir = base_config.get('save_dir', './checkpoints_classification')
    results_dir = Path(base_save_dir) / timestamp
    
    results = {
        'timestamp': timestamp,
        'total': len(configs),
        'successful': [],
        'failed': []
    }
    
    for i, cfg in enumerate(configs, 1):
        run_id = create_run_identifier(cfg, i)
        
        cfg = deepcopy(cfg)  # Extra safety
        cfg['save_dir'] = str(results_dir / run_id)
        cfg['wandb_run_name'] = run_id
        
        temp_config_path = temp_dir / f"{run_id}.json"
        with open(temp_config_path, 'w') as f:
            json.dump(cfg, indent=2, fp=f)
        
        print(f"\n{'='*60}")
        print(f"Starting run {i}/{len(configs)}: {run_id}")
        print(f"{'='*60}\n")
        
        try:
            run_training(str(temp_config_path))
            
            results['successful'].append(run_id)
            print(f"\n{'='*60}")
            print(f"Completed run {i}/{len(configs)}: {run_id}")
            print(f"{'='*60}\n")
            
        except Exception as e:
            results['failed'].append({'run_id': run_id, 'error': str(e)})
            print(f"\n{'='*60}")
            print(f"Failed run {i}/{len(configs)}: {run_id}")
            print(f"Error: {e}")
            print(f"{'='*60}\n")
            
            continue
          
    results_file = results_dir / "grid_search_results.json"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
      json.dump(results, indent=2, fp=f)
    
    print(f"\n{'='*60}")
    print(f"GRID SEARCH COMPLETE")
    print(f"{'='*60}")
    print(f"Total: {len(configs)}")
    print(f"Successful: {len(results['successful'])}")
    print(f"Failed: {len(results['failed'])}")
    if results['failed']:
        print("\nFailed runs:")
        for fail in results['failed']:
            print(f"  - {fail['run_id']}: {fail['error']}")
    print(f"\nResults in: {base_save_dir}/{timestamp}/")
    print(f"{'='*60}\n")
    
    if results['failed']:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()