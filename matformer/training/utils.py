#matformer/utils.py
"""
Generic utils to load config, detect best device, compile a model...
"""
import json
import ast
from matformer.model_config import ModelConfig
import torch
from importlib import import_module
from datetime import datetime

def apply_overrides(cfg, overrides=None):
    if overrides is None:
        return cfg
    try:
        assert isinstance(overrides,dict)
    except AssertionError:
        print("Misuse of apply_overrides function: overrides must be a dict")
        raise
    for key, val in overrides.items():
        keys = key.split('.')
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        try:
            d[keys[-1]] = json.loads(val)
        except (TypeError, ValueError):
            try:
                d[keys[-1]] = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                d[keys[-1]] = val
    return cfg


def load_config(config_path=None, overrides=None, config_string=None):
    """
    """
    if config_path is not None:
        try:
            assert config_string is None
        except AssertionError:
            print("Misuse of load_config function: both config_path and config_string were specified")
            raise
        with open(config_path, 'r') as f:
            config_string=f.read()
    json_config=json.loads(config_string)
        
    cfg = json_config
    cfg.setdefault('save_dir', './checkpoints')
    

    if overrides is not None:
        cfg = apply_overrides(cfg, overrides)

    model_class = cfg['model_class'] 
    if cfg['model_config'].get('training_objective') is None:
        print("Training objective not specified.")
        cfg['model_config']['training_objective'] = "autoregressive" if model_class == "Autoregressive_Model" else "masked"
        print(f"Inferred: {cfg['model_config']['training_objective']} from the model class. This behaviour will be deprecated. ")
    cfg['model_config']['tokenizer_type']=cfg['tokenizer']['type']
    cfg['model_config']['tokenizer_name']=cfg['tokenizer']['pretrained_name']
    if 'wanted_from_strategy' not in cfg['data'].keys():
         cfg['data']['wanted_from_strategy']='chunked_tokens'
    model_config_dict = cfg['model_config']
    train_config_dict = cfg['training']
    data_config_dict = cfg['data']
    tok_config_dict = cfg['tokenizer']
    model_config = ModelConfig(**model_config_dict) # Returns a ModelConfig object
    return model_config, train_config_dict, data_config_dict, tok_config_dict, cfg

def detect_device():
    if torch.cuda.is_available():
        accelerator = 'gpu'
        device_string = 'cuda'
    elif torch.backends.mps.is_available():
        accelerator = device_string = 'mps'
    else:
        accelerator = device_string = 'cpu'
    return accelerator,device_string
    
def compile_model(model):
    return torch.compile(model)

def get_model_class(model_class: str):
    """
    From a string return a Matformer's ModelClass. String and class name must match.
    """
    try:
        module = import_module("matformer.transformer_blocks")
    except:
        print("FATAL: {model_class} not present in Matformer's transformer_blocks definitions.")
        raise
    return getattr(module, model_class)

