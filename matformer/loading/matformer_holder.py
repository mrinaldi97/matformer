import json
import re
from collections import namedtuple
from itertools import chain
from pathlib import Path

import einx
import torch
import yaml
from safetensors.torch import load_file

from matformer.matformer_registry import registry
from matformer.loading.utils import  _get_path, _axis_size, _ShardedWeights, StableWeights, StateHolder
from matformer.loading.checkpoint_holder import CheckpointHolder

@registry.register("checkpoint_holder", "matformer", "default")  
class MatformerCheckpointHolder(CheckpointHolder):
    def __init__(self, path, map_location=None, what_to_reset=None):
        """
        1. '.ckpt' file: weights, configs and states from a Matformer checkpoint
        2. raw weights (.pt / .safetensors): weights only, no configs
        3. folder: all yaml as configs + exactly one weight file
        """
        super().__init__(path, map_location)
        p = self.path
        self.checkpoint, self.configs = None, None
        self.what_to_reset=what_to_reset
        if p.is_file() and p.suffix == '.ckpt':
            self.checkpoint = torch.load(p, map_location=map_location, weights_only=False)
            self.state_dict = self.checkpoint['state_dict']
            self.configs = self._configs_from_checkpoint(self.checkpoint)
        elif p.is_file():
            self._load_state(p, map_location)
        elif p.is_dir():
            from matformer.matformer_configs import MatformerConfigs
            self.configs = MatformerConfigs.create_configs(p.rglob('*.yaml'))
            weight_files = list(chain(p.rglob('*.pt'), p.rglob('*.safetensors')))
            if not weight_files:
                raise ValueError(f"No weight file (.pt, .safetensors) found in {p}")
            if len(weight_files) > 1:
                raise ValueError(f"Multiple weight files found in {p}. Move extra files to subfolders.")
            self._load_state(weight_files[0], map_location)

    def _load_state(self, path, map_location=None):
        if path.suffix == '.safetensors':
            self.state_dict = load_file(path, device=map_location or "cpu")
        else:
            self.state_dict = torch.load(path, map_location=map_location, weights_only=True, mmap=True)

    def get_configs(self):
        return self.configs

    def get_weights(self, definition=None, wanted=None, discard=None):
        """Native weights are already stable-keyed: returned as-is unless a user definition
        is given. `wanted`/`discard` (mutually exclusive) select the stable keys to return."""
        if definition is not None:
            return super().get_weights(definition, wanted=wanted, discard=discard)
        keep = self._keep_filter(wanted, discard)
        if keep is None:
            return self.state_dict
        return {k: v for k, v in self.state_dict.items() if keep(k)}

    def _raw_weights(self, rename, rules):
        return self.state_dict

    def get_states(self):
        if self.checkpoint is None:
            return None
        c = self.checkpoint
        if isinstance(self.what_to_reset, str):
            self.what_to_reset = self.what_to_reset.split(',')
        self.what_to_reset = {k.strip() for k in (what_to_reset or [])}
        pick = lambda flag, key, default=None: default if flag in self.what_to_reset else c.get(key, default)
        return StateHolder(optimizer=pick('optimizer', 'optimizer'),
                        lr_scheduler=pick('lr_scheduler', 'scheduler'),
                        datamodule=pick('datamodule', 'datamodule'),
                        step=pick('steps', 'step', 0), epoch=pick('steps', 'epoch', 0))

    def _configs_from_checkpoint(self, ckpt):
        from matformer.matformer_configs import MatformerConfigs
        if 'configs' in ckpt:
            return MatformerConfigs.create_configs(stdin=ckpt['configs'])
        hp = ckpt.get('hyper_parameters')
        if hp is not None:
            print("WARNING: You are restoring from a deprecated Lightning checkpoint")
            legacy = hp if isinstance(hp, dict) else {'config': hp}
            return MatformerConfigs.from_sections(self._configs_from_legacy_checkpoint(legacy))
        print("WARNING: The checkpoint doesn't contain a Config")
        return None

    @staticmethod
    def _configs_from_legacy_checkpoint(checkpoint: dict) -> dict:
        """
        TO BE REMOVED once old checkpoints fall out of use. Old checkpoints store a single
        'config' key holding an already-instantiated (old-style) ModelConfig that nested
        tokenizer_config/masking_config directly, instead of those being their own top-level
        sections as they are today. Splits that object into today's shape:
        {'model': ModelConfig, 'tokenizer': TokenizerConfig, 'masking': MaskingConfig}.
        Returned as a plain dict of instances; the caller wraps it via MatformerConfigs.from_sections().
        """        
        from matformer.dataclasses.model_dataclasses import ModelConfig,MaskingConfig,TokenizerConfig
        _d = lambda x: asdict(x) if hasattr(x, '__dataclass_fields__') else dict(x or {})
        d = dict(checkpoint['config'].__dict__)
        tok, mask = _d(d.pop('tokenizer_config', None)), _d(d.pop('masking_config', None))
 
        for k in ('vocab_size', 'pad_token_id', 'bos_token_id', 'eos_token_id', 'mask_token_id'):
            if k in d: tok[k] = d.pop(k)
        d['vocab_size'] = tok.get('vocab_size')          # ModelConfig keeps it too
        if 'tokenizer_name' in d: tok['pretrained_name'] = d.pop('tokenizer_name')
        for k in ('masked_substitution_rate', 'cloze_probability', 'random_probability', 'same_probability'):
            if k in d: mask[k] = d.pop(k)
 
        # Legacy layers kept attention scalars flat; today they live in sequence_mixer_config.
        # Also pre-fill q/k/v dims and head counts (get_layer_config does this lazily, but
        # head construction reads default_layer directly before that runs).
        hidden, heads = d.get('hidden_size'), d.pop('num_attention_heads', None)
        layers = d.get('custom_layers') or {}
        for lay in [d.setdefault('default_layer', {}), *layers.values()]:
            lay['sequence_mixer'] = lay.get('sequence_mixer', 'attention')
            smc = lay.setdefault('sequence_mixer_config', {})
            for f in ('attn_impl', 'sliding_window_size', 'positional_encoding', 'rope_theta'):
                if f in lay: smc.setdefault(f, lay.pop(f))
            if heads is not None:
                smc.setdefault('num_attention_heads', heads)
                smc.setdefault('num_key_value_heads', heads)
            if hidden is not None:
                for dim in ('q_dim', 'k_dim', 'v_dim'): smc.setdefault(dim, hidden)
 
        d['sliding_type'] = d.get('sliding_type') or 'disabled'
        if d.get('loss_type') in (None, 'normal'): d['loss_type'] = 'cross_entropy_loss'
 
        out = {'model':     ModelConfig.from_dict(d, strict=False),
            'tokenizer': TokenizerConfig.from_dict(tok, strict=False),
            'masking':   MaskingConfig.from_dict(mask, strict=False)}
        import os
        from matformer.matformer_tokenizers import MatformerTokenizer
        if (n := os.environ.get("LEGACY_TOKENIZER")):
            out['tokenizer'] = MatformerTokenizer.from_huggingface(n).config
        return out    
