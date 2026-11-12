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
StateHolder = namedtuple('StateHolder', ['optimizer', 'lr_scheduler', 'datamodule', 'step', 'epoch'])

def make_holder(path, **kwargs):
    parts=path.split('://')
    if len(parts)==1:
        holder_type='matformer'
        holder_path=path
    elif len(parts)==2:
        holder_type=parts[0]
        holder_path=parts[1]
    else:
        raise Exception
    return registry.create("checkpoint_holder", holder_type, path=holder_path, **kwargs)

def _get_path(cfg, dotted):
    """Look up 'a.b.c' in a nested dict, with a readable error on failure."""
    node = cfg
    for part in dotted.split("."):
        try:
            node = node[part]
        except (KeyError, TypeError):
            raise KeyError(f"'{dotted}' not found in source config (failed at '{part}')") from None
    return node


def _axis_size(spec, cfg):
    """
    Resolve an `axes` entry of a weight_transform into an int. Accepts an int literal
    or a product of factors separated by '*', each factor being an int literal or a
    dotted path into the source config. e.g. "text_config.linear_key_head_dim * text_config.linear_num_key_heads"
    """
    if isinstance(spec, int):
        return spec
    size = 1
    for factor in str(spec).split("*"):
        factor = factor.strip()
        size *= int(factor) if factor.isdigit() else _get_path(cfg, factor)
    return size


class _ShardedWeights:
    """
    Lazy, dict-like (keys / in / []) view over one or more weight shard files. Only the
    current shard is resident; keys() groups same-shard entries so sequential iteration
    touches each shard once. Format-agnostic: any holder with a {source_key: filename} map can use it.
    """
    def __init__(self, folder, shard_map):
        self.folder, self.shard_map = Path(folder), shard_map
        self._cur_file, self._cur_tensors = None, {}

    def keys(self):
        return sorted(self.shard_map, key=self.shard_map.get)

    def __iter__(self):
        return iter(self.keys())

    def __contains__(self, key):
        return key in self.shard_map

    def __getitem__(self, key):
        fname = self.shard_map[key]
        if fname != self._cur_file:
            path = self.folder / fname
            if path.suffix == ".safetensors":
                self._cur_tensors = load_file(path, device="cpu")
            else:
                try:
                    self._cur_tensors = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
                except RuntimeError:
                    self._cur_tensors = torch.load(path, map_location="cpu", weights_only=False) #For older checkpoints
            self._cur_file = fname
        return self._cur_tensors.get(key)


class StableWeights:
    """
    Lazy {stable_key: tensor} view. Nothing is materialised: items() re-runs the
    streaming resolution, yielding one tensor at a time. Duck-types the part of a dict
    that MatformerModelWrapper.load_stable_state_dict uses (.items()).
    """
    def __init__(self, stream_fn):
        self._stream_fn = stream_fn

    def items(self):
        return self._stream_fn()

    def materialize(self):
        return dict(self.items())
