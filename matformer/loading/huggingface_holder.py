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

@registry.register("checkpoint_holder", "hf", "default")  
class HuggingFaceCheckpointHolder(CheckpointHolder):
    """Local HF-format folder or hub repo id. Only HF specifics live here: config.json, model_type, download."""
    _DEFS = Path(__file__).parent / "adapters" / "huggingface"   # point to wherever the YAMLs actually live

    def __init__(self, path, map_location=None):
        print(f"Path 1 è settata a {path}")
        super().__init__(self._resolve_source(path), map_location)
        print(f"Path 2 è settata a {path}")
        with open(self.path / "config.json") as f:
            self.source_config = json.load(f)
        model_type = self.source_config.get("model_type")
        if not model_type:
            raise ValueError("config.json has no 'model_type' key.")
        self.definition = self._load_definition(model_type)
        self.num_layers = self._get_num_layers(self.source_config)

    def _resolve_source(self, path):
        """Local folder as-is, otherwise download (or reuse cache of) a hub repo. The only hub-specific hook."""
        if Path(path).exists():
            return Path(path)
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(str(path)))

    @staticmethod
    def _get_num_layers(hf):
        """Layer-count key, top level or under text_config; GPT-2 style models use n_layer,
        EXAONE/OLMo-1 style num_layers, DistilBERT n_layers."""
        for key in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
            if key in hf:
                return hf[key]
            if key in hf.get("text_config", {}):
                return hf["text_config"][key]
        raise KeyError("num_hidden_layers (or n_layer/num_layers) not found at top level or under text_config.")

    def _raw_weights(self, rename, rules, aliases):
        index = next(self.path.glob("*.index.json"), None)
        if index is not None:
            shard_map = json.loads(index.read_text())["weight_map"]
        else:
            shard = next(chain(self.path.glob("*.safetensors"), self.path.glob("*.bin")))
            needed = set(rename) | {s for r in rules for s in r["sources"]}
            needed = {c for k in needed for c in self._candidate_source_keys(k, aliases)}
            shard_map = {k: shard.name for k in needed}
        return _ShardedWeights(self.path, shard_map)

    def get_configs(self, configs=None):
        """HF config.json -> MatformerConfigs (optionally merged into an existing one)."""
        from matformer.matformer_configs import MatformerConfigs
        configs = MatformerConfigs() if configs is None else configs
        configs.convert_and_import(self.definition, self.source_config)
        configs.load_config(stdin={"tokenizer": {"type": "huggingface","pretrained_name": str(self.path)}})
        return configs


@registry.register("checkpoint_holder", "modelscope", "default")  
class ModelScopeCheckpointHolder(HuggingFaceCheckpointHolder):
    def _resolve_source(self, path):
        if Path(path).exists():
            return Path(path)
        from modelscope import snapshot_download
        return Path(snapshot_download(str(path)))


class GGUFFCheckpointHolder(CheckpointHolder):
    _DEFS = Path(__file__).parent / "adapters" / "gguf"
