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


def _pick_template(template):
    """tokenizer_config chat_template is a Jinja string, or (newer HF) a list of {name, template}.
    Return the default/first usable string, or None."""
    if isinstance(template, str):
        return template or None
    if isinstance(template, list):
        for entry in template:
            if isinstance(entry, dict) and entry.get("name") in ("default", None, ""):
                return entry.get("template")
        for entry in template:
            if isinstance(entry, dict) and entry.get("template"):
                return entry["template"]
    return None


@registry.register("checkpoint_holder", "hf", "default")
class HuggingFaceCheckpointHolder(CheckpointHolder):
    """Local HF-format folder or hub repo id. Only HF specifics live here: config.json, model_type,
    download, and the chat template that rides next to the tokenizer."""
    _DEFS = Path(__file__).parent / "adapters" / "huggingface"   # point to wherever the YAMLs actually live

    def __new__(cls, path, *args, **kwargs):
        """A hub repo that is really a Matformer checkpoint wrapped for HuggingFace (published by
        huggingface_publish.py, so config.json says model_type 'matformer') is downloaded and handed
        to the Matformer holder, which reads its matformer.yaml + safetensors natively. Returning a
        non-instance of this class from __new__ skips __init__ here, so no HF adapter is built."""
        source = cls._resolve_source(path)
        if cls._is_matformer(source):
            from matformer.loading.matformer_holder import MatformerCheckpointHolder
            return MatformerCheckpointHolder(source, *args, **kwargs)
        return super().__new__(cls)

    def __init__(self, path, map_location=None):
        super().__init__(self._resolve_source(path), map_location)
        with open(self.path / "config.json") as f:
            self.source_config = json.load(f)
        model_type = self.source_config.get("model_type")
        if not model_type:
            raise ValueError("config.json has no 'model_type' key.")
        self.definition = self._load_definition(model_type)
        self.num_layers = self._get_num_layers(self.source_config)

    @classmethod
    def _resolve_source(cls, path):
        """Local folder as-is, otherwise download (or reuse cache of) a hub repo. The only hub-specific hook."""
        if Path(path).exists():
            return Path(path)
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(str(path)))

    @staticmethod
    def _is_matformer(source):
        """True when a resolved source is a HuggingFace-wrapped Matformer checkpoint."""
        config = Path(source) / "config.json"
        if not config.is_file():
            return False
        try:
            return json.loads(config.read_text()).get("model_type") == "matformer"
        except (json.JSONDecodeError, OSError):
            return False

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

    def get_chat_template(self):
        """The template HF keeps next to the tokenizer: tokenizer_config.json 'chat_template' (a
        Jinja string, or a list of named templates where 'default' wins), else a chat_template.jinja
        sidecar. A template too unusual to read is skipped, never fatal."""
        config = self.path / "tokenizer_config.json"
        if config.is_file():
            try:
                template = json.loads(config.read_text(encoding="utf-8")).get("chat_template")
            except (json.JSONDecodeError, OSError):
                template = None
            source = _pick_template(template)
            if source:
                return source
        return self._read_jinja_sidecar()


@registry.register("checkpoint_holder", "modelscope", "default")
class ModelScopeCheckpointHolder(HuggingFaceCheckpointHolder):
    @classmethod
    def _resolve_source(cls, path):
        if Path(path).exists():
            return Path(path)
        from modelscope import snapshot_download
        return Path(snapshot_download(str(path)))


class GGUFFCheckpointHolder(CheckpointHolder):
    _DEFS = Path(__file__).parent / "adapters" / "gguf"
