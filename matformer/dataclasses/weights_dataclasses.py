from matformer.matformer_registry import registry
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Union
from matformer.dataclasses.matformer_config import MatformerMultiConfig

"""
This file is not well-formed according to the Matformer's principles because
 it doesn't contain just the dataclasses, but also the implementations 
for loading torch/safetensors/hf weights.
"""

@dataclass
class WeightConfig(MatformerMultiConfig):
    """Base for a single additional-weights source. Not registered itself —
    only its typed variants (torch/safetensors/hf) are."""
    path: str = None
    weight_mapping: dict = field(default_factory=dict)
    definition: Optional[dict] = None                         # adapter definition override (weight_mapping/weight_transforms) handed to the checkpoint holder
    wanted: Optional[Union[List[str], str]] = None            # stable keys (or fnmatch patterns) to inherit; mutually exclusive with discard
    discard: Optional[Union[List[str], str]] = None           # stable keys (or fnmatch patterns) not to inherit; mutually exclusive with wanted


@registry.register_dataclass("weights", "torch", default=True)
@dataclass
class TorchWeightConfig(WeightConfig):
    pass


@registry.register_dataclass("weights", "safetensors")
@dataclass
class SafetensorWeightConfig(WeightConfig):
    pass


@registry.register_dataclass("weights", "hf")
@dataclass
class HFWeightConfig(WeightConfig):
    pass

class WeightSource:
    """Base for additional-weight loaders. Constructed via instantiate(weight_cfg),
    so __init__ receives the WeightConfig itself as `config`.

    load() returns (state_dict, weight_map): weight_map is {source_key: stable_name}
    or None, ready to hand straight to MatformerModelWrapper._load_stable_state_dict.
    """
    def __init__(self, config):
        self.config = config
        self.path = config.path
        self.weight_mapping = config.weight_mapping or None

    def load(self):
        raise NotImplementedError


@registry.register("weights", "torch", "default")
class TorchWeights(WeightSource):
    def load(self):
        import torch
        state_dict = torch.load(Path(self.path), map_location="cpu", weights_only=False)
        return state_dict.get("state_dict", state_dict), self.weight_mapping


@registry.register("weights", "safetensors","default")
class SafetensorWeights(WeightSource):
    def load(self):
        from safetensors.torch import load_file
        return load_file(Path(self.path), device="cpu"), self.weight_mapping

@registry.register("weights", "hf", "default")
class HFWeights(WeightSource):
    """WeightSource for an HF repo used as an additional (non-primary) weight
    source, e.g. loading only lm_head from a donor model."""
    def __init__(self, path, weight_mapping=None, transform_rules=None):
        self.path = path
        self.weight_mapping = weight_mapping or None
        self.transform_rules = transform_rules or None

    def load(self):
        from matformer.huggingface.hf_model_loader import get_hf_repo, _hf_state_dict_and_mapping
        hf_config_path, _ = get_hf_repo(self.path)
        return _hf_state_dict_and_mapping(hf_config_path, self.weight_mapping, self.transform_rules)