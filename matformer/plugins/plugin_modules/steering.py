# matformer/plugins/plugin_modules/steering.py
"""Activation steering as a plugin: add `alpha * direction` to whatever passes through a socket.

Mount it on a residual-stream socket (`*layer_output`, `*post_attn`, ...); the direction is a
vector in the model's hidden size. A negative alpha ablates the same direction, a positive one
pushes the activation towards it. The direction can be given inline or read from a JSON file, so
a UI can send a short config while a study can reuse a saved vector."""
import json
from dataclasses import dataclass
from typing import Optional

import torch

from matformer.matformer_registry import registry
from matformer.plugins.plugin_classes import Plugin, PluginConfig


@registry.register_dataclass("plugin", "steering")
@dataclass
class SteeringConfig(PluginConfig):
    direction: Optional[list] = None       # the vector itself
    direction_file: Optional[str] = None   # JSON list of numbers
    vector_store: Optional[str] = None     # or a vector in store (matformer.vectors)
    vector_backend: str = "lmdb"
    vector_databank: str = "vectors"
    vector_name: Optional[str] = None
    alpha: float = 1.0                     # added as alpha * direction
    positions: str = "all"                 # all | last: steer every token or only the last one


@registry.register("plugin", "steering", "default", requires=["torch"])
class Steering(Plugin):
    name = "steering"

    def __init__(self, config, model_config, cache):
        super().__init__(config, model_config, cache)
        self.alpha = float(getattr(config, "alpha", 1.0))
        self.positions = getattr(config, "positions", "all") or "all"
        raw = getattr(config, "direction", None)
        path = getattr(config, "direction_file", None)
        if raw is None and path:
            with open(path) as handle:
                raw = json.load(handle)
        vector_name = getattr(config, "vector_name", None)
        if raw is None and vector_name:
            from matformer import vectors
            store = vectors.store({"filename": getattr(config, "vector_store", None),
                                   "backend": getattr(config, "vector_backend", "lmdb"),
                                   "databank": getattr(config, "vector_databank", "vectors"),
                                   "readonly": True})
            record = vectors.find(store, vector_name)
            if record is None:
                raise ValueError(f"steering vector '{vector_name}' is not in the store")
            raw = vectors.vector_of(store, record).tolist()
        self.direction = None if raw is None else torch.tensor(raw, dtype=torch.float32)

    def set_direction(self, direction, alpha=None):
        self.direction = None if direction is None else torch.tensor(direction, dtype=torch.float32)
        if alpha is not None:
            self.alpha = float(alpha)
        return self.direction

    def forward(self, x, *args, **kwargs):
        if self.direction is None or not self.alpha:
            return None
        tensor = x if isinstance(x, torch.Tensor) else getattr(x, "tensor", None)
        if tensor is None or tensor.shape[-1] != self.direction.numel():
            return None
        direction = self.direction.to(device=tensor.device, dtype=tensor.dtype)
        delta = direction * self.alpha
        changed = tensor.clone()
        if self.positions == "last" and changed.ndim >= 2:
            changed[..., -1, :] = changed[..., -1, :] + delta
        else:
            changed = changed + delta
        if isinstance(x, torch.Tensor):
            return changed
        from dataclasses import replace
        return replace(x, tensor=changed)
