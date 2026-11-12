"""Functional Localizer à la "The LLM Language Network: A Neuroscientific Approach for
Identifying Causally Task-Relevant Units" (AlKhamissi, Tuckute, Bosselut & Schrimpf, 2024),
adapted to Matformer and to binary irony classification.

Protocol (same as llm-localization, with class 1/0 as positive/negative stimuli):
  - read the residual stream after every block, mean-pooled per sample;
  - score every unit (layer, dim) with the Welch t-statistic of |activation| across classes;
  - top-k units = the functional localizer mask;
  - lesion = zero those units of the residual after each block; compare against lesions of
    equally many random units (and random units drawn from the same t-value percentile band).

Three plugins (mounted by run_functional_localizer.py):
  - FuncLocalScan   (last socket): accumulates |act| moments per class -> t-ranking.
  - ResidualAblator (every layer socket): zeroes its assigned dims (.dims) of the residual.
  - FuncLocalEval   (last socket): tracks the accuracy of the classifier head.
"""

import json
import re
from dataclasses import dataclass, replace

import torch

from matformer.matformer_registry import registry
from matformer.plugins.plugin_classes import Plugin, PluginConfig
from matformer.plugins.plugin_modules.dla_stats import _as_token_matrix


def mean_pool(tdc):
    """Per-sample mean pooling -> (B, D) float32, using dla_stats' token matrix helpers."""
    matrix, segs = _as_token_matrix(tdc)
    if matrix.ndim != 2:
        return torch.zeros(0)
    return torch.stack([matrix[s].mean(0) if s.stop > s.start else torch.zeros(matrix.shape[-1])
                        for s in segs])


def _labels_of(acts):
    for tdc in acts.values():
        labels = getattr(tdc, "attributes", {}).get("labels")
        if labels is not None:
            return [int(l) for l in labels]
    return None


@registry.register_dataclass("plugin", "func_local_scan")
@dataclass
class FuncLocalScanConfig(PluginConfig):
    ranking_file: str = "results/func_local_ranking.json"
    trunk_regex: str = r"blocks\.layer_\d+$"


@registry.register("plugin", "func_local_scan", "default")
class FuncLocalScan(Plugin):
    """Accumulates |activation| count/mean/variance per unit and class; ranks by Welch t."""

    name = "func_local_scan"

    def __init__(self, config, model_config, cache):
        super().__init__(config, model_config, cache)
        self.config = config
        self.regex = re.compile(config.trunk_regex)
        self.reset()

    def reset(self):
        # address -> [[n, sum, sumsq] per class]
        self.moments = {}

    def forward(self, x, *args, **kwargs):
        acts = self.cache.storage.get("activation_cache", {})
        labels = _labels_of(acts)
        if labels is None:
            return None
        for address, tdc in acts.items():
            if not self.regex.search(address):
                continue
            pooled = mean_pool(tdc).abs()                   # |activation|, as in llm-localization
            if pooled.ndim != 2:
                continue
            m = self.moments.setdefault(address, {"sum": [0.0]*pooled.shape[1],
                                                  "sumsq": [0.0]*pooled.shape[1],
                                                  "n": [0, 0]})
            for y in (0, 1):
                sel = pooled[[i for i, l in enumerate(labels) if l == y]]
                if len(sel):
                    m["sum"] = [a + b for a, b in zip(m["sum"], sel.sum(0).tolist())]
                    m["sumsq"] = [a + b for a, b in zip(m["sumsq"], sel.pow(2).sum(0).tolist())]
                    m["n"][y] += len(sel)
        return None

    def t_values(self):
        """-> list of (address, dim, t) sorted by |t| descending (Welch, on |activations|)."""
        rows = []
        for address, m in self.moments.items():
            n0, n1 = m["n"]
            if n0 < 2 or n1 < 2:
                continue
            for d, (s0, q0) in enumerate(zip(m["sum"], m["sumsq"])):
                s1, q1 = m["sum"][d], m["sumsq"][d]
                mu0, mu1 = s0 / n0, s1 / n1
                v0 = max((q0 - n0 * mu0 * mu0) / (n0 - 1), 0.0)
                v1 = max((q1 - n1 * mu1 * mu1) / (n1 - 1), 0.0)
                denom = (v0 / n0 + v1 / n1) ** 0.5
                rows.append((address, d, (mu1 - mu0) / denom if denom else 0.0))
        rows.sort(key=lambda r: abs(r[2]), reverse=True)
        return rows

    def save_ranking(self, path=None):
        rows = self.t_values()
        with open(path or self.config.ranking_file, "w") as f:
            json.dump(rows, f)
        return rows


@registry.register_dataclass("plugin", "residual_ablator")
@dataclass
class ResidualAblatorConfig(PluginConfig):
    pass


@registry.register("plugin", "residual_ablator", "default")
class ResidualAblator(Plugin):
    """Zeroes the residual dimensions in `.dims` right after its layer (mask * (1-lesion),
    exactly like llm-localization's `hidden_states * (1 - mask[i])`)."""

    name = "residual_ablator"

    def __init__(self, config, model_config, cache):
        super().__init__(config, model_config, cache)
        self.dims = []

    def forward(self, x, *args, **kwargs):
        if not self.dims or not hasattr(x, "tensor") or x.tensor.ndim < 2:
            return None
        t = x.tensor.clone()
        t[..., self.dims] = 0.0
        return replace(x, tensor=t)


@registry.register_dataclass("plugin", "func_local_eval")
@dataclass
class FuncLocalEvalConfig(PluginConfig):
    pass


@registry.register("plugin", "func_local_eval", "default")
class FuncLocalEval(Plugin):
    """Accuracy of the classifier head (argmax of logits), labels from the cache."""

    name = "func_local_eval"

    def __init__(self, config, model_config, cache):
        super().__init__(config, model_config, cache)
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0

    def accuracy(self):
        return self.correct / self.total if self.total else float("nan")

    def forward(self, x, *args, **kwargs):
        out = x if isinstance(x, torch.Tensor) else getattr(x, "tensor", None)
        if out is None or out.ndim != 2 or out.shape[-1] != 2:
            return None
        labels = _labels_of(self.cache.storage.get("activation_cache", {}))
        if labels is None:
            return None
        pred = out.argmax(-1).cpu().tolist()
        self.correct += sum(p == l for p, l in zip(pred, labels))
        self.total += len(labels)
        return None
