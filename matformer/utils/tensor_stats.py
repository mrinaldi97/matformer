"""Tensor statistics shared by weight inspection, activation analysis and module diagnostics.

One function, :func:`compute_tensor_stats`, is the single definition of "what a tensor looks
like": distribution, sparsity, per-vector geometry and (for matrices) spectrum. Every caller --
``TensorDC.stats``, ``ParameterEditor.stats`` and the analysis tasks -- goes through it, so a
statistic added here appears everywhere at once.

The function is deliberately cheap by default: it never touches the autograd graph, it converts
to float32 on the fly, and it only runs an SVD when ``spectral`` allows it (activations and
gradients usually do not need one).

:class:`RunningStats` combines the per-batch dictionaries into a dataset-level summary without
keeping the batches around.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import torch

_DEFAULT_SPECTRAL_ELEMENTS = 4_000_000
_QUANTILE_SAMPLE = 1_000_000


def _scalar(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _quantiles(values: torch.Tensor, probabilities=(0.5, 0.9, 0.99)) -> dict:
    """Quantiles of a (possibly large) 1-D tensor, sampled when it is too big for ``torch.quantile``."""
    if values.numel() == 0:
        return {name: None for name in probabilities}
    flat = values.flatten()
    if flat.numel() > _QUANTILE_SAMPLE:
        step = math.ceil(flat.numel() / _QUANTILE_SAMPLE)
        flat = flat[::step]
    try:
        found = torch.quantile(flat.float(), torch.tensor(probabilities, device=flat.device))
        return {name: float(value) for name, value in zip(probabilities, found)}
    except (RuntimeError, ValueError):
        return {name: None for name in probabilities}


def _energy_shares(x: torch.Tensor, ks=(1, 2, 5, 10)) -> dict:
    """Mean share of each vector's squared norm carried by its largest ``k`` coordinates."""
    if x.numel() == 0:
        return {k: None for k in ks}
    total = x.pow(2).sum(-1).clamp_min(1e-30)
    out = {}
    for k in ks:
        width = min(k, x.shape[-1])
        top = x.pow(2).topk(width, dim=-1).values.sum(-1)
        out[k] = float((top / total).mean())
    return out


def compute_tensor_stats(t: torch.Tensor, *, spectral="auto", outlier_sigma: float = 6.0,
                         vector_stats: bool = True) -> dict | None:
    """Everything useful about one tensor, in a JSON/logger-friendly dictionary.

    Args:
        t: the tensor. Any shape; empty tensors return ``None``.
        spectral: ``True`` always computes an SVD, ``False`` never, ``"auto"`` only below
            ``_DEFAULT_SPECTRAL_ELEMENTS`` (the useful default for weight matrices).
        outlier_sigma: an element is an outlier when ``|x - mean| > outlier_sigma * std``.
        vector_stats: compute per-vector (last-dim) norms and energy concentration.
    """
    if t is None or not torch.is_tensor(t) or t.numel() == 0:
        return None

    source_shape = tuple(t.shape)
    t = t.detach().to(torch.float32)
    nonfinite = int((~torch.isfinite(t)).sum())
    x = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

    mean = x.mean()
    diffs = x - mean
    var = (diffs ** 2).mean()
    std = var.sqrt()
    z = diffs / std.clamp_min(1e-12)
    outlier_threshold = float(outlier_sigma) * float(std)
    outlier_fraction = (diffs.abs() > outlier_threshold).float().mean() if std > 0 else \
        torch.zeros((), device=x.device)

    stats = {
        "shape": source_shape,
        "numel": int(t.numel()),
        "nonfinite": nonfinite,
        "nonfinite_fraction": nonfinite / t.numel(),
        "mean": float(mean),
        "var": float(var),
        "std": float(std),
        "skewness": float((z ** 3).mean()),
        "kurtosis": float((z ** 4).mean() - 3.0),
        "min": float(x.min()),
        "max": float(x.max()),
        "abs_mean": float(x.abs().mean()),
        "abs_max": float(x.abs().max()),
        "rms": float(x.pow(2).mean().sqrt()),
        "l1_norm": float(x.abs().sum()),
        "l2_norm": float(x.norm()),
        "zero_fraction": float((x == 0).float().mean()),
        "sparsity_1e-6": float((x.abs() < 1e-6).float().mean()),
        "sparsity_1e-3": float((x.abs() < 1e-3).float().mean()),
        "outlier_sigma": float(outlier_sigma),
        "outlier_threshold": outlier_threshold,
        "outlier_fraction": float(outlier_fraction),
        "max_abs_z": float(z.abs().max()),
    }

    flattened = x.flatten()
    qs = _quantiles(flattened, (0.01, 0.5, 0.99))
    stats.update({"quantile_1pct": qs[0.01], "quantile_50pct": qs[0.5],
                  "quantile_99pct": qs[0.99]})

    if vector_stats and t.ndim >= 1:
        vectors = x.reshape(-1, x.shape[-1])
        norms = vectors.norm(dim=-1) if vectors.numel() else vectors
        if norms.numel():
            norm_q = _quantiles(norms, (0.5, 0.9, 0.99))
            max_abs = vectors.abs().max(dim=-1).values
            stats.update({
                "vector_norm_mean": float(norms.mean()),
                "vector_norm_std": float(norms.std(unbiased=False)),
                "vector_norm_min": float(norms.min()),
                "vector_norm_max": float(norms.max()),
                "vector_norm_p50": norm_q[0.5],
                "vector_norm_p90": norm_q[0.9],
                "vector_norm_p99": norm_q[0.99],
                "vector_max_abs_mean": float(max_abs.mean()),
                "vector_max_abs_max": float(max_abs.max()),
                "outlier_vectors": int((norms > float(norms.mean())
                                        + outlier_sigma * float(norms.std(unbiased=False))).sum()),
            })
            shares = _energy_shares(vectors, (1, 2, 5, 10))
            for k, value in shares.items():
                stats[f"top{k}_share"] = value
        else:
            for key in ("vector_norm_mean", "vector_norm_std", "vector_norm_min",
                        "vector_norm_max", "vector_norm_p50", "vector_norm_p90",
                        "vector_norm_p99", "vector_max_abs_mean", "vector_max_abs_max",
                        "outlier_vectors", "top1_share", "top2_share", "top5_share",
                        "top10_share"):
                stats[key] = None

    wants_spectrum = spectral is True or (spectral == "auto"
                                          and t.ndim >= 2
                                          and t.numel() <= _DEFAULT_SPECTRAL_ELEMENTS)
    if wants_spectrum and t.ndim >= 2:
        matrix = x.reshape(-1, x.shape[-1])
        try:
            singular = torch.linalg.svdvals(matrix)
            frobenius = singular.norm()
            top = singular[0]
            probability = singular / singular.sum().clamp_min(1e-30)
            stable_rank = (frobenius / top.clamp_min(1e-30)) ** 2
            effective_rank = torch.exp(-(probability * probability.clamp_min(1e-30).log()).sum())
            centered = matrix - matrix.mean(0, keepdim=True)
            eigenvalues = torch.linalg.eigvalsh(centered.T @ centered).clamp_min(0)
            covariance = eigenvalues / eigenvalues.sum().clamp_min(1e-30)
            effective_rank_centered = torch.exp(
                -(covariance * (covariance + 1e-30).log()).sum())
            stats.update({
                "frobenius": float(frobenius),
                "nuclear_norm": float(singular.sum()),
                "spectral_norm": float(top),
                "stable_rank": float(stable_rank),
                "effective_rank": float(effective_rank),
                "effective_rank_centered": float(effective_rank_centered),
                "condition_number": float(top / singular[-1].clamp_min(1e-30)),
                "singular_max": float(top),
                "singular_min": float(singular[-1]),
            })
        except (RuntimeError, torch.linalg.LinAlgError) as problem:
            stats["spectral_error"] = str(problem)
    else:
        for key in ("frobenius", "nuclear_norm", "spectral_norm", "stable_rank",
                    "effective_rank", "effective_rank_centered", "condition_number",
                    "singular_max", "singular_min"):
            stats[key] = None
    return stats


class RunningStats:
    """Combine per-batch statistic dictionaries into one summary.

    Means are weighted (a batch with more tokens counts more), standard deviations are combined
    from the weighted first and second moments, and minima/maxima are tracked exactly. Non-numeric
    and non-finite values are ignored so a single NaN activation cannot poison a whole run.
    """

    def __init__(self, keep_std: bool = True):
        self.keep_std = keep_std
        self.weight = 0.0
        self.samples = 0
        self._sum = OrderedDict()
        self._sum_sq = OrderedDict()
        self._min = {}
        self._max = {}

    def update(self, stats: dict | None, weight: float = 1.0) -> None:
        if not stats:
            return
        try:
            weight = float(weight)
        except (TypeError, ValueError):
            weight = 1.0
        if weight <= 0:
            return
        self.weight += weight
        self.samples += 1
        for key, value in stats.items():
            number = _scalar(value)
            if number is None or not math.isfinite(number):
                continue
            self._sum[key] = self._sum.get(key, 0.0) + number * weight
            self._sum_sq[key] = self._sum_sq.get(key, 0.0) + number * number * weight
            self._min[key] = min(self._min.get(key, number), number)
            self._max[key] = max(self._max.get(key, number), number)

    def summary(self, prefix: str = "") -> dict:
        if not self.weight:
            return {}
        out = {}
        for key, total in self._sum.items():
            mean = total / self.weight
            out[f"{prefix}{key}"] = mean
            if self.keep_std:
                variance = max(self._sum_sq[key] / self.weight - mean * mean, 0.0)
                out[f"{prefix}{key}_std"] = math.sqrt(variance)
            out[f"{prefix}{key}_min"] = self._min[key]
            out[f"{prefix}{key}_max"] = self._max[key]
        return out

    @property
    def empty(self) -> bool:
        return self.samples == 0
