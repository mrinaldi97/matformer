"""
Direct Logit Attribution (DLA) as a Matformer plugin.

Mount an `activation_cache` on the non-identity sockets of the model, then mount this
plugin at the last socket that runs in the forward pass (e.g. `before_unembed`).
At the end of every batch it reads the whole activation cache, and for each cached
activation computes how much that component wrote toward the positive class:

    proj(address) = mean_t [ r_t * < write_t , g * (w[1] - w[0]) > ]

where w[1]-w[0] is the classifier direction, g the gain of the final RMSNorm and
r_t = 1 / rms(residual_t) its per-token scale. Because the residual stream is a plain
sum of component writes, these projections add up to the final logit difference
(up to the classifier bias), so no linear regression is needed for RMSNorm models.

Outputs (written by the plugin, one JSON line per batch + a final summary):
  - output_file:  per-sample projections of every cached address, with labels
  - summary_file: per-address mean projection split by label (+ / - toward irony)
"""

import json
from dataclasses import dataclass

import torch

from matformer.matformer_registry import registry
from matformer.dataclasses.tensors_dataclasses import PaddedTensor, UnpaddedTensor
from matformer.plugins.plugin_classes import Plugin, PluginConfig


@registry.register_dataclass("plugin", "dla_stats")
@dataclass
class DlaStatsConfig(PluginConfig):
    output_file: str = "dla_projections.jsonl"
    summary_file: str = "dla_summary.json"
    cache_key: str = "activation_cache"
    pre_norm_address: str = None      # cache entry holding the residual right before the final norm
    rms_norm: bool = True             # exact per-token 1/rms correction (final norm is RMSNorm)
    eps: float = 1e-6


def _segments(tdc):
    """(batch_index, token_slice) pairs over the flattened token rows of a TensorDC."""
    if isinstance(tdc, UnpaddedTensor):
        cu = tdc.cu_seqlens.tolist()
        return [(i, slice(a, b)) for i, (a, b) in enumerate(zip(cu, cu[1:]))]
    if isinstance(tdc, PaddedTensor):
        padding_mask = tdc.padding_mask.to(torch.bool)
        lengths = (~padding_mask).sum(1).tolist()
        width = padding_mask.shape[1]
        return [(i, slice(i * width, i * width + n)) for i, n in enumerate(lengths)]
    tensor = tdc.tensor
    if tensor.ndim == 3:                       # NormalTensor (B, T, D)
        width = tensor.shape[1]
        return [(i, slice(i * width, (i + 1) * width)) for i in range(tensor.shape[0])]
    return [(0, slice(0, tensor.shape[0]))]    # single flat sample


def _as_token_matrix(tdc):
    """Return (N_tokens, D) float32 CPU view + list of token_slices per sample."""
    if isinstance(tdc, torch.Tensor):
        t = tdc.detach().float().cpu()
        if t.ndim == 3:
            b, width = t.shape[:2]
            segs = [slice(i * width, (i + 1) * width) for i in range(b)]
            return t.reshape(-1, t.shape[-1]), segs
        return t.reshape(-1, t.shape[-1]), [slice(0, t.shape[0])]
    t = tdc.tensor.detach().float().cpu()
    if t.ndim == 3:
        t = t.reshape(-1, t.shape[-1])
    return t, [s for _, s in _segments(tdc)]


@registry.register("plugin", "dla_stats", "default")
class DlaStats(Plugin):
    """Reads the activation cache at the end of the forward pass, projects every cached
    activation of every sample on the label direction, and stores the results as JSONL.
    Call `set_direction()` (and optionally `set_final_norm()`) after mounting."""

    name = "dla_stats"

    def __init__(self, config, model_config, cache):
        super().__init__(config, model_config, cache)
        self.config = config
        self.direction = None       # g * (w[1] - w[0]), set by the driver script
        self.gain = None            # final-norm affine gain, if any
        self.raw_direction = None   # w[1] - w[0], uncorrected (reference numbers)
        self.num_samples = 0
        self.num_batches = 0
        self.bias_diff = 0.0        # b[1] - b[0] of the classifier
        self._totals = {}           # address -> {label: [sum, count]}
        self._file = open(config.output_file, "w")

    # ---------- configuration from the driver script ----------
    def set_direction(self, direction, bias=None):
        """`direction` is the label direction (e.g. classifier.get_direction('classifier.weight'));
        `bias` is the classifier bias, used only to report the constant offset."""
        self.raw_direction = direction.detach().float().cpu().reshape(-1)
        self.direction = self.raw_direction.clone()
        if bias is not None:
            b = bias.detach().float().cpu().reshape(-1)
            self.bias_diff = float(b[1] - b[0]) if b.numel() > 1 else 0.0

    def set_final_norm(self, owner, name="norm"):
        """Enable the exact per-token RMS correction by reading the final norm's affine
        gain through the stable-name API (`owner` is the module owning the norm)."""
        try:
            gain = owner._p(f"{name}.weight")
        except KeyError:
            gain = None                       # norm without affine parameters: g = 1
        if gain is not None:
            self.gain = gain.detach().float().cpu().reshape(-1)
            self.direction = self.direction * self.gain

    # ---------- per-batch computation ----------
    def _residual_scale(self, acts):
        """Per-token 1/rms of the residual stream entering the final norm, plus its
        per-sample token slices. This is the scale that makes the RMSNorm exactly
        decomposable: LN(x_t) = x_t * scale_t (times the gain)."""
        if not self.config.rms_norm:
            return None
        tdc = acts.get(self.config.pre_norm_address)
        if tdc is None:
            return None
        matrix, _ = _as_token_matrix(tdc)
        if matrix.ndim != 2:
            return None
        return 1.0 / matrix.pow(2).mean(-1).add(self.config.eps).sqrt()

    @staticmethod
    def _row_scale(scale, ref_tdc, tdc, n_rows):
        """Map the residual's per-token scale onto another cached activation's token rows,
        aligning sample by sample (handles padded/unpadded layout differences)."""
        _, ref_segs = _as_token_matrix(ref_tdc)
        _, segs = _as_token_matrix(tdc)
        if len(ref_segs) != len(segs):
            return None
        out = torch.ones(n_rows)
        for ref_s, s in zip(ref_segs, segs):
            n = min(ref_s.stop - ref_s.start, s.stop - s.start)
            out[s.start:s.start + n] = scale[ref_s.start:ref_s.start + n]
        return out

    def _token_scores(self, tdc, direction, scale=None, ref_tdc=None):
        """-> per-sample list of mean over tokens of <token, direction> (RMS-corrected)."""
        matrix, segs = _as_token_matrix(tdc)
        if matrix.ndim != 2 or matrix.shape[-1] != direction.shape[-1] or not segs:
            return None
        scores = matrix @ direction                      # (N_tokens,)
        if scale is not None:
            rows = self._row_scale(scale, ref_tdc, tdc, matrix.shape[0]) \
                if ref_tdc is not tdc else scale
            if rows is not None:
                scores = scores * rows
        elif self.config.rms_norm:
            rms = matrix.pow(2).mean(-1).add(self.config.eps).sqrt()
            scores = scores / rms
        return [scores[s].mean().item() if s.stop > s.start else 0.0 for s in segs]

    def _find_labels(self, x, acts):
        for holder in (x, *acts.values()):
            labels = getattr(holder, "attributes", {}).get("labels")
            if labels is not None:
                if isinstance(labels, torch.Tensor):
                    labels = labels.detach().cpu().reshape(-1).tolist()
                return [int(l) for l in labels]
        return None

    def forward(self, x, *args, **kwargs):
        if self.direction is None:
            return None
        acts = self.cache.storage.get(self.config.cache_key, {})
        labels = self._find_labels(x, acts)
        raw = self.raw_direction if self.raw_direction is not None else self.direction
        ref_tdc = acts.get(self.config.pre_norm_address) if self.config.rms_norm else None
        scale = self._residual_scale(acts)
        projections, raw_projections, logit_diff = {}, {}, None
        for address, tdc in sorted(acts.items()):
            proj = self._token_scores(tdc, self.direction, scale, ref_tdc)
            if proj is not None:
                projections[address] = proj
                if self.gain is not None:
                    raw_projections[address] = self._token_scores(tdc, raw, scale, ref_tdc)
        # x is the classifier output when this plugin sits at the very end of the model
        out = x if isinstance(x, torch.Tensor) else getattr(x, "tensor", None)
        if out is not None and out.ndim == 2 and out.shape[-1] == 2:
            logit_diff = (out[:, 1] - out[:, 0]).detach().float().cpu().tolist()
        self._file.write(json.dumps({
            "offset": self.num_samples,
            "labels": labels,
            "logit_diff": logit_diff,
            "projections": projections,
            "projections_raw_direction": raw_projections or None,
        }) + "\n")
        self._file.flush()
        if labels is not None:
            for address, proj in projections.items():
                acc = self._totals.setdefault(address, {})
                for l, v in zip(labels, proj):
                    s, n = acc.get(l, (0.0, 0))
                    acc[l] = (s + v, n + 1)
            self.num_samples += len(labels)
        self.num_batches += 1
        return None

    # ---------- end of run ----------
    def summary(self):
        """Per-address mean projection split by label. `irony_minus_nonirony` is the
        average extra push of each component toward the positive (ironic) class."""
        out = {"n_samples": self.num_samples,
               "classifier_bias_diff": self.bias_diff,
               "rms_correction": self.config.rms_norm,
               "addresses": {}}
        for address, acc in sorted(self._totals.items()):
            entry = {f"mean_label{l}": s / n for l, (s, n) in sorted(acc.items()) if n}
            if len(entry) == 2:
                entry["irony_minus_nonirony"] = entry["mean_label1"] - entry["mean_label0"]
            out["addresses"][address] = entry
        return out

    def save_summary(self):
        with open(self.config.summary_file, "w") as f:
            json.dump(self.summary(), f, indent=2)
        if not self._file.closed:
            self._file.close()
