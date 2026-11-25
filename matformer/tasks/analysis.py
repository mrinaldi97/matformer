"""Analysis tasks: what the weights are, and what the activations do.

Two tasks, both ordinary ``MatformerTask`` implementations, so their output already flows to the
run database, to the GUI and to RPC with no special casing:

``static_model_analysis``
    Reads every parameter (and its current gradient, when one exists) and logs the full tensor
    statistics: distribution, sparsity, per-vector geometry, spectrum, layer and family totals.
    It needs no data; ``requires_data=False`` makes it usable from a validation step that has a
    model but no test set.

``activation_analysis``
    Runs the model over a dataset and records, for every module that produces a ``TensorDC``,
    the same statistics of its output; optionally differentiates a learning-signal loss to get
    gradient statistics, and asks the wrapper for the architecture-specific logs that mixers
    implement through ``_log``. Per-token geometry is part of every activation summary (vector
    norms, quantiles, outliers), so an unusual token is visible without storing every token.

Everything is one implementation of the statistics: ``compute_tensor_stats``. These tasks decide
*where* to look, never *how* a tensor is described.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import ClassVar, Optional

import torch

from matformer.dataclasses.tensors_dataclasses import TensorDC
from matformer.matformer_registry import registry
from matformer.matformer_wrapper import resolve_wrapper
from matformer.tasks.task import MatformerTask, TaskConfig
from matformer.utils.tensor_stats import RunningStats, compute_tensor_stats

_LAYER = re.compile(r"(?:layers|blocks\.layers)\.(\d+)")


def parameter_metadata(name: str) -> dict:
    """Layer, family and kind of one stable parameter name; shared by every analysis view."""
    match = _LAYER.search(name)
    layer = int(match.group(1)) if match else -1
    lowered = name.lower()
    if "attn" in lowered:
        family = "attention"
    elif "mlp" in lowered or "ffn" in lowered:
        family = "mlp"
    elif "embed" in lowered:
        family = "embedding"
    elif "head" in lowered or "classifier" in lowered or "lm_head" in lowered:
        family = "head"
    elif "norm" in lowered:
        family = "norm"
    else:
        family = "other"
    leaf = name.rsplit(".", 1)[-1]
    return {"layer": layer, "family": family, "is_norm": "norm" in lowered,
            "is_bias": leaf not in ("weight", "scale", "gamma")}


def _named_parameters(model) -> list:
    """``[(stable name, parameter)]``, from the wrapper's stable mapping when available."""
    wrapper = resolve_wrapper(model)
    mapping = getattr(wrapper, "_mappings", None)
    if mapping:
        found, seen = [], set()
        for stable, _actual in mapping.items():
            parameter = wrapper._get_weight_pointer(stable)
            if parameter is not None and id(parameter) not in seen:
                seen.add(id(parameter))
                found.append((stable, parameter))
        if found:
            return found
    return list(wrapper.named_parameters())


def _numeric(items, prefix=""):
    """Flatten a nested diagnostic dictionary into plottable numbers."""
    out = {}
    for key, value in (items or {}).items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_numeric(value, f"{name}_"))
        elif isinstance(value, bool):
            out[name] = float(value)
        elif isinstance(value, (int, float)):
            out[name] = float(value)
    return out


def _texture(value):
    """A short, logger-friendly description of a shape tuple."""
    return "x".join(str(one) for one in value) if isinstance(value, (tuple, list)) else value


def _batch_tokens(batch) -> int:
    """Real token count of a batch, whichever TensorDC layout it uses."""
    if isinstance(batch, TensorDC):
        mask = getattr(batch, "padding_mask", None)
        if mask is not None:
            return int((~mask).sum())
        if getattr(batch, "isUnpadded", False):
            return int(batch.tensor.shape[0])
        width = max(int(batch.tensor.shape[-1]), 1)
        return int(batch.tensor.numel() // width)
    if hasattr(batch, "detach"):
        return sum(int(one.numel()) for one in batch.detach())
    return 0


@registry.register_dataclass("task", "static_model_analysis")
@dataclass
class StaticModelAnalysisConfig(TaskConfig):
    _ref_fields: ClassVar[dict] = {}       # weights never need a dataset
    per_parameter: bool = True            # one summary entry per parameter; false = family totals only
    gradients: bool = True                # include the gradients currently attached to the weights
    spectral: str = "auto"                # "auto", "always" or "never"
    outlier_sigma: float = 6.0
    include: list = field(default_factory=list)   # fnmatch patterns on stable names
    exclude: list = field(default_factory=list)
    families: list = field(default_factory=list)  # restrict to these families, empty = all


@registry.register("task", "static_model_analysis", "default")
class StaticModelAnalysis(MatformerTask):
    """Everything that can be known about a model without running it."""

    requires_data = False

    def begin_task(self):
        super().begin_task()
        self._families = defaultdict(RunningStats)
        self._layers = defaultdict(RunningStats)
        self._all = RunningStats()
        self._gradients = defaultdict(RunningStats)
        self._gradient_layers = defaultdict(RunningStats)
        self._gradients_all = RunningStats()

    def _spectral(self):
        return {"auto": "auto", "always": True, "never": False}.get(
            str(self.config.spectral).lower(), "auto")

    def _wanted(self, name: str) -> bool:
        if self.config.include and not any(fnmatch(name, pattern) for pattern in self.config.include):
            return False
        if self.config.exclude and any(fnmatch(name, pattern) for pattern in self.config.exclude):
            return False
        family = parameter_metadata(name)["family"]
        return not self.config.families or family in self.config.families

    def compute(self) -> dict:
        out, counts = {}, defaultdict(lambda: {"parameters": 0, "tensors": 0})
        layer_counts = defaultdict(lambda: {"parameters": 0, "tensors": 0})
        total_elements = total_bytes = trainable = 0
        for name, parameter in _named_parameters(self.model):
            if "alibi" in name or "extra_state" in name:
                continue
            if not self._wanted(name):
                continue
            metadata = parameter_metadata(name)
            stats = compute_tensor_stats(parameter.detach(), spectral=self._spectral(),
                                         outlier_sigma=self.config.outlier_sigma)
            if stats is None:
                continue
            details = dict(stats)
            details.update(metadata)
            details["shape"] = _texture(details.get("shape"))
            details["requires_grad"] = bool(parameter.requires_grad)
            details["dtype"] = str(parameter.dtype)
            details["numel"] = int(parameter.numel())
            total_elements += parameter.numel()
            total_bytes += parameter.numel() * parameter.element_size()
            trainable += parameter.numel() if parameter.requires_grad else 0
            counts[metadata["family"]]["parameters"] += parameter.numel()
            counts[metadata["family"]]["tensors"] += 1
            layer_counts[metadata["layer"]]["parameters"] += parameter.numel()
            layer_counts[metadata["layer"]]["tensors"] += 1
            self._all.update(stats, weight=max(parameter.numel(), 1))
            self._families[metadata["family"]].update(stats, weight=max(parameter.numel(), 1))
            self._layers[metadata["layer"]].update(stats, weight=max(parameter.numel(), 1))
            if self.config.per_parameter:
                prefix = f"weights/{name.replace('.', '/')}/"
                for key, value in details.items():
                    if isinstance(value, (int, float, bool)):
                        out[f"{prefix}{key}"] = value
            gradient = parameter.grad
            if self.config.gradients and gradient is not None:
                gradient_stats = compute_tensor_stats(gradient.detach(), spectral=False,
                                                      vector_stats=False,
                                                      outlier_sigma=self.config.outlier_sigma)
                self._gradients[name].update(gradient_stats, weight=max(parameter.numel(), 1))
                self._gradient_layers[metadata["layer"]].update(
                    gradient_stats, weight=max(parameter.numel(), 1))
                self._gradients_all.update(gradient_stats, weight=max(parameter.numel(), 1))
                if self.config.per_parameter:
                    prefix = f"gradients/{name.replace('.', '/')}/"
                    for key, value in (gradient_stats or {}).items():
                        if isinstance(value, (int, float, bool)):
                            out[f"{prefix}{key}"] = value

        out["model/parameters"] = int(total_elements)
        out["model/trainable_parameters"] = int(trainable)
        out["model/frozen_parameters"] = int(total_elements - trainable)
        out["model/tensors"] = sum(one["tensors"] for one in counts.values())
        out["model/bytes"] = int(total_bytes)
        for family, one in counts.items():
            out[f"model/{family}/parameters"] = one["parameters"]
            out[f"model/{family}/tensors"] = one["tensors"]
        for layer, one in layer_counts.items():
            name = "none" if layer < 0 else str(layer)
            out[f"model/layer/{name}/parameters"] = one["parameters"]
            out[f"model/layer/{name}/tensors"] = one["tensors"]
        out.update(self._all.summary("weights/all/"))
        for family, accumulator in self._families.items():
            out.update(accumulator.summary(f"weights/{family}/"))
        for layer, accumulator in self._layers.items():
            name = "none" if layer < 0 else str(layer)
            out.update(accumulator.summary(f"weights/layer/{name}/"))
        out.update(self._gradients_all.summary("gradients/all/"))
        for name, accumulator in self._gradients.items():
            out.update(accumulator.summary(f"gradients/{name.replace('.', '/')}/"))
        for layer, accumulator in self._gradient_layers.items():
            name = "none" if layer < 0 else str(layer)
            out.update(accumulator.summary(f"gradients/layer/{name}/"))
        return out


@registry.register_dataclass("task", "activation_analysis")
@dataclass
class ActivationAnalysisConfig(TaskConfig):
    learning_signal: Optional[str] = None   # None mirrors the model's own training selector
    gradients: bool = True                  # differentiate the loss to describe gradient flow
    per_parameter_gradients: bool = True
    spectral: str = "never"                 # activations rarely want an SVD
    outlier_sigma: float = 6.0
    include: list = field(default_factory=list)   # fnmatch patterns on module paths
    exclude: list = field(default_factory=list)
    module_logs: bool = True                # architecture-specific `_log` of mixers and modules
    evaluate_mode: bool = True              # eval() for deterministic activations
    profile: bool = False                   # time each architectural component
    profile_memory: bool = True             # CUDA allocator deltas while profiling
    profile_per_module: bool = False        # also report each timed module separately
    record_outliers: bool = False           # store per-token outliers in task_records
    outlier_limit: int = 100                # cap per module and batch


@registry.register("task", "activation_analysis", "default")
class ActivationAnalysis(MatformerTask):
    """Per-module activation statistics, gradient statistics and module-specific diagnostics."""

    def begin_task(self):
        super().begin_task()
        self._wrapper = resolve_wrapper(self.model)
        self._activations = defaultdict(RunningStats)
        self._gradients = defaultdict(RunningStats)
        self._gradients_all = RunningStats()
        self._module_logs = defaultdict(RunningStats)
        self._loss = RunningStats(keep_std=False)
        self._tokens = 0
        self._batches = 0
        self._signal = None
        self._profiler = None
        self._pending_outliers = []
        self._outlier_summary = []
        if self.config.learning_signal:
            self._signal = registry.create("learning_signal", self.config.learning_signal,
                                           wrapper=self.model)
        if self.config.module_logs and hasattr(self._wrapper, "enable_module_diagnostics"):
            self._wrapper.refresh_diagnostics()
            self._wrapper.enable_module_diagnostics(True)
        if self.config.profile:
            factory = getattr(self._wrapper, "component_profiler", None)
            if factory is not None:
                self._profiler = factory(memory=self.config.profile_memory,
                                         per_module=self.config.profile_per_module)
                self._profiler.start()

    def end_task(self):
        if self._profiler is not None:
            self._profiler.stop()
        if self.config.module_logs and hasattr(self._wrapper, "enable_module_diagnostics"):
            self._wrapper.enable_module_diagnostics(False)
        super().end_task()

    def _spectral(self):
        return {"auto": "auto", "always": True, "never": False}.get(
            str(self.config.spectral).lower(), False)

    @staticmethod
    def _token_weight(output) -> float:
        tensor = getattr(output, "tensor", output)
        if not torch.is_tensor(tensor) or tensor.numel() == 0:
            return 0.0
        mask = getattr(output, "padding_mask", None)
        if mask is not None:
            return float((~mask).sum())
        width = max(int(tensor.shape[-1]), 1)
        return float(tensor.numel() // width)

    def _capture(self, name, output):
        if isinstance(output, dict):
            for key, value in output.items():
                self._capture(f"{name}.{key}" if name else str(key), value)
            return
        tensor = getattr(output, "tensor", output)
        if not torch.is_tensor(tensor) or tensor.numel() == 0:
            return
        stats = output.stats(spectral=self._spectral(), outlier_sigma=self.config.outlier_sigma) \
            if isinstance(output, TensorDC) else compute_tensor_stats(
                tensor, spectral=self._spectral(), outlier_sigma=self.config.outlier_sigma)
        self._activations[name].update(stats, weight=max(self._token_weight(output), 1.0))
        if self.config.record_outliers:
            self._remember_outliers(name, output)

    def _remember_outliers(self, name, output):
        """Keep the tokens whose vector norm is unusually large, capped per module and batch."""
        tensor = getattr(output, "tensor", output)
        if not torch.is_tensor(tensor) or tensor.numel() == 0:
            return
        mask = getattr(output, "padding_mask", None)
        flat = tensor.detach().float()
        if mask is not None and flat.dim() >= 2:
            flat, indices = flat[~mask], torch.nonzero(~mask).flatten()
        else:
            indices = torch.arange(flat.reshape(-1, flat.shape[-1]).shape[0])
        vectors = flat.reshape(-1, flat.shape[-1])
        norms = vectors.norm(dim=-1)
        if not norms.numel():
            return
        threshold = float(norms.mean() + self.config.outlier_sigma * norms.std(unbiased=False))
        picks = (norms > threshold).nonzero(as_tuple=True)[0][:self.config.outlier_limit]
        tokenizer = getattr(self.model, "tokenizer", None)
        for pick in picks.tolist():
            token = int(indices[pick])
            token_id = None
            if getattr(self, "_batch_ids", None) is not None and token < len(self._batch_ids):
                token_id = int(self._batch_ids[token])
            text = None
            if token_id is not None and tokenizer is not None:
                try:
                    text = tokenizer.decode([token_id])
                except Exception:
                    text = None
            record = {"module": name, "token": token, "norm": float(norms[pick]),
                      "max_abs": float(vectors[pick].abs().max()),
                      "token_id": token_id, "text": text}
            self._pending_outliers.append(record)
            self._outlier_summary.append(record)

    def _signals(self, batch):
        if self._signal is not None:
            return [self._signal]
        selector = getattr(self._wrapper, "learning_signal_selector", None)
        if selector is not None:
            return list(selector.pick(batch, getattr(self, "_batch_index", None)) or [])
        return []

    def _forward_and_loss(self, batch):
        signals = self._signals(batch)
        if not signals:
            self.model(batch)
            return None
        total = None
        for signal in signals:
            input_sequence, metadata = signal.prepare_inputs(batch)
            predictions = self.model(input_sequence, **signal.model_kwargs(metadata))
            predictions, targets = signal.select_targets(predictions, batch, metadata)
            loss = signal.loss_function(predictions, targets)
            total = loss if total is None else total + loss
        return total

    def _gradient_stats(self, loss):
        parameters = [(name, parameter) for name, parameter in _named_parameters(self.model)
                      if parameter.requires_grad]
        if not parameters:
            return
        try:
            gradients = torch.autograd.grad(loss, [parameter for _, parameter in parameters],
                                            allow_unused=True, retain_graph=False)
        except RuntimeError as problem:
            self._loss.update({"gradient_error": 1.0})
            self.status = f"gradients skipped: {problem}"
            return
        total_norm_sq, max_abs, counted = 0.0, 0.0, 0
        for (name, parameter), gradient in zip(parameters, gradients):
            if gradient is None:
                continue
            detached = gradient.detach()
            stats = compute_tensor_stats(detached, spectral=False, vector_stats=False,
                                         outlier_sigma=self.config.outlier_sigma)
            self._gradients_all.update(stats, weight=max(parameter.numel(), 1))
            if self.config.per_parameter_gradients:
                self._gradients[name].update(stats, weight=max(parameter.numel(), 1))
            total_norm_sq += float(detached.float().pow(2).sum())
            max_abs = max(max_abs, float(detached.abs().max()))
            counted += 1
        self._loss.update({"loss": float(loss.detach()),
                           "grad_norm": math.sqrt(total_norm_sq),
                           "grad_max_abs": max_abs,
                           "grad_tensors": float(counted)})

    def process_batch(self, batch):
        self._batch_index = self._batches
        tensor = getattr(batch, "tensor", None)
        self._batch_ids = tensor.detach().reshape(-1).cpu() if torch.is_tensor(tensor) else None
        self._tokens += _batch_tokens(batch)
        was_training = self.model.training
        if self.config.evaluate_mode:
            self.model.eval()
        loss = None
        context = nullcontext() if self.config.gradients else torch.no_grad()
        try:
            with context:
                if getattr(self._wrapper, "runtime_mode", None) == "virtual":
                    produced = self.model(batch)
                    self._capture("virtual", produced)
                else:
                    tracing = hasattr(self._wrapper, "trace_activations")
                    context_trace = self._wrapper.trace_activations(
                        self._capture, include=self.config.include,
                        exclude=self.config.exclude) if tracing else nullcontext()
                    with context_trace:
                        loss = self._forward_and_loss(batch)
            if self.config.gradients and loss is not None and loss.requires_grad:
                self._gradient_stats(loss)
            elif loss is not None:
                self._loss.update({"loss": float(loss.detach())})
            if self.config.module_logs and hasattr(self._wrapper, "collect_module_diagnostics"):
                for name, values in self._wrapper.collect_module_diagnostics().items():
                    self._module_logs[name].update(_numeric(values), weight=1.0)
            if self._pending_outliers and self.logger is not None \
                    and hasattr(self.logger, "log_records"):
                step, epoch = self._position()
                self.logger.log_records(self.config.name, self._pending_outliers,
                                        step=step, epoch=epoch, task_run=self.execution)
                self._pending_outliers = []
        finally:
            if self.config.evaluate_mode:
                self.model.train(was_training)
            self._batches += 1
        return batch, None

    def compute(self) -> dict:
        out = {"batches": self._batches, "tokens": self._tokens}
        for name, accumulator in self._activations.items():
            out.update(accumulator.summary(f"activations/{name.replace('.', '/')}/"))
        out.update(self._gradients_all.summary("gradients/all/"))
        for name, accumulator in self._gradients.items():
            out.update(accumulator.summary(f"gradients/{name.replace('.', '/')}/"))
        out.update(self._loss.summary())
        for name, accumulator in self._module_logs.items():
            out.update(accumulator.summary(f"diagnostics/{name.replace('.', '/')}/"))
        if self._outlier_summary:
            out["outliers"] = sorted(self._outlier_summary, key=lambda one: one["norm"],
                                     reverse=True)[:self.config.outlier_limit]
        if self._profiler is not None:
            if getattr(self._profiler, "_running", False):
                self._profiler.stop()          # compute() runs before end_task(); close the clock
            out.update(self._profiler.report())
        return out
