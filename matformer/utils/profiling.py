"""Component-level timing and memory, attached from the outside.

The profiler registers forward hooks on modules, groups the calls by architectural component
(embedding, attention, MLP, norm, head, whole block) and accumulates wall time and CUDA allocator
deltas. Nothing in the model knows it exists: no ``timeit`` calls are scattered through the
forward, and a normal training pays nothing because the hooks are attached only while a profiler
is running.

    with ComponentProfiler(model) as profiler:
        model(batch)
    profiler.report()   # {component: {calls, seconds, fractions, memory}}

It is intended for analysis tasks and RPC commands: "where is the step time going?" is answered
by a forward pass on a batch, not by instrumenting every layer by hand.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

import torch


def component_of(name: str, module) -> str:
    """Map a dotted module path/class to a coarse architecture component."""
    lowered = name.lower()
    kind = type(module).__name__.lower()
    if "attn" in lowered or "attention" in kind or "mixer" in lowered:
        return "attention"
    if "mlp" in lowered or "ffn" in lowered:
        return "mlp"
    if "norm" in lowered:
        return "norm"
    if "embed" in lowered:
        return "embedding"
    if "head" in lowered or "classifier" in lowered:
        return "head"
    if "block" in lowered or "layer" in kind:
        return "block"
    return "other"


@dataclass
class _Component:
    calls: int = 0
    seconds: float = 0.0
    max_seconds: float = 0.0
    allocated_delta: float = 0.0
    allocated_max: float = 0.0
    reserved_delta: float = 0.0
    per_module: dict = field(default_factory=lambda: defaultdict(float))


class ComponentProfiler:
    """Forward-hook profiler. Use as a context manager, then call :meth:`report`."""

    def __init__(self, model, include=None, exclude=None, memory: bool = True,
                 per_module: bool = False):
        self.model = model
        self.include = list(include or [])
        self.exclude = list(exclude or [])
        self.memory = bool(memory)
        self.per_module = bool(per_module)
        self.cuda = torch.cuda.is_available()
        self.components = defaultdict(_Component)
        self.started = 0.0
        self.total_seconds = 0.0
        self.memory_start = {}
        self.memory_end = {}
        self._handles = []
        self._calls = defaultdict(list)
        self._module_of = {}
        self._running = False

    def modules(self):
        """Modules worth timing: named modules of the architectural components."""
        from fnmatch import fnmatch
        for name, module in self.model.named_modules():
            if not name:
                continue
            if self.include and not any(fnmatch(name, p) for p in self.include):
                continue
            if self.exclude and any(fnmatch(name, p) for p in self.exclude):
                continue
            if component_of(name, module) == "other":
                continue
            yield name, module

    def _before(self, name):
        start = time.perf_counter()
        event = torch.cuda.Event(enable_timing=True) if self.cuda else None
        if event is not None:
            event.record()
        allocated = torch.cuda.memory_allocated() if (self.cuda and self.memory) else 0
        reserved = torch.cuda.memory_reserved() if (self.cuda and self.memory) else 0
        self._calls[name].append([start, event, allocated, reserved])

    def _after(self, name):
        stack = self._calls.get(name)
        if not stack:
            return
        start, event, allocated, reserved = stack.pop()
        if event is not None:
            after = torch.cuda.Event(enable_timing=True)
            after.record()
            after.synchronize()
            seconds = max(event.elapsed_time(after) / 1000.0, 0.0)
        else:
            seconds = time.perf_counter() - start
        component = component_of(name, self._module_of.get(name))
        record = self.components[component]
        record.calls += 1
        record.seconds += seconds
        record.max_seconds = max(record.max_seconds, seconds)
        if self.per_module:
            entry = self.components[f"{component}/{name}"]
            entry.calls += 1
            entry.seconds += seconds
            entry.max_seconds = max(entry.max_seconds, seconds)
        if self.cuda and self.memory:
            record.allocated_delta += torch.cuda.memory_allocated() - allocated
            record.allocated_max = max(record.allocated_max,
                                       torch.cuda.memory_allocated() - allocated)
            record.reserved_delta += torch.cuda.memory_reserved() - reserved

    def start(self):
        if self._running:
            return self
        self.started = time.perf_counter()
        if self.cuda and self.memory:
            self.memory_start = {"allocated": torch.cuda.memory_allocated(),
                                 "reserved": torch.cuda.memory_reserved(),
                                 "peak": torch.cuda.max_memory_allocated()}
        for name, module in self.modules():
            self._module_of[name] = module
            self._handles.append(module.register_forward_pre_hook(
                lambda _m, _i, _name=name: self._before(_name)))
            self._handles.append(module.register_forward_hook(
                lambda _m, _i, _o, _name=name: self._after(_name)))
        self._running = True
        return self

    def stop(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []
        if self._running:
            self.total_seconds += time.perf_counter() - self.started
        if self.cuda and self.memory:
            self.memory_end = {"allocated": torch.cuda.memory_allocated(),
                               "reserved": torch.cuda.memory_reserved(),
                               "peak": torch.cuda.max_memory_allocated()}
        self._running = False
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()

    def report(self) -> dict:
        total = max(self.total_seconds, 1e-9)
        out = {"profile/total_seconds": self.total_seconds}
        if self.memory_start and self.memory_end:
            for key in ("allocated", "reserved", "peak"):
                out[f"profile/gpu_{key}_start_mib"] = self.memory_start[key] / 2 ** 20
                out[f"profile/gpu_{key}_end_mib"] = self.memory_end[key] / 2 ** 20
            out["profile/gpu_peak_growth_mib"] = max(
                0.0, (self.memory_end["peak"] - self.memory_start["peak"]) / 2 ** 20)
        for component, record in self.components.items():
            prefix = f"profile/{component.replace('/', '.')}/"
            calls = max(record.calls, 1)
            out[f"{prefix}calls"] = record.calls
            out[f"{prefix}seconds"] = record.seconds
            out[f"{prefix}fraction"] = record.seconds / total
            out[f"{prefix}mean_ms"] = 1000.0 * record.seconds / calls
            out[f"{prefix}max_ms"] = 1000.0 * record.max_seconds
            if self.cuda and self.memory:
                out[f"{prefix}cuda_allocated_delta_mib"] = record.allocated_delta / 2 ** 20
                out[f"{prefix}cuda_allocated_max_mib"] = record.allocated_max / 2 ** 20
                out[f"{prefix}cuda_reserved_delta_mib"] = record.reserved_delta / 2 ** 20
        return out
