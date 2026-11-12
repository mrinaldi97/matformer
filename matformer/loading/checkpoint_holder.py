import json
import re
from collections import namedtuple
from fnmatch import fnmatch
from itertools import chain
from pathlib import Path

import einx
import torch
import yaml
from safetensors.torch import load_file

from matformer.matformer_registry import registry
from matformer.loading.utils import  _get_path, _axis_size, _ShardedWeights, StableWeights, StateHolder


class CheckpointHolder:
    """
    Base class for every checkpoint source. A subclass only says WHERE things are:
        - `_DEFS`            folder with its adapter definitions (YAML)
        - __init__           sets `source_config` (raw config dict, used for transform
                             axis sizes), `definition` (adapter dict) and `num_layers`
        - `_raw_weights`     returns a dict-like of raw {source_key: tensor}
        - `get_configs`      converts its native config into MatformerConfigs
    Renaming, N-template expansion, transforms and streaming live here, once.

    Adapter / definition schema (same for YAML adapters and user overrides):
        alias: <model_type>             # optional: load <model_type>.yaml instead (dummy
                                        # yaml for architectures identical to a supported family)
        num_layers: int                 # optional, overrides the holder's own
        source_config: {...}            # optional, overrides the holder's own (for axis sizes)
        weight_mapping:  [ {stable_key: source_key}, ... ]      # 'N' = layer index
        weight_transforms:
          - sources: [source_key, ...]  # fetched and passed positionally to the einx op
            outputs: [stable_key, ...]
            op: rearrange               # any public einx function; default rearrange
            pattern: "(q + k + v) c -> q c, k c, v c"
            axes: {q: <int | dotted.path | a * b>, ...}
        If the op returns one tensor and there are several outputs, it is unbound along axis 0.
    """
    _DEFS: Path | None = None

    def __init__(self, path, map_location=None):
        self.path = Path(path)
        self.map_location = map_location
        self.source_config, self.definition, self.num_layers = {}, None, None
        self.unresolved = []   # source keys that could not be placed; filled while weights stream

    # --- public API
    def get_configs(self):
        raise NotImplementedError

    def get_states(self):
        """Optimizer/scheduler/datamodule/step/epoch, if the source has them."""
        return None

    def get_weights(self, definition=None, wanted=None, discard=None):
        """Lazy stable-weights view. `wanted`/`discard` (mutually exclusive) are stable key
        names or fnmatch patterns, applied while streaming: unwanted weights are never
        fetched, never transformed, and never reported as unresolved."""
        keep = self._keep_filter(wanted, discard)
        definition = self.definition if definition is None else definition
        if definition is None:
            raise ValueError(...)
        rename, rules, aliases = self._expand_definition(definition)
        cfg = definition.get("source_config", self.source_config)
        raw = self._raw_weights(rename, rules, aliases)
        return StableWeights(lambda: self._stream(raw, rename, rules, cfg, aliases, keep))

    @staticmethod
    def _keep_filter(wanted, discard):
        """Stable-key predicate built from the mutually exclusive wanted/discard pair
        (each a key name, an fnmatch pattern, or an iterable of them); None keeps everything."""
        if wanted is not None and discard is not None:
            raise ValueError("'wanted' and 'discard' are mutually exclusive: pass at most one.")
        if wanted is None and discard is None:
            return None
        def _matcher(spec):
            patterns = [spec] if isinstance(spec, str) else list(spec)
            return lambda key: any(fnmatch(key, p) for p in patterns)
        if wanted is not None:
            return _matcher(wanted)
        _discards = _matcher(discard)
        return lambda key: not _discards(key)

    def _raw_weights(self, rename, rules):
        """Dict-like {source_key: tensor}. `rename`/`rules` let lazy sources know which keys matter."""
        raise NotImplementedError

    # --- definitions
    def _load_definition(self, model_type, _seen=None):
        if self._DEFS is None:
            raise NotImplementedError(f"{type(self).__name__} must set `_DEFS`.")
        path = self._DEFS / f"{model_type}.yaml"
        if not path.exists():
            raise NotImplementedError(
                f"No definition for '{model_type}' in {self._DEFS}. "
                f"Supported: {[p.stem for p in self._DEFS.glob('*.yaml')]}")
        with open(path) as f:
            definition = yaml.safe_load(f)
        # A definition that is just a pointer to another one (e.g. a dummy
        # <model_type>.yaml containing only `alias: mistral` for an architecture
        # identical to an already-supported family).
        alias = definition.get("alias") if isinstance(definition, dict) else None
        if alias:
            _seen = (_seen or set()) | {model_type}
            if alias in _seen:
                raise ValueError(f"Alias cycle: {' -> '.join(sorted(_seen | {alias}))}")
            return self._load_definition(alias, _seen)
        return definition

    def _expand_definition(self, definition):
        n = definition.get("num_layers", self.num_layers)
        return (
            self._expand_weight_mapping(definition.get("weight_mapping", {}), n),
            self._expand_transform_rules(definition.get("weight_transforms", []), n),
            definition.get("source_key_aliases", []),
        )
    @staticmethod
    def _candidate_source_keys(key, aliases):
        yield key
        for prefix in aliases:
            if key.startswith(prefix):
                yield key[len(prefix):]
            else:
                yield prefix + key

    def _resolve_source_key(self, raw, key, aliases):
        # alias candidates may be pre-registered in the shard map without an actual
        # tensor (absent from this checkpoint's files): skip those, otherwise a
        # present-but-None first candidate would shadow a later real one
        for candidate in self._candidate_source_keys(key, aliases):
            if candidate in raw and raw[candidate] is not None:
                return candidate
        return None

    # --- N-template machinery
    @staticmethod
    def _sub_n(item: str, i) -> str:
        """Replace only the exact 'N' dot-token with the layer index (other 'N' characters,
        e.g. inside 'LayerNorm', are untouched)."""
        return ".".join(str(i) if p == "N" else p for p in item.split("."))

    @staticmethod
    def expand_template(items, N):
        """['a.N.b', 'c'] -> ['a.0.b', ..., 'a.N-1.b', 'c']"""
        out = []
        for item in items:
            if "N" in item.split("."):
                out.extend(CheckpointHolder._sub_n(item, i) for i in range(N))
            else:
                out.append(item)
        return out

    @staticmethod
    def collapse_template(items):
        """Inverse of expand_template."""
        return list({".".join(re.sub(r"\d+", "N", p) for p in item.split(".")) for item in items})

    @staticmethod
    def _expand_weight_mapping(raw, num_layers):
        """{stable: source} (dict, or YAML list of single-entry dicts), N-templated -> flat {source: stable}."""
        items = raw.items() if isinstance(raw, dict) else (next(iter(d.items())) for d in raw)
        out = {}
        for stable, source in items:
            if "N" in stable.split("."):
                for i in range(num_layers):
                    out[CheckpointHolder._sub_n(source, i)] = CheckpointHolder._sub_n(stable, i)
            else:
                out[source] = stable
        return out

    @staticmethod
    def _expand_transform_rules(raw_rules, num_layers):
        """One concrete rule per layer for rules whose sources/outputs contain 'N'."""
        out = []
        for rule in raw_rules or []:
            has_n = any("N" in s.split(".") for s in rule["sources"] + rule["outputs"])
            for i in (range(num_layers) if has_n else [None]):
                sub = (lambda s, i=i: CheckpointHolder._sub_n(s, i)) if i is not None else (lambda s: s)
                out.append({**rule, "sources": [sub(s) for s in rule["sources"]],
                                    "outputs": [sub(s) for s in rule["outputs"]]})
        return out

    # --- resolution
    def _stream(self, raw, rename, rules, cfg, aliases, keep=None):
        """Yield (stable_key, tensor): first the plain renames, then the transforms.
        `keep`, when given, is a stable-key predicate (see _keep_filter): unwanted weights
        are silently consumed, never materialised."""
        self.unresolved = []
        transform_sources = {s for r in rules for s in r["sources"]}
        consumed = set()

        for source_key, stable_key in rename.items():
            actual = self._resolve_source_key(raw, source_key, aliases)
            if actual is None:
                if keep is None or keep(stable_key):
                    self.unresolved.append(source_key)
                continue
            consumed.add(actual)
            if keep is not None and not keep(stable_key):
                continue
            tensor = raw[actual]
            if tensor is None:
                self.unresolved.append(stable_key)
            else:
                yield stable_key, tensor

        for key in raw.keys():
            if key in consumed or key in transform_sources or "_extra_state" in key:
                continue
            if raw[key] is None:
                continue    # key was only pre-registered (needed-keys/alias candidates)
            self.unresolved.append(key)

        for rule in rules:
            if keep is not None and not any(keep(o) for o in rule["outputs"]):
                continue    # nothing this rule produces is wanted: skip the op entirely
            resolved = {s: self._resolve_source_key(raw, s, aliases) for s in rule["sources"]}
            if any(v is None or raw[v] is None for v in resolved.values()):
                continue
            for stable_key, tensor in self._apply_weight_transform(rule, raw, cfg, resolved).items():
                if keep is None or keep(stable_key):
                    yield stable_key, tensor

    def _apply_weight_transform(self, rule, raw, cfg, resolved):
        """Run one weight_transforms rule through its einx op. Returns {stable_key: tensor}."""
        op_name, pattern, outputs = rule.get("op", "rearrange"), rule["pattern"], rule["outputs"]
        op = None if op_name.startswith("_") else getattr(einx, op_name, None)
        if not callable(op):
            raise ValueError(f"Unknown einx operation '{op_name}' in weight_transform for {outputs}")
        try:
            axes = {name: _axis_size(spec, cfg) for name, spec in rule.get("axes", {}).items()}
            sources = [raw[resolved[s]] for s in rule["sources"]]
            if rule.get("stack"):
                # Merge several source tensors into one (leading axis), e.g. HF's separate
                # q/k/v projections into Matformer's packed qkv_proj. The pattern then
                # operates on the stacked tensor (leading axis 's' by convention).
                sources = [torch.stack(sources, dim=0)]
            result = op(pattern, *sources, **axes)
            parts = list(result) if isinstance(result, (tuple, list)) \
                else [result] if len(outputs) == 1 else list(result.unbind(0))
            if len(parts) != len(outputs):
                raise ValueError(f"produced {len(parts)} tensors for {len(outputs)} outputs")
        except Exception as e:
            raise RuntimeError(f"weight_transform {outputs} failed (einx.{op_name} '{pattern}'): {e}") from e
        return dict(zip(outputs, parts))
    def release(self):
        """Drop references to loaded tensors so their memory can be freed."""
        for attr in ("checkpoint", "state_dict"):
            if hasattr(self, attr):
                setattr(self, attr, None)

