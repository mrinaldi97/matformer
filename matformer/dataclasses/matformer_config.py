# matformer/dataclasses/matformer_config.py
from dataclasses import fields, dataclass, field, asdict, replace

from typing import Optional, ClassVar, Any, get_type_hints, get_origin, get_args, Union

from matformer.matformer_registry import registry
@dataclass
class MatformerConfig:
    """
    Base for all Matformer config dataclasses.
    A subclass can be linked to its registered implementation with the registry decorator:
        @registry.register_dataclass("saver", "JSONLSaver", default=True)
    which stamps _matformer_category / _matformer_variant onto the class.
    When this function is used, load_config.py resolves everything through the registry.
    """
    _merge_fields: ClassVar[frozenset] = frozenset()
    _matformer_category: ClassVar[Optional[str]] = None
    type: ClassVar[Optional[str]] = None
    _matformer_variant:  ClassVar[Optional[str]] = None
    _matformer_multi: ClassVar[bool] = False
    name: Optional[str] = field(default="default", kw_only=True)
    _ref_fields: ClassVar[dict] = {}    
    _mandatory_references: ClassVar[list] = []
    instance = None #When instantiated, the object is cached
    # dict-style accessors
    def __getitem__(self, key: str) -> Any:
        """Allow config[key] access."""
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow config[key] = value assignment."""
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-style .get with a default, mirroring dict.get."""
        return getattr(self, key, default)

    def asdict(self, include_defaults: bool = True) -> dict:
        d = asdict(self)
        if not include_defaults:
            explicit = getattr(self, '_explicit_fields', d.keys())
            d = {k: v for k, v in d.items() if k in explicit}
        if self._matformer_variant is not None:
            d['type'] = self._matformer_variant # Type is preserved for correctly serializing the class
        return d
    def _check_ref_safety(self):
        """_ref_fields pointers get stamped onto the instance as attributes named after the
        target section (see resolve_references), so no dataclass field may share that name."""
        valid = {f.name for f in fields(self)}
        for field_name, target_section in getattr(self, '_ref_fields', {}).items():
            if target_section in valid:
                raise ValueError(f"'{type(self).__name__}' declares a field '{target_section}' that collides with ref target section '{target_section}' (from '{field_name}')")
    @classmethod
    def _field_types(cls):
        """Cached {field_name: resolved type}, so Optional[...]-unwrapping only happens once
        per class rather than on every dict coerced."""
        cached = cls.__dict__.get('_cached_field_types')
        if cached is None:
            cached = get_type_hints(cls)
            cls._cached_field_types = cached
        return cached

    @classmethod
    def _coerce_dict(cls, d: dict) -> dict:
        """Coerce string values onto their field's declared int/float type. Needed because
        PyYAML doesn't recognize exponential notation without a decimal point (e.g. '5e-4') as
        a float -- it stays a string -- and dataclasses never validate/cast field types on
        construction, so it would otherwise silently overwrite a float default and fail
        downstream (e.g. AdamW's `0.0 <= lr` check)."""
        types = cls._field_types()
        out = {}
        for k, v in d.items():
            t = types.get(k)
            if get_origin(t) is Union and type(None) in get_args(t):
                t = next(a for a in get_args(t) if a is not type(None))
            out[k] = t(v) if isinstance(v, str) and t in (float, int) else v
        return out

    def update(self, d: dict, strict: bool = False) -> "MatformerConfig":
        """Like from_dict, but merges onto an existing instance instead of the class defaults,
        and extends (not replaces) the record of which fields were explicitly set.

        Fields in _merge_fields are deep-merged (see deep_merge); a field currently holding a
        MatformerConfig instance (e.g. default_layer) delegates to that instance's own update(),
        so nested configs patch instead of being overwritten wholesale."""
        d = type(self)._coerce_dict(d)
        valid = {f.name for f in fields(self)}
        unknown = d.keys() - valid
        if strict and unknown:
            raise ValueError(f"Unknown field(s) {sorted(unknown)} for {type(self).__name__}")
        resolved = {}
        for k, v in d.items():
            if k not in valid:
                continue
            current = getattr(self, k)
            if k in self._merge_fields and isinstance(current, dict) and isinstance(v, dict):
                resolved[k] = deep_merge(current, v)
            elif isinstance(current, MatformerConfig) and isinstance(v, dict):
                resolved[k] = current.update(v, strict=strict)
            else:
                resolved[k] = v
        new = replace(self, **resolved)
        explicit = getattr(self, '_explicit_fields', set()) | (valid & d.keys())
        object.__setattr__(new, '_explicit_fields', explicit)
        new._check_ref_safety()
        return new
  
# construction
    @classmethod
    def from_dict(cls, d: dict, strict: bool = False) -> "MatformerConfig":
        """Build an instance from a dict. If strict (default), raises ValueError on keys that
        aren't declared dataclass fields; otherwise those keys are silently dropped.

        Also records, on the instance, which fields were explicitly present in 'd' and not
        override with default
        """
        d = cls._coerce_dict(d)
        valid = {f.name for f in fields(cls)}
        unknown = d.keys() - valid
        if strict and unknown:
            raise ValueError(f"Unknown field(s) {sorted(unknown)} for {cls.__name__}; valid fields: {sorted(valid)}")
        instance = cls(**{k: v for k, v in d.items() if k in valid})
        object.__setattr__(instance, '_explicit_fields', valid & d.keys())
        instance._check_ref_safety()
        return instance

    def instantiate(self, use_cache=True, refresh=False, preferred=None, *args, **kwargs):
        """Resolve section_name (and object_name for grouped sections, falling back to that
        group's default) to its MatformerConfig, then build the implementation class registered
        for that config's category/variant -- e.g. configs.instantiate('data', 'train').
        
        Args:
            cache: If True, cache the instance for reuse
            refresh: If True, force re-instantiation even if cached
            preferred: Optional preferred implementation name to pass to registry
        """
        if not refresh and use_cache and self.instance:
            return self.instance
        assert 'config' not in kwargs, "'config' is a reserved kwarg for instantiate()."
        
        # Use registry.create to get the instantiated object
        # The registry will handle priority, environment variables, etc.
        instance = registry.create(
            category=self._matformer_category,
            variant=self._matformer_variant,
            preferred=preferred,
            config=self,  # Pass config as kwarg
            **kwargs
        )
        #if hasattr(instance,'__init__'):
        #    instance.__init__()

        if refresh or use_cache:
            self.instance = instance
            return self.instance
        else:
            return instance   
@dataclass
class MatformerMultiConfig(MatformerConfig):
    """Marker base for multi-entry categories (data, task, validation, ...)."""
    _matformer_multi: ClassVar[bool] = True        

def deep_merge(base: dict, patch: dict) -> dict:
    """Recursively merge `patch` onto `base`: nested dicts merge key-wise, a patch value of
    None deletes that key (so a patch can remove something inherited), everything else
    (including lists) replaces outright. Shared by MatformerConfig.update() for fields
    declared in _merge_fields, and by ModelConfig's per-layer cascade."""
    out = dict(base)
    for k, v in patch.items():
        if v is None and k in out:
            del out[k]
        elif isinstance(out.get(k), dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out    

