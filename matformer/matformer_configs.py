# matformer/matformer_configs.py
"""
Load YAML/JSON config file(s) into a tree of typed MatformerConfig dataclasses, addressable by
attribute access.

Each top-level key must be a category registered in the matformer registry
(registry.config_categories()):

    data:
      type: HFDataset        # optional; falls back to that category's default=True variant
      batch_size: 32

A section's "type" selects the dataclass variant registered under that category; the rest of the
section is passed to that dataclass's from_dict(). "Listed" categories (registry._listed_configs)
may hold several named variants, given as a YAML list:

    data:
      - name: train
        type: HFDataset
      - name: test
        type: HFDataset

which becomes configs.data.train / configs.data.test (the first entry is that group's default,
so configs.data.batch_size also works, proxying to configs.data.train). Re-declaring a section
patches its existing fields rather than replacing the instance (see MatformerConfig.update()).

A file may declare "includes: [other.yaml, ...]" (or a bare string), resolved relative to that
file's own directory and merged in first, so the including file's own content always wins.

Every string of a document (paths in it, includes, override values) goes through MATFORMER_ASSETS
token expansion, so a config may address the shared assets folder without machine-specific paths:

data_root: "$DATA_ASSETS/IRONITA/train.jsonl"   # also $ASSETS, $CONFIG_ASSETS, $CHECKPOINT_ASSETS...

    configs = MatformerConfigs()
    configs.load_config("base.yaml")
    configs.set_overrides(["data.train.batch_size=64", "model.num_labels=5"])
    configs.resolve_references()
    model = configs.instantiate("model")
    configs.save_config("run_config.yaml", exclude=["data", "validation"])

A value may declare several candidates instead of one, separated by a pipe, and the file then
stands for a sweep rather than for a single run:

    training:
      lr: 2e-3 | 1e-4
    model:
      positional_encoding: alibi | [alibi, rope]

The candidates land in `configs.sweeps` and the configs themselves keep the first one, so a config
nobody expands is still a perfectly valid run. `configs.sweep_variants()` is the expansion: one
detached config per combination, each renamed so its logs, checkpoints and saved outputs can never
collide with another point's. See matformer/sweeps.py.
"""
import os
import json
import dataclasses
import yaml
from collections import defaultdict
from matformer.matformer_registry import registry
from matformer.utils import matformer_assets
from matformer.dataclasses.matformer_config import MatformerConfig
from matformer import sweeps

def _set_nested(d: dict, dotted_path: str, value):
    *path, leaf = dotted_path.split('.')
    for key in path:
        d = d.setdefault(key, {})
    d[leaf] = value
def _get_nested(mapping, path):
    if isinstance(path, str):
        path = path.split(".")
    value = mapping
    for part in path:
        if not isinstance(value, dict) or part not in value:
            raise KeyError(part)
        value = value[part]
    return value


_MISSING = object()


def _literal(text):
    """An override's right-hand side as a Python value. What is not valid YAML on its own -- a
    sweep declaration like `[alibi,rope] | alibi` -- stays the string the sweep parser splits."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text.strip().strip('"\'')


def _patch_value(node, keys, value, path=''):
    """Set `keys` -- a path inside one config instance -- to `value`, and return the instance.

    `update()` writes the leaf; a path that goes through a plain dict or list field rebuilds just
    that container first, so 'training.lr', 'task.test.topk' and 'data.train.label2id.0' all
    address something real."""
    if not keys:
        return value
    head, *rest = keys
    if not rest:
        return node.update({head: value})
    inner = node.get(head)
    if isinstance(inner, MatformerConfig):
        return node.update({head: _patch_value(inner, rest, value, path)})
    patched = _patch_container(inner, rest, value)
    if patched is _MISSING:
        raise ValueError(f"'{path}': no such field inside '{head}' of {type(node).__name__}")
    return node.update({head: patched})


def _patch_container(node, keys, value):
    """The dict/list twin of _patch_value: copy the branch, set the leaf, hand the whole rebuilt
    container back to the config that owns it. _MISSING when the path is not there."""
    head, *rest = keys
    index = _lookup(node, head)
    if index is _MISSING:
        return _MISSING
    child = value if not rest else _patch_container(node[index], rest, value)
    if child is _MISSING:
        return _MISSING
    out = list(node) if isinstance(node, (list, tuple)) else dict(node)
    out[index] = child
    return out


def _lookup(node, head):
    """The key `head` names inside a plain container -- dict keys and list positions both arrive
    from a dotted path as strings, and YAML dict keys may be ints."""
    number = head.lstrip('-').isdigit()
    if isinstance(node, dict):
        if head in node:
            return head
        if number and int(head) in node:
            return int(head)
    elif isinstance(node, (list, tuple)) and number and 0 <= int(head) < len(node):
        return int(head)
    return _MISSING
class GroupConfig:
    """Holds one or more named MatformerConfig instances under a shared section, e.g.
    configs.data.train / configs.data.test. MatformerConfigs itself is the root of this same
    structure (parent=None, section_name=None), keyed by section name instead of variant
    name."""

    def __init__(self, section_name=None, parent=None):
        self.section_name = section_name
        self.parent = parent
        self.objects = []
        self.default = None

    def list_configs(self):
        return self.objects
    def __iter__(self):
        for k,v in self.items():
            yield k,v
    def items(self):
        """Dict-style (name, config) pairs over this group's entries"""
        return [(name, getattr(self, name)) for name in self.objects]
    def add_or_update_dataclass(self, section_name, object_name, section_config, dataclass_obj):
        """Register/patch one entry. At the root (parent is None), entries are keyed by
        section_name (configs.model); inside a group they're keyed by object_name, with the
        first entry becoming that group's default (configs.data -> configs.data.train)."""
        if self.parent is not None:
            assert section_name == self.section_name, (
                f"FATAL: This is the GroupConfig for {self.section_name}, but you are trying "
                f"to add a config with section name: {section_name}"
            )
            if self.section_name not in self.parent.objects:
                self.parent.objects.append(self.section_name)
        
        key = object_name if self.parent is not None else section_name
        is_new = key not in self.objects
        
        if is_new:
            setattr(self, key, dataclass_obj.from_dict(section_config, strict=False))
            self.objects.append(key)
        else:
            existing = getattr(self, key)
            
            # If incoming config does not specify a type, preserve existing (more specific) type
            if 'type' not in section_config and type(existing) is not dataclass_obj:
                dataclass_obj = type(existing)
            
            if type(existing) is not dataclass_obj:
                # Type change with explicit type field
                skip = set(getattr(existing, '_linked_fields', [])) | set(getattr(existing, '_ref_fields', {}).keys())
                explicit_fields = getattr(existing, '_explicit_fields', set())
                base_fields = {
                    k: v for k, v in existing.asdict(include_defaults=False).items() 
                    if k not in skip and k in explicit_fields
                }
                merged_config = {**base_fields, **section_config}
                setattr(self, key, dataclass_obj.from_dict(merged_config, strict=False))
            else:
                setattr(self, key, existing.update(section_config))

        if is_new and self.parent is not None and self.default is None:
            self.default = key

    def set_default(self, name):
        assert name in self.objects, f"You are trying to set {name} as default, but it is not registered in this group"
        self.default = name

    def instances(self):
        """Flatten every dataclass instance held here, recursing into nested groups."""
        out = []
        for name in self.objects:
            obj = getattr(self, name)
            out.extend(obj.instances()) if isinstance(obj, GroupConfig) else out.append(obj)
        return out
    def remove_section(self, section_name: str):
        """Remove a top-level section from the config tree."""
        if section_name in self.objects:
            self.objects.remove(section_name)
        if hasattr(self, section_name):
            delattr(self, section_name)
        return self
    def asdict(self, include_defaults=False):
        return [getattr(self, name).asdict(include_defaults) for name in self.objects]
    """
    def __getattr__(self, attribute):
        Unknown attribute access falls through to the group's default entry, e.g.
        configs.data.batch_size resolving via configs.data.train (the default)
        if self.default:
            return getattr(getattr(self, self.default), attribute)
        raise AttributeError(f"'{self.section_name}' has no attribute '{attribute}' and no default entry is set")
    """
    def __repr__(self):
        return f"<{self.section_name}: {', '.join(self.objects)} (default={self.default})>"
    

class MatformerConfigs(GroupConfig):
    def __init__(self):
        super().__init__()
        self.known_categories = registry.list_categories()
        # Sweeps: the candidates each field declares ('training.lr' -> [2e-3, 1e-4]), and the point
        # of a sweep this very instance stands for (None unless it is one of sweep_variants()).
        self.sweeps, self.active_sweep = {}, None

    def load_config(self, configs_paths=None, stdin=None, resolve=True):
        """Load one or more YAML/JSON files, or a YAML/JSON string via `stdin`. Each file's own
        'includes' are resolved relative to that file's directory (see import_config)."""
        assert configs_paths or stdin is not None, "Provide either config file path(s) or a stdin string"
        if stdin is not None:
            if isinstance(stdin,dict):
                self.import_config(stdin)
            else:
                self.import_config(yaml.safe_load(stdin))
        if configs_paths is not None:
            for config_path in ([configs_paths] if isinstance(configs_paths, str) else configs_paths):
                config_path = matformer_assets.expand(config_path)
                extension = config_path.split('.')[-1]
                assert extension in ('yaml', 'json'), "Please provide either a yaml or a json file."
                with open(config_path) as f:
                    config_dict = yaml.safe_load(f) if extension == 'yaml' else json.load(f)
                self.import_config(config_dict, base_dir=os.path.dirname(os.path.abspath(config_path)))
        if resolve:
            self.resolve_references()
    def import_config(self, config_dict, base_dir='.', sweeps_on=True):
        # 0. Magic MATFORMER_ASSETS tokens ($ASSETS, $DATA_ASSETS, ...) become absolute paths.
        config_dict = matformer_assets.expand_tree(config_dict)
        # 1. Resolve and merge includes first, relative to this dict's own file -- so this
        # file's own content is imported afterwards and wins on conflicts.
        includes = config_dict.pop('includes', None)
        if includes:
            includes = [includes] if isinstance(includes, str) else includes
            self.load_config([os.path.join(base_dir, inc) for inc in includes], resolve=False)
        # 2. Check if all the sections declared have their corresponding category in the registry
        declared_sections = config_dict.keys()
        if declared_sections - self.known_categories:
            raise ValueError(f"Unknown config section(s) {sorted(declared_sections - self.known_categories)}. Available: {sorted(self.known_categories)}")
        for section_name, sections_config in config_dict.items():
            listed = section_name in registry.list_categories(multi_only=True)
            if not listed:
                assert isinstance(sections_config, dict), f"You are providing a list of configs for section '{section_name}', but it supports only a single configuration. More than one {section_name} sections cannot co-exist in the same config."
            if listed and not hasattr(self, section_name):
                setattr(self, section_name, GroupConfig(section_name, parent=self))
            for section_config in (sections_config if isinstance(sections_config, list) else [sections_config]):
                section_type = section_config.get('type', None)
                object_name = section_config.get('name', None)
                dataclass_obj = registry.get_definition(section_name, section_type)
                if not dataclass_obj:
                    print(f"ERRORE: il registro ha emesso None per {section_name}, {section_type}")
                if object_name is None:
                    object_name = getattr(dataclass_obj, 'name', None) or f'default_{section_type}'
                target = getattr(self, section_name) if listed else self
                if sweeps_on:      # 'lr: 2e-3 | 1e-4' declares a sweep, not a broken float
                    prefix = f"{section_name}.{object_name}" if listed else section_name
                    section_config, found, touched = sweeps.harvest(section_config, prefix)
                    self.sweeps.update(found)
                    for path in touched:    # re-declared without pipes: that dimension is closed
                        if path not in found:
                            self.sweeps.pop(path, None)
                target.add_or_update_dataclass(section_name, object_name, section_config, dataclass_obj)

    def set_value(self, path, value, declare=True, resolve=True):
        """Set one value in place, addressed the way the configs themselves address it:
        'training.lr', 'task.test.topk', and -- going through plain dict/list fields --
        'data.train.label2id.0'. With `declare`, a value holding pipes declares a sweep instead of
        setting a single value (see matformer/sweeps.py)."""
        section, *rest = str(path).split('.')
        held = getattr(self, section, None)
        if held is None:
            raise ValueError(f"'{path}': these configs have no '{section}' section "
                             f"({', '.join(self.objects) or 'none loaded'})")
        if declare:
            value = sweeps.declare(path, value, self.sweeps)
        if isinstance(held, GroupConfig):
            if rest and rest[0] in held.objects:
                name, rest = rest[0], rest[1:]
            elif held.default is not None:
                name = held.default
            else:
                raise ValueError(f"'{path}': section '{section}' has no entry named "
                                 f"{rest[:1]} and no default one ({', '.join(held.objects)})")
            setattr(held, name, _patch_value(getattr(held, name), rest, value, path))
        else:
            setattr(self, section, _patch_value(held, rest, value, path))
        if resolve:
            self.resolve_references()
        return self

    def set_override(self, overrides):
        """Apply hydra-style dotlist override(s), e.g. 'data.train.batch_size=64'. The entry
        name may be omitted for a listed section with a single/default entry ('model.num_labels=5').
        An override whose value holds pipes ('training.lr=2e-3|1e-4') declares a sweep."""
        if isinstance(overrides, str):
            overrides = [overrides]
        for override in overrides:
            path, value = override.split('=', 1)
            self.set_value(path.strip(), matformer_assets.expand(_literal(value)))

    # -- sweeps -------------------------------------------------------------------------
    def checkpoint_path(self, name=None):
        """The saved model these configs say to load, or None when they say nothing: the config
        answer to `--model`, and what load_model falls back to when no path was given."""
        group = getattr(self, 'checkpoint', None)
        if group is None:
            return None
        if not isinstance(group, GroupConfig):
            return group.get('path')
        entry = getattr(group, name or group.default, None)
        return entry.get('path') if entry is not None else None

    def check_sweeps(self):
        """Fail on a swept path that addresses nothing.

        A dimension pointing at a field that does not exist would be substituted with its first
        candidate, silently produce N identical runs, and name them all differently: the one sweep
        mistake that goes unnoticed in a table of results, so it is raised instead."""
        for path in self.sweeps:
            section, *rest = path.split('.')
            node = getattr(self, section, None)
            if node is None:
                raise ValueError(f"swept '{path}': these configs have no '{section}' section "
                                 f"({', '.join(self.objects) or 'none loaded'})")
            if isinstance(node, GroupConfig):
                if rest and rest[0] in node.objects:
                    node, rest = getattr(node, rest[0]), rest[1:]
                elif node.default is not None:
                    node = getattr(node, node.default)
                else:
                    raise ValueError(f"swept '{path}': section '{section}' has no entries yet")
            for key in rest:
                valid = ([f.name for f in dataclasses.fields(node)]
                         if hasattr(node, '__dataclass_fields__') else [])
                if valid and key not in valid:
                    raise ValueError(f"swept '{path}': '{key}' is not a field of "
                                     f"{type(node).__name__} (available: {', '.join(sorted(valid))}). "
                                     f"Hint: 'type' and 'name' pick which config an entry *is*, "
                                     f"and cannot be swept.")
                node = node.get(key)
        return self

    def model_axes(self):
        """The `checkpoint` entries standing for more than one saved model, as sweep dimensions.

        A folder with a cadence, or a `a | b` path, is a *series* of models -- and a series is just
        another dimension, so the models a run evaluates cross-multiply with the learning rates it
        sweeps exactly as the two folders of checkpoints cross-multiply with each other. An entry
        that resolves to one model adds nothing: it is simply the model the run loads."""
        group = getattr(self, 'checkpoint', None)
        if not isinstance(group, GroupConfig):
            return {}
        axes = {}
        for name, entry in group.items():
            path = f"checkpoint.{name}.path"
            found = [one for candidate in (self.sweeps.get(path) or [entry.path])
                     for one in entry.expand(candidate)]
            if len(found) > 1:
                axes[path] = found
        return axes

    def sweep_points(self, selection=None):
        """Every combination the declared candidates span, in product order; `selection` narrows it
        down to point numbers or name globs ('1,3', 'lr-2e-3*'). Empty when nothing is swept."""
        self.check_sweeps()
        axes = self.model_axes()
        return sweeps.select(sweeps.plan({**axes,
                                         **{path: values for path, values in self.sweeps.items()
                                            if path not in axes}}), selection)

    def sweep_variant(self, point):
        """One point of the sweep, applied to a detached copy of these configs (matformer/sweeps.py)."""
        return sweeps.apply(self, point)

    def sweep_variants(self, selection=None):
        """One ready-to-run config per point of the sweep, and exactly [self] when nothing is
        swept -- so a caller never has to know which of the two it is facing.

        A variant shares nothing with the configs it came from: it is re-imported from their dict
        form, has its point applied, and has every artifact name its config classes declare
        (`_sweep_scoped`) tagged with the point's short name. The trainer, the loggers
        and the activation stores therefore keep reading their own configs, and each point of a
        sweep lands on a shelf of its own: one run folder, one checkpoint stem, one predictions
        file per point -- which is what makes the whole product comparable afterwards."""
        points = self.sweep_points(selection)
        return [self] if not points else [sweeps.apply(self, point) for point in points]

    def resolve_references(self, strict=False):
        """Validate that every '_ref_fields' entry declared on a config instance (e.g.
        {"ref_data": "data"}) points to a real section, and a real entry within it if grouped."""
        for instance in self.instances():
            for field_name, target_section in getattr(instance, '_ref_fields', {}).items():
                if not hasattr(self, target_section):
                    if target_section in getattr(instance,'_mandatory_fields',[]):
                        raise ValueError(f"'{instance} {instance.name}' requires section '{target_section}' (field '{field_name}'), but it is absent from the config")
                    else:
                        setattr(instance,target_section,None)
                else:
                    target = getattr(self, target_section)
                    referenced_name = getattr(instance, field_name)
                    if not isinstance(target,GroupConfig):
                        raise ValueError("You used _ref_fields but the requested section is a non-listed section, thus you must use _linked_fields instead.")
                    if isinstance(target, GroupConfig) and referenced_name not in target.objects:
                        raise ValueError(f"'{instance.name}.{field_name}' references '{referenced_name}', not a registered variant of '{target_section}'")
                    if isinstance(target,GroupConfig):
                        pointer=getattr(getattr(self,target_section),referenced_name)
                    setattr(instance,target_section,pointer)
            for field_name in getattr(instance,'_linked_fields',[]):
                if strict:
                    if not hasattr(self,field_name):
                        raise ValueError(f"'{instance} {instance.name}' requires section {field_name}, but it is absent from the config. Available sections: {self.list_configs()}")
                pointer=getattr(self,field_name,None)
                setattr(instance,field_name,pointer)
    def instantiate(self, section_name, object_name=None, *args, **kwargs):
        """Resolve section_name (and object_name for grouped sections, falling back to that
        group's default) to its MatformerConfig, then build the implementation class registered
        for that config's category/variant -- e.g. configs.instantiate('data', 'train')."""
        assert 'config' not in kwargs, "'config' is a reserved kwarg for instantiate()."
        target = getattr(self, section_name)
        if isinstance(target, GroupConfig):
            target = getattr(target, object_name or target.default)
        impl_cls = registry.create(target._matformer_category, target._matformer_variant, config=target, **kwargs)
        return impl_cls

    def asdict(self, include_defaults=False):
        return {name: getattr(self, name).asdict(include_defaults) for name in self.objects}

    def save_config(self, path=None, fmt='yaml', include_defaults=False, exclude=None):
        """Serialize back to the same nested shape as the input, with includes already flattened
        in (never round-tripped as 'includes'). `exclude` drops whole sections, e.g.
        exclude=['data', 'validation'] to publish a bare model config."""
        exclude = set(exclude or [])
        data = {name: getattr(self, name).asdict(include_defaults) for name in self.objects if name not in exclude}
        if fmt=='dict':
            dumped=data
        else:
            dumped = yaml.safe_dump(data, sort_keys=False) if fmt == 'yaml' else json.dumps(data, indent=2)
            if path:
                with open(path, 'w') as f:
                    f.write(dumped)
        return dumped
    def convert_and_import(self, definition, source: dict) -> "MatformerConfigs":
        if not isinstance(definition, dict):
            with open(definition) as f:
                definition = yaml.safe_load(f)

        built = defaultdict(dict)

        # 1. defaults
        for section, section_defaults in definition.get('defaults', {}).items():
            try:
                built[section].update(section_defaults)
            except Exception as e:
                print(f"Problem with interpretation of section {section} with defaults {section_defaults}")
                raise e
        # 2. mappings – now correctly interprets field_map as:
        #    {target_field: source_path}
        for key, field_map in definition.items():
            if not key.endswith('_mapping'):
                continue
            if key == 'weight_mapping':   # skip: not a config mapping
                continue
            section = key[:-len('_mapping')]
            for target_field, source_path in field_map.items():
                try:
                    value = _get_nested(source, source_path)
                except (KeyError, TypeError):
                    continue
                _set_nested(built[section], target_field, value)

        # 3. derived expressions
        for dotted, expr in definition.get('derived', {}).items():
            section, field = dotted.split('.', 1)
            value = eval(str(expr), {}, source)
            _set_nested(built[section], field, value)

        self.load_config(stdin=dict(built))
        return self
    def __repr__(self):
        return f"MatformerConfigs({', '.join(self.objects)})"

    @classmethod
    def create_configs(cls, configs_paths=None, overrides=None, stdin=None):
        """Factory method to instantiate, load, and optionally override configurations."""
        config_inst = cls()
        if configs_paths or stdin is not None:
            config_inst.load_config(configs_paths=configs_paths, stdin=stdin)
        if overrides:
            config_inst.set_override(overrides)
        return config_inst
    def merge_sections(self, sections: dict):
        """Shallow section-level override: each key in `sections` fully replaces (or adds) that
        top-level section. Mirrors the old dict.update()-style merge but on the object itself --
        e.g. lets a caller-provided partial config override specific sections of a
        checkpoint-loaded MatformerConfigs without going through the YAML/registry pipeline."""
        for name, value in sections.items():
            if name not in self.objects:
                self.objects.append(name)
            setattr(self, name, value)
        return self

    @classmethod
    def from_sections(cls, sections: dict) -> "MatformerConfigs":
        """Build a MatformerConfigs from already-instantiated {section_name: MatformerConfig}
        pairs, bypassing the registry/YAML pipeline entirely -- e.g. for legacy-checkpoint
        config reconstruction or the hf-loading path, where the sections are built by hand."""
        object = cls().merge_sections(sections)    
        object.resolve_references()
        return object  
