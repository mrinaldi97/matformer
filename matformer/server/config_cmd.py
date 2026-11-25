# matformer/server/config_cmd.py
"""Structured config access, built on the framework itself:

  • config.read   parses through MatformerConfigs (includes resolved, references linked, defaults
                  applied exactly as a real run would see them) and pairs every resulting
                  dataclass with its field schema (kinds, docs, Literal options, path detection).
                  A blank file still yields the full skeleton of default sections: the wizard
                  starts from what Matformer *can* be configured, not from an empty page.
  • config.save   minimal document surgery on the file itself: override strings are parsed by
                  the framework's yaml literals and set as nested keys. Includes are preserved:
                  editing a child config keeps the base linked, only what you touched is written.
  • config.new /  wizard primitives: blank or inherit-from-base ({includes: [base]}).
  • config.sections / config.add_entry   live registry view for the Add-entry picker.
No raw config parsing is duplicated anywhere: understanding configs stays MatformerConfigs' job."""
from pathlib import Path

from matformer.server.core import rpc
from matformer.server.fs import resolve
from matformer.utils import matformer_assets

MULTI_SECTIONS = ("data", "task")


@rpc("config.read")
def config_read(conn, p):
    """{path} -> {path, sections: [{section, name, variant, fields:[{name,label,kind,default,
    doc,options?}]}]}. Resolved via MatformerConfigs; default skeleton when the file is empty."""
    from matformer.matformer_configs import MatformerConfigs
    from matformer.utils.schema import fields_schema
    path = str(resolve(p["path"]))
    sections = []
    try:
        configs = MatformerConfigs.create_configs([path])
        for section, name, inst in _entries(configs):
            try:
                fields = fields_schema(type(inst), skip=("name",))
            except Exception:
                continue
            for f in fields:
                value = getattr(inst, f["name"], None)
                if f["kind"] == "json" and value is not None:
                    import json
                    value = json.dumps(value, default=str)
                if f["name"] == "presentation":
                    f["kind"], f["options"] = "radio", list(_PRESENTATIONS)
                f["default"] = value
            sections.append({"section": section, "name": name, "variant": type(inst).__name__,
                             "fields": fields})
    except Exception:
        sections = []
    if not sections:
        sections = _default_skeleton()
    return {"path": path, "sections": sections}


# The task page and the config editor render the same choice with the same widget; the list lives
# with the tasks, not with either renderer.
_PRESENTATIONS = ("probs", "argmax", "full")


def _entries(configs):
    """(section, entry_name_or_None, dataclass) for every config instance of a loaded file."""
    for section in list(configs.objects):
        obj = getattr(configs, section)
        if hasattr(obj, "objects") and hasattr(obj, "items"):
            for name, child in obj.items():
                if hasattr(child, "__dataclass_fields__"):
                    yield section, name, child
        elif hasattr(obj, "__dataclass_fields__"):
            yield section, None, obj


def _default_skeleton():
    """Every config dataclass the registry knows (single-entry categories), with class defaults:
    what a brand-new config can declare. Multi-entry sections come next via Add entry."""
    import dataclasses
    from matformer.matformer_registry import registry
    from matformer.utils.schema import fields_schema
    out = []
    info = registry.get_registry_info(include_implementations=False)
    for category, cat in sorted(info.items()):
        if category in MULTI_SECTIONS:
            continue
        try:
            definition = registry.get_definition(category, cat.get("default_variant"))
        except Exception:
            continue
        if not (isinstance(definition, type) and dataclasses.is_dataclass(definition)):
            continue
        try:
            out.append({"section": category, "name": None, "variant": definition.__name__,
                        "fields": fields_schema(definition, skip=("name",))})
        except Exception:
            continue
    return out


@rpc("config.save")
def config_save(conn, p):
    """Apply override strings to the file's own document (includes stay linked):
    {path, overrides: ["section[.entry].field=value"]}. value parsed as yaml literal."""
    import yaml
    doc = _raw_doc(p["path"])
    for o in p.get("overrides") or []:
        _set_in_doc(doc, o)
    Path(str(resolve(p.get("target") or p["path"]))).write_text(
        yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), "utf-8")
    return {"msg": f"saved {p.get('target') or p['path']}"}


def _raw_doc(path):
    import yaml
    file = resolve(path)
    return yaml.safe_load(file.read_text("utf-8")) if file.exists() else {}


def _set_in_doc(doc, override):
    import yaml
    path, _, value = override.partition("=")
    parts = path.split(".")
    # a path inside the assets folder is stored back as its token: the file stays relocatable
    value = matformer_assets.shrink(yaml.safe_load(value))
    section = parts[0]
    rest = parts[1:]
    if section in MULTI_SECTIONS:
        entry, field = (rest + [None])[:2]
        node = doc.setdefault(section, [])
        match = next((e for e in node if isinstance(e, dict) and e.get("name") == entry), None)
        if match is None:
            match = {"name": entry or "default"}
            node.append(match)
        if field:
            match[field] = value
    else:
        node = doc.setdefault(section, {})
        for p in rest[:-1]:
            node = node.setdefault(p, {})
        if rest:
            node[rest[-1]] = value


@rpc("config.home")
def config_home(conn, p):
    """Where a new config belongs: the assets configs folder when one is configured, the repo's
    configs/ otherwise. The wizard opens there instead of asking the user to type a path."""
    from matformer.server import REPO
    from matformer.utils import matformer_assets
    return {"path": str(matformer_assets.area("configs") or (REPO / "configs"))}


@rpc("config.new")
def config_new(conn, p):
    """Wizard: create a config. `{path, sections: ["model", "training", …]}` writes those sections
    with their registry defaults spelled out; `base` instead of (or with) sections inherits, keeping
    the link with {includes: [base]} so only what changes is ever written on top."""
    import yaml
    target = str(resolve(p["path"]))
    doc = {"includes": [matformer_assets.shrink(str(resolve(p["base"])))]} if p.get("base") else {}
    for section in p.get("sections") or []:
        doc[section] = _section_defaults(section)
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    Path(target).write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), "utf-8")
    return {"msg": f"created {target}", "path": target}


def _section_defaults(section):
    """A section as a plain mapping of its declared defaults, None values aside: a new config
    starts with the names that exist, not with an empty block."""
    import dataclasses
    from matformer.matformer_registry import registry
    from matformer.utils.schema import default_of

    try:
        info = registry.get_registry_info(include_implementations=False)[section]
        definition = registry.get_definition(section, info.get("default_variant"))
    except Exception:
        return {}
    if not (isinstance(definition, type) and dataclasses.is_dataclass(definition)):
        return {}
    def clean(value):
        if dataclasses.is_dataclass(value):
            return dataclasses.asdict(value)
        if isinstance(value, (list, tuple)):
            return [clean(one) for one in value]
        if isinstance(value, dict):
            return {key: clean(one) for key, one in value.items()}
        return value

    return {f.name: clean(default_of(f)) for f in dataclasses.fields(definition)
            if not f.name.startswith("_") and default_of(f) is not None}


@rpc("config.write")
def config_write(conn, p):
    """Save a whole document as written by hand: parsed as YAML first, so a broken file is refused
    with the parser's own message instead of landing on disk."""
    import yaml
    target = str(resolve(p["path"]))
    try:
        yaml.safe_load(p["text"])
    except yaml.YAMLError as problem:
        raise ValueError(f"not valid YAML: {problem}")
    Path(target).write_text(p["text"], "utf-8")
    return {"msg": f"written {target}"}


@rpc("config.derive")
def config_derive(conn, p):
    """'Duplicate to inherit': the child keeps {includes: [source]} + only the overrides."""
    doc = {"includes": [matformer_assets.shrink(str(resolve(p["source"])))]}
    for o in p.get("overrides") or []:
        _set_in_doc(doc, o)
    import yaml
    target = str(resolve(p["target"]))
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    Path(target).write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), "utf-8")
    return {"msg": f"derived {target} from {p['source']}"}


@rpc("config.sections")
def config_sections(conn, p):
    """Sections and variants for the Add-entry picker, straight from the live registry."""
    from matformer.matformer_registry import registry
    data = registry.get_registry_info(include_implementations=False)
    return [{"category": category, "default": cat.get("default_variant"),
             "variants": sorted(cat["variants"])} for category, cat in sorted(data.items())]


@rpc("config.add_entry")
def config_add_entry(conn, p):
    """Declare a new entry (a second task, a dataset split...) in a listed section."""
    doc = _raw_doc(p["path"])
    node = doc.setdefault(p["section"], [])
    entry = {"name": p["name"]}
    if p.get("variant"):
        entry["type"] = p["variant"]
    if not any(isinstance(e, dict) and e.get("name") == p["name"] for e in node):
        node.append(entry)
    import yaml
    Path(resolve(p["path"])).write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), "utf-8")
    return {"msg": f"added {p['section']}.{p['name']}"}


@rpc("config.fields_catalog")
def config_fields_catalog(conn, p):
    """Every registered config dataclass flattened to a field catalog:
    [{path: "training.lr", section, variant, kind, default, doc, options?}] — the GUI builds its
    override pickers (and any future parameter explorer) on top of this single command."""
    from matformer.matformer_registry import registry
    from matformer.utils.schema import fields_schema
    out = []
    info = registry.get_registry_info(include_implementations=False)
    for category, cat in sorted(info.items()):
        for variant in sorted(cat["variants"]):
            try:
                definition = registry.get_definition(category, variant)
            except Exception:
                continue
            if not isinstance(definition, type):
                continue
            for f in fields_schema(definition, skip=("name",)):
                out.append({"path": f"{category}.{f['name']}", "section": category,
                            "variant": variant, "kind": f["kind"], "default": f["default"],
                            "doc": f["doc"], "options": f.get("options", [])})
    return out


@rpc("config.delete_section")
def config_delete_section(conn, p):
    """Drop a whole section (or one named entry of a listed section) from the file."""
    import yaml
    doc = _raw_doc(p["path"])
    section = p["section"]
    if p.get("name") and isinstance(doc.get(section), list):
        doc[section] = [e for e in doc[section]
                        if not (isinstance(e, dict) and e.get("name") == p["name"])]
        if not doc[section]:
            doc.pop(section)
    else:
        doc.pop(section, None)
    Path(str(resolve(p["path"]))).write_text(yaml.safe_dump(doc, sort_keys=False,
                                                            allow_unicode=True), "utf-8")
    return {"msg": f"removed {section}" + (f".{p['name']}" if p.get("name") else "")}
