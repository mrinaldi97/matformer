"""Wizard definitions: TOML documents describing a guided, multi-step configuration.

A wizard is renderer independent. It describes steps, fields, conditions, presets and one action.
The server turns a filled wizard into RPC work; a GUI, a TUI or the documentation can render the
same document without knowing anything about training, MDAT or annotation.

The schema is small on purpose:

    [[steps]]
    id = "model"
    title = "..."
    when = { field = "mode", is = "scratch" }
    generator = "config"             # fields generated from registered dataclasses
    preset = "${mode}_${model_kind}"
    sections = ["model", "tokenizer"]

A field is a plain table: name, kind, label, doc, default, options, required, advanced, source,
depends_on, when. Generated config fields are dotted paths ("model.hidden_size"), so one answer
language builds the YAML document the action needs.

Actions are declarative too. `train` writes a config and queues it; `mdat` queues a sequence of
MDatCommands calls; `annotate` queues an annotation job. Values may reference answers with
`${answers.name}`.
"""
from __future__ import annotations

import dataclasses
import json
import re
import tomllib
from pathlib import Path

DEFINITIONS = Path(__file__).resolve().parent / "definitions"
_REF = re.compile(r"\$\{([^}]+)\}")
SECTIONS = ("model", "tokenizer", "masking", "data", "training", "validation", "task",
            "checkpoint", "saver", "weights")


def definitions() -> list:
    return sorted(DEFINITIONS.glob("*.toml"))


def load(name: str) -> dict:
    path = DEFINITIONS / f"{name}.toml"
    if not path.exists():
        raise ValueError(f"no wizard '{name}' (available: {', '.join(one.stem for one in definitions())})")
    wizard = tomllib.loads(path.read_text("utf-8"))
    wizard["id"] = wizard.get("id") or name
    wizard["steps"] = wizard.get("steps") or []
    wizard.setdefault("presets", {})
    return wizard


def catalog() -> list:
    rows = []
    for path in definitions():
        try:
            wizard = load(path.stem)
        except Exception:
            continue
        rows.append({"id": wizard["id"], "title": wizard.get("title", wizard["id"]),
                     "description": wizard.get("description", ""),
                     "category": wizard.get("category", "General"),
                     "steps": len(wizard["steps"]), "order": wizard.get("order", 100)})
    rows.sort(key=lambda row: (row["category"], row["order"], row["title"]))
    return rows


# ── conditions and substitution ──────────────────────────────────────────────────────────────
def _values(wizard, answers) -> dict:
    """Answers plus every static default, so a condition can read a choice before it is submitted."""
    values = {}
    for step in wizard["steps"]:
        for field in step.get("fields", []):
            values.setdefault(field["name"], field.get("default"))
    values.update({key: value for key, value in (answers or {}).items() if value is not None})
    return values


def _test(condition, values) -> bool:
    if not condition:
        return True
    if isinstance(condition, list):
        return all(_test(one, values) for one in condition)
    if "any" in condition:
        return any(_test(one, values) for one in condition["any"])
    if "all" in condition:
        return all(_test(one, values) for one in condition["all"])
    if "not" in condition:
        return not _test(condition["not"], values)
    field = condition.get("field")
    value = values.get(field)
    if "is" in condition:
        return value == condition["is"]
    if "equals" in condition:
        return value == condition["equals"]
    if "in" in condition:
        return value in condition["in"]
    if "not_in" in condition:
        return value not in condition["not_in"]
    if condition.get("truthy"):
        return bool(value)
    if "exists" in condition:
        return (value is not None) == bool(condition["exists"])
    return True


def substitute(value, answers):
    """Recursive ${answers.a.b} substitution. A string that is one reference keeps its type.
    Unknown references stay as written, so the caller can add its own namespaces (files.NAME)
    before the action is executed."""
    if isinstance(value, str):
        match = _REF.fullmatch(value.strip())
        if match:
            try:
                return _lookup(match.group(1), answers)
            except KeyError:
                return value
        return _REF.sub(lambda m: _safe_lookup(m.group(1), answers), value)
    if isinstance(value, list):
        return [substitute(one, answers) for one in value]
    if isinstance(value, dict):
        return {key: substitute(one, answers) for key, one in value.items()}
    return value


def _safe_lookup(path, answers):
    try:
        return str(_lookup(path, answers))
    except KeyError:
        return "${" + path + "}"


def _lookup(path, answers):
    if path.startswith("answers."):
        path = path[len("answers."):]
    node = answers or {}
    if path in node:
        return node[path]
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"missing answer '{path}'")
        node = node[part]
    return node


def _resolved(value, answers):
    result = substitute(value, answers)
    if isinstance(result, str) and "${" in result:
        return None
    return result


# ── generated config fields ──────────────────────────────────────────────────────────────────
def _preset(wizard, name) -> dict:
    presets = wizard.get("presets", {})
    if name not in presets:
        raise ValueError(f"wizard '{wizard['id']}': unknown preset '{name}'")
    preset = dict(presets[name])
    parent = preset.pop("extends", None)
    if parent:
        base = _preset(wizard, parent)
        for key, value in preset.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                base[key] = {**base[key], **value}
            elif isinstance(value, list) and isinstance(base.get(key), list):
                base[key] = list(dict.fromkeys(base[key] + value))
            else:
                base[key] = value
        preset = base
    return preset


def _flatten_values(mapping, prefix=""):
    out = {}
    for key, value in (mapping or {}).items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_flatten_values(value, path))
        else:
            out[path] = value
    return out


def _jsonable(value):
    if dataclasses.is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name))
                for field in dataclasses.fields(value)}
    if isinstance(value, tuple):
        return list(value)
    return value


def _config_fields(section, variant, preset, only=None):
    from matformer.matformer_registry import registry
    from matformer.utils import schema

    definition = registry.get_definition(section, variant)
    values = _flatten_values(preset.get("values", {}))
    focus = set(preset.get("focus", []))
    required = set(preset.get("required", []))
    skip = {"name", "type", "hooks", "custom_layers",
            "selector_config", "head_config", "missing_label_values"}
    entries = []

    def walk(cls, prefix, instance, docs, sections_of):
        if docs is None:
            docs = schema.field_docs(cls)
        if sections_of is None:
            sections_of = schema.field_sections(cls)
        for field in dataclasses.fields(cls):
            if field.name in skip or field.name.startswith("_"):
                continue
            path = f"{prefix}.{field.name}"
            default = schema.default_of(field)
            if field.name in ("default_layer", "sequence_mixer_config"):
                inner = getattr(instance, field.name, default) if instance is not None else default
                if inner is not None and dataclasses.is_dataclass(inner):
                    walk(type(inner), path, inner, None, None)
                continue
            if instance is not None:
                default = getattr(instance, field.name, default)
            if path in values:
                default = values[path]
            kind = schema.kind_of(field)
            if kind == "json":
                default = _jsonable(default)
            entry = {"name": path, "label": field.name.replace("_", " ").capitalize(),
                     "kind": kind, "default": default,
                     "doc": docs.get(field.name, ""),
                     "section": sections_of.get(field.name, prefix),
                     "advanced": not (path in focus or any(path.startswith(f + ".") for f in focus)),
                     "required": path in required}
            if kind == "choice":
                entry["options"] = schema.literal_options(field)
            entries.append(entry)

    instance = None
    try:
        instance = definition.from_dict({})
    except Exception:
        instance = None
    walk(definition, section, instance, schema.field_docs(definition), schema.field_sections(definition))
    if only:
        entries = [entry for entry in entries if entry["name"].split(".")[1] in only]
    return entries


def _variants(step, preset, answers):
    variants = dict(preset.get("variants", {}))
    for section, rule in (step.get("variant_by") or {}).items():
        reference = rule.get("field")
        if isinstance(reference, str) and not reference.startswith("${"):
            value = answers.get(reference)
        else:
            value = _resolved(reference, answers)
        table = rule.get("map", {})
        if isinstance(value, str) and value in table:
            variants[section] = table[value]
        elif value is not None:
            variants[section] = value
    return variants


def _generated_fields(step, wizard, answers):
    preset_name = _resolved(step.get("preset"), answers)
    if not preset_name:
        return []
    try:
        preset = _preset(wizard, preset_name)
    except ValueError:
        return []
    variants = _variants(step, preset, answers)
    sections = step.get("sections") or preset.get("sections") or []
    fields = []
    for section in sections:
        variant = variants.get(section)
        if variant is None:
            from matformer.matformer_registry import registry
            info = registry.get_registry_info(include_implementations=False).get(section, {})
            variant = info.get("default_variant")
        try:
            fields.extend(_config_fields(section, variant, preset))
        except Exception:
            continue
    return fields


# ── schema ──────────────────────────────────────────────────────────────────────────────────
def schema(wizard, answers=None, provider=None) -> dict:
    answers = dict(answers or {})
    values = _values(wizard, answers)
    steps = []
    for step in wizard["steps"]:
        if not _test(step.get("when"), values):
            continue
        fields = list(step.get("fields", []))
        if step.get("generator") == "config":
            fields = _generated_fields(step, wizard, values)
        for field in fields:
            if field.get("source") and provider is not None:
                try:
                    field["options"] = provider(field["source"], values, field)
                except Exception as problem:
                    field["options"] = []
                    field["source_error"] = str(problem)
        steps.append({"id": step["id"], "title": step.get("title", step["id"]),
                      "description": step.get("description", ""),
                      "fields": fields})
    return {"id": wizard["id"], "title": wizard.get("title", wizard["id"]),
            "description": wizard.get("description", ""),
            "category": wizard.get("category", "General"),
            "steps": steps, "values": values}


def validate(wizard, answers) -> list:
    values = _values(wizard, answers)
    errors = []
    for step in wizard["steps"]:
        if not _test(step.get("when"), values):
            continue
        fields = list(step.get("fields", []))
        if step.get("generator") == "config":
            fields = _generated_fields(step, wizard, values)
        for field in fields:
            if not _test(field.get("when"), values):
                continue
            if not field.get("required"):
                continue
            value = (answers or {}).get(field["name"], field.get("default"))
            if value is None or (isinstance(value, str) and not value.strip()):
                errors.append(f"{field.get('label', field['name'])} is required")
    return errors


# ── action building ─────────────────────────────────────────────────────────────────────────
def _set(doc, path, value):
    parts = path.split(".")
    node = doc
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def config_document(wizard, answers) -> dict:
    """The YAML document the answers describe, built from the same generated fields the GUI saw."""
    from matformer.matformer_registry import registry
    values = _values(wizard, answers)
    multi = set(registry.list_categories(multi_only=True))
    doc = {}
    data_name = values.get("data_name") or "train"
    variants = {}
    for step in wizard["steps"]:
        if step.get("generator") != "config" or not _test(step.get("when"), values):
            continue
        preset_name = _resolved(step.get("preset"), values)
        if not preset_name:
            continue
        preset = _preset(wizard, preset_name)
        variants.update(_variants(step, preset, values))
        for field in _generated_fields(step, wizard, values):
            if not _test(field.get("when"), values):
                continue
            value = values.get(field["name"], field.get("default"))
            if value is None:
                continue
            _set(doc, field["name"], value)
    for section, section_doc in list(doc.items()):
        if section in multi:
            entry = section_doc if isinstance(section_doc, dict) else {}
            entry["name"] = data_name if section == "data" else entry.get("name") or section
            doc[section] = [entry]
    for section, variant in variants.items():
        if section not in doc or variant is None:
            continue
        if section in multi:
            doc[section][0]["type"] = variant
        else:
            doc[section]["type"] = variant
    if values.get("model_variant") and "model" in doc:
        doc["model"]["type"] = values["model_variant"]

    # A test split repeats the data section with another root.
    test_root = values.get("test_data_root")
    if test_root:
        source = dict(doc.get("data", [{}])[0])
        source["name"] = "test"
        source["data_root"] = test_root
        if values.get("test_file_format"):
            source["file_format"] = values["test_file_format"]
        if values.get("test_split"):
            source["split"] = values["test_split"]
        doc["data"] = list(doc.get("data", [])) + [source]

    training = doc.get("training")
    data = doc.get("data") or []
    if isinstance(training, dict) and data:
        first = data[0] if isinstance(data, list) else data
        if training.get("batch_size") is None and first.get("batch_size"):
            training["batch_size"] = first["batch_size"]
    _add_validation(doc, values)
    return doc


def _unique(base, taken):
    name = base or "task"
    candidate, index = name, 2
    while candidate in taken:
        candidate = f"{name}_{index}"
        index += 1
    taken.add(candidate)
    return candidate


def _task_data_ref(doc, data, fallback):
    """A task may name an existing data entry or point at a file. A file becomes a lazy entry
    with the same module settings as the training split, and the task points at its new name."""
    data_entries = doc.get("data") or []
    names = {entry.get("name") for entry in data_entries if isinstance(entry, dict)}
    if not data:
        return fallback
    if data in names:
        return data
    if not data_entries:
        return fallback
    base = dict(data_entries[0])
    name = _unique(f"task_data_{len(data_entries)}", names)
    base["name"] = name
    base["data_root"] = data
    suffix = str(data).lower().rsplit(".", 1)[-1] if "." in str(data) else ""
    if suffix in ("json", "jsonl", "csv", "conll", "conllu"):
        base["type"] = "lazy"
        base["file_format"] = suffix
    elif suffix == "mdat" or str(data).endswith("/"):
        base["type"] = "lazy"
        base["file_format"] = "mdat"
    base.pop("mdat_dataset", None)
    doc["data"] = list(data_entries) + [base]
    return name


def _add_validation(doc, values):
    if not values.get("validation_enabled"):
        return
    rows = values.get("validation_tasks") or []
    if isinstance(rows, str):
        rows = [rows]
    rows = [row if isinstance(row, dict) else {"type": row} for row in rows]
    test_ref = "test" if values.get("test_data_root") else "train"
    learning_signal = None
    signals = ((doc.get("training") or {}).get("learning_signals") or [])
    if signals:
        learning_signal = signals[0]
    tasks, schedules, taken = [], [], set()
    for row in rows:
        kind = row.get("type") or row.get("kind")
        if not kind:
            continue
        name = _unique(row.get("name") or kind, taken)
        if kind == "evaluation":
            entry = {"name": name, "type": "evaluation",
                     "ref_data": _task_data_ref(doc, row.get("data"), test_ref),
                     "learning_signal": learning_signal}
            tasks.append(entry)
        elif kind in ("static_model_analysis", "activation_analysis"):
            entry = {"name": name, "type": kind}
            if kind == "activation_analysis":
                entry["ref_data"] = _task_data_ref(doc, row.get("data"), test_ref)
            tasks.append(entry)
        elif kind == "sandbox_train":
            tasks.append({"name": name, "type": "sandbox_train",
                          "sandbox_configs": list(values.get("validation_sandbox_configs") or [])})
    for task in tasks:
        schedules.append({"name": f"{task['name']}_schedule", "ref_task": task["name"],
                          **({"every_n_epochs": values["validation_every_n_epochs"]}
                             if values.get("validation_every_n_epochs") else {}),
                          **({"every_n_steps": values["validation_every_n_steps"]}
                             if values.get("validation_every_n_steps") else {})})
    if tasks:
        doc["task"] = tasks
        doc["validation"] = schedules


def _effective_values(wizard, answers) -> dict:
    values = _values(wizard, answers)
    try:
        doc = config_document(wizard, answers)
    except Exception:
        doc = {}
    values.update(_flatten_values(doc))
    return values


def _applicable(action, values):
    action = dict(action)
    for key in ("commands", "files"):
        if key in action:
            action[key] = [one for one in action[key] if _test(one.get("when"), values)]
    return action


def plan(wizard, answers) -> dict:
    values = _effective_values(wizard, answers) if (wizard.get("action") or {}).get("kind") == "train" else _values(wizard, answers)
    action = _applicable(substitute(wizard.get("action") or {}, values), values)
    result = {"id": wizard["id"], "title": wizard.get("title", wizard["id"]),
              "action": action.get("kind"), "lines": [], "errors": validate(wizard, answers)}
    if action.get("kind") == "train":
        doc = config_document(wizard, answers)
        import yaml
        result["config"] = doc
        result["yaml"] = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)
        result["name"] = action.get("name") or (doc.get("training") or {}).get("run_name") or "training"
        result["project"] = action.get("project") or (doc.get("training") or {}).get("project_name") or "wizard"
        result["lines"] = [f"Action: queue a training job",
                           f"Run: {result['name']}", f"Project: {result['project']}",
                           f"Config: {len(result['yaml'].splitlines())} lines generated"]
    elif action.get("kind") == "mdat":
        result["container"] = action.get("container")
        result["commands"] = [one for one in action.get("commands", []) if one.get("cmd")]
        result["files"] = action.get("files", [])
        result["lines"] = [f"Action: queue an MDAT job",
                           f"Container: {result['container']}",
                           f"Operations: {', '.join(one['cmd'] for one in result['commands'])}"]
    elif action.get("kind") == "annotate":
        result["lines"] = [f"Action: queue an annotation job",
                           f"Input: {action.get('input')}",
                           f"Output: {action.get('output') or '(automatic)'}",
                           f"Task: {action.get('task') or 'evaluation'}"]
    else:
        result["lines"] = ["No action declared by this wizard"]
    return result


def action(wizard, answers) -> dict:
    values = _effective_values(wizard, answers) if (wizard.get("action") or {}).get("kind") == "train" else _values(wizard, answers)
    return _applicable(substitute(wizard.get("action") or {}, values), values)


def to_json(value):
    return json.dumps(value, default=str)
