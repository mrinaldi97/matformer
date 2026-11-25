# matformer/server/wizard_cmd.py
"""Wizard RPC: definitions are TOML files, execution is the framework's own job queue.

The GUI (or any other renderer) asks for a schema, sends answers, sees a summary and submits.
Nothing about training, MDAT or annotation is reimplemented here: `train` writes the config the
wizard describes and calls the same `train.start` path, `mdat` queues the same MDatCommands facade,
`annotate` queues a job that loads its own model. A wizard is a description, not a second engine.
"""
from __future__ import annotations

import json
import re
import tempfile
from datetime import datetime
from pathlib import Path

from matformer import wizards
from matformer.server.core import rpc


# ── runtime options for fields that depend on a container ───────────────────────────────────
def _provider(source, answers, field):
    if source.startswith("mdat."):
        container = answers.get("container") or answers.get("path")
        if not container:
            return []
        try:
            from matformer.data.mdat.container import MDatContainer
            ds = MDatContainer.load_dataset(str(container), readonly=True)
            try:
                if source == "mdat.datasets":
                    return list(ds.list_submdat())
                if source == "mdat.strategies":
                    return list(ds.list_strategies())
                if source == "mdat.views":
                    return list(ds.list_views())
            finally:
                try:
                    ds.close()
                except Exception:
                    pass
        except Exception:
            return []
    return []


def _schema(wizard, answers):
    return wizards.schema(wizard, answers, provider=_provider)


@rpc("wizard.list")
def wizard_list(conn, p):
    """Every wizard definition: id, title, description, category and step count."""
    return wizards.catalog()


@rpc("wizard.context")
def wizard_context(conn, p):
    """What a wizard may prefill: the model currently loaded on this server, when there is one."""
    from matformer.server import tasks_host
    session = tasks_host.STATE.get("session")
    source = tasks_host.STATE.get("source")
    return {"model": source, "loaded": session is not None,
            "task": session.task_name if session else None,
            "tasks": list(session.configs.task.list_configs()) if session else []}


@rpc("wizard.schema")
def wizard_schema(conn, p):
    """The visible steps and their fields for the answers so far. Generated config fields are
    expanded from their registered dataclasses; a field with `source` is filled from live data."""
    wizard = wizards.load(p["id"])
    return _schema(wizard, p.get("answers") or {})


@rpc("wizard.plan")
def wizard_plan(conn, p):
    """Validate the answers and describe the work: the generated YAML for a training, the command
    list for MDAT, the annotation target. Nothing is queued and no model is built."""
    wizard = wizards.load(p["id"])
    answers = p.get("answers") or {}
    result = wizards.plan(wizard, answers)
    result["errors"] = wizards.validate(wizard, answers)
    return result


@rpc("wizard.submit")
def wizard_submit(conn, p):
    """Execute the wizard: queue a training, an MDAT job or an annotation job. Returns the job id
    and a short description of what was written."""
    wizard = wizards.load(p["id"])
    answers = p.get("answers") or {}
    errors = wizards.validate(wizard, answers)
    if errors:
        raise ValueError("; ".join(errors))
    action = wizards.action(wizard, answers)
    _check_resolved(action)
    kind = action.get("kind")
    if kind == "train":
        return _submit_train(wizard, answers, action)
    if kind == "mdat":
        return _submit_mdat(action)
    if kind == "annotate":
        return _submit_annotate(action)
    if kind == "collect":
        return _submit_collect(wizard, answers, action)
    raise ValueError(f"unknown wizard action '{kind}'")


def _check_resolved(value, path="action"):
    """A placeholder left after substitution means a required answer is missing. Say which one
    instead of handing `${answers.x}` to a job."""
    if isinstance(value, str):
        match = re.search(r"\$\{([^}]+)\}", value)
        if match and not match.group(1).startswith("files."):
            raise ValueError(f"unresolved '{match.group(1)}' in {path}")
    elif isinstance(value, list):
        for i, one in enumerate(value):
            _check_resolved(one, f"{path}[{i}]")
    elif isinstance(value, dict):
        for key, one in value.items():
            _check_resolved(one, f"{path}.{key}")


def _submit_train(wizard, answers, action):
    import yaml
    from matformer.server import REPO
    from matformer.server.jobs import train_start
    from matformer.utils import matformer_assets

    plan = wizards.plan(wizard, answers)
    folder = matformer_assets.area("configs") or (REPO / "configs")
    folder = Path(folder) / "wizards"
    folder.mkdir(parents=True, exist_ok=True)
    name = str(action.get("name") or plan.get("name") or "training")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-") or "training"
    path = folder / f"{safe}-{datetime.now():%Y%m%d-%H%M%S}.yaml"
    path.write_text(yaml.safe_dump(plan["config"], sort_keys=False, allow_unicode=True), "utf-8")

    options = {key: value for key, value in (action.get("options") or {}).items()
               if value not in (None, "")}
    if options.get("device") == "auto":
        options.pop("device", None)
    params = {"config": [str(path)], "name": name,
              "project": action.get("project") or "wizard", "options": {}}
    params.update({key: options.pop(key) for key in ("device", "gpus", "save_model",
                                                     "save_plugins") if key in options})
    queued = train_start(None, params)
    return {"kind": "train", "run": queued["run"], "config": str(path), "name": name,
            "project": params["project"]}


def _submit_mdat(action):
    from matformer.server.jobs import enqueue

    files = _materialize_files(action)
    action = wizards.substitute(action, {"files": files}) if files else action
    _check_resolved(action)
    commands = []
    for one in action.get("commands") or []:
        args = {key: value for key, value in (one.get("args") or {}).items()
                if value not in (None, "")}
        commands.append({"cmd": one["cmd"], "args": args})
    if not commands:
        raise ValueError("this wizard produced no MDAT operation")
    container = action.get("container")
    job = enqueue("mdat", action.get("name") or commands[0]["cmd"], "mdat",
                  {"container": container, "readonly": False,
                   "open": bool(action.get("open", True)), "commands": commands},
                  "matformer.data.mdat.tui:execute")
    return {"kind": "mdat", "run": job, "container": container,
            "files": {key: str(value) for key, value in files.items()}}


def _materialize_files(action):
    """Files a wizard declares (a pretokenization strategy, for example) are written to a private
    folder and addressed as ${files.NAME} in the command list."""
    files = {}
    for entry in action.get("files") or []:
        folder = Path(tempfile.mkdtemp(prefix="matformer-wizard-"))
        suffix = entry.get("suffix", "json")
        path = Path(entry["path"]) if entry.get("path") else folder / f"{entry['id']}.{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        content = entry.get("content")
        if suffix == "json" or isinstance(content, (dict, list)):
            path.write_text(json.dumps(content, indent=2, default=str), "utf-8")
        else:
            path.write_text(str(content), "utf-8")
        files[entry["id"]] = str(path)
    return files


def _submit_collect(wizard, answers, action):
    """Write the data config the wizard describes and queue one forward that fills the store."""
    import yaml
    from pathlib import Path
    from matformer.server import REPO
    from matformer.server.task_jobs import enqueue_task
    from matformer.utils import matformer_assets

    document = wizards.config_document(wizard, answers)
    addresses = action.get("addresses") or ""
    addresses = [one.strip() for one in str(addresses).split(",") if one.strip()]
    store = dict(action.get("store") or {})
    backend = store.get("backend") or "lmdb"
    document["task"] = [{"type": "forward", "name": "forward", "ref_data": "train"}]
    document["plugin"] = [
        {"type": "activation_cache", "name": "collect_cache", "targets": addresses,
         "move_to_cpu": bool(action.get("move_to_cpu", False))},
        {"type": f"activation_store_{backend}", "name": "collect_store",
         "filename": store.get("filename"), "databank": store.get("databank") or "activations",
         "targets": action.get("store_socket") or (addresses[0] if addresses else "*"),
         "only_first": True, "readonly": False, "flush_every": 8, "preload": False},
    ]
    folder = matformer_assets.area("configs") or (REPO / "configs")
    folder = Path(folder) / "wizards"
    folder.mkdir(parents=True, exist_ok=True)
    name = str(action.get("name") or "collect")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-") or "collect"
    path = folder / f"{safe}-{datetime.now():%Y%m%d-%H%M%S}.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True), "utf-8")

    extra = action.get("cfg") or []
    extra = [extra] if isinstance(extra, str) else list(extra)
    vectorize = dict(action.get("vectorize") or {})
    if "enabled" in vectorize:
        vectorize["enabled"] = bool(vectorize["enabled"]) and str(vectorize["enabled"]).lower()             not in ("false", "0", "none", "")
    positions = vectorize.get("positions")
    if isinstance(positions, str) and positions.strip():
        try:
            vectorize["positions"] = json.loads(positions)
        except json.JSONDecodeError:
            vectorize["positions"] = [int(one) for one in positions.replace(" ", "").split(",")
                                      if one]
    if not store.get("filename"):
        raise ValueError("collect needs the activation store filename")
    spec = {"mode": "run", "cfg": [str(path), *extra], "model": action.get("model") or None,
            "device": action.get("device") or None, "task": "forward",
            "label_key": action.get("label_key") or "label", "vectorize": vectorize}
    job = enqueue_task(spec, name=name, project="collect")
    return {"kind": "collect", "run": job, "config": str(path), "store": store,
            "vectorize": vectorize}


def _submit_annotate(action):
    from matformer.server.task_jobs import enqueue_task

    spec = {key: value for key, value in action.items()
            if key in ("model", "cfg", "device", "task", "input", "output", "format",
                       "keys_only", "limit", "batch_size", "dataset", "column", "label",
                       "max_new_tokens", "temperature") and value not in (None, "", 0)}
    spec.setdefault("format", "jsonl")
    spec.setdefault("mode", "annotate")
    name = Path(str(action.get("input") or "annotation")).stem or "annotation"
    job = enqueue_task(spec, name=name, project="annotate")
    return {"kind": "annotate", "run": job, "spec": spec}
