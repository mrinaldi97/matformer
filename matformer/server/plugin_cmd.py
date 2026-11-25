# matformer/server/plugin_cmd.py
"""Plugins and weight-space interventions over RPC.

Everything here is the same machinery the training code already uses: PluginManager for mounting
a plugin on a socket, ParameterEditor for editing, freezing, ablating, projecting out and undoing
one stable parameter. A UI (or a script) can therefore mount a steering kernel, ablate a
direction, inspect the result and restore it, without a new code path."""
import fnmatch

import torch

from matformer.server.core import rpc


def _model():
    from matformer.server import tasks_host
    session = tasks_host.session()
    return session.model


def _editor():
    model = _model()
    if not hasattr(model, "edit"):
        raise RuntimeError("the loaded model does not expose the parameter editor")
    return model


def _named(model):
    from matformer.tasks.analysis import _named_parameters
    return _named_parameters(model)


def _kind_fields(kind):
    from matformer.matformer_registry import registry
    definition = registry.get_definition("plugin", kind)
    if definition is None:
        return {}
    try:
        out = definition.from_dict({}).asdict()
    except Exception:
        return {}
    for internal in ("name", "targets", "type"):
        out.pop(internal, None)
    return out


@rpc("plugin.kinds")
def plugin_kinds(conn, p):
    """Every plugin the registry knows, with the defaults of its config."""
    from matformer.matformer_registry import registry
    info = registry.get_registry_info(category="plugin")
    rows = []
    for kind in info.get("plugin", {}).get("variants", {}):
        rows.append({"kind": kind, "defaults": _kind_fields(kind)})
    return {"plugins": rows}


@rpc("plugin.sockets")
def plugin_sockets(conn, p):
    """The sockets of the loaded model: address, what is mounted, who owns the module."""
    model = _model()
    pattern = p.get("pattern") or "*"
    found = model.find_socket(pattern)
    rows = [{"address": address, "mounted": socket.inspect(),
             "count": len(socket.components),
             "owner": type(socket.owner).__name__}
            for address, socket in sorted(found.items())]
    return {"sockets": rows, "pattern": pattern}


@rpc("plugin.list")
def plugin_list(conn, p):
    """Mounted plugins: id, socket, kind, class, config."""
    model = _model()
    rows = [{"id": plugin.id, "socket": getattr(getattr(plugin, "socket", None), "address", ""),
             "kind": getattr(plugin, "kind", ""), "class": type(plugin).__name__,
             "name": getattr(plugin, "name", ""), "origin": model.plugin_origin(plugin)}
            for plugin in model.plugins.values()]
    return {"plugins": rows}


@rpc("plugin.attach")
def plugin_attach(conn, p):
    """Mount a plugin: {socket: pattern, kind, config: {...}?, name?}. A pattern that matches more
    than one socket mounts one instance per socket, exactly like a config file would."""
    from matformer.matformer_registry import registry
    model, kind = _model(), p.get("kind")
    if not kind:
        raise RuntimeError("plugin.attach needs the plugin kind")
    definition = registry.get_definition("plugin", kind)
    if definition is None:
        raise RuntimeError(f"unknown plugin '{kind}'")
    config = p.get("config") or {}
    cfg = definition(**config) if isinstance(definition, type) else definition.from_dict(config)
    mounted = model.attach(p.get("socket") or "*", cfg, name=p.get("name") or kind)
    return {"mounted": [{"id": one.id, "socket": getattr(one.socket, "address", ""),
                         "kind": getattr(one, "kind", ""), "class": type(one).__name__}
                        for one in mounted]}


@rpc("plugin.detach")
def plugin_detach(conn, p):
    """Unmount one plugin by id (see plugin.list)."""
    model = _model()
    plugin = model.plugins.get(p.get("id"))
    if plugin is None:
        raise RuntimeError(f"no mounted plugin '{p.get('id')}'")
    socket, name = plugin.socket, plugin.name
    model.detach(socket.address, name)
    return {"msg": f"{p['id']} unmounted"}


@rpc("param.list")
def param_list(conn, p):
    """Stable weights of the loaded model: shape, dtype, norm, grad and whether an edit is live.
    `pattern` is a glob on the stable name; `limit` bounds the rows."""
    model = _editor()
    pattern = p.get("pattern") or "*"
    limit = int(p.get("limit") or 500)
    rows = []
    for name, parameter in _named(model):
        if not fnmatch.fnmatchcase(name, pattern):
            continue
        with torch.no_grad():
            norm = float(parameter.detach().float().norm())
        rows.append({"name": name, "shape": list(parameter.shape),
                     "dtype": str(parameter.dtype),
                     "requires_grad": bool(parameter.requires_grad),
                     "norm": norm,
                     "edited": float(model.delta(name)) if name in model._store() else 0.0})
        if len(rows) >= limit:
            break
    return {"parameters": rows, "pattern": pattern, "total": sum(
        1 for name, _ in _named(model) if fnmatch.fnmatchcase(name, pattern))}


@rpc("param.stats")
def param_stats(conn, p):
    """Full tensor statistics of one stable parameter (the analysis task's own numbers)."""
    model = _editor()
    return model.stats(p["name"])


@rpc("param.edit")
def param_edit(conn, p):
    """Overwrite a parameter (a scalar broadcasts everywhere), optionally freezing it afterwards."""
    model = _editor()
    value = p.get("value", 0)
    if isinstance(value, list):
        value = torch.tensor(value)
    model.edit(p["name"], value=value, freeze_grad=p.get("freeze_grad", True))
    return {"msg": f"{p['name']} edited", "delta": model.delta(p["name"])}


@rpc("param.freeze")
def param_freeze(conn, p):
    """Zero the gradient of a parameter (values untouched), or of all but a mask."""
    model = _editor()
    train = p.get("train")
    model.freeze(p["name"], train=None if train is None else torch.tensor(train))
    return {"msg": f"{p['name']} frozen"}


@rpc("param.project_out")
def param_project_out(conn, p):
    """Remove a direction from a weight matrix's input or output space (ablation), and keep it out
    of the gradients. `direction` is a list (or list of lists for a subspace)."""
    model = _editor()
    direction = torch.tensor(p["direction"])
    model.project_out(p["name"], direction, side=p.get("side", "in"),
                      freeze_grad=p.get("freeze_grad", True))
    return {"msg": f"direction removed from {p['name']}",
            "leak": model.leak(p["name"], direction, side=p.get("side", "in")),
            "delta": model.delta(p["name"])}


@rpc("param.directions")
def param_directions(conn, p):
    """Top-k singular vectors of a parameter, input or output space: the natural ablation axes."""
    model = _editor()
    vectors = model.directions(p["name"], k=int(p.get("k") or 1), side=p.get("side", "in"))
    return {"directions": vectors.tolist()}


@rpc("param.leak")
def param_leak(conn, p):
    """How much of the parameter's energy still lies along a direction (0 after a clean ablation)."""
    model = _editor()
    return {"leak": model.leak(p["name"], torch.tensor(p["direction"]),
                               side=p.get("side", "in"))}


@rpc("param.delta")
def param_delta(conn, p):
    """Relative change of one parameter compared to its backup: 0 when it was never edited."""
    model = _editor()
    return {"name": p["name"], "delta": model.delta(p["name"])}


@rpc("param.restore")
def param_restore(conn, p):
    """Undo edits and gradient hooks for one parameter, or for all of them."""
    model = _editor()
    name = p.get("name")
    model.restore(name)
    return {"msg": f"restored {name or 'every edited parameter'}"}


# ── steering: a named direction from a store, mounted on a socket ───────────────────────────────
@rpc("steer.apply")
def steer_apply(conn, p):
    """Mount the `steering` plugin from a stored vector.
    {vector: {filename, backend?, databank?, name}, socket?, alpha?, positions?}."""
    from matformer import vectors as V
    model = _model()
    spec = dict(p.get("vector") or {})
    physical = V.store({**spec, "readonly": True})
    record = V.find(physical, spec.get("name"), spec.get("kind"))
    if record is None:
        raise RuntimeError(f"no vector '{spec.get('name')}' in {physical.databank}")
    alpha = p.get("alpha", record.get("alpha", 1.0))
    socket = p.get("socket") or record.get("address") or "*"
    # The plugin loads the direction itself from the store: no need to ship thousands of floats.
    config = {"vector_store": spec.get("filename") or spec.get("path"),
              "vector_backend": spec.get("backend") or "lmdb",
              "vector_databank": spec.get("databank") or "vectors",
              "vector_name": record["name"],
              "alpha": float(alpha), "positions": p.get("positions") or "all"}
    meta = V.as_public(record)
    from matformer.matformer_registry import registry
    definition = registry.get_definition("plugin", "steering")
    cfg = definition(**config) if isinstance(definition, type) else definition.from_dict(config)
    mounted = model.attach(socket, cfg, name="steering")
    return {"vector": meta, "socket": socket, "alpha": alpha,
            "mounted": [{"id": one.id, "socket": getattr(one.socket, "address", "")}
                        for one in mounted]}


@rpc("steer.clear")
def steer_clear(conn, p):
    """Unmount every steering plugin, leaving the weights and the other plugins untouched."""
    model = _model()
    removed = []
    for plugin in list(model.plugins.values()):
        if getattr(plugin, "kind", "") == "steering":
            model.detach(plugin.socket.address, plugin.name)
            removed.append(plugin.id)
    return {"removed": removed}
