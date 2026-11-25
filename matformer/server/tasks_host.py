# matformer/server/tasks_host.py
"""Task execution host: the server owns exactly one TaskSession (one model on the GPU), and every
command here is a thin mapping onto the machinery run_task.py already drives — TaskSession, its
tasks and their configs. Nothing is reimplemented: the GUI form for a task is *generated from the
task's own dataclass* (fields, defaults, docstrings), so a new task type is editable from the GUI
the moment it is registered, without touching the GUI or this file.

Streaming: the token callback plugs into TaskSession.feed(streamer=...), the same path the console
uses, and pushes `token` events to the GUI. `task.stop` interrupts at the next token."""
import dataclasses
import json
import os
import threading

from matformer.server.core import broadcast, rpc
from matformer import sweeps

LOCK = threading.Lock()
CANCEL = threading.Event()
STATE = {"session": None, "configs": None, "source": None}
LAST = {"rows": None, "task": None}


class TaskStopped(Exception):
    """Raised inside the streamer to unwind a running generation."""


def session():
    if STATE["session"] is None:
        raise RuntimeError("No model loaded: use task.load first.")
    return STATE["session"]


SKIP = {"name", "type", "data", "batch_preprocessors", "batch_postprocessors"}


@rpc("task.types")
def task_types(conn, p):
    """The task types registered in the framework (generation, evaluation, masking, ...)."""
    from matformer.tasks.session import TaskSession
    return TaskSession.task_types()


def _release():
    """Drop the loaded session and give its GPU memory back. Caller holds LOCK."""
    import gc
    session = STATE["session"]
    if session is not None:
        model = getattr(session, "model", None)
        for plugin in list((getattr(model, "plugins", {}) or {}).values()):
            close = getattr(plugin, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        store = getattr(model, "activation_store", None)
        close = getattr(store, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
    STATE["session"], STATE["configs"], STATE["source"] = None, None, None
    gc.collect()
    if "torch" in __import__("sys").modules:
        __import__("torch").cuda.empty_cache()


@rpc("task.load")
def task_load(conn, p):
    """Build the session: {model: path|hf://id|None, cfg: [YAML], overrides: [...],
    device, attention, tokenizer}. Heavy: takes seconds, runs in the worker thread. Loading over
    an existing session releases it first, so a reload cannot silently stack two models on the GPU."""
    import torch
    from matformer.matformer_configs import MatformerConfigs
    from matformer.matformer_wrapper import load_model
    from matformer.tasks.session import TaskSession
    if p.get("attention"):
        os.environ["MATFORMER_ATTENTION_NORMAL"] = p["attention"]
    configs = MatformerConfigs.create_configs(p.get("cfg") or [])
    if p.get("tokenizer"):
        configs.import_config({"tokenizer": {"pretrained_name": p["tokenizer"]}})
    if p.get("virtual"):
        # A virtual model replays a cached activation store instead of running the backbone:
        # the store's stamped model config builds the sockets, the plugins see the same addresses.
        virtual = dict(p["virtual"])
        backend = str(virtual.pop("backend", "lmdb")).lower()
        check_input = bool(virtual.pop("check_input", False))
        configs.import_config({
            "training": {"virtual_only": True},
            "virtual_model": {"check_input": check_input,
                              "input_address": virtual.pop("input_address", "encoder.model_input")},
            "activation_store": {"type": backend, "readonly": True, **virtual}})
        p = {**p, "model": None}
    device = torch.device(p.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    with LOCK:
        if STATE["session"] is not None:
            _release()
        loaded = load_model(path=p.get("model"), map_location=device,
                            dtype=torch.bfloat16, inference=True, configs=configs)
        STATE["session"] = TaskSession(loaded["model"], loaded["configs"], task=p.get("task"),
                                       overrides=p.get("overrides"), source=p.get("model"),
                                       presentation=p.get("presentation"))
        STATE["configs"], STATE["source"] = loaded["configs"], p.get("model")
    broadcast("task_state", _state_dict())
    return task_state(conn, p)


@rpc("task.state")
def task_state(conn, p):
    """What is loaded and which task is active, plus the configured task names."""
    return _state_dict()


def _state_dict():
    s = STATE["session"]
    template = getattr(s.model, "chat_template", None) if s else None
    return {"loaded": s is not None, "source": STATE["source"],
            "task": s.task_name if s else None,
            "tasks": list(s.configs.task.list_configs()) if s else [],
            "sweep": s.configs.active_sweep.name if s and s.configs.active_sweep else None,
            "sweeps": len(STATE["configs"].sweep_points()) if STATE["configs"] is not None else 0,
            "device": str(s.model.device) if s else None,
            # Chat template at a glance: whether the model has one, and the toggles it exposes.
            "chat_template": bool(template),
            "chat_features": template.features if template else []}


@rpc("task.sweeps")
def task_sweeps(conn, p):
    """The sweep the loaded configs declare: its points, their names, what each one changes. The
    model is not rebuilt to answer this, nor to run one of them -- `task.sweep` switches."""
    with LOCK:
        base = STATE["configs"] or session().configs
        points = base.sweep_points()
        return {"runs": len(points), "dimensions": sorted({path for one in points for path in one.values}),
                "crosses_model": sweeps.crosses_model(base),
                "active": base.active_sweep.name if base.active_sweep else None,
                "points": [{"index": one.index, "name": one.name, "describe": one.describe()}
                           for one in points]}


@rpc("task.sweep")
def task_sweep(conn, p):
    """Switch the session onto one point of its sweep, or back onto the plain configs with no
    point. The model stays where it is: what changes is the run around it -- its task, its data,
    the names its outputs are written under.

    A point that reaches the sections a model is *built* from cannot be switched this way (the
    weights are already on the GPU): those sweeps want one process per point, which is what
    run_task.py and the training queue do."""
    with LOCK:
        s, base = session(), STATE["configs"] or session().configs
        name = p.get("point")
        points = {one.name: one for one in base.sweep_points()}
        if name and name not in points:
            raise ValueError(f"'{name}' is not a point of this sweep: {', '.join(points) or 'none'}")
        if name and sweeps.crosses_model(base):
            raise ValueError(f"'{name}' changes the model itself: run this sweep from "
                             f"run_task.py or the training queue, which build one model per point")
        s.configs = sweeps.apply(base, points[name]) if name else base
        s._task = None
    broadcast("task_state", _state_dict())
    return {"sweep": s.configs.active_sweep.name if s.configs.active_sweep else None,
            "task": s.task_name}


@rpc("task.use")
def task_use(conn, p):
    """Switch to a configured task or instantiate a new one from a task type."""
    with LOCK:
        s = session()
        result = {"task": s.use(p["name"])}
    broadcast("task_state", _state_dict())
    return result


@rpc("task.unload")
def task_unload(conn, p):
    """Drop the model: release the session, collect, and give the GPU memory back."""
    with LOCK:
        _release()
    broadcast("task_state", _state_dict())
    return {"msg": "model unloaded"}


@rpc("task.fields")
def task_fields(conn, p):
    """Schema of the active task's config, values included: [{name, label, kind, default, doc,
    options?}] — the GUI renders it with the very same field language as the plugin TOMLs."""
    from matformer.tasks.session import PRESENTATIONS
    from matformer.tasks.task import TaskConfig
    from matformer.utils.schema import fields_schema
    s = session()
    base = {f.name for f in dataclasses.fields(TaskConfig)}
    fields = fields_schema(type(s.task_config), skip=SKIP)
    for e in fields:
        value = getattr(s.task_config, e["name"], None)
        e["interactive"] = e["name"] in base
        if e["name"] == "presentation":
            e.update(kind="radio", options=list(PRESENTATIONS))
        if isinstance(value, bool) or e["kind"] == "bool":
            value = bool(value)
        if e["kind"] == "json" and value is not None:
            value = json.dumps(value, default=str)
        e["default"] = value
    return {"task": s.task_name, "type": type(s.task_config).__name__, "fields": fields}


@rpc("task.set")
def task_set(conn, p):
    """Change one parameter: same language as the console's `set` (bare name = current task).
    JSON values become YAML literals (the override engine parses them back)."""
    value = p["value"]
    value = "null" if value is None else (value if isinstance(value, str) else json.dumps(value))
    with LOCK:
        session().configure(p["path"], value)
    return {"msg": f"{p['path']} = {value}"}


@rpc("task.feed")
def task_feed(conn, p):
    """One text through the active task. Generation streams token events; classification and
    masking additionally return `cells` — one per predicted token/position, with the top-k — so
    the GUI (or any renderer) can draw an interactive prediction. The full distribution stays
    server-side (LAST), fetched on demand with task.probs: compute everything cheap, show on
    demand."""
    s = session()
    stream = p.get("stream", s.wants_stream)
    CANCEL.clear()
    conn.stop.clear()          # a previous Stop is not a permanent property of the connection

    def emit(next_tokens, step):
        if CANCEL.is_set() or conn.stop.is_set():
            raise TaskStopped()
        delta = emit.decoder.push(next_tokens[0].item())
        if delta:
            conn.push("token", {"t": delta, "task": s.task_name})

    from matformer.tasks.session import StreamDecoder
    batch = s.batch_for(p["text"])
    emit.decoder = StreamDecoder(s.tokenizer, batch.pad().pad_right().tensor[0].tolist(),
                                 getattr(s.tokenizer.config, "eos_token_id", None))
    try:
        with LOCK:
            rows = s.feed(p["text"], stream=stream, save=p.get("save", False),
                          streamer=emit if stream else None, batch=batch)
    except TaskStopped:
        return {"reason": "stopped", "status": s.status()}
    LAST["rows"], LAST["task"] = rows, s.task_name
    rendered = s.task.present(rows)
    if rendered is None:
        from matformer.tasks.session import format_rows
        rendered = format_rows(rows, stream)
    first = rows[0] if rows and isinstance(rows[0], dict) else {}
    return {"rendered": rendered, "status": s.status(), "reason": "done",
            "presentation": getattr(s.task_config, "presentation", "probs"),
            # Chat template visibility: always tell whether it was applied and, when it was, hand
            # back the exact prompt that reached the model so a UI can show/hide it.
            "used_chat_template": bool(getattr(s.task, "chat_mode", False)),
            "chat_prompt": getattr(s.task, "rendered_prompt", None),
            "token_level": getattr(s.task, "token_level", None),
            "cells": _cells(s, rows), "tokens": first.get("tokens"),
            "positions": first.get("positions"), "sentence": first.get("sentence"),
            "vocab_head": first.get("vocab_head"), "restored": first.get("restored")}


def _name(s, cls_id):
    """Class id -> human name: the label set when there is one, the vocabulary token otherwise."""
    i2l = getattr(s.task, "id2label", None)
    if i2l and int(cls_id) in i2l:
        return i2l[int(cls_id)]
    # A vocabulary head predicts token ids, not task labels: decode the id instead of printing it.
    try:
        text = s.tokenizer.decode([int(cls_id)])
        if text:
            return text
    except Exception:
        pass
    try:
        return surface(s.tokenizer.tokens([int(cls_id)])[0])
    except Exception:
        return str(int(cls_id))


def _cells(s, rows):
    """One cell per prediction, ready to draw and to click: winning class, its probability, the
    runner-ups, and `index` — the row of the stored distribution task.probs has to read, so the
    detail stays correct even when only some positions are shown. Masking labels its holes, token
    classification its words (a word is labelled by its first subword); a sequence-level head has
    one cell however many tokens were tokenised."""
    if not rows or not isinstance(rows[0], dict) or "topk_ids" not in rows[0]:
        return None
    from matformer.tasks.task import per_position, surface
    row = rows[0]
    ids, probs = per_position(row["topk_ids"]), per_position(row["topk_probs"])
    tokens = row.get("tokens") or []
    if isinstance(tokens, str):
        tokens = [tokens]
    holes = row.get("holes")
    if isinstance(holes, str):
        holes = [holes]
    positions = row.get("positions")
    if hasattr(positions, "tolist"):
        positions = positions.tolist()
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    if not isinstance(positions, (list, tuple)):
        positions = None
    if not isinstance(tokens, (list, tuple)):
        tokens = None
    # A masked sentence has the full subword sequence and the positions that were hidden. That is
    # the only case where the GUI can put every prediction back where the token was.
    in_place = bool(row.get("sentence") and positions and tokens and len(positions) <= ids.shape[0])
    if in_place:
        picks = [int(i) for i in positions if int(i) < len(tokens)]
        cells = []
        for row_index, position in enumerate(picks):
            if row_index >= ids.shape[0]:
                break
            cells.append({"tok": surface(tokens[position]), "label": _name(s, ids[row_index][0]),
                          "p": float(probs[row_index][0]), "index": int(row_index),
                          "position": int(position),
                          "top": [[_name(s, c), float(q)] for c, q in
                                  zip(ids[row_index].tolist(), probs[row_index].tolist())]})
        return cells
    elif holes:
        picks, names = list(range(min(len(holes), ids.shape[0]))), [surface(h) for h in holes]
    elif positions and len(positions) <= ids.shape[0]:
        picks, names = [int(i) for i in positions], [surface(t) for t in tokens]
    elif ids.shape[0] == 1:
        picks, names = [0], [" ".join(surface(t) for t in tokens) or "sequence"]
    else:
        picks = list(range(ids.shape[0]))
        pieces = row.get("pieces") or []
        if pieces:
            names = [str(one) for one in pieces][:len(picks)]
        else:
            names = [surface(t) for t in tokens][:len(picks)]
        names = names or [str(i + 1) for i in picks]
    return [{"tok": names[j] if j < len(names) else str(p),
             "label": _name(s, ids[p][0]), "p": float(probs[p][0]), "index": int(p),
             "position": int(p),
             "top": [[_name(s, c), float(q)] for c, q in zip(ids[p].tolist(), probs[p].tolist())]}
            for j, p in enumerate(picks)]


@rpc("task.probs")
def task_probs(conn, p):
    """Everything known about one prediction: the distribution as a (class, prob) Table, sorted
    descending and capped, next to the stats of that very position. On demand, so that feeding a
    text stays cheap and the detail is one click away. Evaluated and masked tokens have the whole
    vocabulary or class set behind them; generated ones the candidates that were kept."""
    from matformer.utils.table import Table
    from matformer.tasks.task import per_position, top_k
    s = session()
    row = (LAST.get("rows") or [None])[0]
    if not isinstance(row, dict):
        raise RuntimeError("no prediction stored: run a text first")
    i = max(int(p.get("index") or 0), 0)
    if "probs" in row:
        dist = per_position(row["probs"])
        i = min(i, dist.shape[0] - 1)
        dist = dist[i].float()
        k = top_k(int(p.get("top", 50)), dist.shape[-1])
        values, indices = dist.topk(k)
        ordered = dist.sort(descending=True).values
        stats = {"position": i, "classes": int(dist.shape[-1]),
                 "entropy": float(-(dist * (dist + 1e-12).log()).sum())}
        targets = row.get("targets")
        if targets is not None and targets.numel() > i:
            gold = int(targets.reshape(-1)[i])
            stats["gold"] = _name(s, gold)
            stats["gold_probability"] = float(dist[gold]) if gold < dist.shape[-1] else 0.0
    else:
        ids, probs = per_position(row["topk_ids"]), per_position(row["topk_probs"])
        i = min(i, ids.shape[0] - 1)
        values, indices = probs[i].float(), ids[i]
        stats = {"position": i, "candidates": int(values.shape[-1]),
                 "classes": int(getattr(s.model.config, "num_classes", None)
                                or s.model.config.vocab_size)}
    stats["predicted"] = _name(s, indices[0]) if values.numel() else None
    stats["probability"] = float(values[0]) if values.numel() else 0.0
    stats["margin"] = float(values[0] - values[1]) if values.numel() > 1 else 1.0
    if "gold" in stats:
        stats["correct"] = stats["gold"] == stats["predicted"]
    return {"distribution": Table(["class", "prob"],
                                  [{"class": _name(s, c), "prob": float(q)}
                                   for c, q in zip(indices.tolist(), values.tolist())]),
            "stats": stats}


@rpc("task.details")
def task_details(conn, p):
    """Every position of the last run side by side: position, what was there, and the candidates
    the model had — one row each, so a whole generation can be read as a table instead of by
    clicking token after token."""
    from matformer.utils.table import Table
    s = session()
    cells = _cells(s, LAST.get("rows") or [])
    if not cells:
        raise RuntimeError("no per-token detail: run a text with topk_present set")
    rows = [{"position": i, "token": cell["tok"], "prediction": cell["label"],
             "probability": cell["p"],
             "candidates": "  ".join(f"{name} {q:.1%}" for name, q in cell["top"][1:5])}
            for i, cell in enumerate(cells)]
    return Table(["position", "token", "prediction", "probability", "candidates"], rows)


@rpc("task.stop")
def task_stop(conn, p):
    """Interrupt the running generation at the next token."""
    CANCEL.set()
    conn.stop.set()
    return {"msg": "stop requested"}


@rpc("task.run")
def task_run(conn, p):
    """Run the whole task over its dataset (batch mode) and return the summary."""
    with LOCK:
        summary = session().run()
    from matformer.tasks.session import format_summary
    return {"summary": summary, "text": format_summary(summary)}


@rpc("task.annotate")
def task_annotate(conn, p):
    """Annotate a jsonl/csv file or mdat with the active task (see the console's `annotate`)."""
    with LOCK:
        summary = session().annotate(p["input"], output_path=p.get("output"),
                                     column=p.get("column", "text"), label=p.get("label"),
                                     limit=p.get("limit"), batch_size=p.get("batch_size"),
                                     output_format=p.get("format"),
                                     keys_only=p.get("keys_only", False),
                                     mdat_dataset=p.get("dataset"), chat=p.get("chat"))
    return {"output": summary.pop("output", None), "rows": summary.pop("rows", None),
            "summary": summary}


@rpc("model.internals")
def model_internals(conn, p):
    """What the loaded model actually is: runtime, attention implementation, mounted plugins,
    cache — the quick check 'which kernel am I running?' without opening a terminal."""
    from matformer.utils.table import Table
    s = session()
    m = s.model
    rows = [{"what": "runtime", "name": m.runtime_mode, "detail": str(m.device)},
            {"what": "backbone", "name": type(m.model).__name__,
             "detail": f"causal={getattr(m.config, 'is_causal', None)} "
                       f"layers={getattr(m.config, 'num_hidden_layers', getattr(m.config, 'n_layers', None))}"}]
    for name, plug in (getattr(m, "plugins", {}) or {}).items():
        rows.append({"what": "plugin", "name": name, "detail": type(plug).__name__})
    cache = getattr(m, "cache", None)
    if cache is not None:
        kv = cache.storage.get("paged_kv") if hasattr(cache, "storage") else None
        rows.append({"what": "cache", "name": type(cache).__name__, "detail": str(kv or "no paged kv")})
    attn = __import__("os").environ.get("MATFORMER_ATTENTION_NORMAL", "")
    if attn:
        rows.append({"what": "attention", "name": attn, "detail": "via MATFORMER_ATTENTION_NORMAL"})
    return Table(["what", "name", "detail"], rows)


@rpc("task.chat")
def task_chat(conn, p):
    """Chat template state of the active task: whether one is available, which toggles the template
    reads, and the current use/system settings. Read with no params; set any of use/system/options
    by passing them. This is the one place a GUI, console or agent learns how to drive the template."""
    from matformer.utils.table import Table
    s = session()
    template = getattr(s.model, "chat_template", None)
    # All three go through the config so they survive the task being re-instantiated by a later
    # configure(); the generation task reads system_prompt from its config every turn.
    if "use" in p:
        s.configure("use_chat_template", bool(p["use"]))
    if "options" in p:
        s.configure("chat_options", p["options"] or None)
    if "system" in p:
        s.configure("system_prompt", p["system"] or None)
    cfg = s.task.config
    return {"available": bool(template),
            "features": template.features if template else [],
            "use": bool(getattr(cfg, "use_chat_template", False)),
            "system_prompt": getattr(cfg, "system_prompt", None) or getattr(s.task, "system_prompt", None),
            "options": getattr(cfg, "chat_options", None),
            "add_generation_prompt": getattr(cfg, "add_generation_prompt", None)}


@rpc("task.info")
def task_info(conn, p):
    """The console's `model`: model, tokenizer, memory and stats, as text."""
    return {"text": session().model_info()}


@rpc("task.status")
def task_status(conn, p):
    s = session()
    return {"text": s.status(), "stats": s.stats.report()}


def _passthrough(name, method):
    """Expose a task method (clear, show_context, set_system_prompt, ...) as task.<name>."""
    @rpc(f"task.{name}")
    def handler(conn, p, _m=method):
        with LOCK:
            return {"text": getattr(session().task, _m)(*(p.get("args") or ())) or ""}


for _alias, _method in (("clear", "clear"), ("context", "show_context"),
                        ("system", "set_system_prompt"), ("template", "set_chat_template")):
    _passthrough(_alias, _method)
