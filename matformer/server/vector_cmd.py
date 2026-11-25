# matformer/server/vector_cmd.py
"""Vectors over RPC: save, find, compare, label by similarity, and embed texts.

The records live in the activation store (see matformer.vectors); this module is the thin wire
between that module and a UI. One call can also turn texts into vectors, so a user can build a
prototype set and label a document without leaving the client."""
import json

from matformer.server.core import rpc
from matformer import vectors as V


def _spec(p):
    spec = dict(p.get("store") or {})
    for key in ("filename", "path", "backend", "databank", "readonly"):
        if key in p and key not in spec:
            spec[key] = p[key]
    if not (spec.get("filename") or spec.get("path")):
        raise ValueError("the call needs a vector store: {filename, backend, databank}")
    return spec


def _record(p):
    data = {**p, **p.get("record", {})}
    if "vector" not in data and "direction" in data:
        data["vector"] = data["direction"]
    return V.record_from_dict(data)


@rpc("vectors.save")
def vectors_save(conn, p):
    """Store one vector: {store, name, vector|direction, kind?, label?, address?, alpha?,
    description?, tags?, metric?, model?, source?, pool?, normalize?, meta?}."""
    record = _record(p)
    if record.vector is None and p.get("direction") is not None:
        record.vector = p["direction"]
    saved = V.save(V.store(_spec(p)), record)
    return {"vector": V.as_public(saved)}


@rpc("vectors.list")
def vectors_list(conn, p):
    """Records in a store, optionally filtered by kind, label, tag or address."""
    physical = V.store(_spec(p))
    rows = V.list_records(physical, kind=p.get("kind"), label=p.get("label"),
                          tag=p.get("tag"), address=p.get("address"))
    try:
        databanks = physical.list_databank()
    except Exception:
        databanks = [physical.databank]
    return {"vectors": [V.as_public(row) for row in rows], "databank": physical.databank,
            "databanks": list(databanks or [physical.databank]),
            "filename": getattr(getattr(physical, "config", None), "filename", None)}


@rpc("vectors.get")
def vectors_get(conn, p):
    """One record, whole: its metadata and the vector as a list of numbers."""
    physical = V.store(_spec(p))
    record = V.find(physical, p["name"], p.get("kind"))
    if record is None:
        raise RuntimeError(f"no vector '{p['name']}'")
    return {"vector": V.as_public(record), "direction": V.vector_of(physical, record).tolist()}


@rpc("vectors.delete")
def vectors_delete(conn, p):
    """Hide a record (append-only store: a tombstone keeps the history)."""
    physical = V.store({**_spec(p), "readonly": False})
    return {"deleted": V.delete(physical, p["name"], p.get("kind"))}


@rpc("vectors.nearest")
def vectors_nearest(conn, p):
    """Top-k neighbours of a raw vector or of a stored record: {store, vector | name, k, kind?}."""
    physical = V.store(_spec(p))
    query = p.get("vector")
    if query is None:
        record = V.find(physical, p["name"], p.get("kind"))
        if record is None:
            raise RuntimeError(f"no vector '{p['name']}'")
        query = V.vector_of(physical, record).tolist()
    return {"neighbours": V.nearest(physical, query, k=p.get("k", 5), kind=p.get("search_kind"),
                                    label=p.get("label"), metric=p.get("metric", "cosine"))}


@rpc("vectors.classify")
def vectors_classify(conn, p):
    """Label a vector by similarity to labelled prototypes/embeddings (top-k weighted vote)."""
    physical = V.store(_spec(p))
    query = p.get("vector")
    if query is None:
        record = V.find(physical, p["name"], p.get("kind"))
        if record is None:
            raise RuntimeError(f"no vector '{p['name']}'")
        query = V.vector_of(physical, record).tolist()
    return V.classify(physical, query, k=p.get("k", 5), kind=p.get("search_kind"),
                      label=p.get("filter_label"), metric=p.get("metric", "cosine"))


@rpc("vectors.compare")
def vectors_compare(conn, p):
    """Cosine/dot/L2 between two records, or between a record and a raw vector."""
    physical = V.store(_spec(p))
    return V.compare(physical, p["a"], p["b"], metric=p.get("metric", "cosine"))


@rpc("vectors.export")
def vectors_export(conn, p):
    """Portable JSON snapshot (metadata + numbers); writes `path` on the server when given."""
    physical = V.store(_spec(p))
    payload = V.export_json(physical, kind=p.get("kind"))
    if p.get("path"):
        with open(p["path"], "w") as handle:
            json.dump(payload, handle)
        return {"path": p["path"], "vectors": len(payload["vectors"])}
    return payload


@rpc("vectors.import")
def vectors_import(conn, p):
    """Load a snapshot from `payload` or from `path` (the server reads the file)."""
    payload = p.get("payload")
    if payload is None and p.get("path"):
        with open(p["path"]) as handle:
            payload = json.load(handle)
    physical = V.store({**_spec(p), "readonly": False})
    saved = V.import_json(physical, payload)
    return {"imported": len(saved), "vectors": [V.as_public(one) for one in saved]}


@rpc("vectors.embed")
def vectors_embed(conn, p):
    """Turn texts into vectors at one socket: {texts, address, pool: mean|last, save?, store?,
    kind?, label (or labels), name_prefix?, normalize?}. Returns one vector per text."""
    import torch
    from matformer.server import tasks_host
    texts = p.get("texts") or ([p["text"]] if p.get("text") else [])
    address = p.get("address")
    if not texts or not address:
        raise ValueError("vectors.embed needs texts and the socket address to read")
    session = tasks_host.session()
    model = session.model
    cache = getattr(model, "cache", None)
    if cache is None:
        raise RuntimeError("the loaded model has no activation cache")
    temporary = None
    mounted = [plug for plug in (model.plugins or {}).values()
               if getattr(plug, "kind", "") == "activation_cache"
               and getattr(getattr(plug, "socket", None), "address", None) == address]
    if not mounted:
        from matformer.matformer_registry import registry
        definition = registry.get_definition("plugin", "activation_cache")
        cfg = definition(**{}) if isinstance(definition, type) else definition
        temporary = model.attach(address, cfg, name="embed_cache")
    try:
        batch = session.batch(texts)
        cache.storage.setdefault("activation_cache", {}).clear()
        with torch.no_grad():
            model(batch)
        captured = (cache.storage.get("activation_cache") or {}).get(address)
        if captured is None:
            raise RuntimeError(f"the socket '{address}' produced no activation cache entry")
        tensor = captured.tensor if hasattr(captured, "tensor") else captured
        mask = getattr(captured, "padding_mask", None)
        flat = tensor.float()
        cu = getattr(captured, "cu_seqlens", None)
        if mask is not None:
            valid = (~mask).unsqueeze(-1)
            pooled = (flat * valid).sum(1) / valid.sum(1).clamp_min(1)
            if p.get("pool") == "last":
                lengths = (~mask).sum(1).clamp_min(1) - 1
                pooled = torch.stack([flat[i, length] for i, length in enumerate(lengths)])
        elif cu is not None:
            # varlen batches arrive as one long (tokens, D) tensor plus the sequence offsets.
            segments = [flat[cu[i]:cu[i + 1]] for i in range(len(cu) - 1)]
            pooled = torch.stack([(seg[-1] if p.get("pool") == "last" else seg.mean(0))
                                  for seg in segments])
        elif flat.ndim == 3:
            if p.get("pool") == "last":
                pooled = flat[:, -1]
            else:
                pooled = flat.mean(1)
        elif flat.ndim == 2:
            pooled = flat.mean(0, keepdim=True)
        else:
            pooled = flat.flatten().unsqueeze(0)
        embeddings = pooled.detach().cpu()
        result = {"address": address, "dim": int(embeddings.shape[1]),
                  "vectors": embeddings.tolist(), "texts": texts}
        if p.get("save"):
            if not p.get("store"):
                raise ValueError("save=True needs a store")
            physical = V.store({**_spec(p), "readonly": False})
            labels = p.get("labels")
            if labels is None and p.get("label") is not None:
                labels = [p["label"]] * len(texts)
            prefix = p.get("name_prefix") or "embedding"
            saved = []
            for index, one in enumerate(embeddings):
                label = labels[index] if labels and index < len(labels) else None
                name = (p.get("names") or [None] * len(texts))[index] \
                    if p.get("names") else f"{prefix}-{int(__import__('time').time())}-{index}"
                saved.append(V.save(physical, V.VectorRecord(
                    name=name, vector=one.tolist(), kind=p.get("kind") or "embedding",
                    label=label, address=address, description=texts[index][:300],
                    model=getattr(session, "source", None), pool=p.get("pool") or "mean",
                    metric=p.get("metric") or "cosine",
                    normalize=p.get("normalize", True),
                    meta={"text": texts[index]})))
            result["saved"] = [V.as_public(one) for one in saved]
        return result
    finally:
        if temporary:
            for plug in temporary:
                model.detach(plug.socket.address, plug.name)
