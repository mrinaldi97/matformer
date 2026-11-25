# matformer/server/store_cmd.py
"""Activation stores read through the plugins that already own them.

A store is not a second kind of object: it is the plugin a training run mounts (DuckDB or LMDB),
found in `model.plugins` or in `model.activation_store` for a virtual model. Mounting, switching
databanks and writing during a forward are already plugin jobs; this module only *reads* them and
turns one address into numbers a UI can draw. Statistics are computed on a sample of sequences, so
a store with millions of rows answers quickly."""
import threading

import torch

from matformer.server.core import rpc

LOCK = threading.Lock()
STATE = {"current": None, "detached": None}


def _is_store(plugin):
    """A store is identified by what it can do, not by the name it was given when mounted: a
    config calls it `collect_store`, the class is still ActivationLMDBStore."""
    return (hasattr(plugin, "databank") and hasattr(plugin, "available_addresses")
            and hasattr(plugin, "flush"))


def _detached():
    return STATE.get("detached")


def _stores():
    """Every activation store: mounted on the loaded model, the model's own, or a detached file."""
    found = {}
    try:
        from matformer.server import tasks_host
        model = tasks_host.session().model
    except Exception:
        model = None
    for plugin in (getattr(model, "plugins", {}) or {}).values():
        if _is_store(plugin):
            found[getattr(plugin, "id", type(plugin).__name__)] = plugin
    own = getattr(model, "activation_store", None)
    if own is not None:
        found.setdefault("model.activation_store", own)
    if _detached() is not None:
        found["detached"] = _detached()
    return found


def _store(p):
    stores = _stores()
    if not stores:
        raise RuntimeError("no activation store on this model: mount one with plugin.attach "
                           "(kind activation_store_lmdb/duckdb), or load a virtual model")
    chosen = p.get("store") or STATE["current"]
    if chosen and chosen in stores:
        return stores[chosen]
    if len(stores) == 1:
        return next(iter(stores.values()))
    names = ", ".join(sorted(stores))
    raise RuntimeError(f"several stores are available ({names}): choose one with store.use")


def _state(store, ident):
    try:
        model = store.load_model_config()
    except Exception:
        model = None
    try:
        databanks = store.list_databank()
    except Exception:
        databanks = [store.databank]
    return {"store": ident, "backend": type(store).__name__,
            "path": getattr(getattr(store, "config", None), "filename", None),
            "databank": store.databank, "databanks": list(databanks or []),
            "sequences": len(store),
            "addresses": list(store.available_addresses() or []),
            "model": (model or {}).get("name") if isinstance(model, dict) else None}


@rpc("store.state")
def store_state(conn, p):
    """The stores this model owns: where they are, how many sequences, which addresses."""
    with LOCK:
        stores = _stores()
        if not stores:
            return {"stores": [], "backend": None}
        rows = [_state(one, ident) for ident, one in stores.items()]
        current = STATE["current"] if STATE["current"] in stores else rows[0]["store"]
        return {"stores": rows, "current": current}


@rpc("store.inspect")
def store_inspect(conn, p):
    """Open a store file read-only for inspection, without mounting it on a model:
    {filename, backend, databank}. Use store.forget to close it."""
    import os
    from matformer import vectors as V
    filename = p.get("filename") or p.get("path")
    if not filename or not os.path.exists(filename):
        raise FileNotFoundError(filename or "store file")
    with LOCK:
        if _detached() is not None:
            try:
                _detached().close()
            except Exception:
                pass
        STATE["detached"] = V.store({"filename": filename,
                                     "backend": str(p.get("backend") or "lmdb").lower(),
                                     "databank": p.get("databank") or "activations",
                                     "readonly": True})
        return _state(STATE["detached"], "detached")


@rpc("store.forget")
def store_forget(conn, p):
    """Close the store opened with store.inspect."""
    with LOCK:
        if _detached() is not None:
            try:
                _detached().close()
            except Exception:
                pass
        STATE["detached"] = None
    return {"msg": "detached store closed"}


@rpc("store.use")
def store_use(conn, p):
    """Pick the store the read commands work on when a model holds more than one."""
    with LOCK:
        ident = p.get("store")
        if ident not in _stores():
            raise RuntimeError(f"no store '{ident}': see store.state")
        STATE["current"] = ident
        return _state(_stores()[ident], ident)


@rpc("store.switch")
def store_switch(conn, p):
    """Make another databank active; rewind=True restarts replay at sequence 0 (evaluation)."""
    with LOCK:
        store = _store(p)
        store.switch_databank(p["databank"], rewind=p.get("rewind", False))
        return _state(store, p.get("store") or STATE["current"] or next(iter(_stores())))


@rpc("store.flush")
def store_flush(conn, p):
    """Write the batches still buffered in the store (mounting a store flushes every `flush_every`
    batches: after a short capture this is what makes the rows visible)."""
    with LOCK:
        store = _store(p)
        store.flush()
        return _state(store, p.get("store") or STATE["current"] or next(iter(_stores())))


@rpc("store.close")
def store_close(conn, p):
    """Flush and close the selected store; the file stays where it is."""
    with LOCK:
        store = _store(p)
        store.close()
        return {"msg": "activation store closed"}


def _sample(store, address, start, limit):
    ids = list(range(int(start), min(int(start) + int(limit), len(store))))
    if not ids:
        raise RuntimeError(f"the store has {len(store)} sequences: nothing to read")
    batch = store.fetch(address, ids)
    tensor = batch.tensor if hasattr(batch, "tensor") else batch
    mask = getattr(batch, "padding_mask", None)
    return ids, tensor.detach().cpu(), (None if mask is None else (~mask).cpu())


def _values(tensor, mask):
    tensor = tensor.float()
    data = tensor.reshape(-1) if mask is None else tensor[mask]
    return data[torch.isfinite(data)]


@rpc("store.summary")
def store_summary(conn, p):
    """Statistics of one address over a sample of sequences: distribution, norms, sparsity.
    `limit` bounds how many sequences are read (default 64); `start` picks where the sample begins."""
    address = p.get("address")
    if not address:
        raise RuntimeError("choose an address (see store.state)")
    with LOCK:
        store = _store(p)
        ids, tensor, mask = _sample(store, address, p.get("start", 0), p.get("limit", 64))
    values = _values(tensor, mask)
    if values.numel() == 0:
        return {"address": address, "sequences": len(ids), "empty": True}
    flat = tensor.float()
    norms = ((flat * mask.unsqueeze(-1)).flatten(1) if mask is not None
             else flat.reshape(len(ids), -1)).norm(dim=1)
    quantiles = torch.tensor([.01, .05, .5, .95, .99])
    q = torch.quantile(values, quantiles).tolist()
    return {"address": address, "sequences": len(ids), "tokens": int(values.numel()),
            "shape": list(tensor.shape[1:]), "dtype": str(tensor.dtype),
            "mean": float(values.mean()), "std": float(values.std()),
            "abs_mean": float(values.abs().mean()), "min": float(values.min()),
            "max": float(values.max()), "sparsity": float((values == 0).float().mean()),
            "norms": {"mean": float(norms.mean()), "std": float(norms.std()) if norms.numel() > 1 else 0.0,
                      "min": float(norms.min()), "max": float(norms.max())},
            "quantiles": dict(zip(["p1", "p5", "p50", "p95", "p99"], q)),
            "per_sequence_norms": norms.tolist(), "ids": ids}


@rpc("store.histogram")
def store_histogram(conn, p):
    """A histogram of the values of one address: {edges, counts}, ready to be drawn by any UI."""
    address, bins = p.get("address"), int(p.get("bins") or 40)
    with LOCK:
        store = _store(p)
        ids, tensor, mask = _sample(store, address, p.get("start", 0), p.get("limit", 64))
    values = _values(tensor, mask)
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        hi = lo + 1e-9
    counts = torch.histc(values, bins=bins, min=lo, max=hi)
    edges = torch.linspace(lo, hi, bins + 1)
    return {"address": address, "ids": ids, "bins": bins,
            "edges": edges.tolist(), "counts": counts.tolist()}


@rpc("store.matrix")
def store_matrix(conn, p):
    """One point per sequence: mean vector of an address projected on its two principal axes.
    A cheap way to see whether a databank clusters (labels, models, runs) before plotting it."""
    address = p.get("address")
    with LOCK:
        store = _store(p)
        ids, tensor, mask = _sample(store, address, p.get("start", 0), p.get("limit", 256))
    flat = tensor.float()
    if mask is not None:
        flat = (flat * mask.unsqueeze(-1)).flatten(1) / mask.sum(1, keepdim=True).clamp_min(1)
    else:
        flat = flat.reshape(len(ids), -1)
    flat = flat - flat.mean(0, keepdim=True)
    if min(flat.shape) < 2:
        raise RuntimeError("need at least two sequences and two dimensions")
    _, _, v = torch.pca_lowrank(flat, q=2)
    projected = flat @ v[:, :2]
    metas = store.metadata(ids) if hasattr(store, "metadata") else [None] * len(ids)
    return {"address": address, "axes": ["pc1", "pc2"],
            "variance": [float(x) for x in projected.var(0)],
            "points": [{"id": i, "x": float(xy[0]), "y": float(xy[1]),
                        "meta": (m if isinstance(m, (str, dict)) else None)}
                       for i, xy, m in zip(ids, projected, metas)]}


@rpc("store.sequences")
def store_sequences(conn, p):
    """Metadata rows: {start, limit} -> one row per stored sequence."""
    with LOCK:
        store = _store(p)
        start, limit = int(p.get("start") or 0), int(p.get("limit") or 100)
        ids = list(range(start, min(start + limit, len(store))))
        metas = store.metadata(ids) if ids else []
    rows = []
    for sid, meta in zip(ids, metas):
        row = {"id": sid}
        if isinstance(meta, dict):
            row.update({str(k): v for k, v in meta.items()})
        elif meta is not None:
            row["meta"] = meta
        rows.append(row)
    return {"rows": rows, "total": len(store)}


@rpc("store.tensor")
def store_tensor(conn, p):
    """One sequence, whole, when it is small enough to read; otherwise its headers and a slice.
    `max_values` bounds the payload (default 4000)."""
    address, sid = p.get("address"), int(p.get("id") or 0)
    limit = int(p.get("max_values") or 4000)
    with LOCK:
        store = _store(p)
        batch = store.fetch(address, [sid])
    tensor = batch.tensor if hasattr(batch, "tensor") else batch
    mask = getattr(batch, "padding_mask", None)
    values = _values(tensor.cpu(), None if mask is None else (~mask).cpu())
    cut = min(values.numel(), limit)
    return {"address": address, "id": sid, "shape": list(tensor.shape),
            "dtype": str(tensor.dtype), "total_values": int(values.numel()),
            "truncated": values.numel() > cut, "values": values[:cut].tolist()}


@rpc("store.config")
def store_config(conn, p):
    """The model configuration stamped in the store, when there is one."""
    with LOCK:
        store = _store(p)
        return {"model": store.load_model_config()}
