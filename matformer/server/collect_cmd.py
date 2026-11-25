# matformer/server/collect_cmd.py
"""Collect activations as a queue job, and pool an existing store into vectors.

The job is the same forward task the console runs; the config carries the activation-cache/store
plugins (see the collect wizard), so this module only queues it and prints where the vectors go.
The second call pools an existing store now, to try mean/last/position pooling without a new run."""
from datetime import datetime

from matformer.server.core import rpc
from matformer.server.task_jobs import enqueue_task


@rpc("collect.start")
def collect_start(conn, p):
    """Queue a forward over an existing config: {cfg: [...], model, device, vectorize, label_key,
    name, overrides}. The config must already mount the activation store plugins."""
    if not p.get("cfg"):
        raise ValueError("collect.start needs the config files (they define data and plugins)")
    name = p.get("name") or f"collect-{datetime.now():%H%M%S}"
    vectorize = dict(p.get("vectorize") or {})
    if vectorize.get("enabled") and not vectorize.get("filename"):
        raise ValueError("vectorize.enabled needs vectorize.filename")
    spec = {"mode": "run", "cfg": p["cfg"], "model": p.get("model"),
            "device": p.get("device"), "task": p.get("task") or "forward",
            "overrides": p.get("overrides"), "vectorize": vectorize,
            "label_key": p.get("label_key") or "label"}
    job = enqueue_task(spec, name=name, project=p.get("project") or "collect")
    return {"job": job, "name": name, "vectorize": vectorize}


@rpc("collect.vectorize")
def collect_vectorize(conn, p):
    """Pool an existing activation store into vectors now (no new forward)."""
    from matformer import vectors as V
    from matformer.vectors.pooling import pool_activation_store
    store = dict(p.get("store") or {})
    vector = dict(p.get("vectorize") or {})
    if not store.get("filename") or not vector.get("filename"):
        raise ValueError("collect.vectorize needs store.filename and vectorize.filename")
    source = V.store({**store, "readonly": True})
    saved = pool_activation_store(source, vector, p.get("label_key") or "label",
                                  p.get("model"))
    return {"saved": saved, "file": vector["filename"], "databank": vector.get("databank")}
