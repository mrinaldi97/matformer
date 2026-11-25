# matformer/vectors.py
"""Vectors as first-class objects: directions, embeddings, prototypes and probes.

The physical storage is the activation store the framework already has (LMDB or DuckDB, one
databank per collection, metadata per sequence), so a vector inherits its format, its portability
and its ability to hold millions of rows without a second database. What this module adds is the
*meaning*: every vector has a name, a kind, an optional label, the position it belongs to, the
metric it should be compared with, and free metadata (model, source, description, tags).

Two uses are indistinguishable to the store and only differ in what reads them:

* a steering direction (`kind="steering"`) is mounted on a socket by the steering plugin;
* an embedding or a prototype (`kind="embedding"`, `"prototype"`) is compared with a new vector
  to label a document, an activation or a prompt.

Nothing here is server-specific: the server RPC and the plugins both call this module, so a
script and the GUI save, find and compare the same records."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import torch

_COLLECTIONS = ("steering", "embeddings", "prototypes", "probes", "vectors")
_KIND_FIELDS = ("name", "kind", "label", "address", "alpha", "description", "tags",
                "metric", "model", "source", "pool", "normalize", "meta")
_CACHE = {}


@dataclass
class VectorRecord:
    name: str
    vector: Any
    kind: str = "vector"
    label: Optional[str] = None
    address: Optional[str] = None       # socket / layer / position this vector belongs to
    alpha: float = 1.0
    description: str = ""
    tags: list = field(default_factory=list)
    metric: str = "cosine"              # cosine | dot | l2
    model: Optional[str] = None
    source: Optional[str] = None
    pool: Optional[str] = None          # mean | last, for embeddings
    normalize: bool = False
    meta: dict = field(default_factory=dict)


def normalize_kind(kind: str) -> str:
    kind = str(kind or "vector").strip().lower()
    return kind or "vector"


def store(spec):
    """Open (and cache) the physical store named by {filename, backend, databank, readonly}."""
    from matformer.matformer_registry import registry
    filename = spec.get("filename") or spec.get("path")
    if not filename:
        raise ValueError("a vector store needs a filename")
    backend = str(spec.get("backend") or "lmdb").lower()
    databank = spec.get("databank") or "vectors"
    readonly = bool(spec.get("readonly", False))
    key = (filename, backend, databank, readonly)
    if key in _CACHE:
        return _CACHE[key]
    kind = f"activation_store_{backend}"
    cls = registry.get_definition("plugin", kind)
    if cls is None:
        raise ValueError(f"backend '{backend}' has no activation store")
    cfg = cls.from_dict({"filename": filename, "databank": databank,
                         "readonly": readonly, "preload": True})
    opened = registry.create("plugin", kind, config=cfg, model_config=None, cache=None)
    _CACHE[key] = opened
    return opened


def _physical(meta):
    """The address a record is stored under physically. New records use one; older ones used the
    semantic address itself, so both are readable."""
    return meta.get("_address") or meta.get("address") or "vector"


def _records(physical, drop_deleted=True):
    """Latest record per (kind, name), plus its id. A later save of the same name wins."""
    ids = list(range(len(physical)))
    metas = physical.metadata(ids) if ids else []
    found = {}
    for sid, meta in zip(ids, metas):
        if not isinstance(meta, dict) or not meta.get("name"):
            continue
        found[(meta.get("kind", "vector"), meta["name"])] = dict(meta, id=sid)
    rows = list(found.values())
    if drop_deleted:
        rows = [r for r in rows if not r.get("deleted")]
    return sorted(rows, key=lambda r: (r.get("kind", ""), r["name"]))


def find(physical, name, kind=None):
    for record in _records(physical):
        if record["name"] == name and (kind is None or record.get("kind") == kind):
            return record
    return None


def vector_of(physical, record) -> torch.Tensor:
    batch = physical.fetch(_physical(record), [record["id"]])
    tensor = batch.tensor if hasattr(batch, "tensor") else batch
    return tensor.detach().float().flatten()


def save(physical, record: VectorRecord) -> dict:
    """Store one vector as a new sequence; the previous record with the same name stays readable
    as history but is no longer the current one."""
    if not record.name:
        raise ValueError("a vector needs a name")
    tensor = torch.as_tensor(record.vector, dtype=torch.float32).flatten()
    if tensor.numel() == 0:
        raise ValueError("the vector is empty")
    norm = float(tensor.norm())
    if record.normalize or record.metric == "cosine" and record.kind in ("embedding", "prototype"):
        tensor = tensor / max(norm, 1e-12)
    meta = {
        "schema": 1, "name": record.name, "kind": normalize_kind(record.kind),
        "label": record.label, "address": record.address, "alpha": float(record.alpha),
        "description": record.description, "tags": list(record.tags or []),
        "metric": record.metric, "model": record.model, "source": record.source,
        "pool": record.pool, "dim": int(tensor.numel()), "norm": norm,
        "normalized": bool(record.normalize or
                           (record.metric == "cosine" and record.kind in ("embedding", "prototype"))),
        "created": time.time(), "meta": record.meta or {}, "_address": "vector",
    }
    sid = int(physical.next_id)
    physical._write([(sid, json.dumps(meta))], [physical._encode(sid, "vector", tensor)])
    physical.next_id = sid + 1
    physical.flush()
    return dict(meta, id=sid)


def delete(physical, name, kind=None) -> bool:
    """Hide a record: a tombstone keeps the history (the store is append-only) but `list` and
    `get` stop returning it."""
    record = find(physical, name, kind)
    if record is None:
        return False
    meta = {**record, "deleted": True, "deleted_at": time.time(), "_address": _physical(record),
            "id": None}
    meta.pop("id", None)
    sid = int(physical.next_id)
    tensor = vector_of(physical, record)
    physical._write([(sid, json.dumps(meta))], [physical._encode(sid, "vector", tensor)])
    physical.next_id = sid + 1
    physical.flush()
    return True


def list_records(physical, kind=None, label=None, tag=None, address=None) -> list:
    rows = _records(physical)
    if kind:
        rows = [r for r in rows if r.get("kind") == normalize_kind(kind)]
    if label is not None:
        rows = [r for r in rows if r.get("label") == label]
    if tag:
        rows = [r for r in rows if tag in (r.get("tags") or [])]
    if address:
        rows = [r for r in rows if r.get("address") == address]
    return rows


def _matrix(physical, rows):
    return torch.stack([vector_of(physical, row) for row in rows]) if rows else torch.empty(0)


def _score_matrix(vectors, query, metric):
    query = torch.as_tensor(query, dtype=torch.float32).flatten()
    if metric == "cosine":
        a = vectors / vectors.norm(dim=1, keepdim=True).clamp_min(1e-12)
        q = query / query.norm().clamp_min(1e-12)
        return a @ q
    if metric == "dot":
        return vectors @ query
    if metric == "l2":
        return -(vectors - query).norm(dim=1)
    raise ValueError(f"unknown metric '{metric}' (cosine, dot, l2)")


def nearest(physical, query, k=5, kind=None, label=None, metric="cosine") -> list:
    rows = list_records(physical, kind=kind, label=label)
    if not rows:
        return []
    vectors = _matrix(physical, rows)
    if vectors.shape[1] != torch.as_tensor(query).flatten().numel():
        raise ValueError(f"dimension mismatch: store {vectors.shape[1]}, query "
                         f"{torch.as_tensor(query).flatten().numel()}")
    scores = _score_matrix(vectors, query, metric)
    order = torch.argsort(scores, descending=True)[:max(1, int(k))]
    return [{**{key: rows[i].get(key) for key in ("name", "kind", "label", "address",
                                                  "description", "tags", "id")},
             "score": float(scores[i])} for i in order.tolist()]


def classify(physical, query, k=5, kind=None, label=None, metric="cosine") -> dict:
    """Label a vector by its neighbours: the top-k vote, weighted by similarity, plus the best
    per-label score. The same call labels a document embedding, an activation or a prompt."""
    neighbours = nearest(physical, query, k=k, kind=kind, label=label, metric=metric)
    votes, best = {}, {}
    for one in neighbours:
        name = one.get("label")
        if name is None:
            continue
        weight = max(one["score"], 0.0) if metric != "l2" else 1.0 / (1.0 + abs(one["score"]))
        votes[name] = votes.get(name, 0.0) + weight
        best[name] = max(best.get(name, -1e30), one["score"])
    total = sum(votes.values()) or 1.0
    ranking = [{"label": name, "vote": value / total, "best": best[name]}
               for name, value in votes.items()]
    ranking.sort(key=lambda row: (row["vote"], row["best"]), reverse=True)
    return {"labels": ranking, "neighbours": neighbours, "k": len(neighbours)}


def compare(physical, first, second, metric="cosine") -> dict:
    """Similarity between two stored vectors, or between one stored and one raw vector."""
    def tensor_of(value):
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, list):
            return torch.as_tensor(value, dtype=torch.float32).flatten()
        record = find(physical, value["name"], value.get("kind")) if isinstance(value, dict) else \
            find(physical, value)
        if record is None:
            raise ValueError(f"no vector '{value}'")
        return vector_of(physical, record)
    a, b = tensor_of(first), tensor_of(second)
    if a.numel() != b.numel():
        raise ValueError(f"dimension mismatch: {a.numel()} vs {b.numel()}")
    if metric == "cosine":
        score = float(torch.dot(a / a.norm().clamp_min(1e-12),
                                b / b.norm().clamp_min(1e-12)))
    elif metric == "dot":
        score = float(torch.dot(a, b))
    elif metric == "l2":
        score = -float((a - b).norm())
    else:
        raise ValueError(f"unknown metric '{metric}'")
    return {"metric": metric, "score": score}


def export_json(physical, kind=None) -> dict:
    """A portable snapshot: metadata + the numbers, without the backend's file format."""
    rows = list_records(physical, kind=kind)
    return {"databank": physical.databank, "vectors": [
        {**{key: row.get(key) for key in _KIND_FIELDS},
         "vector": vector_of(physical, row).tolist()} for row in rows]}


def import_json(physical, payload) -> list:
    saved = []
    for one in (payload or {}).get("vectors") or []:
        saved.append(save(physical, VectorRecord(**{k: one[k] for k in _KIND_FIELDS
                                                    if k in one}, vector=one["vector"])))
    return saved


def record_from_dict(data: dict) -> VectorRecord:
    fields = {key: value for key, value in (data or {}).items() if key in _KIND_FIELDS}
    if "vector" not in (data or {}):
        raise ValueError("a vector record needs 'vector'")
    return VectorRecord(**fields, vector=data["vector"])


def as_public(meta: dict) -> dict:
    """The record as a UI should see it, without the physical address or the timestamps it did
    not ask for."""
    return {key: meta.get(key) for key in
            ("id", "name", "kind", "label", "address", "alpha", "description", "tags",
             "metric", "model", "source", "pool", "dim", "norm", "normalized", "created",
             "meta")}
