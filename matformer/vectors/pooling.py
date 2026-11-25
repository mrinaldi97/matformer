# matformer/vectors/pooling.py
"""Pool a collected activation store into vectors, with the dataset metadata attached.

This is the one implementation used by both the collect wizard's job and `collect.vectorize`: a
store is a store, whether it was just written or read back from disk."""
from __future__ import annotations

import json

import torch


def pool_tensor(tensor, mask=None, cu_seqlens=None, pool="mean", positions=None):
    """One vector per sequence from (batch, tokens, D), a padded batch or a varlen tensor."""
    flat = tensor.float()
    if mask is not None:
        rows = [flat[i][~mask[i]] for i in range(flat.shape[0])]
    elif cu_seqlens is not None:
        rows = [flat[cu_seqlens[i]:cu_seqlens[i + 1]] for i in range(len(cu_seqlens) - 1)]
    elif flat.ndim == 3:
        rows = [flat[i] for i in range(flat.shape[0])]
    else:
        rows = [flat]
    out = []
    for row in rows:
        if row.numel() == 0:
            out.append(torch.zeros(tensor.shape[-1]))
            continue
        current = row
        if positions:
            picked = [current[p] for p in positions if 0 <= p < current.shape[0]]
            current = torch.stack(picked) if picked else current
        if pool == "last":
            value = current[-1]
        elif pool == "first":
            value = current[0]
        elif pool == "max":
            value = current.max(0).values
        else:
            value = current.mean(0)
        out.append(value)
    return torch.stack(out)


def _label_of(meta, key):
    if not isinstance(meta, dict):
        return None
    value = meta.get(key) if key else None
    if value is None and key != "label":
        value = meta.get("label", meta.get("labels"))
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def pool_activation_store(source, vector_spec, label_key="label", model_source=None):
    """Read every address of an already-open activation store and save one vector per sequence.
    `source` is the live store plugin (or any object with the ActivationStore read API)."""
    from matformer import vectors as V
    vector_spec = dict(vector_spec or {})
    target = V.store({"filename": vector_spec["filename"],
                      "backend": vector_spec.get("backend") or "lmdb",
                      "databank": vector_spec.get("databank") or "vectors", "readonly": False})
    kind = vector_spec.get("kind") or "embedding"
    pool = vector_spec.get("pool") or "mean"
    positions = vector_spec.get("positions") or None
    limit = int(vector_spec.get("limit") or len(source))
    saved = 0
    for address in source.available_addresses():
        ids = list(range(min(limit, len(source))))
        for start in range(0, len(ids), 128):
            chunk = ids[start:start + 128]
            batch = source.fetch(address, chunk)
            tensor = batch.tensor if hasattr(batch, "tensor") else batch
            mask = getattr(batch, "padding_mask", None)
            cu = getattr(batch, "cu_seqlens", None)
            pooled = pool_tensor(tensor.detach().cpu(),
                                 mask.detach().cpu() if mask is not None else None,
                                 cu.tolist() if cu is not None else None, pool, positions)
            metas = source.metadata(chunk)
            for index, sid in enumerate(chunk):
                meta = metas[index] if index < len(metas) else None
                label = _label_of(meta, label_key)
                layer = address.split(".")[-2] if "." in address else "act"
                name = f"{vector_spec.get('name_prefix') or 'act'}-{layer}-{sid}"
                V.save(target, V.VectorRecord(
                    name=name, vector=pooled[index].tolist(), kind=kind, label=label,
                    address=address, description=(json.dumps(meta)[:300] if meta else address),
                    model=model_source, source=vector_spec.get("source"),
                    pool=pool, metric=vector_spec.get("metric") or "cosine",
                    normalize=pool == "mean", meta={"sequence": sid, "metadata": meta,
                                                    "pool_positions": positions}))
                saved += 1
    print(f"vectorized {saved} sequences into {vector_spec['filename']} ({kind})", flush=True)
    return saved
