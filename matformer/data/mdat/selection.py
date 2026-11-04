"""Build temporary document indexes without changing an MDAT on disk."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import tempfile
from typing import Dict, Optional, Sequence

import numpy as np

from .utils import get_datatype_for_numpy


@dataclass(frozen=True)
class DatasetSelection:
    """Select whole documents from a named view.

    Budgets count cached tokenizer tokens, excluding padding and special tokens
    added during loading. Documents that do not fit are skipped; later, smaller
    documents may still fit. ``domain_tokens`` sets independent caps, not ratios.
    With no ``domains``, its keys also identify the domains to include.
    ``seed=None`` preserves source order; an integer deterministically permutes
    candidate positions using constant memory (not a uniform random shuffle).
    """

    domains: Optional[Sequence[str]] = None
    max_tokens: Optional[int] = None
    domain_tokens: Optional[Dict[str, int]] = None
    seed: Optional[int] = None

    def __post_init__(self):
        def validate_budget(name, value):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer token count")

        if self.max_tokens is not None:
            validate_budget("max_tokens", self.max_tokens)
        if self.domains is not None:
            if isinstance(self.domains, str) or not self.domains:
                raise ValueError("domains must be a non-empty list of subdataset names")
            if any(not isinstance(name, str) or not name for name in self.domains):
                raise ValueError("Each domain must be a non-empty subdataset name")
            if len(set(self.domains)) != len(self.domains):
                raise ValueError("domains must not contain duplicates")
        if self.domain_tokens is not None:
            if not isinstance(self.domain_tokens, dict) or not self.domain_tokens:
                raise ValueError("domain_tokens must be a non-empty mapping of names to caps")
            for name, value in self.domain_tokens.items():
                if not isinstance(name, str) or not name:
                    raise ValueError("domain_tokens keys must be subdataset names")
                validate_budget(f"domain_tokens[{name!r}]", value)
            if self.domains is not None and set(self.domain_tokens) - set(self.domains):
                raise ValueError("domain_tokens contains a domain excluded by domains")
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int)):
            raise ValueError("seed must be an integer or None")

    def resolve_domains(self, available):
        names = list(self.domains if self.domains is not None else
                     self.domain_tokens if self.domain_tokens is not None else available)
        unknown = set(names) - set(available)
        if unknown:
            raise ValueError(f"Domains absent from the source view: {sorted(unknown)}")
        return names


def _positions(size, seed):
    """A reproducible bijection over positions; no list of corpus IDs in RAM."""
    if seed is None or size < 2:
        yield from range(size)
        return
    rng = random.Random(seed)
    stride = rng.randrange(1, size)
    while math.gcd(stride, size) != 1:
        stride = rng.randrange(1, size)
    offset = rng.randrange(size)
    for index in range(size):
        yield (stride * index + offset) % size


def source_references(dataset, seed=None):
    """Read exact view membership, including partial views' pointer files."""
    metadata = dataset.db.get_manifest(view=dataset.current_view)
    pointer_format = metadata.get("shuffle_struct_format")
    if pointer_format:
        record = struct.Struct(pointer_format)
        path = Path(dataset.shuffling_path) / f"{dataset.current_view}.mdat"
        count, remainder = divmod(path.stat().st_size, record.size)
        if remainder:
            raise ValueError(f"Incomplete document pointer in {path}")
        expected = dataset.db.get_view_totals(dataset.current_view)["document_number"]
        if count != expected:
            raise ValueError(f"View index has {count} documents; metadata declares {expected}")
        with path.open("rb") as stream:
            for position in _positions(count, seed):
                if seed is not None:
                    stream.seek(position * record.size)
                yield record.unpack(stream.read(record.size))
        return

    if dataset.db.ask_view(dataset.current_view, "partial_submdats"):
        raise ValueError("A partial source view needs a document pointer file")
    # A view containing only preserved domains needs no stored shuffle file.
    from bisect import bisect_right
    ids = sorted(dataset.db.ask_view(dataset.current_view, "preserved_submdats"))
    boundaries = [0]
    for domain_id in ids:
        boundaries.append(boundaries[-1] + dataset.db.get_manifest(submdat=domain_id)["document_number"])
    for position in _positions(boundaries[-1], seed):
        domain_position = bisect_right(boundaries, position) - 1
        yield ids[domain_position], position - boundaries[domain_position]


class SelectedView:
    """Temporary pointer/count files and exact statistics for a selection.

    Only indexes are written to the system temporary directory. Source LMDBs,
    SQLite metadata, and saved views are untouched. Call ``close`` to release
    the indexes (also done by ``MatformerDataset.close``).
    """

    pointer_format = "<QQ"
    _pointer = struct.Struct(pointer_format)
    _count = struct.Struct("<Q")

    def __init__(self, dataset, config, domains):
        self.config = config
        self.source_view = dataset.current_view
        self.strategy_name = (dataset.current_strategy.strategy_name
                              if dataset.current_strategy is not None else None)
        self._directory = tempfile.TemporaryDirectory(prefix="matformer-selection-")
        self.pointer_path = Path(self._directory.name) / "documents.mdat"
        self.count_path = Path(self._directory.name) / "chunks.bin"
        self.documents = 0
        self.tokens = 0 if self.strategy_name else None
        self.chunks = 0
        self.domain_stats = {name: {"documents": 0, "tokens": 0 if self.strategy_name else None}
                             for name in domains}
        self._length_cache = {}
        try:
            self._build(dataset, domains)
        except BaseException:
            self.close()
            raise

    def _build(self, dataset, domains):
        config = self.config
        if (config.max_tokens is not None or config.domain_tokens is not None) and not self.strategy_name:
            raise ValueError("Token budgets require a pretokenization strategy")
        strategy = dataset.current_strategy
        if strategy is not None:
            if (strategy.splitter_arguments or {}).get("splitter_decides_chunking"):
                raise ValueError("Selections require standard stored chunk boundaries; custom chunking is unsupported")
            token_size = np.dtype(get_datatype_for_numpy(strategy.tokens_datatype)).itemsize
            chunk_size = np.dtype(get_datatype_for_numpy(strategy.chunks_datatype)).itemsize
            for name in domains:
                if "tokens" not in dataset.loaded_submdats[name].pretok_db:
                    raise ValueError(f"Domain {name!r} has no cached tokens for {self.strategy_name!r}")

        names = dataset.db.get_dsmap("id")
        selected_names = set(domains)
        digest = hashlib.sha256(json.dumps({"view": self.source_view, "strategy": self.strategy_name,
                                           "selection": asdict(config)}, sort_keys=True).encode())
        with self.pointer_path.open("wb") as pointers, self.count_path.open("wb") as counts:
            for domain_id, document_id in source_references(dataset, config.seed):
                if config.max_tokens is not None and self.tokens >= config.max_tokens:
                    break
                if config.domain_tokens and all(
                    name in config.domain_tokens and stats['tokens'] >= config.domain_tokens[name]
                    for name, stats in self.domain_stats.items()
                ):
                    break
                name = names[domain_id]
                if name not in selected_names:
                    continue
                stats = self.domain_stats[name]
                cap = (config.domain_tokens or {}).get(name)
                if cap is not None and stats["tokens"] >= cap:
                    continue
                token_count = chunk_count = 0
                if strategy is not None:
                    cache = dataset.loaded_submdats[name].pretok_db
                    token_bytes = cache["tokens"][document_id]
                    if token_bytes is None or len(token_bytes) % token_size:
                        raise ValueError(f"Invalid cached tokens for {name}[{document_id}]")
                    token_count = len(token_bytes) // token_size
                    if config.max_tokens is not None and self.tokens + token_count > config.max_tokens:
                        continue
                    if cap is not None and stats["tokens"] + token_count > cap:
                        continue
                    if "chunks" in cache:
                        chunk_bytes = cache["chunks"][document_id]
                        if chunk_bytes is None or len(chunk_bytes) % chunk_size:
                            raise ValueError(f"Invalid cached chunks for {name}[{document_id}]")
                        chunk_count = len(chunk_bytes) // chunk_size
                pointer = self._pointer.pack(domain_id, document_id)
                count = self._count.pack(chunk_count)
                pointers.write(pointer)
                counts.write(count)
                digest.update(pointer + count + struct.pack("<Q", token_count))
                self.documents += 1
                self.chunks += chunk_count
                stats["documents"] += 1
                if strategy is not None:
                    self.tokens += token_count
                    stats["tokens"] += token_count
        self.fingerprint = digest.hexdigest()

    def worker_lengths(self, multiplier=1, workers=1, chunked=False):
        """Exact lengths for the iterator's document-level rank assignment."""
        if workers < 1 or multiplier < 1:
            raise ValueError("workers and multiplier must be positive")
        key = multiplier, workers, chunked
        if key not in self._length_cache:
            lengths = [0] * workers
            with self.count_path.open("rb") as stream:
                offset = 0
                while True:
                    block = stream.read(8 * 65536)
                    if not block:
                        break
                    for index, (chunks,) in enumerate(struct.iter_unpack("<Q", block), start=offset):
                        lengths[index % workers] += (chunks + multiplier - 1) // multiplier if chunked else 1
                    offset += len(block) // 8
            self._length_cache[key] = lengths
        return list(self._length_cache[key])

    def summary(self):
        return {"source_view": self.source_view, "strategy": self.strategy_name,
                "documents": self.documents, "tokens": self.tokens,
                "chunks": self.chunks, "domains": self.domain_stats,
                "selection": asdict(self.config), "fingerprint": self.fingerprint}

    def close(self):
        self._directory.cleanup()
