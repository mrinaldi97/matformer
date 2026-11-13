"""LMDB-backed key/value storage used by mdats (datasets) and pretokenization caches.

Compression backend is chosen via the USE_ZLIB environment variable
(zstandard by default). Keys are stored as decimal strings; ``length`` tracks
one past the largest key ever written and is persisted under ``__len__``.
"""
from __future__ import annotations

import os
import math
from typing import Dict

import lmdb
import orjson

USE_ZLIB = os.environ.get('USE_ZLIB', 'false').lower() == 'true'

if USE_ZLIB:
    import zlib
    def compress_func(data, level):
        return zlib.compress(data, level)
    def decompress_func(data):
        return zlib.decompress(data)
    COMPRESSION_BACKEND = "zlib"
else:
    import zstandard as zstd
    _zstd_compressors = {}
    _zstd_decompressor = zstd.ZstdDecompressor()

    def compress_func(data, level):
        if level not in _zstd_compressors:
            _zstd_compressors[level] = zstd.ZstdCompressor(level=level)
        return _zstd_compressors[level].compress(data)

    def decompress_func(data):
        return _zstd_decompressor.decompress(data)

    COMPRESSION_BACKEND = "zstd"


class LMDBDataset:
    """LMDB key/value store wrapper.

    Use the ``open()`` factory (not the constructor) when the same file may be
    opened more than once in a process (e.g. two containers linking the same
    dataset, or the host container and a linked view): compatible handles are
    shared through a reference-counted registry, and the environment is only
    really closed when the last holder calls close().
    """

    _registry: Dict[str, 'LMDBDataset'] = {}

    @classmethod
    def open(cls, path, readonly=True, lock=False, compressed=False, compression_level=0,
             map_size=1 << 44, batch_size=50000):
        key = os.path.abspath(path)
        existing = cls._registry.get(key)
        if existing is not None:
            if existing.compressed != compressed or existing.compression_level != compression_level:
                raise RuntimeError(f"{path}: already opened in this process with different "
                                   "compression settings.")
            if existing.readonly and not readonly:
                raise RuntimeError(f"{path}: already opened read-only in this process; "
                                   "cannot open it for writing. Close the reader first.")
            # A read request can share an existing handle (writable or not):
            # lmdb forbids a second environment on the same path anyway.
            existing._refcount += 1
            if not readonly and map_size > existing.map_size:
                existing.env.set_mapsize(math.ceil(map_size))
                existing.map_size = math.ceil(map_size)
            return existing
        return cls(path, readonly=readonly, lock=lock, compressed=compressed,
                   compression_level=compression_level, map_size=map_size, batch_size=batch_size)

    def __init__(self, path, readonly=True, lock=False, compressed=False, compression_level=0, map_size=1 << 44, batch_size=50000):
        self.env = lmdb.open(
            path,
            subdir=os.path.isdir(path),
            readonly=readonly,
            lock=lock,
            readahead=True,
            meminit=False,
            max_readers=1,
            map_size=map_size
        )
        self.map_size = map_size
        self.path = path
        self.readonly = readonly
        self.compressed = compressed
        self.compression_level = compression_level
        self.batch_size = batch_size
        self.batch_buffer = []
        self.compression_attempts = 100  # Transient zstd/zlib failures are retried before giving up
        self.compression_backend = COMPRESSION_BACKEND
        self._refcount = 1
        LMDBDataset._registry[os.path.abspath(path)] = self
        with self.env.begin(write=False) as txn:
            try:
                self.length = int(txn.get(b"__len__").decode())
            except Exception:
                self.length = 0

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def __len__(self):
        return self.length

    def write(self, obj, key=None, batched=False):
        """
        "Obj" can be:
            * A byte object, inserted directly
            * Any other type, orjson serialized (e.g. python dictionaries)
        If batched==True, obj is a list and its items are written in order.

        "key" is honored: data is written at str(key) and length advances
        accordingly. Without a key, items are appended after the current
        length. Errors (e.g. full map) raise.
        """
        if batched and key is not None:
            raise ValueError("LMDBDataset.write: a key must not be provided when data is batched")

        batch = obj if isinstance(obj, list) else [obj]
        if len(batch) > self.batch_size:
            raise ValueError(f"Batch size {len(batch)} exceeds maximum {self.batch_size}")

        for offset, data in enumerate(batch):
            item_key = self.length + offset if key is None else int(key) + offset
            if not isinstance(data, bytes):
                data = orjson.dumps(data)
            if self.compressed:
                data = compress_func(data, self.compression_level)
            self.batch_buffer.append((str(item_key).encode(), data))
            if item_key >= self.length:
                self.length = item_key + 1

            if len(self.batch_buffer) >= self.batch_size:
                self._flush_batch()

    def write_batch_with_keys(self, key_data_pairs):
        """
        Write multiple (key, data) pairs using the existing batch buffer
        mechanism (auto-flushing on batch_size). The tracked length is kept
        consistent so that close() persists a correct __len__.
        """
        for key, data in key_data_pairs:
            if not isinstance(data, bytes):
                data = orjson.dumps(data)
            if self.compressed:
                data = compress_func(data, self.compression_level)

            self.batch_buffer.append((str(key).encode(), data))
            if key >= self.length:
                self.length = key + 1

            if len(self.batch_buffer) >= self.batch_size:
                self._flush_batch()

    def _flush_batch(self):
        if not self.batch_buffer:
            return
        with self.env.begin(write=True) as txn:
            for key, data in self.batch_buffer:
                txn.put(key, data)
        self.batch_buffer.clear()

    def close(self):
        self._flush_batch()  # Flush remaining items
        self._refcount -= 1
        if self._refcount > 0:
            return  # still in use by another holder in this process
        key = os.path.abspath(self.path)
        if LMDBDataset._registry.get(key) is self:
            del LMDBDataset._registry[key]
        if not self.readonly:
            with self.env.begin(write=True) as txn:
                txn.put(b"__len__", str(self.length).encode())  # Synchronizing correct length
            self.env.sync()
        self.env.close()

    def __getitem__(self, key):
        raw_key = str(key).encode()
        with self.env.begin(write=False) as txn:
            data = txn.get(raw_key)
            if data is None:
                raise IndexError(f"Index {key} not found in LMDB dataset.")
            if not self.compressed:
                return data
            try:
                return decompress_func(data)
            except Exception:
                return self._decompress_with_retries(raw_key, data)

    def _decompress_with_retries(self, raw_key, data):
        """Guard against the known rare transient zstd/zlib decompression bug.

        Retries the decompression a fixed number of times; if it never
        succeeds, returns None so that training is not broken (the event is
        logged to a local file).
        """
        safe_path = ''.join(c if c.isalnum() or c in '_-' else '_' for c in self.path)
        log_file = f"compression_error_logs_{safe_path}.txt"
        print(f"LMDBDataset {self.path} WARNING: error loading data at index {raw_key}.")
        print(f"It seems there is a bug with {self.compression_backend}.")
        print(f"First, let's try to load the data again {self.compression_attempts} times.")
        for attempt in range(1, self.compression_attempts + 1):
            try:
                recovered = decompress_func(data)
                print(f"It worked at attempt: {attempt}")
                with open(log_file, "a") as l:
                    l.write(f"{self.compression_backend} error at {attempt}. Recovered after {attempt}/{self.compression_attempts} attempts.\n")
                return recovered
            except Exception:
                pass
        print(f"It didn't work after {self.compression_attempts} attempts. Returning None to avoid breaking training. Event logged in ./{log_file}")
        with open(log_file, "a") as l:
            l.write(f"{self.compression_backend} error at {raw_key}. Not recovered after {self.compression_attempts} attempts.\n")
        return None
