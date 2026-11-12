
"""
A Matformer hook to store activations into a database.
This can be very useful for things such as mechanistic interpretability studies or model debug. 

"""

import json
from dataclasses import dataclass
import numpy as np
import pyarrow as pa
import duckdb
import torch
import copy
from pathlib import Path
from matformer.matformer_registry import registry
from matformer.dataclasses.tensors_dataclasses import PaddedTensor, UnpaddedTensor,NormalTensor
from matformer.plugins.plugin_classes import Plugin, PluginConfig
from collections import defaultdict
#TOGLIERE
from transformers import AutoTokenizer
import atexit
@registry.register_dataclass("plugin", "activation_cache", default=True)
@dataclass
class ActivationCacheConfig(PluginConfig):
    move_to_cpu: bool = False
@registry.register("plugin", "activation_cache", "default", requires=["torch"])
class ActivationCache(Plugin):
    def __init__(self, config, model_config, cache):
        super().__init__(config, model_config, cache)
        self.config = config
        self.cache = cache
        self.move_to_cpu = config.move_to_cpu
        self.cache.storage.setdefault('activation_cache', {})
    def forward(self, x, *args, **kwargs):
        self.cache.storage['activation_cache'][self.socket.address] = copy.deepcopy(x).to('cpu') if self.move_to_cpu else x
        return None

import atexit, json
from dataclasses import dataclass, asdict

import torch

from matformer.dataclasses.tensors_dataclasses import PaddedTensor, UnpaddedTensor
from matformer.plugins.plugin_classes import Plugin, PluginConfig


@dataclass
class ActivationStoreBaseConfig(PluginConfig):
    filename: str = "activations"
    databank: str = "default"     # which set of tables to read/write
    flush_every: int = 8          # batches per write
    preload: bool = True          # read the needed addresses into RAM once, then serve from memory
    readonly: bool = False        # True for replay-only runs: no writer lock, so several processes can read the same file


def _samples(tdc):
    """Per-sample CPU views of ONE device->host copy, padding removed."""
    t = tdc.tensor.detach().cpu()
    if isinstance(tdc, UnpaddedTensor):
        cs = tdc.cu_seqlens.cpu().tolist()
        return [t[a:b] for a, b in zip(cs, cs[1:])]
    if isinstance(tdc, PaddedTensor):
        return [r[k] for r, k in zip(t, ~tdc.padding_mask.cpu())]
    return [t]   # NormalTensor


class ActivationStore(Plugin):
    """Backend-independent activation store. Subclasses implement the `_primitives` below."""

    def __init__(self, config, model_config=None, cache=None):
        super().__init__(config, model_config, cache)
        self.cache, self.flush_every, self.preload = cache, config.flush_every, config.preload
        self._preloaded, self._positions, self._closed = {}, {}, False
        self._open(config)
        # Stamp the model config once; an existing store is never overwritten
        if model_config is not None and self._load_model_config() is None:
            cfg = model_config.asdict() if hasattr(model_config, "asdict") else asdict(model_config)
            self._save_model_config(json.dumps(cfg, default=str))
        self._activate(config.databank)
        atexit.register(self.close)     # the unflushed tail must never be lost

    # ---------- backend primitives ----------
    def _open(self, config): raise NotImplementedError
    def _close(self): raise NotImplementedError
    def _select_databank(self, name): raise NotImplementedError        # create if missing
    def list_databank(self): raise NotImplementedError
    def _count(self): raise NotImplementedError                        # next free sequence id
    def _write(self, seq_rows, act_rows): raise NotImplementedError    # (id, meta_json|None), (sid, address, dtype, shape, raw)
    def _read(self, addresses, ids=None): raise NotImplementedError    # {address: {sid: tensor}}
    def available_addresses(self): raise NotImplementedError
    def metadata(self, ids): raise NotImplementedError
    def _load_model_config(self): raise NotImplementedError            # json str or None
    def _save_model_config(self, text): raise NotImplementedError

    # ---------- databanks ----------
    def _activate(self, name):
        self._select_databank(name)
        self.databank, self.next_id = name, self._count()
        self.n_batches, self.seq_rows, self.act_rows = 0, [], []
        self.position = self._positions.get(name, 0)

    def switch_databank(self, name, rewind=False):
        """Make `name` the active databank (created empty if new). Each databank remembers its own replay
        position; rewind=True restarts it from sequence 0 (use it for evaluation)."""
        if name != self.databank:
            self.flush()
            self._positions[self.databank] = self.position
            self._activate(name)
        if rewind:
            self.position = 0

    def __len__(self):
        return self.next_id

    # ---------- writing ----------
    @staticmethod
    def _encode(sid, address, row):
        row = row.contiguous()
        raw = row.flatten().view(torch.uint8).numpy().tobytes()      # native dtype bytes (bfloat16 ok)
        return (sid, address, str(row.dtype), list(row.shape), raw)

    def forward(self, x, *args, **kwargs):
        acts = self.cache.storage.get('activation_cache', {})
        metas = self.cache.storage.get('metadata_cache')
        if not acts:
            return None
        ids = range(self.next_id, self.next_id + x.batch_size)
        self.seq_rows += [(i, json.dumps(metas[k]) if metas else None) for k, i in enumerate(ids)]
        for address, tdc in acts.items():
            self.act_rows += [self._encode(i, address, row) for i, row in zip(ids, _samples(tdc))]
        self.next_id += x.batch_size
        self.n_batches += 1
        if self.n_batches % self.flush_every == 0:
            self.flush()
        return None

    def flush(self):
        if not (self.seq_rows or self.act_rows):
            return
        self._write(self.seq_rows, self.act_rows)
        self.seq_rows, self.act_rows = [], []
        self._preloaded = {k: v for k, v in self._preloaded.items() if k[0] != self.databank}   # stale now

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.flush()
        self._close()

    # ---------- reading ----------
    @staticmethod
    def _decode(dtype, shape, blob):
        dtype = getattr(torch, dtype.removeprefix("torch."))          # "torch.bfloat16" -> torch.bfloat16
        return torch.frombuffer(bytearray(blob), dtype=dtype).reshape(shape)

    def fetch_all(self, ids, addresses=None):
        """Fetch sequences `ids` -> ({address: PaddedTensor}, n_fetched). `addresses`: str, list, or None (all)."""
        self.flush()                                                   # make pending rows visible
        ids = [i for i in ids if i < len(self)]
        addresses = [addresses] if isinstance(addresses, str) else (addresses or self.available_addresses())
        addresses = tuple(sorted(addresses))
        if not ids:
            return {}, 0
        if self.preload:
            key = (self.databank, addresses)
            if key not in self._preloaded:
                self._preloaded[key] = self._read(addresses)          # one scan of the whole databank
            data = self._preloaded[key]
        else:
            data = self._read(addresses, ids)
        for a in addresses:
            missing = set(ids) - data.get(a, {}).keys()
            if missing:
                raise KeyError(f"Address '{a}' is missing {len(missing)} of the {len(set(ids))} requested sequences")
        return {a: PaddedTensor.from_rows([data[a][i] for i in ids]) for a in addresses}, len(ids)

    def fetch(self, address, ids):
        return self.fetch_all(ids, address)[0][address]

    def load_model_config(self):
        text = self._load_model_config()
        return json.loads(text) if text else None


def convert(src, dst, databank="default", step=256):
    """Copy one databank between stores of any backend (e.g. old DuckDB file -> new LMDB file)."""
    dst.switch_databank(databank)
    addrs = tuple(src.available_addresses())
    for s in range(0, len(src), step):
        ids = list(range(s, min(s + step, len(src))))
        data = src._read(addrs, ids)
        metas = src.metadata(ids)
        dst._write([(i, json.dumps(m) if m else None) for i, m in zip(ids, metas)],
                   [dst._encode(i, a, data[a][i]) for a in addrs for i in ids])
    dst.next_id = dst._count()
    if dst.load_model_config() is None and src._load_model_config():
        dst._save_model_config(src._load_model_config())

#DUCKDB
import re
from dataclasses import dataclass
from collections import defaultdict

import duckdb
import pyarrow as pa

from matformer.matformer_registry import registry
#from matformer.activation_store import ActivationStore, ActivationStoreBaseConfig   # adjust the import path


@registry.register_dataclass("plugin", "activation_store_duckdb")
@dataclass
class ActivationDuckDBStoreConfig(ActivationStoreBaseConfig):
    filename: str = "activations.duckdb"
    threads: int = 23
    memory_limit: str = "350GB"


@registry.register_dataclass("activation_store", "duckdb", default=True)
@dataclass
class ActivationStoreConfig(ActivationDuckDBStoreConfig):
    pass    # distinct class, so it gets its own _matformer_category/_variant


@registry.register("activation_store", "duckdb", "default", requires=["torch", "duckdb", "pyarrow"])
@registry.register("plugin", "activation_store_duckdb", "default", requires=["torch", "duckdb", "pyarrow"])
class ActivationDuckDBStore(ActivationStore):
    def _open(self, config):
        self.readonly = config.readonly
        # read_only connections take a shared lock: several replay processes may open the same
        # file at once, but (like LMDB readonly) nothing can be created nor written.
        self.con = duckdb.connect(config.filename, read_only=config.readonly)
        for q in (f"PRAGMA threads={config.threads}",
                  f"PRAGMA memory_limit='{config.memory_limit}'",
                  "SET preserve_insertion_order=true"):
            self.con.execute(q)
        if not config.readonly:
            self.con.execute("CREATE TABLE IF NOT EXISTS model_config (config JSON)")

    def _close(self):
        self.con.close()

    def _select_databank(self, name):
        # "default" keeps the original table names, so old DBs open unchanged
        suffix = "" if name == "default" else "__" + re.sub(r"\W", "_", name)
        self.t_act, self.t_seq = f"activation_store{suffix}", f"sequence{suffix}"
        if self.readonly:
            return    # the databank must already exist: a read-only store cannot create it
        self.con.execute(f"CREATE TABLE IF NOT EXISTS {self.t_seq} (id BIGINT, meta JSON)")
        self.con.execute(f"CREATE TABLE IF NOT EXISTS {self.t_act} "
                         "(sequence_id BIGINT, address VARCHAR, dtype VARCHAR, shape INTEGER[], data BLOB)")

    def list_databank(self):
        names = [r[0] for r in self.con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'activation_store%'").fetchall()]
        return ["default" if n == "activation_store" else n.removeprefix("activation_store__") for n in names]

    def _count(self):
        return self.con.execute(f"SELECT coalesce(max(id) + 1, 0) FROM {self.t_seq}").fetchone()[0]

    def _write(self, seq_rows, act_rows):
        for table, rows, cols in ((self.t_seq, seq_rows, ["id", "meta"]),
                                  (self.t_act, act_rows, ["sequence_id", "address", "dtype", "shape", "data"])):
            if rows:
                self.con.register("_tmp", pa.table(dict(zip(cols, zip(*rows)))))
                self.con.execute(f"INSERT INTO {table} SELECT * FROM _tmp")
                self.con.unregister("_tmp")

    def _read(self, addresses, ids=None):
        where, params = "address IN (SELECT unnest(?))", [list(addresses)]
        if ids is not None:
            where, params = "sequence_id IN (SELECT unnest(?)) AND " + where, [list(set(ids))] + params
        out = defaultdict(dict)
        for address, sid, dtype, shape, blob in self.con.execute(
                f"SELECT address, sequence_id, dtype, shape, data FROM {self.t_act} WHERE {where}", params).fetchall():
            out[address][sid] = self._decode(dtype, shape, blob)
        return out

    def available_addresses(self):
        return [r[0] for r in self.con.execute(f"SELECT DISTINCT address FROM {self.t_act}").fetchall()]

    def metadata(self, ids):
        ids = list(ids)
        by_id = dict(self.con.execute(f"SELECT id, meta FROM {self.t_seq} WHERE id IN (SELECT unnest(?))", [ids]).fetchall())
        return [json.loads(by_id[i]) if by_id[i] else None for i in ids]

    def _load_model_config(self):
        row = self.con.execute("SELECT config FROM model_config").fetchone()
        return row[0] if row else None

    def _save_model_config(self, text):
        self.con.execute("INSERT INTO model_config VALUES (?)", [text])        


# lmdb
import json, re, struct
from collections import defaultdict
from dataclasses import dataclass

import lmdb

from matformer.matformer_registry import registry
#from matformer.activation_store import ActivationStore, ActivationStoreBaseConfig


@registry.register_dataclass("plugin", "activation_store_lmdb")
@dataclass
class ActivationLMDBStoreConfig(ActivationStoreBaseConfig):
    filename: str = "activations.lmdb"
    preload: bool = False         # point reads are cheap: no need to hold the databank in RAM
    map_size: int = 1 << 42       # max size in bytes (4 TB); sparse, so it costs nothing until used


@registry.register_dataclass("activation_store", "lmdb")
@dataclass
class ActivationStoreLMDBConfig(ActivationLMDBStoreConfig):
    pass


def _key(sid, address=b""):
    return struct.pack(">Q", sid) + address           # big-endian: byte order == numeric order


@registry.register("activation_store", "lmdb", "default", requires=["torch", "lmdb"])
@registry.register("plugin", "activation_store_lmdb", "default", requires=["torch", "lmdb"])
class ActivationLMDBStore(ActivationStore):
    """Layout: one sub-db per databank and kind. act__<name>: key = sid(8B) + address, value = header + raw bytes.
    seq__<name>: key = sid, value = meta json. model_config: the stamped model config."""

    def _open(self, config):
        self.readonly = config.readonly
        self.env = lmdb.open(config.filename, subdir=False, readonly=config.readonly, max_dbs=256,
                             map_size=config.map_size, readahead=False, meminit=False)
        self.cfg_db = self.env.open_db(b"model_config", create=not config.readonly)

    def _close(self):
        if not self.readonly:                 # a read-only env has nothing to flush and sync() would fail
            self.env.sync()
        self.env.close()

    def _select_databank(self, name):
        name = re.sub(r"\W", "_", name)
        create = not self.readonly
        self.act_db = self.env.open_db(f"act__{name}".encode(), create=create)
        self.seq_db = self.env.open_db(f"seq__{name}".encode(), create=create)

    def list_databank(self):
        with self.env.begin() as txn:                  # sub-db names are the keys of the main db
            return [k.decode().removeprefix("act__") for k, _ in txn.cursor() if k.startswith(b"act__")]

    def _count(self):
        with self.env.begin(db=self.seq_db) as txn:
            cur = txn.cursor()
            return struct.unpack(">Q", cur.key())[0] + 1 if cur.last() else 0

    # value = 2-byte header length + json [dtype, shape] + raw bytes
    @staticmethod
    def _pack(dtype, shape, raw):
        head = json.dumps([dtype, shape]).encode()
        return struct.pack(">H", len(head)) + head + raw

    def _unpack(self, value):
        n = struct.unpack(">H", value[:2])[0]
        dtype, shape = json.loads(bytes(value[2:2 + n]))
        return self._decode(dtype, shape, value[2 + n:])      # _decode copies: the buffer dies with the txn

    def _write(self, seq_rows, act_rows):
        seq = sorted((_key(i), (m or "null").encode()) for i, m in seq_rows)
        act = sorted((_key(sid, a.encode()), self._pack(dt, shape, raw)) for sid, a, dt, shape, raw in act_rows)
        with self.env.begin(write=True) as txn:               # one atomic transaction per flush
            txn.cursor(self.seq_db).putmulti(seq)
            txn.cursor(self.act_db).putmulti(act)

    def _read(self, addresses, ids=None):
        out = defaultdict(dict)
        with self.env.begin(db=self.act_db, buffers=True) as txn:
            if ids is None:                                    # full scan; values of other addresses are never touched
                wanted = set(addresses)
                for k, v in txn.cursor():
                    address = bytes(k[8:]).decode()
                    if address in wanted:
                        out[address][struct.unpack(">Q", k[:8])[0]] = self._unpack(v)
            else:                                              # point lookups
                for sid in set(ids):
                    for a in addresses:
                        v = txn.get(_key(sid, a.encode()))
                        if v is not None:
                            out[a][sid] = self._unpack(v)
        return out

    def available_addresses(self):
        with self.env.begin(db=self.act_db, buffers=True) as txn:
            cur = txn.cursor()
            if not cur.first():
                return []
            first, addresses = bytes(cur.key()[:8]), []
            for k in cur.iternext(values=False):
                if bytes(k[:8]) != first:
                    break
                addresses.append(bytes(k[8:]).decode())
            return addresses

    def metadata(self, ids):
        with self.env.begin(db=self.seq_db) as txn:
            return [json.loads(txn.get(_key(i)) or b"null") for i in ids]

    def _load_model_config(self):
        with self.env.begin(db=self.cfg_db) as txn:
            v = txn.get(b"config")
            return v.decode() if v else None

    def _save_model_config(self, text):
        with self.env.begin(db=self.cfg_db, write=True) as txn:
            txn.put(b"config", text.encode())        