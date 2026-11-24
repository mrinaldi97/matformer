"""MDat: a single dataset — raw data (LMDB stores) + pretokenization strategies.

An MDat can live in two ways:

* **internal** to a container (MDatContainer): its folder is
  ``<container>/datasets/<name>`` and all its rows live in the container
  catalog;
* **standalone**: ``MDat.create_standalone(path)`` makes a self-contained
  dataset folder (``mdat.db`` + ``datasets/`` + ``pretok/`` + ``shuffling/``).
  Physically the folder is a minimal container holding exactly this one
  dataset ("its own stupid container"), so a standalone dataset is immediately
  shufflable, viewable and trainable on its own; at the same time any number
  of other containers can ``link_dataset()`` it.

When a container loads a linked dataset it builds an MDat bound to the host
container, but reading *configuration* (storage databases, strategy stats)
goes through the dataset's own catalog (``config_db``/``config_id``), while
``submdat_path``/``pretok_base_path`` point inside the external folder.

A *generation* counter is bumped whenever the document set changes; adding a
pretokenization strategy does NOT bump it (that is harmless for containers
that already link the dataset).
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import uuid as uuid_lib
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

try:
    from tqdm import tqdm
except:
    pass
try:
    import orjson
except:
    orjson = None

from .lmdb_store import LMDBDataset
from .mdat_common import (
    MDatIsReadOnly,
    RandomPermutator,
    SubMdatAlreadyExists,
    SubMdatMissesStrategy,
    SubmDatNotFound,
    sanify_name,
)
from .preprocessing import AtlasIterator, HuggingFaceIterator, JSONIterator, LMDBIterator
from .strategy import PretokenizationStrategy, process_documents_batch

# Strategy attachment defaults
DEFAULT_PRETOK_MAP_SIZE = 1 << 41
DEFAULT_BATCH_SIZE = 50000
# How many chunk multipliers get their iteration length precomputed
MAX_PRECOMPUTED_MULTIPLIER = 100


class MDat:
    def __init__(self) -> None:
        """Initialize an (unbound) MDat. Use load_submdat / from_external /
        create_submdat to obtain a bound instance."""
        self.current_strategy = None
        self.default_wanted_from_dbs = None
        self.default_wanted_from_strategy = None
        self.storage_db = dict()  # pointers to the LMDB storage databases
        self.default_map_size = 1 << 33
        self.linked = False
        self.default_raw = False
        self.default_add_special_tokens = True

    # ------------------------------------------------------------- binding
    def _bind(self, container: 'MDatContainer', submdat_name: str,
              folder_name: str = None, storage_root: str = None,
              config_db=None, config_id: int = None, row: Dict[str, Any] = None) -> None:
        """Bind this MDat to its host container (and, for linked datasets, to
        the external folder/catalog where its bytes and rows actually live)."""
        from .container import MDatContainer
        if not isinstance(container, MDatContainer):
            raise TypeError(f"container must be an MDatContainer, got {type(container)}")
        self.container = container
        self.db = container.db
        self.mdat_path = container.mdat_path  # host container root
        self.readonly = container.readonly
        self.submdat_name = submdat_name  # local name (alias) inside the container
        self.folder_name = folder_name or submdat_name  # folder name on disk
        self.storage_root_path = storage_root or container.mdat_path
        self.submdat_path = os.path.join(self.storage_root_path, container.DATASETS_DIR, self.folder_name)
        self.pretok_base_path = os.path.join(self.storage_root_path, container.PRETOK_DIR)
        self.config_db = config_db or container.db
        self.config_id = config_id
        self.row = row if row is not None else {}

    @property
    def mdat(self):
        """Deprecated alias for .container."""
        return self.container

    def __str__(self):
        info = f"MDat: {self.submdat_name}, Documents: {self.len}"
        if self.current_strategy is not None:
            strategy_stats = self.config_db.get_strategy_submdats(self.current_strategy.strategy_name, self.config_id)
            if strategy_stats:
                info += f" Tokens: {strategy_stats['total_tokens']} Chunks: {strategy_stats['total_chunks']}"
        gaps = self.desynced_strategies() if isinstance(getattr(self, 'config_id', None), int) else []
        if gaps:
            info += f" DESYNCED: {gaps} (run sync_strategy)"
        return info

    # NOTE: load_submdat/create_submdat serve BOTH the standalone datasets
    # (a dataset inside its own mini-container) and the legacy internal
    # datasets of a shared container; only the latter are BACKWARD-COMPAT.
    @classmethod
    def load_submdat(cls, container: 'MDatContainer', submdat_name: str) -> 'MDat':
        """
        Load an existing internal dataset of a container.
        Args:
            container: host MDatContainer
            submdat_name: name of the dataset
        """
        instance = cls()
        instance._bind(container, submdat_name)
        if instance.submdat_name not in instance.container.list_submdat():
            raise SubmDatNotFound(f"Dataset '{submdat_name}' is not registered in the container")
        if not os.path.isdir(instance.submdat_path):
            # A container pretokenized on faster storage often keeps only `pretok/` where it is trained:
            # the raw folders are gone, and nothing needs them once a strategy serves the documents. That
            # is a legal state, not a broken one -- with neither raw nor pretokenized bytes, it is.
            if not instance.pretokenized_strategies():
                raise FileNotFoundError(f"{instance.submdat_name} is registered but its folder is missing at {instance.submdat_path}.")
            instance.raw_missing = True
        else:
            instance.raw_missing = False
        instance.row = instance.config_db.get_submdat_row(submdat_name)
        instance.submdat_id = instance.config_db.get_dsmap(key='name', value=submdat_name)
        instance.config_id = instance.submdat_id
        instance.len = instance.row['document_number'] or 0
        instance.set_default_wanted()
        instance._load_storage_dbs()
        return instance

    @classmethod
    def from_external(cls, container: 'MDatContainer', alias: str, row: Dict[str, Any],
                      external_root: str, external_db, external_name: str,
                      external_container=None) -> 'MDat':
        """
        Build an MDat for a *linked* dataset: bytes and config come from the
        external standalone folder; the local (aliased) bookkeeping belongs to
        the host container.
        """
        instance = cls()
        instance._bind(container, alias,
                       folder_name=external_name, storage_root=external_root,
                       config_db=external_db, config_id=row['ds_map_id'], row=row)
        instance.submdat_id = container.db.get_dsmap(key='name', value=alias)
        instance.linked = True
        instance.external_container = external_container
        instance.external_name = external_name
        instance.len = row['document_number'] or 0
        instance.set_default_wanted()
        instance._load_storage_dbs()
        return instance

    # ------------------------------------------------------ identity & staleness
    @property
    def uuid(self) -> str:
        """Stable identity of this dataset (generated on first access)."""
        row = self.config_db.get_submdat_row(self.config_id)
        identity = row.get('uuid')
        if not identity:
            identity = uuid_lib.uuid4().hex
            self.config_db.update_submdat_row(self.config_id, {'uuid': identity})
        return identity

    @property
    def generation(self) -> int:
        """Bumped every time the document set changes (never by strategies)."""
        row = self.config_db.get_submdat_row(self.config_id)
        return int(row.get('generation') or 0)

    def _bump_generation(self) -> int:
        return self.config_db.bump_generation(self.config_id)

    def _after_data_change(self):
        """Refresh in-memory length and every bookkeeping copy after ingestion."""
        self._bump_generation()
        if self.linked:
            live = self.config_db.get_submdat_row(self.config_id)
            self.container.db.update_submdat_row(self.submdat_id, {
                'document_number': live.get('document_number') or 0,
                'generation': live.get('generation') or 0,
            })
            ext = getattr(self, 'external_container', None)
            if ext is not None:
                self.container._mirror_linked_dataset(self.submdat_id, ext, self.external_name)
        self.container._update_default_view(self.submdat_id)
        # Appended rows lengthened the default view: any shuffle file and any
        # distributed precompute that includes this dataset is now stale.
        on_change = getattr(self.container, '_on_dataset_changed', None)
        if on_change is not None:
            on_change(self.submdat_id)

    # ----------------------------------------------------------- storage dbs
    def _check_writable(self):
        if self.readonly or (self.linked and self.container.readonly):
            raise MDatIsReadOnly(f"Dataset '{self.submdat_name}' is read-only"
                                 + (" (linked dataset in a read-only container)" if self.linked else ""))

    def _open_lmdb(self, path, comp_level, map_size, b_size=None, create=False):
        return LMDBDataset.open(
            path,
            compressed=comp_level > 0,
            readonly=False if create else self.readonly,
            compression_level=comp_level,
            map_size=math.ceil(map_size),
            batch_size=b_size or DEFAULT_BATCH_SIZE,
        )

    def _load_map_size(self, db_info):
        """Stored dbs are opened with a map size 20% larger than their current
        disk size (falls back to the default for empty/unknown ones). During an
        ingestion a larger, input-sized map can be forced."""
        info = self.config_db.get_storage_db_info(db_info['id'])
        disk_size = info.get('disk_size')
        declared = int(info.get('map_size') or 0)
        if not isinstance(disk_size, int) or disk_size <= 1:
            size = self.default_map_size
        else:
            size = disk_size * 1.2
        # Never shrink below the map size the database was created with: the
        # on-disk file size is smaller than it (pages are added lazily), and an
        # append/pretokenization must still fit.
        size = max(size, declared)
        forced = getattr(self, '_forced_map_size', None)
        return max(size, forced) if forced else size

    def pretokenized_strategies(self) -> list:
        """Strategies that hold a pretokenized copy of this dataset on disk."""
        base = self.pretok_base_path
        if not os.path.isdir(base):
            return []
        return [one for one in sorted(os.listdir(base))
                if os.path.isdir(os.path.join(base, one, self.folder_name))]

    def _load_storage_dbs(self, batch_size=None):
        """Open all the storage databases ('data', 'meta', extras) of this dataset. Ones that are not on
        disk are skipped rather than fatal: a dataset served by a pretokenization strategy has no use
        for the raw bytes it was tokenized from."""
        for dbs in self.config_db.get_submdat_databases(self.config_id):
            db_type = _role_of(dbs)
            db_path = os.path.join(self.submdat_path, db_type) + '.dat'
            # LMDB stores can be a file or a directory depending on how they were
            # created (subdir=os.path.isdir at open time), so test existence, not dir-ness.
            if not os.path.exists(db_path):
                self.missing_storage = getattr(self, 'missing_storage', []) + [db_type]
                continue
            self.storage_db[db_type] = self._open_lmdb(db_path, dbs['compression_level'],
                                                       self._load_map_size(dbs), b_size=batch_size)

    def _create_storage_db(self, db_type, compression_level, map_size=None, disk_size=None, extra_data=None, batch_size=None):
        """Create one storage database ('data'/'meta'/extra) and register it."""
        path = os.path.join(self.submdat_path, db_type) + '.dat'
        size = map_size or self.default_map_size
        db_pointer = self._open_lmdb(path, compression_level, size, b_size=batch_size, create=True)
        db_id = self.config_db.add_database(_type='LMDB', compression_level=compression_level,
                                            map_size=size, disk_size=disk_size, extra_data=extra_data)
        return db_pointer, db_id

    def _load_pretok_dbs(self, strategy, batch_size=None):
        """Open the pretokenization output databases of `strategy`."""
        self.pretok_db = {}
        for sdb in self.config_db.get_strategy_dbs(strategy, self.config_id):
            db_name = _pretok_role_of(sdb)
            db_path = os.path.join(self.pretok_base_path, strategy, self.folder_name, db_name + '.dat')
            self.pretok_db[db_name] = self._open_lmdb(db_path, sdb['compression_level'],
                                                      self._load_map_size(sdb), b_size=batch_size)

    def _create_pretok_dbs(self, strategy, compression_level, map_size, batch_size=None):
        """Create the pretokenization output databases declared by `strategy`."""
        if map_size is None:
            raise ValueError("map_size must be specified when creating a new strategy's storage DB")
        pretok_folder = os.path.join(self.pretok_base_path, strategy, self.folder_name)
        os.makedirs(pretok_folder, exist_ok=True)
        self.pretok_db = {}
        strategy_row = self.config_db.get_strategy_row(strategy)
        db_names = json.loads(strategy_row['required_databases']) if isinstance(strategy_row.get('required_databases'), str) else (strategy_row.get('required_databases') or [])
        for db_name in db_names:
            db_path = os.path.join(pretok_folder, db_name + '.dat')
            self.pretok_db[db_name] = self._open_lmdb(db_path, compression_level, map_size, b_size=batch_size, create=True)
            db_id = self.config_db.add_database(_type='LMDB', compression_level=compression_level,
                                                map_size=map_size, disk_size=None, extra_data=None)
            is_tokens = 1 if db_name == 'tokens' else 0
            is_chunks = 1 if db_name == 'chunks' else 0
            is_extra = json.dumps(db_name) if db_name not in ('tokens', 'chunks') else None
            self.config_db.add_strategy_db(database_id=db_id, strategy_name=strategy, submdat_id=self.config_id,
                                           datatype=None, is_tokens=is_tokens, is_chunks=is_chunks,
                                           is_extra=is_extra, is_complete=0)

    def _close_storage_dbs(self, _type='storage'):
        if _type != 'storage':
            return
        for name, db in self.storage_db.items():
            print("Closed db: ", name)
            db.close()
        self.storage_db = {}

    # -------------------------------------------------------------- creation
    @classmethod
    def create_submdat(cls, container: 'MDatContainer', submdat_name: str,
                       compression_levels: Dict[str, int] = None, map_sizes: Dict[str, int] = None,
                       overwrite: bool = False, db_types: List[str] = ['meta', 'data'],
                       data_type: str = 'text', data_key: str = 'text',
                       modality: str = 'text',
                       hybrid_data: List = []) -> 'MDat':
        """
        Create a new internal dataset inside `container` and load it.
        Args:
            container: host MDatContainer
            submdat_name: Name of the new dataset
            compression_levels: {"db_name": level} per storage db (default 0)
            map_sizes: {"db_name": map_size} per storage db (default 1<<33)
            db_types: Storage databases to create (default ['meta','data'])
            data_type: Data type
            data_key: Original key of the data field (ex. 'text')
            modality: Modality of the data
        """
        compression_levels = compression_levels or {}
        map_sizes = map_sizes or {}
        instance = cls()
        instance._bind(container, submdat_name)
        if instance.readonly:
            raise MDatIsReadOnly
        if instance.submdat_name in instance.container.list_submdat():
            if instance.container.get_submdat(instance.submdat_name) is not None:  # None is the creation placeholder
                raise SubMdatAlreadyExists(f"Dataset '{submdat_name}' already exists")
        if os.path.isdir(instance.submdat_path):
            raise FileExistsError("Dataset folder is present but not registered in the container")
        instance.submdat_id = instance.config_db.add_submdat(submdat_name)
        instance.config_id = instance.submdat_id

        os.makedirs(instance.submdat_path, exist_ok=True)
        for db_type in db_types:
            try:
                db_pointer, db_id = instance._create_storage_db(
                    db_type, compression_level=compression_levels.get(db_type, 0),
                    map_size=map_sizes.get(db_type))
                is_data = 1 if db_type == 'data' else 0
                is_meta = 1 if db_type == 'meta' else 0
                is_extra = json.dumps(db_type) if db_type not in ('data', 'meta') else None
                instance.config_db.link_submdat_database(instance.config_id, db_id, raw_data_bytes=0,
                                                         is_data=is_data, is_meta=is_meta, is_extra=is_extra)
                instance.storage_db[db_type] = db_pointer
            except Exception as e:
                instance.config_db.remove_submdat(instance.config_id)
                raise e

        manifest = {
            "name": submdat_name,
            "modality": json.dumps(modality),
            "data_key": data_key,
            "document_number": 0,
            "errors_counters": json.dumps({}),
            "data_type": json.dumps(data_type)
        }
        instance.config_db.update_submdat_row(instance.config_id, manifest)
        instance.container._update_default_view(instance.submdat_id)

        # Close the freshly-created environments: the canonical load below
        # re-opens them from the catalog (same process, single set of handles).
        instance._close_storage_dbs()
        return cls.load_submdat(container, submdat_name)

    # ------------------------------------------------------- standalone API
    @classmethod
    def create_standalone(cls, path: str, name: str = None,
                          compression_levels: Dict[str, int] = None, map_sizes: Dict[str, int] = None,
                          db_types: List[str] = ('meta', 'data'), data_type: str = 'text',
                          data_key: str = 'text', modality: str = 'text') -> 'MDat':
        """
        Create a self-contained dataset folder that is *its own* minimal
        container: it can be ingested, pretokenized, shuffled, viewed and
        trained directly, and it can be linked by any number of other
        containers with MDatContainer.link_dataset().
        """
        from .container import MDatContainer
        name = sanify_name(name or os.path.basename(os.path.normpath(os.path.abspath(path))))
        container = MDatContainer.create_new(path)
        return container.add_submdat(name, compression_levels=compression_levels or {},
                                     map_sizes=map_sizes or {}, db_types=list(db_types),
                                     data_type=data_type, data_key=data_key, modality=modality)

    @classmethod
    def open_standalone(cls, path: str, readonly: bool = False) -> 'MDat':
        """Open a dataset folder created with create_standalone()."""
        from .container import MDatContainer
        container = MDatContainer.load_dataset(path, readonly=readonly)
        names = container.list_submdat()
        if len(names) != 1:
            raise ValueError(f"{path} is not a standalone dataset folder (contains {len(names)} datasets)")
        return container[names[0]]

    def shuffle(self, view_name: str = 'default'):
        """(Standalone use) shuffle one of this dataset's own views."""
        return self.container.shuffle(view_name)

    def register_new_view(self, view_dict):
        """(Standalone use) register a view on this dataset's own container."""
        return self.container.register_new_view(view_dict)

    def set_iteration_modality(self, *args, **kwargs):
        """(Standalone use) configure iteration on this dataset's container."""
        return self.container.set_iteration_modality(*args, **kwargs)

    def set_training_strategy(self, strategy_name: str, max_seq_len: int = None):
        """(Standalone use) full container-side strategy setup (totals, multipliers)."""
        return self.container.set_strategy(strategy_name, max_seq_len=max_seq_len)

    def __iter__(self, with_prefetch: bool = True):
        """(Standalone use) iterate this dataset through its own container."""
        return self.container.__iter__(with_prefetch=with_prefetch)

    def __next__(self):
        return self.container.__next__()

    # ------------------------------------------------------------- reading
    def set_default_wanted(self, from_dbs: list = ['full'], from_strategy: list = [], raw: bool = False, add_special_tokens=True):
        """
        Set the default elements returned by __getitem__:
            from_dbs: db names to return; "meta" is parsed as a dictionary,
                      anything else is returned raw; "full" returns everything.
            from_strategy: strategy outputs to return ('tokens', 'chunks',
                         'chunked_tokens', ...); "full" returns all of them.
        """
        self.default_wanted_from_dbs = from_dbs
        self.default_wanted_from_strategy = from_strategy
        self.default_raw = raw
        self.default_add_special_tokens = add_special_tokens

    def __getitem__(self, key):
        return self._compose_return(key=key, wanted_from_dbs=self.default_wanted_from_dbs,
                                    wanted_from_strategy=self.default_wanted_from_strategy,
                                    raw=self.default_raw)

    def raw_bytes(self, key: int) -> bytes:
        """The raw stored document bytes (decompressed), bypassing composition."""
        return self.storage_db['data'][key]

    def _compose_return(self, key: int, wanted_from_dbs: list, wanted_from_strategy: list, raw: bool = False,
                        add_special_tokens: bool = True, strategy_name=None, on_the_fly_mode=True,
                        max_seq_len=None, decode_strings=True):
        composed = {'submdat_name': self.submdat_name, 'key': key}
        wanted_from_dbs = wanted_from_dbs or []
        # 'full' (as a bare string or inside the list) returns meta fields + data
        full = wanted_from_dbs == 'full' or 'full' in wanted_from_dbs
        if full:
            composed.update(orjson.loads(self.storage_db['meta'][key]))
            composed.update({'data': self.storage_db['data'][key]})
        if 'data' in wanted_from_dbs and not full:
            if decode_strings:
                text = self.storage_db['data'][key].decode('utf-8')
            else:
                text = self.storage_db['data'][key]
            if not raw:
                composed.update({'data': text})
            else:
                return text
        if 'meta' in wanted_from_dbs and not full:
            composed.update({'meta': orjson.loads(self.storage_db['meta'][key])})
        if self.current_strategy is not None and wanted_from_strategy:
            composed.update(self.current_strategy(key=key, cache_dict=self.pretok_db, max_seq_len=max_seq_len,
                                                  wanted_from_strategy=wanted_from_strategy,
                                                  add_special_tokens=add_special_tokens))
        return composed

    def get_generator(self, progress_bar=False, wanted_from_dbs='full', wanted_from_strategy=None, raw=False,
                      randomize=False, seed=27, start=0, end=None):
        """
        Generator over the elements of this dataset.
        Args:
            progress_bar: show a tqdm progress bar
            wanted_from_dbs / wanted_from_strategy / raw: composition of each item
                (same semantics as set_default_wanted)
            randomize: iterate in randomized order (deterministic per seed)
            start / end: optional half-open key range (used to pretokenize only
                the rows an append added)
        """
        end = self.len if end is None else min(int(end), self.len)
        start = max(int(start), 0)
        permutator = RandomPermutator(max_len=end - start, seed=seed) if randomize else None
        iterator = tqdm(range(start, end)) if progress_bar else range(start, end)
        for key in iterator:
            if permutator is not None:
                key = start + permutator(key - start)
            yield self._compose_return(key=key, wanted_from_dbs=wanted_from_dbs,
                                       wanted_from_strategy=wanted_from_strategy, raw=raw)

    def __len__(self):
        return self.len

    def get_row(self) -> Dict[str, Any]:
        """The catalog row describing this dataset (its 'manifest')."""
        return self.config_db.get_submdat_row(self.submdat_name if not self.linked else self.config_id)

    def get_current_strategy(self):
        return self.current_strategy

    # ----------------------------------------------------- direct row editing
    def _storage_role_file(self, role: str) -> str:
        return os.path.join(self.submdat_path, role) + '.dat'

    def _persist_storage_state(self, data_delta: int = 0, meta_delta: int = 0) -> None:
        """Refresh disk sizes and the recorded uncompressed byte counts after a
        write/edit/delete. Deltas are accumulated by the caller so that a
        streaming writer pays for one catalog update, not one per row."""
        deltas = {'data': int(data_delta), 'meta': int(meta_delta)}
        for dbs in self.config_db.get_submdat_databases(self.config_id):
            role = _role_of(dbs)
            update = {}
            path = self._storage_role_file(role)
            if os.path.exists(path):
                update['disk_size'] = os.path.getsize(path)
            if role in deltas:
                update['raw_data_bytes'] = max(0, int(dbs.get('raw_data_bytes') or 0) + deltas[role])
            if update:
                self.config_db.update_submdat_database(
                    submdat_id=self.config_id, database_id=dbs['id'], data=update)

    def _invalidate_pretokenization(self) -> None:
        """Edits and deletions change what every stored key means, so every
        strategy must be pretokenized again. Clearing coverage (and the stats
        it is inferred from) makes desynced_strategies() report the full range;
        pretokenize_submdat() rewrites the rows of the new document set."""
        for strategy_name in self.config_db.list_strategies():
            if not self.config_db.get_strategy_submdats(strategy_name, self.config_id):
                continue
            self.config_db.set_strategy_coverage(strategy_name, self.config_id, [])
            self.config_db.clear_strategy_submdat_precomp(strategy_name, self.config_id)
            self.config_db.update_strategy_submdat(strategy_name, self.config_id, {
                'total_tokens': 0, 'max_tokens_per_doc': 0, 'total_chunks': 0,
                'max_chunks_per_doc': 0, 'processed_docs': 0,
            })

    def _encode_record(self, record, data_column):
        """Split one record into (data bytes or None, metadata dict, serialized metadata)."""
        if not isinstance(record, dict):
            raise TypeError(f"An mdat row must be a dict, got {type(record)}")
        payload = dict(record)
        data = None
        if 'data' in self.storage_db and data_column is not None and data_column in payload:
            data = payload.pop(data_column)
            if isinstance(data, str):
                data = data.encode('utf-8')
            if not isinstance(data, bytes):
                raise TypeError(f"Column '{data_column}' must be str or bytes to be stored as raw data, "
                                f"got {type(data)}")
        try:
            serialized = orjson.dumps(payload) if orjson else json.dumps(payload).encode()
        except Exception as exc:
            raise TypeError(f"Metadata for an mdat row is not JSON serializable: {exc}") from exc
        return data, payload, serialized

    def _write_one(self, record, data_column='text', key=None):
        """Write one row without touching the catalog. Returns
        (key, data_delta, meta_delta, is_new) where the deltas are the change in
        uncompressed bytes with respect to the row that was there before."""
        self._check_writable()
        is_new = key is None or int(key) >= self.len
        key = int(self.len) if key is None else int(key)
        if key < 0:
            raise ValueError("Row keys must be non-negative")
        has_data_db = 'data' in self.storage_db
        has_meta_db = 'meta' in self.storage_db
        if not has_data_db and not has_meta_db:
            raise RuntimeError(f"Dataset '{self.submdat_name}' has no writable storage databases")
        if has_data_db and data_column is not None and data_column not in record and is_new:
            raise ValueError(f"Column '{data_column}' is missing from the new row "
                             f"(keys: {sorted(record)})")
        data, payload, serialized = self._encode_record(record, data_column)
        old_data_len = 0
        old_meta_len = 0
        if not is_new:
            if has_data_db:
                try:
                    old_data_len = len(self.storage_db['data'][key])
                except IndexError:
                    old_data_len = 0
            if has_meta_db:
                try:
                    old_meta_len = len(self.storage_db['meta'][key])
                except IndexError:
                    old_meta_len = 0
        if has_data_db and data is not None:
            self.storage_db['data'].write(data, key=key)
        if has_meta_db:
            if payload or has_data_db:
                self.storage_db['meta'].write(serialized, key=key)
        self.len = max(self.len, key + 1)
        return key, (len(data) if data is not None else 0) - old_data_len, \
            (len(serialized) if (payload or has_data_db) else 0) - old_meta_len, is_new

    def commit_storage_changes(self, data_delta: int = 0, meta_delta: int = 0,
                               invalidate_pretokenization: bool = False) -> None:
        """Flush the storage databases and persist length, sizes and (optionally)
        the invalidation that any edit/delete requires."""
        for db in self.storage_db.values():
            db.sync_length()
        self.config_db.update_submdat_row(self.config_id, {'document_number': int(self.len)})
        self._persist_storage_state(data_delta, meta_delta)
        if invalidate_pretokenization:
            self._invalidate_pretokenization()
        self._after_data_change()
        refresh = getattr(self.container, '_refresh_active_view', None)
        if refresh is not None:
            refresh()

    def write_document(self, record, data_column='text', key=None):
        """Write (append or overwrite) one row and persist it immediately.

        Passing a key inside ``[0, len)`` overwrites that document, which breaks
        any pretokenization of this dataset; keys at or beyond ``len`` append.
        Use an MDatWriter for a streaming loop: it batches these catalog writes.
        """
        key, data_delta, meta_delta, is_new = self._write_one(record, data_column, key)
        self.commit_storage_changes(data_delta, meta_delta, invalidate_pretokenization=not is_new)
        return key

    def update_document(self, key, record, data_column='text'):
        """Overwrite a single existing row (breaks pretokenization)."""
        if int(key) >= self.len:
            raise IndexError(f"Row {key} does not exist (len={self.len}); use write_document to append")
        return self.write_document(record, data_column=data_column, key=key)

    def update_rows(self, rows, data_column='text'):
        """Overwrite several rows in one catalog update.

        ``rows`` is a mapping ``{key: record}`` or an iterable of
        ``(key, record)`` pairs. Every key must already exist. Any edit breaks
        pretokenization, so all strategies are invalidated afterwards.
        """
        items = rows.items() if isinstance(rows, dict) else rows
        data_delta = 0
        meta_delta = 0
        keys = []
        for key, record in items:
            key = int(key)
            if key < 0 or key >= self.len:
                raise IndexError(f"Row {key} does not exist (len={self.len})")
            key, row_data_delta, row_meta_delta, _ = self._write_one(record, data_column, key=key)
            data_delta += row_data_delta
            meta_delta += row_meta_delta
            keys.append(key)
        if keys:
            self.commit_storage_changes(data_delta, meta_delta, invalidate_pretokenization=True)
        return keys

    def delete_documents(self, keys, compact=True) -> int:
        """Delete rows and keep keys sequential by default.

        Keys are sequential numbers, so a hole would break iteration. With
        ``compact=True`` every surviving row is shifted down to fill the gap
        (O(n) over the dataset, but no decompression/recompression). Deletion
        always breaks pretokenization: coverages and strategy stats are reset.
        """
        if not compact:
            raise NotImplementedError("Only compact deletion is supported: mdat keys are sequential")
        self._check_writable()
        old_len = int(self.len)
        selected = sorted({int(key) for key in keys})
        selected = [key for key in selected if 0 <= key < old_len]
        if not selected:
            return 0
        data_delta = 0
        meta_delta = 0
        if 'data' in self.storage_db:
            data_delta = -sum(len(self.storage_db['data'][key]) for key in selected)
        if 'meta' in self.storage_db:
            meta_delta = -sum(len(self.storage_db['meta'][key]) for key in selected)
        for db in self.storage_db.values():
            db.delete_many(selected, compact=True)
        self.len = old_len - len(selected)
        self.commit_storage_changes(data_delta, meta_delta, invalidate_pretokenization=True)
        return len(selected)

    # ------------------------------------------------------------- strategies
    def _attach_strategy(self, strategy_name: str, add: bool = False, compression_level: int = 0,
                         map_size: int = DEFAULT_PRETOK_MAP_SIZE, batch_size: int = DEFAULT_BATCH_SIZE):
        """
        Attach a pretokenization strategy to this dataset (used internally by
        the container during iteration and by pretokenize()).
        Args:
            strategy_name: the strategy to attach
            add: if True, register the strategy on this dataset (creating its
                 output databases); the pretokenization itself must follow,
                 ending with add_strategy_end()
        """
        if len(self.config_db.get_strategy_submdats(strategy_name=strategy_name, submdat_id=self.config_id)) == 0:
            if not add:
                raise SubMdatMissesStrategy(f"Dataset '{self.submdat_name}' has no pretokenization '{strategy_name}'")
            self._check_writable()
            print(f"Adding {strategy_name} to {self.submdat_name}")
            self.config_db.add_strategy_submdat(strategy_name, self.config_id)
            pretok_folder = os.path.join(self.pretok_base_path, strategy_name, self.folder_name)
            os.makedirs(pretok_folder, exist_ok=True)
            self.current_strategy = PretokenizationStrategy(
                db=self.config_db, strategy_name=strategy_name,
                pretok_path=self.pretok_base_path, functions_path=self.container.functions_path)
            self._create_pretok_dbs(strategy_name, compression_level=compression_level,
                                    map_size=map_size, batch_size=batch_size)
        else:
            # Load the strategy: prefer the container's already-built object
            container_strategy = self.container.current_strategy
            if container_strategy is not None and container_strategy.strategy_name == strategy_name:
                self.current_strategy = container_strategy
            else:
                self.current_strategy = PretokenizationStrategy(
                    db=self.config_db, strategy_name=strategy_name,
                    pretok_path=self.pretok_base_path, functions_path=self.container.functions_path)
            self._load_pretok_dbs(strategy_name)

    def set_strategy(self, *args, **kwargs):
        """Deprecated alias for _attach_strategy on a dataset object; on a
        standalone dataset prefer set_training_strategy() (container-side)."""
        return self._attach_strategy(*args, **kwargs)

    def add_strategy_end(self, strategy_name: str, stats: Dict[str, Any]):
        """Close the pretokenization databases of a strategy and persist its
        stats, precomputed lengths and disk sizes (into the dataset's own
        catalog)."""
        self._check_writable()
        for db in self.pretok_db.values():
            db.close()
        self._persist_strategy_stats(strategy_name, stats, finalize=True)

        # Record each output database's disk size
        pretok_folder = os.path.join(self.pretok_base_path, strategy_name, self.folder_name)
        for sdb in self.config_db.get_strategy_dbs(strategy_name, self.config_id):
            db_name = _pretok_role_of(sdb)
            db_file = os.path.join(pretok_folder, f"{db_name}.dat")
            if os.path.exists(db_file):
                self.config_db.update_strategy_db(database_id=sdb['db_id'], data={'disk_size': os.path.getsize(db_file)})
            else:
                print(f"Missing pretokenization database: {db_file}")

        if self.linked:
            # Refresh the host container's mirrors of this dataset
            ext = getattr(self, 'external_container', None)
            if ext is not None:
                self.container._mirror_linked_dataset(self.submdat_id, ext, self.external_name)

    def _persist_strategy_stats(self, strategy_name, stats, finalize=False):
        data = {
            'total_tokens': stats.get('total_tokens', 0),
            'max_tokens_per_doc': stats.get('max_tokens_per_doc', 0),
            'total_chunks': stats.get('total_chunks', 0),
            'max_chunks_per_doc': stats.get('max_chunks_per_doc', 0),
            'processed_docs': stats.get('processed_docs', 0),
        }
        if finalize:
            data['finalized_at'] = datetime.utcnow().isoformat()
        self.config_db.update_strategy_submdat(strategy_name, self.config_id, data)
        for multiplier, precomp_len in enumerate(stats.get('precomputed_lengths', [])):
            if precomp_len > 0:
                self.config_db.set_strategy_submdat_precomp(
                    strategy_name, self.config_id, multiplier, precomp_len)

    def _flush_pretok_dbs(self):
        for db in self.pretok_db.values():
            db.sync_length()

    # ------------------------------------------------------------ coverage
    @staticmethod
    def _merge_ranges(ranges):
        """Canonical (sorted, non-overlapping, adjacent-merged) half-open ranges."""
        out = []
        for start, end in sorted((int(s), int(e)) for s, e in ranges if int(e) > int(s)):
            if out and start <= out[-1][1]:
                out[-1] = (out[-1][0], max(out[-1][1], end))
            else:
                out.append((start, end))
        return out

    def _coverage_ranges(self, strategy_name, materialize=False):
        """Rows this strategy has durably pretokenized. Old datasets that predate
        coverage rows are inferred from their finalized stats; materialize=True
        writes that inference to the catalog (needed before extending it)."""
        rows = self.config_db.get_strategy_coverage(strategy_name, self.config_id)
        if rows:
            return self._merge_ranges([(r['start'], r['end']) for r in rows])
        stats = self.config_db.get_strategy_submdats(strategy_name, self.config_id)
        processed = int((stats or {}).get('processed_docs') or 0)
        if processed > 0:
            inferred = [(0, processed)]
            if materialize:
                self.config_db.set_strategy_coverage(strategy_name, self.config_id, inferred)
            return inferred
        return []

    def _missing_ranges(self, strategy_name, total=None, materialize=False):
        total = int(self.len if total is None else total)
        missing, cursor = [], 0
        for start, end in self._coverage_ranges(strategy_name, materialize=materialize):
            if start > cursor:
                missing.append((cursor, min(start, total)))
            cursor = max(cursor, end)
            if cursor >= total:
                break
        if cursor < total:
            missing.append((cursor, total))
        return [(s, e) for s, e in missing if e > s]

    def _add_coverage(self, strategy_name, start, end):
        if end <= start:
            return
        ranges = self._coverage_ranges(strategy_name, materialize=True)
        ranges.append((start, end))
        self.config_db.set_strategy_coverage(strategy_name, self.config_id,
                                             self._merge_ranges(ranges))

    def desynced_strategies(self):
        """Pretokenization gaps introduced by appends: one (strategy, start, end)
        tuple per missing range. Empty when every strategy covers every row."""
        total = int((self.config_db.get_submdat_row(self.config_id) or {}).get('document_number') or 0)
        out = []
        for strategy_name in self.config_db.list_strategies():
            if not self.config_db.get_strategy_submdats(strategy_name, self.config_id):
                continue
            for start, end in self._missing_ranges(strategy_name, total):
                out.append((strategy_name, start, end))
        return out

    # ---------------------------------------------------------- pretokenization
    def pretokenize_submdat(self, strategy_name, progress_bar=True, compression_level=0,
                            chunking_strict_checks=False, parallel=True, num_processes=None,
                            batch_size=256, map_size=1 << 40, disk_size_multiplier=4,
                            checkpoint_every=100_000, resume=False):
        """
        Pretokenize this dataset with a registered strategy: runs the
        strategy's splitter over every document (optionally in parallel) and
        writes the tokens/chunks/extra outputs into LMDB databases, then
        persists the stats (total/max tokens and chunks, precomputed iteration
        lengths up to MAX_PRECOMPUTED_MULTIPLIER chunk multipliers).

        Writing conventions:
            1) "tokens": serialized with the smallest int width fitting the
               vocabulary (int8/int16/int32; float for patches/multimodal).
            2) "chunks": the splitter returns [(0,5),(5,9),(9,13)...]; they are
               converted to lengths (5,4,4,14) and stored as a numpy array with
               the smallest width fitting the max chunk length.
            3) "extra": stored as-is (bytes; strings utf-8; else orjson).
        """
        if resume:
            # Repair/resume the ranges that were not durably finished instead
            # of starting over (checkpointed every `checkpoint_every` rows).
            return self.sync_strategy(
                strategy_name, progress_bar=progress_bar, compression_level=compression_level,
                chunking_strict_checks=chunking_strict_checks, parallel=parallel,
                num_processes=num_processes, batch_size=batch_size,
                disk_size_multiplier=disk_size_multiplier, checkpoint_every=checkpoint_every)
        total = int((self.config_db.get_submdat_row(self.config_id) or {}).get('document_number') or 0)
        return self._pretokenize_ranges(
            strategy_name, ranges=[(0, total)], merge_stats=False,
            progress_bar=progress_bar, compression_level=compression_level,
            chunking_strict_checks=chunking_strict_checks, parallel=parallel,
            num_processes=num_processes, batch_size=batch_size, map_size=map_size,
            disk_size_multiplier=disk_size_multiplier, checkpoint_every=checkpoint_every)

    def sync_strategy(self, strategy_name, progress_bar=True, compression_level=0,
                      chunking_strict_checks=False, parallel=True, num_processes=None,
                      batch_size=256, disk_size_multiplier=4, checkpoint_every=100_000):
        """Pretokenize only the rows added since `strategy_name` last ran.

        This is the repair for a desynced dataset: it reuses exactly the normal
        pretokenization pipeline, but walks only the missing ranges and merges
        its statistics with the existing ones. Interrupted runs resume from the
        last durable checkpoint, because coverage is recorded as they progress.
        """
        self._check_writable()
        if not self.config_db.get_strategy_submdats(strategy_name, self.config_id):
            raise SubMdatMissesStrategy(
                f"Dataset '{self.submdat_name}' has no pretokenization '{strategy_name}'")
        total = int((self.config_db.get_submdat_row(self.config_id) or {}).get('document_number') or 0)
        self.len = total
        missing = self._missing_ranges(strategy_name, total, materialize=True)
        if not missing:
            return {'strategy': strategy_name, 'ranges': [], 'processed_docs': 0}
        print(f"Syncing '{strategy_name}' on '{self.submdat_name}': {missing}")
        result = self._pretokenize_ranges(
            strategy_name, ranges=missing, merge_stats=True,
            progress_bar=progress_bar, compression_level=compression_level,
            chunking_strict_checks=chunking_strict_checks, parallel=parallel,
            num_processes=num_processes, batch_size=batch_size, map_size=None,
            disk_size_multiplier=disk_size_multiplier, checkpoint_every=checkpoint_every)
        result['ranges'] = missing
        return result

    def _pretokenize_ranges(self, strategy_name, ranges, merge_stats,
                            progress_bar=True, compression_level=0,
                            chunking_strict_checks=False, parallel=True, num_processes=None,
                            batch_size=256, map_size=1 << 40, disk_size_multiplier=4,
                            checkpoint_every=100_000):
        self._check_writable()
        max_multiplier = MAX_PRECOMPUTED_MULTIPLIER
        data_disk_size = self.config_db.get_submdat_info(submdat_id=self.config_id)['raw_data_bytes'] or 0
        inferred_map_size = data_disk_size * disk_size_multiplier if data_disk_size else map_size
        if inferred_map_size:
            # LMDB needs room for its pages/overhead even for tiny datasets.
            inferred_map_size = max(inferred_map_size, 1 << 30)
        if inferred_map_size:
            print(f"Original data size is {data_disk_size}. A multiplier of {disk_size_multiplier} is used to initialize the pretokenization storage dataset(s). If this value ({inferred_map_size}) is too large, you'll get memory errors; if too small, pretokenization will not fit the database. Adjust disk_size_multiplier in case of problems.")
        self._attach_strategy(strategy_name, add=True, compression_level=compression_level,
                              map_size=inferred_map_size or (1 << 40))
        strategy = self.current_strategy

        stats = {
            'total_tokens': 0, 'max_tokens_per_doc': 0,
            'total_chunks': 0, 'max_chunks_per_doc': 0,
            'processed_docs': 0,
            'precomputed_lengths': [0] * (max_multiplier + 1),
        }
        if merge_stats:
            existing = self.config_db.get_strategy_submdats(strategy_name, self.config_id) or {}
            for field in ('total_tokens', 'max_tokens_per_doc', 'total_chunks',
                          'max_chunks_per_doc', 'processed_docs'):
                stats[field] = int(existing.get(field) or 0)
            for precomp in self.config_db.get_strategy_precomp(
                    strategy_name=strategy_name, submdat_id=self.config_id):
                multiplier = int(precomp['multiplier'])
                if 0 <= multiplier <= max_multiplier:
                    stats['precomputed_lengths'][multiplier] = int(precomp['precomputed_length'] or 0)
        else:
            self.config_db.set_strategy_coverage(strategy_name, self.config_id, [])
            self.config_db.clear_strategy_submdat_precomp(strategy_name, self.config_id)
        if merge_stats:
            self._coverage_ranges(strategy_name, materialize=True)

        progress_bar_generator = progress_bar if not parallel else False
        progress_start = None

        def checkpoint(upto):
            nonlocal progress_start
            if progress_start is not None and upto > progress_start:
                self._flush_pretok_dbs()
                self._persist_strategy_stats(strategy_name, stats, finalize=False)
                self._add_coverage(strategy_name, progress_start, upto)
            progress_start = upto

        def make_generator(start, end):
            return self.get_generator(
                progress_bar=progress_bar_generator,
                wanted_from_dbs=strategy.wants_from_db,
                raw=strategy.wants_raw, start=start, end=end)

        if not parallel:
            def process_batch(batch):
                keys, docs = zip(*batch)
                batch_results = strategy.pretokenize_batch(list(docs))
                for k, split_result in zip(keys, batch_results):
                    for db_name in strategy.returns:
                        if db_name not in split_result:
                            continue
                        if db_name == 'tokens':
                            token_bytes = strategy.prepare_tokens_for_storage(split_result['tokens'])
                            num_tokens = len(split_result['tokens'])
                            stats['total_tokens'] += num_tokens
                            stats['max_tokens_per_doc'] = max(stats['max_tokens_per_doc'], num_tokens)
                            self.pretok_db[db_name].write(key=k, obj=token_bytes)
                        elif db_name == 'chunks':
                            tokens_length = len(split_result.get('tokens', [])) if chunking_strict_checks else None
                            chunk_bytes = strategy.prepare_chunks_for_storage(
                                split_result['chunks'],
                                max_tokens_per_chunk=strategy.chunk_size,
                                tokens_length=tokens_length,
                                strict_checks=chunking_strict_checks)
                            chunk_count = len(split_result['chunks'])
                            stats['total_chunks'] += chunk_count
                            stats['max_chunks_per_doc'] = max(stats['max_chunks_per_doc'], chunk_count)
                            for j in range(1, max_multiplier + 1):
                                stats['precomputed_lengths'][j] += math.ceil(chunk_count / j)
                            self.pretok_db[db_name].write(obj=chunk_bytes, key=k)
                        else:
                            obj_bytes = strategy.prepare_extra_data_for_storage(split_result[db_name])
                            self.pretok_db[db_name].write(obj=obj_bytes, key=k)
                    stats['processed_docs'] += 1

            for range_start, range_end in ranges:
                progress_start = range_start
                processed_since_checkpoint = 0
                batch = []
                upto = range_start
                for offset, doc in enumerate(make_generator(range_start, range_end)):
                    key = range_start + offset
                    batch.append((key, doc))
                    if len(batch) >= batch_size:
                        process_batch(batch)
                        processed_since_checkpoint += len(batch)
                        upto = batch[-1][0] + 1
                        batch = []
                        if checkpoint_every and processed_since_checkpoint >= checkpoint_every:
                            checkpoint(upto)
                            processed_since_checkpoint = 0
                if batch:
                    process_batch(batch)
                    upto = batch[-1][0] + 1
                checkpoint(upto)
        else:
            if num_processes is None:
                num_processes = mp.cpu_count()
            strategy._initialize_components()

            def process_batch_and_write(batch_data):
                sub_batches = [batch_data[i:i + batch_size] for i in range(0, len(batch_data), batch_size)]
                results = pool.starmap(process_documents_batch, [
                    (sub_batch, strategy.manifest, self.pretok_base_path,
                     self.container.functions_path, strategy_name,
                     chunking_strict_checks, strategy.chunk_size, strategy.returns, max_multiplier)
                    for sub_batch in sub_batches])
                db_batches = {db_name: [] for db_name in strategy.returns}
                for worker_db_data, worker_stats in results:
                    stats['total_tokens'] += worker_stats['total_tokens']
                    stats['total_chunks'] += worker_stats['total_chunks']
                    stats['processed_docs'] += worker_stats['processed_docs']
                    stats['max_tokens_per_doc'] = max(stats['max_tokens_per_doc'], worker_stats['max_tokens_per_doc'])
                    stats['max_chunks_per_doc'] = max(stats['max_chunks_per_doc'], worker_stats['max_chunks_per_doc'])
                    stats['precomputed_lengths'] = (np.array(stats['precomputed_lengths']) +
                                                    np.array(worker_stats['precomputed_lengths'])).tolist()
                    for db_name in strategy.returns:
                        if db_name in worker_db_data:
                            db_batches[db_name].extend(worker_db_data[db_name])
                for db_name, key_data_pairs in db_batches.items():
                    if key_data_pairs:
                        self.pretok_db[db_name].write_batch_with_keys(key_data_pairs)

            with mp.Pool(processes=num_processes) as pool:
                for range_start, range_end in ranges:
                    progress_start = range_start
                    processed_since_checkpoint = 0
                    batch = []
                    upto = range_start
                    total_docs = max(range_end - range_start, 0)
                    pbar = tqdm(total=total_docs, desc="Pretokenizing", disable=not progress_bar)
                    for offset, doc in enumerate(make_generator(range_start, range_end)):
                        key = range_start + offset
                        batch.append((key, doc))
                        if len(batch) >= batch_size * num_processes:
                            process_batch_and_write(batch)
                            processed_since_checkpoint += len(batch)
                            upto = batch[-1][0] + 1
                            pbar.update(len(batch))
                            batch = []
                            if checkpoint_every and processed_since_checkpoint >= checkpoint_every:
                                checkpoint(upto)
                                processed_since_checkpoint = 0
                    if batch:
                        process_batch_and_write(batch)
                        upto = batch[-1][0] + 1
                        pbar.update(len(batch))
                    pbar.close()
                    checkpoint(upto)

        self.add_strategy_end(strategy_name=strategy_name, stats=stats)
        return {'strategy': strategy_name, 'processed_docs': stats['processed_docs'],
                'total_tokens': stats['total_tokens'], 'total_chunks': stats['total_chunks']}

    # ------------------------------------------------------------------ ingest
    @staticmethod
    def _ingest_logger(logger):
        if logger:
            return logger

        class PrintLogger:
            def warning(self, message):
                print("Warning: ", message)

            def error(self, message):
                print("Error: ", message)

            def info(self, message):
                print("Info: ", message)

        return PrintLogger()

    def _build_iterator(self, dataset_type, dataset_path, dataset_args, dataset_config,
                        progress_bar, logger_fn, skip=0):
        """Instantiate the input iterator for a source format. Returns
        (generator, inferred_map_size); append and ingestion share it."""
        inferred_map_size = None
        if dataset_type == 'jsonl':
            generator_fn = JSONIterator(json_path=dataset_path, dataset_args=dataset_args,
                                        progress_bar=progress_bar, logger=logger_fn, skip=skip)
            file_size = os.path.getsize(dataset_path)
            inferred_map_size = max(1 << 30, file_size * 5)
        elif dataset_type == 'lmdb':
            generator_fn = LMDBIterator(dataset_path, dataset_args=dataset_args,
                                        progress_bar=progress_bar, logger=logger_fn, skip=skip)
        elif dataset_type == 'hf':
            generator_fn = HuggingFaceIterator(dataset_path=dataset_path, logger=logger_fn,
                                               dataset_args=dataset_args, dataset_config=dataset_config,
                                               dataset_split='train', progress_bar=True, skip=skip)
        elif dataset_type == 'atlas':
            generator_fn = AtlasIterator(path=dataset_path, dataset_args=dataset_args,
                                         progress_bar=progress_bar, logger=logger_fn, skip=skip)
        elif dataset_type in ('sqlite', 'csv', 'files'):
            raise NotImplementedError(f"Dataset type '{dataset_type}' is not implemented")
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        return generator_fn, inferred_map_size

    def _raw_bytes_by_role(self):
        """Currently recorded raw byte counts per storage role ('data', 'meta')."""
        out = {'data': 0, 'meta': 0}
        for dbs in self.config_db.get_submdat_databases(self.config_id):
            role = _role_of(dbs)
            if role in out:
                out[role] = int(dbs.get('raw_data_bytes') or 0)
        return out

    def _write_documents(self, generator_fn, data_key, logger_fn, start_key,
                         do_transform=False, do_filtering=False, checkpoint_every=None,
                         on_checkpoint=None):
        """Write the documents yielded by `generator_fn` under consecutive keys
        starting at `start_key`. This is the single ingestion loop shared by
        fresh ingestion and appends."""
        errors_counters = {
            'hasDataError': 0,
            'generatorReturnedNone': 0,
            'missingDataKey': 0,
            'data<=10': 0,
        }
        raw_data_bytes = 0
        raw_meta_bytes = 0
        written = 0
        consumed = 0
        for local_i, item in enumerate(generator_fn):
            consumed = local_i + 1
            if item is None:
                errors_counters['generatorReturnedNone'] += 1
                continue
            if do_transform:
                pass
            if data_key in item:
                data = item[data_key]
            elif 'raw_content' in item:
                data = item['raw_content']
            else:
                logger_fn.warning(f"Data key '{data_key}' not found in item {local_i}. Item has keys {item.keys()}")
                errors_counters['missingDataKey'] += 1
                continue

            if isinstance(data, str):
                data = data.encode('utf-8')
            if not isinstance(data, bytes):
                logger_fn.warning(f"Data is of type {type(data)} but it should be either string or bytes")
                errors_counters['missingDataKey'] += 1
                continue
            if len(data) <= 10:
                logger_fn.warning("Data is smaller than 10 bytes. Skipping")
                errors_counters['data<=10'] += 1
                continue

            item.pop(data_key, None)
            item.pop('raw_content', None)
            if do_filtering:
                logger_fn.warning("Filter functions not implemented")

            key = start_key + written
            try:
                self.storage_db['data'].write(data, key=key)
            except Exception as e:
                logger_fn.error(f"Error writing data item {key}: {e}")
                errors_counters['hasDataError'] += 1
            try:
                serialized = orjson.dumps(item) if orjson else json.dumps(item).encode()
            except Exception as e:
                logger_fn.error(f'Serialization of metadata for item {key} failed: {e}')
                errors_counters['hasDataError'] += 1
                continue
            try:
                self.storage_db['meta'].write(serialized, key=key)
            except Exception as e:
                logger_fn.error(f"Error writing metadata item {key}: {e}")
                errors_counters['hasDataError'] += 1

            raw_data_bytes += len(data)
            raw_meta_bytes += len(serialized)
            written += 1
            if checkpoint_every and on_checkpoint and written % checkpoint_every == 0:
                # Flush the buffers and persist the durable length before the
                # checkpoint claims those documents are done.
                for db in self.storage_db.values():
                    db.sync_length()
                on_checkpoint(written, raw_data_bytes, raw_meta_bytes, errors_counters, consumed)
        return written, raw_data_bytes, raw_meta_bytes, errors_counters, consumed

    def convert_to_submdat(self, dataset_type, dataset_path, dataset_args={}, data_key='text',
                           modality='text', logger=False, progress_bar=True, do_transform=False,
                           do_filtering=False, dataset_config='', append=False, resume=False,
                           checkpoint_every=100_000):
        """
        Populate this dataset's databases from an external format.
        dataset_type: 'jsonl' | 'lmdb' | 'hf' | 'atlas' (see preprocessing.py).
        Returns the partial manifest written (document_number, errors_counters).

        With append=True the new documents are written after the last existing
        one instead of overwriting from key 0. With resume=True an interrupted
        ingestion of the same source resumes from its last checkpoint.
        """
        self._check_writable()
        logger_fn = self._ingest_logger(logger)

        if not self.storage_db:
            logger_fn.error(f"Databases are not loaded for dataset {self.submdat_name}")
            return None
        if 'data' not in self.storage_db or 'meta' not in self.storage_db:
            raise RuntimeError("Appending/ingesting documents requires both the 'data' and 'meta' storage databases.")

        row = self.config_db.get_submdat_row(self.config_id)
        old_docs = int(row.get('document_number') or 0)
        start_key = old_docs if append else 0
        source_skip = 0
        log_id = None
        resumed_rows = 0
        resumed_consumed = 0
        operation = 'append' if append else 'ingest'
        if resume:
            pending = self.config_db.get_resumable_ingestion(self.config_id, dataset_path)
            if pending:
                start_key = int(pending['start_key']) + int(pending['rows_inserted'])
                resumed_rows = int(pending['rows_inserted'])
                # The source is re-read from its beginning: skip exactly the
                # source records this operation already consumed (its
                # checkpoint), not the dataset rows that existed before it.
                source_skip = int(pending.get('source_consumed') or pending['rows_inserted'])
                resumed_consumed = source_skip
                log_id = pending['id']
                operation = pending['operation']
                print(f"Resuming {operation} of {dataset_path} at row {start_key} "
                      f"({source_skip} source documents skipped).")
            else:
                print(f"No interrupted ingestion found for {dataset_path}; starting from the beginning.")

        if not append and not resume:
            old_docs = 0
        base_raw = self._raw_bytes_by_role() if (append or resume) else {'data': 0, 'meta': 0}
        try:
            base_errors = json.loads(row.get('errors_counters') or '{}') if (append or resume) else {}
        except (TypeError, ValueError):
            base_errors = {}
        base_errors = {k: int(v or 0) for k, v in base_errors.items()}

        try:
            generator_fn, inferred_map_size = self._build_iterator(
                dataset_type, dataset_path, dataset_args, dataset_config,
                progress_bar, logger_fn, skip=source_skip)
        except Exception as e:
            logger_fn.error(str(e))
            return None

        # Re-open the storage dbs with a map size that fits the input. For an
        # append never shrink the existing map: the old rows must stay there.
        if inferred_map_size:
            map_size = inferred_map_size
            if append or resume:
                existing_disk = sum(
                    (self.config_db.get_storage_db_info(d['id']) or {}).get('disk_size') or 0
                    for d in self.config_db.get_submdat_databases(self.config_id)
                    if d.get('disk_size'))
                map_size = max(inferred_map_size, int(existing_disk * 1.2))
            print(f"Inferred size of the input file: {inferred_map_size}. Reloading the DB with the appropriate size.")
            self._forced_map_size = map_size
            self._close_storage_dbs(_type='storage')
            self._load_storage_dbs()
            self._forced_map_size = None

        if log_id is None:
            log_id = self.config_db.add_ingestion_log(
                self.config_id, operation, dataset_path, start_key, 0, status='running')

        def persist_progress(written, raw_data_bytes, raw_meta_bytes, errors, consumed, final=False):
            documents_number = start_key + written
            self.len = documents_number
            merged_errors = dict(base_errors)
            for key, value in errors.items():
                merged_errors[key] = merged_errors.get(key, 0) + int(value or 0)
            self.config_db.update_submdat_row(self.config_id, {
                'document_number': documents_number,
                'errors_counters': json.dumps(merged_errors)})
            disk_sizes = {
                'data': os.path.getsize(os.path.join(self.submdat_path, 'data.dat')),
                'meta': os.path.getsize(os.path.join(self.submdat_path, 'meta.dat')),
            }
            for dbs in self.config_db.get_submdat_databases(self.config_id):
                role = _role_of(dbs)
                if role not in disk_sizes:
                    raise NotImplementedError("Extra databases are not supported by convert_to_submdat")
                update = {'disk_size': disk_sizes[role]}
                if role == 'data':
                    update['raw_data_bytes'] = base_raw['data'] + raw_data_bytes
                elif role == 'meta':
                    update['raw_data_bytes'] = base_raw['meta'] + raw_meta_bytes
                self.config_db.update_submdat_database(
                    submdat_id=self.config_id, database_id=dbs['id'], data=update)
            self.config_db.update_ingestion_log(
                log_id, {'rows_inserted': resumed_rows + written,
                         'source_consumed': resumed_consumed + consumed,
                         'status': 'complete' if final else 'running'})

        written, raw_data_bytes, raw_meta_bytes, errors_counters, consumed = self._write_documents(
            generator_fn, data_key, logger_fn, start_key=start_key,
            do_transform=do_transform, do_filtering=do_filtering,
            checkpoint_every=checkpoint_every,
            on_checkpoint=persist_progress)

        persist_progress(written, raw_data_bytes, raw_meta_bytes, errors_counters, consumed, final=True)
        documents_number = start_key + written
        self.len = documents_number

        # Persist the durable length and leave the dataset usable in this
        # process: the handles were already opened with an input-sized map.
        for db in self.storage_db.values():
            db.sync_length()
        # Refresh the default view, the generation counter and any container mirror.
        self._after_data_change()
        return {
            'document_number': documents_number,
            'rows_inserted': written,
            'errors_counters': json.dumps(errors_counters),
        }

    def append_to_submdat(self, dataset_type, dataset_path, dataset_args={}, data_key=None,
                          logger=False, progress_bar=True, do_transform=False,
                          do_filtering=False, dataset_config='', resume=False,
                          checkpoint_every=100_000):
        """Append the documents of an external source after the last row.

        This is convert_to_submdat(append=True): same iterators, same validation,
        same statistics and history, same resumable checkpoints.
        """
        if getattr(self, 'raw_missing', False):
            raise RuntimeError("The raw storage of this dataset is not available here; append where it lives.")
        data_key = data_key or (self.config_db.get_submdat_row(self.config_id).get('data_key')
                                or 'text')
        return self.convert_to_submdat(dataset_type, dataset_path, dataset_args=dataset_args,
                                       data_key=data_key, logger=logger,
                                       progress_bar=progress_bar, do_transform=do_transform,
                                       do_filtering=do_filtering, dataset_config=dataset_config,
                                       append=True, resume=resume,
                                       checkpoint_every=checkpoint_every)

    def ingestion_history(self, limit=100):
        """The append/ingestion log of this dataset (newest first)."""
        return self.config_db.get_ingestion_log(self.config_id, limit=limit)


def _is_mdat_catalog_folder(path: str) -> bool:
    return (os.path.isfile(os.path.join(path, 'mdat_container.db'))
            or os.path.isfile(os.path.join(path, 'mdat.db')))


class MDatWriter:
    """Small streaming writer for the data/meta databases of an mdat.
    It is the mdat analogue of ``open("file.jsonl", "a")``: the common case is
    a path pointing at a standalone dataset (created on first write), a data
    column (default ``text``) and a loop::

        with mdat_writer("corpus.mdat", data_column="text") as md:
            for document in documents:
                md.write(document)

    The data column goes to the raw ``data`` database, every other field to
    ``meta``. If the path is a multi-dataset container, pass ``dataset=NAME``;
    a missing dataset is created when ``create=True``.

    ``keys_only=True`` (or ``data_column=None``) stores only metadata, creating
    no data database: this is what the annotation "keys only" output uses.
    ``write(record, key=N)`` overwrites an existing row and invalidates every
    pretokenization strategy; ``delete(keys)`` compacts the databases so keys
    remain sequential.
    """

    def __init__(self, path, data_column='text', dataset=None, create=True,
                 compression_levels=None, map_sizes=None, keys_only=False,
                 modality='text', name=None):
        self.path = path
        self.dataset = dataset
        self.create = create
        self.keys_only = bool(keys_only) or data_column is None
        self.data_column = None if self.keys_only else data_column
        self.compression_levels = compression_levels or {}
        self.map_sizes = map_sizes or {}
        self.modality = modality
        self.name = name
        self.container = None
        self.mdat = None
        self._data_delta = 0
        self._meta_delta = 0
        self._invalidate = False
        self._written = 0
        self._closed = False
        self._open_target()

    def _open_target(self):
        from .container import MDatContainer
        path = os.path.abspath(self.path)
        if self.dataset is not None:
            if os.path.isdir(path) and _is_mdat_catalog_folder(path):
                self.container = MDatContainer.load_dataset(path, readonly=False)
            elif self.create:
                self.container = MDatContainer.create_new(path)
            else:
                raise FileNotFoundError(f"No mdat container at {path}")
            if self.dataset in self.container.list_submdat():
                self.mdat = self.container[ self.dataset ] if self.dataset in self.container.loaded_submdats \
                    else self.container._ensure_loaded(self.dataset)
            elif self.create:
                db_types = ['meta'] if self.keys_only else ['meta', 'data']
                self.mdat = self.container.add_submdat(
                    self.dataset,
                    compression_levels=self.compression_levels,
                    map_sizes=self.map_sizes,
                    db_types=db_types,
                    data_key=self.data_column or 'text',
                    modality=self.modality)
            else:
                raise SubmDatNotFound(f"Dataset '{self.dataset}' is not in {path}")
        else:
            if os.path.isdir(path) and _is_mdat_catalog_folder(path):
                self.mdat = MDat.open_standalone(path, readonly=False)
                self.container = self.mdat.container
            elif self.create:
                db_types = ['meta'] if self.keys_only else ['meta', 'data']
                self.mdat = MDat.create_standalone(
                    path, name=self.name,
                    compression_levels=self.compression_levels,
                    map_sizes=self.map_sizes,
                    db_types=db_types,
                    data_key=self.data_column or 'text',
                    modality=self.modality)
                self.container = self.mdat.container
            else:
                raise FileNotFoundError(f"No standalone mdat at {path}")

    def write(self, record, key=None) -> int:
        """Write one record and return its key. A key inside the dataset
        overwrites that row; a missing/out-of-range key appends."""
        if self._closed:
            raise RuntimeError("MDatWriter is closed")
        key, data_delta, meta_delta, is_new = self.mdat._write_one(
            record, data_column=self.data_column, key=key)
        self._data_delta += data_delta
        self._meta_delta += meta_delta
        if not is_new:
            self._invalidate = True
        self._written += 1
        return key

    def update(self, key, record) -> int:
        """Overwrite an existing row (breaks pretokenization)."""
        if int(key) >= self.mdat.len:
            raise IndexError(f"Row {key} does not exist (len={self.mdat.len})")
        self._invalidate = True
        return self.write(record, key=key)

    def edit(self, rows) -> int:
        """Overwrite several rows: a ``{key: record}`` mapping or key/record pairs."""
        items = rows.items() if isinstance(rows, dict) else rows
        count = 0
        for key, record in items:
            self.update(key, record)
            count += 1
        return count

    def delete(self, keys, compact=True) -> int:
        """Delete rows (compacting keys) and break pretokenization."""
        count = self.mdat.delete_documents(keys, compact=compact)
        self._written += count
        return count

    def flush(self) -> None:
        """Persist buffered rows, length, sizes and invalidation to the catalog."""
        if self._closed:
            return
        self.mdat.commit_storage_changes(
            self._data_delta, self._meta_delta,
            invalidate_pretokenization=self._invalidate)
        self._data_delta = 0
        self._meta_delta = 0
        self._invalidate = False

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        self._closed = True
        if self.container is not None:
            self.container.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class MDatReader:
    """Iterate the rows of an mdat without loading it into memory.

    Reading a normal dataset yields its metadata fields plus the data column::

        with mdat_reader("corpus.mdat") as md:
            for row in md:
                print(row)

    ``keys_only=True`` yields only metadata (no raw data is read). ``include_key``
    adds the sequential row key to each record, which is what annotation needs
    to write results keyed by the source row.
    """

    def __init__(self, path, dataset=None, data_column=None, keys_only=False,
                 include_key=False, readonly=True):
        from .container import MDatContainer
        self.path = path
        self.dataset = dataset
        self.keys_only = keys_only
        self.include_key = include_key
        self.readonly = readonly
        self.container = None
        self.mdat = None
        self._open_target()
        row = self.mdat.get_row() or {}
        self.data_column = data_column or row.get('data_key') or 'text'
        self.has_data = 'data' in self.mdat.storage_db and not keys_only
        self._index = 0
        self._length = int(self.mdat.len)
        self._decode = orjson.loads if orjson else lambda value: json.loads(value.decode())

    def _open_target(self):
        from .container import MDatContainer
        path = os.path.abspath(self.path)
        if self.dataset is not None:
            self.container = MDatContainer.load_dataset(path, readonly=self.readonly)
            self.mdat = self.container[self.dataset] if self.dataset in self.container.loaded_submdats \
                else self.container._ensure_loaded(self.dataset)
        else:
            self.mdat = MDat.open_standalone(path, readonly=self.readonly)
            self.container = self.mdat.container

    def __iter__(self):
        return self

    def __len__(self):
        return self._length

    def __next__(self):
        if self._index >= self._length:
            raise StopIteration
        key = self._index
        self._index += 1
        row = {}
        meta_db = self.mdat.storage_db.get('meta')
        if meta_db is not None:
            row.update(self._decode(meta_db[key]))
        if self.has_data:
            data = self.mdat.storage_db['data'][key]
            try:
                row[self.data_column] = data.decode('utf-8')
            except UnicodeDecodeError:
                row[self.data_column] = data
        if self.include_key:
            row['key'] = key
        return row

    def close(self) -> None:
        if self.container is not None:
            self.container.close()
            self.container = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def mdat_writer(path, data_column='text', dataset=None, create=True, **kwargs) -> MDatWriter:
    """Open an mdat for streaming writes (see MDatWriter)."""
    return MDatWriter(path, data_column=data_column, dataset=dataset, create=create, **kwargs)


def mdat_reader(path, dataset=None, data_column=None, keys_only=False,
                include_key=False, readonly=True) -> MDatReader:
    """Open an mdat for streaming reads (see MDatReader)."""
    return MDatReader(path, dataset=dataset, data_column=data_column, keys_only=keys_only,
                      include_key=include_key, readonly=readonly)


def _role_of(dbs_row) -> str:
    """Storage-db role name ('data', 'meta' or the extra name)."""
    if dbs_row['is_data']:
        return 'data'
    if dbs_row['is_meta']:
        return 'meta'
    if dbs_row['is_extra'] and isinstance(dbs_row['is_extra'], str):
        return json.loads(dbs_row['is_extra'])
    return 'extra'


def _pretok_role_of(sdb_row) -> str:
    """Pretokenization-db role name ('tokens', 'chunks' or the extra name)."""
    if sdb_row['is_tokens'] == 1:
        return 'tokens'
    if sdb_row['is_chunks'] == 1:
        return 'chunks'
    if sdb_row['is_extra'] and isinstance(sdb_row['is_extra'], str):
        return json.loads(sdb_row['is_extra'])
    return 'extra'


# Backward compatibility
SubMdat = MDat


# Backward compatibility with the old monolithic mdat.py: everything that used
# to be importable from this module still is, resolved lazily so that there is
# no circular import between mdat.py and container.py.
_LAZY_CONTAINER_NAMES = ('MDatContainer', 'MatformerDataset')
_LAZY_MODULE_NAMES = {
    'DatabaseManager': '.sql_database',
    'LMDBDataset': '.lmdb_store',
    'PretokenizationStrategy': '.strategy',
    'process_documents_batch': '.strategy',
    'JSONIterator': '.preprocessing',
    'LMDBIterator': '.preprocessing',
    'HuggingFaceIterator': '.preprocessing',
    'AtlasIterator': '.preprocessing',
}


def __getattr__(name):
    if name in _LAZY_CONTAINER_NAMES:
        from . import container
        return getattr(container, name)
    if name in _LAZY_MODULE_NAMES:
        import importlib
        module = importlib.import_module(_LAZY_MODULE_NAMES[name], __package__)
        return getattr(module, name)
    from . import mdat_common
    if hasattr(mdat_common, name):
        return getattr(mdat_common, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
