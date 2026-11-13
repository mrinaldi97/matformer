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

from .lmdb import LMDBDataset
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
            raise FileNotFoundError(f"{instance.submdat_name} is registered but its folder is missing at {instance.submdat_path}.")
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
        disk size (falls back to the default for empty/unknown ones)."""
        disk_size = self.config_db.get_storage_db_info(db_info['id'])['disk_size']
        if not isinstance(disk_size, int) or disk_size <= 1:
            return self.default_map_size
        return disk_size * 1.2

    def _load_storage_dbs(self, batch_size=None):
        """Open all the storage databases ('data', 'meta', extras) of this dataset."""
        for dbs in self.config_db.get_submdat_databases(self.config_id):
            db_type = _role_of(dbs)
            db_path = os.path.join(self.submdat_path, db_type) + '.dat'
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

    def get_generator(self, progress_bar=False, wanted_from_dbs='full', wanted_from_strategy=None, raw=False, randomize=False, seed=27):
        """
        Generator over the elements of this dataset.
        Args:
            progress_bar: show a tqdm progress bar
            wanted_from_dbs / wanted_from_strategy / raw: composition of each item
                (same semantics as set_default_wanted)
            randomize: iterate in randomized order (deterministic per seed)
        """
        permutator = RandomPermutator(max_len=self.len, seed=seed) if randomize else None
        iterator = tqdm(range(self.len)) if progress_bar else range(self.len)
        for key in iterator:
            if permutator is not None:
                key = permutator(key)
            yield self._compose_return(key=key, wanted_from_dbs=wanted_from_dbs,
                                       wanted_from_strategy=wanted_from_strategy, raw=raw)

    def __len__(self):
        return self.len

    def get_row(self) -> Dict[str, Any]:
        """The catalog row describing this dataset (its 'manifest')."""
        return self.config_db.get_submdat_row(self.submdat_name if not self.linked else self.config_id)

    def get_current_strategy(self):
        return self.current_strategy

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

        self.config_db.update_strategy_submdat(strategy_name, self.config_id, {
            'finalized_at': datetime.utcnow().isoformat(),
            'total_tokens': stats.get('total_tokens', 0),
            'max_tokens_per_doc': stats.get('max_tokens_per_doc', 0),
            'total_chunks': stats.get('total_chunks', 0),
            'max_chunks_per_doc': stats.get('max_chunks_per_doc', 0),
            'processed_docs': stats.get('processed_docs', 0)
        })

        for multiplier, precomp_len in enumerate(stats.get('precomputed_lengths', [])):
            if precomp_len > 0:
                self.config_db.set_strategy_submdat_precomp(strategy_name, self.config_id, multiplier, precomp_len)

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

    # ---------------------------------------------------------- pretokenization
    def pretokenize_submdat(self, strategy_name, progress_bar=True, compression_level=0,
                            chunking_strict_checks=False, parallel=True, num_processes=None,
                            batch_size=256, map_size=1 << 40, disk_size_multiplier=4):
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
        self._check_writable()
        max_multiplier = MAX_PRECOMPUTED_MULTIPLIER
        # Map size for the output dbs: raw data size x multiplier (tunable)
        data_disk_size = self.config_db.get_submdat_info(submdat_id=self.config_id)['raw_data_bytes'] or 0
        inferred_map_size = data_disk_size * disk_size_multiplier if data_disk_size else map_size
        print(f"Original data size is {data_disk_size}. A multiplier of {disk_size_multiplier} is used to initialize the pretokenization storage dataset(s). If this value ({inferred_map_size}) is too large, you'll get memory errors; if too small, pretokenization will not fit the database. Adjust disk_size_multiplier in case of problems.")
        self._attach_strategy(strategy_name, add=True, compression_level=compression_level, map_size=inferred_map_size)
        strategy = self.current_strategy
        stats = {
            'total_tokens': 0,
            'max_tokens_per_doc': 0,
            'total_chunks': 0,
            'max_chunks_per_doc': 0,
            'processed_docs': 0,
            'precomputed_lengths': [0] * (max_multiplier + 1),
        }
        # What does the splitter want from the dataset's databases? (default: raw data)
        progress_bar_generator = progress_bar if not parallel else False
        generator = self.get_generator(
            progress_bar=progress_bar_generator,
            wanted_from_dbs=strategy.wants_from_db,
            raw=strategy.wants_raw
        )

        if not parallel:
            def process_batch(batch, stats):
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
                                strict_checks=chunking_strict_checks
                            )
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

            batch = []
            for key, doc in enumerate(generator):
                batch.append((key, doc))
                if len(batch) >= batch_size:
                    process_batch(batch, stats)
                    batch = []
            if batch:
                process_batch(batch, stats)
        else:
            if num_processes is None:
                num_processes = mp.cpu_count()
            self.current_strategy._initialize_components()

            def process_batch_and_write(batch_data, stats):
                """Process a batch across workers and write the merged results."""
                sub_batches = [batch_data[i:i + batch_size] for i in range(0, len(batch_data), batch_size)]

                results = pool.starmap(process_documents_batch, [
                    (sub_batch, self.current_strategy.manifest, self.pretok_base_path,
                     self.container.functions_path, strategy_name,
                     chunking_strict_checks, strategy.chunk_size, strategy.returns, max_multiplier)
                    for sub_batch in sub_batches
                ])

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
                return stats

            with mp.Pool(processes=num_processes) as pool:
                batch = []
                total_docs = len(self)
                pbar = tqdm(total=total_docs, desc="Pretokenizing", disable=not progress_bar)

                for key, doc in enumerate(generator):
                    batch.append((key, doc))
                    if len(batch) >= batch_size * num_processes:
                        stats = process_batch_and_write(batch, stats)
                        pbar.update(len(batch))
                        batch = []

                if batch:
                    stats = process_batch_and_write(batch, stats)
                    pbar.update(len(batch))
                pbar.close()

        # Close the databases and persist the stats
        self.add_strategy_end(strategy_name=strategy_name, stats=stats)

    # ------------------------------------------------------------------ ingest
    def convert_to_submdat(self, dataset_type, dataset_path, dataset_args={}, data_key='text',
                           modality='text', logger=False, progress_bar=True, do_transform=False,
                           do_filtering=False, dataset_config=''):
        """
        Populate this dataset's databases from an external format.
        dataset_type: 'jsonl' | 'lmdb' | 'hf' | 'atlas' (see preprocessing.py).
        Returns the partial manifest written (document_number, errors_counters).
        """
        self._check_writable()

        class PrintLogger:
            def warning(self, message):
                print("Warning: ", message)

            def error(self, message):
                print("Error: ", message)

            def info(self, message):
                print("Info: ", message)

        logger_fn = logger if logger else PrintLogger()

        if not self.storage_db:
            logger_fn.error(f"Databases are not loaded for dataset {self.submdat_name}")
            return None

        # Instantiate the generator for the requested input format
        # dataset_args example: json_batch_read=False, files_recurse_in_folder=True,
        #                       files_metadata_type='multiple_json', csv_has_header=False,
        #                       csv_data_field=0, hf_split='train', hf_subdataset=None
        inferred_map_size = None
        if dataset_type == 'jsonl':
            generator_fn = JSONIterator(json_path=dataset_path, dataset_args=dataset_args,
                                        progress_bar=progress_bar, logger=logger_fn)
            file_size = os.path.getsize(dataset_path)
            inferred_map_size = max(1 << 30, file_size * 5)
        elif dataset_type == 'lmdb':
            generator_fn = LMDBIterator(dataset_path, dataset_args=dataset_args,
                                        progress_bar=progress_bar, logger=logger_fn)
        elif dataset_type == 'hf':
            generator_fn = HuggingFaceIterator(dataset_path=dataset_path, logger=logger_fn,
                                               dataset_args=dataset_args, dataset_config=dataset_config,
                                               dataset_split='train', progress_bar=True)
        elif dataset_type == 'atlas':
            generator_fn = AtlasIterator(path=dataset_path, dataset_args=dataset_args,
                                         progress_bar=progress_bar, logger=logger_fn)
        elif dataset_type in ('sqlite', 'csv', 'files'):
            logger_fn.error(f"Dataset type '{dataset_type}' is not implemented")
            return None
        else:
            error = f"Unsupported dataset type: {dataset_type}"
            logger_fn.error(error)
            return error

        # Re-open the storage dbs with a map size sized for the input, if known
        if inferred_map_size:
            print(f"Inferred size of the input file: {inferred_map_size}. Reloading the DB with the appropriate size.")
            self.default_map_size = inferred_map_size
            self._close_storage_dbs(_type='storage')
            self._load_storage_dbs()

        errors_counters = {
            'hasDataError': 0,
            'generatorReturnedNone': 0,
            'missingDataKey': 0,
            'data<=10': 0,
        }
        raw_data_bytes = 0
        raw_meta_bytes = 0

        for i, item in enumerate(generator_fn):
            if item is None:
                errors_counters['generatorReturnedNone'] += 1
                continue
            # An optional transformer could be plugged here (not implemented)
            if do_transform:
                pass

            # Extract the document body; 'raw_content' is a compatibility
            # fallback for redpajama-style records.
            if data_key in item:
                data = item[data_key]
            elif 'raw_content' in item:
                data = item['raw_content']
            else:
                logger_fn.warning(f"Data key '{data_key}' not found in item {i}. Item has keys {item.keys()}")
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

            # Optional filtering (language id, quality metrics...) not implemented
            if do_filtering:
                logger_fn.warning("Filter functions not implemented")

            try:
                self.storage_db['data'].write(data, key=i)
            except Exception as e:
                logger_fn.error(f"Error writing data item {i}: {e}")
                errors_counters['hasDataError'] += 1
            try:
                serialized = orjson.dumps(item) if orjson else json.dumps(item).encode()
            except Exception as e:
                logger_fn.error(f'Serialization of metadata for item {i} failed: {e}')
                errors_counters['hasDataError'] += 1
                continue

            try:
                self.storage_db['meta'].write(serialized, key=i)
            except Exception as e:
                logger_fn.error(f"Error writing metadata item {i}: {e}")
                errors_counters['hasDataError'] += 1

            raw_data_bytes += len(data)
            raw_meta_bytes += len(serialized)

        # Close the storage databases and record sizes
        documents_number = len(self.storage_db['data'])
        self.storage_db['data'].close()
        self.storage_db['meta'].close()
        db_data_bytes = os.path.getsize(os.path.join(self.submdat_path, 'data.dat'))
        db_meta_bytes = os.path.getsize(os.path.join(self.submdat_path, 'meta.dat'))

        for dbs in self.config_db.get_submdat_databases(self.config_id):
            if dbs['is_data']:
                self.config_db.update_submdat_database(submdat_id=self.config_id, database_id=dbs['id'],
                                                       data={'raw_data_bytes': raw_data_bytes, 'disk_size': db_data_bytes})
            elif dbs['is_meta']:
                self.config_db.update_submdat_database(submdat_id=self.config_id, database_id=dbs['id'],
                                                       data={'raw_data_bytes': raw_meta_bytes, 'disk_size': db_meta_bytes})
            else:
                raise NotImplementedError("Extra databases are not supported by convert_to_submdat")

        partial_manifest = {
            'document_number': documents_number,
            'errors_counters': json.dumps(errors_counters)
        }
        self.config_db.update_submdat_row(self.config_id, partial_manifest)
        self.len = documents_number
        # Refresh the default view, the generation counter and any container mirror
        self._after_data_change()
        return partial_manifest


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
    'LMDBDataset': '.lmdb',
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
