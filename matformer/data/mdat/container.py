"""MDatContainer (formerly MatformerDataset): a container of mdats.

An MDatContainer is the training-time object: it owns views, shuffle pointer
files and distributed precomputes, and it references datasets. Datasets can be

* **internal**: their folder lives under ``<container>/datasets/<name>`` and
  their catalog rows live in the container's own ``mdat_container.db``;
* **linked** (``link_dataset``): a *standalone* MDat folder (created with
  ``MDat.create_standalone``) that lives anywhere on any filesystem. Only a
  relative path is stored, resolved against ``$MATFORMER_MDAT_ROOT`` first and
  the container folder second, so an entire dataset tree can be relocated by
  exporting one environment variable. The external dataset's rows (stats,
  strategies, precomputed lengths) are mirrored into the container catalog at
  load time; the container reads data/pretok bytes from the external folder.

Linked datasets can be reused by any number of containers. Editing a linked
dataset (adding documents, adding a new pretokenization strategy) never
invalidates other containers' strategies: document-set changes
(generation counter) invalidate shuffle files/views, in which case the
container warns at load and offers ``repair()`` to reshuffle affected views.

The class still answers to the old name ``MatformerDataset``.
"""
from __future__ import annotations

import json
import math
import os
import queue
import shutil
import struct
import threading
import time
from bisect import bisect_right
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from tqdm import tqdm
except:
    pass
from torch.utils.data import IterableDataset

from .mdat import MDat
from .mdat_common import (
    MDatAlreadyPresent,
    MDatIsReadOnly,
    MDatNotFound,
    MDatNotShuffled,
    NameAlreadyUsed,
    RandomPermutator,
    SubmDatNotFound,
    SubMdatMissesStrategy,
    TooManySubmDats,
    sanify_name,
)
from .sql_database import DatabaseManager
from .strategy import PretokenizationStrategy

# Environment variable that linked dataset relative paths are resolved against.
MDAT_ROOT_ENV = 'MATFORMER_MDAT_ROOT'


class _ShuffleFileReader:
    """Owns the single read handle on the active view's shuffle pointer file.

    Pointer records are ``(submdat_id, doc_id)`` structs; one reader instance
    is created per active view and shared by the main thread and the prefetch
    thread.
    """

    def __init__(self, path: str, struct_format: str):
        self.path = path
        self.struct_format = struct_format
        self.struct_size = struct.calcsize(struct_format)
        self._f = None

    def _ensure_open(self):
        if self._f is None:
            self._f = open(self.path, 'rb')

    def seek_document(self, index: int):
        self._ensure_open()
        self._f.seek(index * self.struct_size)

    def read_pair(self) -> Optional[tuple]:
        """Return the next (submdat_id, doc_id), or None at end of file."""
        self._ensure_open()
        data = self._f.read(self.struct_size)
        if not data:
            return None
        return struct.unpack(self.struct_format, data)

    def close(self):
        if self._f is not None:
            try:
                self._f.close()
            except Exception:
                pass
            self._f = None


class MDatContainer(IterableDataset):
    DB_FILENAME = 'mdat_container.db'
    LEGACY_DB_FILENAME = 'mdat.db'  # will be removed
    DEFAULT_DSMAP_BITS = 8

    # Directory structure:
    DATASETS_DIR = 'datasets'
    SHUFFLING_DIR = 'shuffling'
    VIEWS_DIR = 'views'
    PRETOK_DIR = 'pretok'
    FUNCTIONS_DIR = 'functions'

    
    _linked_container_cache: Dict[str, 'MDatContainer'] = {}

    def __init__(self) -> None:
        """
        Initialize an empty container.
        The container is the entry point for mdats and pretok strategies:
        loaded_submdats holds the MDat objects indexed by (local) name, while
        linked datasets' catalogs are opened on demand and cached per path.
        *If a pretokenization strategy is chunk-based (ex. text pretokenized in
        chunks of 512 tokens), a chunk_multiplier elevates the target sequence
        length required by the model.
        *Several "views" of the container can be defined, for example to
        prepare a smaller set for ablation studies.
        """
        self._cached_length = None
        self.manifest: Dict[str, Any] = {}
        self.loaded_submdats: Dict[str, Any] = {}  # MDat objects, by local name
        self.pretok_strategies: Dict[str, Any] = {}
        self.readonly: bool = False
        self.current_strategy = None
        self.total_documents = 0
        self.total_tokens = None  # Populated when a strategy is loaded
        self.total_chunks = None
        self.total_divided_chunks = None
        self.chunk_multiplier = 1
        self._shuffle_reader: Optional[_ShuffleFileReader] = None
        self._active_view_shuffled = False

    #manifest
    def _get_manifest_attr(self, key: str, default: Any = None) -> Any:
        """Get manifest attribute (possible to set a default)"""
        return self.manifest.get(key, default)

    def _set_manifest_attr(self, key: str, value: Any, save: bool = False) -> None:
        """Set manifest attribute, but don't update the file unless 'save' is True"""
        self.manifest[key] = value
        if save:
            self.db.update_manifest(self.manifest)

    # load/create
    @classmethod
    def load_dataset(cls, path: str, shuffle: bool = False,
                     ds_view: Optional[str] = None, distributed: bool = False,
                     readonly: bool = False, create_if_not_existent: bool = False,
                     batch_size: Optional[int] = None, prefetch_buffer=32768) -> 'MDatContainer':
        """
        Load an existing container from a path. All views and linked datasets
        are populated and ready to be used.

        Args:
            path: Container root directory path
            shuffle: Shuffle the active view if it is not shuffled yet
            ds_view: View to load (default: 'default')
            distributed: Enable distributed training mode
            readonly: Load in read-only mode
            create_if_not_existent: Create the container if missing
            batch_size: Batch size used by chunked_for_recurrence modality
            prefetch_buffer: Size of the prefetch queue (documents)
        """
        instance = cls()
        instance._set_paths(path)
        instance.readonly = readonly
        instance._cached_length = None
        instance.dist = None
        instance.batch_size = batch_size
        instance.prefetch_buffer = prefetch_buffer
        if instance._container_db_path() is not None:
            instance.db = DatabaseManager(instance.db_path)
            instance.db.connect()
            instance.manifest = instance.db.get_manifest()
            instance.pretok_strategies = {}
            if distributed:
                instance._set_distributed_training()
            else:
                instance.dist = None
            instance.set_view(ds_view if ds_view is not None else 'default')
            if shuffle and not instance._active_view_shuffled:
                struct_format = instance.shuffle(instance.current_view)
                instance.db.update_view_row(instance.current_view, {'shuffle_struct_format': struct_format})
                instance._active_view_shuffled = True
            instance.check_linked_datasets()
            instance.set_iteration_modality('document', with_meta=True)  # Default: raw document + metadata
        else:
            if create_if_not_existent:
                return cls.create_new(path=path, overwrite=False)
            else:
                raise MDatNotFound(f"MDat container not found at {path}")

        return instance

    @classmethod
    def load_catalog(cls, path: str, readonly: bool = False) -> 'MDatContainer':
        """
        Open only the catalog of a container/dataset folder: no views, no
        storage (LMDB) environments. Used for linked datasets.
        """
        instance = cls()
        instance._set_paths(path)
        instance.readonly = readonly
        instance.dist = False
        instance.world_size = 1
        instance.rank_size = 0
        db_path = instance._container_db_path()
        if db_path is None:
            raise MDatNotFound(f"No container catalog found at {path}")
        instance.db = DatabaseManager(db_path)
        instance.db.connect()
        instance.manifest = instance.db.get_manifest()
        instance.current_view = None
        instance._active_view = None
        instance._active_view_shuffled = False
        instance.batch_size = None
        instance.prefetch_buffer = 0
        return instance

    @classmethod
    def create_new(cls, path: str, overwrite: bool = False,
                   dsmap_bits: int = None, modalities: list = ['text']) -> 'MDatContainer':
        """
        Create a new empty container and load it.

        Args:
            path: Directory path for the new container
            overwrite: Overwrite an existing container if present
            dsmap_bits: Dataset mapping bits (default 8, i.e. max 255 datasets)
        """
        if dsmap_bits is None:
            dsmap_bits = cls.DEFAULT_DSMAP_BITS

        instance = cls()
        instance._set_paths(path)
        instance.readonly = False

        if instance.dataset_exists():
            if not overwrite:
                raise MDatAlreadyPresent(f"Container already exists at {path}")
            else:
                shutil.rmtree(instance.mdat_path)

        instance._create_directory_structure()

        instance.db = DatabaseManager(instance.db_path)
        instance.db.connect()
        instance.db._initialize_db()
        instance.db.create_manifest(dsmap_bits, modalities)
        instance.db.add_view('default')

        return cls.load_dataset(path, readonly=False)

    def _container_db_path(self) -> Optional[str]:
        new_path = os.path.join(self.mdat_path, self.DB_FILENAME)
        if os.path.exists(new_path):
            return new_path
        legacy_path = os.path.join(self.mdat_path, self.LEGACY_DB_FILENAME)
        if os.path.exists(legacy_path):
            return legacy_path
        return None

    def _set_paths(self, path: str) -> None:
        self.mdat_path = path
        resolved = self._container_db_path()
        self.db_path = resolved if resolved is not None else os.path.join(path, self.DB_FILENAME)
        self.pretok_path = os.path.join(path, self.PRETOK_DIR)
        self.dataset_path = os.path.join(path, self.DATASETS_DIR)
        self.views_path = os.path.join(path, self.VIEWS_DIR)
        self.shuffling_path = os.path.join(path, self.SHUFFLING_DIR)
        self.functions_path = os.path.join(path, self.FUNCTIONS_DIR)

    def dataset_exists(self) -> bool:
        """Check if a container catalog exists at this path."""
        return self._container_db_path() is not None

    def close(self):
        """Release all resources: shuffle reader, LMDB storage environments
        (ref-counted) and the catalog connection. After close() the same
        folder can be reopened writable in the same process."""
        self.stop_prefetch()
        for sbm in list(getattr(self, 'loaded_submdats', {}).values()):
            if sbm is not None:
                try:
                    sbm._close_storage_dbs()
                except Exception:
                    pass
        self.loaded_submdats = {}
        reader = getattr(self, '_shuffle_reader', None)
        if reader is not None:
            reader.close()
            self._shuffle_reader = None
        if getattr(self, 'db', None) is not None:
            try:
                self.db.connect().close()
            except Exception:
                pass
            self.db.conn = None

    def _create_directory_structure(self) -> None:
        os.makedirs(self.mdat_path, exist_ok=True)
        os.makedirs(self.dataset_path, exist_ok=True)
        os.makedirs(self.shuffling_path, exist_ok=True)
        os.makedirs(self.views_path, exist_ok=True)
        os.makedirs(self.pretok_path, exist_ok=True)
        os.makedirs(self.functions_path, exist_ok=True)

    #  dataset access
    def __getitem__(self, submdat_name: str) -> Any:
        """
        Direct access to a loaded dataset (MDat object).
        Args:
            submdat_name: Local name (alias) of the dataset
        """
        if hasattr(self, 'loaded_submdats') and submdat_name in self.loaded_submdats:
            return self.loaded_submdats[submdat_name]
        raise SubmDatNotFound(f"Dataset '{submdat_name}' not found or not loaded in the current view.")

    def get_current_document(self) -> Any:
        """Get current document."""
        return self.current_document

    def _set_distributed_training(self) -> None:
        """Setup distributed training config."""
        try:
            import torch.distributed as dist
            torch_dist_available = True
        except:
            torch_dist_available = False
        if torch_dist_available and dist.is_available() and dist.is_initialized():
            self.dist = True
            self.world_size = dist.get_world_size()
            self.rank_size = dist.get_rank()
        else:
            self.dist = False
            self.world_size = 1
            self.rank_size = 0

    # ---------------------------------------------------- internal datasets
    # BACKWARD-COMPAT(internal): "internal" datasets (folder under
    # <container>/datasets/, rows inside the container catalog) are the legacy
    # layout of the old submdats. The new world is standalone MDat folders +
    # link_dataset(). When the legacy path is dropped, remove: add_submdat,
    # the internal branches of _load_submdat/_populate_submdat, and
    # MDat.load_submdat/create_submdat.
    def _ensure_loaded(self, sbm_name) -> MDat:
        """Return the loaded MDat for a name, loading it on demand."""
        sbm = self.loaded_submdats.get(sbm_name)
        if sbm is None:
            self._load_submdat(sbm_name)
            sbm = self.loaded_submdats[sbm_name]
        return sbm

    def _load_submdat(self, sbm_name):
        """Load a single dataset row of this container (internal or linked)."""
        row = self.db.get_submdat_row(sbm_name)
        if row.get('source_rel_path'):
            self.loaded_submdats[sbm_name] = self._load_linked_submdat(sbm_name, row)
        else:
            self.loaded_submdats[sbm_name] = MDat.load_submdat(self, sbm_name)

    def _populate_submdat(self) -> None:
        """Load every dataset used by the current view (skipping loaded ones)."""
        self.total_documents = self.db.get_view_totals(view_name=self.current_view)['document_number']
        view_data = self.db.ask_view(view=self.current_view, keys=['preserved_submdats', 'partial_submdats'])
        for sbm_id in view_data['preserved_submdats'] + view_data['partial_submdats']:
            sbm_name = self.db.get_dsmap(key='id', value=sbm_id)
            if self.loaded_submdats.get(sbm_name) is not None:
                continue  # Already loaded: never re-open LMDB environments
            self._load_submdat(sbm_name)

    # BACKWARD-COMPAT(internal): legacy layout, see note above _ensure_loaded
    def add_submdat(self, submdat_name: str, compression_levels: Dict = {}, map_sizes: Dict = {},
                    db_types: List[str] = ['meta', 'data'], data_type: str = 'text', data_key: str = 'text',
                    modality: str = 'text', hybrid_data: List = [],
                    reshuffle: bool = False, round_robin_insertion: bool = False, reshuffle_at_the_end: bool = False) -> MDat:
        """
        Add a new *internal* (legacy layout) dataset to this container.
        Args:
            submdat_name: Name of the dataset
            compression_levels: {"db_name": compression_level} (0-9)
            map_sizes: {"db_name": map_size} LMDB map size (default: 1<<33)
            db_types: Databases to create (default ['meta','data'])
            data_type: Data type
            data_key: Original key of the data field (ex. 'text')
            modality: Modality of the data
        """
        if self.readonly:
            raise MDatIsReadOnly
        self._check_dsmap_capacity()
        if submdat_name in self.list_submdat():
            raise NameAlreadyUsed(f"Dataset name '{submdat_name}' already used")
        self.loaded_submdats[submdat_name] = None  # Placeholder allowing creation
        new = MDat.create_submdat(self, submdat_name=submdat_name,
                                  compression_levels=compression_levels, map_sizes=map_sizes,
                                  db_types=db_types, data_type=data_type, modality=modality,
                                  hybrid_data=hybrid_data)
        self.loaded_submdats[submdat_name] = new
        return new

    def _check_dsmap_capacity(self):
        max_submdat = 255 if self._get_manifest_attr('dsmap_bits', 8) == 8 else 65535
        if self.db.get_dsmap('max_id') >= max_submdat:
            raise TooManySubmDats

    def list_submdat(self) -> List[str]:
        """List all dataset names (internal and linked)."""
        return list(self.db.get_dsmap(key='name').keys())

    def get_submdat(self, submdat_name: str) -> MDat:
        """Get a loaded dataset by name."""
        if submdat_name in self.loaded_submdats:
            return self.loaded_submdats[submdat_name]
        raise SubmDatNotFound(f"Dataset '{submdat_name}' is not loaded")

    #  linked datasets
    def link_dataset(self, path: str, alias: str = None) -> Dict[str, Any]:
        """
        Register a standalone MDat folder (created with MDat.create_standalone)
        under a local alias. Nothing is copied: only a relative path is stored,
        so the same dataset folder can be linked by many containers (possibly
        on many machines) and moved around by relocating $MATFORMER_MDAT_ROOT.

        The dataset's stats, strategies, precomputed lengths and storage-db
        configuration are mirrored into the container catalog (refreshed at
        every load). The dataset is added to the 'default' view.

        Returns the mirror summary dict.
        """
        if self.readonly:
            raise MDatIsReadOnly
        resolved = os.path.realpath(path)
        if not self._is_catalog_folder(resolved):
            raise MDatNotFound(f"No dataset catalog (mdat_container.db / mdat.db) found at {path}")
        ext = self._open_linked_container(resolved)
        ext_names = ext.list_submdat()
        if len(ext_names) != 1:
            raise ValueError(f"{path} is not a standalone dataset folder ({len(ext_names)} datasets found). Use a folder created with MDat.create_standalone().")
        ext_name = ext_names[0]
        alias = sanify_name(alias or ext_name)
        if alias in self.list_submdat():
            raise NameAlreadyUsed(f"Dataset name '{alias}' already used in this container")
        self._check_dsmap_capacity()

        ext_row = ext.db.get_submdat_row(ext_name)
        uuid = ext_row.get('uuid')
        if not uuid:
            import uuid as uuid_lib
            uuid = uuid_lib.uuid4().hex
            ext.db.update_submdat_row(ext_row['ds_map_id'], {'uuid': uuid})
        stored_rel = self._store_relative_link(resolved)
        local_id = self.db.add_submdat_link(
            alias, stored_rel, uuid=uuid,
            data_type=ext_row.get('data_type'), modality=ext_row.get('modality'),
            data_key=ext_row.get('data_key'),
            document_number=ext_row.get('document_number') or 0,
            generation=ext_row.get('generation') or 0,
        )
        self._mirror_linked_dataset(local_id, ext, ext_name)
        self._update_default_view(local_id)
        return self.db.get_submdat_row(local_id)

    def unlink_dataset(self, alias: str) -> None:
        """
        Remove a *linked* dataset from this container. The external dataset
        folder is never touched. Views that used it lose its rows: their shuffle
        files become invalid and must be recreated (see repair()).
        """
        if self.readonly:
            raise MDatIsReadOnly
        row = self.db.get_submdat_row(alias)
        if not row:
            raise SubmDatNotFound(f"Dataset '{alias}' not found in this container")
        if not row.get('source_rel_path'):
            raise ValueError(f"'{alias}' is an internal dataset. Use remove_submdat(..., delete_data=True) instead.")
        self.db.remove_submdat(row['ds_map_id'])
        self.loaded_submdats.pop(alias, None)
        print(f"Unlinked dataset '{alias}'. Shuffle files/views that referenced it are now invalid: run repair() to rebuild them.")

    def remove_submdat(self, submdat_name: str, delete_data: bool = False) -> None:
        """
        Remove any dataset from this container.

        Internal datasets: their catalog rows (and cascaded strategy stats,
        view links, precomputes) are deleted; with delete_data=True the
        dataset folder and its pretokenization folders are deleted too.
        Linked datasets: unlink only (the external folder is never deleted).
        Either way, shuffle files/views referencing it become invalid.
        """
        if self.readonly:
            raise MDatIsReadOnly
        row = self.db.get_submdat_row(submdat_name)
        if not row:
            raise SubmDatNotFound(f"Dataset '{submdat_name}' not found in this container")
        if row.get('source_rel_path'):
            self.unlink_dataset(submdat_name)
            return
        if delete_data:
            sbm_folder = os.path.join(self.dataset_path, submdat_name)
            shutil.rmtree(sbm_folder, ignore_errors=True)
            for strategy in self.list_strategies():
                shutil.rmtree(os.path.join(self.pretok_path, strategy, submdat_name), ignore_errors=True)
        self.db.remove_submdat(row['ds_map_id'])
        self.loaded_submdats.pop(submdat_name, None)
        print(f"Removed dataset '{submdat_name}' from the container. Views/shuffle files referencing it are invalid: run repair().")

    # linked infrastructure
    def _store_relative_link(self, dataset_path: str) -> str:
        """Most portable relative representation of an external dataset path:
        against $MATFORMER_MDAT_ROOT when possible, container-relative
        otherwise (absolute paths are kept only when no relation exists)."""
        env_root = os.environ.get(MDAT_ROOT_ENV)
        if env_root:
            env_root = os.path.realpath(env_root)
            if dataset_path.startswith(env_root + os.sep):
                return os.path.relpath(dataset_path, env_root)
        try:
            return os.path.relpath(dataset_path, os.path.realpath(self.mdat_path))
        except ValueError:
            return dataset_path 

    @staticmethod
    def _is_catalog_folder(path: str) -> bool:
        """True if the folder holds a container/dataset catalog (either name)."""
        return (os.path.exists(os.path.join(path, MDatContainer.DB_FILENAME)) or
                os.path.exists(os.path.join(path, MDatContainer.LEGACY_DB_FILENAME)))

    def _resolve_linked_path(self, stored: str) -> str:
        """Resolve a stored relative link to an existing dataset folder."""
        candidates = []
        if os.path.isabs(stored):
            candidates.append(stored)
        else:
            env_root = os.environ.get(MDAT_ROOT_ENV)
            if env_root:
                candidates.append(os.path.join(env_root, stored))
            candidates.append(os.path.join(self.mdat_path, stored))
        for candidate in candidates:
            if self._is_catalog_folder(candidate):
                return os.path.realpath(candidate)
        raise MDatNotFound(
            f"Linked dataset not found. Tried: {[os.path.realpath(c) for c in candidates]}. "
            f"(Set ${MDAT_ROOT_ENV} if the dataset filesystem moved.)")

    def _open_linked_container(self, resolved_path: str) -> 'MDatContainer':
        """Open (once per process, per path) the catalog of a linked dataset.
        Catalog-only: the host container opens the LMDB environments itself."""
        cached = MDatContainer._linked_container_cache.get(resolved_path)
        if cached is not None:
            return cached
        ext = MDatContainer.load_catalog(resolved_path, readonly=False)
        MDatContainer._linked_container_cache[resolved_path] = ext
        return ext

    def _load_linked_submdat(self, alias: str, row: Dict[str, Any]) -> MDat:
        resolved = self._resolve_linked_path(row['source_rel_path'])
        ext = self._open_linked_container(resolved)
        ext_name = ext.list_submdat()[0]
        sbm = MDat.from_external(self, alias=alias, row=ext.db.get_submdat_row(ext_name),
                                 external_root=resolved, external_db=ext.db, external_name=ext_name,
                                 external_container=ext)
        self._mirror_linked_dataset(row['ds_map_id'], ext, ext_name)
        return sbm

    def _mirror_linked_dataset(self, local_id: int, ext: 'MDatContainer', ext_name: str):
        """Refresh the container's mirrors of an external dataset: storage-db
        configuration (so document/byte totals work locally), strategy
        definitions, per-strategy stats, precomputed lengths and output-db
        rows, remapped to this container's ids."""
        ext_id = ext.db.get_dsmap(key='name', value=ext_name)
        # Storage databases ('data'/'meta'/extras) as local mirrors: needed by
        # get_submdat_info / view byte totals. LMDB files themselves stay in
        # the external folder.
        c = self.db.connect().cursor()
        c.execute("DELETE FROM submdat_database WHERE submdat_id=?", (local_id,))
        for dbs in ext.db.get_submdat_databases(ext_id):
            mirrored_db_id = self.db.upsert_mirrored_database_ref(dbs)
            self.db.link_submdat_database(local_id, mirrored_db_id,
                                          raw_data_bytes=dbs.get('raw_data_bytes') or 0,
                                          is_data=dbs['is_data'], is_meta=dbs['is_meta'],
                                          is_extra=dbs['is_extra'])
        for strategy_name in ext.db.list_strategies():
            strategy_row = ext.db.get_strategy_row(strategy_name)
            stats = ext.db.get_strategy_submdats(strategy_name, ext_id)
            if not stats:
                continue  # strategy not applied to this dataset
            self.db.upsert_strategy_row(strategy_row)
            self.db.upsert_strategy_submdat(strategy_name, local_id, stats)
            for precomp in ext.db.get_strategy_precomp(strategy_name=strategy_name, submdat_id=ext_id):
                self.db.set_strategy_submdat_precomp(strategy_name, local_id, precomp['multiplier'], precomp['precomputed_length'])
            for sdb in ext.db.get_strategy_dbs(strategy_name, ext_id):
                mirrored_db_id = self.db.upsert_mirrored_database_ref(sdb)
                self.db.upsert_mirrored_strategy_db(mirrored_db_id, strategy_name, local_id, sdb)

    def check_linked_datasets(self, verbose: bool = True) -> Dict[str, List[str]]:
        """
        Compare every linked dataset's live state (generation, document count)
        with the mirror taken at link/refresh time. Prints a warning when a
        container is stale (its shuffle files / views may be invalid) and
        returns {alias: [issues]}.
        """
        issues: Dict[str, List[str]] = {}
        for row in self.db.list_submdats():
            stored = row.get('source_rel_path')
            if not stored:
                continue
            alias = row['name']
            try:
                resolved = self._resolve_linked_path(stored)
                ext = self._open_linked_container(resolved)
                ext_row = ext.db.get_submdat_row(ext.list_submdat()[0])
            except MDatNotFound as e:
                issues[alias] = [str(e)]
                continue
            mirrored_gen = row.get('generation') or 0
            live_gen = ext_row.get('generation') or 0
            mirrored_docs = row.get('document_number') or 0
            live_docs = ext_row.get('document_number') or 0
            if live_gen != mirrored_gen or live_docs != mirrored_docs:
                issues[alias] = [f"linked dataset changed: {live_docs} documents now vs {mirrored_docs} recorded "
                                 f"(generation {mirrored_gen} -> {live_gen})"]
        if issues and verbose:
            print("=" * 70)
            print("WARNING: this container is OUT OF SYNC with one or more linked datasets:")
            for alias, problem in issues.items():
                print(f"  - {alias}: {'; '.join(problem)}")
            print("Shuffle files, views and distributed precomputes that involve them may be invalid.")
            print("Fix automatically with: container.repair()   (syncs mirrors and reshuffles views)")
            print("=" * 70)
        return issues

    def sync_linked_datasets(self) -> Dict[str, Any]:
        """Refresh mirrors and recorded generation/document counts of all
        linked datasets. Returns {alias: 'ok'|'missing'}."""
        status = {}
        for row in self.db.list_submdats():
            stored = row.get('source_rel_path')
            if not stored:
                continue
            alias = row['name']
            try:
                resolved = self._resolve_linked_path(stored)
            except MDatNotFound:
                status[alias] = 'missing'
                continue
            ext = self._open_linked_container(resolved)
            ext_name = ext.list_submdat()[0]
            ext_row = ext.db.get_submdat_row(ext_name)
            self._mirror_linked_dataset(row['ds_map_id'], ext, ext_name)
            self.db.update_submdat_row(row['ds_map_id'], {
                'document_number': ext_row.get('document_number') or 0,
                'generation': ext_row.get('generation') or 0,
            })
            status[alias] = 'ok'
        return status

    def repair(self, reshuffle_views: bool = True) -> List[str]:
        """
        Automatic fix after linked datasets changed (or were removed):
        1. syncs all mirrors (stats, strategies, precomputed lengths);
        2. drops distributed precomputes (they must be recomputed);
        3. reshuffles every view that had a shuffle file.
        Returns the list of reshuffled views.
        """
        if self.readonly:
            raise MDatIsReadOnly
        self.sync_linked_datasets()
        self.db.connect().execute("DELETE FROM pretok_strategy_view_distributed_length")
        reshuffled = []
        if reshuffle_views:
            for view_name in self.list_views():
                if self.db.get_view_row(view_name).get('shuffle_struct_format'):
                    self.set_view(view_name)  # loads the datasets this view needs
                    struct_format = self.shuffle(view_name)
                    self.db.update_view_row(view_name, {'shuffle_struct_format': struct_format})
                    reshuffled.append(view_name)
        print(f"Repair done. Reshuffled views: {reshuffled or 'none'}")
        return reshuffled

    # ---------------------------------------------------- views and shuffling
    def shuffle(self, view_name=None):
        """
        Create (or recreate) the shuffled pointer file of a view.
        Instead of shuffling the data, a file of pointers is written in the
        shuffling folder: one (submdat_id, doc_id) struct per selected
        document. submdat_id width follows dsmap_bits, doc_id width follows the
        largest dataset. Views with partially-selected datasets stop early once
        every per-dataset criterion (documents or bytes) is satisfied.

        Returns the struct format string of the pointer file.
        """
        if self.readonly:
            raise MDatIsReadOnly

        # 1. Interrogate the view to determine the preserved/partial submdats
        partial_ids = self.db.ask_view(view=view_name, keys='partial_submdats')
        preserved_ids = self.db.ask_view(view=view_name, keys='preserved_submdats')
        assert len(partial_ids) + len(preserved_ids) > 0, "You are trying to shuffle an empty view."
        terminate_early = len(partial_ids) > 0
        all_selected_ids = (partial_ids + preserved_ids)

        # 2. Mapping between "local ids" (random permutation space) and real ids
        all_selected_ids.sort()
        mapping = {k: v for (k, v) in enumerate(all_selected_ids)}

        ds_map = self.db.get_dsmap(key='id')
        # 3. Datasets length and cumulative length (for bisect_right)
        totlen = 0
        cu_dslens = [0]
        selected_lengths = []
        for i in range(0, len(all_selected_ids)):
            L = len(self._ensure_loaded(ds_map[mapping[i]]))
            totlen += L
            selected_lengths.append(L)
            cu_dslens.append(totlen)
        maxlen = max(selected_lengths) if selected_lengths else 0

        # 4. Pointers struct size
        doc_id_type = 'B' if maxlen <= 255 else 'H' if maxlen <= 65535 else 'I' if maxlen <= 4294967295 else 'Q'
        submdat_id_type = 'B' if self._get_manifest_attr('dsmap_bits', 8) == 8 else 'H'
        struct_format = submdat_id_type + doc_id_type

        # 5. Output path
        os.makedirs(self.shuffling_path, exist_ok=True)
        shuffle_path = os.path.join(self.shuffling_path, f'{view_name}.mdat')

        # 6. Random permutator over the whole selection
        permutator = RandomPermutator(totlen)

        # Byte-based termination reads raw documents; force a data-only return
        # configuration on the involved datasets for the whole shuffle.
        involved = [self._ensure_loaded(ds_map[sbm_id]) for sbm_id in all_selected_ids]
        saved_wanted = [(sbm, sbm.default_wanted_from_dbs, sbm.default_wanted_from_strategy, sbm.default_raw)
                        for sbm in involved]
        for sbm in involved:
            sbm.set_default_wanted(from_dbs=['data'], from_strategy=[])
        try:
            terminators = {}
            if terminate_early:
                # 7. "Terminators": counters that stop the selection early
                for sbm_id in all_selected_ids:
                    # sel_info contains at least 'bytes_criteria' and 'document_number'
                    sel_info = self.db.get_view_submdat(view_name=view_name, submdat_id=sbm_id)
                    if sbm_id in preserved_ids:
                        criteria = 'L'
                        terminator = len(self._ensure_loaded(ds_map[sbm_id]))
                    else:
                        bc = sel_info.get('bytes_criteria')
                        dn = sel_info.get('document_number')
                        if bc is not None and bc > 0:
                            criteria = 'B'
                            terminator = int(bc)
                        elif dn is not None and dn > 0:
                            criteria = 'L'
                            terminator = int(dn)
                        else:
                            print(f"View '{view_name}': no usable criterion for submdat {ds_map[sbm_id]}, skipping it")
                            continue
                    terminators[sbm_id] = (0, 0, terminator, criteria)

                def accumulator_equal_terminator(_id):
                    accL, accB, limit, crit = terminators[_id]
                    if crit == 'L':
                        return 1 if accL >= limit else 0
                    if crit == 'B':
                        return 1 if accB >= limit else 0
                    raise ValueError(f"Unknown terminator criterion: {crit}")

                def accumulators_equal_terminators():
                    return all(accumulator_equal_terminator(k) for k in terminators)

                # 8. Shuffling the pointers with early termination
                with open(shuffle_path, 'wb') as f:
                    pbar = tqdm(range(totlen))
                    for idx in pbar:
                        permuted_idx = permutator(idx)
                        loc_id = bisect_right(cu_dslens, permuted_idx) - 1
                        global_id = mapping[loc_id]
                        doc_id = permuted_idx - cu_dslens[loc_id]
                        if accumulators_equal_terminators():
                            break
                        if accumulator_equal_terminator(global_id):
                            continue
                        accL, accB, term, crit = terminators[global_id]
                        accL += 1
                        if crit == 'B':
                            accB += len(self._ensure_loaded(ds_map[global_id]).raw_bytes(doc_id))
                        terminators[global_id] = (accL, accB, term, crit)
                        postfix = {}
                        doc_terms = [t for t in terminators.values() if t[3] == 'L']
                        byte_terms = [t for t in terminators.values() if t[3] == 'B']
                        if doc_terms:
                            postfix['docs'] = f"{sum(t[0] for t in doc_terms)}/{sum(t[2] for t in doc_terms)}"
                        if byte_terms:
                            postfix['MB'] = f"{sum(t[1] for t in byte_terms)/(1024**2):.1f}/{sum(t[2] for t in byte_terms)/(1024**2):.1f}"
                        pbar.set_postfix(postfix)
                        f.write(struct.pack(struct_format, global_id, doc_id))

                # Record the actually selected documents/bytes per submdat
                for sbm_id, (accL, accB, term, crit) in terminators.items():
                    if sbm_id in preserved_ids:
                        submdat_info = self.db.get_submdat_info(submdat_id=sbm_id)
                        update_dict = {'document_number': submdat_info.get('document_number'),
                                       'bytes_criteria': submdat_info.get('raw_data_bytes')}
                    else:
                        update_dict = {'document_number': accL, 'bytes_criteria': accB}
                    self.db.update_view_submdat(view_name=view_name, submdat_id=sbm_id, data=update_dict)
            else:
                # Full shuffle: every document of every selected submdat
                with open(shuffle_path, 'wb') as f:
                    for idx in tqdm(range(totlen)):
                        permuted_idx = permutator(idx)
                        loc_id = bisect_right(cu_dslens, permuted_idx) - 1
                        global_id = mapping[loc_id]
                        doc_id = permuted_idx - cu_dslens[loc_id]
                        f.write(struct.pack(struct_format, global_id, doc_id))
        finally:
            for sbm, dbs, strat, raw in saved_wanted:
                sbm.set_default_wanted(from_dbs=dbs, from_strategy=strat, raw=raw)

        self.db.update_view_row(view_name, {"shuffle_struct_format": struct_format})
        # The pointer file changed: drop any stale reader, and refresh the
        # active-view flag so iteration can start right after shuffling.
        reader = getattr(self, '_shuffle_reader', None)
        if reader is not None:
            reader.close()
            self._shuffle_reader = None
        if view_name == getattr(self, '_active_view', None):
            self._active_view_shuffled = True
        return struct_format

    def _update_default_view(self, submdat_id):
        submdat_info = self.db.get_submdat_info(submdat_id=submdat_id)
        self.db.link_submdat_view(submdat_id=submdat_id, view_name='default', is_skipped=0, is_partial=0, is_preserved=1,
                                  document_number=submdat_info.get('document_number'),
                                  bytes_criteria=submdat_info.get('raw_data_bytes'))

    def register_new_view(self, view_dict):
        """
        How views work in a container.

        Each view is initialized from a view descriptor, a Python dictionary:
        {
            "view_name": "example",
            "selection": {
                "dataset_1": "2G",       # 2 Gigabytes
                "dataset_2": 800,        # 800 Documents
                "dataset_3": "preserved",# All documents preserved
                "dataset_4": "80M",      # 80 Megabytes
                "dataset_5": 0.2,        # 20% of documents (proportion-based)
                "dataset_6": "skipped"   # Dataset excluded entirely
            },
            "shuffle": true/false        # Whether to shuffle right after creation
        }

        - The 'selection' field must include every dataset of the container.
          Each value can be:
            * an integer => number of documents to select;
            * a float in [0,1] => proportion of documents to select;
            * a string ending in K/M/G/T => byte-based selection (bytes are
              computed on the raw 'data' field) (e.g. '500M');
            * 'preserved' => all documents included;
            * 'skipped' => dataset excluded entirely.
        - Any partial selection (integer, float or bytes) forces shuffling,
          regardless of the explicit 'shuffle' parameter.
        """
        view_name = view_dict.get('view_name')
        selection = view_dict.get('selection', {})
        shuffle = bool(view_dict.get('shuffle', False))
        if self.readonly and shuffle:
            raise MDatIsReadOnly("shuffling a new view requires a writable container")

        if view_name in self.db.list_views():
            raise NameAlreadyUsed(f"View '{view_name}' already exists. Remove it first with remove_view().")

        all_submdats = list(self.list_submdat())
        name_to_id = {name: self.db.get_dsmap(key='name', value=name) for name in all_submdats}

        if set(all_submdats) != set(selection.keys()):
            raise ValueError("Selection must include all available datasets")

        targets = {}
        partial_submdats = []
        skipped_submdats = []
        preserved_submdats = []
        submdat_bytes_decided = []
        for name, value in selection.items():
            ds_id = name_to_id[name]
            if isinstance(value, str):
                sval = value.strip().lower()
                if sval == 'skipped':
                    skipped_submdats.append(name)
                elif sval == 'preserved':
                    preserved_submdats.append(name)
                elif sval[-1].upper() in ('K', 'M', 'G', 'T'):
                    submdat_bytes_decided.append(ds_id)
                    mult = {'K': 1024, 'M': 1024**2, 'G': 1024**3, 'T': 1024**4}[sval[-1].upper()]
                    targets[ds_id] = int(float(sval[:-1]) * mult)
                    partial_submdats.append(name)
                else:
                    raise ValueError(f"Invalid selection value for '{name}': {value}")
            elif isinstance(value, (int, float)):
                targets[ds_id] = value
                partial_submdats.append(name)
            else:
                raise ValueError(f"Unsupported selection type for '{name}': {type(value)}")

        # Create view in database
        self.db.add_view(view_name)
        ds_map = self.db.get_dsmap(key='name')

        for sbm in all_submdats:
            ds_id = ds_map[sbm]
            bytes_criteria = 0
            partial_doc_number = 0

            if sbm in skipped_submdats:
                doc_number = 0
                bytes_criteria = 0
            elif sbm in preserved_submdats:
                doc_number = len(self._ensure_loaded(sbm))
                bytes_criteria = self.db.get_submdat_info(ds_id)['raw_data_bytes']
            elif ds_id in targets:
                val = targets[ds_id]
                if ds_id in submdat_bytes_decided:
                    partial_doc_number = 0
                    bytes_criteria = int(val)
                elif isinstance(val, int):
                    partial_doc_number = val
                elif isinstance(val, float):
                    partial_doc_number = int(len(self._ensure_loaded(sbm)) * val)
                else:
                    raise ValueError(f"Unexpected target value for {sbm}: {val}")

                doc_number = partial_doc_number
            else:
                raise ValueError(f"Unexpected missing selection for submdat: {sbm}")

            self.db.link_submdat_view(
                ds_id, view_name,
                sbm in skipped_submdats,
                sbm in partial_submdats,
                sbm in preserved_submdats,
                doc_number,
                bytes_criteria
            )

        if partial_submdats:
            if not shuffle:
                print("Partial selections detected; shuffling enforced for random document selection.")
            shuffle = True

        shuffle_struct_format = self.shuffle(view_name) if shuffle else None
        self.db.update_view_row(view_name, {"shuffle_struct_format": shuffle_struct_format})
        return self.db.get_view_row(view_name)

    def remove_view(self, view_name):
        """Delete a view, its shuffle file and its precomputes."""
        if self.readonly:
            raise MDatIsReadOnly
        if view_name == 'default':
            raise ValueError("The 'default' view cannot be removed")
        self.db.remove_view(view_name)
        shuffle_file = os.path.join(self.shuffling_path, f'{view_name}.mdat')
        if os.path.exists(shuffle_file):
            os.remove(shuffle_file)

    def _get_shuffle_reader(self) -> _ShuffleFileReader:
        """Lazily open the pointer-file reader of the active view."""
        reader = getattr(self, '_shuffle_reader', None)
        if reader is not None:
            return reader
        active = getattr(self, '_active_view', None)
        if not active:
            raise MDatNotShuffled("No active view set. Call set_view().")
        view_meta = self.db.get_view_row(active)
        if not view_meta:
            raise ValueError(f"View '{active}' not registered")
        struct_format = view_meta.get('shuffle_struct_format')
        if not struct_format:
            raise MDatNotShuffled(
                f"View '{active}' is not shuffled. Fix: ds.shuffle('{active}') "
                f"(or load_dataset(..., shuffle=True), or in the mdat shell: `shuffle {active}`).")
        self._shuffle_reader = _ShuffleFileReader(
            os.path.join(self.shuffling_path, f'{active}.mdat'), struct_format)
        return self._shuffle_reader

    def _reset_document_pointer(self):
        """Reset document index and file pointer."""
        self._seek_document_pointer(0)

    def _seek_document_pointer(self, index):
        """Go to document index inside the shuffle file (if the view has one)."""
        self.document_index = index
        reader = getattr(self, '_shuffle_reader', None)
        if reader is None and getattr(self, '_active_view_shuffled', False):
            reader = self._get_shuffle_reader()
        if reader is not None:
            print("Seeking into shuffle file...")
            reader.seek_document(index)

    def set_view(self, view_name):
        """Set the active view (closing the previous shuffle-file reader)."""
        reader = getattr(self, '_shuffle_reader', None)
        if reader is not None:
            reader.close()
            self._shuffle_reader = None

        if view_name is None:
            raise ValueError("A valid view name must be provided")

        view_meta = self.db.get_view_row(view_name)
        if not view_meta:
            raise ValueError(f"View '{view_name}' not found")

        self._active_view = view_name
        self.current_view = view_name
        self.document_index = 0
        self.len = self.db.get_view_totals(view_name=view_name)['document_number']
        self._active_view_shuffled = view_meta.get('shuffle_struct_format') is not None
        self._populate_submdat()

    # ------------------------------------------------ distributed precompute
    def precompute_chunks_distributed(self, num_workers=840):
        """
        Precompute chunk counts for distributed training.

        Args:
            num_workers: Number of workers to precompute for (should be highly composite)
        """
        original_modality = self.current_iteration_modality
        self.set_iteration_modality('chunks')

        worker_chunk_counts = [0] * num_workers
        self.__iter__(with_prefetch=False)
        doc_index = 0

        with tqdm(total=len(self)) as pbar:
            try:
                while True:
                    doc = next(self)
                    worker_id = doc_index % num_workers
                    num_chunks = len(doc['object']['chunks'])
                    chunks_yielded = math.ceil(num_chunks / self.chunk_multiplier) if num_chunks > 0 else 0
                    worker_chunk_counts[worker_id] += chunks_yielded
                    doc_index += 1
                    pbar.update(1)
            except StopIteration:
                pass

        self.set_iteration_modality(original_modality)

        self.db._store_distributed_precomputed_length(
            view_name=self.current_view,
            strategy_name=self.current_strategy.strategy_name,
            chunk_multiplier=self.chunk_multiplier,
            num_workers=num_workers,
            worker_chunk_counts=worker_chunk_counts
        )

        return {
            'num_workers': num_workers,
            'total_chunks': sum(worker_chunk_counts),
            'min_chunks': min(worker_chunk_counts),
            'max_chunks': max(worker_chunk_counts)
        }

    # ------------------------------------------------------ prefetch & loading
    def start_prefetch(self, max_prefetch=32768, shuffled=True):
        """
        Start a background thread that fills a queue with upcoming documents.
        The thread reads the (single) shuffle-file reader sequentially; the
        main thread must not read or seek it while prefetching.
        """
        if getattr(self, "_prefetch_thread", None) is not None:
            return

        if shuffled and not getattr(self, '_active_view_shuffled', None):
            raise MDatNotShuffled(
                f"View '{getattr(self, 'current_view', None)}' is not shuffled: training/iteration requires "
                "a shuffled view. Fix: ds.shuffle(view) on a writable container, or in the mdat shell: `shuffle <view>`.")

        if not hasattr(self, 'prefetch_index'):
            self.prefetch_index = self.document_index

        if shuffled:
            self._get_shuffle_reader()

        ds_map = self.db.get_dsmap(key='id')

        self._prefetch_queue = queue.Queue(max_prefetch)
        self._prefetch_stop = threading.Event()

        def _prefetch_loop():
            while not self._prefetch_stop.is_set():
                if self._prefetch_queue.full():
                    time.sleep(0.001)
                    continue
                try:
                    doc, self.prefetch_index = self._load_next_document_inner(ds_map, self.prefetch_index, shuffled)
                    self._prefetch_queue.put((self.prefetch_index, doc))
                except StopIteration:
                    self._prefetch_queue.put(None)  # Sentinel for exhausted dataset
                    return
                except Exception as e:
                    print(f"[Prefetch Thread Error] {e}")
                    self._prefetch_queue.put(None)
                    return

        t = threading.Thread(target=_prefetch_loop, daemon=True)
        t.start()
        print(f"Prefetch thread {t} started with queue size {max_prefetch}. Increase or decrease this value according to your RAM to improve performance.")
        self._prefetch_thread = t

    def stop_prefetch(self):
        if getattr(self, "_prefetch_thread", None) is None:
            return

        self._prefetch_stop.set()

        queue_ = getattr(self, "_prefetch_queue", None)
        if queue_ is not None:
            while True:  # Draining the queue
                try:
                    queue_.get_nowait()
                except queue.Empty:
                    break

        self._prefetch_thread.join(timeout=2.0)
        self._prefetch_thread = None
        self._prefetch_queue = None

    def load_next_document(self, shuffled=True):
        if shuffled and not getattr(self, '_active_view_shuffled', None):
            raise MDatNotShuffled(
                f"View '{getattr(self, 'current_view', None)}' is not shuffled. "
                "Fix: ds.shuffle(view) on a writable container, or in the mdat shell: `shuffle <view>`.")

        if not hasattr(self, 'document_index'):
            self.document_index = 0

        if getattr(self, "_prefetch_queue", None) is not None:
            while True:
                try:
                    from_queue = self._prefetch_queue.get(timeout=15)
                    if from_queue is None:
                        self.stop_prefetch()
                        raise StopIteration
                    prefetch_index, doc = from_queue
                    self.document_index = prefetch_index  # The global document index is updated here
                    return doc
                except queue.Empty:
                    if self._prefetch_thread is None or not self._prefetch_thread.is_alive():
                        self.stop_prefetch()
                        raise RuntimeError("Prefetch thread died unexpectedly.")
        else:
            ds_map = self.db.get_dsmap(key='id')
            if shuffled:
                self._get_shuffle_reader()
            current_document, self.document_index = self._load_next_document_inner(ds_map, self.document_index, shuffled)
            return current_document

    def _load_next_document_inner(self, ds_map, local_document_index, shuffled=True):
        """Fetch the document at local_document_index (shuffled or sequential).

        In distributed mode, shuffle entries not belonging to this rank are
        skipped (index still advances, so all ranks stay in lockstep).
        """
        if shuffled:
            reader = self._get_shuffle_reader()
            while True:
                pair = reader.read_pair()
                if pair is None:
                    raise StopIteration
                if self.dist and local_document_index % self.world_size != self.rank_size:
                    local_document_index += 1
                    continue  # Skip this shuffle entry (other rank's document)
                submdat_id, doc_id = pair
                current_document = self.loaded_submdats[ds_map[submdat_id]][doc_id]
                local_document_index += 1
                return current_document, local_document_index
        current_pos = 0
        for submdat_id in sorted(ds_map.keys()):
            submdat_name = ds_map[submdat_id]
            if submdat_name in self.loaded_submdats:
                submdat_len = len(self.loaded_submdats[submdat_name])
                if local_document_index < current_pos + submdat_len:
                    doc_id = local_document_index - current_pos
                    current_document = self.loaded_submdats[submdat_name][doc_id]
                    return current_document, local_document_index + 1
                current_pos += submdat_len
        raise StopIteration

    # --------------------------------------------------------------- strategies
    def deregister_strategy(self, strategy_name: str, delete_files: bool = False) -> None:
        """
        Deregister a strategy from this container (its per-dataset stats and
        precomputes go with it). With delete_files=True the strategy's
        pretokenization folders under this container are deleted as well.
        Linked datasets keep their own copy of the strategy: it reappears at
        the next sync if it is still applied there.
        Args:
            strategy_name: Strategy name to remove
        """
        if self.readonly:
            raise MDatIsReadOnly
        self.db.remove_strategy(strategy_name)
        if delete_files:
            shutil.rmtree(os.path.join(self.pretok_path, strategy_name), ignore_errors=True)

    def register_strategy(self, strategy_name: str, strategy_dict: Dict[str, Any]) -> None:
        """
        Register a new strategy in this container.
        Args:
            strategy_name: Name of strategy
            strategy_dict: Strategy configuration
        """
        if self.readonly:
            raise MDatIsReadOnly
        strategy_name = sanify_name(strategy_name)
        if strategy_name in self.list_strategies():
            raise NameAlreadyUsed(f"Strategy '{strategy_name}' already registered")
        strategy = PretokenizationStrategy.from_dict(self.db, self.pretok_path, self.functions_path, strategy_name, strategy_dict)
        strategy.save()

    def strategies(self):
        return self.list_strategies()  # shortcut

    def list_views(self):
        return self.db.list_views()

    def list_strategies(self):
        return list(self.db.list_strategies())

    # ------------------------------------------------------------ state dict
    def get_state_dict(self):
        """All the pointers required to restart training mid-epoch."""
        state_dict = {
            "iteration_modality": self.current_iteration_modality,
            "document_index": getattr(self, "document_index", 0),
            "current_chunk_step": getattr(self, "current_chunk_step", 0),
            "_iteration_count": getattr(self, "_iteration_count", 0),
            "_recurrent_documents": getattr(self, "_recurrent_documents", []),
            "_recurrent_steps": getattr(self, "_recurrent_steps", []),
            "_max_iterations": getattr(self, "_max_iterations", None),
            "_cached_length": getattr(self, "_cached_length", None),
            "saved_world_size": 1 if not self.dist else self.world_size
        }
        return state_dict

    def restore_state_dict(self, state_dict):
        self.document_index = int(state_dict["document_index"])
        self.current_chunk_step = int(state_dict["current_chunk_step"])
        self._iteration_count = int(state_dict["_iteration_count"])
        self._recurrent_documents = list(state_dict["_recurrent_documents"])
        self._recurrent_steps = list(state_dict["_recurrent_steps"])
        self._max_iterations = None if state_dict["_max_iterations"] is None else int(state_dict['_max_iterations'])
        self._cached_length = None if state_dict["_cached_length"] is None else int(state_dict['_cached_length'])
        self._skip_reinit = True

    # ------------------------------------------------------------- iteration
    def __iter__(self, with_prefetch=True):
        # Reset the iteration (ex. for a new epoch)
        self.stop_prefetch()

        # Reset state (unless resuming mid-epoch via restore_state_dict)
        self.current_document = None
        if not hasattr(self, '_skip_reinit'):
            self.current_chunk_step = 0
            self.document_index = 0
            self._iteration_count = 0

            if hasattr(self, '_max_iterations'):
                del self._max_iterations

            # Rewind the pointer file for the new epoch
            if getattr(self, '_active_view_shuffled', False):
                self._get_shuffle_reader().seek_document(0)
            if with_prefetch and self.prefetch_buffer:
                self.start_prefetch(max_prefetch=self.prefetch_buffer)
        else:
            print("MDAT is resuming training from external logic.")
            print("Document index: ", self.document_index)
            print("Chunk step: ", self.current_chunk_step)
            print("Max iterations: ", self._max_iterations)
            print("Iteration count: ", self._iteration_count)
            self._seek_document_pointer(self.document_index)
            if with_prefetch and self.prefetch_buffer:
                self.start_prefetch(max_prefetch=self.prefetch_buffer)
        return self

    def _get_next_chunk_from_document(self, document, chunk_step):
        """
        Extract the next chunk (chunk_multiplier base chunks) of a document.
        Returns: (chunk, next_chunk_step, has_more_chunks)
        """
        if document is None or 'chunked_tokens' not in document:
            return None, 0, False
        chunks_list = document['chunked_tokens']
        total_chunks = len(chunks_list)
        if chunk_step >= total_chunks:
            return None, 0, False  # Document exhausted
        end_step = min(chunk_step + self.chunk_multiplier, total_chunks)
        selected_chunks = chunks_list[chunk_step:end_step]
        new_chunk_step = end_step
        chunk = np.concatenate(selected_chunks).tolist()
        if getattr(self, 'max_seq_len', None) is not None:
            if len(chunk) > self.max_seq_len:
                print(f"WARNING: A sequence is longer than max length ({len(chunk)} > {self.max_seq_len}) and it's truncated")
                chunk = chunk[:self.max_seq_len]
        has_more_chunks = new_chunk_step < total_chunks
        return chunk, new_chunk_step, has_more_chunks

    def __next__(self):
        if not hasattr(self, '_iteration_count'):
            self._iteration_count = 0
        if self.dist:
            if not hasattr(self, '_max_iterations'):
                self._max_iterations = len(self)
            if self._iteration_count >= self._max_iterations:
                raise StopIteration
        result = None
        is_same_document = False

        try:
            if self.current_iteration_modality in ['document', 'tokens', 'chunks']:  # Entire document
                self.current_document = self.load_next_document()
                result = self.current_document

            elif self.current_iteration_modality == 'chunked_tokens':  # Sequential chunked
                if not hasattr(self, 'current_chunk_step'):
                    self.current_chunk_step = 0
                    self.current_document = None

                while True:
                    is_document_exhausted = (self.current_document is None or
                                             'chunked_tokens' not in self.current_document or
                                             self.current_chunk_step >= len(self.current_document['chunked_tokens']))

                    if is_document_exhausted:
                        self.current_document = self.load_next_document()
                        self.current_chunk_step = 0
                        is_same_document = False
                    else:
                        is_same_document = True

                    result, self.current_chunk_step, has_more = self._get_next_chunk_from_document(self.current_document, self.current_chunk_step)
                    if result is not None:
                        break
            elif self.current_iteration_modality == 'chunked_for_recurrence_temp':

                if not hasattr(self, '_recurrent_documents'):
                    self._recurrent_documents = [None] * self.batch_size
                    self._recurrent_steps = [0] * self.batch_size
                    self._recurrent_has_more = [False] * self.batch_size

                    for i in range(self.batch_size):
                        doc = self.load_next_document()
                        self._recurrent_documents[i] = doc
                        self._recurrent_steps[i] = 0
                        self._recurrent_has_more[i] = True

                cell = self._iteration_count % self.batch_size
                doc = self._recurrent_documents[cell]
                step = self._recurrent_steps[cell]

                if not self._recurrent_has_more[cell]:
                    doc = self.load_next_document()
                    self._recurrent_documents[cell] = doc
                    step = 0
                    self._recurrent_steps[cell] = 0
                    self._recurrent_has_more[cell] = True
                    is_same_document = False
                else:
                    is_same_document = (step > 0)

                result, new_step, has_more = self._get_next_chunk_from_document(doc, step)

                if result is None:
                    raise RuntimeError(
                        f"Document in cell {cell} returned empty chunk unexpectedly."
                    )

                self._recurrent_steps[cell] = new_step
                self._recurrent_has_more[cell] = has_more
            elif self.current_iteration_modality == 'chunked_for_recurrence':
                if not hasattr(self, '_recurrent_documents'):
                    self._recurrent_documents = [None] * self.batch_size
                    self._recurrent_steps = [0] * self.batch_size
                    for i in range(self.batch_size):
                        self._recurrent_documents[i] = self.load_next_document()

                document_cell_index = self._iteration_count % self.batch_size
                current_document = self._recurrent_documents[document_cell_index]
                current_chunk_step = self._recurrent_steps[document_cell_index]

                is_document_exhausted = (current_document is None or
                                         'chunked_tokens' not in current_document or
                                         current_chunk_step >= len(current_document['chunked_tokens']))

                if is_document_exhausted:
                    self._recurrent_documents[document_cell_index] = self.load_next_document()
                    self._recurrent_steps[document_cell_index] = 0
                    is_same_document = False
                else:
                    is_same_document = current_chunk_step > 0

                while True:
                    result, new_step, has_more = self._get_next_chunk_from_document(
                        self._recurrent_documents[document_cell_index],
                        self._recurrent_steps[document_cell_index]
                    )
                    if result is not None:
                        break
                    self._recurrent_documents[document_cell_index] = self.load_next_document()
                    self._recurrent_steps[document_cell_index] = 0

                self._recurrent_steps[document_cell_index] = new_step

            else:
                raise ValueError(f"Unknown iteration modality: {self.current_iteration_modality}")
            return_dict = {"object": result, "modality": "text"}
            if self.wants_metadata and self.current_iteration_modality in ['chunked_tokens', 'chunked_for_recurrence']:
                # Metadata (everything except the pretokenized payload) accompanies chunked modalities
                return_dict['metadata'] = {x: self.current_document[x] for x in self.current_document if x != 'chunked_tokens'}
            self._iteration_count += 1
            if self.dist:
                return_dict["worker_has_finished"] = None
                self._last_item = result
            if self.current_iteration_modality == 'chunked_for_recurrence':
                return_dict["is_same_document"] = is_same_document
            return return_dict

        except StopIteration:
            if self.dist:
                self._iteration_count += 1
                if self.current_iteration_modality == 'chunked_for_recurrence':
                    return {"object": [0], "worker_has_finished": True, "modality": "text", "is_same_document": False}
                return {"object": [0], "worker_has_finished": True, "modality": "text"}
            else:
                raise

    def set_iteration_modality(self, modality, with_meta=False, return_raw=False, add_special_tokens=True):
        supported_modalities = ['document', 'tokens', 'chunked_tokens', 'strategy_default', 'chunks', 'chunked_for_recurrence']
        if modality in supported_modalities:
            self.current_iteration_modality = modality
        else:
            print(f"{modality} not in {supported_modalities}")
        if modality == 'document':
            wanted_from_strategy = []
            wanted_from_dbs = ['data']
            self.len = self.total_documents
        elif modality == 'tokens':
            wanted_from_strategy = ['tokens']
            wanted_from_dbs = []
            self.len = self.total_documents
        elif modality == 'chunks':
            wanted_from_strategy = ['chunks']
            wanted_from_dbs = []
            self.len = self.total_documents
        elif modality in ['chunked_tokens', 'chunked_for_recurrence']:
            wanted_from_strategy = ['chunked_tokens']
            wanted_from_dbs = []
            self.len = self.total_divided_chunks
            if modality == 'chunked_for_recurrence':
                assert self.batch_size is not None
        elif modality == 'strategy_default':
            wanted_from_strategy = ['strategy_default']
            wanted_from_dbs = []
            self.len = 0
        else:
            raise ValueError(f"Unsupported modality: {modality}")
        if with_meta:
            self.wants_metadata = True
            wanted_from_dbs.append('meta')
        else:
            self.wants_metadata = False
        # Propagate the requested composition to every loaded dataset
        for sbm in self.loaded_submdats.values():
            sbm.set_default_wanted(from_dbs=wanted_from_dbs, from_strategy=wanted_from_strategy,
                                   raw=return_raw, add_special_tokens=add_special_tokens)

    def get_iteration_modality(self):
        return self.current_iteration_modality

    def set_strategy(self, strategy_name, max_seq_len=None):
        # 1. Load the strategy object
        self.current_strategy = PretokenizationStrategy.from_dict(db=self.db, strategy_name=strategy_name,
                                                                  pretok_path=self.pretok_path,
                                                                  functions_path=self.functions_path)
        # 2. Set the chunk multiplier from the model's sequence length: it must
        #    be an exact multiple of the strategy's base chunk size.
        if max_seq_len is not None:
            base_chunk_size = self.current_strategy.chunk_size
            if base_chunk_size is not None:
                assert max_seq_len % base_chunk_size == 0, \
                    f"max_seq_len {max_seq_len} must be a multiple of the strategy base chunk size {base_chunk_size}"
                self.chunk_multiplier = (max_seq_len // base_chunk_size)
                self.max_seq_len = max_seq_len
        else:
            self.max_seq_len = self.current_strategy.chunk_size
            self.chunk_multiplier = 1
        # 3. Initialize counters
        self.total_chunks = 0
        self.total_tokens = 0
        self.total_divided_chunks = 0
        # 4. Accumulate stats over the datasets of the current view. For
        #    chunk-based strategies the iteration length comes from the
        #    precomputed lengths at the current chunk multiplier:
        #    preserved datasets use the per-dataset precomputes, partial ones
        #    the per-view precomputes.
        preserved_ids = self.db.ask_view(view=self.current_view, keys='preserved_submdats')
        partial_ids = self.db.ask_view(view=self.current_view, keys='partial_submdats')
        used_submdats = preserved_ids + partial_ids
        ds_map = self.db.get_dsmap(key='id')
        covered = 0
        missed = []
        for sbm_id in used_submdats:
            sbm = self[ds_map[sbm_id]]
            try:
                sbm.set_strategy(strategy_name)
                strategy_stats = self.db.get_strategy_submdats(strategy_name, sbm_id)
                self.total_chunks += strategy_stats.get('total_chunks', 0)
                self.total_tokens += strategy_stats.get('total_tokens', 0)
                if 'chunks' in self.current_strategy.returns:
                    precomp = self.db.get_strategy_precomp(
                        strategy_name=strategy_name,
                        submdat_id=sbm_id,
                        view_name=None if sbm_id in preserved_ids else self.current_view,
                        multiplier=self.chunk_multiplier)
                    if precomp:
                        self.total_divided_chunks += precomp[0].get('precomputed_length', 0)
                covered += 1
            except SubMdatMissesStrategy:
                missed.append(ds_map[sbm_id])
            except Exception as e:
                print(f"Error accumulating stats for submdat {ds_map[sbm_id]} with strategy {strategy_name}: {e}")
        if missed:
            print(f"Datasets not pretokenized with strategy '{strategy_name}' (excluded from this strategy): {missed}")
        if covered == 0:
            print(f"No dataset of view '{self.current_view}' is pretokenized with '{strategy_name}'.")
            self.current_strategy = None

    # ---------------------------------------------------------------- export
    def export_view(self, output_file, view='default', wanted='document', data_field_name='text', with_meta=True, limiter=None):
        """
        Export all the documents of a view into a JSONL file.
        Args:
            limiter: stop after this many documents (None = export everything)
        Returns:
            the number of documents written.
        """
        self.set_view(view)
        self._reset_document_pointer()
        self.set_iteration_modality(wanted, with_meta=with_meta)

        import orjson
        counter = 0
        try:
            with open(output_file, "wb") as f:
                for wrapped in tqdm(self):
                    if limiter is not None and counter >= limiter:
                        break
                    item = wrapped['object']
                    data = {
                        "group": item['submdat_name'],
                        data_field_name: item['data']
                    }
                    if with_meta:
                        meta = {k: v for k, v in item.items() if k not in ('submdat_name', 'data')}
                        data["meta"] = meta
                    f.write(orjson.dumps(data))
                    f.write(b"\n")
                    counter += 1
        finally:
            self.stop_prefetch()
        return counter

    def get_distributed_length_before_training(self, num_devices=1):
        """
        Lightning initializes the dataset without torch.distributed set when it
        first gathers the dataset info, however the dataset length is required
        for correct learning-rate scheduling. Given a target number of devices,
        return the maximum distributed length: the one that will actually be
        used during training.
        """
        if self.current_view is None:
            raise ValueError("No active view: call set_view() before asking for the distributed length.")
        if self.current_strategy is None:
            raise ValueError("No strategy set: call set_strategy() before asking for the distributed length "
                             "(distributed lengths are defined on chunked iterations).")
        if self.current_iteration_modality in ['chunked_tokens', 'chunked_for_recurrence']:
            try:
                max_len = 0
                for w in range(num_devices):
                    w_len = self.db.get_worker_length_distributed(
                        target_num_workers=num_devices, target_worker_id=w,
                        view_name=self.current_view, strategy_name=self.current_strategy.strategy_name,
                        chunk_multiplier=self.chunk_multiplier)
                    if w_len > max_len:
                        max_len = w_len
                return max_len
            except ValueError as e:
                raise ValueError(
                    f"{e}\nFix: on a writable container run "
                    f"ds.precompute_chunks_distributed(num_workers) with a value divisible by "
                    f"{num_devices} (shell command: `precompute-distributed {num_devices}`), then retry.") from None
        else:
            raise ValueError(
                f"Distributed length requested but the iteration modality is "
                f"'{self.current_iteration_modality}': distributed per-worker lengths only exist for "
                "'chunked_tokens'/'chunked_for_recurrence'. Set a chunked modality or train non-distributed.")

    def __len__(self):
        """
        The len behaves differently depending on the configuration:
        1) no strategy / non-chunked modality: the document number of the view;
        2) chunked modality: the number of iterations at the current
           chunk_multiplier (total_divided_chunks);
        3) distributed + chunked: the (all-reduced maximum) per-worker length.
        """
        if self.dist and self.current_iteration_modality in ['chunked_tokens', 'chunked_for_recurrence']:
            try:
                if self._cached_length is not None:
                    return self._cached_length
                this_worker_length = self.db.get_worker_length_distributed(
                    target_num_workers=self.world_size, target_worker_id=self.rank_size,
                    view_name=self.current_view, strategy_name=self.current_strategy.strategy_name,
                    chunk_multiplier=self.chunk_multiplier)
                print(f"WORKER {self.rank_size} got length {this_worker_length}")
                # Broadcast maximum across all workers so everyone agrees
                import torch.distributed as dist
                import torch
                if dist.is_initialized():
                    length_tensor = torch.tensor([this_worker_length], dtype=torch.long).cuda()
                    dist.all_reduce(length_tensor, op=dist.ReduceOp.MAX)
                    max_length = length_tensor.item()
                    print(f"Max length set to {max_length}")
                    self._cached_length = max_length
                    return max_length
                else:
                    return this_worker_length
            except ValueError:
                print("I need to precompute lengths for an incompatible workers set up")
                self.precompute_chunks_distributed(self.world_size)
                return self.__len__()
        else:
            return self.len

    def create_range_view(self, new_view_name: str, source_view: str = 'default',
                          start: int = 0, end: int = None) -> dict:
        """
        Create a new view slicing [start, end) documents from source_view's
        shuffle file, preserving their order (no re-shuffling). This is the
        building block of the "partition" feature and of head views.

        Per-multiplier iteration lengths for the new (partial) datasets are
        estimated proportionally from the source precomputes and stored as
        view precomputes, so chunked modalities get a usable len().

        Args:
            new_view_name: Name for the new view.
            source_view: View to slice from (must be shuffled).
            start: First document index (inclusive, default 0).
            end: Last document index (exclusive, None = to the end).
        """
        if self.readonly:
            raise MDatIsReadOnly("creating a range view requires a writable container")
        source_meta = self.db.get_view_row(source_view)
        if not source_meta:
            raise ValueError(f"Source view '{source_view}' not found.")
        struct_format = source_meta.get('shuffle_struct_format')
        if not struct_format:
            raise MDatNotShuffled(
                f"Source view '{source_view}' has no shuffle file, cannot slice it. "
                "Fix: shuffle it first (ds.shuffle('view'), or in the shell: `shuffle view`).")

        source_total = self.db.get_view_totals(view_name=source_view)['document_number']
        start = max(int(start or 0), 0)
        end = source_total if end is None else min(int(end), source_total)
        if end <= start:
            raise ValueError(f"Empty range [{start}, {end}) for view '{source_view}' ({source_total} documents).")
        if new_view_name in self.db.list_views():
            raise NameAlreadyUsed(f"View '{new_view_name}' already exists.")

        source_shuffle_path = os.path.join(self.shuffling_path, f'{source_view}.mdat')
        new_shuffle_path = os.path.join(self.shuffling_path, f'{new_view_name}.mdat')
        os.makedirs(self.shuffling_path, exist_ok=True)
        struct_size = struct.calcsize(struct_format)

        self.db.add_view(new_view_name)
        doc_counts = {}  # submdat_id -> documents selected in the window
        with open(source_shuffle_path, 'rb') as src, open(new_shuffle_path, 'wb') as dst:
            src.seek(start * struct_size)
            for _ in tqdm(range(end - start), desc=f"Building range view '{new_view_name}'"):
                raw = src.read(struct_size)
                if not raw:
                    break  # source file shorter than expected; stop cleanly
                dst.write(raw)
                submdat_id, doc_id = struct.unpack(struct_format, raw)
                doc_counts[submdat_id] = doc_counts.get(submdat_id, 0) + 1
        selected_total = sum(doc_counts.values())
        window = float(selected_total) / max(source_total, 1)

        # Datasets appearing in the window are "partial"; the others "skipped"
        for sbm_id in set(self.db.get_dsmap(key='id').keys()):
            if sbm_id in doc_counts:
                self.db.link_submdat_view(submdat_id=sbm_id, view_name=new_view_name,
                                          is_skipped=0, is_partial=1, is_preserved=0,
                                          document_number=doc_counts[sbm_id], bytes_criteria=0)
            else:
                self.db.link_submdat_view(submdat_id=sbm_id, view_name=new_view_name,
                                          is_skipped=1, is_partial=0, is_preserved=0,
                                          document_number=0, bytes_criteria=0)

        # Proportional view precomputed lengths for every strategy/multiplier
        # the source datasets have (so len() works in chunked modalities)
        precomp_rows = []
        for strategy_name in self.list_strategies():
            for sbm_id, docs_selected in doc_counts.items():
                sbm_total = (self.db.get_submdat_row(sbm_id).get('document_number') or 0)
                for precomp in self.db.get_strategy_precomp(strategy_name=strategy_name, submdat_id=sbm_id):
                    estimate = math.ceil((precomp['precomputed_length'] or 0) * docs_selected / max(sbm_total, 1))
                    precomp_rows.append((strategy_name, new_view_name, sbm_id,
                                         precomp['multiplier'], estimate))
        self.db.set_strategy_view_precomp_many(precomp_rows)

        self.db.update_view_row(new_view_name, {'shuffle_struct_format': struct_format})
        print(f"Range view '{new_view_name}': {selected_total} documents from '{source_view}'[{start}:{end}].")
        return self.db.get_view_row(new_view_name)

    def ensure_range_view(self, source_view: str, start: int = None, end: int = None) -> str:
        """Return the name of a range view slicing [start, end) of source_view,
        creating it on first use (deterministic name). Used by the
        partition_start/partition_end data configuration."""
        view_name = f"{source_view}_p{start if start is not None else 0}_{end if end is not None else 'end'}"
        if view_name not in self.db.list_views():
            self.create_range_view(view_name, source_view=source_view, start=start or 0, end=end)
        return view_name

    def create_head_view(self, n: int, new_view_name: str, source_view: str = 'default') -> dict:
        """New view with the first n documents of source_view (see create_range_view)."""
        return self.create_range_view(new_view_name, source_view=source_view, start=0, end=n)

    def __str__(self):
        stringa = ""
        stringa += ("\n---- Matformer Dataset Container ----")
        for k in self.manifest.keys():
            stringa += (f"\n{k}: {self.manifest[k]}")
        stringa += (f"\n Current strategy: {self.current_strategy}")
        stringa += (f"\n Total documents: {self.total_documents}")
        stringa += (f"\n Total tokens: {self.total_tokens}")
        stringa += (f"\n Total chunks: {self.total_chunks}")
        stringa += (f"\n Total divided chunks: {self.total_divided_chunks}")
        stringa += (f"\n Current iteration modality: {self.current_iteration_modality}")
        stringa += (f"\n Datasets: {self.list_submdat()}")
        return stringa


# Backward compatibility: the container used to be the only class in town.
MatformerDataset = MDatContainer
