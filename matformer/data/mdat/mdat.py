#!/usr/bin/env python3
"""Dataset orchestration, named views, and resumable document/chunk iteration.

Storage, preprocessing, and CLI implementations live in the neighboring modules.
Historical imports from this module remain supported.
"""
from __future__ import annotations

# Keep the historical direct-script CLI invocation working.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    # Otherwise this folder's lmdb.py shadows the external lmdb package.
    script_directory = Path(__file__).resolve().parent
    sys.path[:] = [entry for entry in sys.path if Path(entry).resolve() != script_directory]
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "matformer.data.mdat"

import math
import os
import copy
import shutil
import struct
import queue
import threading
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm
from torch.utils.data import IterableDataset, get_worker_info

from .sql_database import DatabaseManager
from .submdat import SubMdat
from .preprocessing import PretokenizationStrategy
from .utils import RandomPermutator, sanify_name
from .selection import DatasetSelection, SelectedView

from .exceptions import (
    MDatAlreadyPresent, MDatIsReadOnly, MDatNotFound, MDatNotShuffled, NameAlreadyUsed, SubMdatMissesStrategy, SubmDatNotFound, TooManySubmDats,
)

class MatformerDataset(IterableDataset):
    DB_FILENAME = 'mdat.db'
    DEFAULT_DSMAP_BITS = 8

    # Directory structure:
    DATASETS_DIR = 'datasets'
    SHUFFLING_DIR = 'shuffling'
    VIEWS_DIR = 'views'
    PRETOK_DIR = 'pretok'
    FUNCTIONS_DIR = 'functions'

    def __init__(self) -> None:
        """
        Initialize an empty MDat.
        The MDat is the only proper entry point for submdats and pretok_strategies.
        Both are instances of classes, in particular loaded_submdats contains pointers
        to SubMDat objects and pretok_strategies to PretokStrategy objects.
        Mdat is a very flexible dataset manager that allows for a very efficient pretokenization of the raw
        data following a given pretokenization strategy. It is possible to maintain in the same mdat several
        pretokenization strategy.
        *If the pretokenization strategy is chunk-based (ex. a text pretokenized in chunks of 512 tokens), it is possible
        to set a chunk_multiplier to elevate the target sequence length required by the dataset.
        *It is possible to set up several different "views" of the dataset, for example to prepare a smaller set for
        ablation studies;
        *It is possible (WIP) to prepare a "finalized" lightning-fast version of the dataset where modifications are no longer
        possible but reading time is reduced to minimum (ideal for larger trainings)
        """
        self._cached_length=None
        self.manifest: Dict[str, Any] = {}
        self.loaded_submdats: Dict[str, Any] = {} # Dictionary that contains the pointers to the loaded submdats, indexed by names
        self.pretok_strategies: Dict[str, Any] = {}
        self.readonly: bool = False
        self.current_strategy = None
        self.total_documents = 0
        self.total_tokens = None #This will be populated when a strategy is loaded
        self.total_chunks = None
        self.total_divided_chunks = None
        self.chunk_multiplier = 1 # Standard chunk multiplier: 1
        self.selection = None

    def _get_manifest_attr(self, key: str, default: Any = None) -> Any:
        """Get manifest attribute (possible to set a default)"""
        return self.manifest.get(key, default)

    def _set_manifest_attr(self, key: str, value: Any, save: bool = False) -> None:
        """Set manifest attribute, but don't update the file unless 'save' is True"""
        self.manifest[key] = value
        if save:
            self.db.update_manifest(data=self.manifest)

    @classmethod
    def load_dataset(cls, path: str, shuffle: bool = False,
                      ds_view: Optional[str] = None, distributed: bool = False,
                      readonly: bool = False, create_if_not_existent: bool = False, batch_size: Optional[int] = None,
                      prefetch_buffer=32768, *, strategy=None, max_seq_len=None,
                      selection=None) -> 'MatformerDataset':
        """
        Load an existing Mdat from a path. The loaded Mdat will have all the submdats, views and pretokenization strategies
        populated and ready to be used.
        Here, it can be also set the readonly flag and the presence of distributed training.

        Args:
            path: Dataset root directory path
            shuffle: Enable dataset shuffling
            ds_view: Dataset view to load
            distributed: Enable distributed training mode
            readonly: Load in read-only mode
            create_if_not_existent: Create dataset if missing
            strategy: Optional pretokenization strategy to activate
            max_seq_len: Desired sequence length for the strategy
            selection: DatasetSelection or dict with domains, max_tokens,
                domain_tokens, and seed. Applied within ds_view, without writing
                to the dataset. Token budgets preserve whole documents.

        ``shuffle`` is retained for compatibility. Existing pointer files define
        iteration order; loading never rewrites them. Use selection.seed to
        reproducibly change the order of a temporary selection.
        """

        instance = cls()
        instance._set_paths(path)
        instance.readonly = readonly
        instance._cached_length=None
        instance.dist=None
        instance.batch_size=batch_size
        instance.prefetch_buffer=prefetch_buffer
        if instance._mdat_exists(instance.db_path):
            instance.db = DatabaseManager(instance.db_path, readonly=readonly)
            try:
                instance.db.connect()
                instance.manifest = instance.db.get_manifest()
                instance.pretok_strategies = {}
                if distributed:
                    instance._set_distributed_training()
                else:
                    instance.dist=None
                if ds_view is not None:
                    instance.set_view(ds_view)
                else:
                    instance.set_view('default')
                #instance._populate_submdat() moved to view
                #instance._populate_strategies() changed to on-demand
                instance.set_iteration_modality('document', with_meta=True) #Default: get a non-tokenized document with metadata
                if selection is not None:
                    instance.select(selection, strategy=strategy, max_seq_len=max_seq_len)
                elif strategy is not None:
                    instance.set_strategy(strategy, max_seq_len=max_seq_len)
            except BaseException:
                instance.close()
                raise
        else:
            if create_if_not_existent:
                # Create and then load the dataset
                created = cls.create_new(path=path, overwrite=False)
                created.close()
                return cls.load_dataset(path, shuffle=shuffle, ds_view=ds_view,
                                        distributed=distributed, readonly=readonly,
                                        batch_size=batch_size, prefetch_buffer=prefetch_buffer,
                                        strategy=strategy, max_seq_len=max_seq_len, selection=selection)
            else:
                raise MDatNotFound(f"MDat not found at {path}")

        return instance

    def select(self, selection, *, strategy=None, max_seq_len=None):
        """Replace the temporary selection, always starting from current_view.

        Domains are subdataset names. Token counts use the selected strategy's
        stored tokens, before special tokens and padding are added. Inspect
        ``selection_summary`` for the achieved budget and per-domain totals.
        """
        config = selection if isinstance(selection, DatasetSelection) else DatasetSelection(**selection)
        if self.current_iteration_modality == 'chunked_for_recurrence':
            raise ValueError('Selections currently support document, tokens, chunks, and chunked_tokens iteration; '
                             'the recurrent loader cannot guarantee complete document consumption')
        active_ids = (self.db.ask_view(self.current_view, 'preserved_submdats') +
                      self.db.ask_view(self.current_view, 'partial_submdats'))
        names = self.db.get_dsmap('id')
        domains = config.resolve_domains([names[index] for index in active_ids])
        strategy_name = strategy or (self.current_strategy.strategy_name if self.current_strategy else None)
        if strategy_name is None and (config.max_tokens is not None or config.domain_tokens is not None):
            raise ValueError('Token budgets require a pretokenization strategy')
        self.stop_prefetch()
        if strategy_name is not None:
            if max_seq_len is None and self.current_strategy is not None:
                max_seq_len = self.max_seq_len
            active_domains = [name for name in domains if (config.domain_tokens or {}).get(name) != 0]
            self.set_strategy(strategy_name, max_seq_len=max_seq_len, domains=active_domains)
        selected = SelectedView(self, config, domains)
        self._clear_selection()
        self.selection = selected
        self.total_documents = selected.documents
        self.total_tokens = selected.tokens
        self.total_chunks = selected.chunks
        self.total_divided_chunks = sum(selected.worker_lengths(self.chunk_multiplier, chunked=True))
        self._active_view_shuffled = True
        self._cached_length = None
        self._worker_length_cache = {}
        self._reset_iteration_state()
        self.set_iteration_modality(self.current_iteration_modality, self.wants_metadata,
                                    self.return_raw, self.add_special_tokens)
        return self

    @property
    def selection_summary(self):
        return self.selection.summary() if self.selection is not None else None

    def _clear_selection(self):
        if getattr(self, '_shuffle_file', None) is not None:
            self._shuffle_file.close()
            self._shuffle_file = None
        if self.selection is not None:
            self.selection.close()
            self.selection = None

    def _reset_iteration_state(self):
        self.current_document = None
        self.document_index = 0
        self.current_chunk_step = 0
        self._iteration_count = 0
        for name in ('_skip_reinit', '_max_iterations', '_recurrent_documents',
                     '_recurrent_steps', '_recurrent_has_more'):
            self.__dict__.pop(name, None)

    def close(self):
        """Release temporary indexes, prefetch threads, LMDBs, and SQLite."""
        self.stop_prefetch()
        self._clear_selection()
        for subdataset in self.loaded_submdats.values():
            if subdataset is not None:
                subdataset.close()
        self.loaded_submdats.clear()
        if getattr(self, 'db', None) is not None:
            self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @classmethod
    def create_new(cls, path: str, overwrite: bool = False,
                   dsmap_bits: int = None, modalities: list = ['text']) -> 'MatformerDataset':
        """
        Create a new empty Mdat and load it.

        Args:
            path: Directory path for new dataset
            overwrite: Overwrite existing dataset if present
            dsmap_bits: Dataset mapping bits (defaults to 8 bit, meaning maximum 255 sub-datasets)
        """
        if dsmap_bits is None:
            dsmap_bits = cls.DEFAULT_DSMAP_BITS

        instance = cls()
        instance._set_paths(path)
        instance.readonly = False

        if instance.dataset_exists():
            if not overwrite:
                raise MDatAlreadyPresent(f"Dataset already exists at {path}")
            else:
                shutil.rmtree(instance.mdat_path)

        # Create directory structure
        instance._create_directory_structure()

        # Initialize database
        instance.db = DatabaseManager(instance.db_path)
        instance.db.connect()
        instance.db._initialize_db()

        # Create the default manifest
        instance.db.create_manifest(dsmap_bits,modalities)


        # Create default view
        instance.db.add_view('default')
        instance.db.close()

        # Now load the created dataset
        return cls.load_dataset(path, readonly=False)

    def _set_paths(self, path: str) -> None:
        """
        Private function: set the correct paths
        Args:
            path: Root dataset directory path
        """
        self.mdat_path = path
        self.db_path = os.path.join(self.mdat_path, self.DB_FILENAME)
        self.pretok_path = os.path.join(path, self.PRETOK_DIR)
        self.dataset_path = os.path.join(path, self.DATASETS_DIR)
        self.views_path = os.path.join(path, self.VIEWS_DIR)
        self.shuffling_path = os.path.join(path, self.SHUFFLING_DIR)
        self.functions_path = os.path.join(path, self.FUNCTIONS_DIR)

    def _mdat_exists(self, db_path: str) -> bool:
        """Check if manifest exists."""
        return os.path.exists(db_path)

    def _create_directory_structure(self) -> None:
        """Private function: create the mdat's directory structure"""
        os.makedirs(self.mdat_path, exist_ok=True)
        os.makedirs(self.dataset_path, exist_ok=True)
        os.makedirs(self.shuffling_path, exist_ok=True)
        os.makedirs(self.views_path, exist_ok=True)
        os.makedirs(self.pretok_path, exist_ok=True)
        os.makedirs(self.functions_path, exist_ok=True)

    def __getitem__(self, submdat_name: str) -> Any:
        """
        Direct access to submdat.
        Args:
            submdat_name: Name of the sub-dataset to retrieve
        """
        if hasattr(self, 'loaded_submdats') and submdat_name in self.loaded_submdats:
            return self.loaded_submdats[submdat_name]
        raise SubmDatNotFound(f"Sub-dataset '{submdat_name}' not found or not loaded in the current view.")

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
        if torch_dist_available and dist.is_available() and dist.is_initialized(): # Dirty multi-gpu stuff, temporanea
            self.dist = True
            self.world_size = dist.get_world_size()
            self.rank_size = dist.get_rank()
        else:
            self.dist = False
            self.world_size = 1
            self.rank_size = 0

    def _load_submdat(self, sbm_name):
        """Helper function"""
        self.loaded_submdats[sbm_name] = SubMdat.load_submdat(self, sbm_name) # SubMdats are initialized with parent reference so they can inherit things such as pretok strategies,readonly...

    def _populate_submdat(self) -> None:
        """Populate loaded submdats."""
        # Interrogate the current view to get datasets' number of documents according to the view
        #self.total_documents = self.db.get_manifest(view=self.current_view)['total_documents']
        self.total_documents=self.db.get_view_totals(view_name=self.current_view)['document_number']
        view_data = self.db.ask_view(view=self.current_view, keys=['preserved_submdats', 'partial_submdats'])
        for sbm_id in view_data['preserved_submdats'] + view_data['partial_submdats']: #Not loading the skipped submdats
            sbm_name = self.db.get_dsmap(key='id', value=sbm_id)
            self._load_submdat(sbm_name)

    def add_submdat(self, submdat_name: str, compression_levels: Dict = {}, map_sizes: Dict = {},
            db_types: List[str] = ['meta', 'data'], data_type: str = 'text', data_key: str = 'text', modality: str = 'text', hybrid_data: List = [],
            reshuffle: bool = False, round_robin_insertion: bool = False, reshuffle_at_the_end: bool = False) -> Any:
        """
        Add new submdat.
        Args:
            submdat_name: Name of submdat
            compression_levels: Compression config (A dict: {"db_name":compression:_level})
            map_sizes: Map size config (A dict: {"db_name":map_size})
            db_types: Database types
            data_type: Data type
            data_key: Data key
            hybrid_data: Hybrid data config
            reshuffle: Reshuffle after add
            round_robin_insertion: Use round robin
            reshuffle_at_the_end: Reshuffle at end
        """
        if self.readonly:
            raise MDatIsReadOnly
        # 0. Check if it fits in the mdat
        max_submdat = 255 if self._get_manifest_attr('dsmap_bits', 8) == 8 else 65535
        if self.db.get_dsmap('max_id') >= max_submdat:
            raise TooManySubmDats
        # 1. Check if the name is already used
        if submdat_name in self.list_submdat():
            raise NameAlreadyUsed
        self.loaded_submdats[submdat_name] = None # A placeholder to allow creation of submdat
        # 2. Create the submdat
        new = SubMdat.create_submdat(self, submdat_name=submdat_name, compression_levels=compression_levels, map_sizes=map_sizes, db_types=db_types, data_type=data_type, modality=modality, hybrid_data=hybrid_data)
        # 3. Default view must be updated
        #self.mdat._update_default_view(self.submdat_id)
        # End: load the submdat and returns the newly created SubMdat
        self.loaded_submdats[submdat_name] = new
        return new

    def list_submdat(self) -> List[str]:
        """List submdat names."""
        #return list(self.loaded_submdats.keys())
        return list(self.db.get_dsmap(key='name').keys())

    def get_submdat(self, submdat_name: str) -> Any:
        """
        Get submdat by name.
        Args:
            submdat_name: Name of submdat
        """
        if submdat_name in self.loaded_submdats.keys():
            return self.loaded_submdats[submdat_name]
        else:
            raise SubmDatNotFound

    def dataset_exists(self) -> bool:
        """Check if dataset exists."""
        return os.path.exists(os.path.join(self.mdat_path, self.DB_FILENAME))

    # Methods for the views and shuffling logic
    def shuffle(self, view_name=None):
        """
        This function will create a shuffled version of the dataset.
        Instead of actually shuffling the data, this function will create a file with pointers
        in the shuffle folder. For now, it supports only a shuffle of the whole dataset (shuffle/default.mdat)

        The pointer file is a file with structs:
        submdat_id,doc_id
        The datatype of submdat_id can either be 8 bit or 16 bit uint depending on the number of submdat (dsmap_bits)
        The datatype of doc_id depends from the largest number of documents in the submdats

        The shuffle file's struct format is returned to the caller
        """
        from bisect import bisect_right
        view_name = view_name or self.current_view
        if self.readonly:
            raise MDatIsReadOnly

        # 1. Interrogate the view to determine the preserved/selected submdats
        partial_ids = self.db.ask_view(view=view_name, keys='partial_submdats')
        preserved_ids = self.db.ask_view(view=view_name, keys='preserved_submdats')
        assert len(partial_ids)+len(preserved_ids)>0,"You are trying to shuffle an empty view."
        terminate_early=len(partial_ids)>0
        all_selected_ids=(partial_ids+preserved_ids)

        # 2. Create a mapping between "local_ids", used for the random permutation operation, and the actual id of the submdat
        all_selected_ids.sort()
        mapping={k:v for (k,v) in enumerate(all_selected_ids)}

        ds_map=self.db.get_dsmap(key='id')
        # 3. Calculate datasets length and cumulative length for bisect_right
        totlen=0
        cu_dslens=[0]
        selected_lengths=[]
        for i in range(0,len(all_selected_ids)):
            global_id=mapping[i] # The submdat_id
            L=len(self[ds_map[mapping[i]]])
            totlen+=L
            selected_lengths.append(L)
            cu_dslens.append(totlen)
        maxlen = max(selected_lengths) if selected_lengths else 0

        # 4. Defining the pointers struct size
        doc_id_type = 'B' if maxlen <= 255 else 'H' if maxlen <= 65535 else 'I' if maxlen <= 4294967295 else 'Q'
        submdat_id_type = 'B' if self._get_manifest_attr('dsmap_bits', 8) == 8 else 'H'
        struct_format = submdat_id_type + doc_id_type

        # 5. Output path
        os.makedirs(self.shuffling_path, exist_ok=True)
        shuffle_path = os.path.join(self.shuffling_path, f'{view_name}.mdat')

        # 6. Initializing the random permutator
        permutator = RandomPermutator(totlen)
        if terminate_early:
            # 7. Create the "terminators" => counters that terminates the selection earlier
            terminators={}
            # Terminator:
            # "ID": (accumulator_L,accumulator_B,terminator,criteria) #criteria L for lengths B for bytes
            for sbm_id in all_selected_ids:
                # sel_info is a dict containing at least keys 'bytes_criteria' and 'document_number' (None when not set)
                sel_info = self.db.get_view_submdat(view_name=view_name, submdat_id=sbm_id)
                if sbm_id in preserved_ids:
                    criteria='L'
                    terminator=len(self[ds_map[sbm_id]])
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
                        print("terminator is set to 0. It may be a bug")
                        continue
                terminators[sbm_id]=(0,0,terminator,criteria)


            def accumulator_equal_terminator(_id):
                v=terminators[_id]
                if v[3]=='L':
                    if v[0]>=v[2]:
                        return 1
                elif v[3] =='B':
                    if v[1]>=v[2]:
                        return 1
                else:
                    raise Exception
                return 0

            def accumulators_equal_terminators():
                terminated=0
                target=0
                for k in terminators.keys():
                    target+=1
                    if accumulator_equal_terminator(k):
                        terminated+=1
                return target==terminated

        # 8. Shuffling datasets' pointers
        with open(shuffle_path, 'wb') as f:
            pbar = tqdm(range(totlen))
            for idx in pbar:
                permuted_idx = permutator(idx)
                loc_id = bisect_right(cu_dslens, permuted_idx) - 1
                global_id = mapping[loc_id]
                doc_id = permuted_idx - cu_dslens[loc_id]
                # Early termination in case of view creation of partially chosen submdat
                if terminate_early:
                    if accumulators_equal_terminators():
                        return struct_format
                    if accumulator_equal_terminator(global_id):
                        continue
                    if terminators[global_id][3]=='L':
                        # update length accumulator (document count) for this submdat
                        accL, accB, term, crit = terminators[global_id]
                        accL += 1
                        terminators[global_id] = (accL, accB, term, crit)
                    else:
                        # update bytes accumulator for this submdat
                        accL, accB, term, crit = terminators[global_id]
                        accL+=1
                        accB += len(self[ds_map[global_id]][doc_id]['data'])
                        terminators[global_id] = (accL, accB, term, crit)
                    # Update progress bar with partial selection info
                    has_byte_criteria = any(t[3]=='B' for t in terminators.values())
                    has_doc_criteria = any(t[3]=='L' for t in terminators.values())
                    postfix = {}
                    if has_doc_criteria:
                        total_docs = sum(t[0] for t in terminators.values() if t[3]=='L')
                        target_docs = sum(t[2] for t in terminators.values() if t[3]=='L')
                        postfix['docs'] = f'{total_docs}/{target_docs}'
                    if has_byte_criteria:
                        total_bytes = sum(t[1] for t in terminators.values() if t[3]=='B')
                        target_bytes = sum(t[2] for t in terminators.values() if t[3]=='B')
                        postfix['MB'] = f'{total_bytes/(1024**2):.1f}/{target_bytes/(1024**2):.1f}'
                    pbar.set_postfix(postfix)
                    f.write(struct.pack(struct_format, global_id, doc_id))
                else:
                    # Write everything
                    f.write(struct.pack(struct_format, global_id, doc_id))

        if terminate_early:
            # Update the view with the computed number of selected documents and bytes length
            for sbm_id, (accL, accB, term, crit) in terminators.items():
                update_dict = {}
                if sbm_id in preserved_ids:
                    # Retrieve submdat metadata for preserved datasets
                    submdat_info = self.db.get_submdat_info(submdat_id=sbm_id)
                    update_dict['document_number'] = submdat_info.get('doc_count')
                    update_dict['bytes_criteria'] = submdat_info.get('raw_data_bytes')
                else:
                        update_dict['document_number'] = accL
                        update_dict['bytes_criteria'] = accB

                self.db.update_view_submdat(view_name=view_name, submdat_id=sbm_id, data=update_dict)
        self.db.update_manifest(data={"shuffle_struct_format": struct_format}, view=view_name)
        return struct_format


    def _update_default_view(self, submdat_id):
        submdat_info = self.db.get_submdat_info(submdat_id=submdat_id)
        self.db.link_submdat_view(submdat_id=submdat_id,view_name='default',is_skipped=0,is_partial=0,is_preserved=1,document_number=submdat_info.get('document_number'),bytes_criteria=submdat_info.get('raw_data_bytes'))

        #self.shuffle('default', is_shuffled=True)

    def register_new_view(self, view_dict):
        """
        How views work in Mdat.

        Each view must be initialized from a view descriptor, a Python dictionary with the following fields:
        {
            "view_name": "example",
            "selection": {
                "dataset_1": "2G",       # 2 Gigabytes
                "dataset_2": 800,        # 800 Documents
                "dataset_3": "preserved",# All documents preserved
                "dataset_4": "80M",      # 80 Megabytes
                "dataset_5": 0.2,        # 20% of documents (proportion-based)
                "dataset_6": "skipped"   # Skipped submdat
            },
            "shuffle": true/false        # Decide if to directly apply shuffling after view creation.
        }

        - The 'selection' field determines how submdats are handled.
          Each submdat key maps to a selection descriptor, which can be:
            * an integer => number of documents to select;
            * a float in [0,1] => proportion of documents to select;
            * a string ending in K/M/G/T => byte-based selection (byte computation is carried on only on the 'data' field) (e.g. '500M');
            * the string 'preserved' => all documents included;
            * the string 'skipped' => submdat excluded entirely.
        - If any submdat has a partial selection (integer, float, or byte-based), shuffling is always enforced,
          regardless of the explicit 'shuffle' parameter.
        """
        view_name = view_dict.get('view_name')
        selection = view_dict.get('selection', {})
        shuffle = bool(view_dict.get('shuffle', False))

        all_submdats = list(self.list_submdat())
        name_to_id = {name: self.db.get_dsmap(key='name', value=name) for name in all_submdats}

        if set(all_submdats) != set(selection.keys()):
            raise ValueError("Selection must include all available submdats")

        targets = {}
        partial_submdats = []
        skipped_submdats = []
        preserved_submdats = []
        submdat_bytes_decided=[]
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
                bytes_criteria=0
            elif sbm in preserved_submdats:
                doc_number = len(self[sbm])
                bytes_criteria = self.db.get_submdat_info(ds_id,)['raw_data_bytes']
            elif ds_id in targets:
                val = targets[ds_id]
                if ds_id in submdat_bytes_decided:
                    partial_doc_number=0
                    bytes_criteria = int(val)
                elif isinstance(val, int):
                    partial_doc_number = val
                elif isinstance(val, float):
                    partial_doc_number = int(len(self[sbm]) * val)
                else:
                    raise Exception

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

        # Recompute chunks for pretokenized submdats if required
        # self._recompute_view_chunks(view_name)

        if partial_submdats:
            if not shuffle:
                print("Partial selections detected; shuffling enforced for random document selection.")
            shuffle = True

        shuffle_struct_format = self.shuffle(view_name) if shuffle else None
        self.db.update_manifest(data={"shuffle_struct_format": shuffle_struct_format}, view=view_name)
        return self.db.get_manifest(view=view_name)


    def _recompute_view_chunks(self, view_name):
        """
        Recompute chunk counts for pretokenized submdats in a view
        """
        from tqdm import tqdm

        strategies = self.db.connect().cursor().execute("SELECT strategy_name FROM pretok_strategy").fetchall()

        for (strategy_name,) in strategies:
            strategy_info = self.db.get_manifest(strategy=strategy_name)
            if not strategy_info or not strategy_info.get('chunk_size'):
                continue

            partial_ids = self.db.ask_view(view=view_name, keys='partial_submdats')

            for submdat_id in tqdm(partial_ids, desc=f"Recomputing chunks for {strategy_name}"):
                strategy_submdat_info = self.db.get_strategy_submdats(strategy_name, submdat_id)
                if not strategy_submdat_info:
                    continue

                view_submdat = self.db.connect().cursor().execute(
                    "SELECT document_number FROM view_submdat WHERE view_name = ? AND submdat_id = ?",
                    (view_name, submdat_id)
                ).fetchone()

                if not view_submdat:
                    continue

                doc_count = view_submdat[0]
                total_chunks_in_submdat = strategy_submdat_info.get('total_chunks', 0)
                total_docs_in_submdat = self.db.get_manifest(submdat=submdat_id).get('document_number', 1)

                # Estimate chunks proportionally
                estimated_chunks = int(total_chunks_in_submdat * doc_count / max(total_docs_in_submdat, 1))

                # Precompute for different multipliers
                for multiplier in range(1, 101):
                    precomp = math.ceil(estimated_chunks / multiplier)
                    self.db.set_strategy_view_precomp(strategy_name, view_name, submdat_id, multiplier, precomp)

    def _init_shuffle_file(self):
        """Initialize shuffle file for the active view."""
        if not hasattr(self, '_shuffle_file') or self._shuffle_file is None:
            if self.selection is not None:
                self._shuffle_struct_format = self.selection.pointer_format
                self._shuffle_struct_size = struct.calcsize(self._shuffle_struct_format)
                self._shuffle_file = open(self.selection.pointer_path, 'rb')
                return
            active = getattr(self, '_active_view', None)
            if not active:
                raise MDatNotShuffled("No active view set. Call set_view().")

            view_meta = self.db.get_manifest(view=active)
            if view_meta is None:
                raise ValueError(f"View '{active}' not registered")

            struct_format = view_meta.get('shuffle_struct_format')
            if not struct_format:
                raise MDatNotShuffled(f"View '{active}' is not shuffled")

            shuffle_file_path = os.path.join(self.shuffling_path, f'{active}.mdat')

            self._shuffle_struct_format = struct_format
            self._shuffle_struct_size = struct.calcsize(struct_format)
            self._shuffle_file = open(shuffle_file_path, 'rb')

    def _reset_document_pointer(self):
        """Reset document index and file pointer."""
        self._seek_document_pointer(0)

    def _seek_document_pointer(self, index):
        """Go to document index."""
        self.document_index = index
        if hasattr(self, '_shuffle_file') and self._shuffle_file:
            print("Seeking into shuffle file...")
            self._shuffle_file.seek(index * self._shuffle_struct_size)

    def set_view(self, view_name):
        """
        Set the active view.
        """
        if view_name is None:
            raise ValueError("A valid view name must be provided")

        view_meta = self.db.get_manifest(view=view_name)
        if not view_meta:
            raise ValueError(f"View '{view_name}' not found in manifest")

        self.stop_prefetch()
        self._clear_selection()
        for subdataset in self.loaded_submdats.values():
            if subdataset is not None:
                subdataset.close()
        self.loaded_submdats.clear()
        self._cached_length = None
        self._worker_length_cache = {}
        self._reset_iteration_state()
        self._active_view = view_name
        self.current_view = view_name
        self.document_index = 0
        # Set the MDAT length to the number of documents in the view
        self.len = self.db.get_view_totals(view_name=view_name)['document_number']
        self._active_view_shuffled = view_meta.get('shuffle_struct_format') is not None
        # Load the actual submdats
        self._populate_submdat()
        if self.current_strategy is not None:
            self.set_strategy(self.current_strategy.strategy_name, getattr(self, 'max_seq_len', None))
        if hasattr(self, 'current_iteration_modality'):
            self.set_iteration_modality(self.current_iteration_modality, self.wants_metadata,
                                        self.return_raw, self.add_special_tokens)
    def precompute_chunks_distributed(self, num_workers=840):
        """
        Precompute chunk counts for distributed training.

        Args:
            num_workers: Number of workers to precompute for (should be highly composite)
        """
        #if 'chunked_tokens' not in self.current_strategy.returns:
        #    raise ValueError(f"Current strategy '{self.current_strategy.strategy_name}' does not return chunked_tokens")

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
    def start_prefetch(self, max_prefetch=32768, shuffled=None):
        """Prefetch documents, preserving failures instead of silently dropping data."""
        if getattr(self, '_prefetch_thread', None) is not None:
            return
        shuffled = self._active_view_shuffled if shuffled is None else shuffled
        if shuffled:
            self._init_shuffle_file()
        self.prefetch_index = self.document_index
        ds_map = self.db.get_dsmap('id')
        self._prefetch_queue = queue.Queue(max_prefetch)
        self._prefetch_stop = threading.Event()

        def enqueue(item):
            while not self._prefetch_stop.is_set():
                try:
                    self._prefetch_queue.put(item, timeout=0.1)
                    return
                except queue.Full:
                    continue

        def prefetch_loop():
            while not self._prefetch_stop.is_set():
                try:
                    document, self.prefetch_index = self._load_next_document_inner(
                        ds_map, self.prefetch_index, shuffled)
                    enqueue((self.prefetch_index, document))
                except StopIteration:
                    enqueue(None)
                    return
                except Exception as error:
                    enqueue(error)
                    return

        self._prefetch_thread = threading.Thread(target=prefetch_loop, daemon=True)
        self._prefetch_thread.start()

    def stop_prefetch(self):
            if not hasattr(self, "_prefetch_thread") or self._prefetch_thread is None:
                return

            self._prefetch_stop.set()

            if hasattr(self, "_prefetch_queue") and self._prefetch_queue is not None:
                while not self._prefetch_queue.empty(): #Draining the queue
                    try:
                        self._prefetch_queue.get_nowait()
                    except queue.Empty:
                        break

            # Do not replace the queue/event or close LMDB while an old reader
            # can still access them (slow shared-filesystem reads may exceed 2s).
            self._prefetch_thread.join()
            self._prefetch_thread = None
            self._prefetch_queue = None

    def load_next_document(self, shuffled=None):
        shuffled = self._active_view_shuffled if shuffled is None else shuffled
        if self.selection is not None:
            shuffled = True  # The temporary index defines selected membership.
        if getattr(self, '_prefetch_queue', None) is not None:
            while True:
                try:
                    result = self._prefetch_queue.get(timeout=1)
                except queue.Empty:
                    if self._prefetch_thread is None or not self._prefetch_thread.is_alive():
                        self.stop_prefetch()
                        raise RuntimeError('Prefetch thread exited without an end marker')
                    continue
                if result is None:
                    self.stop_prefetch()
                    raise StopIteration
                if isinstance(result, Exception):
                    self.stop_prefetch()
                    raise result
                self.document_index, document = result
                return document
        if shuffled:
            self._init_shuffle_file()
        document, self.document_index = self._load_next_document_inner(
            self.db.get_dsmap('id'), self.document_index, shuffled)
        return document

    def _load_next_document_inner(self, ds_map, local_document_index, shuffled=True, ) -> None:
        """Load next document."""
        if shuffled:
            if not hasattr(self, '_shuffle_file') or self._shuffle_file is None:
                raise Exception
            # skip until we find a document good for this worker
            while True:
                data = self._shuffle_file.read(self._shuffle_struct_size)
                if not data:
                    raise StopIteration

                if self.dist:
                    if local_document_index % self.world_size != self.rank_size:
                        local_document_index += 1
                        continue  # Skip this shuffle entry

                submdat_id, doc_id = struct.unpack(self._shuffle_struct_format, data)
                current_document = self.loaded_submdats[ds_map[submdat_id]][doc_id]
                local_document_index += 1
                break
        else:
            if self.dist:
                local_document_index += (self.rank_size - local_document_index) % self.world_size
            current_pos = 0
            for submdat_id in sorted(ds_map.keys()):
                submdat_name = ds_map[submdat_id]
                if submdat_name in self.loaded_submdats:
                    submdat_len = len(self.loaded_submdats[submdat_name])
                    if local_document_index < current_pos + submdat_len:
                        doc_id = local_document_index - current_pos
                        current_document = self.loaded_submdats[submdat_name][doc_id]
                        local_document_index += 1
                        return current_document, local_document_index
                    current_pos += submdat_len
            raise StopIteration
        return current_document, local_document_index
    # Methods for the pretokenization-registry:
    # In the manifest, names of pretokenizations strategies should be saved
    # Each strategy will live in ds_root/pretok/strategy_name
    # Each strategy will have its ds_root/pretok/strategy_name/strategy.json
    # Each pretokenization will live in ds_root/pretok/strategy_name/submdat_name/
    # Each pretokenization will have important stats in ds_root/pretok/strategy_name/submdat_name/
    def deregister_strategy(self, strategy_name: str) -> None:
        """
        Deregister strategy.
        Args:
            strategy_name: Strategy name to remove
        """
        if self.readonly:
            raise MDatIsReadOnly
        c = self.db.connect().cursor()
        c.execute("DELETE FROM pretok_strategy WHERE strategy_name = ?", (strategy_name,))

    def register_strategy(self, strategy_name: str, strategy_dict: Dict[str, Any]) -> Optional[Any]:
        """
        Register new strategy.
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
        if strategy:
            strategy.save()

    def strategies(self):
        return self.list_strategies() #a shortcut
    def list_views(self):
        return self.db.list_views()
    def get_state_dict(self):
        """
        Return a dictionary with all the required pointers in order to restart training

        """
        state_dict={
            "iteration_modality":self.current_iteration_modality,
            "document_index":getattr(self,"document_index",0),
            "current_chunk_step":getattr(self,"current_chunk_step",0),
            "_iteration_count":getattr(self,"_iteration_count",0),
            "_recurrent_documents":getattr(self,"_recurrent_documents",[]),
            "_recurrent_steps":getattr(self,"_recurrent_steps",[]),
            "_max_iterations":getattr(self,"_max_iterations",None),
            "_cached_length":getattr(self,"_cached_length",None),
            "saved_world_size":1 if not self.dist else self.world_size
        }
        state_dict['selection_fingerprint'] = self.selection.fingerprint if self.selection else None
        state_dict['current_document'] = getattr(self, 'current_document', None)
        state_dict['dataset_config'] = self._checkpoint_config()
        return copy.deepcopy(state_dict)

    def _checkpoint_config(self):
        return {
            'view': self.current_view,
            'strategy': self.current_strategy.strategy_name if self.current_strategy else None,
            'max_seq_len': getattr(self, 'max_seq_len', None),
            'batch_size': self.batch_size,
            'with_meta': self.wants_metadata,
            'return_raw': self.return_raw,
            'add_special_tokens': self.add_special_tokens,
        }
    def restore_state_dict(self,state_dict):
        expected = self.selection.fingerprint if self.selection else None
        if state_dict.get('selection_fingerprint') != expected:
            raise ValueError('Checkpoint dataset selection differs from the loaded selection')
        if state_dict.get('dataset_config', self._checkpoint_config()) != self._checkpoint_config():
            raise ValueError('Checkpoint view, strategy, sequence length, or loading options differ')
        if state_dict['iteration_modality'] != self.current_iteration_modality:
            raise ValueError('Checkpoint iteration modality differs from the loaded dataset')
        self.stop_prefetch()
        self.document_index = int(state_dict["document_index"])
        self.current_chunk_step = int(state_dict["current_chunk_step"])
        self._iteration_count = int(state_dict["_iteration_count"])
        self._recurrent_documents=copy.deepcopy(state_dict["_recurrent_documents"])
        self._recurrent_steps=list(state_dict["_recurrent_steps"])
        self._max_iterations=None if state_dict["_max_iterations"] is None else int(state_dict['_max_iterations'])
        self._cached_length=None if state_dict["_cached_length"] is None else int(state_dict['_cached_length'])
        self.current_document = copy.deepcopy(state_dict.get('current_document'))

        self._skip_reinit=True
        #self._seek_document_pointer(self.document_index)


    def __iter__(self,with_prefetch=True):
            if get_worker_info() is not None:
                raise RuntimeError('MDAT manages its own prefetching; use DataLoader(num_workers=0)')
            # Reset the iteration (ex. for a new epoch)
            self.stop_prefetch()

            # 2. Reset State
            if not hasattr(self,'_skip_reinit'):
                self._reset_iteration_state()

                if hasattr(self, '_max_iterations'):
                    del self._max_iterations

                # Reset file pointer for the new thread to use
                if hasattr(self, '_shuffle_file') and self._shuffle_file is not None:
                    self._shuffle_file.seek(0)
                if with_prefetch:
                    if self.prefetch_buffer is not None and self.prefetch_buffer>0:
                        self.start_prefetch(max_prefetch=self.prefetch_buffer,
                                            shuffled=self._active_view_shuffled)
            else:
                print("MDAT is resuming training from external logic.")
                print("Document index: ",self.document_index)
                print("Chunk step: ", self.current_chunk_step)
                print("Max iterations: ",self._max_iterations)
                print("Iteration count: ", self._iteration_count)
                shuffled=self._active_view_shuffled
                if shuffled:
                    if not hasattr(self, '_shuffle_file') or self._shuffle_file is None:
                        self._init_shuffle_file()
                self._seek_document_pointer(self.document_index)
                if with_prefetch:
                    if self.prefetch_buffer is not None and self.prefetch_buffer>0:
                        self.start_prefetch(max_prefetch=self.prefetch_buffer, shuffled=shuffled)
                del self._skip_reinit
            return self


    def _get_next_chunk_from_document(self, document, chunk_step):
            """
            Helper method to extract the next chunk from a given document and update the step.
            Returns: (chunk, next_chunk_step, has_more_chunks)
            """
            if document is None or 'chunked_tokens' not in document:
                return None, 0, False
            chunks_list = document['chunked_tokens']
            total_chunks = len(chunks_list)
            if chunk_step >= total_chunks:
                return None, 0, False # Document exhausted
            end_step = min(chunk_step + self.chunk_multiplier, total_chunks)
            selected_chunks = chunks_list[chunk_step:end_step]
            new_chunk_step = end_step
            if self.add_special_tokens and len(selected_chunks) > 1:
                # Cached chunks each have BOS/EOS. A longer packed context has
                # one pair, preserving every content token within this document.
                bos, eos = self.current_strategy.bos_token_id, self.current_strategy.eos_token_id
                if any(len(part) < 2 or part[0] != bos or part[-1] != eos for part in selected_chunks):
                    raise ValueError('Stored chunks do not have the declared BOS/EOS boundaries')
                chunk = [bos] + [int(token) for part in selected_chunks for token in part[1:-1]] + [eos]
            else:
                chunk = np.concatenate(selected_chunks).tolist()
            if getattr(self, 'max_seq_len', None) is not None:
                if len(chunk) > self.max_seq_len:
                    raise ValueError(f'Packed chunk has {len(chunk)} tokens, exceeds {self.max_seq_len}; '
                                     'reserve space for BOS/EOS when pretokenizing. No tokens were truncated.')
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
                if self.current_iteration_modality in ['document', 'tokens', 'chunks']: # Entire document
                    self.current_document=self.load_next_document()
                    result = self.current_document

                elif self.current_iteration_modality == 'chunked_tokens': # Sequential chunked
                    if not hasattr(self, 'current_chunk_step'):
                        self.current_chunk_step = 0
                        self.current_document = None

                    while True:
                        is_document_exhausted = (self.current_document is None or
                                                 'chunked_tokens' not in self.current_document or
                                                 self.current_chunk_step >= len(self.current_document['chunked_tokens']))

                        if is_document_exhausted:
                            self.current_document=self.load_next_document()
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
                if self.wants_metadata and self.current_iteration_modality in ['chunked_tokens','chunked_for_recurrence']:
                    #The dataloader wants metadata, those are appended in chunked_tokens modalities
                    return_dict['metadata']={x: self.current_document[x] for x in self.current_document if x not in 'chunked_tokens'}
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
        if self.selection is not None and modality == 'chunked_for_recurrence':
            raise ValueError('Selections currently support document, tokens, chunks, and chunked_tokens iteration; '
                             'the recurrent loader cannot guarantee complete document consumption')
        self.return_raw = return_raw
        self.add_special_tokens = add_special_tokens
        self._cached_length = None
        self._worker_length_cache = {}
        if modality in ('chunked_tokens', 'chunked_for_recurrence'):
            if self.current_strategy is None or 'chunks' not in self.current_strategy.returns:
                raise ValueError('Chunk iteration requires a strategy with stored chunks')
        supported_modalities = ['document', 'tokens', 'chunked_tokens', 'strategy_default','chunks','chunked_for_recurrence']
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
        elif modality in ['chunked_tokens','chunked_for_recurrence']:
            wanted_from_strategy = ['chunked_tokens']
            wanted_from_dbs = []
            self.len = self.total_divided_chunks
            if modality=='chunked_for_recurrence':
                assert self.batch_size is not None
        elif modality == 'strategy_default':
            wanted_from_strategy = ['strategy_default']
            wanted_from_dbs = []
            self.len = 0
        else:
            raise Exception
        if with_meta:
            self.wants_metadata=True
            wanted_from_dbs.append('meta')
        else:
            self.wants_metadata=False
        # Set the iteration modality in each submdat
        for sbm in self.loaded_submdats.values():
            sbm.set_default_wanted(from_dbs=wanted_from_dbs, from_strategy=wanted_from_strategy, raw=return_raw, add_special_tokens=add_special_tokens)

    def get_iteration_modality(self):
        return self.current_iteration_modality

    def set_strategy(self, strategy_name, max_seq_len=None, *, domains=None):
        """Activate cached tokenization, validating all selected domains.

        ``domains`` is used internally while constructing a selection. Normal
        callers can pass domain names through ``select`` or ``load_dataset``.
        Changing the strategy on a selection recomputes its token budget.
        """
        if self.selection is not None and domains is None:
            return self.select(self.selection.config, strategy=strategy_name, max_seq_len=max_seq_len)
        strategy = PretokenizationStrategy.from_dict(
            db=self.db, strategy_name=strategy_name,
            pretok_path=self.pretok_path, functions_path=self.functions_path)
        base_size = strategy.chunk_size
        target_size = max_seq_len if max_seq_len is not None else base_size
        if target_size is not None and (isinstance(target_size, bool) or not isinstance(target_size, int)):
            raise ValueError('max_seq_len must be an integer')
        if target_size is not None and base_size is not None:
            if target_size <= 0 or target_size % base_size:
                raise ValueError('max_seq_len must be a positive multiple of the strategy chunk size')
        names = list(domains) if domains is not None else list(self.loaded_submdats)
        for name in names:
            subdataset = self.loaded_submdats[name]
            if not self.db.get_strategy_submdats(strategy_name, subdataset.submdat_id):
                raise SubMdatMissesStrategy(f"Domain {name!r} has no pretokenized strategy {strategy_name!r}")
        self.current_strategy = strategy
        self.max_seq_len = target_size
        self.chunk_multiplier = target_size // base_size if target_size and base_size else 1
        self._cached_length = None
        self._worker_length_cache = {}
        self.total_tokens = self.total_chunks = self.total_divided_chunks = 0
        for name in names:
            self.loaded_submdats[name].set_strategy(strategy_name)
        # A selection computes exact counts while building its temporary index.
        if domains is not None:
            return
        recount = set()
        partial = set(self.db.ask_view(self.current_view, 'partial_submdats'))
        for name in names:
            subdataset = self.loaded_submdats[name]
            domain_id = subdataset.submdat_id
            stats = self.db.get_strategy_submdats(strategy_name, domain_id)
            precomputed = self.db.get_strategy_precomp(
                strategy_name, submdat_id=domain_id, multiplier=self.chunk_multiplier)
            if domain_id in partial or ('chunks' in strategy.returns and not precomputed):
                recount.add(domain_id)
                continue
            self.total_tokens += stats.get('total_tokens', 0) or 0
            self.total_chunks += stats.get('total_chunks', 0) or 0
            if precomputed:
                self.total_divided_chunks += precomputed[0]['precomputed_length']
        if recount:
            from .selection import source_references
            from .utils import get_datatype_for_numpy
            token_size = np.dtype(get_datatype_for_numpy(strategy.tokens_datatype)).itemsize
            chunk_size = np.dtype(get_datatype_for_numpy(strategy.chunks_datatype)).itemsize
            id_to_name = self.db.get_dsmap('id')
            for domain_id, document_id in source_references(self):
                if domain_id not in recount:
                    continue
                cache = self.loaded_submdats[id_to_name[domain_id]].pretok_db
                tokens = len(cache['tokens'][document_id]) // token_size
                chunks = len(cache['chunks'][document_id]) // chunk_size if 'chunks' in cache else 0
                self.total_tokens += tokens
                self.total_chunks += chunks
                self.total_divided_chunks += (chunks + self.chunk_multiplier - 1) // self.chunk_multiplier
        if getattr(self, 'current_iteration_modality', None) in ('chunked_tokens', 'chunked_for_recurrence'):
            self.len = self.total_divided_chunks

    def export_view(self,output_file,view='default',wanted='document',data_field_name='text',with_meta=True,limiter=None):
        """
        A function (WIP) to export all the documents from a view into an output file
        Currently supports only documents (with/without metadata) and JSONL output.

        """
        from tqdm import tqdm
        import orjson

        if wanted not in ('document', 'default'):
            raise ValueError('JSONL export supports document records only')
        if limiter is not None and (not isinstance(limiter, int) or limiter < 0):
            raise ValueError('limiter must be a non-negative integer or None')
        self.set_view(view)
        self._reset_document_pointer()
        self.set_iteration_modality('document', with_meta=with_meta)

        counter=0
        with open(output_file, "wb") as f:
            if limiter == 0:
                return 0
            for item in tqdm(self):
                item = item['object']
                data = {
                    "group": item['submdat_name'],
                    data_field_name: item['data']
                }

                if with_meta:
                    # Metadata is without "submdat_name" and "data"
                    meta = {k: v for k, v in item.items() if k not in ('submdat_name', 'data')}
                    data["meta"] = meta

                f.write(orjson.dumps(data))
                f.write(b"\n")
                counter += 1
                if limiter is not None and counter >= limiter:
                    break
        self.stop_prefetch()
        return counter



    def __str__(self):
        stringa = ""
        stringa += ("\n---- Matformer Dataset ----")
        for k in self.manifest.keys():
            stringa += (f"\n{k}: {self.manifest[k]}")
        stringa += (f"\n Current strategy: {self.current_strategy}")
        stringa += (f"\n Total documents: {self.total_documents}")
        stringa += (f"\n Total tokens: {self.total_tokens}")
        stringa += (f"\n Total chunks: {self.total_chunks}")
        stringa += (f"\n Total divided chunks: {self.total_divided_chunks}")
        stringa += (f"\n Current iteration modality: {self.current_iteration_modality}")
        stringa += ("\n Loaded sumbdats: ")
        return stringa

    def list_strategies(self):
        """
        List registered strategies
        """
        return list(self.db.list_strategies())

    def get_distributed_length_before_training(self, num_devices=1):
        """Return the longest rank's epoch length without writing metadata."""
        return max(self._worker_lengths(num_devices))

    def _worker_lengths(self, num_devices):
        if not isinstance(num_devices, int) or num_devices < 1:
            raise ValueError('num_devices must be a positive integer')
        chunked = self.current_iteration_modality in ('chunked_tokens', 'chunked_for_recurrence')
        if self.selection is not None:
            return self.selection.worker_lengths(self.chunk_multiplier, num_devices, chunked)
        if not chunked:
            quotient, remainder = divmod(self.total_documents, num_devices)
            return [quotient + int(rank < remainder) for rank in range(num_devices)]
        key = self.current_view, self.current_strategy.strategy_name, self.chunk_multiplier, num_devices
        if key in getattr(self, '_worker_length_cache', {}):
            return self._worker_length_cache[key]
        try:
            lengths = [self.db.get_worker_length_distributed(
                target_num_workers=num_devices, target_worker_id=rank,
                view_name=self.current_view, strategy_name=self.current_strategy.strategy_name,
                chunk_multiplier=self.chunk_multiplier) for rank in range(num_devices)]
        except ValueError:
            # Legacy datasets may lack lengths for this worker count. Compute
            # exact counts in memory so read-only loading remains read-only.
            from .selection import source_references
            from .utils import get_datatype_for_numpy
            size = np.dtype(get_datatype_for_numpy(self.current_strategy.chunks_datatype)).itemsize
            names = self.db.get_dsmap('id')
            lengths = [0] * num_devices
            for position, (domain_id, document_id) in enumerate(source_references(self)):
                cache = self.loaded_submdats[names[domain_id]].pretok_db['chunks']
                chunks = len(cache[document_id]) // size
                lengths[position % num_devices] += (chunks + self.chunk_multiplier - 1) // self.chunk_multiplier
        if not hasattr(self, '_worker_length_cache'):
            self._worker_length_cache = {}
        self._worker_length_cache[key] = lengths
        return lengths
    def __len__(self):
        """Count selected documents/chunks; distributed ranks share a max length."""
        if self.dist:
            return max(self._worker_lengths(self.world_size))
        return self.len
    def create_head_view(self, n: int, new_view_name: str, source_view: str = 'default') -> dict:
        """
        *** Added using an LLM, not yet tested but should work ***
        Create a new view containing the first n documents from source_view's
        shuffle file, preserving their original order. No re-shuffling.

        Args:
            n:               Number of documents to take from the head of source_view.
            new_view_name:   Name for the new view.
            source_view:     View to slice from (default: 'default').

        Returns:
            The new view's manifest dict.
        """
        # 1. Validate source view and its shuffle file
        source_meta = self.db.get_manifest(view=source_view)
        if not source_meta:
            raise ValueError(f"Source view '{source_view}' not found.")
        struct_format = source_meta.get('shuffle_struct_format')
        if not struct_format:
            raise MDatNotShuffled(f"Source view '{source_view}' has no shuffle file.")

        source_total = self.db.get_view_totals(view_name=source_view)['document_number']
        if n > source_total:
            raise ValueError(f"Requested {n} documents but source view only has {source_total}.")

        source_shuffle_path = os.path.join(self.shuffling_path, f'{source_view}.mdat')
        struct_size = struct.calcsize(struct_format)

        # 2. Create the new view entry in the DB (must exist before link_submdat_view calls)
        if new_view_name in self.db.list_views():
            raise NameAlreadyUsed(f"View '{new_view_name}' already exists.")
        self.db.add_view(new_view_name)

        # 3. Read the first n entries and accumulate per-submdat stats
        doc_counts   = {}   # submdat_id -> number of docs selected
        bytes_counts = {}   # submdat_id -> raw bytes of selected docs

        new_shuffle_path = os.path.join(self.shuffling_path, f'{new_view_name}.mdat')
        os.makedirs(self.shuffling_path, exist_ok=True)

        with open(source_shuffle_path, 'rb') as src, \
             open(new_shuffle_path, 'wb') as dst:

            for _ in tqdm(range(n), desc=f"Building head view '{new_view_name}'"):
                raw = src.read(struct_size)
                if not raw:
                    break  # source file shorter than expected; stop cleanly
                dst.write(raw)

                submdat_id, doc_id = struct.unpack(struct_format, raw)
                doc_counts[submdat_id]  = doc_counts.get(submdat_id, 0) + 1

                # Accumulate raw bytes from the actual data DB
                #submdat_name = ds_map[submdat_id]
                #raw_bytes = len(self.loaded_submdats[submdat_name].storage_db['data'][doc_id])
                #bytes_counts[submdat_id] = bytes_counts.get(submdat_id, 0) + raw_bytes

        # 4. Register every submdat against the new view
        #    Submdats that appear in the head are "partial"; others are "skipped".
        all_submdat_ids = set(self.db.get_dsmap(key='id').keys())   # all known IDs

        for sbm_id in all_submdat_ids:
            if sbm_id in doc_counts:
                self.db.link_submdat_view(
                    submdat_id      = sbm_id,
                    view_name       = new_view_name,
                    is_skipped      = 0,
                    is_partial      = 1,
                    is_preserved    = 0,
                    document_number = doc_counts[sbm_id],
                    bytes_criteria  = bytes_counts.get(sbm_id, 0),
                )
            else:
                self.db.link_submdat_view(
                    submdat_id      = sbm_id,
                    view_name       = new_view_name,
                    is_skipped      = 1,
                    is_partial      = 0,
                    is_preserved    = 0,
                    document_number = 0,
                    bytes_criteria  = 0,
                )

        # 5. Store the shuffle struct format so set_view / _init_shuffle_file work normally
        self.db.update_manifest(
            data={'shuffle_struct_format': struct_format},
            view=new_view_name,
        )

        return self.db.get_manifest(view=new_view_name)



# Resolve historical helper imports lazily, without circular CLI imports.
def __getattr__(name):
    import importlib
    for module_name in ("exceptions", "lmdb", "preprocessing", "submdat", "utils", "user_interface"):
        module = importlib.import_module(f"{__package__}.{module_name}")
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main(argv=None):
    from .user_interface import main as cli_main
    return cli_main(argv)


if __name__ == "__main__":
    main()
