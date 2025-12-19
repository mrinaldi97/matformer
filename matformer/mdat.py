#!/usr/bin/env python3
from __future__ import annotations
import math
import os
import json
import shutil
import struct
import traceback
import importlib
try:
    from tqdm import tqdm  
except:
    pass
import numpy as np
#from lmdb_dataset import LMDBDataset
try:
    import orjson
except:
    orjson=None
from typing import Optional, Dict, Any, Union, List, Set
import multiprocessing as mp
from functools import partial
from torch.utils.data import IterableDataset
from bisect import bisect_right
import queue
import threading
import time

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
                      readonly: bool = False, create_if_not_existent: bool = False, batch_size: Optional[int] = None,prefetch_buffer=32768 ) -> 'MatformerDataset':
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
        """
        
        instance = cls()
        instance._set_paths(path)
        instance.readonly = readonly
        instance._cached_length=None
        instance.dist=None
        instance.batch_size=batch_size
        instance.prefetch_buffer=prefetch_buffer
        if instance._mdat_exists(instance.db_path): 
            instance.db = DatabaseManager(instance.db_path) 
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
        else:
            if create_if_not_existent:
                # Create and then load the dataset
                return cls.create_new(path=path, overwrite=False)
            else:
                raise MDatNotFound(f"MDat not found at {path}")
        
        return instance
    
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
        manifest = { 
            "dsmap_bits": dsmap_bits,
            "modalities": json.dumps(modalities) 
        }    
        instance.db.create_manifest(dsmap_bits,modalities)  

        
        # Create default view
        instance.db.add_view('default')
        
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
            import torch
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
        
        strategies = self.db.connect(check_same_thread=False).cursor().execute("SELECT strategy_name FROM pretok_strategy").fetchall() 
        
        for (strategy_name,) in strategies: 
            strategy_info = self.db.get_manifest(strategy=strategy_name) 
            if not strategy_info or not strategy_info.get('chunk_size'): 
                continue 
            
            partial_ids = self.db.ask_view(view=view_name, keys='partial_submdats') 
            
            for submdat_id in tqdm(partial_ids, desc=f"Recomputing chunks for {strategy_name}"): 
                strategy_submdat_info = self.db.get_strategy_submdats(strategy_name, submdat_id) 
                if not strategy_submdat_info: 
                    continue 
                
                view_submdat = self.db.connect(check_same_thread=False).cursor().execute( 
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
        if hasattr(self, '_shuffle_file') and self._shuffle_file:
            try:
                self._shuffle_file.close()
            except Exception:
                pass
            self._shuffle_file = None

        if view_name is None:
            raise ValueError("A valid view name must be provided")

        view_meta = self.db.get_manifest(view=view_name) 
        if not view_meta: 
            raise ValueError(f"View '{view_name}' not found in manifest")

        self._active_view = view_name
        self.current_view = view_name 
        self.document_index = 0
        # Set the MDAT length to the number of documents in the view
        self.len = self.db.get_view_totals(view_name=view_name)['document_number']
        self._active_view_shuffled = view_meta.get('shuffle_struct_format') is not None 
        # Load the actual submdats
        self._populate_submdat()
    def precompute_chunks_distributed(self, num_workers=120):
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
                    chunks_yielded = num_chunks // self.chunk_multiplier
                    worker_chunk_counts[worker_id] += chunks_yielded
                    doc_index += 1
                    pbar.update(1)
            except StopIteration:
                pass
        
        self.set_iteration_modality(original_modality)
        
        self.db._store_distributed_precomputed_length(
            view_name=self.current_view,
            strategy_name=self.current_strategy.strategy_name,
            num_workers=num_workers,
            worker_chunk_counts=worker_chunk_counts
        )
        
        return {
            'num_workers': num_workers,
            'total_chunks': sum(worker_chunk_counts),
            'min_chunks': min(worker_chunk_counts),
            'max_chunks': max(worker_chunk_counts)
        }        
    def start_prefetch(self, max_prefetch=32768, shuffled=True):
            """
            Starts a background thread that fills a queue with upcoming documents.
            """
            if hasattr(self, "_prefetch_thread") and self._prefetch_thread is not None:
                return 

            if shuffled and not getattr(self, '_active_view_shuffled', None):
                 raise MDatNotShuffled("Dataset view is not shuffled.")
            
            if not hasattr(self, 'document_index'):
                self.document_index = 0
                
            if shuffled:
                if not hasattr(self, '_shuffle_file') or self._shuffle_file is None:
                    self._init_shuffle_file()
            
            
            ds_map = self.db.get_dsmap(key='id')


            self._prefetch_queue = queue.Queue(max_prefetch)
            self._prefetch_stop = threading.Event()

            def _prefetch_loop():
                while not self._prefetch_stop.is_set():
                    if self._prefetch_queue.full():
                        time.sleep(0.001)
                        continue
                    
                    try:
                        doc,self.document_index = self._load_next_document_inner(
                            ds_map, 
                            self.document_index, 
                            shuffled
                        )
                                
                        self._prefetch_queue.put(doc)
                        
                    except StopIteration:
                        self._prefetch_queue.put(None) #Sentinel for exhausted dataset
                        return
                    except Exception as e:
                        print(f"[Prefetch Thread Error] {e}")
                        self._prefetch_queue.put(None)
                        return

            t = threading.Thread(target=_prefetch_loop, daemon=True)
            t.start()
            print(f"Prefetch thread {t} started with queue size {max_prefetch}. Increase or decrease this value according to your RAM to improve performance. ")
            self._prefetch_thread = t
            
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

            self._prefetch_thread.join(timeout=2.0)
            self._prefetch_thread = None
            self._prefetch_queue = None
            
    def load_next_document(self, shuffled=True):
            if shuffled and not getattr(self, '_active_view_shuffled', None):
                 raise MDatNotShuffled("Dataset view is not shuffled.")
            
            if not hasattr(self, 'document_index'):
                 self.document_index = 0
                 
            
            if hasattr(self, "_prefetch_queue") and self._prefetch_queue is not None:
                while True:
                    try:
                        doc = self._prefetch_queue.get(timeout=15)
                        if doc is None:
                            self.stop_prefetch()
                            raise StopIteration
                        return doc
                    except queue.Empty:
                        if self._prefetch_thread is None or not self._prefetch_thread.is_alive():
                            self.stop_prefetch()
                            raise RuntimeError("Prefetch thread died unexpectedly.")
                        continue
            else:
                ds_map = self.db.get_dsmap(key='id')
                
                if shuffled:
                    if not hasattr(self, '_shuffle_file') or self._shuffle_file is None:
                        self._init_shuffle_file()
                
                current_document, self.document_index = self._load_next_document_inner(
                    ds_map, 
                    self.document_index, 
                    shuffled
                )
                return current_document

    def _load_next_document_inner(self, ds_map, document_index, shuffled=True, ) -> None:
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
                    if document_index % self.world_size != self.rank_size:
                        document_index += 1
                        continue  # Skip this shuffle entry
                
                submdat_id, doc_id = struct.unpack(self._shuffle_struct_format, data)
                current_document = self.loaded_submdats[ds_map[submdat_id]][doc_id]
                document_index += 1
                break
        else:
            current_pos = 0
            for submdat_id in sorted(ds_map.keys()): 
                submdat_name = ds_map[submdat_id] 
                if submdat_name in self.loaded_submdats:
                    submdat_len = len(self.loaded_submdats[submdat_name])
                    if self.document_index < current_pos + submdat_len:
                        doc_id = self.document_index - current_pos
                        current_document = self.loaded_submdats[submdat_name][doc_id]
                        document_index += 1
                        return
                    current_pos += submdat_len
            raise StopIteration
        return current_document, document_index
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
        c = self.db.connect(check_same_thread=False).cursor() 
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
            "iteration_modality":self.iteration_modality,
            "document_index":getattr(self,"document_index",0),
            "current_chunk_step":getattr(self,"current_chunk_step",0),
            "_iteration_count":getattr(self,"_iteration_count",0),
            "_recurrent_documents":getattr(self,"_recurrent_documents",[]),
            "_recurrent_steps":getattr(self,"_recurrent_steps",[]),
            "_max_iteration":getattr(self,"_max_iterations",None),
            "_cached_length":getattr(self,"_cached_length",None),
            "saved_world_size":1 if not self.dist else self.world_size
        }
        return state_dict
        pass
    def restore_state_dict(self,state_dict):
        self.document_index = int(state_dict["document_index"])
        self.current_chunk_step = int(state_dict["current_chunk_step"])
        self._iteration_count = int(state_dict["_iteration_count"])
        self._recurrent_documents=list(state_dict["_recurrent_documents"]
        self._recurrent_steps=list(state_dict["_recurrent_steps"]
        self._max_iteration=None if state_dict["_max_iterations"] is None else int(state_dict['max_iterations']
        self._cached_length=None if state_dict["_cached_length"] is None else int(state_dict['_cached_length']
        
        self._skip_reinit=True
        self._seek_document_pointer(self.document_index)
        
        
    def __iter__(self,with_prefetch=True):
            # Reset the iteration (ex. for a new epoch)  
            self.stop_prefetch()
            
            # 2. Reset State
            self.current_document = None
            if not hasattr(self,'_skip_reinit'):
                self.current_chunk_step = 0
                self.document_index = 0
                self._iteration_count = 0
                
                if hasattr(self, '_max_iterations'):
                    del self._max_iterations
                
                # Reset file pointer for the new thread to use
                if hasattr(self, '_shuffle_file') and self._shuffle_file is not None:
                    self._shuffle_file.seek(0)
                if with_prefetch:
                    if self.prefetch_buffer is not None and self.prefetch_buffer>0:
                        self.start_prefetch(max_prefetch=self.prefetch_buffer)
            else:
                print("MDAT is resuming training from external logic.")
                print("Document index: ",self.document_index)
                print("Chunk step: ", self.current_chunk_step)
                shuffled=True #TODO
                if shuffled:
                    if not hasattr(self, '_shuffle_file') or self._shuffle_file is None:
                        self._init_shuffle_file()
                self._seek_document_pointer(self.document_index)
                if with_prefetch:
                    if self.prefetch_buffer is not None and self.prefetch_buffer>0:
                        self.start_prefetch(max_prefetch=self.prefetch_buffer)                
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
            wanted_from_dbs.append('meta')
        # Set the iteration modality in each submdat
        for sbm in self.loaded_submdats.values():
            sbm.set_default_wanted(from_dbs=wanted_from_dbs, from_strategy=wanted_from_strategy, raw=return_raw, add_special_tokens=add_special_tokens)
        
    def get_iteration_modality(self):
        return self.current_iteration_modality

    def set_strategy(self, strategy_name, max_seq_len=None):
        #1. Load the strategy object
        self.current_strategy = PretokenizationStrategy.from_dict(db=self.db,strategy_name=strategy_name,pretok_path=self.pretok_path,functions_path=self.functions_path)
        #2. Set the chunk multiplier
        """
        A method to set the multiplier based on the max. sequence length required by the model
        If the required sequence length is not a multiple of strategy's base sequence length,
        an exception will be thrown.
        """
        if max_seq_len is not None:
            base_chunk_size = self.current_strategy.chunk_size
            if base_chunk_size is not None:
               assert max_seq_len % base_chunk_size == 0
               self.chunk_multiplier=(max_seq_len // base_chunk_size)
               self.max_seq_len = max_seq_len
        else:
            self.max_seq_len=self.current_strategy.chunk_size
            self.chunk_multiplier=1        
        #3. Initialize counters
        self.total_chunks = 0
        self.total_tokens = 0    
        self.total_divided_chunks = 0    
        #4. Select datasets from the current view. If datasets are 'partial' in the view, use precomputed chunks from partial datasets
        asking_view=dict() #Sporco
        asking_view['partial_submdats']=self.db.ask_view(view=self.current_view, keys='partial_submdats')  #Sporco
        asking_view['preserved_submdats']=self.db.ask_view(view=self.current_view, keys='preserved_submdats')  #Sporco
        used_submdats=asking_view['preserved_submdats']+asking_view['partial_submdats']
        ds_map=self.db.get_dsmap(key='id')
        strategy=strategy_name #Sporco
        for sbm_id in used_submdats:
            try:
                sbm=self[ds_map[sbm_id]]
                sbm.set_strategy(strategy_name)
                strategy_stats = self.db.get_strategy_submdats(strategy_name, sbm_id) 
                self.total_chunks += strategy_stats.get('total_chunks', 0) 
                self.total_tokens += strategy_stats.get('total_tokens', 0) 
                if 'chunks' in self.current_strategy.returns:
                    #This strategy has token chunks, need precomputation to get actual length dependent on max_seq_len
                    if sbm_id in asking_view['preserved_submdats']:
                        precomp = self.db.get_strategy_precomp(strategy_name=strategy, submdat_id=sbm_id, multiplier=self.chunk_multiplier) 
                    else:
                        precomp = self.db.get_strategy_precomp(strategy_name=strategy, submdat_id=sbm_id, view=self.current_view,multiplier=self.chunk_multiplier)   
                if precomp: 
                    self.total_divided_chunks += precomp[0].get('precomputed_length', 0) 
            except SubMdatMissesStrategy:
                print(f"Submdat {sbm.submdat_name} is not pretokenized with strategy {strategy}.")
                self.total_chunks = 0 
                self.total_tokens = 0 
                self.total_divided_chunks = 0 
                self.current_strategy = None 
            except Exception as e:
               print(e)
    def export_view(self,output_file,view='default',wanted='document',data_field_name='text',with_meta=True,limiter=None):
        """
        A function (WIP) to export all the documents from a view into an output file
        Currently supports only documents (with/without metadata) and JSONL output.
        
        """
        from tqdm import tqdm
        import orjson

        self.set_view(view)
        self._reset_document_pointer()
        self.set_iteration_modality(wanted, with_meta=with_meta)

        counter=0
        with open(output_file, "wb") as f:
            for item in tqdm(self):
                if limiter:
                    counter+=1
                    if counter>limiter:
                        return 0
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
            
    def get_distributed_length_before_training(self,num_devices=1):
        """
        Lightning initialize the dataset with torch.dist not set when first gathering the dataset info
        however, dataset length is required for correct learning rate scheduling calculation
        This function, given a target of gpu, returns the maximum distributed length, the one that will actually be used
        during training
        """
        assert self.current_view is not None
        assert self.current_strategy is not None
        if self.current_iteration_modality in ['chunked_tokens','chunked_for_recurrence']:
            max_len=0
            for w in range(num_devices):
                w_len=self.db.get_worker_length_distributed(target_num_workers=num_devices, target_worker_id=w, view_name=self.current_view, strategy_name=self.current_strategy.strategy_name)
                if w_len>max_len:
                    max_len=w_len
            return max_len
        else:
            print("You are asking for distributed length but you are not using chunked_tokens modality.")
            raise Exception
    def __len__(self):
        """
        The len behaves in different ways:
        1) If nothing is specified, it return the summed length of each submdat
        2) If a view is specified, it computes the length according to the view
        3) If a pretokenization strategy is set, it returns the number of chunks according to max_seq_len (necessary for model training)
        """
        if self.dist and self.current_iteration_modality in ['chunked_tokens','chunked_for_recurrence']:
            try:
                if self._cached_length is not None:
                    return self._cached_length
                # Get this worker's actual length

                this_worker_length = self.db.get_worker_length_distributed(target_num_workers=self.world_size, target_worker_id=self.rank_size, view_name=self.current_view, strategy_name=self.current_strategy.strategy_name)
                print(f"WORKER {self.rank_size} got length {this_worker_length}")
                # Broadcast maximum across all workers so everyone agrees
                import torch.distributed as dist
                import torch
                if dist.is_initialized():
                    length_tensor = torch.tensor([this_worker_length], dtype=torch.long).cuda()
                    dist.all_reduce(length_tensor, op=dist.ReduceOp.MAX)
                    max_length = length_tensor.item()
                    print(f"Max length set to {max_length}")
                    self._cached_length=max_length
                    return max_length
                else:
                    return this_worker_length
            except ValueError:
                # Precompute
                print("I need to precompute lengths for an incompatible workers set up")
                self.precompute_chunks_distributed(self.world_size)
                return self.__len__() 
        else:
            return self.len


# Exceptions
if True:
     #This "if True" is just to easily collapse the exceptions in my editor, will be removed
    class MDatNotFound(Exception):
        """Raised when dataset is not found."""
        pass

    class MDatAlreadyPresent(Exception):
        """Raised when dataset already exists and overwrite is False."""
        pass

    class SubmDatNotFound(Exception):
        """Raised when sub-dataset is not found."""
        pass
    class SubMdatMissesStrategy(Exception):
        pass

    class MDatIsReadOnly(Exception):
        """Raised when trying to modify readonly dataset."""
        pass

    class TooManySubmDats(Exception):
        """Raised when too many submdats."""
        #print("FATAL: You've reached the maximum amount of Sub-dataset allowed in the Mdat. Please, increase the DSMAP_BITS in the manifest.")
        pass

    class MDatAlreadyShuffled(Exception):
        """Raised when mdat is already shuffled."""
        pass

    class NameAlreadyUsed(Exception):
        """Raised when name is already used."""
        pass

    class StrategyNotRegistered(Exception):
        """Raised when strategy is not registered."""
        pass

import sqlite3
from typing import Optional
from datetime import datetime


class DatabaseManager:
    def __init__(self, db_path: str):
        """
        DatabaseManager manages the SQLite connection and schema initialization.
        """
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None

    def connect(self):
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path, isolation_level=None, detect_types=sqlite3.PARSE_DECLTYPES,check_same_thread=False)
            self.conn.execute("PRAGMA foreign_keys = ON;")
        return self.conn

    def _initialize_db(self):
        """
        Initializes the SQLite database schema
        """
        conn = self.connect()
        cur = conn.cursor()

        # Tabella "MANIFEST" (in realtà, sono semplicemente attributi)
        #     dsmap_bits //SHORTINT (8,16 or 32), it indicates with how many bits the id of the subdataset should be represented. If 8 is chosen, it will be much more efficient, but the subdatasets must be limited to 255
        #     modalities //JSON ex {"text"} or {"text","audio","image"}...
        cur.execute("""
        CREATE TABLE IF NOT EXISTS manifest (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dsmap_bits INTEGER NOT NULL CHECK (dsmap_bits IN (8,16,32,64)),
            modalities TEXT NOT NULL -- JSON
        );
        """)

        # Tabella "SUBMDAT"
        # * A Mdat doesn't directly contain data. Data is contained in the sub-mdat objects, subclasses of the parent dataset. In this way, it is possible to easily separe logically different dataset to be used to train a model
        #
        #     name //STR
        #     ds_map_id //PROGRESSIVE-ID
        #     data_type //JSON ex {"text"} or {"text","audio","image"}
        #     document_number //LONG-UNSIGNED-INT how many documents in the dataset
        #     modality //JSON ex {"text"} or {"text","audio","image"}...
        #     data_key //Short string, the original key of the data (ex. 'text')
        #     errors_counters //JSON
        #     finalized_at //DATETIME
        cur.execute("""
        CREATE TABLE IF NOT EXISTS submdat (
            ds_map_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            data_type TEXT,              -- JSON
            document_number INTEGER CHECK (document_number >= 0),
            modality TEXT,               -- JSON
            data_key TEXT,
            errors_counters TEXT,       -- JSON (can be NULL)
            finalized_at TEXT           -- DATETIME stored as TEXT
        );
        """)

        # Tabella "DATABASE"
        # * Actually, not even the Submdat holds the data. Data is contained in highly efficient KV stores, such as LMDB. This table contains all the useful references to directly call a database. It is used also by other parts of the mdat, for example, the pretok strategy manager.
        #    
        #     ID //PROGRESSIVE-ID
        #     Type //STRING ex LMDB
        #     Compression_level //SHORTINT (0-9), level of compression
        #     map_size //LONGINT lmdb requires a "map_size" parameter
        #     disk_size //LONGINT the actual size on disk
        #     extra_data //JSON (for further expansions)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS database_ref (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL,
            compression_level INTEGER CHECK (compression_level BETWEEN 0 AND 9),
            map_size INTEGER,
            disk_size INTEGER,
            extra_data TEXT  -- JSON
        );
        """)

        # Tabella "SUBMDAT-DATABASE"
        # *This table maps the relevant databases of the submdat. Usually, we will just have one dataset with raw data (data) and one with metadata (meta). However, for further expansion (classification, multimodality...) it is possible to map an arbitrary amount of datasets
        #     submdat_id //linked to submdat
        #     database_id //linked to database
        #     raw_data_bytes // LONGINT the amount of bytes of the original data (for example, chars in case of documents)
        #     is_data //BOOL (mututally exclusive)
        #     is_meta //BOOL (mututally exclusive)
        #     is_extra //JSON or NULL (mututally exclusive)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS submdat_database (
            submdat_id INTEGER NOT NULL,
            database_id INTEGER NOT NULL,
            raw_data_bytes INTEGER CHECK (raw_data_bytes >= 0),
            is_data INTEGER DEFAULT 0 CHECK (is_data IN (0,1)),
            is_meta INTEGER DEFAULT 0 CHECK (is_meta IN (0,1)),
            is_extra TEXT, -- JSON stored as TEXT when extra metadata is present; NULL otherwise
            PRIMARY KEY (submdat_id, database_id),
            FOREIGN KEY (submdat_id) REFERENCES submdat(ds_map_id) ON DELETE CASCADE,
            FOREIGN KEY (database_id) REFERENCES database_ref(id) ON DELETE CASCADE
        );
        """)

        # Tabella "PRETOK_STRATEGY"
        # * The MDAT can store, alongside with the data, many different "pretokenizations" of the data, for faster retrieval during model training.
        # * Each pretok strategy can call a "splitter" (either built-in or user-defined) that handles the data processing, ex, splitting in chunks
        #
        #     strategy_name //STR
        #     modality  //JSON ex {"text"} or {"text","audio","image"}...
        #     tokenizer_type //STR ex "huggingface"
        #     tokenizer_name //STR ex "sapienzanlp/Minerva-350M-base-v1.0"
        #     tokenizer_args //JSON, extra arguments for the tokenizer
        #     bos_token_id //SINGLE INT
        #     eos_token_id //SINGLE INT
        #     mask_token_id //SINGLE INT
        #     splitter_class //STR ex "split_and_tokenize_by_nltk_sentences"
        #     splitter_init //JSON, arguments for the splitter's init
        #     splitter_arguments //JSON, arguments for the splitter's call
        #     chunk_size //INT, ex 1024 if the base chunk are 1024tok long
        #     wants_from_db //JSON, a list of required databases from the SUMBDAT to process the data, example, a text pretokenization strategy would just require "data" that contains raw text, thus ["data"]
        #     returns //JSON ex {"tokens","chunked_tokens"}
        #     required_databases //JSON ex {"tokens","chunks"}
        cur.execute("""
        CREATE TABLE IF NOT EXISTS pretok_strategy (
            strategy_name TEXT PRIMARY KEY,
            modality TEXT,             -- JSON stored as TEXT
            tokenizer_type TEXT,
            tokenizer_name TEXT,
            tokenizer_args TEXT,       -- JSON stored as TEXT
            bos_token_id INTEGER,
            eos_token_id INTEGER,
            mask_token_id INTEGER,
            wants_raw INTEGER DEFAULT 1, --BOOL STORED AS INT
            splitter_class TEXT,
            splitter_init TEXT,        -- JSON stored as TEXT
            splitter_arguments TEXT,   -- JSON stored as TEXT
            chunk_size INTEGER CHECK (chunk_size >= 0),
            wants_from_db TEXT,        -- JSON stored as TEXT (list of required db names)
            returns TEXT,              -- JSON stored as TEXT
            chunks_datatype TEXT, 
            vocab_size INT,                           
            tokens_datatype TEXT,              
            required_databases TEXT    -- JSON stored as TEXT
        );
        """)

        # Tabella "PRETOK_STRATEGY-SUBMDAT"
        # The table that maps each pretokenization strategy to a submdat
        #
        #     strategy_name //linked to strategy
        #     submdat_id //linked to submdat
        #     finalized_at //DATETIME
        #     total_tokens //LONGINT
        #     max_tokens_per_doc //LONGINT
        #     total_chunks //LONGINT
        #     max_chunks_per_doc //LONGINT
        #     processed_docs //LONGINT
        cur.execute("""
        CREATE TABLE IF NOT EXISTS pretok_strategy_submdat (
            strategy_name TEXT NOT NULL,
            submdat_id INTEGER NOT NULL,
            finalized_at TEXT,  -- ISO8601 DATETIME stored as TEXT (can be NULL)
            total_tokens INTEGER CHECK (total_tokens >= 0),
            max_tokens_per_doc INTEGER CHECK (max_tokens_per_doc >= 0),
            total_chunks INTEGER CHECK (total_chunks >= 0),
            max_chunks_per_doc INTEGER CHECK (max_chunks_per_doc >= 0),
            processed_docs INTEGER CHECK (processed_docs >= 0),
            PRIMARY KEY (strategy_name, submdat_id),
            FOREIGN KEY (strategy_name) REFERENCES pretok_strategy(strategy_name) ON DELETE CASCADE,
            FOREIGN KEY (submdat_id) REFERENCES submdat(ds_map_id) ON DELETE CASCADE
        );
        """)

        # Tabella "PRETOK_STRATEGY-SUBMDAT-PRECOMPUTED-LENGTH"
        # * In case of chunked tokens, it is useful to precompute the length of iteration in case a multiplier of base_chunk_size is set. These data are used when entire subdatasets are selected.
        #     strategy_name //linked to strategy
        #     submdat_id //linked to submdat
        #     multiplier //SHORT
        #     precomputed_length //LONGINT 
        cur.execute("""
        CREATE TABLE IF NOT EXISTS pretok_strategy_submdat_precomputed_length (
            strategy_name TEXT NOT NULL,
            submdat_id INTEGER NOT NULL,
            multiplier INTEGER NOT NULL CHECK (multiplier >= 0),
            precomputed_length INTEGER CHECK (precomputed_length >= 0),
            PRIMARY KEY (strategy_name, submdat_id, multiplier),
            FOREIGN KEY (strategy_name) REFERENCES pretok_strategy(strategy_name) ON DELETE CASCADE,
            FOREIGN KEY (submdat_id) REFERENCES submdat(ds_map_id) ON DELETE CASCADE
        );
        """)

        # Tabella "PRETOK_STRATEGY-DB"
        # The table that maps each pretokenization strategy to a submdat through one or more db
        #     strategy_name //linked to strategy
        #     submdat_id //linked to submdat  
        #     id //PROGRESSIVE ID
        #     is_complete //BOOL
        #     datatype //str ex. 'uint16', 'uint32, 'float', 'tensor'...
        #     is_tokens //BOOL (mututally exclusive)
        #     is_chunks //BOOL (mututally exclusive)
        #     is_extra //JSON or NULL   (mututally exclusive)   
        cur.execute("""
        CREATE TABLE IF NOT EXISTS pretok_strategy_db (
            db_id INTEGER NOT NULL,
            strategy_name TEXT NOT NULL,
            submdat_id INTEGER NOT NULL,
            is_complete INTEGER DEFAULT 0 CHECK (is_complete IN (0,1)),
            datatype TEXT,
            is_tokens INTEGER DEFAULT 0 CHECK (is_tokens IN (0,1)),
            is_chunks INTEGER DEFAULT 0 CHECK (is_chunks IN (0,1)),
            is_extra TEXT, -- JSON stored as TEXT (nullable)
            FOREIGN KEY (strategy_name) REFERENCES pretok_strategy(strategy_name) ON DELETE CASCADE,
            FOREIGN KEY (db_id) REFERENCES database_ref(id) ON DELETE CASCADE,
            FOREIGN KEY (submdat_id) REFERENCES submdat(ds_map_id) ON DELETE CASCADE,
            CHECK (
                (COALESCE(is_tokens,0) + COALESCE(is_chunks,0) + CASE WHEN is_extra IS NOT NULL THEN 1 ELSE 0 END) = 1
            )
        );
        """)

        # Tabella "PRETOK_STRATEGY-VIEW-SUBMDAT-PRECOMPUTED-LENGTH"
        # * In case of chunked tokens, it is useful to precompute the length of iteration in case a multiplier of base_chunk_size is set. These data are used when a partial submdat is included in a view.
        #     strategy_name //linked to strategy
        #     view_id //linked to view
        #     submdat_id //linked to submdat
        #     multiplier //SHORT
        #     precomputed_length //LONGINT 
        cur.execute("""
        CREATE TABLE IF NOT EXISTS pretok_strategy_view_submdat_precomputed_length (
            strategy_name TEXT NOT NULL,
            view_name TEXT NOT NULL,
            submdat_id INTEGER NOT NULL,
            multiplier INTEGER NOT NULL CHECK (multiplier >= 0),
            precomputed_length INTEGER CHECK (precomputed_length >= 0),
            PRIMARY KEY (strategy_name, view_name, submdat_id, multiplier),
            FOREIGN KEY (strategy_name) REFERENCES pretok_strategy(strategy_name) ON DELETE CASCADE,
            FOREIGN KEY (view_name) REFERENCES view(view_name) ON DELETE CASCADE,
            FOREIGN KEY (submdat_id) REFERENCES submdat(ds_map_id) ON DELETE CASCADE
        );
        """)

        # Tabella "VIEW"
        # * In a Mdat, it is possible to build a "view" over a dataset, for example choosing only some example to build a smaller set for ablation studies
        #     view_name //STR (KEY)
        #     shuffle_struct_format //STRING 
        cur.execute("""
        CREATE TABLE IF NOT EXISTS view (
            view_name TEXT PRIMARY KEY,
            shuffle_struct_format TEXT
        );
        """)

        # Tabella "VIEW-SUMBDAT"
        # * For each view-submdat couple, how it is considered in the view
        #     view_name KEY
        #     sumbdat_id KEY
        #     is_skipped BOOL (mututally exclusive) //in case the submdat is skipped in this view
        #     is_preserved BOOL (mututally exclusive) //in case the submdat is entirely taken
        #     is_partial BOOL (mututally exclusive) //in case only a part of the submdat is taken
        #     document_number //NULL if skipped or preserved, or the number of selected documents
        #     bytes_criteria //LONGINT if the criteria to choose the view was data size, that number is shown here for reference
        cur.execute("""
        CREATE TABLE IF NOT EXISTS view_submdat (
            view_name TEXT NOT NULL,
            submdat_id INTEGER NOT NULL,
            is_skipped INTEGER DEFAULT 0 CHECK (is_skipped IN (0,1)),
            is_preserved INTEGER DEFAULT 0 CHECK (is_preserved IN (0,1)),
            is_partial INTEGER DEFAULT 0 CHECK (is_partial IN (0,1)),
            document_number INTEGER,  -- NULL if skipped or preserved, otherwise number of selected documents
            bytes_criteria INTEGER,   -- reference if selection was based on data size
            PRIMARY KEY (view_name, submdat_id)      
        );
        """)
        # Table "PRETOK_STRATEGY-VIEW-DISTRIBUTED-LENGTH"
        # * Precomputed chunk counts for distributed training configurations
        #     strategy_name //linked to strategy
        #     view_name //linked to view
        #     num_workers //INT number of workers this was precomputed for
        #     worker_id //INT the worker rank (0 to num_workers-1)
        #     chunk_count //LONGINT number of chunks this worker processes
        cur.execute("""
        CREATE TABLE IF NOT EXISTS pretok_strategy_view_distributed_length (
            strategy_name TEXT NOT NULL,
            view_name TEXT NOT NULL,
            num_workers INTEGER NOT NULL CHECK (num_workers > 0),
            worker_id INTEGER NOT NULL CHECK (worker_id >= 0),
            chunk_count INTEGER NOT NULL CHECK (chunk_count >= 0),
            PRIMARY KEY (strategy_name, view_name, num_workers, worker_id),
            FOREIGN KEY (strategy_name) REFERENCES pretok_strategy(strategy_name) ON DELETE CASCADE,
            FOREIGN KEY (view_name) REFERENCES view(view_name) ON DELETE CASCADE,
            CHECK (worker_id < num_workers)
        );
        """)

        # Add index
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_pretok_distributed_lookup 
        ON pretok_strategy_view_distributed_length(view_name, strategy_name, num_workers);
        """)
        # Indexes to speed up common access patterns
        cur.execute("CREATE INDEX IF NOT EXISTS idx_submdat_dsmap ON submdat(ds_map_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_submdat_name ON submdat(name);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_dbref_type ON database_ref(type);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pretok_strategy_modality ON pretok_strategy(modality);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pretok_strategy_submdat ON pretok_strategy_submdat(submdat_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pretok_strategy_db_strategy ON pretok_strategy_db(strategy_name);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_submdat_database_submdat ON submdat_database(submdat_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_view_submdat_view ON view_submdat(view_name);")

        conn.commit()

    def _dict_from_row(self, cursor, row):
        if not row:
            return {}
        return dict(zip([x[0] for x in cursor.description], row))

    def _fetchone(self, q, params=()):
        c = self.connect().cursor()
        r = c.execute(q, params).fetchone()
        return self._dict_from_row(c, r)

    def _fetchall(self, q, params=()):
        c = self.connect().cursor()
        rows = c.execute(q, params).fetchall()
        return [dict(zip([x[0] for x in c.description], r)) for r in rows]

    def _update_table(self, table, data, where_clause=None, where_params=()):
        if not data:
            return
        cols = ",".join([f"{k}=?" for k in data.keys()])
        q = f"UPDATE {table} SET {cols}"
        params = list(data.values())
        if where_clause:
            q += f" WHERE {where_clause}"
            params.extend(where_params)
        c = self.connect().cursor()
        c.execute(q, tuple(params))
        self.connect().commit()

    def _insert(self, table, data):
        """Generic insert. `data` is dict of col->value. Returns lastrowid."""
        cols = ",".join(data.keys())
        vals = ",".join(["?" for _ in data])
        q = f"INSERT INTO {table}({cols}) VALUES({vals})"
        c = self.connect().cursor()
        c.execute(q, tuple(data.values()))
        self.connect().commit()
        return c.lastrowid


    def get_manifest(self, submdat=None, strategy=None, view=None):
        """Return a single dict from manifest OR a row from submdat/pretok_strategy/view.
        Maintains original call signatures for compatibility.
        """
        if not submdat and not strategy and not view:
            return self._fetchone("SELECT * FROM manifest LIMIT 1")

        if submdat:
            where = ("ds_map_id=?" if isinstance(submdat, int) else "name=?", (submdat,))
            return self._fetchone(f"SELECT * FROM submdat WHERE {where[0]}", where[1])
        if strategy:
            return self._fetchone("SELECT * FROM pretok_strategy WHERE strategy_name=?", (strategy,))
        return self._fetchone("SELECT * FROM view WHERE view_name=?", (view,))

    def create_manifest(self, dsmap_bits, modalities):
        """Keep original signature: inserts into manifest."""
        return self._insert("manifest", {"dsmap_bits": dsmap_bits, "modalities": json.dumps(modalities)})

    def get_dsmap(self, key, value=None):
        """Unified accessor for dsmap-related lookups. Returns either scalar or dict as original."""
        c = self.connect().cursor()
        if key == "name" and value is not None:
            r = c.execute("SELECT ds_map_id FROM submdat WHERE name=?", (value,)).fetchone()
            return r[0] if r else None
        if key == "id" and value is not None:
            r = c.execute("SELECT name FROM submdat WHERE ds_map_id=?", (value,)).fetchone()
            return r[0] if r else None
        if key == "max_id":
            r = c.execute("SELECT MAX(ds_map_id) FROM submdat").fetchone()
            return (r[0] or 0)
        if key == "name":
            rows = c.execute("SELECT ds_map_id,name FROM submdat").fetchall()
            return {r[1]: r[0] for r in rows}
        rows = c.execute("SELECT ds_map_id,name FROM submdat").fetchall()
        return {r[0]: r[1] for r in rows}

    def update_manifest(self, data, submdat=None, strategy=None, view=None):
        data = dict(data)
        c = self.connect().cursor()
        cols_info = [x[1] for x in c.execute("PRAGMA table_info('manifest')").fetchall()]
        if 'finalized_at' in cols_info:
            data['finalized_at'] = datetime.utcnow().isoformat()

        if submdat:
            table = 'submdat'
            where_clause = ('ds_map_id=?' if isinstance(submdat, int) else 'name=?')
            where_params = (submdat,)
        elif strategy:
            table = 'pretok_strategy'
            where_clause = 'strategy_name=?'
            where_params = (strategy,)
        elif view:
            table = 'view'
            where_clause = 'view_name=?'
            where_params = (view,)
        else:
            table = 'manifest'
            where_clause = None
            where_params = ()

        self._update_table(table, data, where_clause, where_params)

    def ask_view(self, view, keys):
        c = self.connect().cursor()
        if isinstance(keys, str):
            keys = [keys]
        mapping = {
            'preserved_submdats': 'is_preserved',
            'partial_submdats': 'is_partial',
            'skipped_submdats': 'is_skipped'
        }
        res = {}
        for k in keys:
            col = mapping[k]
            rows = c.execute(f"SELECT submdat_id FROM view_submdat WHERE view_name=? AND {col}=1", (view,)).fetchall()
            res[k] = [r[0] for r in rows]
        return res if len(keys) > 1 else res[keys[0]]

    def add_database(self, _type, compression_level, map_size, disk_size, extra_data):
        return self._insert('database_ref', {
            'type': _type,
            'compression_level': compression_level,
            'map_size': map_size,
            'disk_size': disk_size,
            'extra_data': extra_data
        })

    def link_submdat_view(self, submdat_id, view_name, is_skipped, is_partial, is_preserved, document_number, bytes_criteria):
        """Insert or update link into view_submdat (UPSERT semantics)."""
        flags = (int(bool(is_skipped)), int(bool(is_partial)), int(bool(is_preserved)))
        if sum(flags) != 1:
            raise ValueError("Exactly one of is_skipped, is_partial, is_preserved must be true")

        c = self.connect().cursor()

        # Check if the row already exists
        exists = c.execute(
            "SELECT 1 FROM view_submdat WHERE submdat_id=? AND view_name=?",
            (submdat_id, view_name)
        ).fetchone()

        if exists:
            # Update existing entry
            c.execute("""
                UPDATE view_submdat
                SET is_skipped=?, is_partial=?, is_preserved=?, document_number=?, bytes_criteria=?
                WHERE submdat_id=? AND view_name=?
            """, (flags[0], flags[1], flags[2], document_number, bytes_criteria, submdat_id, view_name))
        else:
            # Insert new entry
            c.execute("""
                INSERT INTO view_submdat(submdat_id, view_name, is_skipped, is_partial, is_preserved, document_number, bytes_criteria)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (submdat_id, view_name, flags[0], flags[1], flags[2], document_number, bytes_criteria))

        self.connect().commit()



    def link_submdat_database(self, submdat_id, database_id, raw_data_bytes, is_data, is_meta, is_extra):
        return self._insert('submdat_database', {
            'submdat_id': submdat_id,
            'database_id': database_id,
            'raw_data_bytes': raw_data_bytes,
            'is_data': int(bool(is_data)),
            'is_meta': int(bool(is_meta)),
            'is_extra': int(bool(is_extra))
        })

    def add_submdat(self, submdat_name):
        return self._insert('submdat', {'name': submdat_name})

    def add_view(self, view_name):
        c = self.connect().cursor()
        c.execute("INSERT INTO view(view_name,shuffle_struct_format) VALUES (?,?)", (view_name, None))
        self.connect().commit()
        return c.lastrowid

    def remove_submdat(self, submdat_id):
        c = self.connect().cursor()
        c.execute("DELETE FROM submdat WHERE ds_map_id=?", (submdat_id,))
        self.connect().commit()
    def get_storage_db_info(self,database_id):
        q="SELECT * FROM database_ref WHERE database_ref.id = ?"
        return self._fetchone(q,(int(database_id),))

        
    def get_submdat_databases(self, submdat_id, database_id=None, database_type=None):
        q = """
            SELECT dr.*, sd.raw_data_bytes, sd.is_data, sd.is_meta, sd.is_extra
            FROM database_ref dr
            JOIN submdat_database sd ON dr.id = sd.database_id
            WHERE sd.submdat_id = ?
        """
        params = [submdat_id]
        if database_id is not None:
            q += " AND sd.database_id = ?"
            params.append(database_id)
        result = self._fetchall(q, tuple(params))
        if database_type=='data':
            for r in result:
                if r['is_data']==1:
                    return r
        elif database_type=='meta':
            for r in result:
                if r['is_meta']==1:
                    return r            
        return result[0] if database_id is not None and result else result

    def update_submdat_database(self, submdat_id, database_id, data):
        dr_allowed = {'type','compression_level','map_size','disk_size','extra_data'}
        sd_allowed = {'raw_data_bytes','is_data','is_meta','is_extra','doc_count'}
        dr_fields = {k: v for k, v in data.items() if k in dr_allowed}
        sd_fields = {k: v for k, v in data.items() if k in sd_allowed}
        if dr_fields:
            self._update_table('database_ref', dr_fields, 'id=?', (database_id,))
        if sd_fields:
            self._update_table('submdat_database', sd_fields, 'submdat_id=? AND database_id=?', (submdat_id, database_id))

    def get_strategy_submdats(self, strategy_name, submdat_id=None):
        q = "SELECT * FROM pretok_strategy_submdat WHERE strategy_name = ?"
        params = [strategy_name]
        if submdat_id is not None:
            q += " AND submdat_id = ?"
            params.append(submdat_id)
        rows = self._fetchall(q, tuple(params))
        return rows[0] if submdat_id is not None and rows else rows

    def update_strategy_submdat(self, strategy_name, submdat_id, data):
        self._update_table('pretok_strategy_submdat', data, 'strategy_name=? AND submdat_id=?', (strategy_name, submdat_id))

    def add_strategy_submdat(self, strategy_name, submdat_id, data=None):
        data = data or {}
        payload = {'strategy_name': strategy_name, 'submdat_id': submdat_id}
        payload.update(data)
        return self._insert('pretok_strategy_submdat', payload)

    def remove_strategy_submdat(self, strategy_name, submdat_id):
        c = self.connect().cursor()
        c.execute("DELETE FROM pretok_strategy_submdat WHERE strategy_name=? AND submdat_id=?", (strategy_name, submdat_id))
        self.connect().commit()

    def get_strategy_dbs(self, strategy_name, submdat_id, db_id=None):
        q = """SELECT psd.*, dr.* FROM pretok_strategy_db psd
               JOIN database_ref dr ON psd.db_id = dr.id
               WHERE psd.strategy_name=? AND psd.submdat_id=?"""
        params = [strategy_name, submdat_id]
        if db_id is not None:
            q += " AND psd.id=?"
            params.append(db_id)
        rows = self._fetchall(q, tuple(params))
        return rows[0] if db_id is not None and rows else rows

    def update_strategy_db(self, database_id, data):
        psd_allowed = {'is_complete', 'datatype', 'is_tokens', 'is_chunks', 'is_extra'}
        dr_allowed = {'type', 'compression_level', 'map_size', 'disk_size', 'extra_data'}

        psd_fields = {k: v for k, v in data.items() if k in psd_allowed}
        dr_fields = {k: v for k, v in data.items() if k in dr_allowed}

        if psd_fields:
            self._update_table('pretok_strategy_db', psd_fields, 'db_id=?', (database_id,))

        if dr_fields:
            self._update_table('database_ref', dr_fields, 'id=?', (database_id,))


    def add_strategy_db(self, database_id, strategy_name, submdat_id, datatype, is_tokens,is_chunks,is_extra,is_complete=1):
        c = self.connect().cursor()
        c.execute("INSERT INTO pretok_strategy_db(db_id,strategy_name,submdat_id,datatype,is_tokens,is_chunks,is_extra,is_complete) VALUES(?,?,?,?,?,?,?,?)", (database_id,strategy_name, submdat_id,datatype,is_tokens,is_chunks,is_extra,is_complete))
        db_id = c.lastrowid
        self.connect().commit()
        return db_id

    def remove_strategy_db(self, db_id):
        c = self.connect().cursor()
        c.execute("DELETE FROM pretok_strategy_db WHERE id=?", (db_id,))
        self.connect().commit()
        
    def set_strategy_view_precomp(self, strategy_name, view_name, submdat_id, multiplier, precomputed_length):
        c = self.connect().cursor()
        c.execute("""INSERT OR REPLACE INTO pretok_strategy_view_submdat_precomputed_length
                     (strategy_name,view_name,submdat_id,multiplier,precomputed_length) VALUES(?,?,?,?,?)""",
                  (strategy_name, view_name, submdat_id, multiplier, precomputed_length))
        self.connect().commit()
        
    def get_strategy_precomp(self, strategy_name, view_name=None, submdat_id=None, multiplier=None):
        if view_name:
            table='pretok_strategy_view_submdat_precomputed_length'
        else:
            table='pretok_strategy_submdat_precomputed_length'
        q = f"SELECT * FROM {table} WHERE strategy_name=?"
        params = [strategy_name]
        if view_name:
            q += " AND view_name=?"
            params.append(view_name)
        if submdat_id:
            q += " AND submdat_id=?"
            params.append(submdat_id)
        if multiplier:
            q += " AND multiplier=?"
            params.append(multiplier)
        return self._fetchall(q, tuple(params))

    def set_strategy_submdat_precomp(self, strategy_name, submdat_id, multiplier, precomputed_length):
        c = self.connect().cursor()
        c.execute("""INSERT OR REPLACE INTO pretok_strategy_submdat_precomputed_length
                     (strategy_name,submdat_id,multiplier,precomputed_length) VALUES(?,?,?,?)""",
                  (strategy_name, submdat_id, multiplier, precomputed_length))
        self.connect().commit()

    def list_views(self):
        rows = self._fetchall("SELECT view_name FROM view")
        return [row['view_name'] for row in rows]
    def list_strategies(self):
        rows = self._fetchall("SELECT strategy_name FROM pretok_strategy")
        return [row['strategy_name'] for row in rows]

    def get_submdat_info(self, submdat_id):
        submdat_id=int(submdat_id)
        r = self._fetchone(
            """
            SELECT s.ds_map_id AS submdat_id, s.document_number,
                   sd.raw_data_bytes
            FROM submdat AS s
            LEFT JOIN submdat_database AS sd ON sd.submdat_id = s.ds_map_id
            WHERE s.ds_map_id = ? AND sd.is_data = ?
            """,
            (submdat_id,1,)
        )
        return r



    def get_view_submdat(self, view_name, submdat_id):
        return self._fetchone(
            "SELECT view_name, submdat_id, is_skipped, is_preserved, is_partial, document_number, bytes_criteria FROM view_submdat WHERE view_name=? AND submdat_id=?",
            (view_name, submdat_id)
        )

    def update_view_submdat(self, view_name, submdat_id, data):
        allowed = {'is_skipped','is_preserved','is_partial','document_number','bytes_criteria'}
        upd = {k: v for k, v in (data or {}).items() if k in allowed}
        if not upd:
            return
        self._update_table('view_submdat', upd, 'view_name=? AND submdat_id=?', (view_name, submdat_id))

    def get_view_totals(self, view_name, criterion='active'):
        """Return aggregated document_number and bytes_criteria for a view.
        criterion: 'active' (default) : rows where is_skipped=0
                   'preserved' : rows where is_preserved=1
                   'all' : all rows
        Returns dict { 'document_number': int, 'bytes_criteria': int }
        """
        c = self.connect().cursor()
        if criterion == 'preserved':
            where = 'WHERE view_name=? AND is_preserved=1'
        elif criterion == 'all':
            where = 'WHERE view_name=?'
        else:
            where = 'WHERE view_name=? AND is_skipped=0'
        q = f"SELECT SUM(COALESCE(document_number,0)) as document_number, SUM(COALESCE(bytes_criteria,0)) as bytes_criteria FROM view_submdat {where}"
        r = c.execute(q, (view_name,)).fetchone()
        if not r:
            return {'document_number': 0, 'bytes_criteria': 0}
        return {'document_number': int(r[0] or 0), 'bytes_criteria': int(r[1] or 0)}
    def _store_distributed_precomputed_length(self, view_name, strategy_name, num_workers, worker_chunk_counts):
        """
        Store precomputed chunk counts for distributed training.
        
        Args:
            view_name: Name of the view
            strategy_name: Name of the pretokenization strategy
            num_workers: Number of workers this was precomputed for
            worker_chunk_counts: List of chunk counts, one per worker
        """
        conn = self.connect()
        cur = conn.cursor()
        
        # Delete existing entries
        cur.execute("""
            DELETE FROM pretok_strategy_view_distributed_length 
            WHERE view_name = ? AND strategy_name = ? AND num_workers = ?
        """, (view_name, strategy_name, num_workers))
        
        # Insert new entries
        for worker_id, chunk_count in enumerate(worker_chunk_counts):
            cur.execute("""
                INSERT INTO pretok_strategy_view_distributed_length 
                (view_name, strategy_name, num_workers, worker_id, chunk_count)
                VALUES (?, ?, ?, ?, ?)
            """, (view_name, strategy_name, num_workers, worker_id, chunk_count))
        
        conn.commit()
        print(f"Stored precomputed lengths for {num_workers} workers in database")


    def get_worker_length_distributed(self, target_num_workers, target_worker_id, view_name, strategy_name):
        """
        Get the precomputed chunk count for a specific worker.
        If exact match exists, return it. If precomputed for a multiple, calculate from that.
        
        Args:
            target_num_workers: Number of workers in current training setup
            target_worker_id: ID of this worker (rank)
        
        Returns:
            int: Number of chunks this worker should process
        """

        
        conn = self.connect()
        cur = conn.cursor()
        
        # Try exact match first
        result = cur.execute("""
            SELECT chunk_count FROM pretok_strategy_view_distributed_length
            WHERE view_name = ? AND strategy_name = ? 
            AND num_workers = ? AND worker_id = ?
        """, (view_name, strategy_name, target_num_workers, target_worker_id)).fetchone()
        
        if result:
            return result[0]
        
        # No exact match - try to find a precomputed multiple
        available = cur.execute("""
            SELECT DISTINCT num_workers FROM pretok_strategy_view_distributed_length
            WHERE view_name = ? AND strategy_name = ?
            ORDER BY num_workers DESC
        """, (view_name, strategy_name)).fetchall()
        
        for (precomputed_workers,) in available:
            if precomputed_workers % target_num_workers == 0:

                
                # Fetch all worker counts from the precomputed configuration
                precomputed_counts = cur.execute("""
                    SELECT worker_id, chunk_count FROM pretok_strategy_view_distributed_length
                    WHERE view_name = ? AND strategy_name = ? AND num_workers = ?
                    ORDER BY worker_id
                """, (view_name, strategy_name, precomputed_workers)).fetchall()
                
                # Sum the relevant workers
                total = 0
                for worker_id, chunk_count in precomputed_counts:
                    if worker_id % target_num_workers == target_worker_id:
                        total += chunk_count
                
                return total
        
        raise ValueError(
            f"No precomputed length found for {target_num_workers} workers. "
            f"Available configurations: {[w[0] for w in available]}. "
            f"Please run precompute_chunks_distributed() with a number divisible by {target_num_workers}"
        )

        
class SubMdat:
    def __init__(self, parent_mdat: 'MatformerDataset', submdat_name: str) -> None:
        """Initialize SubMdat instance."""
        self.current_strategy = None
        self.default_wanted_from_dbs = None
        self.default_wanted_from_strategy = None
        self.storage_db = dict() #Dictionary containing the pointers to the databases with the actual storage
        self.default_map_size=1<<33 #WARNING
    def __str__(self):
        return self.submdat_name

    def common_initialization(self, parent_mdat: 'MatformerDataset', submdat_name: str) -> None:
        """Common initialization for both load and create."""
        if not isinstance(parent_mdat, MatformerDataset):
            raise Exception
        self.mdat = parent_mdat
        self.db = parent_mdat.db 
        self.mdat_path = parent_mdat.mdat_path
        self.readonly = self.mdat.readonly
        self.submdat_name = submdat_name
        self.submdat_path = os.path.join(self.mdat.dataset_path, submdat_name)
    
    @classmethod
    def load_submdat(cls, parent_mdat: 'MatformerDataset', submdat_name: str) -> 'SubMdat':
        """
        Load existing submdat.
        Args:
            parent_mdat: Parent MatformerDataset instance
            submdat_name: Name of submdat to load
        """
        instance = cls(parent_mdat, submdat_name)
        instance.common_initialization(parent_mdat, submdat_name)
        if instance.submdat_name not in instance.mdat.list_submdat():
            raise SubmDatNotFound   
        if not os.path.isdir(instance.submdat_path): 
            raise FileNotFoundError(f"{instance.submdat_name} is registered but not existing in {instance.mdat_path}.")
        instance.manifest = instance.db.get_manifest(submdat=submdat_name) 
        instance.submdat_id = instance.db.get_dsmap(key='name', value=submdat_name) 
        instance.len = instance.manifest['document_number'] 
        instance.set_default_wanted()
        # Initialize the storage DB; it could be considered to make this part 'lazy', i.e. load the DBs only when actually required (for example, avoid loading 'data' if only 'meta' is required), low priority
        instance._load_storage_db(_id=instance.submdat_id,load_all=True)
        return instance
    def _close_storage_dbs(self,_type='storage'):
        if _type=='storage':
            for name,db in self.storage_db.items():
                print("Closed db: ",name)
                db.close()
                del db
            del self.storage_db
            self.storage_db={}
        else:
            pass
    def _load_storage_db(self, _id=None, _type='storage', batch_size=50000, load_all=False, strategy=None, create=False, compression_level=None, disk_size=None, extra_data=None, db_type=None, map_size=None): 
        """
        An helper function that loads or creates a storage DB by getting its path from the meta-database and returns its pointer
        
        Args:
            _id: Database ID for loading existing DB (or submdat_id for pretok)
            _type: Type of DB ('storage' or 'pretok'/'strategy')
            load_all: Load all databases for the submdat
            strategy: Strategy name for pretok DBs
            create: If True, create new DB instead of loading
            compression_level: Compression level for new DB
            map_size: Map size for new DB (default: 1<<40)
            disk_size: Disk size for new DB
            extra_data: Extra data for new DB
            db_type: Type of DB to create ('data', 'meta', etc.)
            batch_size: Batch size for pretok DBs
        """
        def _create_lmdb(path, comp_level, map_size, b_size=None):
            params = {
                'compressed': comp_level > 0,
                'readonly': False if create else self.readonly,
                'compression_level': comp_level,
                'map_size': math.ceil(map_size)
            }
            if b_size:
                params['batch_size'] = b_size
            else:
                params['batch_size']= 50000
            return LMDBDataset(path, **params)
        
        def _get_map_size(db_info):
            if create:
                return self.default_map_size
            else:
                disk_size=self.db.get_storage_db_info(db_info['id'])['disk_size']
                if not isinstance(disk_size,int) or disk_size<=1:
                    return self.default_map_size
                else:
                    return disk_size*1.2 #Map size 20% larger than db size for safety
        
        if _type == 'storage':
            if create:
                path = os.path.join(self.submdat_path, db_type) + '.dat' if not strategy else os.path.join()
                db_pointer = _create_lmdb(path, compression_level,map_size=self.default_map_size,b_size=batch_size)
                db_id = self.db.add_database(_type='LMDB', compression_level=compression_level, map_size=self.default_map_size, disk_size=disk_size, extra_data=extra_data) 
                return db_pointer, db_id
            
            dbs_list = self.db.get_submdat_databases(self.submdat_id) if load_all else [self.db.get_submdat_databases(submdat_id=self.submdat_id, database_id=_id)]
            for dbs in dbs_list:
                db_type = 'data' if dbs['is_data'] else 'meta' if dbs['is_meta'] else (json.loads(dbs['is_extra']) if dbs['is_extra'] else 'extra')
                db_path = os.path.join(self.submdat_path, db_type) + '.dat'
                self.storage_db[db_type] = _create_lmdb(db_path, dbs['compression_level'], _get_map_size(dbs),b_size=batch_size)
        else: #type=='pretok'
            
            pretok_path = os.path.join(self.mdat.pretok_path, strategy, self.submdat_name)
            os.makedirs(pretok_path, exist_ok=True)
            self.pretok_db = {}
            
            db_names = json.loads(self.db.get_manifest(strategy=strategy)['required_databases']) if create else []
            db_configs = [(name, compression_level, map_size) for name in db_names] if create else [
                (
                    'tokens' if db['is_tokens']==1 else 'chunks' if db['is_chunks']==1 else (json.loads(db['is_extra']) if db['is_extra']!=0 else 'extra'),
                    db['compression_level'],
                    _get_map_size(db)
                )
                for db in self.db.get_strategy_dbs(strategy, _id)
            ]
            
            for db_name, comp_level, _map_size in db_configs:
                db_path = os.path.join(pretok_path, db_name + '.dat')
                db_pointer = _create_lmdb(db_path, comp_level, _map_size, batch_size or 50000)
                
                if create:
                    assert map_size is not None,"Map size should be specified when creating a new strategy's storage DB"
                    db_id = self.db.add_database(_type='LMDB', compression_level=comp_level, map_size=_map_size, disk_size=disk_size, extra_data=extra_data)
                    is_tokens = 1 if db_name == 'tokens' else 0
                    is_chunks = 1 if db_name == 'chunks' else 0
                    is_extra = json.dumps(db_name) if db_name not in ['tokens', 'chunks'] else None
                    self.db.add_strategy_db(database_id=db_id,strategy_name=strategy, submdat_id=self.submdat_id, datatype=None, is_tokens=is_tokens,is_chunks=is_chunks,is_extra=is_extra,is_complete=0)
                
                self.pretok_db[db_name] = db_pointer

        

    @classmethod
    def create_submdat(cls, parent_mdat: 'MatformerDataset', submdat_name: str, 
                      compression_levels: Dict[str, int], map_sizes: Dict[str, int], 
                      overwrite: bool = False, db_types: List[str] = ['meta', 'data'], 
                      data_type: str = 'text', data_key: str = 'text', 
                      modality: str = 'text', 
                      hybrid_data: List = []) -> 'SubMdat':
        """
        Create new submdat and load it.
        Args:
            parent_mdat: Parent MatformerDataset instance
            submdat_name: Name of new submdat
            compression_levels: Compression config per db type
            map_sizes: Map size config per db type
            overwrite: Overwrite if exists
            db_types: Database types to create (default: ['meta','data']
            data_type: Data type
            data_key: Data key
            hybrid_data: Hybrid data config
        """
        instance = cls(parent_mdat, submdat_name)
        instance.common_initialization(parent_mdat, submdat_name)
        if instance.readonly:
            raise MDatIsReadOnly       
        # Check if submdat already exists (that's redundant if submdat is called by a parent mdat)
        if instance.submdat_name in instance.mdat.list_submdat():
            if instance.mdat.get_submdat(submdat_name) is not None: #None is the placeholder for a submdat being created
                raise SubMdatAlreadyExists
        if os.path.isdir(instance.submdat_path):
            raise FileExistsError("SubMdat not registered in Mdat, but folder is present")
        instance.submdat_id = instance.db.add_submdat(submdat_name) # Add a placeholder submdat in the meta-database
                             
        # Create submdat directory
        os.makedirs(instance.submdat_path, exist_ok=True)        
        for db_type in db_types:        
            try:
                db_pointer, db_id = instance._load_storage_db(_type='storage',compression_level=compression_levels[db_type], disk_size=None, extra_data=None,db_type=db_type,create=True) 
                is_data = 1 if db_type == 'data' else 0 
                is_meta = 1 if db_type == 'meta' else 0 
                is_extra = json.dumps(db_type) if db_type not in ['data', 'meta'] else None 
                instance.db.link_submdat_database(instance.submdat_id, db_id, raw_data_bytes=0, is_data=is_data, is_meta=is_meta, is_extra=is_extra) 
                instance.storage_db[db_type] = db_pointer 
            except Exception as e:
                instance.db.remove_submdat(instance.submdat_id)
                raise e
                
        # A basic manifest, without data stats yet     
        manifest = { 
            "name": submdat_name,   
            "modality": json.dumps(modality), 
            "data_key": data_key,
            "document_number": 0, 
            "errors_counters": json.dumps({}), 
            "data_type": json.dumps(data_type) 
        }       
        instance.db.update_manifest(submdat=instance.submdat_id, data=manifest) 
        #Update the default view
        instance.mdat._update_default_view(instance.submdat_id) 

        # Now load the created submdat
        return cls.load_submdat(parent_mdat, submdat_name)
               
    def set_default_wanted(self, from_dbs: list = ['full'], from_strategy: list = [], raw: bool = False, add_special_tokens=True):
        """
        Set the default elements returned by the submat when called with __getitem__
        Args:
            from_dbs: the names of the db to be returned. "meta" is reserved for dictionaries, all the others are returned raw. The special key "full" will return all the databases.
            from_strategy: the names of the strategy's items to be returned, for example 'tokens' or 'chunks'. The special key "full" will return all the items produced by the pretokenization strategy.
        """
        self.default_wanted_from_dbs = from_dbs
        self.default_wanted_from_strategy = from_strategy
        self.default_raw = raw
        self.default_add_special_tokens = add_special_tokens
        
    def __getitem__(self, key):
        return self._compose_return(key=key, wanted_from_dbs=self.default_wanted_from_dbs, wanted_from_strategy=self.default_wanted_from_strategy, raw=self.default_raw)

    def _compose_return(self, key: int, wanted_from_dbs: list, wanted_from_strategy: list, raw: bool = False, add_special_tokens: bool = True, strategy_name=None, on_the_fly_mode=True, max_seq_len=None, decode_strings=True):
        composed = {'submdat_name': self.submdat_name, 'key': key}
        if wanted_from_dbs == 'full':
            composed.update(orjson.loads(self.storage_db['meta'][key])) 
            composed.update({'data': self.storage_db['data'][key]}) 
        if 'data' in wanted_from_dbs:
            if decode_strings:
                text = self.storage_db['data'][key].decode('utf-8') 
            else:
                text = self.storage_db['data'][key] 
            if not raw:
                composed.update({'data': text})
            else:
                return(text)
        if 'meta' in wanted_from_dbs:
            composed.update({'meta':orjson.loads(self.storage_db['meta'][key])}) 
        if self.current_strategy is not None:
            if wanted_from_strategy is not None:
               composed.update(self.current_strategy(key=key, cache_dict=self.pretok_db, max_seq_len=max_seq_len, wanted_from_strategy=wanted_from_strategy, add_special_tokens=add_special_tokens))
        return composed

    def get_generator(self, progress_bar=False, wanted_from_dbs='full', wanted_from_strategy=None, raw=False, randomize=False, seed=27):
        """
        Returns a generator over submdat elements. 
        Args:
            progress_bar: enable/disable a tqdm progress bar
            from_dbs: the names of the db to be returned. "meta" is reserved for dictionaries, all the others are returned raw. The special key "full" will return all the databases.
            from_strategy: the names of the strategy's items to be returned, for example 'tokens' or 'chunks'. The special key "full" will return all the items produced by the pretokenization strategy.
        """
        if randomize:
            permutator = RandomPermutator(max_len=self.len, seed=seed)
        if progress_bar:
            from tqdm import tqdm
            for key in tqdm(range(self.len)):
                if randomize:
                    key = permutator(key)
                yield self._compose_return(key=key, wanted_from_dbs=wanted_from_dbs, wanted_from_strategy=wanted_from_strategy, raw=raw)
        else:
            for key in (range(self.len)):
                if randomize:
                    key = permutator(key)
                yield self._compose_return(key=key, wanted_from_dbs=wanted_from_dbs, wanted_from_strategy=wanted_from_strategy, raw=raw)                           

    def __len__(self):
        return self.len
    
    def __str__(self):
        print(f"Submdat: {self.manifest['name']}, Documents: {self.manifest['document_number']}") 
        if self.current_strategy is not None:
            strategy_stats = self.db.get_strategy_submdats(self.current_strategy.strategy_name, self.submdat_id) 
            print(f"Tokens: {strategy_stats['total_tokens']} Chunks: {strategy_stats['total_chunks']}") 

    def get_manifest(self):  
        return self.db.get_manifest(submdat=self.submdat_name)
    
    def set_strategy(self, strategy_name: str, add: bool = False,compression_level: int = 0, map_size: int = 1<<41, batch_size: int = 50000):
        """
        Set the Pretokenization Strategy to be used in the Submdat
        """
        #1. Check if the requested strategy is associated with the submdat
        if len(self.db.get_strategy_submdats(strategy_name=strategy_name,submdat_id=self.submdat_id))==0:
            if add:
                """
                Begin of the process of adding a strategy.
                It's important to call add_strategy_end after the Pretokenization process to be sure databases are closed, stats and manifest saved
                """
                print(f"Adding {strategy_name} to {self.submdat_name}")
                self.db.add_strategy_submdat(strategy_name, self.submdat_id)
                pretok_path = os.path.join(self.mdat.pretok_path, strategy_name, self.submdat_name)
                os.makedirs(pretok_path, exist_ok=True)
                self.current_strategy=PretokenizationStrategy(db=self.db,strategy_name=strategy_name,pretok_path=self.mdat.pretok_path,functions_path=self.mdat.functions_path)
                self._load_storage_db(_id=self.submdat_id,_type='strategy',strategy=strategy_name,create=True,compression_level=compression_level,map_size=map_size,batch_size=batch_size)
            else:
                raise SubMdatMissesStrategy
        else:
            #2. Load the strategy
            self.current_strategy = self.mdat.current_strategy
            #3. Load the strategy's storage db(s)
            self._load_storage_db(_id=self.submdat_id,_type='strategy',strategy=strategy_name,create=False)
     
    def add_strategy_end(self, strategy_name, stats: dict[str, Any]):
        if self.readonly:
            raise MdatIsReadOnly
        for db in self.pretok_db.values():
            db.close()
        
        self.db.update_strategy_submdat(strategy_name, self.submdat_id, { 
            'finalized_at': datetime.utcnow().isoformat(), 
            'total_tokens': stats.get('total_tokens', 0), 
            'max_tokens_per_doc': stats.get('max_tokens_per_doc', 0), 
            'total_chunks': stats.get('total_chunks', 0), 
            'max_chunks_per_doc': stats.get('max_chunks_per_doc', 0), 
            'processed_docs': stats.get('processed_docs', 0) 
        }) 
        
        for i, precomp_len in enumerate(stats.get('precomputed_lengths', [])): 
            if precomp_len > 0: 
                self.db.set_strategy_submdat_precomp(strategy_name, self.submdat_id, i, precomp_len) 
                
        
        # Update database's disk size in database_ref
        pretok_path = os.path.join(self.mdat.pretok_path, strategy_name, self.submdat_name)
        strategy_dbs = self.db.get_strategy_dbs(strategy_name, self.submdat_id)
        print(pretok_path)
        print(strategy_dbs)
        for db in strategy_dbs:
            if db['is_tokens'] == 1:
                db_name = 'tokens'
            elif db['is_chunks'] == 1:
                db_name = 'chunks'
            else:
                db_name = json.loads(db['is_extra']) if db['is_extra'] else 'extra'
            
            db_file = os.path.join(pretok_path, f"{db_name}.dat")
            if os.path.exists(db_file):
                
                disk_size = os.path.getsize(db_file)
                self.db.update_strategy_db(database_id=db['db_id'], data={'disk_size': disk_size})
            else:
                print(f"Non trovo {db_file}")
                
    def get_current_strategy(self):
        return self.current_strategy

    def pretokenize_submdat(self, strategy_name,progress_bar=True, compression_level=0, chunking_strict_checks=False, parallel=True, num_processes=None, batch_size=256, map_size=1<<40,disk_size_multiplier=4):

        max_multiplier = 100     #Precompute 100 times the base sequence length     
        # 1. Start adding the strategy to the submdat
        # Inferred map_size = 4 times the data db size
        data_disk_size=self.db.get_submdat_info(submdat_id=self.submdat_id)['raw_data_bytes']
        inferred_map_size=data_disk_size*disk_size_multiplier
        print(f"Original data size is f{data_disk_size}. A multiplier of {disk_size_multiplier} is set to initialize the pretokenization storage dataset(s). If this value ({inferred_map_size}) is too large, you'll get memory errors, if too small, pretokenization will not fit the database. Adjust this value in case of problems.")
        self.set_strategy(strategy_name=strategy_name, compression_level=compression_level, add=True, map_size=inferred_map_size)
        strategy=self.current_strategy
        # Initialize stats
        stats = {
            'total_tokens': 0,
            'max_tokens_per_doc': 0,
            'total_chunks': 0,
            'max_chunks_per_doc': 0,
            'processed_docs': 0,
            'precomputed_lengths':[0] * (int(max_multiplier)+1),
        }
        # 4. Initialize the splitter
        # 5. What does the splitter wants from Submdats'databases? [Default: raw_data]
        # 6. Initialize the generator
        progress_bar_generator=progress_bar if not parallel else False
        generator = self.get_generator(
            progress_bar=progress_bar_generator,
            wanted_from_dbs=strategy.wants_from_db,
            raw=strategy.wants_raw
        )
        
        """
        Writing strategies:
            1) For "tokens", we can use the following datatypes:
                    - int8
                    - int16
                    - int32
                    - Experimentally, we tried int24 but still we haven't reached satisfying results
                    - float (to be improved, for patches, especially multimodal)
               The datatype per token it's chosen by the strategy as the best to fit max_vocab_size
            2) For "chunks", we expect from the splitter an array of tuples like [(0,5),(5,9),(9,13),(13,27)...]
               Each tuple represent the chunks of data (token, most commonly, but also raw data) present in each split
               These data are first converted into lengths, for example (5,4,4,14) and stored as a numpy array
               The datatype for the array is the minimum enough to handle max_sequence_length (int8,int16,int32)
            3) "Extra" is freely chosen by the splitter. Thus, if present, we will just store extra_db[key]=extra as it is.
               Byte objects are required, if a string is returned, it is converted to utf-8, if other stuff, we try to serialize with orjson or
               raise exception (is responsibility of the splitter to return the correct datatype)
        """
        if not parallel:
            def process_batch(batch, stats):
                keys, docs = zip(*batch)
                batch_results = strategy.pretokenize_batch(list(docs))

                for k, split_result in zip(keys, batch_results):
                    for db_name in strategy.returns:
                        if db_name in split_result:
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
                                stats['precomputed_lengths'][0] = 0
                                for j in range(1, max_multiplier):
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

            # remaining docs
            if batch:
                process_batch(batch, stats)
        else:
            # PARALLEL VERSION
            if num_processes is None:
                num_processes = mp.cpu_count()
            self.current_strategy._initialize_components()
            def process_batch_and_write(batch_data, stats):
                """Process a batch and write results to databases."""
                sub_batches = [batch_data[i:i+batch_size] for i in range(0, len(batch_data), batch_size)]
                
                results = pool.starmap(process_documents_batch, [
                    (sub_batch,self.current_strategy.manifest, self.mdat.pretok_path, self.mdat.functions_path, strategy_name, 
                     chunking_strict_checks, strategy.chunk_size, strategy.returns, max_multiplier)
                    for sub_batch in sub_batches
                ])
                
                # Collect results by database and merge stats
                db_batches = {db_name: [] for db_name in strategy.returns}
                
                for worker_db_data, worker_stats in results:
                    stats['total_tokens'] += worker_stats['total_tokens']
                    stats['total_chunks'] += worker_stats['total_chunks']
                    stats['processed_docs'] += worker_stats['processed_docs']
                    stats['max_tokens_per_doc'] = max(stats['max_tokens_per_doc'], worker_stats['max_tokens_per_doc'])
                    stats['max_chunks_per_doc'] = max(stats['max_chunks_per_doc'], worker_stats['max_chunks_per_doc'])
                    stats['precomputed_lengths'] = (np.array(stats['precomputed_lengths']) + np.array(worker_stats['precomputed_lengths'])).tolist()
                    # Collect database writes
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
                
                # Process remaining items
                if batch:
                    stats = process_batch_and_write(batch, stats)
                    pbar.update(len(batch))
                pbar.close()
        # E. Close the DB and update the manifest
        self.add_strategy_end(strategy_name=strategy_name, stats=stats)
            
    def convert_to_submdat(self, dataset_type, dataset_path, dataset_args={}, data_key='text', modality='text', logger=False, progress_bar=True, do_transform=False, do_filtering=False,dataset_config=''): 
        """
        A function to populate the submdat's databases with data coming from other datasets' formats
        """
        if self.readonly:
            raise MdatIsReadOnly   
        class PrintLogger:
            def warning(self, message):
                print("Warning: ", message)
            def error(self, message):
                print("Error: ", message)
            def info(self, message):
                print("Info: ", message)                
        logger_fn = logger if logger else PrintLogger()
        
        if not hasattr(self, 'storage_db'):
            logger_fn.error(f"Databases are not loaded for submat {self.submdat_name}") 
            return None
        
        #If there is orjson, better
        try:
            import orjson
        except:
            orjson = False
        # Instantiating the correct generator for the dataset_type
        """
        Example dataset_args: json_batch_read=False, files_recurse_in_folder=True, 
                              files_metadata_type='multiple_json', csv_has_header=False, csv_data_field=0,
                              hf_split='train', hf_subdataset=None,
        """
        inferred_map_size=None
        if dataset_type == 'jsonl':
            #from datasets_iterators import JSONIterator
            generator_fn = JSONIterator(json_path=dataset_path, dataset_args=dataset_args, progress_bar=progress_bar, logger=logger_fn)
            #generator_fn = JSONLIteratorFast(json_path=dataset_path, dataset_args=dataset_args, progress_bar=progress_bar, logger=logger_fn)
            file_size=os.path.getsize(dataset_path)
            inferred_map_size=max(1<<30,file_size*5)
        elif dataset_type == 'lmdb':
            #from datasets_iterators import LMDBIterator
            generator_fn = LMDBIterator(dataset_path, dataset_args, data_key, progress_bar=progress_bar, logger=logger_fn)
        elif dataset_type == 'hf':
            generator_fn= HuggingFaceIterator(dataset_path=dataset_path,logger=logger_fn,dataset_args=dataset_args,dataset_config=dataset_config,dataset_split='train',progress_bar=True,data_key=None)
        elif dataset_type == 'sqlite':
            return
        elif dataset_type == 'atlas':
            generator_fn = AtlasIterator(path=dataset_path, dataset_args=dataset_args, progress_bar=progress_bar, logger=logger_fn)
        elif dataset_type == 'csv':
            return
        elif dataset_type == 'files':
            return
        else:
            error = f"Unsupported dataset type: {dataset_type}"
            logger_fn.error(error)
            return error
        
        print(f"Inferred size of the input file: {inferred_map_size}. Reloading the DB with the appropriate size. ")
        if inferred_map_size:
            self.default_map_size=inferred_map_size
            self._close_storage_dbs(_type='storage')
            self._load_storage_db(_id=self.submdat_id,load_all=True)
        # Initialize stats (error stats and raw size stats)
        errors_counters = dict()  
        errors_counters['hasDataError'] = 0 
        errors_counters['generatorReturnedNone'] = 0
        errors_counters['missingDataKey'] = 0
        errors_counters['data<=10']=0
        n_filtered = 0
        raw_data_bytes = 0
        raw_meta_bytes = 0
        
        # Iterating over the generator
        for i, item in enumerate(generator_fn):
            if item is None:
                errors_counters['generatorReturnedNone'] += 1
                continue
            # A transformer function can be specified by the user (useful, not implemented yet)
            if do_transform:
                # item = transformer_function(item)
                pass
            else:
                if data_key not in item:
                    if 'raw_content' in item:
                         data=item['raw_content'] # Fix for redpajama TODO remove it
                    else:
                       warning = f"Data key '{data_key}' not found in item {i}. Item has keys {item.keys()}"
                       logger_fn.warning(warning)
                       errors_counters['missingDataKey'] += 1
                       continue
                else:
                    data = item[data_key]

            if isinstance(data, str):
                data = data.encode('utf-8')
            if not isinstance(data, bytes):
                logger_fn.warning(f"Data is of types {type(data)} but it should be either string or bytes")
            if len(data)<=10:
                logger_fn.warning(f"Data is smaller than 10 bytes. Skipping")
                errors_counters['data<=10']+=1
                continue
            try: # Fix for Redpajama, TODO Remove it 
               del item[data_key]
            except:
               del item['raw_content']

            
            # Data can be passed through filters for selection (ex. language identification, quality metrics...)
            filtered = False
            if do_filtering:
                 pass
                 logger_fn.warning("Filter functions not implemented")
            
            if filtered:
                continue
            
            returned_error = self.storage_db['data'].write(data, key=i) 
            if returned_error is not None:
                errors_counters['hasDataError'] += 1    
            try:
                if orjson:
                    try:
                        serialized = orjson.dumps(item)
                    except:
                        serialized = json.dumps(item).encode()  
                else:
                    serialized = json.dumps(item).encode()
            except Exception as e:
                logger_fn.error(f'Serialization of item id {i} failed: {e}')
                continue
                
            error = self.storage_db['meta'].write(serialized, key=i) 
            if error is not None:
                errors_counters['hasDataError'] += 1 
            # Computing the size of the data just inserted
            raw_data_bytes += len(data)
            raw_meta_bytes += len(serialized)
            
        # Close the databases 
        documents_number = len(self.storage_db['data']) 
        self.storage_db['data'].close()
        self.storage_db['meta'].close()
        db_data_bytes=os.path.getsize(os.path.join(self.submdat_path,'data.dat'))
        db_meta_bytes=os.path.getsize(os.path.join(self.submdat_path,'meta.dat'))
        # Computing size on disk for the new submdat
        db_disk_bytes = 0
        for root, dirs, files in os.walk(self.submdat_path):
            for file in files:
                db_disk_bytes += os.path.getsize(os.path.join(root, file))
        # Update the db for data and meta databases
        for dbs in self.db.get_submdat_databases(self.submdat_id): 
            if dbs['is_data']:
                self.db.update_submdat_database(submdat_id=self.submdat_id, database_id=dbs['id'], data={'raw_data_bytes': raw_data_bytes, 'disk_size': db_data_bytes}) 
            elif dbs['is_meta']:
                self.db.update_submdat_database(submdat_id=self.submdat_id, database_id=dbs['id'], data={'raw_data_bytes': raw_meta_bytes, 'disk_size': db_meta_bytes}) 
            else:
                raise NotImplementedError       
        
        partial_manifest = { 
            'document_number': documents_number, 
            'errors_counters': json.dumps(errors_counters) 
        } 
        self.db.update_manifest(submdat=self.submdat_id, data=partial_manifest) 
        # Updating the default view with the new documents
        self.mdat._update_default_view(self.submdat_id) 
        return partial_manifest
       
def get_datatype_for_numpy(datatype):
        if isinstance(datatype,int):
            datatype=str(datatype)
        if datatype=='8' or datatype=='uint8':
            return np.uint8
        if datatype=='16' or datatype=='uint16':
            return np.uint16
        if datatype=='32' or datatype=='uint32':
            return np.uint32
        if datatype=='float16':
            return np.float16
        if datatype=='float32':
            return np.float32
        if datatype=='float64':
            return np.float64
        else:
            raise SyntaxError
def sanify_name(_input):
    return ''.join(c if c.isalnum() or c in '_-' else '_' for c in _input)
    
class RandomPermutator:
    def __init__(self,max_len,seed=27):
        self.max_len=max_len
        random.seed(seed)
        self.a=self.create_coprime()  
        self.b=random.randint(0,max_len-1) 
        
    def create_coprime(self):
        while True:
            a=random.randint(1,self.max_len-1)
            if math.gcd(a,self.max_len) == 1: 
                return a
                
    def __call__(self,i):
        if i<=self.max_len:
            return (self.a*i+self.b)%self.max_len
        else:
            raise Exception("Index out of range")       
def process_documents_batch(sub_batch,strategy_manifest, pretoken_path, functions_path, strategy_name,
                           chunking_strict_checks, chunk_size, strategy_returns, max_multiplier):
    """
    Worker function that processes documents and returns organized data by database type.
    """
    strategy = PretokenizationStrategy.from_dict(db=None,strategy_name=strategy_name, pretok_path=pretoken_path, functions_path=functions_path,strategy_dict=strategy_manifest)
    
    # Organize results by database type
    db_data = {db_name: [] for db_name in strategy_returns}
    
    # Local stats for this worker
    local_stats = {
        'total_tokens': 0,
        'max_tokens_per_doc': 0,
        'total_chunks': 0,
        'max_chunks_per_doc': 0,
        'processed_docs': 0,
        'precomputed_lengths':[0] * (int(max_multiplier)+1),
        'base_chunk_size':strategy.chunk_size
    }
    
    # Split sub_batch into keys and documents
    keys = [key for key, _ in sub_batch]
    docs = [doc for _, doc in sub_batch]

    # Run batched pretokenization
    split_results = strategy.pretokenize_batch(docs)

    # Process results in sync with keys
    for key, split_result in zip(keys, split_results):
        local_stats['processed_docs'] += 1

        for db_name in strategy_returns:
            if db_name not in split_result:
                continue

            if db_name == 'tokens':
                token_bytes = strategy.prepare_tokens_for_storage(split_result['tokens'])
                token_count = len(split_result['tokens'])
                local_stats['total_tokens'] += token_count
                local_stats['max_tokens_per_doc'] = max(local_stats['max_tokens_per_doc'], token_count)
                db_data[db_name].append((key, token_bytes))

            elif db_name == 'chunks':
                tokens_length = len(split_result.get('tokens', [])) if chunking_strict_checks else None
                chunk_bytes = strategy.prepare_chunks_for_storage(
                    split_result['chunks'],
                    max_tokens_per_chunk=chunk_size,
                    tokens_length=tokens_length,
                    strict_checks=chunking_strict_checks,
                )
                chunk_count = len(split_result['chunks'])
                local_stats['total_chunks'] += chunk_count
                local_stats['max_chunks_per_doc'] = max(local_stats['max_chunks_per_doc'], chunk_count)
                local_stats['precomputed_lengths'][0] = 0
                for j in range(1, max_multiplier):
                    local_stats['precomputed_lengths'][j] += math.ceil(chunk_count / j)
                db_data[db_name].append((key, chunk_bytes))

            else:
                obj_bytes = strategy.prepare_extra_data_for_storage(split_result[db_name])
                db_data[db_name].append((key, obj_bytes))

    return db_data, local_stats

# Exceptions
if True:
    class NameAlreadyUsed(Exception):
        pass

    class StrategyNotRegistered(Exception):
        pass

    class MDatNotShuffled(Exception):
        pass

    class MDatNotFound(Exception):
        pass

    class MDatAlreadyPresent(Exception):
        pass

    class MDatAlreadyShuffled(Exception):
        pass

    class TooManySubmDats(Exception):
        pass

    class MdatIsWriteOnly(Exception):
        pass

    class SubmDatNotFound(Exception):
        pass

# LMDBDataset's stuff (imports are here because probably it will be moved to a different file):
import os
import lmdb
import orjson
import pickle

USE_ZLIB = os.environ.get('USE_ZLIB', 'false').lower() == 'true'

if USE_ZLIB:
    import zlib
    compress_func = lambda data, level: zlib.compress(data, level)
    decompress_func = lambda data: zlib.decompress(data)
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
    def __init__(self, path, readonly=True, lock=False, compressed=False, compression_level=0,map_size=1<<44,batch_size=50000):
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
        self.map_size=map_size
        self.path=path
        self.readonly=readonly
        self.compressed=compressed
        self.compression_level=compression_level
        self.batch_size = batch_size
        self.batch_buffer = []
        self.compression_attempts=100 #Trying to avoid compression bugs
        self.compression_backend = COMPRESSION_BACKEND
        with self.env.begin(write=False) as txn:
            try:
                self.length = int(txn.get(b"__len__").decode())
            except:
                self.length=0
    def set_batch_size(self,batch_size):
        self.batch_size=batch_size
    def __len__(self):
        return self.length
    def start_transaction(self):
        self.txn=self.env.begin(write=True)
    def close_transaction(self):
        pass

            
    def write(self, obj, key=None, batched=False):
        """
        "Obj" can be:
            * A byte object, directly inserted
            * Other types, orjson serialized
        If batched==True:
            * A list of data is expected
        Return "current_index" if data is correctly added, None if error happened
        """
        
        if key==None and batched==False:
            key=self.length
        if key is not None and batched==True:
            print("LMDBDataset Warning: You should not provide a key if the input data is batched! Key value is ignored.")
        
        batch = obj if isinstance(obj, list) else [obj]
        if len(batch) > self.batch_size:
            raise ValueError(f"Batch size {len(batch)} exceeds maximum {self.batch_size}")
        
        for data in batch:
            if not isinstance(data, bytes):
                data = orjson.dumps(data)  # Serialize non-bytes objects, for example python dictionaries
            if self.compressed:
                data = compress_func(data, self.compression_level)
            self.batch_buffer.append((str(self.length).encode(), data))
            self.length += 1
            
            if len(self.batch_buffer) >= self.batch_size:
                self._flush_batch()
    def write_batch_with_keys(self, key_data_pairs):
            """
            Write multiple (key, data) pairs using the existing batch buffer mechanism.
            This leverages the existing batching and auto-flushing logic.
            
            Args:
                key_data_pairs: list of (key, data) tuples where key is the document index
                               and data is the object to store
            """
            if not key_data_pairs:
                return
            
            for key, data in key_data_pairs:
                if not isinstance(data, bytes):
                    data = orjson.dumps(data)
                if self.compressed:
                    data = compress_func(data, self.compression_level)
                
                # Add to batch buffer - this will auto-flush when batch_size is reached
                self.batch_buffer.append((str(key).encode(), data))
                
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
            if self.readonly == False:
                with self.env.begin(write=True) as txn:
                    txn.put(b"__len__", str(self.length).encode())  # Synchronizing correct length
                self.env.sync()
            self.env.close()
                
            
        
    def __getitem__(self, key):
        key=str(key).encode()
        with self.env.begin(write=False) as txn:
            data = txn.get(key)
            if data is None:
                raise IndexError(f"Index {key} not found in LMDB dataset.")
            if self.compressed:
                try:
                    #print("Attenzione! AL momento sqlite segna compresso ma dati non compressi. Facile da risolvere")
                    return decompress_func(data)
                except:
                    safe_path = ''.join(c if c.isalnum() or c in '_-' else '_' for c in self.path)
                    print(f"LMDBDataset {self.path} WARNING: error loading data at index {key}.")
                    print(f"It seems there is a bug with {self.compression_backend}.")
                    print(f"First, let's try to load the data again {self.compression_attempts} times.")
                    for i in range(1,self.compression_attempts):
                        try:
                            x=decompress_func(data)
                            print(f"It worked at attemp: {i}")
                            with open(f"compression_error_logs_{safe_path}.txt","a") as l:
                                l.write(f"{self.compression_backend} error at {i}. Recovered after {i}/{self.compression_attempts} attempts.\n")
                            return x
                        except:
                            pass
                    print(f"It didn't worked after {self.compression_attempts}. Returning a dummy structure to avoid breaking training. Event logged in ./compression_error_logs_{safe_path}.txt")
                    with open(f"compression_error_logs_{safe_path}.txt","a") as l:
                        l.write(f"{self.compression_backend} error at {i}. Not recovered after {self.compression_attempts} attempts.\n")
                    return None
            else:
                return data


import os
from tqdm import tqdm
import orjson
import json
from contextlib import nullcontext

def ToBeFixed():
    return nullcontext()
def HuggingFaceIterator(dataset_path,logger,dataset_args,dataset_config='',dataset_split='train',progress_bar=True,data_key=None):
     from datasets import load_dataset
     ds=load_dataset(dataset_path,dataset_config)[dataset_split]
     for item in tqdm(ds):
       yield(item)
def JSONIterator(json_path, logger, dataset_args={}, progress_bar=True, batch_size_bytes=16 * 1024 * 1024):
    file_size = os.path.getsize(json_path)
    
    ProgressBar = tqdm if progress_bar else lambda *args, **kwargs: ToBeFixed()
    
    with open(json_path, 'rb', buffering=batch_size_bytes) as f:
        with ProgressBar(total=file_size, unit='B', unit_scale=True, desc="Processing JSONL file...") as pbar:
            i = 0
            while True:
                batch_of_lines = f.readlines(batch_size_bytes)
                if not batch_of_lines:
                    break

                for row_bytes in batch_of_lines:
                    try:
                        data = orjson.loads(row_bytes)
                    except Exception as e:
                        try:
                            logger.warning(f"Row: {i} orjson failed: {e}, falling back to json.")
                            data = json.loads(row_bytes)
                        except Exception as e:
                            logger.error(f"Row {i} Failed to parse JSON row: {e}")
                            data = None

                    if data is not None:
                        pbar.update(len(row_bytes))
                        yield data
                        
                    i += 1

def AtlasIterator(path,logger,dataset_args={},progress_bar=True):
   from torch_atlas_ds import AtlasDataset
   db = AtlasDataset(path) 
   ProgressBar = tqdm if progress_bar else lambda *args, **kwargs: ToBeFixed() 
   for i in ProgressBar(range(len(db))):
       try:
           yield db[i]
       except Exception as e:
           logger.error(f"Error reading Atlas item {i}: {e}")
           yield None

def LMDBIterator(lmdb_path,logger,dataset_args={},progress_bar=True):
   db = LMDBDataset(lmdb_path) 
   ProgressBar = tqdm if progress_bar else lambda *args, **kwargs: ToBeFixed() 
   for i in ProgressBar(range(len(db))):
       try:
           yield db[i]
       except Exception as e:
           logger.error(f"Error reading LMDB item {i}: {e}")
           yield None

import random

import sys

class PretokenizationStrategy:
    def __init__(self, db: 'DatabaseManager', pretok_path: str, functions_path: str, strategy_name: str): 
        """
        Initialize a PretokenizationStrategy by loading from existing configuration.
        A strategy can be used:
            1) To pretokenize a submdat
            2) To retrieve the chunked/pretokenized data from the pretokenization databases
            3) To perform an on_the_fly tokenization/chunking
        """
        self.db = db 
        self.strategy_name = strategy_name
        self.mdat_pretok_path = pretok_path
        self.functions_path = functions_path
        self.on_the_fly_warning = False
        self.on_the_fly_mode = True
        
        # Load strategy configuration
        self._load_configuration()
        self.initialized = False

    @classmethod
    def from_dict(cls, db: 'DatabaseManager', pretok_path: str, functions_path: str, strategy_name: str, strategy_dict: Dict[str, Any]=None) -> 'PretokenizationStrategy': 
        """
        Create a new PretokenizationStrategy from a dictionary configuration.
        """
        instance = cls.__new__(cls)
        instance.db = db 
        instance.strategy_name = strategy_name
        instance.mdat_pretok_path = pretok_path
        instance.functions_path = functions_path
        instance.on_the_fly_warning = False
        instance.on_the_fly_mode = True
        # This part should be more elegant
        if strategy_dict is not None:
            instance._create_from_dict(strategy_dict) #Used for creation of a strategy and for parallel processing
        else:
            instance._load_configuration() # Normal use. Config taken from the database
        instance.initialized = False
        return instance
    def _create_from_dict(self, strategy_dict: Dict[str, Any]):
        self.manifest = strategy_dict
        """Create configuration from dictionary."""
        required_keys = ['strategy_name', 'tokenizer_type', 'tokenizer_name',
                         'splitter_class', 'splitter_init', 'modality',
                         'chunk_size', 'wants_from_db', 'returns', 'required_databases']
        self.initialized=False
        for k in required_keys:
            if k not in strategy_dict.keys():
                raise MissingStrategyKey(f"Missing required key: {k}")

        self.strategy_name = strategy_dict['strategy_name']
        self.tokenizer_type = strategy_dict['tokenizer_type']
        self.tokenizer_name = strategy_dict['tokenizer_name']
        self.bos_token_id = strategy_dict.get('bos_token_id')
        self.eos_token_id = strategy_dict.get('eos_token_id')
        self.mask_token_id = strategy_dict.get('mask_token_id')
        self.tokenizer_args = strategy_dict.get('tokenizer_args', {})
        self.splitter_class = strategy_dict['splitter_class']
        self.splitter_init = strategy_dict['splitter_init']
        self.splitter_arguments = strategy_dict.get('splitter_arguments')
        self.chunk_size = strategy_dict['chunk_size']
        self.modality = strategy_dict['modality']
        self.wants_from_db = strategy_dict['wants_from_db']
        self.wants_raw = strategy_dict.get('wants_raw', False)
        self.returns = strategy_dict['returns']
        self.supports_batch = strategy_dict.get('supports_batch', False)
        self.required_databases = strategy_dict['required_databases']

        def dict_or_tokenizer(param):
            """Load parameter from dict if present, otherwise from tokenizer after initialization."""
            if param in strategy_dict:  # Respect explicit None in config
                if strategy_dict[param] is None:
                    print(f"Parameter {param} is set as None in the config. Check if this is intended.")
                setattr(self, param, strategy_dict[param])
            else:
                print(f"You haven't specified {param} in the config. Trying to load from tokenizer...")
                self._initialize_components()
                val = getattr(self.tokenizer, param, None)
                setattr(self, param, val)
                strategy_dict[param] = val


        # Load special tokens IDs Special tokens are not saved into the chunks (to easily allow chunks merging, but added on the fly)
        dict_or_tokenizer('bos_token_id')
        dict_or_tokenizer('eos_token_id')
        dict_or_tokenizer('mask_token_id')

        # Determine tokens datatype based on vocabulary size
        dict_or_tokenizer('vocab_size')
        if self.vocab_size <= 255:
            self.tokens_datatype = 'uint8'
        elif self.vocab_size <= 65535:
            self.tokens_datatype = 'uint16'
        else:
            self.tokens_datatype = 'uint32'
        strategy_dict['tokens_datatype'] = self.tokens_datatype

        # Determine chunks datatype based on chunk size
        if self.chunk_size <= 255:
            self.chunks_datatype = 'uint8'
        elif self.chunk_size <= 65535:
            self.chunks_datatype = 'uint16'
        else:
            self.chunks_datatype = 'uint32'
        strategy_dict['chunks_datatype'] = self.chunks_datatype

        self.manifest = strategy_dict

    def _load_configuration(self):
        """Load strategy configuration from JSON fields in DB."""
        manifest = self.db.get_manifest(strategy=self.strategy_name)
        if not manifest:
            raise StrategyNotFound(f"Strategy configuration not found: {self.strategy_name}")

        json_defaults = {
            'tokenizer_args': {},
            'splitter_init': {},
            'splitter_arguments': None,
            'modality': "text",
            'wants_from_db': [],
            'returns': [],
            'required_databases': []
        }

        for field, default in json_defaults.items():
            value = manifest.get(field)
            if value is None or not isinstance(value, str):
                manifest[field] = default
            else:
                try:
                    manifest[field] = json.loads(value)
                except json.JSONDecodeError:
                    manifest[field] = default

        self._create_from_dict(manifest)


    def _initialize_components(self):
        """Initialize tokenizer and splitter components."""
        sys.path.append('../') #DIRTY stuff to load matformertokenizer
        if self.initialized==False:
            from matformer.matformer_tokenizers import MatformerTokenizer
            self.tokenizer = MatformerTokenizer(
                tokenizer_type=self.tokenizer_type, 
                tokenizer_name=self.tokenizer_name, 
                tokenizer_args=self.tokenizer_args
            )

            def tokenizer_if_minus_one(self, parameter, tokenizer):
                """If parameter is -1, try to fetch it from tokenizer; otherwise leave unchanged."""
                if getattr(self, parameter, None) == -1:
                    try:
                        if parameter=='vocab_size':
                            setattr(self,parameter,len(tokenizer)) #More reliable, includes special tokens
                        else:
                            setattr(self, parameter, getattr(tokenizer, parameter, None))
                    except Exception:
                        print(f"Cannot find parameter {parameter} in the tokenizer. It stays to None")

            for p in ('bos_token_id', 'eos_token_id', 'mask_token_id', 'vocab_size'):
                tokenizer_if_minus_one(self, p, self.tokenizer)

            """Initialize the splitter class with dynamic import capability."""
            splitter_cls = self._find_splitter_class(self.splitter_class)

            # Prepare initialization arguments
            init_args = self.splitter_init.copy() if self.splitter_init else {}
            init_args['chunk_size'] = self.chunk_size - 2 #Reduce the max sequence length by two to allow special tokens
            init_args['tokenizer'] = self.tokenizer

            # Initialize splitter
            self.splitter = splitter_cls(**init_args)
            self.initialized = True
            self.save() #Saving in order to update token dtype and chunks dtype

        

    def _find_splitter_class(self, class_name: str):
        """Find splitter class in globals or import from functions directory."""
        if class_name in globals():  # First check if it exists in current globals
            return globals()[class_name]
        
        for filename in os.listdir(self.functions_path):  # Try to find the class in any Python file in the functions directory
            if not filename.endswith(".py") or filename.startswith("__"):
                continue
            file_path = os.path.join(self.functions_path, filename)
            try:
                spec = importlib.util.spec_from_file_location(class_name, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                if hasattr(module, class_name):
                    cls = getattr(module, class_name)
                    if cls is None:
                        raise ValueError(f"Module {filename} defines {class_name} = None")
                    return cls
            except Exception as e:
                print(f"Failed to import {filename}: {e}")
                traceback.print_exc()
                continue    


    def save(self):
        """Insert or update strategy configuration in the database."""
        if self.db is not None:
            manifest = self.manifest.copy()
            manifest['strategy_name'] = self.strategy_name

            for k, v in manifest.items():
                # serialize dicts, lists, and tuples
                if isinstance(v, (dict, list, tuple)):
                    manifest[k] = json.dumps(v)
                # bool to int
                elif isinstance(v, bool):
                    manifest[k] = int(v)

            existing = self.db.get_manifest(strategy=self.strategy_name)
            if existing:
                self.db.update_manifest(manifest, strategy=self.strategy_name)
            else:
                self.db._insert("pretok_strategy", manifest)
        else:
            pass #This situation should happen only in multiprocessing mode


    def check_data_compatibility(self, db_names: List[str], modality: str) -> bool:
        """Check if strategy is compatible with given data characteristics."""
        if modality != self.modality:
            return False
        
        for required_db in self.wants_from_db:
            if required_db not in db_names:
                return False
        
        return True
        
    def pretokenize_batch(self, batch):
        if not self.initialized:
            self._initialize_components()

        if self.supports_batch:
            # Use the splitter's batched implementation
            return self.splitter.batched(batch)
        else:
            # single-document pretokenization
            results = []
            for document in batch:
                results.append(self.pretokenize_document(document))
            return results

    def pretokenize_document(self, document):
        """
        Pretokenize a single document using the configured splitter.
        This function can be called either from the extern, in order to perform a pretokenization
        and cache the values, or from the intern, in order to perform an on-the-fly tokenization
        """
        if not self.initialized:
            self._initialize_components()
        # Check if the data dict is compatible with what the splitter wants
        if not self.wants_raw:      
            for required_key in self.wants_from_db:
                if required_key not in document:
                    raise MissingDataKey(f"Required key '{required_key}' not found")
        
        # Call the splitter and return the data to be inserted into the db
        #Is it batched?
        split_result = self.splitter(document)
        
        # Validate that splitter returned expected outputs
        for expected_output in self.returns:
            if expected_output not in split_result:
                raise MissingSplitterOutput(f"Splitter did not return expected output: {expected_output}")
        
        return split_result

    def prepare_chunks_for_storage(self, chunks_tuples: List[tuple], max_tokens_per_chunk: Optional[int] = None, 
                                 tokens_length: Optional[int] = None, strict_checks: bool = False) -> bytes:
        """
        Convert chunk tuples to numpy array for storage.
        It converts the token to a numpy ndarray with the correct datatype and return raw bytes
        """
        if isinstance(chunks_tuples, list):
            # Pointers is a list of tuples ex: [(0,15), (15,36), (36,92)]
            # Transformed to a size array, ready to be stored in the pointers db:
            
            starts = np.array([c[0] for c in chunks_tuples])
            ends = np.array([c[1] for c in chunks_tuples])
            sizes = (ends - starts)
            if strict_checks:
                assert sizes.sum() == (tokens_length if tokens_length is not None else sizes.sum())
                max_size = sizes.max() if sizes.size > 0 else 0
                if max_tokens_per_chunk is not None:
                    assert max_size <= max_tokens_per_chunk
            
            chunks_dtype = get_datatype_for_numpy(self.chunks_datatype)
            return sizes.astype(chunks_dtype).tobytes(order='C')
        else:
            raise InvalidChunksFormat(f"The Splitter returned {type(chunks_tuples)} as chunks instead of a list")

    def prepare_tokens_for_storage(self, tokens: List[int]) -> bytes:
        """Convert tokens list to numpy array bytes for storage."""
        tokens_dtype = get_datatype_for_numpy(self.tokens_datatype)
        token_array = np.array(tokens, dtype=tokens_dtype)
        return token_array.tobytes(order='C')

    def prepare_extra_data_for_storage(self, obj: Any) -> bytes:
        """Prepare extra data for storage, converting to bytes if needed."""
        if isinstance(obj, bytes):
            return obj
        elif isinstance(obj, np.ndarray):
            return obj.tobytes(order='C')
        elif isinstance(obj, str):
            return obj.encode('utf-8')
        else:
            if orjson:
                return orjson.dumps(obj)
            else:
                return json.dumps(obj).encode('utf-8')

    def __call__(self, key: int, data_dict: Optional[Dict] = None, cache_dict: Optional[Dict] = None, 
                 max_seq_len: Optional[int] = None, wanted_from_strategy: List = [], add_special_tokens: bool = True) -> Dict[str, Any]:
        """
        Process a document, either from cache or by pretokenizing.
        """

        # Should we perform the pretokenization or use the cached values?
        if cache_dict is not None:
            # Use cached pretokenized data
            return_dict = {}
            for db_name in self.returns:
                if db_name in wanted_from_strategy:
                    return_dict[db_name] = self._retrieve_from_storage(cache_dict[db_name][key], db_name)
            
            # Chunking tokens
            if 'chunked_tokens' in wanted_from_strategy and 'tokens' in cache_dict and 'chunks' in cache_dict:
                splitter_args = self.splitter_arguments or {}
                if splitter_args.get('splitter_decides_chunking', False):
                        #Leave the chunking logic to the custom splitter (for special cases)
                        self._initialize_components()
                        wanted_dbs_for_chunking = splitter_args.get('wanted_dbs_for_chunking', ['tokens', 'chunks'])
                        data_from_dbs = dict()
                        for db in wanted_dbs_for_chunking:
                            data_from_dbs[db] = self._retrieve_from_storage(cache_dict[db][key], db)
                        return_dict['chunked_tokens'] = self.splitter.chunk_tokens(data_from_dbs)

                else: #Default chunking logic (splitter is not even intialized, faster)
                    tokens_data = self._retrieve_from_storage(cache_dict['tokens'][key], 'tokens')
                    chunks_data = self._retrieve_from_storage(cache_dict['chunks'][key], 'chunks')
                    return_dict['chunked_tokens'] = self._chunk_tokens(tokens_data, chunks_data, add_special_tokens)
            return return_dict
        else:
            # We need to call splitter for pretokenization
            cached = self.pretokenize_document(key=key, data_dict=data_dict)
            
            
            if wanted_from_strategy:
                return_dict = {}
                for db_name in wanted_from_strategy:
                    if db_name in cached:
                        return_dict[db_name] = cached[db_name]
                    elif db_name == 'chunked_tokens' and 'tokens' in cached and 'chunks' in cached:
                        return_dict['chunked_tokens'] = self._chunk_tokens(cached['tokens'], cached['chunks'], add_special_tokens)
                return return_dict
            
            return cached

    def _retrieve_from_storage(self, stored_bytes: bytes, db_name: str):
        """Retrieve and decode data from storage based on database type."""
        if db_name == 'tokens':
            tokens_dtype = get_datatype_for_numpy(self.tokens_datatype)
            return np.frombuffer(stored_bytes, dtype=tokens_dtype).tolist()
        elif db_name == 'chunks':
            chunks_dtype = get_datatype_for_numpy(self.chunks_datatype)
            chunk_sizes = np.frombuffer(stored_bytes, dtype=chunks_dtype)
            chunks = []
            start = 0  # Python int to avoid overflow
            for size in chunk_sizes:
                size = int(size) 
                chunks.append((start, start + size))
                start += size
            return chunks            
        else:
            # For extra data, try to decode as string first, then JSON
            try:
                return stored_bytes.decode('utf-8')
            except UnicodeDecodeError:
                if orjson:
                    return orjson.loads(stored_bytes)
                else:
                    return json.loads(stored_bytes.decode('utf-8'))

    def _chunk_tokens(self, tokens: List[int], chunks: List[tuple], add_special_tokens: bool = True) -> List[List[int]]:
        """Split tokens into chunks based on chunk boundaries."""

        chunked_tokens = []
        for start, end in chunks:
            chunk = tokens[start:end]
            if add_special_tokens:  
                    chunk = [self.bos_token_id] + chunk + [self.eos_token_id]
            chunked_tokens.append(chunk)
        return chunked_tokens
    def _chunk_tokens(self, tokens: List[int], chunks: List[tuple], add_special_tokens: bool = True) -> List[List[int]]:
        """Split tokens into chunks based on chunk boundaries."""
        chunked_tokens = []
        for start, end in chunks:
            chunk = tokens[start:end]
            if add_special_tokens:
                chunk = [self.bos_token_id] + chunk + [self.eos_token_id]
            chunked_tokens.append(chunk)
        return chunked_tokens


# Exception classes
if True:
    class MissingStrategyKey(Exception):
        pass

    class StrategyNotFound(Exception):
        pass

    class SplitterClassNotFound(Exception):
        pass

    class MissingDataKey(Exception):
        pass

    class MissingSplitterOutput(Exception):
        pass

    class InvalidChunksFormat(Exception):
        pass

from transformers import AutoTokenizer

from typing import List, Dict, Union, Any
from nltk.tokenize import PunktTokenizer

class SmallSplitter:
    def __init__(self, language: str, chunk_size: int, tokenizer: Any):
        self.punkt = PunktTokenizer(language)
        self.tokenizer = getattr(tokenizer, "tokenizer", tokenizer) 
        self.chunk_size = chunk_size

    def _get_encoding_data(self, encoding) -> tuple:
        if hasattr(encoding, "ids"): 
            if len(encoding.ids)<=10:
               print(f"WARNING: Sequence (in batch) has just {len(encoding.ids)} tokens.")
               try:
                  print(self.tokenizer.decode(encoding.ids))
               except:
                  pass
            return encoding.ids, encoding.offsets
        if len(encoding['input_ids'])<=10:
            print(f"WARNING: Sequence has just {len(encoding['input_ids'])} tokens.")
            try:
               print(self.tokenizer.decode(encoding['input_ids']))
            except:
               pass
        return encoding["input_ids"], encoding["offset_mapping"]

    def _process(self, text: str, encoding) -> Dict[str, Any]:
        ids, offsets = self._get_encoding_data(encoding)
        sent_spans = self.punkt.span_tokenize(text)
        
        chunks, curr_chunk = [], []
        tok_idx, n_tokens = 0, len(ids)

        for _, sent_end in sent_spans:
            sent_toks = []
            while tok_idx < n_tokens and offsets[tok_idx][0] < sent_end:
                sent_toks.append(ids[tok_idx])
                tok_idx += 1
            
            # For very long sequences
            if len(sent_toks) > self.chunk_size:
                if curr_chunk: chunks.append(curr_chunk); curr_chunk = []
                for i in range(0, len(sent_toks), self.chunk_size):
                    sub = sent_toks[i : i + self.chunk_size]
                    if len(sub) == self.chunk_size: chunks.append(sub)
                    else: curr_chunk.extend(sub)
                continue

            # Normal length
            if len(curr_chunk) + len(sent_toks) > self.chunk_size:
                chunks.append(curr_chunk)
                curr_chunk = list(sent_toks)
            else:
                curr_chunk.extend(sent_toks)

        if curr_chunk: chunks.append(curr_chunk)

        ranges, cursor = [], 0
        flat_tokens = []
        for c in chunks:
            flat_tokens.extend(c)
            ranges.append((cursor, cursor + len(c)))
            cursor += len(c)
            
        return {"tokens": flat_tokens, "chunks": ranges}

    def __call__(self, document: str) -> Dict[str, Any]:
        encoding = self.tokenizer(document, return_offsets_mapping=True, add_special_tokens=False)
        return self._process(document, encoding)

    def batched(self, batch: List[str]) -> List[Dict[str, Any]]:
        encodings = self.tokenizer(
            batch, return_offsets_mapping=True, add_special_tokens=False, 
            verbose=False, padding=False, truncation=False
        )
        
        return [
            self._process(doc, {
                'input_ids': encodings['input_ids'][i],
                'offset_mapping': encodings['offset_mapping'][i]
            })
            for i, doc in enumerate(batch)
        ]

class split_and_tokenize_by_nltk_sentences_aligned:
    def __init__(self, language, chunk_size, tokenizer):
        from typing import List, Tuple
        import time
        import difflib
        import re
        from tqdm import tqdm
        import numpy as np
        self.punkt_tokenizer = PunktTokenizer(language) 
        self.tokenizer = tokenizer.tokenizer #Accessing the huggingface tokenizer inside MatformerTokenizer
        self.language = language
        self.max_tokens = chunk_size
    
    def get_sentence_spans(self, document: str):
        spans = [x for x in self.punkt_tokenizer.span_tokenize(document)]
        starts = [s[0] for s in spans]
        ends = [s[0] for s in spans[1:]] + [len(document)]
        spans = list(zip(starts, ends))
        return spans

    def align(self, sentencespans, encoding):
        tokenspans = []
        try:
            mapping = np.array(encoding["offset_mapping"])
            starts = mapping[:, 0]
            ends = mapping[:, 1]

            for sent_start, sent_end in sentencespans:
                tokenstart = int(np.searchsorted(starts, sent_start, side="right")-1)
                tokenend = int(np.searchsorted(ends, sent_end, side="left")+1)
                tokenspans.append((max(tokenstart, 0), min(tokenend, len(mapping))))
            if len(tokenspans)>0:
                tokenspans[-1]=(tokenspans[-1][0],len(mapping))
                starts = [s[0] for s in tokenspans]
                ends = [s[0] for s in tokenspans[1:]] + [len(mapping)]
                tokenspans = list(zip(starts, ends))            
            else:
                print("WARNING: Empty sequence")
        except Exception as e:
            print(f"Error. {e}")
            print(encoding)
        return tokenspans
    def trim_long_sequences(self, aligned_spans, maxlen):
        #print(maxlen)   
        new_aligned_spans = []
        
        for x in aligned_spans:
            span_length = x[1] - x[0]
            
            if span_length > maxlen:
                # Split the long span into multiple chunks of maxlen
                start = x[0]
                while start < x[1]:
                    end = min(start + maxlen, x[1])
                    new_aligned_spans.append((start, end))
                    start = end
            else:
                new_aligned_spans.append(x)
        
        return new_aligned_spans


    def lenspan(self, start, end):
        return end[1] - start[0]

    def create_final_spans(self, aligned_spans, maxlen):
        span_idx_start = 0
        span_idx_end = 0
        finalspans = []

        while span_idx_end < len(aligned_spans):
            if self.lenspan(aligned_spans[span_idx_start], aligned_spans[span_idx_end]) > maxlen:
                finalspans.append(
                    (aligned_spans[span_idx_start][0], aligned_spans[span_idx_end - 1][1])
                )
                span_idx_start = span_idx_end
            else:
                span_idx_end += 1
                if span_idx_end >= len(aligned_spans):
                    finalspans.append(
                        (aligned_spans[span_idx_start][0], aligned_spans[span_idx_end - 1][1])
                    )

        return finalspans

    def __call__(self, document, batch_idx=None, batched_encoding=None):
        if isinstance(document, bytes):
            document = document.decode("utf-8")
        if not isinstance(document, str):
            raise Exception("Document must be a string or bytes")

        if batch_idx is not None and batched_encoding is not None:
            encoding = batched_encoding[batch_idx]
        else:
            encoding = self.tokenizer(document, return_offsets_mapping=True, add_special_tokens=False)

        if hasattr(encoding, "ids"):  # it's a tokenizers.Encoding
            all_tokens = encoding.ids
            offset_mapping = encoding.offsets
        else:  # it's a BatchEncoding
            all_tokens = encoding["input_ids"]
            offset_mapping = encoding["offset_mapping"]
        try:
            if len(all_tokens) == 0:
                print("WARNING: Empty sequence")
            elif len(all_tokens) == 1:
                print("WARNING: Sequence with just one token!")
            else:
                pass
        except:
            print(f"DEBUG: Conteggio token fallito! {all_tokens}")
        spans = self.get_sentence_spans(document)
        aligned_spans = self.align(spans, {"offset_mapping": offset_mapping})
        aligned_spans = self.trim_long_sequences(aligned_spans, self.max_tokens)
        finalspans = self.create_final_spans(aligned_spans, self.max_tokens)

        chunk_ranges = [(span[0], span[1]) for span in finalspans]

        return {
            "tokens": all_tokens,
            "chunks": chunk_ranges,
        }


    def batched(self, batch):
        batched_encoding = self.tokenizer(batch, return_offsets_mapping=True, add_special_tokens=False)
        results = []
        for i, document in enumerate(batch):
            results.append(self(document, batch_idx=i, batched_encoding=batched_encoding))
        return results
        
class split_and_tokenize_by_nltk_sentences:
    def __init__(self,language,chunk_size, tokenizer):
        from typing import List, Tuple
        import time
        import difflib
        import re
        from tqdm import tqdm
        from nltk.tokenize import PunktTokenizer        
        self.punkt_tokenizer = PunktTokenizer(language) 
        self.tokenizer=tokenizer
        self.language=language
        self.max_tokens=chunk_size
    def __call__(self,document):
        if isinstance(document,bytes):
            document=document.decode('utf-8')
        if not isinstance(document,str):
            raise Exception
        """
        Divide un documento in chunk di token rispettando i confini delle frasi.
        
        Args:
            document: Il testo da processare
            max_tokens: Numero massimo di token per chunk
            punkt_tokenizer: Tokenizer per le sentence di NLTK
            tokenizer: Tokenizer di HuggingFace
            language: Lingua del testo (default: italiano)
        
        Returns:
            all_tokens: Lista di tutti i token del documento (appiattita)
            token_chunks: Lista di chunk di token
            chunk_ranges: Lista di tuple (start, end) per ogni chunk
        """
        # Ottengo con NLTK gli span di ciascuna frase
        spans = [x for x in self.punkt_tokenizer.span_tokenize(document)]
        
        # Riallineo gli span in modo che la fine del primo coincida con l'inizio del secondo e così via
        starts = [s[0] for s in spans]
        ends = [s[0] for s in spans[1:]] + [len(document)]
        spans = list(zip(starts, ends))
        
        # Tokenizzo span per span fin quando non ho finito gli span, dividendo a gruppi da max_tokens token
        spans.reverse()
        token_chunks = []  # La lista con i blocchi di token divisi per segmento
        current_tokens_list = []  # Lista token di appoggio

        while len(spans) != 0:
            current_span = spans.pop()
            stringa_span = document[current_span[0]:current_span[1]]
            tokens_span = self.tokenizer.encode(stringa_span, add_special_tokens=False)
            
            if len(tokens_span) + len(current_tokens_list) <= self.max_tokens:
                # Questo span entra, lo aggiungo
                current_tokens_list.extend(tokens_span)
            else:
                # Questo span NON entra in max token. Salvo gli appoggi precedenti e passo a un nuovo segmento
                if current_tokens_list:  # Solo se non è vuoto
                    token_chunks.append(current_tokens_list)
                
                if len(tokens_span) > self.max_tokens:
                    # Siamo nella situazione in cui una singola "sentence" supera il numero di token massimi.
                    num_chunks = math.ceil(len(tokens_span) / self.max_tokens)
                    token_chunks_split = []
                    
                    for i in range(num_chunks):
                        start_idx = i * self.max_tokens
                        end_idx = min((i + 1) * self.max_tokens, len(tokens_span))
                        token_chunks_split.append(tokens_span[start_idx:end_idx])
                    
                    # Aggiungi tutti i chunk tranne l'ultimo
                    token_chunks.extend(token_chunks_split[:-1])
                    current_tokens_list = token_chunks_split[-1]
                else:
                    current_tokens_list = tokens_span
        
        if current_tokens_list:  # Salvo eventuali residui
            token_chunks.append(current_tokens_list)

        # Crea la lista di tutti i token
        all_tokens = []
        for chunk in token_chunks:
            all_tokens.extend(chunk)
        
        # Crea le tuple (start, end) per ogni chunk
        chunk_ranges = []
        current_idx = 0
        for chunk in token_chunks:
            chunk_length = len(chunk)
            chunk_ranges.append((current_idx, current_idx + chunk_length - 1))
            current_idx += chunk_length
        
        return {
            "tokens":all_tokens,
            "chunks":chunk_ranges
        }       
        pass                
    def _batched():
        raise NotImplementedError

import argparse
import json
import os
import sys
import subprocess
import shlex
import glob
from pathlib import Path
from typing import Dict, Any, Optional, Sequence
def fmt_bytes(n: int) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.2f} {u}"
        n /= 1024
    return f"{n:.2f} PB"


def parse_size(s: str) -> int:
    s = s.strip().upper()
    if not s:
        return 0
    unit = s[-1]
    if unit in "KMGTP":
        val = float(s[:-1])
        exponent = {"K": 1, "M": 2, "G": 3, "T": 4, "P": 5}[unit]
        return int(val * (1024**exponent))
    try:
        return int(float(s))
    except ValueError:
        return 0


def cmd_create(path: str, overwrite: bool, dsmap_bits: int, modalities: Sequence[str]):
    MatformerDataset.create_new(
        path, overwrite=overwrite, dsmap_bits=dsmap_bits, modalities=modalities
    )
    return f"Dataset created at {path}"


def cmd_load(
    path: str, readonly: bool = False, view: Optional[str] = None, shuffle: bool = False
):
    ds = MatformerDataset.load_dataset(
        path, readonly=readonly, ds_view=view, shuffle=shuffle
    )
    return {
        "path": path,
        "ds_object": ds,
        "documents": ds.total_documents,
        "submdats": len(ds.list_submdat()),
        "strategies": len(ds.list_strategies()),
    }


def cmd_add_submdat(
    path: str,
    name: str,
    data_compress: int = 9,
    meta_compress: int = 9,
    data_mapsize: str = "350G",
    meta_mapsize: str = "350G",
    modality: str = "text",
    ds: Optional[MatformerDataset] = None, 
):
    ds = ds or MatformerDataset.load_dataset(path, readonly=False)
    ds.add_submdat(
        submdat_name=name,
        compression_levels={"data": data_compress, "meta": meta_compress},
        map_sizes={"data": parse_size(data_mapsize), "meta": parse_size(meta_mapsize)},
        modality=modality,
    )
    return f"Submdat '{name}' added"


def cmd_batch_import(
    path: str,
    folder: str,
    data_key: str = "text",
    data_compress: int = 9,
    meta_compress: int = 9,
    data_mapsize: str = "500G",
    meta_mapsize: str = "500G",
    pattern: str = "*.jsonl",
    recursive: bool = False,
    modality: str = "text",
    ds: Optional[MatformerDataset] = None, 
):
    ds = ds or MatformerDataset.load_dataset(path, readonly=False)
    glob_path = os.path.join(folder, "**", pattern) if recursive else os.path.join(folder, pattern)
    files = glob.glob(glob_path, recursive=recursive)
    
    if not files:
        raise RuntimeError(f"No files matching '{glob_path}'")
    ok = 0
    for fp in files:
        name = Path(fp).stem
        print(f"Adding file {fp}...")
        sub = ds.add_submdat(
            name,
            compression_levels={"data": data_compress, "meta": meta_compress},
            map_sizes={
                "data": parse_size(data_mapsize),
                "meta": parse_size(meta_mapsize),
            },
            modality=modality,
        )
        res = sub.convert_to_submdat("jsonl", fp, data_key=data_key, progress_bar=True)
        if res:
            ok += 1
    return f"Batch import done: {ok}/{len(files)} succeeded"


def cmd_convert(path: str, submdat: str, _type: str, source: str, data_key: str = "text", ds: Optional[MatformerDataset] = None):
    ds = ds or MatformerDataset.load_dataset(path, readonly=False)
    sub = ds[submdat]
    res = sub.convert_to_submdat(_type, source, data_key=data_key, progress_bar=True)
    if not res:
        raise RuntimeError("Conversion failed")
    return f"Converted {res.get('document_number', 0)} documents"


def cmd_list_submdats(path: str, detailed: bool = False, ds: Optional[MatformerDataset] = None):
    ds = ds or MatformerDataset.load_dataset(path, readonly=True)
    subs = ds.list_submdat()
    rows = []
    for name in subs:
        m = ds.db.get_manifest(submdat=name)
        rows.append(
            {
                "name": name,
                "docs": m.get("document_number", 0),
                "key": m.get("data_key"),
                "strategies": json.loads(m.get("pretokenization_strategies", "[]")),
            }
        )
    return rows


def cmd_list_strategies(path: str, ds: Optional[MatformerDataset] = None):
    ds = ds or MatformerDataset.load_dataset(path, readonly=True)
    return ds.list_strategies()


def cmd_register_strategy(path: str, name: str, json_path: str, ds: Optional[MatformerDataset] = None):
    ds = ds or MatformerDataset.load_dataset(path, readonly=False)
    with open(json_path) as f:
        ds.register_strategy(name, json.load(f))
    return f"Strategy '{name}' registered"


def cmd_pretokenize(
    path: str,
    submdat: str,
    strategy: str,
    parallel: bool = False,
    processes: Optional[int] = None,
    compression: int = 9,
    ds: Optional[MatformerDataset] = None,
):
    ds = ds or MatformerDataset.load_dataset(path, readonly=False)
    ds.set_strategy(strategy)
    sub = ds.get_submdat(submdat)
    sub.pretokenize_submdat(
        strategy,
        progress_bar=True,
        parallel=parallel,
        num_processes=processes,
        compression_level=compression,
    )
    return "Pretokenization complete"

def cmd_pretokenize_batch(
    path: str,
    submdats: Sequence[str],
    strategy: str,
    parallel: bool = False,
    processes: Optional[int] = None,
    compression: int = 9,
    ds: Optional[MatformerDataset] = None,
):
    """Pretokenize multiple submdats with the same strategy."""
    ds = ds or MatformerDataset.load_dataset(path, readonly=False)
    
    results = []
    failed = []
    
    for submdat_name in submdats:
        try:
            print(f"\nPretokenizing '{submdat_name}' with strategy '{strategy}'...")
            sub = ds.get_submdat(submdat_name)
            sub.pretokenize_submdat(
                strategy,
                progress_bar=True,
                parallel=parallel,
                num_processes=processes,
                compression_level=compression,
            )
            results.append(submdat_name)
            print(f"Completed: {submdat_name}")
        except Exception as e:
            failed.append((submdat_name, str(e)))
            print(f"Failed: {submdat_name} - {e}")
    
    summary = f"\nBatch pretokenization complete: {len(results)}/{len(submdats)} succeeded"
    if failed:
        summary += f"\nFailed ({len(failed)}):"
        for name, err in failed:
            summary += f"\n  - {name}: {err}"
    
    return summary

def cmd_export_view_template(path: str, output: str = "mock_view.json", ds: Optional[MatformerDataset] = None):

    ds = ds or MatformerDataset.load_dataset(path, readonly=True)
    submdats = ds.list_submdat()
    
    selection_dict = {name: "preserved" for name in submdats}
    
    tpl = {
        "view_name": "example_view",
        "selection": selection_dict,
        "shuffle": True
    }
    
    try:
        Path(output).write_text(json.dumps(tpl, indent=2))
        return f"View template written to {output}. Edit this file before creating the view."
    except Exception as e:
        return f"Error writing template: {e}"


def cmd_create_view(path: str, json_path: str, shuffle: bool = True, ds: Optional[MatformerDataset] = None):
    ds = ds or MatformerDataset.load_dataset(path, readonly=False)
    try:
        with open(json_path) as f:
            view_dict = json.load(f)
    except Exception as e:
        return f"Error reading JSON file {json_path}: {e}"
        
    if not shuffle:
        view_dict['shuffle'] = False
    
    if True:
        ds.register_new_view(view_dict)
        

            
        return f"View '{view_dict.get('view_name', 'N/A')}' created"



def cmd_list_views(path: str, ds: Optional[MatformerDataset] = None):
    ds = ds or MatformerDataset.load_dataset(path, readonly=True)
    views=ds.list_views()
    returns=[]
    for v in views:
        returns.append((v,ds.db.get_view_totals(v)['document_number']))

    return returns


def cmd_set_view(path: str, view: str, ds: Optional[MatformerDataset] = None):
    ds = ds or MatformerDataset.load_dataset(path, readonly=False)
    ds.set_view(view)
    return f"Active view set to '{view}'"
    
def cmd_export_view_data(
    path: str,
    view: str = "default",
    output: str = "export_view.jsonl",
    wanted: str = "document",
    with_meta: bool = True,
    limiter: Optional[int] = None,
    ds: Optional[MatformerDataset] = None
):
    ds = ds or MatformerDataset.load_dataset(path, readonly=True)
    ds.export_view(output_file=output, wanted=wanted, with_meta=with_meta, limiter=limiter, view=view)
    return f"View exported to '{output}'"

def cmd_shuffle_view(path: str, view: str, ds: Optional[MatformerDataset] = None):
    ds = ds or MatformerDataset.load_dataset(path, readonly=False)
    ds.shuffle(view)
    return f"View '{view}' shuffled"


def cmd_info(path: str, verbose: bool = False, ds: Optional[MatformerDataset] = None):
    ds = ds or MatformerDataset.load_dataset(path, readonly=True)
    info = {
        "path": path,
        "documents": ds.total_documents,
        "submdats": len(ds.list_submdat()),
        "strategies": len(ds.list_strategies()),
    }
    if verbose:
        info["manifest"] = ds.manifest
    return info


class InteractiveShell:
    """
    [REFACTOR]: Revamped TUI to hold the dataset object in memory,
    eliminating subprocess calls for massive performance gains and UI cleanup.
    """
    def __init__(self):
        self.dataset_path: Optional[str] = None
        self.ds: Optional[MatformerDataset] = None

    def _clear(self):
        """Clears the terminal screen."""
        os.system("cls" if os.name == "nt" else "clear")

    def _pause(self):
        """Waits for user to press Enter."""
        try:
            input("\nPress Enter to continue...")
        except (KeyboardInterrupt, EOFError):
            pass 

    def _call_cmd(self, func, *args, **kwargs):
        """Helper to call command functions, print results, and handle errors."""
        if True: #for debug
        #try:
            if self.ds:
                kwargs["ds"] = self.ds
            
            result = func(*args, **kwargs)
            print(f"\n {result}")
        #except Exception as e:
        #    print(f"\n Error: {e}")

    def prompt(self, text, default=None, cast=str):
        try:
            raw = input(
                f"{text}{f' [{default}]' if default is not None else ''}: "
            ).strip()
            if not raw and default is not None:
                return default
            if raw is None: 
                return None
            return cast(raw)
        except (ValueError):
            print("Invalid input – try again")
            return self.prompt(text, default, cast)
        except (KeyboardInterrupt, EOFError):
            print() 
            return None 

    def pick(self, options, title="Select"):
        if not options:
            print("\nNo options to pick from.")
            return None
            
        print(f"\n{title}:")
        for i, opt in enumerate(options, 1):
            print(f"  {i}. {opt}")
        while True:
            try:
                idx_raw = self.prompt("Choice", cast=int)
                if idx_raw is None: 
                    return None
                idx = int(idx_raw)
                if 1 <= idx <= len(options):
                    return options[idx - 1]
            except (ValueError):
                pass
            print("Out of range – try again")

    def main_loop(self):
        while True:
            self._clear()
            print("===  mdat interactive ===")
            if self.dataset_path:
                print(f"Current dataset: {self.dataset_path}")
            
            print("\n--- Main Menu ---")
            print("1. Create dataset")
            print("2. Load dataset")
            if self.dataset_path:
                print("3. Submdat operations")
                print("4. Strategy operations")
                print("5. View operations")
                print("6. Dataset info")
                print("7. Close dataset")
            print("0. Exit")
            
            choice_raw = self.prompt("Select", cast=int)
            if choice_raw is None: # Ctrl+C
                choice = 0 
            else:
                choice = choice_raw

            if choice == 0:
                print("Exiting.")
                break
            
            pause = True 
            if choice == 1:
                path = self.prompt("Path")
                if path:
                    overwrite = self.prompt("Overwrite (y/n)", "n").lower() == "y"
                    bits = self.prompt("dsmap bits", 8, cast=int)
                    self._call_cmd(
                        cmd_create, 
                        path=path, 
                        overwrite=overwrite, 
                        dsmap_bits=bits, 
                        modalities=["text"]
                    )
                else:
                    pause = False 
            
            elif choice == 2:
                path = self.prompt("Path")
                if path:
                    try:
                        info = cmd_load(path, readonly=False) 
                        self.ds = info["ds_object"]
                        self.dataset_path = path
                        print(f"\n Dataset loaded: {info['documents']:,} docs, "
                              f"{info['submdats']} submdats, {info['strategies']} strategies")
                    except Exception as e:
                        self.ds = None
                        self.dataset_path = None
                        print(f"\n Error loading dataset: {e}")
                else:
                    pause = False
            
            elif choice == 7:
                self.ds = None
                self.dataset_path = None
                print("\nDataset closed.")
                pause = True # Pause to show message
            
            elif self.dataset_path and choice == 6:
                try:
                    info = cmd_info(self.dataset_path, ds=self.ds)
                    print("\n--- Dataset Info ---")
                    print(f"Path: {info['path']}")
                    print(f"Documents: {info['documents']:,}")
                    print(f"Submdats: {info['submdats']}")
                    print(f"Strategies: {info['strategies']}")
                except Exception as e:
                    print(f"\n Error getting info: {e}")

            elif self.dataset_path and choice == 3:
                self.submdat_menu()
                pause = False 
            
            elif self.dataset_path and choice == 4:
                self.strategy_menu()
                pause = False
            
            elif self.dataset_path and choice == 5:
                self.view_menu()
                pause = False
            
            elif choice > 7:
                print("Invalid choice.")
            
            if pause:
                self._pause()

    def submdat_menu(self):
        while True:
            self._clear()
            print(f"--- Submdat Operations [{self.dataset_path}] ---")
            print("1. List")
            print("2. Add")
            print("3. Batch import folder")
            print("4. Convert external data")
            print("5. Pretokenize (single)")
            print("6. Pretokenize (batch)")  
            print("0. Back to Main Menu")
            
            c = self.prompt("Select", cast=int)
            if c is None or c == 0:
                return

            pause = True
            if c == 1:
                try:
                    rows = cmd_list_submdats(self.dataset_path, ds=self.ds)
                    if not rows:
                        print("\nNo submdats.")
                    else:
                        print("\n--- Submdats ---")
                        print(f"{'Name':<25} {'Docs':>10}  {'Key':<10}  Strategies")
                        print("-" * 60)
                        for r in rows:
                            print(
                                f"{r['name']:<25} {r['docs']:>10,}  {r['key'] or 'N/A':<10}  {','.join(r['strategies'])}"
                            )
                except Exception as e:
                    print(f"\n Error listing submdats: {e}")
            
            elif c == 2:
                name = self.prompt("Name")
                if name:
                    self._call_cmd(cmd_add_submdat, path=self.dataset_path, name=name)
                else:
                    pause = False

            elif c == 3:
                folder = self.prompt("Folder")
                if folder:
                    key = self.prompt("Data key", "text")
                    self._call_cmd(
                        cmd_batch_import,
                        path=self.dataset_path,
                        folder=folder,
                        data_key=key,
                    )
                else:
                    pause = False
            
            elif c == 4:
                subs = cmd_list_submdats(self.dataset_path, ds=self.ds)
                if not subs:
                    print("No submdats to convert into.")
                    pause = True
                    continue
                name = self.pick([r["name"] for r in subs], "Select Target Submdat")
                print(f"Selected {name}")
                if name:
                    tp = self.pick(["jsonl", "lmdb", "hf", "atlas"], "Select Source Type")
                    if tp:
                        src = self.prompt("Source path")
                        if src:
                            self._call_cmd(
                                cmd_convert,
                                path=self.dataset_path,
                                submdat=name,
                                _type=tp,
                                source=src,
                            )
                        else:
                            pause = False
                    else:
                        pause = False
                else:
                    pause = False
            
            elif c == 5:  # Single pretokenize
                subs = cmd_list_submdats(self.dataset_path, ds=self.ds)
                if not subs:
                    print("No submdats to pretokenize.")
                    pause = True
                    continue
                name = self.pick([r["name"] for r in subs], "Select Submdat")
                if name:
                    strategies = cmd_list_strategies(self.dataset_path, ds=self.ds)
                    if not strategies:
                        print("No strategies registered. Cannot pretokenize.")
                        pause = True
                        continue
                    st = self.pick(strategies, "Select Strategy")
                    if st:
                        self._call_cmd(
                            cmd_pretokenize,
                            path=self.dataset_path,
                            submdat=name,
                            strategy=st,
                            parallel=True,
                        )
                    else:
                        pause = False
                else:
                    pause = False
            
            elif c == 6:  # NEW: Batch pretokenize
                subs = cmd_list_submdats(self.dataset_path, ds=self.ds)
                if not subs:
                    print("No submdats to pretokenize.")
                    pause = True
                    continue
                
                # Get strategy first
                strategies = cmd_list_strategies(self.dataset_path, ds=self.ds)
                if not strategies:
                    print("No strategies registered. Cannot pretokenize.")
                    pause = True
                    continue
                
                st = self.pick(strategies, "Select Strategy")
                if not st:
                    pause = False
                    continue
                
                # Select submdats
                print("\n--- Select Submdats to Pretokenize ---")
                print("Options:")
                print("  1. All submdats")
                print("  2. Select multiple individually")
                
                choice = self.prompt("Choice", cast=int)
                
                selected_submdats = []
                if choice == 1:
                    selected_submdats = [r["name"] for r in subs]
                    print(f"\nSelected all {len(selected_submdats)} submdats")
                elif choice == 2:
                    print("\nEnter submdat names (comma-separated):")
                    available_names = [r["name"] for r in subs]
                    print(f"Available: {', '.join(available_names)}")
                    
                    names_input = self.prompt("Submdats")
                    if names_input:
                        selected_submdats = [n.strip() for n in names_input.split(",")]
                        # Validate
                        invalid = [n for n in selected_submdats if n not in available_names]
                        if invalid:
                            print(f"Invalid submdat names: {', '.join(invalid)}")
                            selected_submdats = []
                else:
                    pause = False
                    continue
                
                if selected_submdats:
                    confirm = self.prompt(
                        f"Pretokenize {len(selected_submdats)} submdat(s) with '{st}'? (y/n)", 
                        "y"
                    ).lower()
                    if confirm == "y":
                        self._call_cmd(
                            cmd_pretokenize_batch,
                            path=self.dataset_path,
                            submdats=selected_submdats,
                            strategy=st,
                            parallel=True,
                        )
                    else:
                        print("Cancelled.")
                else:
                    pause = False
            
            else:
                print("Invalid choice.")
            
            if pause:
                self._pause()

    def strategy_menu(self):
        while True:
            self._clear()
            print(f"--- Strategy Operations [{self.dataset_path}] ---")
            print("1. List")
            print("2. Register")
            print("0. Back to Main Menu")
            
            c = self.prompt("Select", cast=int)
            if c is None or c == 0:
                return

            pause = True
            if c == 1:
                try:
                    strategies = cmd_list_strategies(self.dataset_path, ds=self.ds)
                    if not strategies:
                        print("\nNo strategies registered.")
                    else:
                        print("\n--- Registered Strategies ---")
                        print("\n".join(f"- {s}" for s in strategies))
                except Exception as e:
                    print(f"\n Error listing strategies: {e}")
            
            elif c == 2:
                name = self.prompt("Strategy name")
                if name:
                    json_path = self.prompt("JSON file")
                    if json_path:
                        self._call_cmd(
                            cmd_register_strategy,
                            path=self.dataset_path,
                            name=name,
                            json_path=json_path,
                        )
                    else:
                        pause = False
                else:
                    pause = False
            else:
                print("Invalid choice.")
            
            if pause:
                self._pause()
    def _pick_view(self, prompt_msg="Select a view"):
            views = [v[0] for v in cmd_list_views(self.dataset_path, ds=self.ds)]
            if not views:
                print("\nNo views available.")
                return None
            return self.pick(views, prompt_msg)
    def view_menu(self):

        while True:
            self._clear()
            print(f"--- View Operations [{self.dataset_path}] ---")
            print("1. List")
            print("2. Export mock view (JSON)")  # <-- UPDATED TEXT
            print("3. Create from JSON")
            print("4. Set active")
            print("5. Shuffle")
            print("6. Export data to file")
            print("0. Back to Main Menu")

            c = self.prompt("Select", cast=int)
            if c is None or c == 0:
                return
            
            pause = True
            if c == 1:
                if True:
                    views = cmd_list_views(self.dataset_path, ds=self.ds)
                    if not views:
                        print("\nNo views.")
                    else:
                        print("\n--- Views ---")
                        print(f"{'Name':<30} {'Docs':<10}")
                        print("-" * 40)
                        for name, docs in views:
                            doc_str = f"{docs:<10,}" if docs is not None else "N/A"
                            print(f"{name:<30} {doc_str}")
                #except Exception as e:
                #    print(f"\n Error listing views: {e}")
            
            elif c == 2:
                out = self.prompt("Output file", "mock_view.json") 
                if out:
                    self._call_cmd(cmd_export_view_template, path=self.dataset_path, output=out)
                    print(f"Edit {out} then use option 3")
                else:
                    pause = False
            
            elif c == 3:
                json_path = self.prompt("JSON file")
                if json_path:
                    self._call_cmd(cmd_create_view, path=self.dataset_path, json_path=json_path)
                else:
                    pause = False
            
            elif c in [4, 5]:  # Set or Shuffle
                v = self._pick_view(f"Select View to {'set' if c==4 else 'shuffle'}")
                if v:
                    func = cmd_set_view if c == 4 else cmd_shuffle_view
                    self._call_cmd(func, path=self.dataset_path, view=v)
                else:
                    pause = False

            elif c == 6:  
                v = self._pick_view("Select View to export")
                if not v:
                    pause = False
                    continue

                out = self.prompt("Output file", "export_view.jsonl")
                if not out:
                    pause = False
                    continue

                wanted = self.pick(["document", "tokens", "chunked_tokens"], "Select item type")
                with_meta = self.prompt("Include metadata? (y/n)", "y").lower() == "y"
                limiter_raw = self.prompt("Limiter (max items, empty for no limit)", "")
                limiter = int(limiter_raw) if limiter_raw.strip() else None

                self._call_cmd(
                    cmd_export_view_data,
                    path=self.dataset_path,
                    view=v,
                    output=out,
                    wanted=wanted,
                    with_meta=with_meta,
                    limiter=limiter
                )
                print(f"View '{v}' exported to {out}")
                            
            
            else:
                print("Invalid choice.")
            
            if pause:
                self._pause()


def build_cli():
    p = argparse.ArgumentParser(
        description="mdat – manage Matformer datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--tui", action="store_true", help="Start interactive TUI mode")
    sp = p.add_subparsers(dest="cmd", required=False, help="Sub-command to run")

    path_parser = argparse.ArgumentParser(add_help=False)
    path_parser.add_argument("--path", required=True, help="Path to the dataset")

    submdat_parser = argparse.ArgumentParser(add_help=False)
    submdat_parser.add_argument(
        "--data-compress", type=int, default=0, help="Data compression level"
    )
    submdat_parser.add_argument(
        "--meta-compress", type=int, default=0, help="Metadata compression level"
    )
    submdat_parser.add_argument(
        "--data-mapsize", default="10G", help="LMDB map size for data"
    )
    submdat_parser.add_argument(
        "--meta-mapsize", default="1G", help="LMDB map size for metadata"
    )

    # --- Command Definitions ---
    p_create = sp.add_parser(
        "create", parents=[path_parser], help="Create a new dataset"
    )
    p_create.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing dataset"
    )
    p_create.add_argument("--dsmap-bits", type=int, default=8)
    p_create.add_argument("--modalities", nargs="+", default=["text"])

    p_load = sp.add_parser("load", parents=[path_parser], help="Load a dataset")
    p_load.add_argument("--readonly", action="store_true")
    p_load.add_argument("--view", help="Load a specific view")
    p_load.add_argument("--shuffle", action="store_true", help="Shuffle view on load")

    p_add_sub = sp.add_parser(
        "add-submdat",
        parents=[path_parser, submdat_parser],
        help="Add a new, empty submdat",
    )
    p_add_sub.add_argument("--name", required=True, help="Name of the new submdat")
    p_add_sub.add_argument("--modality", default="text")

    p_batch = sp.add_parser(
        "batch-import",
        parents=[path_parser, submdat_parser],
        help="Import a folder of files",
    )
    p_batch.add_argument("--folder", required=True, help="Folder to import from")
    p_batch.add_argument("--data-key", default="text", help="JSON key for data")
    p_batch.add_argument("--pattern", default="*.jsonl", help="File pattern to match")
    p_batch.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    p_batch.add_argument("--modality", default="text")

    p_conv = sp.add_parser(
        "convert", parents=[path_parser], help="Convert external data into a submdat"
    )
    p_conv.add_argument("--submdat", required=True, help="Target submdat name")
    p_conv.add_argument(
        "--type",
        required=True,
        choices=["jsonl", "lmdb", "hf", "atlas"],
        help="Source data type",
    )
    p_conv.add_argument("--source", required=True, help="Source file or path")
    p_conv.add_argument("--data-key", default="text")

    p_ls_sub = sp.add_parser(
        "list-submdats", parents=[path_parser], help="List submdats in the dataset"
    )
    p_ls_sub.add_argument("--detailed", action="store_true", help="Show more details")

    sp.add_parser(
        "list-strategies", parents=[path_parser], help="List registered strategies"
    )

    p_reg_strat = sp.add_parser(
        "register-strategy",
        parents=[path_parser],
        help="Register a new strategy from JSON",
    )
    p_reg_strat.add_argument("--name", required=True, help="Name for the strategy")
    p_reg_strat.add_argument("--json", required=True, help="Path to the strategy JSON file")

    p_pretok = sp.add_parser(
        "pretokenize", parents=[path_parser], help="Pretokenize a submdat"
    )
    p_pretok.add_argument("--submdat", required=True, help="Submdat to pretokenize")
    p_pretok.add_argument("--strategy", required=True, help="Strategy to use")
    p_pretok.add_argument(
        "--parallel", action="store_true", help="Use parallel processing"
    )
    p_pretok.add_argument("--processes", type=int, help="Number of processes to use")
    p_pretok.add_argument("--compression", type=int, default=0)

    p_export_view = sp.add_parser(
        "export-view", 
        parents=[path_parser], 
        help="Export a mock view JSON (all submdats 'preserved')"  # <-- UPDATED HELP
    )
    p_export_view.add_argument(
        "--output", 
        default="mock_view.json",  # <-- UPDATED DEFAULT
        help="Output file name"
    )
    p_export_view_data = sp.add_parser(
        "export-view-stream",
        parents=[path_parser],
        help="Stream export the active view to JSONL (TB-scale, selectable metadata, type, and limiter)"
    )
    p_export_view_data.add_argument("--output", default="export_view.jsonl", help="Output file path")
    p_export_view_data.add_argument(
        "--wanted", default="document", choices=["document", "default"], help="Type of items to export"
    )
    p_export_view_data.add_argument(
        "--view", type=str,default="default", help="View to export"
    )
    p_export_view_data.add_argument(
        "--meta/--no-meta", dest="with_meta", default=True, help="Include metadata in export"
    )
    p_export_view_data.add_argument(
        "--limiter", type=int, default=None, help="Maximum number of items to export"
    )
    p_create_view = sp.add_parser(
        "create-view", parents=[path_parser], help="Create a view from JSON"
    )
    p_create_view.add_argument("--json", required=True, help="Path to the view JSON file")
    p_create_view.add_argument(
        "--no-shuffle",
        dest="shuffle",
        action="store_false",
        help="Do not shuffle the view (overrides 'shuffle' key in JSON)",  # <-- UPDATED HELP
    )

    sp.add_parser("list-views", parents=[path_parser], help="List all views")

    p_set_view = sp.add_parser(
        "set-view", parents=[path_parser], help="Set the active view"
    )
    p_set_view.add_argument("--view", required=True, help="View name to activate")

    p_shuffle_view = sp.add_parser(
        "shuffle-view", parents=[path_parser], help="Shuffle a view"
    )
    p_shuffle_view.add_argument("--view", required=True, help="View name to shuffle")

    p_shuffle_view = sp.add_parser(
        "shuffle-view", parents=[path_parser], help="Shuffle a view"
    )
    p_shuffle_view.add_argument("--view", required=True, help="View name to shuffle")

    p_info = sp.add_parser(
        "info", parents=[path_parser], help="Show dataset info"
    )
    p_info.add_argument("--verbose", action="store_true", help="Show full manifest")
    #Batch pretokenize
    p_pretok_batch = sp.add_parser(
        "pretokenize-batch", 
        parents=[path_parser], 
        help="Pretokenize multiple submdats with the same strategy"
    )
    p_pretok_batch.add_argument(
        "--submdats", 
        nargs="+", 
        required=True, 
        help="Submdat names to pretokenize (space-separated)"
    )
    p_pretok_batch.add_argument("--strategy", required=True, help="Strategy to use")
    p_pretok_batch.add_argument(
        "--parallel", action="store_true", help="Use parallel processing"
    )
    p_pretok_batch.add_argument("--processes", type=int, help="Number of processes to use")
    p_pretok_batch.add_argument("--compression", type=int, default=0)
    p_pretok_batch.add_argument(
        "--all", 
        action="store_true", 
        help="Pretokenize all submdats (ignores --submdats)"
    )
    return p


def main(argv=None):
    argv = argv or sys.argv[1:]
    
    if not argv or (argv and argv[0] == "--tui"):
        InteractiveShell().main_loop()
        return

    parser = build_cli()
    args = parser.parse_args(argv)

    if not args.cmd:
        parser.print_help()
        return

    cmd_func_map = {
        "create": cmd_create,
        "load": cmd_load,
        "add-submdat": cmd_add_submdat,
        "batch-import": cmd_batch_import,
        "convert": cmd_convert,
        "list-submdats": cmd_list_submdats,
        "list-strategies": cmd_list_strategies,
        "register-strategy": cmd_register_strategy,
        "pretokenize": cmd_pretokenize,
        "pretokenize-batch": cmd_pretokenize_batch,  
        "export-view": cmd_export_view_template,
        "export-view-data": cmd_export_view_data,
        "create-view": cmd_create_view,
        "list-views": cmd_list_views,
        "set-view": cmd_set_view,
        "shuffle-view": cmd_shuffle_view,
        "info": cmd_info,
    }

    try:
        kwargs = vars(args)
        cmd_name = kwargs.pop("cmd")
        kwargs.pop("tui", None)

        cmd_func = cmd_func_map.get(cmd_name)
        if not cmd_func:
            raise NotImplementedError(f"Command '{cmd_name}' not implemented in main")

        result = cmd_func(**kwargs)

        # --- Handle printing results for CLI ---
        if cmd_name == "load":
            info = result
            print(
                f"Dataset loaded: {info['documents']:,} docs, "
                f"{info['submdats']} submdats, {info['strategies']} strategies"
            )
        elif cmd_name == "pretokenize-batch" and kwargs.get("all"):
            from matformer_dataset import MatformerDataset
            ds = MatformerDataset.load_dataset(kwargs["path"], readonly=True)
            subs = [r["name"] for r in cmd_list_submdats(kwargs["path"], ds=ds)]
            kwargs["submdats"] = subs
            kwargs.pop("all")
            print(f"Pretokenizing all {len(subs)} submdats...")            
        elif cmd_name == "list-submdats":
            rows = result
            if not rows:
                print("No submdats")
            elif args.detailed:
                for r in rows:
                    print(
                        f"{r['name']:<25} – {r['docs']:>10,} docs – "
                        f"key={r['key']} – strategies={','.join(r['strategies'])}"
                    )
            else:
                print(f"{'Name':<30} {'Docs':<10}")
                print("-" * 40)
                for r in rows:
                    print(f"{r['name']:<30} {r['docs']:<10,}")
        elif cmd_name == "list-strategies":
            strategies = result
            print("\n".join(strategies) if strategies else "No strategies")
        elif cmd_name == "list-views":
            views = result
            if not views:
                print("No views")
            else:
                print(f"{'Name':<30} {'Docs':<10}")
                print("-" * 40)
                for name, docs in views:
                    print(f"{name:<30} {docs:<10,}")
        elif cmd_name == "info":
            info = result
            print(f"Path: {info['path']}")
            print(f"Documents: {info['documents']:,}")
            print(f"Submdats: {info['submdats']}")
            print(f"Strategies: {info['strategies']}")
            if args.verbose and "manifest" in info:
                print("Manifest:", json.dumps(info["manifest"], indent=2))
        elif result is not None:
            print(result)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

