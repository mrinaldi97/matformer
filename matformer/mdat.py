#!/usr/bin/env python3
import math
import os
import json
import struct
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
class MatformerDataset(IterableDataset):
    MANIFEST_FILENAME = 'manifest.json'
    MANIFEST_VERSION = "1.0"
    MANIFEST_TYPE = "mdat"
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
        """
        self.manifest: Dict[str, Any] = {}
        self.loaded_submdats: Dict[str, Any] = {}
        self.pretok_strategies: Dict[str, Any] = {}
        self.skipped_submats: List = []
        self.readonly: bool = False
        self.current_strategy=None
        self.total_documents=0
        self.total_tokens=None #This will be populated when a strategy is loaded
        self.total_chunks=None
        self.total_divided_chunks=None
        self.chunk_multiplier=1 # Standard chunk multiplier: 1
        
    def _get_manifest_attr(self, key: str, default: Any = None) -> Any:
        """Get manifest attribute (possible to set a default)"""
        return self.manifest.get(key, default)
    
    def _set_manifest_attr(self, key: str, value: Any, save: bool = False) -> None:
        """Set manifest attribute, but don't update the file unless 'save' is True"""
        self.manifest[key] = value
        if save:
            self.update_manifest()
        
    @classmethod
    def load_dataset(cls, path: str, shuffle: bool = False, 
                      ds_view: Optional[str] = None, distributed: bool = False, 
                      readonly: bool = False, create_if_not_existent: bool = False, batch_size:Optional[int]=None) -> 'MatformerDataset':
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
        
        if instance._mdat_exists(instance.manifest_path):
            with open(instance.manifest_path, 'r') as m:
                instance.manifest = json.load(m)
            instance.pretok_strategies = {}
            if distributed:       
                instance._set_distributed_training()
            instance.set_view(ds_view)
            instance._populate_submdat()
            instance._populate_strategies()
            instance.set_iteration_modality('document',with_meta=True) #Default: get a non-tokenized document with metadata
        else:
            if create_if_not_existent:
                # Create and then load the dataset
                return cls.create_new(path=path, overwrite=False)
            else:
                raise MDatNotFound(f"MDat not found at {path}")
        
        return instance
    
    @classmethod
    def create_new(cls, path: str, overwrite: bool = False, 
                   dsmap_bits: int = None) -> 'MatformerDataset':
        """
        Create a new empty Mdat and load it.
        
        Args:
            path: Directory path for new dataset
            overwrite: Overwrite existing dataset if present
            dsmap_bits: Dataset mapping bits (defaults to 8 bit)
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
        
        # Create the default manifest
        instance.manifest = {
            "type": cls.MANIFEST_TYPE,
            "version": cls.MANIFEST_VERSION,
            "dsmap_bits": dsmap_bits,
            "datasets_map": {},
            "views": [],
            "pretok_strategies": [],
            "shuffled": False,
            "isMultimodal": False,
        }
        
        instance.update_manifest()
        
        # Now load the created dataset
        return cls.load_dataset(path, readonly=False)
    
    def _set_paths(self, path: str) -> None:
        """
        Private function: set the correct paths
        Args:
            path: Root dataset directory path
        """
        self.mdat_path = path
        self.manifest_path = os.path.join(self.mdat_path, self.MANIFEST_FILENAME)
        self.pretok_path = os.path.join(path, self.PRETOK_DIR)
        self.dataset_path = os.path.join(path, self.DATASETS_DIR)
        self.views_path = os.path.join(path, self.VIEWS_DIR)
        self.shuffling_path = os.path.join(path, self.SHUFFLING_DIR)
        self.functions_path = os.path.join(path, self.FUNCTIONS_DIR)
    
    def _mdat_exists(self, manifest_path: str) -> bool:
        """Check if manifest exists."""
        return os.path.exists(manifest_path)
    
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
        raise SubmDatNotFound(f"Sub-dataset '{submdat_name}' not found")
    
    def shuffle(self, view_name=None, selected_submdats=None, terminate_early=False, termination_criteria=None, termination_dict=None):
        """
        This function will create a shuffled version of the dataset.
        Instead of actually shuffling the data, this function will create a file with pointers
        in the shuffle folder. For now, it supports only a shuffle of the whole dataset (shuffle/default.mdat)
        
        The pointer file is a file with structs:
        submdat_id,doc_id
        The datatype of submdat_id can either be 8 bit or 16 bit uint depending on the number of submdat (dsmap_bits)
        The datatype of doc_id depends from the largest number of documents in the submdats
        """
        if terminate_early:
            assert termination_criteria in ['bytes','documents_number']
            assert isistance(termination_dict,dict)
            accumulation_dict=dict()
        from bisect import bisect_right
        if self.readonly:
            raise MDatIsReadOnly
        
        # 
        ds_map = self._get_manifest_attr('datasets_map', {})
        if not ds_map:
            raise RuntimeError("No subdatasets found in manifest")        
            
        # Calculate lengths and determine datatypes
        ds_lengths = [len(self[ds_map[str(i)]]) for i in range(1, len(ds_map) + 1)] #Lenghts, indexed by the ID present in the manifest's map
        # Default: use all subdatasets
        if selected_submdats is None:
            selected_ids = list(range(len(ds_map)))  # 0-based: [0, 1, 2, ...]
        else:
            # Convert submdat's names to manifest's map's ID
            selected_ids = [int(i) - 1 for i, name in ds_map.items() if name in selected_submdats]
            selected_ids.sort()       
            
        selected_lengths = [ds_lengths[i] for i in selected_ids]
        cu_dslens = [0]
        s = 0
        for L in selected_lengths:
            s += L
            cu_dslens.append(s)

        totlen = cu_dslens[-1]
        maxlen = max(selected_lengths) if selected_lengths else 0
        
        doc_id_type = 'B' if maxlen <= 255 else 'H' if maxlen <= 65535 else 'I' if maxlen <= 4294967295 else 'Q'
        submdat_id_type = 'B' if self._get_manifest_attr('dsmap_bits', 8) == 8 else 'H'
        struct_format = submdat_id_type + doc_id_type
        
        if view_name is None:
            # If the shuffle is over the entire dataset, these attributes are put in the manifest
            self._set_manifest_attr('shuffle_struct_format', struct_format)
            self._set_manifest_attr('total_documents', totlen)
        else:
            pass
        # Create cumulative lengths and shuffle file
        #cu_dslens = [0] + [sum(ds_lengths[:i+1]) for i in range(len(ds_lengths))]
        
        # Output path
        file_name = view_name or 'default'
        os.makedirs(self.shuffling_path, exist_ok=True)
        shuffle_path = os.path.join(self.shuffling_path, f'{file_name}.mdat')      
        
          
        permutator = RandomPermutator(totlen)
        with open(shuffle_path, 'wb') as f:
            for idx in tqdm(range(totlen)):
                permuted_idx = permutator(idx)
                loc_id = bisect_right(cu_dslens, permuted_idx) - 1
                global_id = selected_ids[loc_id]
                doc_id = permuted_idx - cu_dslens[loc_id]
                # Logic for early termination in case of view creation
                if terminate_early:
                    if accumulators_equal_terminators():
                        raise IterationEnd
                    if accumulators[ds_map[global_id]]>=terminators[ds_map[global_id]]:
                        continue
                    else:
                        if termination_criteria=='documents_number':
                            accumulators[ds_map[global_id]]+=1                      
                        else: #bytes
                            accumulators[ds_map[global_id]]+=len(self[ds_map[global_id]].db['data'][doc_id])
                        f.write(struct.pack(struct_format, global_id, doc_id))
                else:
                    # Write everything
                    f.write(struct.pack(struct_format, global_id, doc_id))
        
        if view_name is None:
            self._set_manifest_attr('shuffled', True, save=True)
        else:
            return struct_format,totlen
    def delete_shuffle(self):
        self._set_manifest_attr('shuffled', False, save=True)
    def _init_shuffle_file(self):
        """Initialize shuffle file"""
        if not hasattr(self, '_shuffle_file') or self._shuffle_file is None:
            self._shuffle_struct_format = self._get_manifest_attr('shuffle_struct_format')
            self._shuffle_struct_size = struct.calcsize(self._shuffle_struct_format)
            shuffle_file_path = os.path.join(self.shuffling_path, 'default.mdat') #Will be modified with more shuffle
            self._shuffle_file = open(shuffle_file_path, 'rb')

    def _reset_document_pointer(self):
        """Reset document index and file pointer"""
        self._seek_document_pointer(0)

    def _seek_document_pointer(self, index):
        """Go to document index."""
        self.document_index = index
        if hasattr(self, '_shuffle_file') and self._shuffle_file:
            self._shuffle_file.seek(index * self._shuffle_struct_size)

    def load_next_document(self, shuffled=True) -> None:
        """Load next document"""
        if shuffled and not self._get_manifest_attr('shuffled', False):
            raise MDatNotShuffled("Dataset is not shuffled. Call shuffle() first or set shuffled=False")
        
        if not hasattr(self, 'document_index'):
            self.document_index = 0
        
        # Handle distributed training - skip documents not for this rank
        if hasattr(self, 'dist') and self.dist:
            if self.document_index % self.world_size != self.rank_size:
                self.document_index += 1
                return self.load_next_document(shuffled)
        
        if shuffled:
            # Initialize shuffle file
            if not hasattr(self, '_shuffle_file') or self._shuffle_file is None:
                self._init_shuffle_file()
            
            # Read from current position
            data = self._shuffle_file.read(self._shuffle_struct_size)
            if not data: 
                raise StopIteration
            
            submdat_id, doc_id = struct.unpack(self._shuffle_struct_format, data)
            ds_map = self._get_manifest_attr('datasets_map', {})
            submdat_name = ds_map[str(submdat_id + 1)]  # Map is 1-indexed
            self.current_document = self.loaded_submdats[submdat_name][doc_id]
        else:
            # Sequential access
            ds_map = self._get_manifest_attr('datasets_map', {})
            current_pos = 0
            
            for ds_id in sorted(int(k) for k in ds_map.keys()):
                submdat_name = ds_map[str(ds_id)]
                if submdat_name in self.loaded_submdats:
                    submdat_len = len(self.loaded_submdats[submdat_name])
                    if self.document_index < current_pos + submdat_len:
                        doc_id = self.document_index - current_pos
                        self.current_document = self.loaded_submdats[submdat_name][doc_id]
                        break
                    current_pos += submdat_len
            else:
               raise StopIteration
        
        self.document_index += 1
    
    def get_current_document(self) -> Any:
        """Get current document."""
        return self.current_document
        
    def _set_distributed_training(self) -> None:
        """Setup distributed training config."""
        try:
            import torch.distributed as dist
            torch_dist_available=True
        except:
            torch_dist_available=False
        if torch_dist_available and dist.is_available() and dist.is_initialized(): # Dirty multi-gpu stuff, temporanea
            self.dist = True
            self.world_size = dist.get_world_size()
            self.rank_size = dist.get_rank()
        else:
            self.dist = False
            self.world_size = 1
            self.rank_size = 0
            
    def _populate_submdat(self) -> None:
        """Populate loaded submdats."""
        self.loaded_submdats = {}
        ds_map = self._get_manifest_attr('datasets_map', {})
        if not ds_map:
            return
        max_key = max(int(k) for k in ds_map.keys())
        self.total_documents=0
        for i in range(1, max_key + 1):
           submdat_name = ds_map.get(str(i))
           if submdat_name is None or submdat_name in self.skipped_submats:
               continue
           # Always construct SubMdat with parent reference so SubMdat can inherit things such as pretok strategies,readonly...
           self.loaded_submdats[submdat_name] = SubMdat.load_submdat(self, submdat_name)
           self.total_documents+=len(self.loaded_submdats[submdat_name])

    def register_new_view(self,view_dict):
        """
        How views work in Mdat.
        Each view must be initialized from a view descriptor, a python dictionary with the following fields:
        {
        "view_name":'example',
        "selection_criteria":['bytes'],['doc_numbers'],['doc_percentage'],
        "skipped_submdats":[... ex 'dataset_3'],
        "selection":{
            "dataset_1":2G, #2 Gigabytes
            "dataset_2":800M, #800 Megabytes
            "dataset_4":8G,
            }
        }
        
        
        """
        
        # 1. Check if the name is available
        if view_name in self._get_manifest_attr('ds_views', []):
            raise NameAlreadyUsed(f"View with name '{view_name}' already registered")
            
        # 2. Check if all the submdats are included in the view, either as skipped or as selected
        all_submdats=list(self.list_submdat())
        submdats_includes=list(selection.keys()).extend(skipped_submdats)
        assert all_submdats==submdats_includes # To be fixed logic and syntax

        # 3. Compute the number of documents in selected datasets to perform the new shuffling
        selected_submdats = list(selection.keys())
        total_docs_selected = sum(len(self[name]) for name in selected_submdats)        
        # 4. Initialize the accumulators to check when the target is reached
        accumulators=dict()
        for s in selection.keys():
            accumulators[s]=0
            
        # 5. Set the target to be reached for each dataset
        targets = {}
        for k, v in selection.items():
            if selection_criteria == 'bytes':
                suffix = v[-1].upper()
                num = float(v[:-1])
                if suffix == 'K':
                    mult = 1024
                elif suffix == 'M':
                    mult = 1024**2
                elif suffix == 'G':
                    mult = 1024**3
                elif suffix == 'T':
                    mult = 1024**4
                else:
                    raise ValueError(f"Invalid size suffix in {v}")
                targets[k] = int(num * mult)
            elif selection_criteria == 'doc_numbers':
                targets[k] = int(v)
            elif selection_criteria == 'doc_percentage':
                targets[k] = int(float(v) * len(self[k]))
            else:
                raise ValueError(f"Invalid selection_criteria: {selection_criteria}")

        
        # 6. Iterates the shuffle keeping pointers only until they fit into the target
        
        # 7. Save the shuffle
        
        # 8. Sync with the pretokenization strategies
        
        
        

        # Success, saving the manifest
            views_list = self._get_manifest_attr('ds_views', [])
            views_list.append(view_name)
            self._set_manifest_attr('ds_views', views_list)
            self.update_manifest()      
    def set_view(self, ds_view: Optional[Union[str, Dict[str, Any]]]) -> None:
        """
        Set dataset view.
        Args:
            ds_view: Dataset view name or dict
        """
        if ds_view is not None:
            views = self._get_manifest_attr('views', {})
            allowed_ds = set(self._get_manifest_attr('datasets_map', {}).values())
            
            if ds_view in views:
                # assert either skipped or selected in ds_view
                view_def = views[ds_view]
                if isinstance(view_def, dict) and 'skipped' in view_def.keys():
                    skipped = set(submdat for submdat in view_def['skipped'])
                    self.selected_ds = allowed_ds - skipped
                elif isinstance(view_def, dict) and 'selected' in view_def:
                    self.selected_ds = set(view_def['selected'])
                else:
                    # If view is present but not in expected format, default to allowed_ds
                    self.selected_ds = allowed_ds
            else:
                # If ds_view provided but not found in manifest, treat as explicit selection list if it is a dict
                if isinstance(ds_view, dict):
                    if 'skipped' in ds_view:
                        skipped = set(ds_view['skipped'])
                        self.selected_ds = allowed_ds - skipped
                    elif 'selected' in ds_view:
                        self.selected_ds = set(ds_view['selected'])
                    else:
                        self.selected_ds = allowed_ds
                else:
                    self.selected_ds = allowed_ds
        else:
            self.selected_ds=set(self._get_manifest_attr('datasets_map', {}).values())
            self.skipped_submats=[]
                         
    def add_submdat(self, submdat_name: str, compression_levels: Dict = {}, map_sizes: Dict = {},
            db_types: List[str] = ['meta','data'], data_type: str = 'text', data_key: str = 'text', hybrid_data: List = [], 
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
        ds_map = self._get_manifest_attr('datasets_map', {})
        if len(ds_map)==0:
            max_ds=0
        else:
            max_ds = int(max([x for x in ds_map.keys()]))
        if max_ds >= max_submdat:
            raise TooManySubmDats        
        # 1. Put into the map (use next integer index)
        ds_map[str(max_ds + 1)] = submdat_name
        # 1. Check if it exists and load/create it
        if submdat_name in self.list_submdat():
            raise NameAlreadyUsed
        self.loaded_submdats[submdat_name]=None # A placeholder to allow creation of submdat
        new = SubMdat.create_submdat(self, submdat_name=submdat_name, compression_levels=compression_levels,map_sizes=map_sizes,db_types=db_types,data_type=data_type,hybrid_data=hybrid_data)
        self._set_manifest_attr('datasets_map',ds_map)
        if self._get_manifest_attr('shuffled', False):
            if reshuffle:
                self.shuffle_mdat()
            if round_robin_insertion:
                self.round_robin_insert(new)
            if reshuffle_at_the_end:
                pass
        else:
                self.delete_shuffle()
        # 3. If until now there were no exception, time to update the manifest
        
        if not reshuffle_at_the_end:
            self.update_manifest()
        # End: load the submdat and returns the newly created SubMdat
        self.loaded_submdats[submdat_name]=new
        return new
    
    def list_submdat(self) -> List[str]:
        """List submdat names."""
        return list(self.loaded_submdats.keys())
        
    def get_submdat(self, submdat_name: str, percentage: float = 1) -> Any:
        """
        Get submdat by name.
        Args:
            submdat_name: Name of submdat
            percentage: Percentage to load
        """
        if submdat_name in self.loaded_submdats.keys():
            return self.loaded_submdats[submdat_name]
        else:
            raise SubmDatNotFound

    def dataset_exists(self) -> bool:
        """Check if dataset exists."""
        return os.path.exists(os.path.join(self.mdat_path, 'manifest.json'))
    
    def update_manifest(self) -> None:
        """Update manifest file."""
        if self.readonly:
            raise MDatIsReadOnly
        else:
            with open(os.path.join(self.mdat_path, 'manifest.json'), 'w') as m:
                m.write(json.dumps(self.manifest))
         
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
        strategies = self._get_manifest_attr('pretok_strategies', [])
        self._set_manifest_attr('pretok_strategies', [s for s in strategies if s != strategy_name])
        self.update_manifest()
        
    def register_strategy(self, strategy_dict: Dict[str, Any]) -> Optional[Any]:
        """
        Register new strategy.
        Args:
            strategy_name: Name of strategy
            strategy_dict: Strategy configuration
        """
        if self.readonly:
            raise MDatIsReadOnly
        strategy_name=strategy_dict['strategy_name']
        strategy_name = sanify_name(strategy_name)
        if strategy_name in self._get_manifest_attr('pretok_strategies', []):
            raise NameAlreadyUsed(f"Strategy '{strategy_name}' already registered")
        strategy=PretokenizationStrategy.from_dict(self.pretok_path,self.functions_path,strategy_name,strategy_dict)
        if strategy:
            strategy.save()
            strategies = self._get_manifest_attr('pretok_strategies', [])
            strategies.append(strategy_name)
            self._set_manifest_attr('pretok_strategies', strategies)
            self.pretok_strategies[strategy_name]=strategy
            self.update_manifest()
            return self.get_strategy(strategy_name)
        else:
            return None

    def _populate_strategies(self) -> None:
        """Populate pretok strategies."""
        for strategy_name in self._get_manifest_attr('pretok_strategies', []):
            self.pretok_strategies[strategy_name]=PretokenizationStrategy(self.pretok_path,self.functions_path,strategy_name)
    
    def get_strategy(self, strategy_name: str) -> Any:
        """
        Get strategy by name.
        Args:
            strategy_name: Strategy name
        """
        if strategy_name not in self.pretok_strategies.keys():
            raise StrategyNotRegistered(f"Strategy '{strategy_name}' not registered")    
        return self.pretok_strategies[strategy_name]
    def strategies(self):
        return self.list_strategies(self) #a shortcut
    def set_max_seq_len(self,max_seq_len: int):
        """
        A method to set the multiplier based on the max. sequence length required by the model
        If the required sequence length is not a multiple of strategy's base sequence length,
        an exception will be thrown.
        """
        if self.current_strategy is not None:
            base_chunk_size=self.current_strategy.chunk_size
            if base_chunk_size is not None:
                assert max_seq_len%base_chunk_size==0
                self.set_multiplier(max_seq_len//base_chunk_size)
                self.max_seq_len=max_seq_len
            else:
                print("MatformerDataset: The strategy {self.current_strategy.strategy_name} doesn't have a base chunk size.")
        else:
            print("MatformerDataset: Impossible to set a max sequence length: load a strategy first.")
    def set_multiplier(self,chunk_multiplier):
        if isinstance(chunk_multiplier,int):
            if chunk_multiplier>100:
                raise Exception
            self.chunk_multiplier=chunk_multiplier
            if self.current_strategy is not None:
                self.set_strategy(self.current_strategy.strategy_name)
            if self.current_iteration_modality is not None:
                self.set_iteration_modality(self.current_iteration_modality)
        else:
            raise TypeError
    def __iter__(self):
        return self
    def __next__(self):
        if self.current_iteration_modality=='document':
            self.load_next_document()
            return self.current_document
        elif self.current_iteration_modality=='tokens':
            self.load_next_document()
            return self.current_document
        elif self.current_iteration_modality=='chunked_tokens':
            # current chunk step contains the chunk we are taking in the current document
            if not hasattr(self, 'current_chunk_step'):
                self.current_chunk_step = 0
                self.current_document = None
            
            while True:
                # Load document if exhausted
                if (self.current_document is None or 
                    'chunked_tokens' not in self.current_document or
                    self.current_chunk_step >= len(self.current_document['chunked_tokens'])):
                    self.load_next_document()
                    self.current_chunk_step = 0
                
                chunks = self.current_document['chunked_tokens']
                end_step = min(self.current_chunk_step + self.chunk_multiplier, len(chunks))
                if end_step > self.current_chunk_step:
                    selected_chunks = chunks[self.current_chunk_step:end_step]
                    self.current_chunk_step = end_step
                    chunk=np.concatenate(selected_chunks).tolist()
                    if getattr(self,'max_seq_len',None) is not None:
                        if len(chunk) > self.max_seq_len:
                            print(f"WARNING: A sequence is longer than max length ({len(chunk)} > {self.max_seq_len}) and it's truncated")
                            chunk = chunk[:self.max_seq_len]
                    return chunk

    def set_iteration_modality(self,modality,with_meta=False, return_raw=False, add_special_tokens=True):
        supported_modalities=['document','tokens','chunked_tokens','strategy_default']
        if modality in supported_modalities:
            self.current_iteration_modality=modality
        else:
            print(f"{modality} not in {supported_modalities}")
        if modality=='document':
            wanted_from_strategy=[]
            wanted_from_dbs=['data']
            self.len=self.total_documents
        elif modality=='tokens':
            wanted_from_strategy=['tokens']
            wanted_from_dbs=[]
            self.len=self.total_documents
        elif modality=='chunked_tokens':
            wanted_from_strategy=['chunked_tokens']
            wanted_from_dbs=[]
            self.len=self.total_divided_chunks
        elif modality=='strategy_default':
            wanted_from_strategy=['strategy_default']
            wanted_from_dbs=[]
            self.len=0 # To be decided            
        else:
            raise Exception
        if with_meta:
            wanted_from_dbs.append('meta')
        # Set the iteration modality in each submdat
        for sbm in self.loaded_submdats.values():
            sbm.set_default_wanted(from_dbs=wanted_from_dbs,from_strategy=wanted_from_strategy,raw=return_raw, add_special_tokens=add_special_tokens)
        
    def get_iteration_modality(self):
        return self.current_iteration_modality
    def set_strategy(self,strategy):
        self.current_strategy=self.get_strategy(strategy)
        self.total_chunks=0
        self.total_tokens=0    
        self.total_divided_chunks=0    
        for sbm in self.loaded_submdats.values():
            try:
                sbm.set_strategy(strategy)
                self.total_chunks+=sbm.strategy_stats['total_chunks']
                self.total_tokens+=sbm.strategy_stats['total_tokens']
                self.total_divided_chunks+=sbm.strategy_stats['precomputed_lengths'][self.chunk_multiplier]
            except SubMdatMissesStrategy:
                print(f"Submdat {sbm.submdat_name} is not pretokenized with strategy {strategy}.")
                total_chunks=0
                total_tokens=0    
                total_divided_chunks=0   
                self.current_strategy=None 
            except Exception as e:
               print(e)
        # Get the stats from each submdat

        # Decide the len of the mdat according to the requested modality
    def __str__(self):
        stringa=""
        stringa+=("\n---- Matformer Dataset ----")
        for k in self.manifest.keys():
            stringa+=(f"\n{k}: {self.manifest[k]}")
        stringa+=(f"\n Current strategy: {self.current_strategy}")
        stringa+=(f"\n Total documents: {self.total_documents}")
        stringa+=(f"\n Total tokens: {self.total_tokens}")
        stringa+=(f"\n Total chunks: {self.total_chunks}")
        stringa+=(f"\n Total divided chunks: {self.total_divided_chunks}")
        stringa+=(f"\n Current iteration modality: {self.current_iteration_modality}")
        stringa+=("\n Loaded sumbdats: ")
        #for sbm in self.loaded_submdats.values():
        #    stringa+=(sbm.__str__())
        return stringa
    def list_strategies(self):
        """
        List registered strategies
        """
        return list(self.pretok_strategies.keys())
    def __len__(self):
        """
        The len behaves in different ways:
        1) If nothing is specified, it return the summed length of each submdat
        2) If a view is specified, it computes the length according to the view
        3) If a pretokenization strategy is set, it returns the number of chunks according to max_seq_len (necessary for model training)
        """
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



class SubMdat:
    def __init__(self, parent_mdat: 'MatformerDataset', submdat_name: str) -> None:
        """Initialize SubMdat instance."""
        self.current_strategy = None
        self.default_wanted_from_dbs = None
        self.default_wanted_from_strategy = None
        self.db = dict()
    
    def __str__(self):
        return self.submdat_name
    def common_initialization(self, parent_mdat: 'MatformerDataset', submdat_name: str) -> None:
        """Common initialization for both load and create."""
        if not isinstance(parent_mdat, MatformerDataset):
            raise Exception
        self.mdat = parent_mdat
        self.mdat_path = parent_mdat.mdat_path
        self.readonly = self.mdat.readonly
        self.submdat_name = submdat_name
        self.submdat_path = os.path.join(self.mdat.dataset_path, submdat_name)
        self.manifest_path = os.path.join(self.submdat_path, 'sub_manifest.json')  
    
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
        #if instance.submdat_name not in instance.mdat.list_submdat():
        #    raise SubmDatNotFound   
        if not os.path.isdir(instance.submdat_path): 
            raise FileNotFoundError(f"{instance.submdat_name} is registered but not existing in {instance.mdat_path}.")
        with open(instance.manifest_path, "r") as m:
            instance.manifest = json.loads(m.read())
        instance.len = instance.manifest['documents_number']
        for db_type in instance.manifest.get('db_types', []):
            db_path = os.path.join(instance.submdat_path, db_type) + '.dat'
            instance.db[db_type] = LMDBDataset(db_path, readonly=instance.readonly, compressed=(instance.manifest['compression_levels'][db_type] > 0), compression_level=instance.manifest['compression_levels'][db_type], map_size=instance.manifest['map_sizes'][db_type])
        return instance
            
    @classmethod
    def create_submdat(cls, parent_mdat: 'MatformerDataset', submdat_name: str, 
                      compression_levels: Dict[str, int], map_sizes: Dict[str, int], 
                      overwrite: bool = False, db_types: List[str] = ['meta','data'], 
                      data_type: str = 'text', data_key: str = 'text', 
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
                if overwrite:
                    import shutil
                    shutil.rmtree(instance.submdat_path)    
        # Check if there are compression levels and map sizes correct for the db types:
        for db_type in db_types:
            if db_type not in compression_levels.keys():
                raise Exception
            if db_type not in map_sizes.keys():
                raise Exception 
        # Create submdat directory
        os.makedirs(instance.submdat_path, exist_ok=True)
        # Create the Databases
        for db_type in db_types:
            instance.db[db_type] = LMDBDataset(os.path.join(instance.submdat_path, db_type) + '.dat', compressed=compression_levels[db_type]>0, readonly=False,
                compression_level=compression_levels[db_type], map_size=map_sizes[db_type])
        # Create the manifest
        instance.manifest = {
                "type": "sub-mdat",
                "name": submdat_name,
                "data_type": data_type,
                "data_key": data_key,
                "hybrid_data": hybrid_data,
                "raw_data_bytes": 0,  
                "raw_meta_bytes": 0,  
                "db_disk_bytes": 0, 
                "db_types": db_types,   
                "compression_levels": compression_levels,                      
                "map_sizes": map_sizes,  
                "documents_number": 0,
                "errors_counters": {},
                "pretokenization_strategies": [],
                "pretokenization_compression": {}
            }
        instance.write_manifest() 
        # Now load the created submdat
        return cls.load_submdat(parent_mdat, submdat_name)
               
    def set_default_wanted(self,from_dbs:list=['full'],from_strategy:list=[],raw:bool=False, add_special_tokens=True):
        """
        Set the default elements returned by the submat when called with __getitem__
        Args:
            from_dbs: the names of the db to be returned. "meta" is reserved for dictionaries, all the others are returned raw. The special key "full" will return all the databases.
            from_strategy: the names of the strategy's items to be returned, for example 'tokens' or 'chunks'. The special key "full" will return all the items produced by the pretokenization strategy.
        """
        self.default_wanted_from_dbs=from_dbs
        self.default_wanted_from_strategy=from_strategy
        self.default_raw=raw
        self.default_add_special_tokens=add_special_tokens
        
    def __getitem__(self,key):
        return self._compose_return(key=key,wanted_from_dbs=self.default_wanted_from_dbs,wanted_from_strategy=self.default_wanted_from_strategy,raw=self.default_raw)
    def _compose_return(self,key:int,wanted_from_dbs:list,wanted_from_strategy:list,raw:bool=False,add_special_tokens:bool=True,strategy_name=None,on_the_fly_mode=True,max_seq_len=None,decode_strings=True):
        composed={'submdat_name':self.submdat_name,'key':key}
        if wanted_from_dbs=='full':
            composed.update(orjson.loads(self.db['meta'][key]))
            composed.update({'data':self.db['data'][key]})
        if 'data' in wanted_from_dbs:
            if decode_strings:
                text=self.db['data'][key].decode('utf-8')
            else:
                text=self.db['data'][key]
            if not raw:
                composed.update({'data':text})
            else:
                return(text)
        if 'meta' in wanted_from_dbs:
            composed.update(orjson.loads(self.db['meta'][key]))
        if self.current_strategy is not None:
            if wanted_from_strategy is not None:
               composed.update(self.current_strategy(key=key,cache_dict=self.pretok_db,max_seq_len=max_seq_len,wanted_from_strategy=wanted_from_strategy,add_special_tokens=add_special_tokens))
        return composed
    def get_generator(self, progress_bar=False,wanted_from_dbs='full',wanted_from_strategy=None,raw=False,randomize=False,seed=27):
        """
        Returns a generator over submdat elements. 
        Args:
            progress_bar: enable/disable a tqdm progress bar
            from_dbs: the names of the db to be returned. "meta" is reserved for dictionaries, all the others are returned raw. The special key "full" will return all the databases.
            from_strategy: the names of the strategy's items to be returned, for example 'tokens' or 'chunks'. The special key "full" will return all the items produced by the pretokenization strategy.
        """
        if randomize:
            permutator=RandomPermutator(max_len=self.len,seed=seed)
        if progress_bar:
            from tqdm import tqdm
            for key in tqdm(range(self.len)):
                if randomize:
                    key=permutator(key)
                yield self._compose_return(key=key,wanted_from_dbs=wanted_from_dbs,wanted_from_strategy=wanted_from_strategy,raw=raw)
        else:
            for key in (range(self.len)):
                if randomize:
                    key=permutator(key)
                yield self._compose_return(key=key,wanted_from_dbs=wanted_from_dbs,wanted_from_strategy=wanted_from_strategy,raw=raw)                           

    def __len__(self):
        return self.len
    
    def __str__(self):
        print(f"Submdat: {self.manifest['name']}, Documents: {self.manifest['documents_number']}")
        if self.current_strategy is not None:
            print(f"Tokens: {self.strategy_stats['total_tokens']} Chunks: {self.strategy_stats['total_chunks']}")
    def get_manifest(self):  
        return self.manifest
    
    def write_manifest(self): 
        if self.readonly:
            raise MdatIsReadOnly
        with open(self.manifest_path, "w") as m:  
            return m.write(json.dumps(self.manifest)) 
  
    def set_strategy(self,strategy_name: str, readonly: bool=True):
        """
        Set the Pretokenization Strategy to be used in the Submdat
        """
        compressed=False #Temporary
        if strategy_name not in self.manifest['pretokenization_strategies']:
            raise SubMdatMissesStrategy
        self.current_strategy=self.mdat.get_strategy(strategy_name)            
        self._load_strategy_dbs(strategy_name, readonly=readonly)
        self._load_strategy_stats(strategy_name)
        
    def _load_strategy_dbs(self, strategy_name, readonly=True, compression_level=None, map_size=None, batch_size=None):
        pretok_path = os.path.join(self.mdat.pretok_path, strategy_name, self.submdat_name)
        os.makedirs(pretok_path,exist_ok=True)
        if compression_level is None:
            compression_level = self.manifest.get("pretokenization_compression", {}).get(strategy_name, 0)
        self.pretok_db = {}
        
        for db_name in self.current_strategy.returns:
            db_path = os.path.join(pretok_path, db_name + '.dat')
            if readonly and not os.path.exists(db_path):
                raise FileNotFoundError(f"Strategy database not found: {db_path}")
            
            self.pretok_db[db_name] = LMDBDataset(
                db_path, 
                compressed=compression_level > 0, 
                readonly=readonly,
                compression_level=compression_level, 
                map_size=map_size or (1<<44), 
                batch_size=batch_size or 50000
            )       
    def add_strategy_start(self, strategy_name:str, compression_level:int =0, map_size:int=1<<44, batch_size:int=50000):
        """
        Begin of the process of adding a strategy.
        It's important to call add_strategy_end after the Pretokenization process to be sure databases are closed, stats and manifest saved
        """
        if self.readonly:
            raise MdatIsReadOnly 
        self.current_strategy = self.mdat.get_strategy(strategy_name)
        self.manifest['pretokenization_strategies'].append(strategy_name)
        
        pretok_path = os.path.join(self.mdat.pretok_path, strategy_name, self.submdat_name)
        os.makedirs(pretok_path, exist_ok=True)
        
        if "pretokenization_compression" not in self.manifest:
            self.manifest["pretokenization_compression"] = {}
        self.manifest["pretokenization_compression"][strategy_name] = compression_level
        
        self._load_strategy_dbs(strategy_name, readonly=False, 
                               compression_level=compression_level, 
                               map_size=map_size, batch_size=batch_size)  
    def _load_strategy_stats(self,strategy_name):
        with open(os.path.join(self.mdat.pretok_path, strategy_name, self.submdat_name,'stats.json'), "r") as f:
            self.strategy_stats=json.load(f)
    def add_strategy_end(self,strategy_name,stats:dict[str,Any]):
        if self.readonly:
            raise MdatIsReadOnly
        for db in self.pretok_db.values():
            db.close()
        with open(os.path.join(self.mdat.pretok_path, strategy_name, self.submdat_name,'stats.json'), "w") as f:
            f.write(json.dumps(stats))
        self.write_manifest()   
      
    def get_current_strategy(self):
        return self.current_strategy
        
    def _prepare_chunks_for_pretokenization(self, chunks_tuples,chunks_dtype,max_tokens_per_chunk=None, tokens_length=None, strict_checks=False):
        """
        A private function, to be called only by pretokenize_submat
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
                    assert max_size <= max_tokens_per_chunk
                return sizes.astype(chunks_dtype).tobytes(order='C')
        else:
                print(f"ERROR: The Splitter/Tokenizer returned {type(chunks)} as chunks instead of a list. This is unrecoverable")

    def pretokenize_submdat(self, strategy_name, strategy_dict=None, register_in_parent_mdat=True, 
                           progress_bar=True, compression_level=0,chunking_strict_checks=False, parallel=True, num_processes=None, batch_size=5000):

        max_multiplier=100     #Precompute 100 times the base sequence length   
        if self.readonly:
            raise MdatIsReadOnly       
        # 1. Check if the strategy is registered in the parent mdat
        try:
            strategy = self.mdat.get_strategy(strategy_name)
        except StrategyNotRegistered:
            if register_in_parent_mdat:
                strategy = self.mdat.register_strategy(strategy_name, strategy_dict)
            else:
                raise StrategyNotRegistered
        
        # 2. Check if the strategy registered in the parent mdat is the same given to the function
        if strategy_dict and strategy != strategy_dict:
            raise StrategyIsDifferent
        
        # 3. Start adding the strategy to the submdat
        self.add_strategy_start(strategy_name=strategy_name, compression_level=compression_level)

        # Initialize stats
        stats = {
            'total_tokens': 0,
            'max_tokens_per_doc': 0,
            'total_chunks': 0,
            'max_chunks_per_doc': 0,
            'processed_docs': 0,
            'precomputed_lengths': [0] * (int(max_multiplier)+1),
            'base_chunk_size': strategy.chunk_size
        }
        
        # 4. Initialize the splitter
        # 5. What does the splitter wants from Submdats'databases? [Default: raw_data]
        # 6. Initialize the generator

        generator = self.get_generator(
            progress_bar=progress_bar,
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

                def process_batch_and_write(batch_data,stats):
                    """Process a batch and write results to databases."""
                    sub_batches = [batch_data[i:i+batch_size] for i in range(0, len(batch_data), batch_size)]
                    
                    results = pool.starmap(process_documents_batch, [
                        (sub_batch, self.mdat.pretok_path, self.mdat.functions_path, strategy_name, 
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
                    
                    for key, doc in enumerate(generator):
                        batch.append((key, doc))
                        
                        if len(batch) >= batch_size * num_processes:
                            stats=process_batch_and_write(batch,stats)
                            batch = []
                    
                    # Process remaining items
                    if batch:
                        stats=process_batch_and_write(batch,stats)
        # E. Close the DB and update the manifest
        self.add_strategy_end(strategy_name=strategy_name,stats=stats)
            
            
    def convert_to_submdat(self, dataset_type, dataset_path, dataset_args={}, data_key='text', modality='text',logger=False,progress_bar=True, do_transform=False, do_filtering=False): 
        """
        A function to populate the submdat's databases with data coming from other datasets' formats
        """
        if self.readonly:
            raise MdatIsReadOnly   
        class PrintLogger:
            def warning(self, message):
                print("Warning: ",message)
            def error(self, message):
                print("Error: ", message)
            def info(self, message):
                print("Info: ", message)                
        logger_fn = logger if logger else PrintLogger()
        
        if not hasattr(self, 'db'):
            logger_fn.error(f"Databases are not loaded for submat {self.submat_name}. Are you sure Submat was created and loaded correctly?")
            return None
        
        #If there is orjson, better
        try:
            import orjson
        except:
            orjson=False
        # Instantiating the correct generator for the dataset_type
        """
        Example dataset_args: json_batch_read=False, files_recurse_in_folder=True, 
                              files_metadata_type='multiple_json', csv_has_header=False, csv_data_field=0,
                              hf_split='train', hf_subdataset=None,
        """
        if dataset_type == 'jsonl':
            #from datasets_iterators import JSONIterator
            generator_fn = JSONIterator(json_path=dataset_path, dataset_args=dataset_args, progress_bar=progress_bar, logger=logger)
        elif dataset_type == 'lmdb':
            #from datasets_iterators import LMDBIterator
            generator_fn = LMDBIterator(dataset_path, dataset_args,data_key, progress_bar=progress_bar,logger=logger)
        elif dataset_type == 'hf':
            generator_fn = HuggingFaceIterator(dataset_path, dataset_args,dataset_path, data_key, progress_bar=progress_bar,logger=logger)
        elif dataset_type == 'sqlite':
            return
        elif dataset_type == 'atlas':
            generator_fn = AtlasIterator(path=dataset_path, dataset_args=dataset_args, progress_bar=progress_bar, logger=logger)
        elif dataset_type == 'csv':
            return
        elif dataset_type == 'files':
            return
        else:
            error=f"Unsupported dataset type: {dataset_type}"
            logger_fn.error(error)
            return error
            
        # Initialize stats (error stats and raw size stats)
        errors_counters=dict()  
        errors_counters['hasDataError'] = 0 
        errors_counters['generatorReturnedNone']=0
        errors_counters['missingDataKey']=0
        n_filtered = 0
        raw_data_bytes=0
        raw_meta_bytes=0
        
        # Iterating over the generator
        for i, item in enumerate(generator_fn):
            if item is None:
                errors_counters['generatorReturnedNone']+=1
                continue
            # A transformer function can be specified by the user (useful, not implemented yet)
            if do_transform:
                # item = transformer_function(item)
                pass
            
            if data_key not in item:
                warning=f"Data key '{data_key}' not found in item {i}. Item has keys {item.keys()}"
                logger_fn.warning(warning)
                errors_counters['missingDataKey']+=1
                continue
            data = item[data_key]
            if isinstance(data,str):
                data=data.encode('utf-8')
            if not isinstance(data,bytes):
                logger_fn.warning(f"Data is of types {type(data)} but it should be either string or bytes")
            del item[data_key]
            
            # Data can be passed through filters for selection (ex. language identification, quality metrics...)
            filtered = False
            if do_filtering:
                 pass
                 logger_fn.warning("Filter functions not implemented")
            
            if filtered:
                continue
            
            returned_error = self.db['data'].write(data, key=i)  
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
                
            error = self.db['meta'].write(serialized, key=i)
            if error is not None:
                hasDataError += 1
            # Computing the size of the data just inserted
            raw_data_bytes+=len(data)
            raw_meta_bytes+=len(serialized)
            
        # Close the databases 
        documents_number=len(self.db['data'])
        self.db['data'].close()
        self.db['meta'].close()

        # Computing size on disk for the new submdat
        db_disk_bytes = 0
        for root, dirs, files in os.walk(self.submdat_path):
            for file in files:
                    db_disk_bytes += os.path.getsize(os.path.join(root, file))
                    
        partial_manifest={}
        partial_manifest['raw_data_bytes']=raw_data_bytes
        partial_manifest['raw_meta_bytes']=raw_meta_bytes
        partial_manifest['db_disk_bytes']=db_disk_bytes
        partial_manifest['documents_number']=documents_number
        partial_manifest['errors_counters']=errors_counters
        self.manifest.update(partial_manifest)
        self.write_manifest()
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
def process_documents_batch(sub_batch, pretoken_path, functions_path, strategy_name,
                           chunking_strict_checks, chunk_size, strategy_returns, max_multiplier):
    """
    Worker function that processes documents and returns organized data by database type.
    """
    strategy = PretokenizationStrategy(strategy_name=strategy_name, pretok_path=pretoken_path, functions_path=functions_path)
    
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
import zlib
import lmdb
import orjson
import pickle
import os

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
        self.zlib_attempts=100 #Tyring to avoid zlib bug
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
                data = zlib.compress(data, self.compression_level)
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
                    data = zlib.compress(data, self.compression_level)
                
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
                    return zlib.decompress(data)
                except:
                    safe_path = ''.join(c if c.isalnum() or c in '_-' else '_' for c in self.path)
                    print(f"LMDBDataset {self.path} WARNING: error loading data at index {key}.")
                    print("It seems there is a bug with zlib.")
                    print(f"First, let's try to load the data again {self.zlib_attempts} times.")
                    for i in range(1,self.zlib_attempts):
                        try:
                            x=zlib.decompress(data)
                            print(f"It worked at attemp: {i}")
                            with open(f"zlib_error_logs_{safe_path}.txt","a") as l:
                                l.write(f"Zlib error at {i}. Recovered after {i}/{self.zlib_attempts} attempts.\n")
                            return x
                        except:
                            pass
                    print(f"It didn't worked after {self.zlib_attempts}. Returning a dummy structure to avoid breaking training. Event logged in ./zlib_error_logs_{safe_path}.txt")
                    with open(f"zlib_error_logs_{safe_path}.txt","a") as l:
                        l.write(f"Zlib error at {i}. Not recovered after {self.zlib_attempts} attempts.\n")
                    return None
            else:
                return data

# Dataset's iterators for creating a new Submdat:

def JSONIterator(json_path,logger,dataset_args={},progress_bar=True):
   file_size = os.path.getsize(json_path)
   ProgressBar = tqdm if progress_bar else lambda *args, **kwargs: ToBeFixed() 
   with open(json_path, 'r') as f:
       with ProgressBar(total=file_size, unit='B', unit_scale=True, desc="Processing JSONL file...") as pbar:
           for i,row in enumerate(f):
               try:
                   if orjson: 
                       try:
                           data = orjson.loads(row)
                       except Exception as e:
                           logger.warning(f"Row: {i} orjson failed: {e}, falling back to json.")
                           data = json.loads(row)
                   else:
                       data = json.loads(row)
               except Exception as e:
                   logger.error(f"Row {i} Failed to parse JSON row: {e}")
                   data = None
               pbar.update(len(row.encode('utf-8')))
               yield data

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
    def __init__(self, pretok_path,functions_path, strategy_name: str):
        """
        Initialize a PretokenizationStrategy by loading from existing configuration.
        A strategy can be used:
            1) To pretokenize a submdat
            2) To retrieve the chunked/pretokenized data from the pretokenization databases
            3) To perform an on_the_fly tokenization/chunking
        """
        #self.mdat = mdat
        self.strategy_name = strategy_name
        self.mdat_pretok_path = pretok_path
        self.functions_path = functions_path
        #self.mdat=None #We don't need mdat anymore
        self.on_the_fly_warning = False
        self.on_the_fly_mode = True
        
        # Load strategy configuration
        self._load_configuration()
        self.initialized=False
        #self._initialize_components()

    @classmethod
    def from_dict(cls, pretok_path,functions_path, strategy_name: str, strategy_dict: Dict[str, Any]) -> 'PretokenizationStrategy':
        """
        Create a new PretokenizationStrategy from a dictionary configuration.
        """
        instance = cls.__new__(cls)
        #instance.mdat = mdat
        instance.strategy_name = strategy_name
        instance.mdat_pretok_path = pretok_path
        instance.functions_path = functions_path
        #instance.mdat=None #We don't need mdat anymore
        instance.on_the_fly_warning = False
        instance.on_the_fly_mode = True
        
        instance._create_from_dict(strategy_dict)
        instance.initialized=False
        #instance._initialize_components()
        return instance

    def _create_from_dict(self, strategy_dict: Dict[str, Any]):
        """Create configuration from dictionary."""
        required_keys = ['strategy_name', 'tokenizer_type', 'tokenizer_name', 
                        'splitter_class', 'splitter_init', 'modality',
                        'chunk_size', 'wants_from_db', 'returns']
        
        for k in required_keys:
            if k not in strategy_dict.keys():
                raise MissingStrategyKey(f"Missing required key: {k}")
                
        self.strategy_name = strategy_dict['strategy_name']
        self.tokenizer_type = strategy_dict['tokenizer_type']
        self.tokenizer_name = strategy_dict['tokenizer_name']
        self.bos_token_id = strategy_dict.get('bos_token_id',None)
        self.eos_token_id = strategy_dict.get('eos_token_id',None)
        self.mask_token_id = strategy_dict.get('mask_token_id',None) 
        self.tokenizer_args = strategy_dict.get('tokenizer_args', {})
        self.splitter_class = strategy_dict['splitter_class']
        self.splitter_init = strategy_dict['splitter_init']
        self.splitter_arguments = strategy_dict.get('splitter_arguments')
        self.chunk_size = strategy_dict['chunk_size']
        self.modality = strategy_dict['modality']
        self.wants_from_db = strategy_dict['wants_from_db']
        self.wants_raw = strategy_dict['wants_raw']
        self.returns = strategy_dict['returns']
        self.supports_batch=strategy_dict.get('supports_batch',False)        
        self.tokens_datatype=strategy_dict.get('tokens_datatype',None)
        self.chunks_datatype=strategy_dict.get('chunks_datatype',None)
    def _load_configuration(self):
        """Load strategy configuration from saved JSON file."""
        config_path = os.path.join(self.mdat_pretok_path, f'{self.strategy_name}.json')
        if not os.path.exists(config_path):
            raise StrategyNotFound(f"Strategy configuration not found: {config_path}")
            
        with open(config_path, 'r') as f:
            strategy_dict = json.load(f)
        
        self._create_from_dict(strategy_dict)

    def _initialize_components(self):
        """Initialize tokenizer and splitter components."""
        sys.path.append('../') #DIRTY stuff to load matformertokenizer
        from matformer.matformer_tokenizers import MatformerTokenizer
        self.tokenizer = MatformerTokenizer(
            tokenizer_type=self.tokenizer_type, 
            tokenizer_name=self.tokenizer_name, 
            tokenizer_args=self.tokenizer_args
        )
        
        # Determine token datatype based on vocab size
        if hasattr(self.tokenizer, 'vocab_size') and self.tokenizer.vocab_size is not None:
            if self.tokenizer.vocab_size <= 255:
                self.tokens_datatype = 'uint8'
            elif self.tokenizer.vocab_size <= 65535:
                self.tokens_datatype = 'uint16'
            else:
                self.tokens_datatype = 'uint32'
        else:
            self.tokens_datatype = getattr(self.tokenizer, 'return_type', 'uint32')
        
        # Determine chunks datatype based on chunk size
        if self.chunk_size <= 255:
            self.chunks_datatype = 'uint8'
        elif self.chunk_size <= 65535:
            self.chunks_datatype = 'uint16'
        else:
            self.chunks_datatype = 'uint32'
        
        # Load special tokens IDs Special tokens are not saved into the chunks (to easily allow chunks merging, but added on the fly)
        self.bos_token_id=self.tokenizer.bos_token_id
        self.eos_token_id=self.tokenizer.eos_token_id
        self.mask_token_id=self.tokenizer.mask_token_id
        
        # Initialize splitter
        
        """Initialize the splitter class with dynamic import capability."""
        splitter_cls = self._find_splitter_class(self.splitter_class)
        
        # Prepare initialization arguments
        init_args = self.splitter_init.copy() if self.splitter_init else {}
        init_args['chunk_size']=self.chunk_size - 2 #Reduce the max sequence length by two to allow special tokens
        init_args['tokenizer'] = self.tokenizer
        
        # Initialize splitter
        self.splitter = splitter_cls(**init_args)
        self.initialized=True
        self.save() #Saving in order to update token dtype and chunks dtype
        
    def _find_splitter_class(self, class_name: str):
        """Find splitter class in globals or import from functions directory."""
        # First check if it exists in current globals
        if class_name in globals():
            return globals()[class_name]
        
        # Try to import from functions directory
        sys.path.insert(0, self.functions_path)
        try:
            # Try to find the class in any Python file in the functions directory
            for filename in os.listdir(self.functions_path):
                if filename.endswith('.py') and not filename.startswith('__'):
                    module_name = filename[:-3]
                    try:
                        spec = importlib.util.spec_from_file_location(
                            module_name, 
                            os.path.join(self.functions_path, filename)
                        )
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        if hasattr(module, class_name):
                            return getattr(module, class_name)
                    except Exception:
                        continue
        finally:
            sys.path.remove(self.functions_path)
        
        raise SplitterClassNotFound(f"Splitter class '{class_name}' not found")

    def save(self):
        """Save strategy configuration to JSON file."""
        config = {
            'strategy_name': self.strategy_name,
            'tokenizer_type': self.tokenizer_type,
            'tokenizer_name': self.tokenizer_name,
            'tokenizer_args': self.tokenizer_args,
            'bos_token_id':getattr(self,'bos_token_id',None),
            'eos_token_id':getattr(self,'eos_token_id',None),
            'mask_token_id':getattr(self,'mask_token_id',None),       
            'splitter_class': self.splitter_class,
            'splitter_init': self.splitter_init,
            'splitter_arguments': self.splitter_arguments,
            'modality': self.modality,
            'chunk_size': self.chunk_size,
            'wants_from_db': self.wants_from_db,
            'wants_raw': self.wants_raw,
            'returns': self.returns,
            'tokens_datatype':getattr(self,'tokens_datatype',None),
            'chunks_datatype':getattr(self,'chunks_datatype',None),
            'supports_batch':getattr(self,'supports_batch',False)
        }
        
        os.makedirs(self.mdat_pretok_path, exist_ok=True)
        config_path = os.path.join(self.mdat_pretok_path, f'{self.strategy_name}.json')
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

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
                 max_seq_len: Optional[int] = None, wanted_from_strategy: List = [], add_special_tokens:bool=True) -> Dict[str, Any]:
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
            # Convert sizes back to tuples
            chunks = []
            start = 0
            for size in chunk_sizes:
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
from nltk.tokenize import PunktTokenizer

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
        flag = False
        new_aligned_spans = []

        for i, x in enumerate(aligned_spans):
            span_length = x[1] - x[0]

            if span_length > maxlen:
                flag = True
                new_aligned_spans.extend(aligned_spans[:i])

                midpoint = x[0] + span_length // 2
                newspans = [(x[0], midpoint), (midpoint, x[1])]
                new_aligned_spans.extend(newspans)

                new_aligned_spans.extend(aligned_spans[i + 1 :])
                break

        if flag:
            return self.trim_long_sequences(new_aligned_spans, maxlen)
        else:
            return aligned_spans

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

####### COMMAND LINE INTERFACE ##########

import sys
import os
import json
import csv
import shutil
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
#from matformer_dataset import MatformerDataset
try:
    import orjson
except Exception:
    orjson = None

def load_mdat(path, create=False):
    return MatformerDataset.load_dataset(path, create_if_not_existent=create)

def setup_logging(mdat_path, name, suppress_warnings=False):
    log_dir = os.path.join(mdat_path, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"mdat:{name}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR if suppress_warnings else logging.INFO)
    ch.setFormatter(fmt)
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(ch); logger.addHandler(fh)
    return logger

def cmd_add_strategy(args):
    mdat = load_mdat(args.mdat_path, create=False)
    if args.strategy_function is not None:
        shutil.copy(args.strategy_function, mdat.functions_path)
    with open(args.strategy_file, "r") as f:
        content = json.load(f) if orjson is None else orjson.loads(f.read())
    returned=mdat.register_strategy(content)
    print(f"Correctly added strategy {returned.strategy_name}")
    

def cmd_remove_strategy(args):
    mdat = load_mdat(args.mdat_path, create=False)
    mdat.deregister_strategy(args.strategy_name)

def cmd_list_strategies(args):
    mdat = load_mdat(args.mdat_path, create=False)
    for s in mdat.list_strategies():
        print(s)

def cmd_info(args):
    mdat = load_mdat(args.mdat_path, create_if_not_existent:=False)
    print(mdat)

def cmd_sub_info(args):
    mdat = load_mdat(args.mdat_path, create=False)
    print(mdat[args.submdat_name])

def cmd_pretokenize(args):
    mdat = load_mdat(args.mdat_path, create=False)
    if args.submdat is None:
        for sb in mdat.list_submdat():
            print(f"Pretokenizing {sb}...")
            mdat[sb].pretokenize_submdat(strategy_name=args.strategy_name, parallel=args.parallel, batch_size=args.batch_size,compression_level=args.compression_level)
    else:
        mdat[args.submdat].pretokenize_submdat(strategy_name=args.strategy_name, parallel=args.parallel, batch_size=args.batch_size, compression_level=args.compression_level)

def cmd_shuffle(args):
    mdat = load_mdat(args.mdat_path, create=False)
    mdat.delete_shuffle()
    mdat.shuffle()

def cmd_create(args):
    MatformerDataset.create_new(args.mdat_path)

def _create_submdat_once(input_path, output_path, name, dataset_type, compress_data, compress_meta, data_key, map_size, do_transform, do_filtering, custom_path, logger):
    mdat = MatformerDataset.load_dataset(path=output_path, create_if_not_existent=True)
    try:
        map_size=1 << (os.path.getsize(input_path) - 1).bit_length() # Optimal file size
        print(f"Auto determined optimal map size: {map_size}")
    except Exception as e:
        print(e)
    submdat = mdat.add_submdat(submdat_name=name,
                              compression_levels={"data": compress_data, "meta": compress_meta},
                              map_sizes={"data": map_size, "meta": map_size},
                              data_type="text", db_types=["meta", "data"])
    dataset_args = {}
    # dataset_type 'jsonl','csv','atlas','huggingface','sqlite','custom','lmdb'
    result = submdat.convert_to_submdat(dataset_type=dataset_type, dataset_path=input_path, dataset_args=dataset_args, data_key=data_key,
                                       modality="text", do_transform=do_transform, do_filtering=do_filtering, logger=logger, progress_bar=True)
    logger.info(f"Sub-MDAT {name} created in {output_path}")
    for k, v in result.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                logger.info(f"({k}) {k2}={v2}")
        else:
            logger.info(f"{k} = {v}")

def cmd_create_submdat(args):
    output_path = args.mdat_path  
    logger = setup_logging(output_path, args.name or "batch_create_submdat", suppress_warnings=args.suppress_warnings)
    if args.batch_csv:
        logger.info(f"Starting batch from {args.batch_csv}")
        with open(args.batch_csv, "r") as batch_file:
            reader = csv.reader(batch_file)
            def process_row(row):
                if len(row) >= 4:
                    input_p, output_p, name, dtype = row[:4]
                    logger.info(f"Processing {name}")
                    _create_submdat_once(input_p, output_p, name, dtype, args.compress_data, args.compress_meta, args.data_key, args.map_size, args.do_transform, args.do_filtering, args.custom, logger)
            with ThreadPoolExecutor(max_workers=4) as ex:
                list(ex.map(process_row, reader))
    else:
        if not args.input_path:
            raise SystemExit("INPUT_PATH is required unless --batch_csv is provided")
        if args.name is None:
            submdat_name=sanify_name(args.input_path.split('.')[-2].strip('/').strip('\\'))
            print("Submdat name: ",submdat_name)
        else:
            submdat_name=args.name
        _create_submdat_once(args.input_path, output_path, submdat_name, args.type, args.compress_data, args.compress_meta, args.data_key, args.map_size, args.do_transform, args.do_filtering, args.custom, logger)
    logger.info("create_submdat completed")

def build_parser():
    p = argparse.ArgumentParser(prog="mdat.py", description="MDAT management CLI")
    p.add_argument("mdat_path", help="Path to MDAT dataset (always required)")

    sp = p.add_subparsers(dest="command", required=True)

    s = sp.add_parser("add_strategy", help="add_strategy strategy_file.json strategy_function.py (optional)")
    s.add_argument("strategy_file")
    s.add_argument("--strategy_function",default=None)
    s.set_defaults(func=cmd_add_strategy)

    s = sp.add_parser("remove_strategy", help="remove_strategy strategy_name")
    s.add_argument("strategy_name")
    s.set_defaults(func=cmd_remove_strategy)

    s = sp.add_parser("list_strategies", help="list_strategies")
    s.set_defaults(func=cmd_list_strategies)

    s = sp.add_parser("info", help="print mdat info")
    s.set_defaults(func=cmd_info)

    s = sp.add_parser("sub", help="print submdat info")
    s.add_argument("--submdat")
    s.set_defaults(func=cmd_sub_info)

    s = sp.add_parser("pretokenize", help="pretokenize strategy_name (optional, if missing pretokenize everything)")
    s.add_argument("strategy_name")
    s.add_argument("--submdat", type=str, default=None)
    s.add_argument("--parallel", type=bool, default=True)
    s.add_argument("--batch_size", type=int, default=500)
    s.add_argument("--compression_level", type=int, default=0)
    s.set_defaults(func=cmd_pretokenize)

    s = sp.add_parser("shuffle", help="shuffle")
    s.set_defaults(func=cmd_shuffle)

    s = sp.add_parser("create", help="create empty MDAT")
    s.set_defaults(func=cmd_create)

    s = sp.add_parser("create_submdat", help="create_submdat INPUT_PATH --type TYPE --name NAME [--batch_csv CSV]")
    s.add_argument("input_path", nargs="?", help="Input dataset path (omit if using --batch_csv)")
    s.add_argument("--type", default='jsonl',required=False, choices=['jsonl', 'csv', 'atlas', 'huggingface', 'sqlite', 'custom', 'lmdb'], dest="type", help="Type of input dataset")
    s.add_argument("--name", required=False, help="Name for the sub-MDAT dataset")
    s.add_argument("--compress_data", type=int, default=0)
    s.add_argument("--compress_meta", type=int, default=0)
    s.add_argument("--data_key", default="text")
    s.add_argument("--modality",type=str,default="text")
    s.add_argument("--batch_csv", dest="batch_csv")
    s.add_argument("--suppress_warnings", action="store_true")
    s.add_argument("--custom", help="Path to custom iterator script")
    s.add_argument("--map_size", type=int, default=1 << 40)
    s.add_argument("--do_transform", action="store_true")
    s.add_argument("--do_filtering", action="store_true")
    s.set_defaults(func=cmd_create_submdat)

    return p

def main(argv):
    parser = build_parser()
    args = parser.parse_args(argv[1:])
    args.mdat_path = args.mdat_path
    args.func(args)

if __name__ == "__main__":
    main(sys.argv)
