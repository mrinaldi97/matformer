from mdat_utils import sanify_name
import os
import json
from lmdb_dataset import LMDBDataset
import orjson
# ===== Added: missing exception classes used in existing/added code (kept minimal) =====
class NameAlreadyUsed(Exception):
    pass

class StrategyNotRegistered(Exception):
    pass
# =======================================================================================

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

# ===== NEW: Allowed pretok db names and filenames (kept simple, referenced by SubMdat) ===
_PRETOK_ALLOWED_DBS = {
    'tokens': 'tokens.dat',
    'chunks': 'chunks.dat',
    'extra':  'extra.dat',
}
# =========================================================================================

class MatformerDataset:
    def __init__(self, path, shuffle=True, ds_view=None, distributed=False):
        self.mdat_path = path
        self.pretok_path=os.path.join(path,'pretok')
        self.dataset_path=os.path.join(path,'datasets')
        self.views_path=os.path.join(path,'views')
        self.readonly = True
        self.load_dataset(path)
        self.pretok_strategies=dict()
        if distributed:       
            self.set_distributed_training()
        self.set_view(ds_view)
        self.populate_submdat() # Now we have self.submdat['submdat_name']==SubMdat object in self.loaded_submdats
    
    @classmethod
    def load_readonly(cls, path, shuffle=True, ds_view=None):
        """Load existing dataset in readonly mode"""
        if not os.path.exists(os.path.join(path, 'manifest.json')):
            raise MDatNotFound
        return cls(path, shuffle=shuffle, ds_view=ds_view)
    
    @classmethod
    def create_or_update(cls, path, create=False, overwrite=True, dsmap_bits=8, shuffle=True, ds_view=None, tokenization_registry=None):
        """Create new dataset or update existing one"""
        instance = cls.__new__(cls)
        instance.mdat_path = path
        instance.readonly = False
        instance.pretok_strategies=dict()
        instance.pretok_path=os.path.join(path,'pretok')
        instance.dataset_path=os.path.join(path,'datasets')
        instance.views_path=os.path.join(path,'views')
        if create:
            instance.create_dataset(overwrite=overwrite, dsmap_bits=dsmap_bits)
        
        instance.load_dataset(path)            
        #if ds_view is not None:
        #    instance.set_view(ds_view)
        #instance.populate_submdat()
        return instance

    def __getitem__(self, submdat_name):
        # Direct access to submdat: mdat[submdat]
        if hasattr(self, 'loaded_submdats') and submdat_name in self.loaded_submdats:
            return self.loaded_submdats[submdat_name]
        raise SubmDatNotFound
    def load_next_document(self):
        """
        
        """
        pass
    
    def get_current_document(self):
        return self.current_document
        
            
    def set_distributed_training(self):
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
            
    def populate_submdat(self):
        self.loaded_submdats = {}
        ds_map = self.manifest.get('datasets_map', {})
        if not ds_map:
            return
        # keys in datasets_map are expected to be numeric strings: "1","2",...
        max_key = max(int(k) for k in ds_map.keys())
        for i in range(1, max_key + 1):
           submdat_name = ds_map.get(str(i))
           if submdat_name is None:
               continue
           # Always construct SubMdat with parent reference so SubMdat can inherit things such as pretok strategies
           self.loaded_submdats[submdat_name] = SubMdat(self, submdat_name)
            
    def set_view(self, ds_view):
        if ds_view is not None:
            views = self.manifest.get('views', {})
            allowed_ds = set(self.manifest.get('datasets_map', {}).values())
            
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
            self.selected_ds=set(self.manifest.get('datasets_map', {}).values())
                    
    def load_dataset(self, path):
        if self.dataset_exists():
            with open(os.path.join(self.mdat_path, 'manifest.json')) as m:
                self.manifest = json.loads(m.read())            
        else:
            raise MDatNotFound
    
    def create_dataset(self, overwrite, dsmap_bits):
        if self.dataset_exists():
            if not overwrite:
                raise MDatAlreadyPresent
            else:
                import shutil
                shutil.rmtree(self.mdat_path)
        # 1. Creating the folder(s)
        os.makedirs(self.mdat_path, exist_ok=True)
        #Where the submdats will live
        os.makedirs(os.path.join(self.mdat_path,'datasets'), exist_ok=True)
        #Where the shuffled pointers will live
        os.makedirs(os.path.join(self.mdat_path,'shuffling'), exist_ok=True)
        #Where the views will live
        os.makedirs(os.path.join(self.mdat_path,'views'), exist_ok=True)
        #Where the pretokenizations will live (updated: pretok root, strategies inside, then submdat folders)
        os.makedirs(os.path.join(self.mdat_path,'pretok'), exist_ok=True)
        
        # 2. Creating an empty manifest.json file
        self.manifest = {
            "type": "mdat",
            "version": "1.0",
            "dsmap_bits": dsmap_bits,
            "datasets_map": {},
            "views": [],
            "pretok_strategies":[],
            "shuffled": False,
            "isMultimodal": False,
        }
        # 3. Write in path/manifest.json
        self.update_manifest()
        
    def add_submdat(self, submdat_name, compression_levels,map_sizes,create=True, reshuffle=False, round_robin_insertion=False, reshuffle_at_the_end=False):
        # 0. Check if it fits in the mdat
        max_submdat = 255 if self.manifest.get('dsmap_bits', 8) == 8 else 65535
        ds_map = self.manifest.setdefault('datasets_map', {})
        if len(ds_map)==0:
            max_ds=0
        else:
            max_ds = int(max([x for x in ds_map.keys()]))
        if max_ds >= max_submdat:
            raise TooManySubmDats        
        # 1. Put into the map (use next integer index)
        ds_map[str(max_ds + 1)] = submdat_name
        # 1. Check if it exists and load/create it
        new = SubMdat(self, submdat_name, create=create, compression_levels=compression_levels,map_sizes=map_sizes)
        if self.manifest.get('shuffled', False):
            if reshuffle:
                self.shuffle_mdat()
            if round_robin_insertion:
                self.round_robin_insert(new)
            if reshuffle_at_the_end:
                pass
            else:
                raise MDatAlreadyShuffled
        # 3. If until now there were no exception, time to update the manifest
        if not reshuffle_at_the_end:
            self.update_manifest()
        # End: returns the newly created SubMdat
        return new
            
    def get_submdat(self, submdat_name, percentage=1):
        if submdat_name in self.manifest.get('datasets_map', {}).values():
            return SubMdat(self, submdat_name, create=False)
        else:
            raise SubmDatNotFound

    def dataset_exists(self):
        return os.path.exists(os.path.join(self.mdat_path, 'manifest.json'))
    def update_manifest(self):
        if self.readonly:
            raise MdatIsWriteOnly
        else:
            with open(os.path.join(self.mdat_path, 'manifest.json'), 'w') as m:
                m.write(json.dumps(self.manifest))
         
    # Methods for the pretokenization-registry:
    # In the manifest, names of pretokenizations strategies should be saved
    # Each strategy will live in ds_root/pretok/strategy_name
    # Each strategy will have its ds_root/pretok/strategy_name/strategy.json
    # Each pretokenization will live in ds_root/pretok/strategy_name/submdat_name/
    # Each pretokenization will have important stats in ds_root/pretok/strategy_name/submdat_name/
    def register_strategy(self, strategy_name, tokenizer_type, tokenizer_name, splitter_function, modality, tokens_datatype,db_names=['tokens','chunks','extra']):
        strategy_name = sanify_name(strategy_name)
        if strategy_name in self.manifest['pretok_strategies']:
            raise NameAlreadyUsed(f"Strategy '{strategy_name}' already registered")
        strategy={
            "strategy_name":strategy_name,
            "tokenizer_type":tokenizer_type,
            "tokenizer_name":tokenizer_name,
            "splitter_function":splitter_function,
            "modality":modality,
            "tokens_datatype":tokens_datatype,
            "db_names":db_names
        }  
        with open(os.path.join(os.path.join(self.mdat_path,'pretok'),'strategy.json'),'w') as s:
            s.write(json.dumps(strategy))
        self.manifest['pretok_strategies'].append(strategy_name)
        self.pretok_strategies[strategy_name]=strategy
        self.update_manifest()

    def get_strategy(self, strategy_name):
        if strategy_name not in self.manifest['pretok_strategies']:
            raise StrategyNotRegistered(f"Strategy '{name}' not registered")
        if strategy_name not in self.pretok_strategies.keys():
            with open(os.path.join(os.path.join(self.pretok_path,strategy_name),'strategy.json'),'r') as s:
                self.pretok_strategies[strategy_name]=json.loads(s.read())
        return self.pretok_strategies[strategy_name]
    

class SubMdat:
    def __init__(self, parent_mdat, submdat_name, create=False,compression_levels=None,map_sizes=None):
        self.db_types={
        'data':'data.dat',
        'meta':'meta.dat'
        }
        # Parent MDat reference (new design: SubMdat always receives parent)
        self.mdat = parent_mdat
        # Use parent's mdat_path
        self.mdat_path = parent_mdat.mdat_path
        self.submdat_name = submdat_name
        self.submdat_path = os.path.join(self.mdat.dataset_path,submdat_name)
        self.manifest_path = os.path.join(self.submdat_path, 'sub_manifest.json')  
        self.current_strategy=None
        self.default_wanted_from_dbs=None
        self.default_wanted_from_strategy=None
        if create:
            self.create_submdat(compression_levels=compression_levels,map_sizes=map_sizes)
        else:
            self.load_submdat()
    def set_default_wanted(self,from_dbs=None,from_strategy=None):
        self.default_wanted_from_dbs=from_dbs
        self.default_wanted_from_strategy=from_strategy
    def __getitem__(self,key):
        return self._compose_return(key=key,wanted_from_dbs=self.default_wanted_from_dbs,wanted_from_strategy=self.default_wanted_from_strategy)
    def _compose_return(self,key,wanted_from_dbs,wanted_from_strategy,raw=False):
        composed={'submdat_name':self.submdat_name,'key':key}
        if wanted_from_dbs=='full':
            composed.update(orjson.loads(self.db['meta'][key]))
            composed.update({'data':self.db['data'][key]})
        if wanted_from_dbs=='data':
            if not raw:
                composed.update({'data':self.db['data'][key]})
            else:
                return self.db['data'][key]
        if wanted_from_dbs=='meta':
            composed.update(orjson.loads(self.db['meta'][key]))
        if self.current_strategy is not None:
            if wanted_from_strategy=='tokens':
                if 'tokens' in current_strategy['db_names']:
                    pass
        return composed
    def get_generator(self, with_tqdm=False,wanted_from_dbs='full',wanted_from_strategy=None,raw=False):
        from tqdm import tqdm
        for key in tqdm(range(self.len)):
            yield self._compose_return(wanted_from_dbs=wanted_from_dbs,wanted_from_strategy=wanted_from_strategy,raw=raw)
    @classmethod
    def create_new(cls, parent_mdat, submdat_name, tokenization_registry=None, **manifest_params):
        return cls(parent_mdat, submdat_name, create=True,tokenization_registry=tokenization_registry, **manifest_params)
    
    
    def create_submdat(self,compression_levels,map_sizes,**manifest_params):
        # Check if submdat already exists
        if os.path.isdir(self.submdat_path):
                raise FileExistsError("SubMdat already present")
                #import shutil
                #shutil.rmtree(self.submdat_path)
        
        # Create submdat directory
        os.makedirs(self.submdat_path, exist_ok=True)
        
        # Create manifest if parameters provided
        if manifest_params:
            self.new_manifest(**manifest_params)
        else:
            self.manifest = None
        for db_type in self.db_types.keys():
             self.create_db(db_type,compression_level=compression_levels[db_type],map_size=map_sizes[db_type])
        
    
    def load_submdat(self):
        if not os.path.isdir(self.mdat_path): 
            raise MDatNotFound(f"{self.mdat_path} not existing.")
        
        if not os.path.isdir(self.submdat_path): 
            raise SubmDatNotFound(f"{self.submdat_name} not existing in {self.mdat_path}.")
        with open(self.manifest_path, "r") as m:
            self.manifest = json.loads(m.read())
  
        self.len=self.manifest['documents_number']
        self.db=dict()
        for db_type in self.db_types.keys():
            if db_type=='data':
                compressed = self.manifest.get('data_compression_level', 0) > 0
            else:
                compressed = self.manifest.get('meta_compression_level', 0) > 0
            self.db[db_type]=LMDBDataset(os.path.join(self.submdat_path,self.db_types[db_type]), compressed=compressed)
    def __len__(self):
        return self.len
    
    def create_db(self, db_type, compression_level=0, map_size=1<<31, batch_size=50000):
        db_file_name=self.db_types[db_type]
        path=os.path.join(self.submdat_path,db_file_name)
        if os.path.exists(path):
            raise FileExistsError
        compressed = compression_level > 0
        self.db = getattr(self, 'db', {})
        if not isinstance(self.db, dict):
            self.db = {}
        self.db[db_type]=LMDBDataset(path,readonly=False,compressed=compressed,compression_level=compression_level,map_size=map_size,batch_size=batch_size)
        return self.db[db_type]
        

    def get_manifest(self):  
        return self.manifest
    
    def write_manifest(self): 
        try:
            with open(self.manifest_path, "w") as m:  
                return m.write(json.dumps(self.manifest)) 
        except Exception as e:
            print(f"Caught exception {e} in writing the manifest {self.manifest_path}") 
            return None 
    
    def new_manifest(self, submdat_name, raw_data_bytes, raw_meta_bytes, db_disk_bytes,documents_number, data_type='text', data_key='text', hybrid_data=[]):
        if self.manifest is None: 
            self.manifest = {
                "type": "sub-mdat",
                "name": submdat_name,
                "data_type": data_type,
                "data_key": data_key,
                "hybrid_data": hybrid_data,
                "raw_data_bytes": raw_data_bytes,  
                "raw_meta_bytes": raw_meta_bytes,  
                "db_disk_bytes": db_disk_bytes,    
                "data_compression_level": self.db['data'].compression_level,                      
                "meta_compression_level": self.db['meta'].compression_level,  
                "map_size_data": self.db['data'].map_size,
                "map_size_meta":self.db['meta'].map_size,
                "documents_number": documents_number,
                "pretokenization_strategies":[]
            }
            self.write_manifest() 
        else:
            print("Can't write a new manifest: the manifest already exists")
        
    
    
    def set_strategy(self,strategy_name,readonly=True):
        compressed=False #Temporary
        if strategy_name not in self.mdat.manifest['pretokenization_strategies']:
            raise StrategyNotRegistered
        if strategy_name not in self.manifest['pretokenization_strategies']:
            raise SubMdatMissesStrategy
        if not os.path.exists(db_path):
            raise FileNotFoundError(db_path)
        self.current_strategy=self.mdat.get_strategy(strategy_name)     
        self.load_strategy_db(readonly=True, compression_level=compressed)
        
    def load_strategy_db(self, strategy_name, readonly=True, compression_level=None, map_size=None, batch_size=None):
        for db_name in self.current_strategy['db_names']:
            self.pretok_db[db_name]=LMDBDataset(os.path.join(self.mdat.pretok_path,self.submdat_name,db_name+'.dat'), readonly=readonly,compression_level=compression_level, map_size=map_size, batch_size=batch_size)
            
    def add_strategy_start(self, strategy_name, compression_level=0, map_size=1<<31, batch_size=50000):
        if strategy_name not in self.mdat.manifest['pretokenization_strategies']:
            raise StrategyNotRegistered
        self.manifest['pretokenization_strategies'].append(strategy_name)       
        self.load_strategy_db(readonly=False, compression_level=compression_level, map_size=map_size,batch_size=batch_size)
    def add_strategy_end(self,stats):
        for db in self.pretok_db.values():
            db.close()
        with open(os.join(self.mdat.pretok_path,self.submdat_name,'stats.json')) as f:
            f.write(json.dumps(stats))
        self.write_manifest()   
    def current_strategy(self):
        return self.current_strategy

        

