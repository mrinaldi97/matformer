"""One named dataset domain, its source records, and pretokenized caches."""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
from datetime import datetime
from typing import Any, Dict, List, TYPE_CHECKING

import numpy as np
import orjson
from tqdm import tqdm

from .exceptions import (
    MDatIsReadOnly, MdatIsReadOnly, SubMdatAlreadyExists,
    SubMdatMissesStrategy, SubmDatNotFound,
)
from .lmdb import LMDBDataset
from .preprocessing import PretokenizationStrategy, process_documents_batch
from .utils import RandomPermutator

if TYPE_CHECKING:
    from .mdat import MatformerDataset

class SubMdat:
    def __init__(self, parent_mdat: 'MatformerDataset', submdat_name: str) -> None:
        """Initialize SubMdat instance."""
        self.current_strategy = None
        self.default_wanted_from_dbs = None
        self.default_wanted_from_strategy = None
        self.storage_db = dict() #Dictionary containing the pointers to the databases with the actual storage
        self.default_map_size=1<<33 #WARNING

    def common_initialization(self, parent_mdat: 'MatformerDataset', submdat_name: str) -> None:
        """Common initialization for both load and create."""
        from .mdat import MatformerDataset

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

    def close(self):
        for name in ('storage_db', 'pretok_db'):
            for database in getattr(self, name, {}).values():
                database.close()
            setattr(self, name, {})
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
                    # Small writable LMDBs need spare pages for metadata commits.
                    return max(self.default_map_size, math.ceil(disk_size * 1.2))

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
            if not self.readonly:
                os.makedirs(pretok_path, exist_ok=True)
            for database in getattr(self, 'pretok_db', {}).values():
                database.close()
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
                db_pointer, db_id = instance._load_storage_db(_type='storage', compression_level=compression_levels.get(db_type, 0), disk_size=None, extra_data=None, db_type=db_type, create=True)
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

        # Close creation handles before reopening the same LMDB paths.
        instance.close()
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
        return self._compose_return(key=key, wanted_from_dbs=self.default_wanted_from_dbs,
                                    wanted_from_strategy=self.default_wanted_from_strategy,
                                    raw=self.default_raw, add_special_tokens=self.default_add_special_tokens)

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
        return f"Submdat: {self.submdat_name}, Documents: {len(self)}"

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
            generator_fn = LMDBIterator(dataset_path, dataset_args=dataset_args, progress_bar=progress_bar, logger=logger_fn)
        elif dataset_type == 'hf':
            generator_fn = HuggingFaceIterator(dataset_path=dataset_path, logger=logger_fn,
                dataset_args=dataset_args, dataset_config=dataset_config,
                dataset_split=dataset_args.get('hf_split', 'train'), progress_bar=progress_bar, data_key=data_key)
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
                logger_fn.warning("Data is smaller than 10 bytes. Skipping")
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


def HuggingFaceIterator(dataset_path, logger, dataset_args=None, dataset_config='',
                        dataset_split='train', progress_bar=True, data_key=None):
    """Read the requested Hugging Face split, with optional load_dataset kwargs."""
    from datasets import load_dataset
    options = dict(dataset_args or {})
    split = options.pop('hf_split', dataset_split)
    config = options.pop('hf_subdataset', dataset_config) or None
    options.pop('split', None)
    ds = load_dataset(dataset_path, config, split=split, **options)
    yield from tqdm(ds, disable=not progress_bar)


def JSONIterator(json_path, logger, dataset_args=None, progress_bar=True,
                 batch_size_bytes=16 * 1024 * 1024):
    """Stream JSONL records with bounded buffering and optional progress output."""
    with open(json_path, 'rb', buffering=batch_size_bytes) as stream:
        with tqdm(total=os.path.getsize(json_path), unit='B', unit_scale=True,
                  desc='Reading JSONL', disable=not progress_bar) as progress:
            for index, row in enumerate(stream):
                progress.update(len(row))
                try:
                    yield orjson.loads(row)
                except (ValueError, TypeError) as error:
                    logger.error(f'Row {index}: invalid JSON: {error}')


def AtlasIterator(path, logger, dataset_args=None, progress_bar=True):
    from torch_atlas_ds import AtlasDataset
    db = AtlasDataset(path)
    for index in tqdm(range(len(db)), disable=not progress_bar):
        try:
            yield db[index]
        except Exception as error:
            logger.error(f'Error reading Atlas item {index}: {error}')
            yield None


def LMDBIterator(lmdb_path, logger, dataset_args=None, progress_bar=True):
    db = LMDBDataset(lmdb_path)
    try:
        for index in tqdm(range(len(db)), disable=not progress_bar):
            yield orjson.loads(db[index])
    finally:
        db.close()
