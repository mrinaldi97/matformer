docstring = """ This script is temporary, but it can be useful before training a large model using Matformer. It will clone an mdat with the cloned mdat being filtered for a certain amount of minimum tokens per document. This is important when the original dataset may contain very short examples that can hinder training abilities. It is a quick&dirty script, this pipeline should definitely be included in the mdat.py file (for example as a filter during the subdmat creation process) to avoid extra-processing and in particular cloning an entire dataset. But it is useful as a fix. """

### CONSTANTS
COMPRESSED = True
MAP_SIZE = 1 << 41  # 1TB
MIN_TOKENS = 20  # Minimum: 20 tokens per document
COMPRESSION_LEVEL = 9
DEBUG=False
DEBUG_DOCUMENTS=1000
BATCH_SIZE = 100000
STRATEGY = "Gettone1024_"
CHUNKS_DTYPE = 'uint16'
### END OF CONSTANTS

print(docstring)

import sys
sys.path.append('../')
from matformer.mdat import LMDBDataset, DatabaseManager
import zlib
import json
import os
import numpy as np
import math
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool

""" It opens directly the LMDB files of the requested submdat """
mdat_path = sys.argv[1]
submdats_arg = sys.argv[2]  # Comma-separated list of submdats
submdats = [s.strip() for s in submdats_arg.split(',')]


def get_dbs(mdat_path, submdat, readonly=False):
    tokens_path = os.path.join(mdat_path, 'pretok', STRATEGY, submdat, 'tokens.dat')
    chunks_path = os.path.join(mdat_path, 'pretok', STRATEGY, submdat, 'chunks.dat')
    data_path = os.path.join(mdat_path, 'datasets', submdat, 'data.dat')
    meta_path = os.path.join(mdat_path, 'datasets', submdat, 'meta.dat')
    
    if readonly == False:
        os.makedirs(os.path.join(mdat_path, 'pretok', STRATEGY, submdat), exist_ok=True)
        os.makedirs(os.path.join(mdat_path, 'datasets', submdat), exist_ok=True)
    
    db_dict = {
        'tokens': LMDBDataset(tokens_path, readonly=readonly, compressed=COMPRESSED, compression_level=COMPRESSION_LEVEL, map_size=MAP_SIZE),
        'chunks': LMDBDataset(chunks_path, readonly=readonly, compressed=COMPRESSED, compression_level=COMPRESSION_LEVEL, map_size=MAP_SIZE),
        'data': LMDBDataset(data_path, readonly=readonly, compressed=COMPRESSED, compression_level=COMPRESSION_LEVEL, map_size=MAP_SIZE),
        'meta': LMDBDataset(meta_path, readonly=readonly, compressed=COMPRESSED, compression_level=COMPRESSION_LEVEL, map_size=MAP_SIZE)
    }
    return db_dict


def parse_chunks(stored_bytes):
    chunk_sizes = np.frombuffer(stored_bytes, dtype=CHUNKS_DTYPE)
    chunks = []
    start = 0
    for size in chunk_sizes:
        size = int(size)
        chunks.append((start, start + size))
        start += size
    try:
          return chunks[-1][1], len(chunks)
    except:
          return 0,0

def process_submdat(submdat):
    worker_position = submdats.index(submdat)
    print(f"\n[{submdat}] processing...")
    
    max_multiplier = 100
    
    pretok_stats = {
        'total_tokens': 0,
        'max_tokens_per_doc': 0,
        'total_chunks': 0,
        'max_chunks_per_doc': 0,
        'processed_docs': 0,
        'precomputed_lengths': [0] * (int(max_multiplier) + 1),
    }
    
    source_db = get_dbs(mdat_path, submdat, readonly=True)
    target_db = get_dbs(mdat_path + '_filtered', submdat, readonly=False)
    
    # Iteration over all the documents 
    if not DEBUG:
        tot_documents = len(source_db['data'])
    else:
        tot_documents = DEBUG_DOCUMENTS
    
    target_index = 0
    discarded_items = 0
    raw_data_bytes = 0
    raw_meta_bytes = 0
    
    for batch_start in tqdm(range(0, tot_documents, BATCH_SIZE), desc=f"[{submdat}]", position=worker_position, leave=True):
        batch_end = min(batch_start + BATCH_SIZE, tot_documents)
        
        batch_to_copy = []
        
        for source_index in range(batch_start, batch_end):
            _object = source_db['chunks'][source_index]
            document_tokens, document_chunks = parse_chunks(_object)
            if document_tokens==0 and document_chunks==0:
                #Error
                with open(f"filtering_error_report_{submdat}.txt","a") as f:
                    f.write(f"{source_index}\n")
            if document_tokens >= MIN_TOKENS:
                batch_to_copy.append({
                    'source_index': source_index,
                    'target_index': target_index,
                    'chunks_object': _object,
                    'document_tokens': document_tokens,
                    'document_chunks': document_chunks
                })
                target_index += 1
            else:
                discarded_items += 1
        
        if batch_to_copy:
            data_batch = []
            tokens_batch = []
            meta_batch = []
            chunks_batch = []
            
            for item in batch_to_copy:
                src_idx = item['source_index']
                tgt_idx = item['target_index']
                
                data_obj = source_db['data'][src_idx]
                tokens_obj = source_db['tokens'][src_idx]
                meta_obj = source_db['meta'][src_idx]
                chunks_obj = item['chunks_object']
                
                data_batch.append((tgt_idx, data_obj))
                tokens_batch.append((tgt_idx, tokens_obj))
                meta_batch.append((tgt_idx, meta_obj))
                chunks_batch.append((tgt_idx, chunks_obj))
                
                pretok_stats['total_tokens'] += item['document_tokens']
                pretok_stats['total_chunks'] += item['document_chunks']
                pretok_stats['processed_docs'] += 1
                pretok_stats['max_tokens_per_doc'] = max(pretok_stats['max_tokens_per_doc'], item['document_tokens'])
                pretok_stats['max_chunks_per_doc'] = max(pretok_stats['max_chunks_per_doc'], item['document_chunks'])
                
                for j in range(1, max_multiplier + 1):
                    pretok_stats['precomputed_lengths'][j] += math.ceil(item['document_chunks'] / j)
                
                raw_data_bytes += len(data_obj)
                raw_meta_bytes += len(meta_obj)
            
            target_db['data'].write_batch_with_keys(data_batch)
            target_db['tokens'].write_batch_with_keys(tokens_batch)
            target_db['meta'].write_batch_with_keys(meta_batch)
            target_db['chunks'].write_batch_with_keys(chunks_batch)
    
    for db_name in ['data', 'tokens', 'meta', 'chunks']:
        source_db[db_name].close()
        target_db[db_name].close()
    
    data_disk_size = os.path.getsize(os.path.join(mdat_path + '_filtered', 'datasets', submdat, 'data.dat'))
    meta_disk_size = os.path.getsize(os.path.join(mdat_path + '_filtered', 'datasets', submdat, 'meta.dat'))
    tokens_disk_size = os.path.getsize(os.path.join(mdat_path + '_filtered', 'pretok', STRATEGY, submdat, 'tokens.dat'))
    chunks_disk_size = os.path.getsize(os.path.join(mdat_path + '_filtered', 'pretok', STRATEGY, submdat, 'chunks.dat'))
    
    print(f"[Worker {submdat}] Processing complete!")
    print(f"[Worker {submdat}] Processed documents: {target_index}")
    print(f"[Worker {submdat}] Discarded items: {discarded_items}")
    print(f"[Worker {submdat}] Total tokens: {pretok_stats['total_tokens']}")
    
    return {
        'submdat': submdat,
        'pretok_stats': pretok_stats,
        'target_index': target_index,
        'discarded_items': discarded_items,
        'raw_data_bytes': raw_data_bytes,
        'raw_meta_bytes': raw_meta_bytes,
        'data_disk_size': data_disk_size,
        'meta_disk_size': meta_disk_size,
        'tokens_disk_size': tokens_disk_size,
        'chunks_disk_size': chunks_disk_size
    }


if __name__ == '__main__':    
    os.makedirs(os.path.join(mdat_path+'_filtered'), exist_ok=True)
    target_db_manager = DatabaseManager(os.path.join(mdat_path + '_filtered', 'mdat.db'))
    target_db_manager.connect()
    target_db_manager._initialize_db()
    target_db_manager.add_view('default')
    source_db_manager = DatabaseManager(os.path.join(mdat_path, 'mdat.db'))
    source_db_manager.connect()
    
    # Copy manifest
    source_manifest = source_db_manager.get_manifest()
    if source_manifest:
        target_db_manager.create_manifest(
            dsmap_bits=source_manifest['dsmap_bits'],
            modalities=json.loads(source_manifest['modalities'])
        )
    
    # Copy pretok_strategy
    source_strategy = source_db_manager.get_manifest(strategy=STRATEGY)
    print("Inserting strategy: ")
    print(source_strategy)
    target_db_manager._insert("pretok_strategy", source_strategy)
    
    # Pre-create all submdats and get their IDs
    submdat_id_map = {}
    for submdat in submdats:
        source_submdat_info = source_db_manager.get_manifest(submdat=submdat)
        target_submdat_id = target_db_manager.add_submdat(submdat)
        submdat_id_map[submdat] = target_submdat_id
        print(f"Pre-created submdat '{submdat}' with ID: {target_submdat_id}")
    
    print(f"\n=== PHASE 2: Processing {len(submdats)} submdats in parallel ===")
    
    #Process submdats in parallel
    num_workers = min(len(submdats), mp.cpu_count())
    print(f"Using {num_workers} parallel workers")
    
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_submdat, submdats)
    
    print("\n=== PHASE 3: Updating database with results (AFTER multiprocessing) ===")
    
    #Update database with all results
    for result in results:
        submdat = result['submdat']
        target_submdat_id = submdat_id_map[submdat]
        pretok_stats = result['pretok_stats']
        
        print(f"\nUpdating database for submdat '{submdat}'...")
        
        # Add database references and link them
        data_db_id = target_db_manager.add_database(
            _type='LMDB',
            compression_level=COMPRESSION_LEVEL,
            map_size=MAP_SIZE,
            disk_size=result['data_disk_size'],
            extra_data=None
        )
        
        meta_db_id = target_db_manager.add_database(
            _type='LMDB',
            compression_level=COMPRESSION_LEVEL,
            map_size=MAP_SIZE,
            disk_size=result['meta_disk_size'],
            extra_data=None
        )
        
        tokens_db_id = target_db_manager.add_database(
            _type='LMDB',
            compression_level=COMPRESSION_LEVEL,
            map_size=MAP_SIZE,
            disk_size=result['tokens_disk_size'],
            extra_data=None
        )
        
        chunks_db_id = target_db_manager.add_database(
            _type='LMDB',
            compression_level=COMPRESSION_LEVEL,
            map_size=MAP_SIZE,
            disk_size=result['chunks_disk_size'],
            extra_data=None
        )
        
        # Link submdat databases
        target_db_manager.link_submdat_database(
            submdat_id=target_submdat_id,
            database_id=data_db_id,
            raw_data_bytes=result['raw_data_bytes'],
            is_data=True,
            is_meta=False,
            is_extra=False
        )
        
        target_db_manager.link_submdat_database(
            submdat_id=target_submdat_id,
            database_id=meta_db_id,
            raw_data_bytes=result['raw_meta_bytes'],
            is_data=False,
            is_meta=True,
            is_extra=False
        )
        
        # Update submdat with document number
        target_db_manager.update_manifest(
            submdat=target_submdat_id,
            data={'document_number': result['target_index']}
        )
        
        # Add strategy-submdat relationship
        target_db_manager.add_strategy_submdat(
            strategy_name=STRATEGY,
            submdat_id=target_submdat_id,
            data={
                'total_tokens': pretok_stats['total_tokens'],
                'max_tokens_per_doc': pretok_stats['max_tokens_per_doc'],
                'total_chunks': pretok_stats['total_chunks'],
                'max_chunks_per_doc': pretok_stats['max_chunks_per_doc'],
                'processed_docs': pretok_stats['processed_docs']
            }
        )
        
        # Add strategy databases
        target_db_manager.add_strategy_db(
            database_id=tokens_db_id,
            strategy_name=STRATEGY,
            submdat_id=target_submdat_id,
            datatype='uint16',
            is_tokens=1,
            is_chunks=0,
            is_extra=None,
            is_complete=1
        )
        
        target_db_manager.add_strategy_db(
            database_id=chunks_db_id,
            strategy_name=STRATEGY,
            submdat_id=target_submdat_id,
            datatype=CHUNKS_DTYPE,
            is_tokens=0,
            is_chunks=1,
            is_extra=None,
            is_complete=1
        )
        
        # Store precomputed lengths
        max_multiplier = 100
        for multiplier in range(1, max_multiplier + 1):
            target_db_manager.set_strategy_submdat_precomp(
                strategy_name=STRATEGY,
                submdat_id=target_submdat_id,
                multiplier=multiplier,
                precomputed_length=pretok_stats['precomputed_lengths'][multiplier]
            )
        
        # Link to default view
        target_db_manager.link_submdat_view(
            submdat_id=target_submdat_id,
            view_name='default',
            is_skipped=0,
            is_partial=0,
            is_preserved=1,
            document_number=result['target_index'],
            bytes_criteria=result['raw_data_bytes']
        )
        
        # Write stats file
        with open(os.path.join(mdat_path + '_filtered', f'{submdat}_new_stats.txt'), "w") as statfile:
            statfile.write(json.dumps(pretok_stats))
            statfile.write('\n')
            statfile.write('Raw data bytes: ')
            statfile.write(str(result['raw_data_bytes']))
            statfile.write('\n Processed documents: ')
            statfile.write(str(result['target_index']))
            statfile.write('\n Discarded items: ')
            statfile.write(str(result['discarded_items']))
    
    print("\n" + "="*60)
    print("="*60)
    for result in results:
        print(f"\n{result['submdat']}:")
        print(f"  Processed documents: {result['target_index']}")
        print(f"  Discarded items: {result['discarded_items']}")
        print(f"  Total tokens: {result['pretok_stats']['total_tokens']}")
    print("="*60)
