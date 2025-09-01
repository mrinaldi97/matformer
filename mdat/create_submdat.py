#!/usr/bin/env python3

"""
create_submdat.py
A script to create a sub-MDAT dataset ready to be included into an MDAT dataset.
Usage:
   python3 create_submdat.py INPUT_PATH OUTPUT_PATH --type {'jsonl','csv','atlas','huggingface:HF_SPLIT','sqlite:DB_TABLE','custom:(arguments)'} --name SBMDAT_NAME --compress_data N --compress_meta N
       --data_key='text' --batch_csv BATCH_CSV_PATH --suppress_warnings
   if "custom" is selected, the script expects --custom ITERATOR_PATH where ITERATOR_PATH is the path of a python script exposing a generator function.
   --data_key, if omitted is "text", points to the key of the dictionary representing data
   --compress_data N compress data with N compression level
   --compress_meta N compress meta with N compression level
   --map_size (default 1<<40)
   
   With batch_csv option it is possible to proceed to convert many sub-MDAT with a single command. The batch csv file will have this structure:
   
   input_path,output_path,name,type
   
   by default, multi-threading will be enabled.
   
   It is still necessary to provide eventual rules about compression and custom generator script by CLI. They will be applied to the entire batch.
   
"""
import os
import json
import csv
import argparse
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from lmdb_dataset import LMDBDataset
from matformer_dataset import MatformerDataset, SubMdat

try:
   import orjson
except ImportError:
   orjson = None



def parse_arguments():
   parser = argparse.ArgumentParser(description='Create a sub-MDAT dataset')
   parser.add_argument('input_path', help='Input path to the dataset')
   parser.add_argument('output_path', help='Path of an MDAT dataset (if not existent it will be created)')
   parser.add_argument('--type', required=True, 
                      choices=['jsonl', 'csv', 'atlas', 'huggingface', 'sqlite', 'custom', 'lmdb'],
                      help='Type of input dataset')
   parser.add_argument('--name', required=True, help='Name for the sub-MDAT dataset')
   parser.add_argument('--compress_data', type=int, default=0, help='Data compression level')
   parser.add_argument('--compress_meta', type=int, default=0, help='Metadata compression level')
   parser.add_argument('--data_key', default='text', help='Key for data in dictionary (default: text)')
   parser.add_argument('--batch_csv', help='Path to batch CSV file for processing multiple datasets')
   parser.add_argument('--suppress_warnings', action='store_true', help='Suppress warnings')
   parser.add_argument('--custom', help='Path to custom iterator script')
   parser.add_argument('--map_size', type=int, default=1<<40, help='Map size for LMDB (default: 1<<40)')
   parser.add_argument('--do_transform', action='store_true', help='Apply transformation function')
   parser.add_argument('--do_filtering', action='store_true', help='Apply filtering functions')
   
   return parser.parse_args()


def setup_logging(mdat_path, submdat_name, suppress_warnings=False):
   #logging to both console and file
   log_dir = os.path.join(mdat_path, 'logs')
   os.makedirs(log_dir, exist_ok=True)
   logger = logging.getLogger('create_submdat')
   logger.setLevel(logging.DEBUG)
   logger.handlers.clear()
   formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
   console_handler = logging.StreamHandler()
   if suppress_warnings:
       console_handler.setLevel(logging.ERROR)
   else:
       console_handler.setLevel(logging.INFO)
   console_handler.setFormatter(formatter)
   logger.addHandler(console_handler)
   file_handler = logging.FileHandler(os.path.join(log_dir, f'{submdat_name}.log'))
   file_handler.setLevel(logging.DEBUG)
   file_handler.setFormatter(formatter)
   logger.addHandler(file_handler)
   return logger


def JSONIterator(json_path):
   logger = logging.getLogger('create_submdat')
   file_size = os.path.getsize(json_path)
   with open(json_path, 'r') as f:
       with tqdm(total=file_size, unit='B', unit_scale=True, desc="Processing JSONL file...") as pbar:
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


def LMDBIterator(lmdb_path):
   logger = logging.getLogger('create_submdat')
   db = LMDBDataset(lmdb_path) 
   for i in tqdm(range(len(db))):
       try:
           yield db[i]
       except Exception as e:
           logger.error(f"Error reading LMDB item {i}: {e}")
           yield None


def create_submdat(input_path, output_path, name, dataset_type, compress_data, compress_meta, data_key, map_size, do_transform=False, do_filtering=False, custom_path=None):
   logger = logging.getLogger('create_submdat')
   # Create or open the MDAT
   mdat = MatformerDataset.create_or_update(output_path, create=True, shuffle=False)
   submdat = mdat.add_submdat(name, create=True, compression_levels={'data':compress_data,'meta':compress_meta}, map_sizes={'data':map_size,'meta':map_size})



   data_db = submdat.db['data']
   meta_db = submdat.db['meta']


   hasDataError = 0 
   
   if dataset_type == 'jsonl':
       generator_fn = JSONIterator(input_path)
   elif dataset_type == 'lmdb':
       generator_fn = LMDBIterator(input_path)
   else:
       logger.error(f"Unsupported dataset type: {dataset_type}")
       return
   
   n_filtered = 0
   raw_data_bytes=0
   raw_meta_bytes=0
   for i, item in enumerate(generator_fn):
       if item is None:
           continue
           
       # A transformer function can be specified by the user
       if do_transform:
           # item = transformer_function(item)
           logger.warning("Transformer function not implemented")
       
       if data_key not in item:
           logger.warning(f"Data key '{data_key}' not found in item {i}. Item has keys {item.keys()}")
           continue
           
       data = item[data_key]
       if isinstance(data,str):
           data=data.encode('utf-8')
       del item[data_key]
       
       # Data can be passed through filters for selection (ex. language identification, quality metrics...)
       filtered = False
       if do_filtering:
           # for filter_fn in filters:
           #     data = filter_fn(data)
           #     if data is None:
           #         filtered = True
           #         n_filtered += 1
           #     else:
           #         filtered = False
           logger.warning("Filter functions not implemented")
       
       if filtered:
           continue
       
       error = data_db.write(data, key=i) 
       if error is not None:
           hasDataError += 1
           
       try:
           if orjson:
               try:
                   serialized = orjson.dumps(item)
               except:
                   serialized = json.dumps(item).encode() 
           else:
               serialized = json.dumps(item).encode()
       except Exception as e:
           logger.error(f'Serialization of item id {i} failed: {e}')
           continue
           
       error = meta_db.write(serialized, key=i)
       if error is not None:
           hasDataError += 1
       # Computing the size of the data just inserted
       raw_data_bytes+=len(data)
       raw_meta_bytes+=len(serialized)
       
   # Close the databases 
   try:
       data_db.close()
   except Exception:
       pass
   try:
       meta_db.close()
   except Exception:
       pass

   # Computing size on disk for the new submdat
   db_disk_bytes = 0
   for root, dirs, files in os.walk(os.path.join(output_path, 'datasets', name)):
       for file in files:
               db_disk_bytes += os.path.getsize(os.path.join(root, file))
   
   if True:
       submdat.new_manifest(
           submdat_name=name,
           raw_data_bytes=raw_data_bytes,
           raw_meta_bytes=raw_meta_bytes,
           db_disk_bytes=db_disk_bytes,
           documents_number=len(data_db),
       )
       mdat.update_manifest()


   
   if hasDataError == 0:
       logger.info(f"Sub-MDAT {name} created successfully in {output_path}. Elements={len(data_db)}. {n_filtered} elements were filtered.")
       print(f"Sub-MDAT {name} created successfully in {output_path}. Elements={len(data_db)}. {n_filtered} elements were filtered.")
   else:
       logger.error(f"Sub-MDAT {name} created with {hasDataError} errors in {output_path}. Elements={len(data_db)}. Check the logs. {n_filtered} elements were filtered.")
       print(f"Sub-MDAT {name} created with {hasDataError} errors in {output_path}. Elements={len(data_db)}. Check the logs. {n_filtered} elements were filtered.")

def main():
   args = parse_arguments()
   
   # Create the MDAT if not existent
   os.makedirs(args.output_path, exist_ok=True)
  
   logger = setup_logging(mdat_path=os.path.join(args.output_path), submdat_name=args.name, suppress_warnings=args.suppress_warnings)

   
   if args.batch_csv is not None:
       logger.info(f"Starting batch processing from {args.batch_csv}")
       with open(args.batch_csv, "r") as batch_file:
           reader = csv.reader(batch_file)
           
           def process_batch_row(row):
               if len(row) >= 4:
                   input_path, output_path, name, dataset_type = row[:4]
                   logger.info(f"Processing batch item: {name}")
                   create_submdat(
                       input_path, output_path, name, dataset_type,
                       args.compress_data, args.compress_meta, args.data_key,
                       args.map_size, args.do_transform, args.do_filtering, args.custom
                   )
           
           with ThreadPoolExecutor(max_workers=4) as executor:
               executor.map(process_batch_row, reader)
   else:
       logger.info(f"Starting single dataset processing: {args.name}")
       create_submdat(
           args.input_path, args.output_path, args.name, args.type,
           args.compress_data, args.compress_meta, args.data_key,
           args.map_size, args.do_transform, args.do_filtering, args.custom
       )
   
   logger.info("Processing completed")


if __name__ == "__main__":
   main()
