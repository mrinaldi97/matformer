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

def create_submdat(input_path, output_path, name, dataset_type, compress_data, compress_meta, data_key, map_size, modality, data_type='text',do_transform=False, do_filtering=False, custom_path=None):
   logger = logging.getLogger('create_submdat')
   dataset_args={}
   mdat = MatformerDataset.load_dataset(path=output_path,create_if_not_existent=True)
   submdat = mdat.add_submdat(submdat_name=name, compression_levels={'data':compress_data,'meta':compress_meta}, map_sizes={'data':map_size,'meta':map_size},
        data_type=data_type,db_types = ['meta','data'])
   result = submdat.convert_to_submdat(dataset_type=dataset_type,dataset_path=input_path,dataset_args=dataset_args,data_key=data_key,modality=modality,do_transform=do_transform,do_filtering=do_filtering,logger=logger,progress_bar=True)
   logger.info(f"Sub-MDAT {name} created successfully in {output_path}")
   for k,v in zip(result.keys(),result.values()): #Result contains useful stuff such as errors, disk size, document number
       if isinstance(v,dict):
           for k2,v2 in zip(v.keys(),v.values()):
               logger.info(f"({k}) {k2}={v2}")
       else:
           logger.info(f"{k} = {v}")
       
def main():
   args = parse_arguments()
   
  
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
