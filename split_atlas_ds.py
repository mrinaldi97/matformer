"""
A script to split text Atlas dataset by the number of bytes. It can be expanded to support tokens instead of bytes.
"""

from torch_atlas_ds import AtlasDataset, AtlasDatasetWriter
import argparse
from tqdm import tqdm

def split_string(string,num_bytes,split_char=' '):
    """Split a string into a list of strings splitted by n_bytes but truncated only at whitespaces"""
    parts=string.split(split_char)
    chunks_list=list()
    split_string=""
    for chunk in parts:
        if len(split_string)+len(chunk)+1>num_bytes:
            chunks_list.append(split_string)
            split_string=chunk+' '
        else:
            split_string+=chunk+' '
    if len(split_string.strip())>1:
        chunks_list.append(split_string.strip())
    return chunks_list
            
def main():
    parser = argparse.ArgumentParser(description='Split items of an Atlas Dataset up to a certain amount of bytes')
    parser.add_argument('--input', type=str, required=True, help='Path to dataset input')
    parser.add_argument('--output', type=str, required=True, help='Path to dataset input')
    parser.add_argument('--bytes', type=int, required=True, help='Max sequence length')
    parser.add_argument('--shard_size', type=int, default=100, help='Output dataset shard size')
    parser.add_argument('--block_size', type=int, default=100, help='Output dataset block size')    
    parser.add_argument('--text_field', type=str, default='text', help='Field of the dict having the text') 
    parser.add_argument('--split_separator', type=str, default=' ', help='Chunks separator. Default: whitespace')
    parser.add_argument('--segment_key', type=str, default='_segment', help='Key for the segment counter in the output dataset')
    args = parser.parse_args()
    
    dataset = AtlasDataset(args.input)
    with AtlasDatasetWriter(args.output, shard_size=args.shard_size, block_size=args.block_size) as writer:
        for item in tqdm(dataset):
            chunks=split_string(item[args.text_field],args.bytes)
            for segment_counter,chunk in enumerate(chunks):
                new_item = item.copy() 
                new_item[args.text_field]=chunk
                new_item[args.segment_key]=segment_counter
                writer.add_example(new_item)
if __name__ == '__main__':
    main()
