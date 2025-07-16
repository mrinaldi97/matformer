"""
A script to split text Atlas dataset by the number of bytes or tokens.
"""

from torch_atlas_ds import AtlasDataset, AtlasDatasetWriter
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer

def split_string(string, num_bytes, split_char=' '):
    """Split a string into a list of strings splitted by n_bytes but truncated only at whitespaces"""
    parts = string.split(split_char)
    chunks_list = list()
    split_string = ""
    for chunk in parts:
        if len(split_string) + len(chunk) + 1 > num_bytes:
            chunks_list.append(split_string)
            split_string = chunk + ' '
        else:
            split_string += chunk + ' '
    if len(split_string.strip()) > 1:
        chunks_list.append(split_string.strip())
    return chunks_list

def split_string_by_tokens(string, tokenizer, num_tokens, split_char=' '):
    """Split a string into a list of strings splitted by number of tokens but truncated only at whitespaces"""
    parts = string.split(split_char)
    chunks_list = list()
    current_chunk = ""
    for chunk in parts:
        trial_chunk = current_chunk + chunk + ' '
        if len(tokenizer.encode(trial_chunk, add_special_tokens=False)) > num_tokens:
            chunks_list.append(current_chunk.strip())
            current_chunk = chunk + ' '
        else:
            current_chunk = trial_chunk
    if len(current_chunk.strip()) > 1:
        chunks_list.append(current_chunk.strip())
    return chunks_list

def main():
    parser = argparse.ArgumentParser(description='Split items of an Atlas Dataset up to a certain amount of bytes or tokens')
    parser.add_argument('--input', type=str, required=True, help='Path to dataset input')
    parser.add_argument('--output', type=str, required=True, help='Path to dataset input')
    parser.add_argument('--bytes', type=int, default=None, help='Max sequence length in bytes')
    parser.add_argument('--tokens', type=int, default=None, help='Max sequence length in tokens using a Hugging Face tokenizer')
    parser.add_argument('--tokenizer', type=str, default='sapienzanlp/Minerva-350M-base-v1.0', help='Tokenizer name for token-based splitting')
    parser.add_argument('--shard_size', type=int, default=100, help='Output dataset shard size')
    parser.add_argument('--block_size', type=int, default=100, help='Output dataset block size')    
    parser.add_argument('--text_field', type=str, default='text', help='Field of the dict having the text') 
    parser.add_argument('--split_separator', type=str, default=' ', help='Chunks separator. Default: whitespace')
    parser.add_argument('--segment_key', type=str, default='_segment', help='Key for the segment counter in the output dataset')
    args = parser.parse_args()

    if args.tokens is not None and args.bytes is not None:
        raise ValueError("You must specify either --bytes or --tokens, not both.")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer) if args.tokens is not None else None
    dataset = AtlasDataset(args.input)
    
    with AtlasDatasetWriter(args.output, shard_size=args.shard_size, block_size=args.block_size) as writer:
        for item in tqdm(dataset):
            text = item[args.text_field]
            if args.tokens is not None:
                chunks = split_string_by_tokens(text, tokenizer, args.tokens, args.split_separator)
            else:
                chunks = split_string(text, args.bytes, args.split_separator)
            for segment_counter, chunk in enumerate(chunks):
                new_item = item.copy() 
                new_item[args.text_field] = chunk
                new_item[args.segment_key] = segment_counter
                writer.add_example(new_item)

if __name__ == '__main__':
    main()

