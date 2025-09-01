"""
Pretokenize Mdat or sub-mdat


A pretokenized sub-mdat will contain an additional folder called "pretok" and inside this folder a folder for each pre-tokenization
"pre-tokenization" is an abstract term: it can refers to classic text tokenization to integer as well as different form of input conversion such as Encoders to floating point values
"""
import os
import json
import argparse
from tqdm import tqdm
import numpy as np

from matformer_dataset import MatformerDataset, SubMdat, TokenizationRegistry
from lmdb_dataset import LMDBDataset


class DummySplitter(MdatSplitter):
    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
    def __call__(self, document):
        return self.tokenizer(document)

try:
    from nltk.tokenize.punkt import PunktSentenceTokenizer as PunktTokenizer
except Exception:
    PunktTokenizer = None

class tokenSplitter_nltkSentence_Greedy(MdatSplitter):
    def __init__(self, tokenizer, max_tokens_per_chunk=512, language='english', **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.language = language
        if PunktTokenizer is not None:
            self.punkt_tokenizer = PunktTokenizer()
        else:
            self.punkt_tokenizer = None



def parse_args():
    parser = argparse.ArgumentParser(description='Pretokenize a sub-MDAT using a registered tokenizer strategy.')
    parser.add_argument('mdat_path', help='Path to MDAT root')
    parser.add_argument('submdat_name', help='SubMDat name to pretokenize')
    parser.add_argument('--strategy', required=True, help='Name of the tokenization strategy (must be registered in the registry)')
    parser.add_argument('--tokenizer', required=False, help='Tokenizer identifier if needed by strategy')
    parser.add_argument('--max_tokens_per_chunk', type=int, default=512)
    parser.add_argument('--language', default='english')
    parser.add_argument('--tokens_compression', type=int, default=0)
    parser.add_argument('--chunks_dtype', default='int32')
    parser.add_argument('--pretok_db_dir', default=None, help='(Ignored) use SubMdat automatic layout')
    return parser.parse_args()

def pretokenize(mdat_path, submdat_name, strategy_name, tokenizer_id=None, max_tokens_per_chunk=512, language='english', tokens_compression=0, chunks_dtype='int32'):
    mdat = MatformerDataset.load_readonly(mdat_path, shuffle=False)

    tokenizer = None
    sub_mdat = mdat.get_submdat(submdat_name)
    # Choose splitter implementation
    splitter = tokenSplitter_nltkSentence_Greedy(tokenizer, max_tokens_per_chunk=max_tokens_per_chunk, language=language)

	# Da cambiare
    compressed_tokens = tokens_compression > 0
    tokens_db = sub_mdat.create_pretok_db(strategy_name, 'tokens', compression_level=tokens_compression) if 'tokens' in splitter.return_types else None
    chunks_db = sub_mdat.create_pretok_db(strategy_name, 'chunks', compression_level=tokens_compression) if 'chunks' in splitter.return_types else None
    extra_db = sub_mdat.create_pretok_db(strategy_name, 'extra', compression_level=tokens_compression) if 'extra' in splitter.return_types else None


    doc_count = len(sub_mdat)
    stats = {'total_tokens': 0, 'total_chunks': 0, 'max_chunks': 0, 'average_chunks': 0}
    total_chunks_acc = 0

    for key in tqdm(range(doc_count), desc="Pretokenizing documents"):
        try:
            raw = sub_mdat.db['data'][key]
        except Exception as e:
            continue

        try:
            tokens_result = splitter(raw)
        except Exception as e:
            # Tokenization failed
            continue

        chunks = tokens_result.get('chunks') if 'chunks' in splitter.return_types else None
        tokens = tokens_result.get('tokens') if 'tokens' in splitter.return_types else None
        extra = tokens_result.get('extra') if 'extra' in splitter.return_types else None

        # Write tokens
        if tokens is not None:
            if isinstance(tokens, (list, tuple, np.ndarray)):
                arr = np.array(tokens)
                tokens_db.write(arr.tobytes(), key=key)
                stats['total_tokens'] += arr.size
            else:
                tokens_db.write(tokens, key=key)
                stats['total_tokens'] += 1
        # Write chunks (pointers)
        if chunks is not None:
            if isinstance(chunks, list):
                # Pointers is a list of tuples ex: [(0,15), (15,36), (36,92)]
                # Transformed to a size array, ready to be stored in the pointers db:
                import numpy as _np
                starts = _np.array([c[0] for c in chunks], dtype=chunks_dtype)
                ends = _np.array([c[1] for c in chunks], dtype=chunks_dtype)
                sizes = (ends - starts)
                assert sizes.sum() == (len(tokens) if tokens is not None else sizes.sum())
                max_size = sizes.max() if sizes.size > 0 else 0
                assert max_size <= max_tokens_per_chunk
                # convert to requested dtype
                sizes = sizes.astype(_np.dtype(chunks_dtype))
                chunks_db.write(sizes.tobytes(), key=key)
                stats['total_chunks'] += sizes.size
                total_chunks_acc += sizes.size
                if sizes.size > stats['max_chunks']:
                    stats['max_chunks'] = sizes.size
            else:
                print(f"ERROR: The Splitter/Tokenizer returned {type(chunks)} as chunks instead of a list. This is unrecoverable")
        if extra is not None:
            print("WARNING: 'extra' not yet implemented. Skipping")

    stats['average_chunks'] = float(total_chunks_acc) / (doc_count if doc_count > 0 else 1)

    # Write stats
    sub_mdat.write_pretok_stats(strategy_name, stats)

    # Close DBs
    for db in [tokens_db, chunks_db, extra_db]:
        try:
            if db is not None:
                db.close()
        except Exception:
            pass

    print("-------------------------------------------")        
    print("Pre-tokenization completed with no errors.")
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Max chunks per document: {stats['max_chunks']}")
    print(f"Average chunks per document: {stats['average_chunks']}")
    print("-------------------------------------------")

def main():
    args = parse_args()
    pretokenize(args.mdat_path, args.submdat_name, args.strategy, tokenizer_id=args.tokenizer, max_tokens_per_chunk=args.max_tokens_per_chunk, language=args.language, tokens_compression=args.tokens_compression, chunks_dtype=args.chunks_dtype)

if __name__ == "__main__":
    main()
