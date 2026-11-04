"""Pretokenization strategies, batch workers, and built-in text splitters."""
from __future__ import annotations

import importlib
import json
import math
import os
import sys
import traceback
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import orjson
from nltk.tokenize import PunktTokenizer

from .exceptions import (
    InvalidChunksFormat, MissingDataKey, MissingSplitterOutput,
    MissingStrategyKey, StrategyNotFound,
)
from .utils import get_datatype_for_numpy

if TYPE_CHECKING:
    from .sql_database import DatabaseManager

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
            if self.db is None or not self.db.readonly:
                self.save()  # Persist inferred token/chunk types only when writable.



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
