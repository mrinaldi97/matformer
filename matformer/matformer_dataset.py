import os
import pickle
from io import BufferedWriter
import struct  
from tqdm import tqdm 
import argparse  
from bisect import bisect_right
import math
import random
from torch_atlas_ds import AtlasDataset, AtlasDatasetWriter  
from transformers import AutoTokenizer  
import multiprocessing as mp  
from functools import partial  
from nltk.tokenize.punkt import PunktTokenizer 
import numpy as np 
import json 
from datasets import load_dataset 


def retrieve_dataset_id(cu_dslens,idx):
    ds_id=bisect_right(cu_dslens,idx)-1
    doc_id=idx-cu_dslens[ds_id]
    return ds_id,doc_id


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


def split_by_tokens(document, max_tokens, punkt_tokenizer, tokenizer, language='italian'):  
    # Ottengo con NLTK gli span di ciascuna frase
    spans = [x for x in punkt_tokenizer.span_tokenize(document)]
    # Riallineo gli span in modo che la fine del primo coincida con l'inizio del secondo e così via
    starts = [s[0] for s in spans]
    ends = [s[0] for s in spans[1:]] + [len(document)]
    spans = list(zip(starts, ends))
    # Tokenizzo span per span fin quando non ho finito gli span, dividendo a gruppi da max_tokens token
    spans.reverse()
    token_output = []  # La lista con i blocchi di token divisi per segmento
    # text_output = []  # Lista di stringhe, ciascuna stringa non supererà max_tokens quando tokenizzate  
    current_tokens_list = []  # Lista token di appoggio
    # current_string = ""  # Stringa di appoggio  

    while len(spans) != 0:
        current_span = spans.pop()
        stringa_span = document[current_span[0]:current_span[1]]
        tokens_span = tokenizer.encode(stringa_span)
        if len(tokens_span) + len(current_tokens_list) <= max_tokens:
            # Questo span entra, lo aggiungo
            # current_string += stringa_span 
            current_tokens_list.extend(tokens_span)
        else:
            # Questo span NON entra in max token. Salvo gli appoggi precedenti e passo a un nuovo segmento
            if len(current_tokens_list) > max_tokens: 
                print(f"WARNING: Chunk exceeded max_tokens ({len(current_tokens_list)} > {max_tokens}), truncating")
                current_tokens_list = current_tokens_list[:max_tokens]
            token_output.append(current_tokens_list)
            # text_output.append(current_string)  
            if len(tokens_span) > max_tokens:
                # Siamo nella situazione in cui una singola "sentence" supera il numero di token massimi.
                token_chunks=[x.tolist() for x in np.array_split(np.array(tokens_span),math.ceil(len(tokens_span)/max_tokens))]
                for chunk in token_chunks[:-1]:  
                    if len(chunk) > max_tokens:
                        print(f"WARNING: Split chunk exceeded max_tokens ({len(chunk)} > {max_tokens}), truncating")
                        chunk = chunk[:max_tokens]
                    token_output.append(chunk)
                # text_chunks=[tokenizer.decode(x, skip_special_tokens=True) for x in token_chunks]  
                # token_output.extend(token_chunks[:-1])  
                # text_output.extend(text_chunks[:-1])  
                current_tokens_list=token_chunks[-1]
                # current_string=text_chunks[-1]  
            else:
                current_tokens_list = tokens_span
                # current_string = stringa_span  
    if len(current_tokens_list) > 0:  # Salvo eventuali residui
        if len(current_tokens_list) > max_tokens:  
            print(f"WARNING: Final chunk exceeded max_tokens ({len(current_tokens_list)} > {max_tokens}), truncating")
            current_tokens_list = current_tokens_list[:max_tokens]
        token_output.append(current_tokens_list)
        # text_output.append(current_string)  

    return token_output  


def process_batch(batch_items, tokenizer_name, num_tokens, text_field, punkt_tokenizer):  
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    results = []
    for item in batch_items:
        text = item[text_field]
        token_chunks = split_by_tokens(text, num_tokens, punkt_tokenizer, tokenizer)  
        # Add tokens_chunks to the item
        item['tokens_chunks'] = token_chunks
        results.append(item)
    return results


def tokenize_dataset(input_data, output_path, tokenizer_name="sapienzanlp/Minerva-350M-base-v1.0", 
                    token_per_segment=1024, text_field='text', num_workers=None, batch_size=100, debug=False):  
    """
    It receives as input three possible iterables: an atlas dataset, a jsonl file or an huggingface dataset object
    """
    
    # Determine input type and create iterator  
    if isinstance(input_data, str):
        if input_data.endswith('.jsonl'):
            # Load jsonl file
            def jsonl_iterator():  # iterator function for jsonl
                with open(input_data, 'r') as f:
                    for line in f:
                        yield json.loads(line)
            dataset_iter = jsonl_iterator()
            #with open(input_data, 'r') as f:
                #total_items = sum(1 for _ in f)
            total_items=0
        else:
            # Assume it's an AtlasDataset path
            dataset = AtlasDataset(input_data)
            dataset_iter = iter(dataset) 
            total_items = len(dataset)
    elif hasattr(input_data, '__iter__'):
        dataset_iter = iter(input_data) 
        try:
            total_items = len(input_data)
        except:
            total_items = None  
    else:
        raise ValueError("Input must be AtlasDataset path, jsonl file, or iterable")
    
    if debug:
        dataset_iter = (x for i, x in enumerate(dataset_iter) if i < 5000) 
        total_items = min(5000, total_items) if total_items else 5000
    
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    nl_tokenizer = PunktTokenizer('italian')
    
    # Process in streaming batches  
    def batch_iterator(iterator, batch_size): 
        batch = []
        for item in iterator:
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:  # Yield remaining items
            yield batch
    
    process_func = partial(
        process_batch,
        tokenizer_name=tokenizer_name,
        num_tokens=token_per_segment,
        text_field=text_field,
        punkt_tokenizer=nl_tokenizer
    )
    
    with AtlasDatasetWriter(output_path, shard_size=256, block_size=10000) as writer:
        with mp.Pool(num_workers) as pool:
            # Stream process batches  
            batch_gen = batch_iterator(dataset_iter, batch_size)
            total_batches = math.ceil(total_items / batch_size) if total_items else None
            
            for batch_results in tqdm(  
                pool.imap(process_func, batch_gen),
                total=total_batches,
                desc="Processing and writing batches"
            ):
                for item in batch_results:
                    writer.add_example(item)
    
    print(f"Created tokenized dataset at {output_path}")


def mergeAtlas(names, tokenizer_type="huggingface", tokenizer_name="sapienzanlp/Minerva-350M-base-v1.0", 
               encoder_type=None, encoder_name=None, token_per_segment=1024, patch_per_segment=1024, shuffle=True):  
    """
    It receives a list of Atlas Datasets. They should be already moved in the current working directory.
    
    """

    precomputed_lengths=[0 for x in range(0,8)] #Lenghts are precomputed up to 8 times the token_per_segment.
    precomputed_byte_lengths=[0 for x in range(0,8)]  # precomputed lengths for bytes
    # 1. Get the name of each dataset
    ds_names=names
    # 2. Open each dataset 
    ds_pointers=[AtlasDataset(x) for x in ds_names]  
    # 3. Get the length of each dataset
    ds_lens=[len(ds) for ds in ds_pointers]
    # 4. Compute the cumulative lenghts
    cu_ds_lens = [0]
    for l in ds_lens:
        cu_ds_lens.append(cu_ds_lens[-1] + l)
    # 5. Compute total documents length
    total_doc_len=sum(ds_lens)
    
    # Initialize statistics 
    total_tokens = 0
    total_chars = 0 
    total_patches = 0
    # min_doc_tokens = float('inf') 
    # max_doc_tokens = 0 
    # min_doc_chars = float('inf') #
    # max_doc_chars = 0 
    
    # 6. Determine the required data types
    ##
    # a. The struct will have format 'ds_pointers,doc_id,token_chunks,patches'
    # b. ds_pointers will be the minimum amount of bytes to represet len(ds_pointers). One or two bytes are usually enough
    # c. doc_id is the largest value, usually it will be 16 bit unsigned, 32 bit unsigned. In some case can even be 64 bit unsigned
    # d. token chunks is hard coded to 16 bit unsigned. If during the computation there are more than 65535 chunks (veery unlikely!) an exception will be raised. It can be missing (usually, it will be present, as it's required for all common models).
    # e. patches is hard coded to 16 bit unsigned. See point (d). It can be missing (usually, it will be missing as it will be used only for experimental encoder+decoder models or multimodal models).
    ##
    
    # Determine ds_pointer data type
    if len(ds_pointers) <= 255:
        ds_pointer_type = 'B'  # 8 bit unsigned
    else:
        ds_pointer_type = 'H'  # 16 bit unsigned
    
    # Determine doc_id data type  
    max_doc_id = max(ds_lens)  
    if max_doc_id <= 65535: # I can use 16 bit unsigned
        doc_id_type='H'  
    elif max_doc_id <= 4294967295: #32 bit unsigned
        doc_id_type='I'  
    else: #64 bit (should be enough!!!)
        doc_id_type='Q' 
    
    # 7. Check if all the datasets have token segments
    hasSegments=True
    for ds in ds_pointers:
        if 'tokens_chunks' in ds[0].keys() and 'tokens_chunks' in ds[-1].keys():
            pass
        else:
            hasSegments=False
    hasPatches=True
    for ds in ds_pointers:
        if 'patches' in ds[0].keys() and 'patches' in ds[-1].keys():
            pass
        else:
            hasPatches=False
            
    # Build struct format string
    struct_format = '<' + ds_pointer_type + doc_id_type  
    if hasSegments:
        struct_format += 'H'  # token_chunks
        struct_format += 'H'  #remainder field for tokens
    if hasPatches:
        struct_format += 'H'  # patches

    # Create permutator if shuffling
    if shuffle:
        permutator = RandomPermutator(total_doc_len)  
    
    # 8. Starting the loooong iteration to shuffle the datasets (or just merge them)
    with open("matformer_dataset.mdat","wb") as raw:  
        # First write the header
        mdat_header = {
            "type":"mdat",
            "version":"1.0", 
            "datasets_map":{i: name for i, name in enumerate(ds_names)},  
            "tokenizer_type":tokenizer_type,  
            "tokenizer_name":tokenizer_name,  
            "encoder_type":encoder_type,  
            "encoder_name":encoder_name,  
            "token_per_segment":token_per_segment, 
            "patch_per_segment":patch_per_segment,
            "precomputed_length":precomputed_lengths,
            "precomputed_byte_length":precomputed_byte_lengths,  
            "ds_pointers":ds_names,
            "ds_lens":ds_lens,  
            "structs":{
                "ds_pointer":ds_pointer_type,
                "item":doc_id_type, 
                "max_chunks":'H' if hasSegments else None,
                "token_remainder":'H' if hasSegments else None, 
                "patches":'H' if hasPatches else None
            }
        }
        pickle.dump(mdat_header, raw)  
        
        # Write zeroed statistics header - will be updated later  
        statistics_format = '<8Q8Q4Q2Q'  # 8*precomputed_lengths + 8*precomputed_byte_lengths + 4*totals + 2*avg
        statistics_size = struct.calcsize(statistics_format)
        statistics_offset = raw.tell()  # Remember position for later update
        raw.write(b'\x00' * statistics_size)  # zeros as placeholder
        
        f=BufferedWriter(raw)
        for i in tqdm(range(total_doc_len)):
            if shuffle:
                idx=permutator(i)  
            else:
                idx=i
            
            # Retrieve the dataset associated with the idx
            ds_id,doc_id=retrieve_dataset_id(cu_ds_lens,idx)  
            doc=ds_pointers[ds_id][doc_id]  
            for ds in ds_pointers:
                for shard in ds.shards:
                    if hasattr(shard, 'last_block'):
                        shard.last_block = None
                        shard.last_block_idx = None            
            # Prepare data to pack
            pack_data = [ds_id, doc_id]
            
            # Update statistics 
            if 'text' in doc:
                doc_chars = len(doc['text'])
                total_chars += doc_chars
                # min_doc_chars = min(min_doc_chars, doc_chars)  
                # max_doc_chars = max(max_doc_chars, doc_chars)  
               
                for j in range(1, 9):
                    if j <= len(precomputed_byte_lengths):
                        chunk_size_bytes = j * token_per_segment
                        doc_byte_chunks = 0
                        start = 0
                        doc_len = len(doc['text'])
                        
                        while start < doc_len:
                            end = min(start + chunk_size_bytes, doc_len)
                            
                            # If we're not at the document end, try to find a word boundary
                            if end < doc_len:
                                # Look back from end to find a space, but limit search to last 100 chars
                                search_start = max(start, end - 100)
                                space_found = False
                                
                                # Search backwards for a space
                                for i in range(end - 1, search_start - 1, -1):
                                    if doc['text'][i] == ' ':
                                        end = i + 1  # Position after the space
                                        space_found = True
                                        break
                                
                                # Ensure forward progress - if no space found or end <= start, use original end
                                if not space_found or end <= start:
                                    end = min(start + chunk_size_bytes, doc_len)
                            
                            if end <= start:
                                end = start + 1  
                            
                            start = end
                            doc_byte_chunks += 1
                            
                            # Safety check to prevent infinite loops
                            if doc_byte_chunks > doc_len: 
                                print(f"WARNING: Byte chunking exceeded limit for document")
                                break
                            
                        precomputed_byte_lengths[j-1] += doc_byte_chunks
            
            # How many token chunks?
            if hasSegments:
                doc_chunks=len(doc['tokens_chunks'])
                pack_data.append(doc_chunks)
                # Calculate total tokens in this document 
                doc_tokens = sum(len(chunk) for chunk in doc['tokens_chunks'])
                total_tokens += doc_tokens
                # min_doc_tokens = min(min_doc_tokens, doc_tokens)  
                # max_doc_tokens = max(max_doc_tokens, doc_tokens)
                
                # Calculate remainder  #
                remainder = doc_tokens % token_per_segment
                pack_data.append(remainder)
                
                # Precompute total chunk length up to 8 times the base segment token count
                # Total chunk length is required for accurate len(ds) estimations for the decided max_len of the model. This is important for accurate training time estimation
                # ex. if token segment is 1024, precompute for 1024,2048,4096,8192,16384,32768,65536,131072
                for j in range(1,9): 
                    if j <= len(precomputed_lengths):  
                        precomputed_lengths[j-1]+=math.ceil(doc_chunks/j)
            
            # How many patches?
            if hasPatches:
                doc_patches=len(doc['patches'])
                pack_data.append(doc_patches)
                total_patches += doc_patches 
            # Add to the struct the "remainder". For example, if tokens are 1972 with a segment size of 1024 the "remainder" will be 948
            # In this way it will be very easy to reconstruct the exact size of each document from the .mdat file, but without
            # adding a significant overhead in file size, because two bytes are more than enough to represent this information even for long documents
            # Add to the map file, currently no sharding (just one file)
            f.write(struct.pack(struct_format, *pack_data))  
        f.flush()
    
    # Update statistics header  - write real statistics after the big iteration
    with open("matformer_dataset.mdat", "r+b") as raw:
        raw.seek(statistics_offset)
        # if min_doc_tokens == float('inf'): 
        #   min_doc_tokens = 0
        # if min_doc_chars == float('inf'):
        #   min_doc_chars = 0
        
        avg_doc_tokens = total_tokens // total_doc_len if total_doc_len > 0 else 0 
        avg_doc_chars = total_chars // total_doc_len if total_doc_len > 0 else 0
        
        # Pack: 8*precomputed_lengths + 8*precomputed_byte_lengths + 4*totals + 2*avg  
        stats_data = (
            *precomputed_lengths,  # 8 values
            *precomputed_byte_lengths,  # 8 values  
            total_tokens, total_chars, total_patches, total_doc_len,  # 4 totals
            # min_doc_tokens, min_doc_chars,  # 2 minimums # Don't need this  
            # max_doc_tokens, max_doc_chars,  # 2 maximums  
            avg_doc_tokens, avg_doc_chars,  # 2 averages
            # *ds_lens  # documents per dataset # Don't need this  
        )

        raw.write(struct.pack(statistics_format, *stats_data))  
    
    print(f"Created matformer_dataset.mdat with {total_doc_len} documents")
        

class MatformerDataset:
    def __init__(self,path,modality='tokens',chunk_size=None,tokens=None,n_bytes=None,state=None):
        """
            Path => The path of a .mdat file
            Modality => 'tokens' to return chunk of tokenized text, 'bytes' to return chunks of raw bytes, 'patches' to return patches  #EDITED BY LLM: added patches modality
            chunk_size => How many chunks to return at each iteration. If the current documents has smaller chunk, return the entire document
            state => The document to restart again if saved; still to be implemented
        """
        if modality=='tokens':
            assert chunk_size is not None or tokens is not None
        elif modality=='patches':  
            assert chunk_size is not None
        else:
            assert n_bytes is not None
            self.chunk_size=n_bytes
        
        self.modality=modality
        try:
            self.fp=open(os.path.join(path,'matformer_dataset.mdat'),"rb")  
        except:
            print(f"ERROR: The path {path} it's not the root of a valid Matformer Dataset (matformer_dataset.mdat file missing)")  
        
        # Loading .mdat metadata
        self.mdat_header=pickle.load(self.fp)
        assert self.mdat_header['type'] == 'mdat'    
        """
            // SAMPLE MATFORMER DATASET MODULE HEADER
            {
            "type":"mdat",
            "version":"1.0",
            "datasets_map":{
                1:'wikipedia',
                2:'liberliber'
                },
            "tokenizer_type":"huggingface",
            "tokenizer_name":"sapienzanlp/Minerva-350M-base-v1.0",
            "token_per_segment":1024,
            "precomputed_length":[x,x,x,x,x,x,x],
            "ds_pointers":['wikipedia','liberliber'],
            structs={
                "ds_pointer":'B',
                "item":'I',
                "max_chunks":H
            }
            }
            // SAMPLE DOCUMENTS POINTERS
            // ds_pointer,item,max_chunks
            [0,1027180096,3]
            [1,9123,5]
            [2,7890098,5] => 3
        """
        
        # Load statistics header  
        statistics_format = '<8Q8Q4Q2Q'  
        statistics_size = struct.calcsize(statistics_format)
        stats_data = struct.unpack(statistics_format, self.fp.read(statistics_size))
        
        # Parse statistics  
        self.precomputed_lengths = list(stats_data[0:8])
        self.precomputed_byte_lengths = list(stats_data[8:16])  
        self.total_tokens = stats_data[16]
        self.total_chars = stats_data[17] 
        self.total_patches = stats_data[18]
        self.total_documents = stats_data[19]
        # self.min_doc_tokens = stats_data[12]  
        # self.min_doc_chars = stats_data[13]  
        # self.max_doc_tokens = stats_data[14]  
        # self.max_doc_chars = stats_data[15]  
        self.avg_doc_tokens = stats_data[20]
        self.avg_doc_chars = stats_data[21]
        
        self.token_per_segment=self.mdat_header['token_per_segment']
        if chunk_size is not None:
            self.chunk_size=chunk_size
        else:
            assert tokens%self.token_per_segment==0
            self.chunk_size=tokens//self.token_per_segment
        
        # Build struct format for reading
        struct_parts = ['<', self.mdat_header['structs']['ds_pointer'], self.mdat_header['structs']['item']]
        if self.mdat_header['structs']['max_chunks']:
            struct_parts.append(self.mdat_header['structs']['max_chunks'])
        if self.mdat_header['structs'].get('token_remainder'):  
            struct_parts.append(self.mdat_header['structs']['token_remainder'])
        if self.mdat_header['structs']['patches']:
            struct_parts.append(self.mdat_header['structs']['patches'])
        self.struct_format = ''.join(struct_parts)  
        self.struct_size = struct.calcsize(self.struct_format)  
        
        # Opening actual dataset(s)
        # Currently, only supports AtlasDatasets but can be easily extended to any kind of dataset (ex. HuggingFace)
        self.ds_pointers=[AtlasDataset(os.path.join(path,x)) for x in self.mdat_header['ds_pointers']]
        
        if modality == 'tokens' and len(self.precomputed_lengths) >= self.chunk_size:  
            self.len=self.precomputed_lengths[self.chunk_size-1]  
        elif modality == 'bytes' and len(self.precomputed_byte_lengths) >= 1:  
            # For bytes, use the first precomputed length as approximation
            self.len = self.precomputed_byte_lengths[0]
        else:
            print(f"WARNING: length for {self.chunk_size*self.token_per_segment} was not computed during dataset creation!")
            self.len=None
        
        self._load_next_document() # Loading the first document
        
    def __len__(self):
        return self.len
        
    def _load_next_document(self):
        try:
            row_data = self.fp.read(self.struct_size)  
            if not row_data:
                raise EOFError
            row = struct.unpack(self.struct_format, row_data)  
            self.current_document = self.ds_pointers[row[0]][row[1]]
            self.current_document_step = 0
            self.current_document_remainder = row[3] if len(row) > 3 and self.mdat_header['structs'].get('token_remainder') else 0  #EDITED BY LLM: store remainder for exact token count
        except (EOFError, StopIteration):
            self.current_document = None
        
    def __iter__(self):
        return self
        
    def __next__(self):
        if self.modality=='tokens':
            return self._next_tokens()
        elif self.modality=='patches':  
            return self._next_patches()
        else:
            return self._next_bytes()
    
    def _next_tokens(self):
        # I expect in the document dictionary the field "tokens_chunks"
        if self.current_document is None:  #EDITED BY LLM: changed assert to if check for graceful handling
            raise StopIteration
        if "tokens_chunks" not in self.current_document.keys():  #EDITED BY LLM: better error message
            raise KeyError(f"Document missing 'tokens_chunks' field. Did you run tokenize_dataset first?")
        
        while self.current_document:
            chunks=self.current_document.get('tokens_chunks')
            start=self.current_document_step*self.chunk_size
            end=start+self.chunk_size
            
            # Am I still inside a document? Shall I retrieve the next document?
            if start>=len(chunks):
                self._load_next_document() # Document is over, going to the next one
                continue 
            
            self.current_document_step+=1
            tokens=[x for z in chunks[start:end] for x in z] # I take chunk_size chunks from the current document
            return tokens
     
    def _next_patches(self):  
        # I expect in the document dictionary the field "patches"
        if self.current_document is None: 
            raise StopIteration
        if "patches" not in self.current_document.keys(): 
            raise KeyError(f"Document missing 'patches' field")
        
        while self.current_document:
            patches=self.current_document.get('patches')
            start=self.current_document_step*self.chunk_size
            end=start+self.chunk_size
            
            # Am I still inside a document? Shall I retrieve the next document?
            if start>=len(patches):
                self._load_next_document() # Document is over, going to the next one
                continue 
            
            self.current_document_step+=1
            patches_output=[x for z in patches[start:end] for x in z] # I take chunk_size chunks from the current document
            return patches_output
            
    def _next_bytes(self):
        while self.current_document:
            document = self.current_document.get('text')
            if not document:  # Handle empty or None document
                self._load_next_document()
                continue
                
            start = self.current_document_step
            doc_len = len(document)

            if start >= doc_len:
                self._load_next_document()
                continue

            end = min(start + self.chunk_size, doc_len)
            
            # If we're not at the document end, try to find a word boundary
            if end < doc_len:
                # Look back from end to find a space, but limit search to last 100 chars
                search_start = max(start, end - 100)
                space_found = False
                
                # Search backwards for a space
                for i in range(end - 1, search_start - 1, -1):
                    if document[i] == ' ':
                        end = i + 1  # Position after the space
                        space_found = True
                        break
                
                # Ensure forward progress - if no space found or end <= start, use original end
                if not space_found or end <= start:
                    end = min(start + self.chunk_size, doc_len)
            
            # Guarantee forward progress
            if end <= start:
                end = min(start + 1, doc_len)  # Move at least 1 character forward
            
            # Extract the actual string only once at the end
            _string = document[start:end]
            self.current_document_step = end
            return _string

        raise StopIteration 
def auto_discover_datasets(): 
    """Auto-discover AtlasDataset directories in current folder"""
    datasets = []
    for item in os.listdir('.'):
        if os.path.isdir(item):
            try:
                AtlasDataset(item) 
                datasets.append(item)
            except:
                continue
    return datasets


def main(): 
    parser = argparse.ArgumentParser(description='Create MDAT dataset from AtlasDatasets')
    parser.add_argument('--datasets', nargs='*', help='Dataset names (auto-discover if not specified)')
    parser.add_argument('--tokenizer-type', default='huggingface', help='Tokenizer type')
    parser.add_argument('--tokenizer-name', default='sapienzanlp/Minerva-350M-base-v1.0', help='Tokenizer name')
    parser.add_argument('--encoder-type', help='Encoder type')
    parser.add_argument('--encoder-name', help='Encoder name')  
    parser.add_argument('--token-per-segment', type=int, default=1024, help='Tokens per segment')
    parser.add_argument('--patch-per-segment', type=int, default=1024, help='Patches per segment')
    parser.add_argument('--no-shuffle', action='store_true', help='Disable shuffling')
    parser.add_argument('--tokenize', action='store_true', help='Tokenize datasets before merging')  
    parser.add_argument('--tokenize-input', type=str, help='Input path for tokenization')  
    parser.add_argument('--tokenize-output', type=str, help='Output path for tokenized dataset')  
    parser.add_argument('--debug', action='store_true', help='Debug mode - limit to 5000 samples')  
    
    args = parser.parse_args()
    
    # If tokenize mode, run tokenization
    if args.tokenize: 
        if not args.tokenize_input or not args.tokenize_output:
            print("ERROR: --tokenize-input and --tokenize-output required for tokenization")
            return
        tokenize_dataset(
            input_data=args.tokenize_input,
            output_path=args.tokenize_output,
            tokenizer_name=args.tokenizer_name,
            token_per_segment=args.token_per_segment,
            debug=args.debug
        )
        return
    
    # Auto-discover datasets if not specified
    if args.datasets is None:
        datasets = auto_discover_datasets()
        if not datasets:
            print("No AtlasDatasets found in current directory!")
            return
        print(f"Auto-discovered datasets: {datasets}")
    else:
        datasets = args.datasets
    
    # Create MDAT dataset
    mergeAtlas(
        names=datasets,
        tokenizer_type=args.tokenizer_type,
        tokenizer_name=args.tokenizer_name,
        encoder_type=args.encoder_type,
        encoder_name=args.encoder_name,
        token_per_segment=args.token_per_segment,
        patch_per_segment=args.patch_per_segment,
        shuffle=not args.no_shuffle
    )


if __name__ == '__main__':  
    main()
