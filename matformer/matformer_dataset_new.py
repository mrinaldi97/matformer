import os
import pickle
import struct
import torch
import random
import re
from collections import deque
import psutil
import torch.distributed as dist
from matformer.matformer_dataset import LMDBDataset
class MatformerDataset(torch.utils.data.IterableDataset):
    def __init__(self, path, modality, chunk_size=None, tokens=None, n_bytes=None, state=None, 
                 byte_tokenizer=None, entropy_model=None, entropy_function=None, 
                 entropy_smoothing=None, return_type='ids'):
        """
            Path => The path of a .mdat file
            Modality => 'tokens' to return chunk of tokenized text, 'bytes' to return chunks of raw bytes, 'patches' to return patches  
            chunk_size => How many chunks to return at each iteration. If the current documents has smaller chunk, return the entire document
            state => The document to restart again if saved; still to be implemented
        """
        self.compute_entropy = entropy_function
        assert return_type in ['ids', 'text', 'dict']
        self.return_type = return_type
        self.dummy_entropy_mode = True
        self.hard_limit = 27  # This is needed just for tests with entropy model
        
        # Modality setup
        if modality == 'tokens':
            assert chunk_size is not None or tokens is not None
        elif modality == 'patches':  
            assert chunk_size is not None
        elif modality == 'bytes':
            print("DEBUG: BYTE MODE ENABLED")
            self.byte_tokenizer = byte_tokenizer
            assert n_bytes is not None
            self.n_bytes = n_bytes
            self.chunk_size = n_bytes
        elif modality == 'entropy_patches':
            print("DEBUG: ENTROPY MODE ENABLED")
            print("Length will not be returned")
            assert entropy_model is not None
            self.entropy_model = entropy_model
            self.n_bytes = n_bytes
            self.chunk_size = chunk_size
            self.current_segment_group = None
            self.current_segment_step = 0  
            self.current_entropy_byte_pos = 0
            self.entropy_smoothing = entropy_smoothing
            
        # Distribution setup
        if dist.is_available() and dist.is_initialized():
            self.dist = True
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.dist = False
            
        self.modality = modality
        
        # File handling
        try:
            self.fp = open(os.path.join(path, 'matformer_dataset.mdat'), "rb")  
        except:
            print(f"ERROR: The path {path} it's not the root of a valid Matformer Dataset (matformer_dataset.mdat file missing)")  
        
        # Loading .mdat metadata
        self.mdat_header = pickle.load(self.fp)
        assert self.mdat_header['type'] == 'mdat'    
        
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
        self.avg_doc_tokens = stats_data[20]
        self.avg_doc_chars = stats_data[21]
        
        # NEW: Document index for random access
        self.doc_index_pos = self.fp.tell()  # Save position of document index
        self.doc_index = None
        self.doc_index_loaded = False
        self.doc_cache = deque(maxlen=1000)  # Simple LIFO cache for documents
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
        # Try to load document index if it fits in memory
        total_memory = psutil.virtual_memory().total
        index_size = self.total_documents * self.struct_size
        if index_size < total_memory * 0.2:  # Use up to 20% of RAM for index
            self._load_doc_index()
        
        self.token_per_segment = self.mdat_header['token_per_segment']
        if modality == 'tokens':
            if chunk_size is not None:
                self.chunk_size = chunk_size
            else:
                print(f"tokens: {tokens}, token_per_segment: {self.token_per_segment}")
                self.chunk_size = tokens // self.token_per_segment
        
 
        
        # Opening actual dataset(s)
        self.ds_pointers = [
            AtlasDataset(os.path.join(path, x)) if t == 'atlas'
            else load_dataset(os.path.join(path, x)) if t == 'hf'
            else LMDBDataset(os.path.join(path, x)) if t == 'lmdb'
            else None
            for x, t in zip(self.mdat_header['ds_pointers'], self.mdat_header['ds_types'])
        ]        
        
        # NEW: Calculate actual length based on total chunks if index is loaded
        if self.doc_index_loaded and modality == 'tokens':
            # Sum all chunks across all documents
            total_chunks = sum(row[2] for row in self.doc_index)  # row[2] is max_chunks
            self.len = total_chunks
        elif modality == 'tokens' and len(self.precomputed_lengths) >= self.chunk_size:  
            self.len = self.precomputed_lengths[self.chunk_size-1]  
        elif modality == 'bytes' and len(self.precomputed_byte_lengths) >= 1:  
            self.len = self.precomputed_byte_lengths[0]
        elif modality == 'entropy_patches':
            print(f"WARNING: length for entropy mode not yet implemented.")
            self.len = None
        else:
            print(f"WARNING: length for {self.chunk_size*self.token_per_segment} was not computed!")
            self.len = None
        
        # NEW: Multi-GPU document distribution if index is loaded
        if self.dist and self.doc_index_loaded:
            # Distribute documents evenly across GPUs
            docs_per_gpu = (self.total_documents + self.world_size - 1) // self.world_size
            self.my_doc_start = self.rank * docs_per_gpu
            self.my_doc_end = min((self.rank + 1) * docs_per_gpu, self.total_documents)
            self.current_doc_idx = self.my_doc_start - 1  # Will be incremented in _load_next_document
        else:
            # Fall back to sequential reading
            self._load_next_document()  # Loading the first document
        
    def _load_doc_index(self):
        """Load the entire document index into memory for random access"""
        try:
            current_pos = self.fp.tell()
            self.fp.seek(self.doc_index_pos)
            
            self.doc_index = []
            for _ in range(self.total_documents):
                row_data = self.fp.read(self.struct_size)
                if not row_data:
                    break
                row = struct.unpack(self.struct_format, row_data)
                self.doc_index.append(row)
                
            self.doc_index_loaded = True
            self.fp.seek(current_pos)  # Restore original position
            print(f"Loaded document index with {len(self.doc_index)} entries")
        except Exception as e:
            print(f"Failed to load document index: {e}")
            self.doc_index_loaded = False
            
    def get_document(self, doc_idx):
        """Direct access to a specific document by index"""
        if not self.doc_index_loaded:
            raise ValueError("Document index not loaded - cannot directly access documents")
            
        if doc_idx < 0 or doc_idx >= len(self.doc_index):
            raise IndexError(f"Document index {doc_idx} out of range")
            
        # Check cache first
        for cached_idx, doc in self.doc_cache:
            if cached_idx == doc_idx:
                return doc
                
        # Not in cache, load from disk
        row = self.doc_index[doc_idx]
        document = self.ds_pointers[row[0]][row[1]]
        
        # Add to cache
        self.doc_cache.append((doc_idx, document))
        return document
        
    def seek(self, chunk_idx):
        """Seek to a specific chunk index in the dataset"""
        if not self.doc_index_loaded:
            raise ValueError("Document index not loaded - cannot seek to chunk")
            
        # Find which document contains this chunk
        cumulative_chunks = 0
        for doc_idx, row in enumerate(self.doc_index):
            doc_chunks = row[2]  # max_chunks field
            if cumulative_chunks <= chunk_idx < cumulative_chunks + doc_chunks:
                # Found the document containing this chunk
                self.current_doc_idx = doc_idx
                self.current_document = self.get_document(doc_idx)
                self.current_document_step = chunk_idx - cumulative_chunks
                return
            cumulative_chunks += doc_chunks
            
        raise IndexError(f"Chunk index {chunk_idx} out of range")
        
    def __len__(self):
        if self.dist and self.len is not None:
            if self.doc_index_loaded:
                # Calculate chunks for this GPU's documents
                my_chunks = sum(self.doc_index[i][2] for i in range(self.my_doc_start, self.my_doc_end))
                return my_chunks
            else:
                return self.len // self.world_size
        return self.len
        
    def _load_next_document(self):
        if self.doc_index_loaded and self.dist:
            # Multi-GPU with index: load from our assigned document range
            self.current_doc_idx += 1
            if self.current_doc_idx >= self.my_doc_end:
                self.current_document = None
                return
                
            self.current_document = self.get_document(self.current_doc_idx)
            self.current_document_step = 0
            self.current_document_remainder = (
                self.doc_index[self.current_doc_idx][3] 
                if len(self.doc_index[self.current_doc_idx]) > 3 and self.mdat_header['structs'].get('token_remainder') 
                else 0
            )
        elif self.doc_index_loaded:
            # Single GPU with index: sequential loading
            self.current_doc_idx += 1
            if self.current_doc_idx >= len(self.doc_index):
                self.current_document = None
                return
                
            self.current_document = self.get_document(self.current_doc_idx)
            self.current_document_step = 0
            self.current_document_remainder = (
                self.doc_index[self.current_doc_idx][3] 
                if len(self.doc_index[self.current_doc_idx]) > 3 and self.mdat_header['structs'].get('token_remainder') 
                else 0
            )
        else:
            # Fallback: original sequential reading from file
            try:
                row_data = self.fp.read(self.struct_size)  
                if not row_data:
                    raise EOFError
                row = struct.unpack(self.struct_format, row_data)  
                self.current_document = self.ds_pointers[row[0]][row[1]]
                self.current_document_step = 0
                self.current_document_remainder = row[3] if len(row) > 3 and self.mdat_header['structs'].get('token_remainder') else 0 
            except (EOFError, StopIteration):
                self.current_document = None

    def _skip_one(self):
        try:
            if self.modality == 'tokens':
                self._next_tokens()
            elif self.modality == 'patches':
                self._next_patches()
            else:
                self._next_bytes()
        except StopIteration:
            raise        
    def __iter__(self):
        # Una modifica estremamente sporca per consentire il training multi-gpu. Bisogna cambiarlo con uno sharding appropriato. In questo modo si limita a saltare gli esempi visti dalle altre gpu (inefficiente!)
        if self.dist:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()      

        self.i = 0
        return self

    def __next__(self):
        if self.dist:
            while self.i % self.world_size != self.rank:
                self._skip_one()
                self.i += 1

        if self.modality == 'tokens':
            sample = self._next_tokens()
        elif self.modality == 'patches':
            sample = self._next_patches()
        elif self.modality=='bytes':
            sample = self._next_bytes()
        elif self.modality=='entropy_patches':
            sample=self._next_entropy_segment()
        else:
            print(f"ERRORE NON IMPLEMENTATO {self.modality}")

        if self.dist:
            self.i += 1
        return sample

    def _next_tokens(self):
        # I expect in the document dictionary the field "tokens_chunks"
        if self.current_document is None:  
            raise StopIteration
        if "tokens_chunks" not in self.current_document.keys():  
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
            
    def _load_next_segment_group(self):
        if self.current_document is None:
            raise StopIteration
        if "text" not in self.current_document:
            raise KeyError("Document missing 'text' field.")

        collected_chunks = []
        while self.current_document:
            chars = self._next_bytes(force_return_type='text', exceed_doc=False)  # <= n_bytes
            if not chars:
                break

            if getattr(self, "dummy_entropy_mode", False):
                # === Dummy entropy mode ===
                words = list(re.finditer(r"\S+", chars))
                start = 0
                new_chunks = []
                for w in words:
                    end = w.end()
                    # Force a cut before exceeding hard_limit
                    if end - start > self.hard_limit:
                        if start < w.start():
                            new_chunks.append(chars[start:w.start()])
                        start = w.start()
                    # With some probability, cut here if still under the limit
                    elif random.random() < 0.3:
                        new_chunks.append(chars[start:end])
                        start = end
                if start < len(chars):
                    new_chunks.append(chars[start:len(chars)])

            else:
                # === Normal entropy mode ===
                new_chunks = self.compute_entropy(
                    self.entropy_model,
                    chars,
                    return_type='chunks',
                    smoothing=self.entropy_smoothing,
                    hard_limit=self.hard_limit
                )

            collected_chunks.extend(new_chunks)

            if len(collected_chunks) >= self.chunk_size:
                break

        self.current_segment_group = collected_chunks
        self.current_segment_step = 0
    def _next_entropy_segment(self):
        if not hasattr(self, 'current_segment_group') or self.current_segment_group is None:  
            self._load_next_segment_group()
        
        while True: 
            start = self.current_segment_step * self.chunk_size  
            end = start + self.chunk_size
            
            if start >= len(self.current_segment_group):  
                self._load_next_segment_group()
                continue  
                
            self.current_segment_step += 1
            segments_output = self.current_segment_group[start:end]  
            if self.return_type == 'ids':
                
                pass  
            elif self.return_type == 'text':
                return segments_output
            else:  # return_type == 'dict'
                return {
                    'text': segments_output,
                    'ids': None 
                }
            
            return segments_output  

        
        

    def _next_document(self):
        document=self.current_document
        self._load_next_document()
        return document    
        
    def _next_bytes(self, force_return_type=None, exceed_doc=True):
        if force_return_type is not None:
            return_type=force_return_type
        else:
            return_type=self.return_type
        while self.current_document:
            document = self.current_document.get('text')
            if not document:
                self._load_next_document()
                continue
                
            start = self.current_document_step
            if start >= len(document):
                self._load_next_document()
                if exceed_doc:
                    continue
                else:
                    return None
                
            chunk = document[start:start + self.n_bytes]
            if start + len(chunk) < len(document):  # Not at document end
                words = chunk.split(' ')
                if len(words) > 1:  
                    chunk = ' '.join(words[:-1]) + ' '  # Keep all but last partial word
            
            if not chunk:
                chunk = document[start:start + 1]
                
            self.current_document_step = start + len(chunk)
            if return_type=='ids':
                return self.byte_tokenizer.encode(chunk)
            elif return_type=='text':
                return chunk
            else:
                return {
                'ids':self.byte_tokenizer.encode(chunk),
                'text':chunk
                }
            
        raise StopIteration
