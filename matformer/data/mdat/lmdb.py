"""LMDB byte storage and compression; the on-disk format is unchanged."""

import os
import lmdb
import orjson

USE_ZLIB = os.environ.get('USE_ZLIB', 'false').lower() == 'true'

if USE_ZLIB:
    import zlib
    compress_func = lambda data, level: zlib.compress(data, level)
    decompress_func = lambda data: zlib.decompress(data)
    COMPRESSION_BACKEND = "zlib"
else:
        import zstandard as zstd
        _zstd_compressors = {}
        _zstd_decompressor = zstd.ZstdDecompressor()

        def compress_func(data, level):
            if level not in _zstd_compressors:
                _zstd_compressors[level] = zstd.ZstdCompressor(level=level)
            return _zstd_compressors[level].compress(data)

        def decompress_func(data):
            return _zstd_decompressor.decompress(data)

        COMPRESSION_BACKEND = "zstd"

class LMDBDataset:
    def __init__(self, path, readonly=True, lock=False, compressed=False, compression_level=0,map_size=1<<44,batch_size=50000):
        self.env = lmdb.open(
            path,
            subdir=os.path.isdir(path),
            readonly=readonly,
            lock=lock,
            readahead=True,
            meminit=False,
            max_readers=1,
            map_size=map_size
        )
        self.map_size=map_size
        self.path=path
        self.readonly=readonly
        self.compressed=compressed
        self.compression_level=compression_level
        self.batch_size = batch_size
        self.batch_buffer = []
        self.compression_attempts=100 #Trying to avoid compression bugs
        self.compression_backend = COMPRESSION_BACKEND
        with self.env.begin(write=False) as txn:
            try:
                self.length = int(txn.get(b"__len__").decode())
            except:
                self.length=0
    def set_batch_size(self,batch_size):
        self.batch_size=batch_size
    def __len__(self):
        return self.length
    def start_transaction(self):
        self.txn=self.env.begin(write=True)
    def close_transaction(self):
        pass


    def write(self, obj, key=None, batched=False):
        """
        "Obj" can be:
            * A byte object, directly inserted
            * Other types, orjson serialized
        If batched==True:
            * A list of data is expected
        Return "current_index" if data is correctly added, None if error happened
        """

        if key==None and batched==False:
            key=self.length
        if key is not None and batched==True:
            print("LMDBDataset Warning: You should not provide a key if the input data is batched! Key value is ignored.")

        batch = obj if isinstance(obj, list) else [obj]
        if len(batch) > self.batch_size:
            raise ValueError(f"Batch size {len(batch)} exceeds maximum {self.batch_size}")

        for data in batch:
            if not isinstance(data, bytes):
                data = orjson.dumps(data)  # Serialize non-bytes objects, for example python dictionaries
            if self.compressed:
                data = compress_func(data, self.compression_level)
            self.batch_buffer.append((str(self.length).encode(), data))
            self.length += 1

            if len(self.batch_buffer) >= self.batch_size:
                self._flush_batch()
    def write_batch_with_keys(self, key_data_pairs):
            """
            Write multiple (key, data) pairs using the existing batch buffer mechanism.
            This leverages the existing batching and auto-flushing logic.

            Args:
                key_data_pairs: list of (key, data) tuples where key is the document index
                               and data is the object to store
            """
            if not key_data_pairs:
                return

            for key, data in key_data_pairs:
                if not isinstance(data, bytes):
                    data = orjson.dumps(data)
                if self.compressed:
                    data = compress_func(data, self.compression_level)

                # Add to batch buffer - this will auto-flush when batch_size is reached
                self.batch_buffer.append((str(key).encode(), data))

                if len(self.batch_buffer) >= self.batch_size:
                    self._flush_batch()
    def _flush_batch(self):
            if not self.batch_buffer:
                return
            with self.env.begin(write=True) as txn:
                for key, data in self.batch_buffer:
                    txn.put(key, data)
            self.batch_buffer.clear()

    def close(self):
            if self.env is None:
                return
            self._flush_batch()  # Flush remaining items
            if self.readonly == False:
                with self.env.begin(write=True) as txn:
                    txn.put(b"__len__", str(self.length).encode())  # Synchronizing correct length
                self.env.sync()
            self.env.close()
            self.env = None



    def __getitem__(self, key):
        key=str(key).encode()
        with self.env.begin(write=False) as txn:
            data = txn.get(key)
            if data is None:
                raise IndexError(f"Index {key} not found in LMDB dataset.")
            if self.compressed:
                try:
                    #print("Attenzione! AL momento sqlite segna compresso ma dati non compressi. Facile da risolvere")
                    return decompress_func(data)
                except:
                    safe_path = ''.join(c if c.isalnum() or c in '_-' else '_' for c in self.path)
                    print(f"LMDBDataset {self.path} WARNING: error loading data at index {key}.")
                    print(f"It seems there is a bug with {self.compression_backend}.")
                    print(f"First, let's try to load the data again {self.compression_attempts} times.")
                    for i in range(1,self.compression_attempts):
                        try:
                            x=decompress_func(data)
                            print(f"It worked at attemp: {i}")
                            with open(f"compression_error_logs_{safe_path}.txt","a") as l:
                                l.write(f"{self.compression_backend} error at {i}. Recovered after {i}/{self.compression_attempts} attempts.\n")
                            return x
                        except:
                            pass
                    print(f"It didn't worked after {self.compression_attempts}. Returning a dummy structure to avoid breaking training. Event logged in ./compression_error_logs_{safe_path}.txt")
                    with open(f"compression_error_logs_{safe_path}.txt","a") as l:
                        l.write(f"{self.compression_backend} error at {i}. Not recovered after {self.compression_attempts} attempts.\n")
                    return None
            else:
                return data
