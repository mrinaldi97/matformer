
"""
A Matformer hook to store activations into a database.
This can be very useful for things such as mechanistic interpretability studies or model debug. 

"""
from matformer.matformer_registry import registry
@registry.register(
    "hooks",
    "activation_cache",
    "default",
    requires=["torch"],
    priority=0,
)
class ActivationCache:
    """
    This module stores the models' activation into the cache.
    All activations will be stored into self.cache.storage in a standard format:
        self.cache.storage[activation_cache][layer_idx][model_position]
    where layer_idx is the layer. Layer will be "-1" for positions that are outside the transformer blocks, such as pre_embed or final_output

    Arguments:
        self
        config
        cache
        layer_idx
        position
        move_to_cpu => Move all tensors to cpu, default to False
        save_metadata => If the tensor has a 'metadata' key, save it. Default to False
    Because metadata are useful only once per sequence, it is suggested to set it to "True" only the first time you call the hook into the model (ex. in the pre_embed position).
    Metadata will be stored in self.cache.storage[metadata_cache], independently of layer_idx and position

    It is your choice how to actually use the cache. You could call another hook at the output of the model to store it in an external file, for example,
    or use it interactively. If you do not consume the cache, it will be overwritten at the next forward pass.

    """
    def __init__(self, config, cache, layer_idx, position, move_to_cpu=False, save_metadata=True):
        super().__init__()
        if layer_idx is None:
            layer_idx=-1 #For saving stuff outside the transformer's layers, such as embeddings, logits...
        self.cache=cache
        if 'activation_cache' not in cache.storage.keys():
            cache.storage['activation_cache']={}
        if 'metadata_cache' not in cache.storage.keys():
            cache.storage['metadata_cache']=None
        self.layer_idx=layer_idx
        self.position=position
        self.move_to_cpu=move_to_cpu
        self.save_metadata=save_metadata

    def forward(self, x, position, *args, **kwargs):
        if self.save_metadata and 'metadata' in x.extra_attributes.keys():
            self.cache.storage['metadata_cache']=x.extra_attributes['metadata']
        self.cache.storage['activation_cache'].setdefault(self.layer_idx, {})[self.position]=x.to('cpu' if self.move_to_cpu else x.device)
        return x

@registry.register(
    "hooks",
    "activation_cache_cleaner",
    "default",
    requires=["torch", "gc"],
    priority=0,
)
class ActivationCacheCleaner:
    """
    A very simple module to be used with ActivationCache.
    It just clears the cache.
    """
    def __init__(self,config,cache,layer_idx,position):
        super().__init__()
        import torch
        import gc
        self.cache=cache

    def forward(self,x, *args, **kwargs):
        import torch
        import gc
        self.cache.storage['activation_cache'].clear()
        self.cache.storage['metadata_cache']=None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return x

@registry.register(
    "hooks",
    "activation_duckdb_store",
    "default",
    requires=["torch", "duckdb"],
    priority=0,
)
class ActivationDuckDBStore:
    """
    Save self.cache.storage['activation_cache'] into a DuckDB file every forward pass.
    pre_embed (raw token ids) is treated as a first-class column on `sequence` rather than a generic
    activation, since it's what you join/group on for per-token statistics. Everything else
    (including logits, if hooked) goes into 'activation_store' unchanged.
    """
    def __init__(self, config, cache, filename: str, detokenize: bool=True, hf_tokenizer_name: str|None=None, save_tokenizer: bool=True, **kwargs):
        super().__init__()
        import duckdb
        self.cache=cache
        self.object_counter=0 # The global counter for un-batched objects
        self.detokenize=detokenize
        self.save_tokenizer=save_tokenizer
        self.tokenizer=AutoTokenizer.from_pretrained(hf_tokenizer_name) if hf_tokenizer_name else None
        self.con=duckdb.connect(filename)
        self.initialize_database()

    def initialize_database(self) -> None:
        self.con.execute("CREATE TABLE IF NOT EXISTS sequence (id BIGINT PRIMARY KEY, meta JSON, text VARCHAR, token_ids INTEGER[])")
        self.con.execute("CREATE TABLE IF NOT EXISTS token (id BIGINT PRIMARY KEY, string VARCHAR)")
        self.con.execute("CREATE TABLE IF NOT EXISTS activation_store (sequence_id BIGINT, layer_idx INTEGER, position VARCHAR, value FLOAT[][], PRIMARY KEY (sequence_id, layer_idx, position))")
        # If save_tokenizer is set, insert into the DB a copy of the tokenizer's vocabulary (ID => token)
        if self.save_tokenizer and self.tokenizer is not None:
            vocab=self.tokenizer.get_vocab()
            self.con.executemany("INSERT OR IGNORE INTO token VALUES (?, ?)", [(i, s) for s, i in vocab.items()])

    def store_in_db(self) -> None:
        activation_cache=self.cache.storage['activation_cache']
        metadata_cache=self.cache.storage['metadata_cache']
        seq_ids, token_ids_by_seq, text_by_seq, activations = None, {}, {}, []
        for layer_idx, content in activation_cache.items():
            for position, tensordc in content.items():
                tensors=[t.tolist() for t in tensordc.pad().detach()] # unpadded per-sequence tensors
                if seq_ids is None:
                    seq_ids=range(self.object_counter, self.object_counter+len(tensors))
                if position=='pre_embed': # raw token ids: first-class on `sequence`, not a generic activation
                    for sid, tensor in zip(seq_ids, tensors):
                        token_ids_by_seq[sid]=tensor
                        if self.detokenize and self.tokenizer is not None:
                            text_by_seq[sid]=self.tokenizer.decode(tensor)
                    continue
                activations.extend((sid, layer_idx, position, tensor) for sid, tensor in zip(seq_ids, tensors))

        sequences=[(s, json.dumps(metadata_cache[i]) if metadata_cache is not None else None, text_by_seq.get(s), token_ids_by_seq.get(s)) for i, s in enumerate(seq_ids)]
        self.con.executemany("INSERT INTO sequence VALUES (?, ?, ?, ?)", sequences)
        self.con.executemany("INSERT INTO activation_store VALUES (?, ?, ?, ?)", activations)
        self.object_counter+=len(seq_ids)

    def forward(self, x: TensorDC, *args, **kwargs) -> TensorDC:
        self.store_in_db()
        return x