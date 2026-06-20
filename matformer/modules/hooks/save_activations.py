
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
        self.activation_cache=cache.storage['activation_cache'][layer_idx][position]
        self.metadata_cache=cache.storage['metadata_cache']
        self.move_to_cpu=move_to_cpu
        self.save_metadata=save_metadata

    def forward(self, x, position, *args, **kwargs):
        if save_metadata and 'metadata' in x.extra_attributes.keys():
          self.metadata_cache=x.extra_attributes['metadata']
        self.activation_cache=x.to('cpu' if self.move_to_cpu else x.device) #The tensor is saved as-it-is  (Question to LLM: Will this cause the returned X to be moved to CPU or will it stay to the original device (which is what I want)?)
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
        self.activation_cache=cache.storage['activation_cache']
        self.metadata_cache=cache.storage['metadata_cache']
    def forward(self,x, *args, **kwargs):
        self.activation_cache=None
        self.metadata_cache=None
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
    def __init__(self,config,cache, filename, layer_idx=None,position=None, detokenize=True, hf_tokenizer_name=None, save_tokenizer=True):
        super().__init__()
        self.activation_cache=cache.storage['activation_cache']
        self.metadata_cache=cache.storage['metadata_cache']
        self.initialize_database()
        self.object_counter=0 # The global counter for un-batched objects
        self.tokenizer=AutoTokenizer.from_pretrained(hf_tokenizer_name)
        self.detokenize=detokenize
        self.save_tokenizer=save_tokenizer
    def initialize_database(self):

        # 2. If save_tokenizer is set, insert into th DB a copy of the tokenizer's vocabulary (tuple, ID => token)
        pass
    def store_in_db():
        to_be_stored=[]
        for layer_idx, content in self.activation_cache.items():
            for position, tensordc in content.items():
                # "Tensordc" may be any TensorDC object, so it could be Unpadded, Padded or a singular tensor.
                # In any case, we need to get a list of tensors if model is working in a batched regime
                tensors=tensordc.pad().detach() # "Detach" returns a list of tensors, without any padding
                # All the tensor in the list are converted into a list
                tensors = [x.tolist() for t in tensors] 
                for i,tensor in enumerate(tensors):
                    # If there are metadata available, we need to unpack them for each object
                    if self.metadata_cache is not None:
                        try:
                            meta=metadata_cache[i]
                        except:
                            raise ValueError('FATAL: Metadata cache present but item not found')
                    else:
                        meta=None
                    # If the "detokenize" option is set, insert also the reconstructed text into the DB
                    if self.detokenize and self.tokenizer is not None and position=='pre_embed':
                        text=self.tokenizer.decode(tensor)
                    else:
                        text=None
                    item={
                        'id':self.object_counter,
                        'layer_idx':layer_idx,
                        'model_position':position,
                        'metadata':meta,
                        'text':text,
                        'value':tensor
                    }
                    self.object_counter+=1
                    self.to_be_stored.append(item)
    def forward(self, x, *args, **kwargs):
        self.store_in_db()
        return x