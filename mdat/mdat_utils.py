import os
import json

def sanify_name(_input):
	return ''.join(c if c.isalnum() or c in '_-' else '_' for c in _input)
	
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
class SubMdatManifest():
    def __init__(self, mdat_path, submdat_name):
        if not os.path.isdir(mdat_path): 
            print(f"ERROR: {mdat_path} not existing.")
            return None
        self.submdat_path = os.path.join(mdat_path, submdat_name)
        if not os.path.isdir(self.submdat_path): 
            print(f"ERROR: {submdat_name} not existing in {mdat_path}.")
            return None     
        self.submdat_name = submdat_name
        self.manifest_path = os.path.join(self.submdat_path, 'manifest.json')
        self.manifest = self.load_manifest()
    
    def load_manifest(self):  
        if not os.path.exists(self.manifest_path):
            return None
        try:
            with open(self.manifest_path, "r") as m:
                return json.loads(m.read())
        except Exception as e:
            print(f"Caught exception {e} in loading the current manifest {self.manifest_path}. Current manifest is None")
            return None
    
    def write_manifest(self): 
        try:
            with open(self.manifest_path, "w") as m:  
                return m.write(json.dumps(self.manifest)) 
        except Exception as e:
            print(f"Caught exception {e} in writing the manifest {self.manifest_path}") 
            return None 
    
    def new_manifest(self, name, raw_data_bytes, raw_meta_bytes, db_disk_bytes, data_compressed,meta_compressed, map_size, documents_number,data_type='text',data_key='text', hybrid_data=[]):
        if self.manifest is None: 
            self.manifest = {
                "type": "sub-mdat",
                "name": name,
                "data_type":data_type,
                "data_key":data_key,
                "hybrid_data":hybrid_data,
                "raw_data_bytes": raw_data_bytes,  
                "raw_meta_bytes": raw_meta_bytes,  
                "db_disk_bytes": db_disk_bytes,    
                "data_compressed_level": data_compressed,                      
                "meta_compression_level": meta_compressed,  
                "map_size": map_size,
                "documents_number": documents_number, 
                "tokenizers": []
            }
            self.write_manifest() 
        else:
            print("Can't write a new manifest: the manifest already exists")
    
    def add_tokenizer(self, tokenizer_type, tokenizer_name, total_tokens,precomputed_lengths, stats, segmentations=None,data_type=int, word_size=16):
        tokenizer_name=sanify_name(tokenizer_name)
        os.makedirs(os.join(os.join(self.submdat_path,'pretok'),tokenizer_name), exist_ok=True)
        new_tokenizer = {
            "tokenizer_type": tokenizer_type,
            "tokenizer_name": tokenizer_name,
            "total_tokens": total_tokens,
            "data_type": data_type,
            "word_size": word_size,
            "segmentations":None,
            "precomputed_lengths": precomputed_lengths,
            "stats": stats
        }
        self.manifest['tokenizers'].append(new_tokenizer) 
        self.write_manifest() 
    
    def read(self):
        return self.manifest
