import sys
sys.path.append('../') # Da sostituire con pyproject.toml eccetera ok per oraimport torch
from torch.utils.data import Dataset
from matformer.tensors_dataclasses import PaddedTensor, NormalTensor
import torch
import random
import json
class JSONLDataset(Dataset):
    def __init__(self, path, debug=False):
        with open(path, 'r', encoding='utf-8') as f:
            self.data=[]
            for i,line in enumerate(f):
                self.data.append(json.loads(line))
                #if i >=301:
                #    break
            #self.data = [json.loads(line) for line in f]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
        
class RandomDataset(Dataset):
    def __init__(self, max_len=27, max_items=200000000):
        self.max_items=max_items
        self.max_len=max_len
    
    def __len__(self):
        return self.max_items
    
    def __getitem__(self, idx):
        return "".join([chr(random.randint(0x0000, 0x058F)) for i in range(random.randint(1,self.max_len))])
        
def generate_random_batch(max_len=27, pad_token=-1, batch_size=16, device='cuda'):
    """Generate random sequence batch for curriculum learning."""
    lengths = torch.randint(1, max_len + 1, (batch_size,), device=device)
    tensors = []
    for length in lengths:
        seq = torch.randint(0, 256, (length.item(),), device=device)
        pad = torch.full((max_len - length.item(),), pad_token, device=device)
        tensors.append(torch.cat([seq, pad]))
    tensors = torch.stack(tensors).int()
    padding_mask = tensors == pad_token
    return PaddedTensor(tensor=tensors, padding_mask=padding_mask), NormalTensor(tensor=lengths.unsqueeze(-1))

def encode_texts(texts, max_len, pad_token, eos_token):
    """Encode text strings to tensors."""
    tokens, lengths = [], []
    for text in texts:
        l=[x for x in text.encode('utf-8')]
        l.append(eos_token)
        l.append(eos_token)
        encoded = torch.tensor(l, dtype=torch.int32)
        length = min(len(encoded), max_len)
        lengths.append(length)
        encoded = encoded[:length]
        if length < max_len:
            encoded = torch.cat([encoded, torch.full((max_len - length,), eos_token, dtype=torch.int32)])
        #if length < max_len:
        #    encoded = torch.cat([encoded, torch.full((max_len - length,), pad_token, dtype=torch.int32)])            
        tokens.append(encoded)
    x = torch.stack(tokens)
    return PaddedTensor(tensor=x, padding_mask=(x == pad_token)), NormalTensor(tensor=torch.tensor(lengths, dtype=torch.int32).unsqueeze(-1))

def collate_fn(batch, max_len, pad_token, eos_token):
    """Collate function for DataLoader."""
    return encode_texts(batch, max_len, pad_token, eos_token)
