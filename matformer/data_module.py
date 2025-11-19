import pytorch_lightning as pl
from torch.utils.data import DataLoader
from matformer.model_config import ModelConfig  
import os
from matformer.mdat import MatformerDataset
import torch
import torch.distributed as dist
from matformer.tensors_dataclasses import TensorDC, NormalTensor, PaddedTensor, UnpaddedTensor


class MatformerDataModule(pl.LightningDataModule):
    def __init__(self, mdat_path: str, iteration_modality, pad_token_id: int, 
                 varlen_strategy='unpadding', with_meta=False, max_seq_len=None, 
                 mdat_strategy=None, mdat_view=None, batch_size=None,distributed=True,num_devices=1):
        super().__init__()
        self.mdat_path = mdat_path
        self.iteration_modality = iteration_modality
        self.with_meta = with_meta
        self.pad_token_id = pad_token_id
        self.varlen_strategy = varlen_strategy
        self.max_seq_len = max_seq_len
        self.mdat_strategy = mdat_strategy
        self.mdat_view = mdat_view
        self.batch_size = batch_size
        self.num_devices=num_devices
        self.distributed_initialized=False
    def setup(self, stage=None):
        if dist.is_initialized():
            self.distributed_initialized=True
        # Initialize dataset here for proper distributed handling
        self.mdat = MatformerDataset.load_dataset(
            path=self.mdat_path,
            readonly=True,
            distributed=dist.is_initialized(),
            shuffle=True,
            ds_view=self.mdat_view,
            batch_size=self.batch_size
        )
        if self.mdat_view is not None:
            self.mdat.set_view(self.mdat_view)        
        if self.mdat_strategy is not None:
            self.mdat.set_strategy(self.mdat_strategy,max_seq_len=self.max_seq_len) 
            
        self.mdat.set_iteration_modality(self.iteration_modality, with_meta=self.with_meta)
        print(f"Len attuale: {len(self)}")
    def collate_fn(self, batch):        
        if batch is None:
            print("WARNING: GOT A None TOKEN SEQUENCES FROM THE DATALOADER!")
            batch = [] 
            
        if self.varlen_strategy == 'nested':
            sequence = torch.nested.nested_tensor(batch, layout=torch.jagged)
            return sequence
        padded_ids = []
        for item in batch:
            if isinstance(item, dict):
                _object = item["object"]
                if item.get("worker_has_finished", False):
                    pass #WIP
            else:
                _object = item

            #padding
            padded_ids.append(
                _object + [self.pad_token_id] * (self.max_seq_len - len(_object))
            )

        tensors = torch.tensor(padded_ids, dtype=torch.long)
        padding_masks = (tensors == self.pad_token_id)
        sequence = PaddedTensor(tensor=tensors, padding_mask=padding_masks)

        if self.varlen_strategy == "unpadding":
            sequence = sequence.unpad()

        return {'sequence':sequence,"worker_has_finished":item.get("worker_has_finished")}        
        """    
        # Pad sequences
        padded_ids = [ids + [self.pad_token_id] * (self.max_seq_len - len(ids)) for ids in batch]
        tensors = torch.tensor(padded_ids, dtype=torch.long)
        padding_masks = (tensors == self.pad_token_id)
        sequence = PaddedTensor(tensor=tensors, padding_mask=padding_masks)
        
        if self.varlen_strategy == 'unpadding':
            sequence = sequence.unpad()   
            
        return sequence 
        """      

    def __len__(self):
        if self.num_devices == 1 or self.distributed_initialized:
            return len(self.mdat) if hasattr(self, 'mdat') else 0
        else:
            return self.mdat.get_distributed_length_before_training(num_devices=self.num_devices)
            
    def train_dataloader(self):
        return DataLoader(
            self.mdat,
            batch_size=self.batch_size,
            num_workers=0,
            collate_fn=self.collate_fn,  
            shuffle=False  
        )

