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
        stacked_recurrence_masks = []
        any_worker_finished = False

        for item in batch:
            recurrent_same = None
            worker_finished = False
            if isinstance(item, dict):
                _object = item["object"]
                if _object is None:
                    print("WARNING: Get a None from the Mdat")
                    print(item)
                    continue
                worker_finished = bool(item.get("worker_has_finished", False))
                recurrent_same = item.get("is_same_document", None)
            else:
                _object = item
            ### To debug the resume of checkpoint
            debug_path = os.environ.get("DEBUG_DATALOADER_SAVE", None)
            if debug_path is not None:
                import json
                if not hasattr(self, "_debug_save_index"):
                    self._debug_save_index = 0
                try:
                    import torch.distributed as dist
                    if dist.is_available() and dist.is_initialized():
                        rank = dist.get_rank()
                    else:
                        rank = 0
                except Exception:
                    rank = 0
                debug_file = f"{debug_path}_rank{rank}.jsonl"
                def _to_serializable(x):
                    if torch.is_tensor(x):
                        return x.detach().cpu().tolist()
                    return x
                record = {
                    "index": self._debug_save_index,
                    "item": _to_serializable(item),
                }
                with open(debug_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record))
                    f.write("\n")
                self._debug_save_index += 1
                
            any_worker_finished = any_worker_finished or worker_finished
            if recurrent_same is True:
                stacked_recurrence_masks.append(True)
            elif recurrent_same is False:
                stacked_recurrence_masks.append(False)
            else:
                stacked_recurrence_masks.append(None)

            assert self.max_seq_len is not None
            padded = _object + [self.pad_token_id] * (self.max_seq_len - len(_object))
            padded_ids.append(padded)
        
        # For recurrence
        if any(mask is not None for mask in stacked_recurrence_masks):
            stacked = [
                m if m is not None else False
                for m in stacked_recurrence_masks
            ]
            recurrence_batch_mask = torch.tensor(stacked, dtype=torch.bool)  # shape [B]
        else:
            recurrence_batch_mask=None
            #extra_follow_keys=None

        # padded ids -> tensor
        tensors = torch.tensor(padded_ids, dtype=torch.long)
        padding_masks = (tensors == self.pad_token_id)
        sequence = PaddedTensor(tensor=tensors, padding_mask=padding_masks, recurrence_mask=recurrence_batch_mask)
        


        if self.varlen_strategy == "unpadding":
            sequence = sequence.unpad()

        return {
            "sequence": sequence,
            "worker_has_finished": any_worker_finished
        }
         
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
        dataloader = DataLoader(
            self.mdat,
            batch_size=self.batch_size,
            num_workers=0,
            collate_fn=self.collate_fn,
            shuffle=False
        )
        dataloader._is_resumable = True
        return dataloader


    def state_dict(self):
        try:
            import torch.distributed as dist
            is_dist = dist.is_available() and dist.is_initialized()
        except Exception:
            is_dist = False

        this_rank_state = self.mdat.get_state_dict()

        # Single GPU
        if not is_dist:
            return {"per_rank": [this_rank_state]}

        
        # Distributed branch
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if rank == 0:
            states = [None for _ in range(world_size)]
            # rank 0 receives from all ranks (including itself)
            dist.gather_object(this_rank_state, object_gather_list=states, dst=0)
            return {"per_rank": states}
        else:
            # non-zero ranks send to rank 0
            dist.gather_object(this_rank_state, dst=0)
            return {}


    def load_state_dict(self, state_dict):
        per_rank = state_dict.get("per_rank", None)
        if per_rank is None:
            return  # checkpoint did not contain data
        try:
            import torch.distributed as dist
            is_dist = dist.is_available() and dist.is_initialized()
            rank = dist.get_rank() if is_dist else 0
            world_size = dist.get_world_size() if is_dist else 1
        except Exception:
            is_dist = False
            rank = 0
            world_size = 1

        saved_world_size = state_dict.get("saved_world_size")
        STRICT_WORLD_SIZE_CHECK = True #It will not work if GPU count changes
        if is_dist and saved_world_size != world_size:
            if STRICT_WORLD_SIZE_CHECK:
                raise RuntimeError(
                    f"Cannot restore dataset state: world size changed "
                    f"(saved={saved_world_size}, current={world_size})."
                )
            else:
                mapped_rank = rank % len(per_rank) #Rivedere la logica se si vuole supportare cambio numero GPU
        else:
            mapped_rank = rank if rank < len(per_rank) else rank % len(per_rank)
            
        state_dict_rank = per_rank[mapped_rank]
        self.mdat.restore_state_dict(state_dict_rank)
        print("Restoring state dict")
        print(self.mdat.document_index)
        print(self.mdat.current_chunk_step)
        if is_dist:
            dist.barrier()

    
    
