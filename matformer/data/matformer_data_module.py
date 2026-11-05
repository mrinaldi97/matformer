"""Connect an MDAT view/selection to the training loop and tensor collation."""
from torch.utils.data import DataLoader
import torch
from pathlib import Path
from matformer.data.mdat.mdat import MatformerDataset
import torch.distributed as dist

class MatformerDataModule:
    """Load a saved view and optionally restrict domains and token budgets.

    ``mdat_selection`` accepts ``domains``, ``max_tokens``, ``domain_tokens``,
    and ``seed``; see :class:`DatasetSelection`. Existing configs need no change.
    """
    def __init__(self, mdat_path: str, iteration_modality, pad_token_id: int,
                 varlen_strategy='unpadding', with_meta=False, max_seq_len=None,
                 mdat_strategy=None, mdat_view=None, batch_size=None,distributed=True,num_devices=1,
                 mdat_selection=None, prefetch_buffer=32768):
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
        self.distributed = distributed
        self.mdat_selection = mdat_selection
        self.prefetch_buffer = prefetch_buffer
        self.distributed_initialized=False
    def setup(self, stage=None):
        self.distributed_initialized = self.distributed and dist.is_initialized()
        if hasattr(self, 'mdat'):
            self.mdat.close()
        self.mdat = MatformerDataset.load_dataset(
            path=self.mdat_path,
            readonly=True,
            distributed=self.distributed_initialized,
            shuffle=True,
            ds_view=self.mdat_view,
            batch_size=self.batch_size,
            strategy=self.mdat_strategy,
            max_seq_len=self.max_seq_len,
            selection=self.mdat_selection,
            prefetch_buffer=self.prefetch_buffer,
        )

        self.mdat.set_iteration_modality(self.iteration_modality, with_meta=self.with_meta)
        # Fast identity check for immutable, staged datasets. This deliberately
        # rejects replacing files in place, even if lengths are unchanged.
        paths = [Path(self.mdat.db_path)]
        pointer = Path(self.mdat.shuffling_path) / f'{self.mdat.current_view}.mdat'
        if pointer.exists():
            paths.append(pointer)
        for subdataset in self.mdat.loaded_submdats.values():
            paths.extend(Path(db.path) for db in getattr(subdataset, 'pretok_db', {}).values())
        self.source_identity = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns)
                                for p in sorted(set(paths))]
        if self.mdat.selection_summary is not None:
            print(f"Dataset selection: {self.mdat.selection_summary}")

    def teardown(self, stage=None):
        if hasattr(self, 'mdat'):
            self.mdat.close()
    def collate_fn(self, batch):
        from matformer.data.collator import TokenCollator
        return TokenCollator(self.pad_token_id, self.max_seq_len,
                             self.varlen_strategy, self.with_meta)(batch)

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
            shuffle=False,
            generator=torch.Generator().manual_seed(0),
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
            return {"per_rank": [this_rank_state], 'source_identity': self.source_identity}


        # Distributed branch
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if rank == 0:
            states = [None for _ in range(world_size)]
            # rank 0 receives from all ranks (including itself)
            dist.gather_object(this_rank_state, object_gather_list=states, dst=0)
            return {"per_rank": states, 'source_identity': self.source_identity}
        else:
            # non-zero ranks send to rank 0
            dist.gather_object(this_rank_state, dst=0)
            return {}


    def load_state_dict(self, state_dict):
        if state_dict.get('source_identity', self.source_identity) != self.source_identity:
            raise ValueError('Dataset files changed since the checkpoint; use a new data stage')
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

        if len(per_rank) != world_size:
            raise ValueError(f'Exact resume requires the same world size: saved={len(per_rank)}, current={world_size}')
        self.mdat.restore_state_dict(per_rank[rank])
