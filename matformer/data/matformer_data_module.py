# matformer/data/matformer_data_module.py
#import pytorch_lightning as pl
from torch.utils.data import DataLoader
from matformer.dataclasses.model_dataclasses import ModelConfig
from matformer.dataclasses.data_dataclasses import DataConfig 
import os
from matformer.data.mdat.mdat import MatformerDataset
from matformer.data.lazy_dataset import LazyIterableDataset
import torch
import torch.distributed as dist
from matformer.dataclasses.tensors_dataclasses import TensorDC, NormalTensor, PaddedTensor, UnpaddedTensor,UnpaddedTensor,PaddedTensor
import torch.nn.functional as F
from matformer.matformer_tokenizers import MatformerTokenizer
from typing import Optional, List, Dict, Any
from matformer.matformer_registry import registry

class MatformerDataModule:
    """
    Abstract base class for all Matformer data modules.

    Subclasses must call ``_assertions`` in their ``__init__`` to validate the
    DataConfig, and must implement ``setup``, ``train_dataloader``, ``__len__``,
    ``state_dict``, and ``load_state_dict``.

    **Adding a new data module** — minimal checklist:

    1. Subclass this class and call ``super().__init__(config, num_devices)``.
    2. Define a ``required_keys`` list and call ``self._assertions('your_name', required_keys)``
       to enforce that ``DataConfig`` carries every field your module needs.
    3. Implement ``setup`` (call ``super().setup()`` first to detect distributed init),
       ``train_dataloader`` (return a DataLoader with ``collate_fn=self.collate_fn``),
       ``__len__``, ``state_dict``, and ``load_state_dict``.
    4. Make your dataset yield items as dicts with the keys documented in :meth:`collate_fn`.
       ``collate_fn`` is inherited and requires no override.
    """

    def __init__(self, config: DataConfig, num_devices: None, *args, **kwargs) -> None:
        """
        Args:
            config:  Fully populated DataConfig for this run.
            num_devices:  Number of devices/processes. Used to compute dataset
                          length before the distributed process group is
                          initialised (e.g. for progress-bar sizing).
        """
        super().__init__()
        self.config:             DataConfig = config
        self.num_devices = num_devices
        if getattr(config,"model",None) is not None and getattr(config.model,"max_position_embeddings",None):
            self.config.max_seq_len = config.model.max_position_embeddings
            print(f"Datamodule: Inheriting {self.config.max_seq_len} from the loaded model as maximum sequence length")
        if getattr(config,"training",None) is not None:
            self.num_devices=config.training.device_count
            self.config.varlen_strategy=config.training.varlen_strategy
            if config.training.batch_size is not None: # A training batch_size of None keeps the data section's own value
                self.config.batch_size=config.training.batch_size
            print(f"Datamodule initialized in training mode. Inheriting {self.num_devices} as device count and {self.config.varlen_strategy} as varlen strategy and {self.config.batch_size} as batch size.")

        self.distributed_initialized: bool       = False

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Called once before training starts.  Detects whether the distributed
        process group is already active and caches the result so that
        ``__len__`` and other methods can branch on it cheaply.

        Subclasses must call ``super().setup(stage)`` **before** their own
        dataset construction so that ``self.distributed_initialized`` is set
        when they need it.

        Args:
            stage: Unused; kept for Lightning/trainer compatibility.
        """
        if dist.is_initialized():
            self.distributed_initialized = True


    def collate_fn(self, batch: List[Any]) -> Dict[str, Any]:
        """
        Converts a list of raw dataset items into a training-ready batch dict.

        **Expected item format** — each element of ``batch`` should be a dict
        with the following keys (all optional except ``"object"``):

        * ``"object"`` *(list[int], required)*: token-id sequence for one example.
        * ``"worker_has_finished"`` *(bool)*: ``True`` on the last item a worker
          will ever yield; triggers the zero-loss guard in the model's forward
          pass so distributed training does not hang on an uneven shard.
        * ``"is_same_document"`` *(bool | None)*: recurrence continuity flag.
          ``True`` => this chunk continues the previous one (pass hidden state
          through). ``False`` => boundary, reset state. ``None`` => no recurrence.
        * ``"metadata"`` *(dict | None)*: arbitrary per-example side-data
          forwarded unchanged; only collected when ``config.keep_metadata``
          is ``True``.

        **Output tensor type** is controlled by ``config.varlen_strategy``:

        * ``"unpadding"`` → :class:`UnpaddedTensor`.  Sequences are
          concatenated into a single flat tensor; ``cu_seqlens`` and ``indices``
          allow kernels (e.g. FlashAttention varlen) to reconstruct boundaries
          without materialising padding.
        * anything else (default ``"padding"``) → :class:`PaddedTensor`.
          Sequences are right-padded to ``config.max_seq_len`` with
          ``config.pad_token_id``; a boolean ``padding_mask`` marks the
          pad positions.

        Both types carry ``extra_attributes={'metadata': ...}`` when
        ``keep_metadata`` is ``True``, and a ``recurrence_mask`` tensor of
        shape ``[B]`` when at least one item supplies ``"is_same_document"``.

        Returns:
            Dict with keys:
            * ``"sequence"`` — a :class:`TensorDC` subclass
              (:class:`PaddedTensor` or :class:`UnpaddedTensor`).

        """
        if batch is None:
            print("Warning: Got a None sequence from the dataloader.")
            batch = []

        token_sequences: List[List[int]] = []
        label_values: List[Any] = []
        if self.config.keep_metadata:
            metadata_sequences: Optional[List[Any]] = []
        else:
            metadata_sequences = None
        stacked_recurrence_masks: List[Optional[bool]] = []
        any_worker_finished: bool = False

        for item in batch:
            recurrent_same: Optional[bool] = None
            worker_finished: bool          = False
            if isinstance(item, dict):
                _object = item["object"]
                if _object is None:
                    print("WARNING: Get a None from the Mdat")
                    print(item)
                    continue
                worker_finished = bool(item.get("worker_has_finished", False))
                recurrent_same  = item.get("is_same_document", None)
            else:
                _object = item

            any_worker_finished = any_worker_finished or worker_finished
            if recurrent_same is True:
                stacked_recurrence_masks.append(True)
            elif recurrent_same is False:
                stacked_recurrence_masks.append(False)
            else:
                stacked_recurrence_masks.append(None)

            assert self.config.max_seq_len is not None
            token_sequences.append(_object)
            label_values.append(item.get('labels') if isinstance(item, dict) else None)  
            if self.config.keep_metadata and 'metadata' in item.keys():
                metadata_sequences.append(item['metadata'])
        # For recurrence
        if any(mask is not None for mask in stacked_recurrence_masks):
            stacked = [
                m if m is not None else False
                for m in stacked_recurrence_masks
            ]
            recurrence_batch_mask: Optional[torch.Tensor] = torch.tensor(stacked, dtype=torch.bool)  # shape [B]
        else:
            recurrence_batch_mask = None
        attributes={}
        following_attributes = {}
        if any_worker_finished:
            attributes['worker_has_finished']=True
        if any(l is not None for l in label_values):
            if getattr(self.config, 'task', None) == 'token-level':
                if self.config.varlen_strategy == "unpadding":
                    following_attributes['labels'] = torch.tensor(
                        [l for seq in label_values for l in seq], dtype=torch.long
                    )
                else:
                    padded = [l + [-100] * (self.config.max_seq_len - len(l)) for l in label_values]
                    following_attributes['labels'] = torch.tensor(padded, dtype=torch.long)
            else:
                label_dtype = torch.float32 if getattr(self.config, 'task', None) == 'regression' else torch.long
                attributes['labels'] = torch.tensor(label_values, dtype=label_dtype)
        if self.config.keep_metadata:
            attributes['metadata']=metadata_sequences
        if recurrence_batch_mask is not None:
            attributes['same_document_mask']=recurrence_batch_mask
        
        if self.config.varlen_strategy == "unpadding":
            # Build UnpaddedTensor directly
            seqlens   = torch.tensor([len(seq) for seq in token_sequences], dtype=torch.int32)
            cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))

            # Concatenate all sequences
            unpadded_tensor = torch.tensor(
                [token for seq in token_sequences for token in seq],
                dtype=torch.long
            )

            # Compute indices: position in flattened padded tensor (batch_size * original_seq_len)
            indices: List[int] = []
            for batch_idx, seq_len in enumerate(seqlens):
                base_offset = batch_idx * self.config.max_seq_len
                indices.extend(range(base_offset, base_offset + seq_len))
            indices_tensor = torch.tensor(indices, dtype=torch.long)

            max_seq_len_value = seqlens.max().item() if len(seqlens) > 0 else 0

            sequence: TensorDC = UnpaddedTensor(
                tensor=unpadded_tensor,
                indices=indices_tensor,
                cu_seqlens=cu_seqlens,
                max_seq_len=max_seq_len_value,
                original_seq_len=self.config.max_seq_len,
                batch_size=len(token_sequences),
                attributes=attributes,
            )
        else:
            padded_ids: List[List[int]] = []
            for seq in token_sequences:
                padded = seq + [self.config.tokenizer.pad_token_id] * (self.config.max_seq_len - len(seq))
                padded_ids.append(padded)
            tensors = torch.tensor(padded_ids, dtype=torch.long)
            padding_masks = (tensors == self.config.tokenizer.pad_token_id)
            sequence = PaddedTensor(
                tensor=tensors,
                padding_mask=padding_masks,
                attributes=attributes,
            )
        return sequence
        """
        return {
            "sequence":            sequence,
            "worker_has_finished": any_worker_finished
        }"""


@registry.register("data", "lazy", "default")
class LazyDataModule(MatformerDataModule):
    """
    A data module that supports lazy loading and on-the-fly tokenization.
    Supported datasets are:
        * HuggingFace datasets
        * JSONL files
        * CSV files
    Chunking is not supported by design: sequences are truncated at the max model length.
    If you want to process datasets with long sequences, please switch to MDAT.
    In general, it is better to use MDAT but this LazyDataModule is useful for quick experimentations
    or light tasks such as instruction-finetuning, benchmarking, batched generations and so on.
    Number of examples for jsonl and csv files by default is NOT pre-computed. If you want it (for progress bar and so on),
    please pass precompute_len=True.

    Args:
        config:  DataConfig instance; must have ``dataloader='lazy'`` and all
                      keys listed in ``required_keys`` (asserted on construction).
        num_devices:  Number of devices/processes (used before distributed init).
        tokenizer:    Optional tokenizer forwarded to :class:`LazyIterableDataset`.
    """

    def __init__(
        self,
        config: DataConfig,
        num_devices: int = 1,
        tokenizer: Optional[MatformerTokenizer] = None, *args, **kwargs
    ) -> None:
        super().__init__(config, num_devices)
        self.tokenizer:    Optional[MatformerTokenizer]                  = tokenizer
        self.lazy_dataset: Optional[LazyIterableDataset] = None
        self._detached_dataset: Optional[LazyIterableDataset] = None
        self._length:      Optional[int]                  = None
        print("DEBUG DEBUG HO INIZIATO IL DATAMODULE")
    
    def setup(self, stage: Optional[str] = None) -> None:
        super().setup(stage)
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        self.lazy_dataset = LazyIterableDataset(
            config=self.config,
            tokenizer=self.tokenizer,
            rank=rank,
            world_size=world_size,
        )
        
        if self.config.precompute_len:
            self._length = self.lazy_dataset.compute_length()
            # Store it in the dataset too
            self.lazy_dataset._length = self._length
    def _label_source(self) -> LazyIterableDataset:
        """Dataset to ask label questions to, building a throwaway one if setup() hasn't run yet."""
        if self.lazy_dataset is not None:
            return self.lazy_dataset
        if getattr(self, "_detached_dataset", None) is None:
            self._detached_dataset = LazyIterableDataset(config=self.config, tokenizer=self.tokenizer)
        return self._detached_dataset

    def get_label_maps(self):
        """(label2id, id2label) in use, inferring them from the data when not configured."""
        return self._label_source().get_label_maps()

    def get_label_info(self):
        """(labels ordered by id, number of labels) for the configured label column."""
        return self._label_source().get_label_info()

    def get_num_labels(self) -> int:
        dataset = self._label_source()
        if self.config.label2id is not None and dataset._label2id is None:
            return len(self.config.label2id)
        return dataset.get_num_labels()

    def get_label_distribution(self):
        """{label id: count}. Requires the full pass over the file, done on demand."""
        return self._label_source().get_label_distribution()

    def get_class_weights(self, strategy: str = "inverse_frequency", normalize: bool = True) -> torch.Tensor:
        weights = self._label_source().get_class_weights(strategy, normalize)
        return torch.tensor(weights, dtype=torch.float32)
    
    def __len__(self) -> int:
        # Returns 0 when precompute_len=False 
        return self._length if self._length is not None else 0

    def train_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.lazy_dataset,
            batch_size=self.config.batch_size,
            num_workers=0,
            collate_fn=self.collate_fn,
            shuffle=False,
        )
        dataloader._is_resumable = True
        
        # Calculate number of batches (accounting for batch size)
        if self._length is not None:
            dataloader.length = (self._length + self.config.batch_size - 1) // self.config.batch_size
        else:
            dataloader.length = None
        
        return dataloader

    def state_dict(self) -> Dict[str, Any]:
        try:
            is_dist = dist.is_available() and dist.is_initialized()
        except Exception:
            is_dist = False

        this_rank_state = self.lazy_dataset.get_state_dict()

        # Single GPU
        if not is_dist:
            return {"per_rank": [this_rank_state]}

        # Distributed branch
        rank       = dist.get_rank()
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

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        per_rank = state_dict.get("per_rank", None)
        if per_rank is None:
            return  # checkpoint did not contain data

        try:
            is_dist   = dist.is_available() and dist.is_initialized()
            rank       = dist.get_rank()       if is_dist else 0
            world_size = dist.get_world_size() if is_dist else 1
        except Exception:
            is_dist    = False
            rank       = 0
            world_size = 1

        saved_world_size     = per_rank[0].get("saved_world_size")
        STRICT_WORLD_SIZE_CHECK = False  # If True, it will not work if GPU count changes
        if is_dist and saved_world_size != world_size:
            if STRICT_WORLD_SIZE_CHECK:
                raise RuntimeError(
                    f"Cannot restore dataset state: world size changed "
                    f"(saved={saved_world_size}, current={world_size})."
                )
            else:
                print(f"WARNING: world size changed")
                print(f"saved={saved_world_size}, current={world_size}")
                mapped_rank = rank  # Rivedere la logica se si vuole supportare cambio numero GPU
        else:
            mapped_rank = rank

        self.lazy_dataset.restore_state_dict(per_rank[mapped_rank])

        if is_dist:
            dist.barrier()


@registry.register("data", "mdat", "default")
class MDATDataModule(MatformerDataModule):
    def __init__(self,config,num_devices=1,*args, **kwargs):
        super().__init__(config, num_devices)      
    def __len__(self):
        if self.num_devices == 1 or self.distributed_initialized:
            return len(self.mdat) if hasattr(self, 'mdat') else 0
        else:
            return self.mdat.get_distributed_length_before_training(num_devices=self.num_devices)
            
    def train_dataloader(self):
        dataloader = DataLoader(
            self.mdat,
            batch_size=self.config.batch_size,
            num_workers=0,
            collate_fn=self.collate_fn,
            shuffle=False
        )
        dataloader._is_resumable = True
        return dataloader

    def setup(self):
        super().setup()
        self.mdat = MatformerDataset.load_dataset(
            path=self.config.data_root,
            readonly=True,
            distributed=dist.is_initialized(),
            shuffle=True,
            ds_view=self.config.mdat_view,
            batch_size=self.config.batch_size
        )
        
        if self.config.mdat_view is not None:
            self.mdat.set_view(self.config.mdat_view)        
        if self.config.mdat_strategy is not None:
            self.mdat.set_strategy(self.config.mdat_strategy,max_seq_len=self.config.max_seq_len) 
            
        self.mdat.set_iteration_modality(self.config.iteration_modality, with_meta=self.config.keep_metadata)
        print(f"Len attuale: {len(self)}")
    
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

        saved_world_size = state_dict["per_rank"][0].get("saved_world_size")
        STRICT_WORLD_SIZE_CHECK = False #If True, It will not work if GPU count changes
        if is_dist and saved_world_size != world_size:
            if STRICT_WORLD_SIZE_CHECK:
                raise RuntimeError(
                    f"Cannot restore dataset state: world size changed "
                    f"(saved={saved_world_size}, current={world_size})."
                )
            else:
                print(f"WARNING: world size changed")
                print(f"saved={saved_world_size}, current={world_size}")
                mapped_rank = rank #Rivedere la logica se si vuole supportare cambio numero GPU
        else:
            mapped_rank = rank
            
        state_dict_rank = per_rank[mapped_rank]
        self.mdat.restore_state_dict(state_dict_rank)
        print("Restoring state dict")
        print(self.mdat.document_index)
        print(self.mdat.current_chunk_step)
        if is_dist:
            dist.barrier()
