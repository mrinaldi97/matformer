from dataclasses import dataclass, field

from typing import Optional, Literal, ClassVar, Dict, List
from matformer.dataclasses.matformer_config import MatformerConfig, MatformerMultiConfig
from matformer.matformer_registry import registry

@dataclass
class DataConfig(MatformerMultiConfig):
    name: Optional[str] = field(default="train", kw_only=True)
    _linked_fields: ClassVar[list] = ["tokenizer", "model", "training"]
    data_root: str = "/data"
    batch_size: int = 32
    num_workers: int = 1
    keep_metadata: bool = False
    head_samples: Optional[int] = None
    partition_start: Optional[int] = None
    partition_end: Optional[int] = None

@registry.register_dataclass("data", "mdat", default=True)
@dataclass
class MDATDataModuleConfig(DataConfig):
    mdat_strategy: Optional[str] = None
    mdat_view: Optional[str] = "default"
    iteration_modality: str = "chunked_tokens"


@registry.register_dataclass("data", "lazy", default=True)
@dataclass
class LazyDataModuleConfig(DataConfig):
    _linked_fields: ClassVar[list] = ['tokenizer', 'training', 'model']
    # LazyDataModule-specific
    file_format: Optional[str] = None          # "jsonl","csv","conll","conllu","hf"
    precompute_len: bool = True
    text_column: Optional[str] = "text"
    input_ids_column: str = "input_ids"
    tokenize: bool = True
    add_eos: bool = False
    add_bos: bool = True
    keep_original_text: bool = False
    split: str = "train"
    max_seq_len: Optional[int] = None
    hf_config: Optional[str] = None
    varlen_strategy: str = "unpadding"
    # Classification additions
    task: Literal["lm", "sentence-level", "token-level", "regression"] = "lm"
    label_column: Optional[str] = None
    label2id: Optional[Dict[str, int]] = None
    id2label: Optional[Dict[int, str]] = None  # filled in automatically when inferred from the data
    scan_labels: Optional[bool] = None         # None=automatic, True=always full pass, False=never (unless mandatory)
    missing_label_values: Optional[List[str]] = None  # extra strings treated as "no label" (null, none, NaN, ... are always included)
    cache_in_memory: bool = False              # still to be implemented, low priority
