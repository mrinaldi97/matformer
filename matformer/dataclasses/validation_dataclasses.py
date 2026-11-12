from dataclasses import dataclass, field
from typing import Optional, ClassVar
from matformer.dataclasses.matformer_config import MatformerMultiConfig
from matformer.matformer_registry import registry


@registry.register_dataclass("validation", "default", default=True)
@dataclass
class ValidationConfig(MatformerMultiConfig):
    ref_task: str = field(default="default", kw_only=True)
    _ref_fields: ClassVar[dict] = {"ref_task": "task"}  # field_name -> target_category
    every_n_steps:    Optional[int] = None
    every_n_epochs:   Optional[int] = None
    every_percent_steps: Optional[float] = None # Warning: this will override any value in every_n_steps!
    name: Optional[str] = None
