from dataclasses import dataclass, field
from typing import Optional, Dict, Callable

@dataclass
class TrainingHooks:
    on_before_optimizer: Optional[Callable] = None  # fn(model, optimizers)
    on_after_optimizer:  Optional[Callable] = None  # fn(model, optimizers)
    on_step_end:         Optional[Callable] = None  # fn(model, step, loss)
    on_epoch_end:        Optional[Callable] = None  # fn(model, epoch)
    on_validation:       Optional[Callable] = None  # fn(model, step)
    periodic:            Dict[int, Callable] = field(default_factory=dict)  # {every_n: fn(model, step)}
