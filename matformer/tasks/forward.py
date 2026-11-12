# matformer/tasks/generation.py
from typing import Optional, ClassVar, Union
import torch
from matformer.tasks.task import MatformerTask, TaskConfig
from matformer.matformer_registry import registry
from dataclasses import dataclass, field
from typing import Optional, ClassVar, Union
@registry.register_dataclass("task", "forward", default=True)
@dataclass
class ForwardTaskConfig(TaskConfig):
    prompt:               Optional[str] = None


@registry.register("task", "forward", "default")
class ForwardTask(MatformerTask):
    """
    A pure forward. Useful if you attach plugins.
    """

    def process_batch(self, batch) -> list[str]:
        _ = self.model(batch)
        return None, None
