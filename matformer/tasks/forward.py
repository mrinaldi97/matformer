# matformer/tasks/generation.py
from typing import Optional, ClassVar, Union
import torch
from matformer.tasks.task import MatformerTask, TaskConfig
from matformer.matformer_registry import registry
from dataclasses import dataclass, field
from typing import Optional, ClassVar, Union
@registry.register_dataclass("task", "forward")
@dataclass
class ForwardTaskConfig(TaskConfig):
    prompt:               Optional[str] = None


@registry.register("task", "forward", "default")
class ForwardTask(MatformerTask):
    """
    A pure forward. Useful if you attach plugins.
    """

    def process_batch(self, batch):
        _ = self.model(batch)
        return batch, None
