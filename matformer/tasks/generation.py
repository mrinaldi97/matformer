# matformer/tasks/generation.py
from typing import Optional, ClassVar, Union
import torch
from matformer.tasks.task import MatformerTask, TaskConfig
from matformer.matformer_registry import registry
from dataclasses import dataclass, field
from typing import Optional, ClassVar, Union
@registry.register_dataclass("task", "generation", default=True)
@dataclass
class GenerationTaskConfig(TaskConfig):
    max_new_tokens:       int           = 200
    min_new_tokens:       int           = 0
    temperature:          float         = 1.0
    top_k:                int           = 0
    top_p:                float         = 0.9
    repetition_penalty:   float         = 1.0
    no_repeat_ngram_size: int           = 0
    do_sample:            bool          = True
    prompt:               Optional[str] = None

    _GENERATE_FIELDS: ClassVar[tuple] = (
        "max_new_tokens", "min_new_tokens", "temperature", "top_k",
        "top_p", "repetition_penalty", "no_repeat_ngram_size", "do_sample",
    )

    def generate_kwargs(self) -> dict:
        return {k: getattr(self, k) for k in self._GENERATE_FIELDS}

@registry.register("task", "generation", "default")
class GenerationTask(MatformerTask):
    """Autoregressive generation over a dataset. Returns decoded text per batch.

    Args:
        tokenizer: MatformerTokenizer; required when config.prompt is set.
    """
    def __init__(self, config: GenerationTaskConfig, model, num_devices=None, logger=None):
        super().__init__(config, model=model, num_devices=num_devices, logger=logger)

    def process_batch(self, batch) -> list[str]:
        ids = self.model.model.generate(
            input_ids=batch,
            output_scores=False, return_timing=False,
            **self.task_config.generate_kwargs(),
        )
        return [self.tokenizer.decode(row.tolist()) for row in ids]
