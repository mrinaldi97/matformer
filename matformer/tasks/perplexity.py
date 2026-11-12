from matformer.tasks import MatformerTask
from matformer.data.savers import MatformerDataSaver
import torch
from typing import Optional

class GenerationTask(MatformerTask):

    def __init__(self, config: TaskConfig, model, data_config,
                 tokenizer=None, saver: Optional[MatformerDataSaver] = None, num_devices=None):
        super().__init__(config, model, data_config, tokenizer, saver)



    def process_batch(self, input_ids, metadata: Optional[list]) -> list[str]:
        pass
        