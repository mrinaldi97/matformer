# matformer/tasks/matformer_task.py
from __future__ import annotations
from abc import ABC, abstractmethod
import torch.distributed as dist
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional, ClassVar, Union
from matformer.dataclasses.matformer_config import MatformerMultiConfig
from matformer.matformer_registry import registry


@dataclass
class TaskConfig(MatformerMultiConfig):
    ref_data: str = field(default="test", kw_only=True)
    ref_saver: str = field(default="default", kw_only=True)
    _ref_fields: ClassVar[dict] = {"ref_data": "data", "ref_saver": "saver"}
    _linked_fields: ClassVar[list] = ['tokenizer']
    _mandatory_references: ClassVar[list] = []
    batch_preprocessors: list = field(default_factory=list)
    batch_postprocessors: list = field(default_factory=list)
    # Saving
    save_batch: bool = True
    save_summary: bool = True
    save_model_config: bool = True  # Store the model configuration in the summary
    save_input_ids: bool = False
    save_attributes: bool = False
    save_following_attributes: bool = False
    flatten_dict: bool = False

class BatchProcessor(ABC):
    """It is a very abstract class easily utilisable to perform any pre/post process operation, for example for complicate evaluations.
    It can both augment data after each batch, as well as accumulating a state to be written in the final summary."""

    @abstractmethod
    def process(self, batch, result=None):
        """ Always receives batch, receives None result if it is a pre-processor. """
        raise NotImplementedError


class MatformerTask(ABC):
    """
    Args:
        task_config: TaskConfig (or subclass) instance.
        model: loaded Matformer model.
        data_config: optional, omit for single-batch/interactive use.
        batch_saver: optional saver invoked whenever process_batch() returns non-None.
        summary_saver: optional saver invoked once, when compute() returns non-None.
        num_devices: num_devices for dataloader.
    """

    def __init__(self, config, model=None, num_devices=None, logger=None, trainer=None):
        self.was_initialized=True
        self.config = config
        self.model = model
        self.single_item_id = 0 # Used to count each single saved row
        self.logger = logger
        self.trainer = trainer
        self._began = False
        self._rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        # Instantiate dataloader
        self.dataloader = None
        if config.data is not None:
            datamodule = config.data.instantiate(num_devices=num_devices)
            datamodule.setup()
            self.dataloader = datamodule.train_dataloader()
            if self.model:
                print("Su model ci sono")
                if self.model.runtime_mode=='virtual':
                    print("Su virtual pure")
                    self.model.activation_store.switch_databank(config.data.name, rewind=True)
                    
        # Instantiate saver
        self.saver=None
        if config.saver is not None:
            self.saver=config.saver.instantiate()

            if config.save_model_config:
                self.model_config=model.configs.asdict()  # The model's config is included in the summary
        else:
            if config.save_summary or config.save_batch:
                print("WARNING: The task asks to save output, but no saver were configured, thus nothing is going to be saved. ")
            config.save_summary = config.save_batch = False
        if self.trainer is not None: # Used if evaluation is performed during model's training
            self.is_training = True
            # epoch/step are refreshed at summary time (they change during training),
            # so here we only store the static metadata.
            self.training_metadata = {
                "project_name": trainer.configs.training.project_name,
                "run_name": trainer.configs.training.run_name
            }
        else:
            self.is_training = False



    def begin_task(self) -> None:
        """Executed before processing the batch"""
        # Tasks can be re-run on cached instances (configs.instantiate caches them), so the
        # databank switch/rewind must happen on every run, not only in __init__. Subclasses
        # overriding this method must call super().begin_task().
        model = getattr(self, 'model', None)
        if model is not None and getattr(model, 'runtime_mode', None) == 'virtual' \
                and getattr(getattr(self, 'config', None), 'data', None) is not None:
            model.activation_store.switch_databank(self.config.data.name, rewind=True)

    def end_task(self) -> None:
        """Executed after having processed the batch"""
        #if self.saver is not None:
        #    self.saver.end()

    def reset(self) -> None:
        """Clear accumulated state."""

    def process_batch(self, batch):
        """
        Run one batch. Return a result to save or return it, or None to no-op. Must be implemented.
        The returned object is going to be saved.
        Must return (batch, result)
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement process_batch().")

    def wrap_process_batch(self,batch):
        """ Calls process batch, running pre and/or post-processor(s) if configured.
        
                    result = self.process_batch(batch)
            if result is not None and self.batch_saver and self.config.save_batch:
                self.save_batch(batch, result) """
        #1. Call any preprocessor(s) in sequence on the batch
        if not hasattr(self,"was_initialized"):
            print("Il task non è mai stato iniziato")
            import sys
            sys.exit(1)        
        for pre in self.config.batch_preprocessors:
           batch, _ = pre(batch)
        batch, result = self.process_batch(batch)
        for post in self.config.batch_postprocessors:
            batch, result = post(batch, result)
        self.save_batch(batch,result)
        return batch, result

    def compute(self) -> Optional[dict]:
        return None

    def save_batch(self, batch, result) -> None:
        """
        - If "config.save_input_ids", the original batch's "input_ids" is extracted and saved (need to pad and detach)
        - If "config.save_attributes", all elements stored as attributes in the original batch, will be saved"
        - If "config.save_following_attributes", all #Not implemented
        """
        if self.config.save_batch and result is not None:
            records = []
            if self.config.save_input_ids:
                detached_inputs=batch.pad().detach()
            if self.config.save_attributes:
                detached_attributes=batch.detach_attributes()
            for i, text in enumerate(result):
                self.single_item_id += 1
                if self.config.flatten_dict:
                    data={"id":self.single_item_id}
                    data.update(result) ##Sistemare: eventuali conflitti chiave
                else:
                    data={
                        "id":self.single_item_id,
                        "result": result
                    }
                if self.config.save_input_ids:
                    data['input_ids']=detached_inputs[i]
                if self.config.save_attributes:
                    data['attributes']=detached_attributes[i]
                if self.config.save_following_attributes:
                    raise NotImplementedError
                    data['following_attributes']=None

                records.append(data)
            self.saver.save_batch(records, rank=self._rank)


    def __call__(self, batch=None, stream_callback=None, delete_keys=None, wanted_keys=None):
        """ Interactive mode """
        if not hasattr(self,"was_initialized"):
            print("Il task non è mai stato iniziato")
            import sys
            sys.exit(1)
        if batch is None:
            return self.run()
        batch, result = self.wrap_process_batch(batch)
        # In interactive mode, it is possible to restrict the returned "result" dictionary only to desired keys
        return clean_result(result,delete_keys,wanted_keys)

    @classmethod
    def clean_result(self, result, delete_keys, wanted_keys):
        """ Cleans a result by removing unwanted information """
        assert (delete_keys is not None) or (wanted_keys is not None),"Either use delete_keys or wanted_keys"
        for k in result.keys():
            if delete_keys is not None:
                if k in delete_keys:
                    result.pop(k)
            if wanted_keys is not None:
                if k not in delete_keys:
                    result.pop(k)
        return result

    def run(self) -> Optional[dict]:
        """Run process_batch() for each batch, then compute() once at the end."""
        assert self.dataloader is not None, "run() needs a data_config at construction time. If you want to run the task interactively, call it without .run()" 
        self.begin_task()
        total_batches = getattr(self.dataloader, 'length', None)        
        for i, batch in tqdm(enumerate(self.dataloader), total=total_batches):
            self.wrap_process_batch(batch)
        self.compute_summary()
        self.end_task()

    def compute_summary(self):   
        summary = self.compute() or {}
        if self.logger is not None: #Logger is called before saver to avoid logging all metadata
            for k, v in summary.items():
                self.logger.log(f"{self.config.name}/{k}", v)
        if self.config.save_summary:
            if self.is_training:
                summary.update(self.training_metadata)
                summary["epoch"] = self.trainer.epoch
                summary["step"] = self.trainer.step
            if self.config.save_model_config:
                summary.update(self.model_config)
            self.saver.save_summary(summary, rank=self._rank)
        return summary