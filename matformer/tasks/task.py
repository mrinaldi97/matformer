# matformer/tasks/matformer_task.py
from __future__ import annotations
from abc import ABC, abstractmethod
import torch
import torch.distributed as dist
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional, ClassVar, Union
from matformer.dataclasses.matformer_config import MatformerMultiConfig
from matformer.matformer_registry import registry


@dataclass
class TaskConfig(MatformerMultiConfig):
    ref_data: str = field(default="test", kw_only=True)
    _ref_fields: ClassVar[dict] = {"ref_data": "data"}
    _linked_fields: ClassVar[list] = ['tokenizer']
    _mandatory_references: ClassVar[list] = []
    batch_preprocessors: list = field(default_factory=list)
    batch_postprocessors: list = field(default_factory=list)
    # Saving: summaries go to the run database; per-item records are opt-in because they can be big.
    save_batch: bool = False
    save_summary: bool = True
    save_input_ids: bool = False
    save_attributes: bool = False
    save_following_attributes: bool = False
    flatten_dict: bool = False
    # How an interactive run is shown: 'probs' (label + confidence, the default), 'argmax'
    # (only the chosen label) or 'full' (everything the task computed). Tasks without a
    # presentation of their own fall back to the generic rendering.
    presentation: str = "probs"

PRESENTATIONS = ("probs", "argmax", "full", "table")

def split_result(result, batch_size: int) -> list:
    """Transpose a batch-level result into per-item results: tensors and lists whose leading
    dimension is the batch are split row-wise, anything else is repeated on every row."""
    if isinstance(result, dict):
        columns = {key: split_result(value, batch_size) for key, value in result.items()}
        return [{key: column[i] for key, column in columns.items()} for i in range(batch_size)]
    if torch.is_tensor(result):
        return list(result) if result.shape and result.shape[0] == batch_size else [result] * batch_size
    if isinstance(result, (list, tuple)) and len(result) == batch_size:
        return list(result)
    return [result] * batch_size


def per_position(tensor):
    """A batch-level result reshaped as (positions, classes): however many leading dimensions
    the batch had, one row per thing that was predicted and one column per class. A single
    prediction stays a single row."""
    while tensor.dim() > 2:
        tensor = tensor[0]
    return tensor.unsqueeze(0) if tensor.dim() == 1 else tensor


def surface(token) -> str:
    """A subword's display form: 'Ġ' and '▁' are a space, 'Ċ' is a newline. Only ever applied to
    what a human reads, never to what is decoded."""
    text = str(token)
    return (text.replace("\u0120", "\u2581").replace("Ġ", "▁")
            .replace("▁", " ").replace("Ċ", "\n").replace("ĉ", "\t"))


def top_k(k, n_classes: int) -> int:
    """How many candidates to keep for a prediction: `k`, or all of them when k is negative, or
    none when it is zero. The GUI's "token detail" knob is one number with three meanings."""
    return 0 if not k else n_classes if k < 0 else min(k, n_classes)


def clean_result(result, delete_keys=None, wanted_keys=None):
    """Filter a result dict down to wanted_keys, or without delete_keys. Non-dicts pass through."""
    if not isinstance(result, dict):
        return result
    if wanted_keys is not None:
        return {k: v for k, v in result.items() if k in wanted_keys}
    if delete_keys is not None:
        return {k: v for k, v in result.items() if k not in delete_keys}
    return result


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
        logger: optional MatformerLogger; receives metrics, summaries and (opt-in) records.
        num_devices: num_devices for dataloader.
    """
    # Analysis tasks (weight statistics, activation sweeps) answer a question on their own and
    # need no dataset; everything else uses the loader built from its `data` reference.
    requires_data: ClassVar[bool] = True

    def __init__(self, config, model=None, num_devices=None, logger=None, trainer=None):
        self.was_initialized=True
        self.config = config
        self.model = model
        self.single_item_id = 0 # Used to count each single saved row
        self.logger = logger
        self.trainer = trainer
        self.record_sink = None     # callable(records): file annotation, set by TaskSession
        self.summary_sink = None    # callable(summary): file annotation, set by TaskSession
        self.execution = None       # logger token linking this task run to its training run
        self._began = False
        self.interactive = False    # True when driven one batch at a time (see __call__)
        self.save_results = True    # False to return results without saving them (see __call__)
        self.stream_callback = None  # Set per call by __call__, read by streamable tasks
        self.metrics = {}             # Numbers of the last run (tokens, seconds), for the console
        self.status = ""              # One line of run-time state the console shows after a run
        self.system_prompt = None     # Chat plumbing, not applied yet (see GenerationTask)
        self.chat_template = None
        self._rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        # Instantiate dataloader
        self.dataloader = None
        data = getattr(config, 'data', None)
        if data is not None:
            datamodule = data.instantiate(num_devices=num_devices,
                                          tokenizer=getattr(model, 'tokenizer', None))
            datamodule.setup()
            self.dataloader = datamodule.train_dataloader()
            if self.model and self.model.runtime_mode == 'virtual':
                self.model.activation_store.switch_databank(data.name, rewind=True)
                    
        if self.trainer is not None: # Used if evaluation is performed during model's training
            self.is_training = True
            self.training_metadata = {
                "project_name": trainer.configs.training.project_name,
                "run_name": trainer.configs.training.run_name
            }
        else:
            self.is_training = False
        # Which point of a sweep this run is, when it belongs to one: written into every summary,
        # so a folder of results says what produced each of them without reading the filenames.
        self.sweep = getattr(getattr(model, 'configs', None), 'active_sweep', None)

    def ensure_began(self) -> None:
        """Run begin_task() on first use, so a task can be driven one batch at a time (interactive
        mode, file annotation) without ever calling run()."""
        if not self._began:
            self.begin_task()
            self._began = True

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

    def reset(self) -> None:
        """Clear accumulated state."""

    def present(self, rows) -> Optional[str]:
        """Human-readable rendering of the rows of one interactive run. Returning None leaves
        the console to its generic dump: tasks that know their own output override this."""
        return None

    def clear(self) -> None:
        """Drop any run-local buffer (generated context and the like)."""
        self.reset()

    def set_system_prompt(self, text) -> str:
        """Store the prompt a future chat template will wrap the conversation with."""
        self.system_prompt = text or None
        return f"system prompt {'cleared' if not text else 'set'} (no chat template applied yet)"

    def set_chat_template(self, text) -> str:
        """Store the chat template name; applying it to the conversation is not implemented yet."""
        self.chat_template = (text or '').strip() or None
        return (f"chat template '{self.chat_template}' selected (applying it is not implemented yet)"
                if self.chat_template else "chat template cleared")

    def process_batch(self, batch):
        """
        Run one batch. Return a result to save or return it, or None to no-op. Must be implemented.
        The returned object is going to be saved.
        Must return (batch, result)
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement process_batch().")

    def wrap_process_batch(self,batch):
        """Calls process batch, running pre and/or post-processor(s) if configured, then saves the
        result. Returns (batch, result)."""
        self.ensure_began()
        # Validation runs receive batches straight from the loader, unlike the training loop:
        # move them (and their attributes, e.g. labels) before any task sees them.
        device = getattr(self.model, "device", None)
        if device is not None and hasattr(batch, "to"):
            try:
                batch = batch.to(device)
            except Exception:
                pass
        #1. Call any preprocessor(s) in sequence on the batch
        for pre in self.config.batch_preprocessors:
           batch, _ = pre(batch)
        batch, result = self.process_batch(batch)
        for post in self.config.batch_postprocessors:
            batch, result = post(batch, result)
        self.save_batch(batch,result)
        return batch, result

    def compute(self) -> Optional[dict]:
        return None

    def _position(self):
        """Step and epoch of the run driving the task, when there is one."""
        trainer = getattr(self, 'trainer', None)
        return (getattr(trainer, 'step', None), getattr(trainer, 'epoch', None)) \
            if trainer is not None else (None, None)

    def records(self, batch, result) -> list:
        """One record per item of the batch: result transposed row-wise (see split_result),
        optionally flattened, plus input tokens and batch attributes when configured."""
        size = len(batch)
        rows = split_result(result, size)
        inputs = batch.pad().detach() if self.config.save_input_ids else None
        attributes = split_result(batch.attributes, size) if self.config.save_attributes else None
        if self.config.save_following_attributes:
            raise NotImplementedError("save_following_attributes is not implemented yet")
        records = []
        for i, row in enumerate(rows):
            self.single_item_id += 1
            data = {"id": self.single_item_id}
            data.update(row if self.config.flatten_dict and isinstance(row, dict) else {"result": row})
            if inputs is not None:
                data['input_ids'] = inputs[i]
            if attributes is not None:
                data['attributes'] = attributes[i]
            records.append(data)
        return records

    def save_batch(self, batch, result) -> None:
        """Send the per-item records to the logger and/or the annotation sink. Opt-in through
        config.save_batch; file annotation sets the sink without touching the config."""
        if not self.save_results or result is None:
            return
        if not self.config.save_batch and self.record_sink is None:
            return
        records = self.records(batch, result)
        step, epoch = self._position()
        if self.config.save_batch and self.logger is not None:
            self.logger.log_records(self.config.name, records, step=step, epoch=epoch,
                                    task_run=self.execution)
        if self.record_sink is not None:
            self.record_sink(records)


    def __call__(self, batch=None, save=False, stream_callback=None, delete_keys=None, wanted_keys=None):
        """ Interactive mode: process a single batch and return its result.
        batch=None runs the whole dataset instead (see run()). save=False keeps the result out of
        the sinks, so probing a model interactively never pollutes the task's output files."""
        if batch is None:
            return self.run()
        self.interactive = True
        self.save_results = save
        self.stream_callback = stream_callback
        _, result = self.wrap_process_batch(batch)
        return clean_result(result, delete_keys, wanted_keys)

    def _start_execution(self):
        """Open the logger's task-execution record: its step/epoch say when it ran, `phase` says
        whether it is an evaluation inside a training or a standalone run."""
        step, epoch = self._position()
        phase = "training" if self.is_training else "standalone"
        self.execution = self.logger.start_task(self.config.name, step=step, epoch=epoch,
                                                phase=phase) \
            if self.logger is not None and hasattr(self.logger, 'start_task') else None

    def _end_execution(self):
        if self.execution is not None:
            self.logger.finish_task(self.execution)
            self.execution = None

    def run(self) -> Optional[dict]:
        """Run process_batch() for each batch, then compute() once at the end.

        A task with ``requires_data = False`` (weight or module analysis) skips the loader and
        goes straight to ``compute()``; a task that really needs data says so instead of failing
        with an obscure ``None`` iteration.
        """
        if self.dataloader is None and self.requires_data:
            raise AssertionError("run() needs a data_config at construction time. If you want to "
                                 "run the task interactively, call it without .run()")
        self.stream_callback = None
        self.interactive = False
        self.save_results = True
        self.begin_task()
        self._began = True
        self._start_execution()
        try:
            if self.dataloader is not None:
                total_batches = getattr(self.dataloader, 'length', None)
                for i, batch in tqdm(enumerate(self.dataloader), total=total_batches):
                    self.wrap_process_batch(batch)
            summary = self.compute_summary()
            self.end_task()
            return summary
        finally:
            self._end_execution()

    def compute_summary(self):
        summary = self.compute() or {}
        if self.sweep is not None:
            summary["sweep"] = self.sweep.name
        step, epoch = self._position()
        if self.is_training:
            summary.update(self.training_metadata)
            summary["epoch"], summary["step"] = epoch, step
        if self.logger is not None:
            for key, value in summary.items():
                self.logger.log(f"{self.config.name}/{key}", value)
            if self.config.save_summary:
                self.logger.log_summary(self.config.name, summary, step=step, epoch=epoch,
                                        task_run=self.execution)
        if self.config.save_summary and self.summary_sink is not None:
            self.summary_sink(summary)
        return summary
