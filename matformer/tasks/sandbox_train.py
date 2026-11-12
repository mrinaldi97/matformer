import gc
import os
import torch
from dataclasses import dataclass, field
from matformer.matformer_registry import registry
from matformer.tasks.task import MatformerTask, TaskConfig
from matformer.matformer_configs import MatformerConfigs
from matformer.training.matformer_trainer import MatformerTrainer

@registry.register_dataclass("task", "sandbox_train")
@dataclass
class SandboxTrainConfig(TaskConfig):
    sandbox_configs: list = field(default_factory=list)
    keep_snapshot: bool = False  # Keep the base-model snapshot on disk after the task (debugging)

@registry.register("task", "sandbox_train", "default")
class SandboxTrain(MatformerTask):
    """This task is used to perform a training without losing the base model.
    Its main utility is to perform fine-tuning tasks as evaluation during model's training
    """
    def __init__(self, config, model=None, tokenizer=None, logger=None, trainer=None):
        assert trainer is not None
        cfgs = config.sandbox_configs
        self.sandbox_configs = cfgs if isinstance(cfgs, list) else [cfgs]
        self.keep_snapshot = config.keep_snapshot
        self.trainer = trainer
        self.fabric = trainer.fabric
        self.logger = logger
        self.snapshot_path = None
        self._raw_model = None
        self._forward_methods = ()
        self._device = None

    def begin_task(self):
        super().begin_task()
        t = self.trainer
        # 1. Snapshot of the base model, inside the checkpoint directory (see notes)
        save_dir = getattr(t.ckpt_manager, 'save_dir', '.')
        os.makedirs(save_dir, exist_ok=True)
        self.snapshot_path = os.path.join(save_dir, f".sandbox_base_step{t.step}.ckpt")
        t.save_checkpoint(step=t.step, epoch=t.epoch, save_path=self.snapshot_path)

        # 2. Offload the outer model and optimizer state to CPU
        self._device = next(t.model.parameters()).device
        raw = getattr(t.model, 'module', t.model)
        self._forward_methods = tuple(getattr(t.model, '_forward_methods', ()))
        for p in raw.parameters():
            p.grad = None
        raw.to('cpu')
        for opt in t.optimizers:
            MatformerTrainer._optimizer_state_to_device(opt, 'cpu')

        # 3. Drop the DDP wrapper (see notes)
        self._raw_model = raw
        t.model = None
        del raw
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.fabric.print(f"[Sandbox] GPU memory after offload: {torch.cuda.memory_allocated() / 2**20:.0f} MiB")

    def run(self):
        for i, cfg in enumerate(self.sandbox_configs):
            # Same mechanism as any base_model finetune: point at the snapshot,
            # let import_config's type-change handling + load_additional_weights'
            task_configs = MatformerConfigs.create_configs(cfg)
            name = getattr(task_configs.training, 'run_name', None) or f'sandbox_{i}'
            print(f"[Sandbox] Running finetune task '{name}'")
            task_configs.set_override(f"model.base_model={self.snapshot_path}")
            task_configs.set_override("training.start_from_scratch=True")
            scoped = self.logger.scoped(f"sandbox/{name}")
            sandbox = MatformerTrainer(
                task_configs, device=self._device, fabric=self.fabric,
                logger=scoped, fabric_loggers=self.trainer.fabric_loggers
            )
            try:
                sandbox.run(save_at_end=False)
            finally:
                scoped.close()  # sends the last row; never closes the parent's backends
                del sandbox
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def end_task(self):
        super().end_task()
        t = self.trainer
        if self._raw_model is not None:
            # Rebuild the wrapper (moves the module back to the device and recreates DDP)
            t.model = self.fabric.setup_module(self._raw_model)
            for name in self._forward_methods:
                t.model.mark_forward_method(name)
            for opt in t.optimizers:
                MatformerTrainer._optimizer_state_to_device(opt, self._device)
            self._raw_model = None
        if self.snapshot_path is not None:
            self.fabric.barrier()  # nobody is still reading the snapshot
            if not self.keep_snapshot and self.fabric.is_global_zero and os.path.exists(self.snapshot_path):
                os.remove(self.snapshot_path)

    def __call__(self):
        try:
            self.begin_task()
            self.run()
        finally:
            self.end_task()