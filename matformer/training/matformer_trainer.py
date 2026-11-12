#matformer/training/matformer_trainer.py
import math, os, torch
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
from matformer.matformer_registry import registry
from matformer.matformer_wrapper import MatformerModelWrapper
from matformer.training.ckpt_manager import CheckpointDirectoryManager
from matformer.matformer_wrapper import load_model
from matformer.training.optimizers import configure_optimizers
from matformer.training.lr_scheduler import configure_lr_schedulers
from matformer.training.logger import MatformerLogger
from matformer.utils import env_variables
from matformer.loading.utils import make_holder
from pathlib import Path
from tqdm import tqdm
class MatformerTrainer:
    def __init__(self, configs, device='auto', logger=None, fabric=None, fabric_loggers=None,
                 hooks=None):
        assert "training" in configs.list_configs(),"You cannot call the trainer without a 'trainer' section in the configs"
        self.configs = configs
        self.device = device
        self.logger = logger
        self.fabric = fabric
        self.fabric_loggers = fabric_loggers if fabric_loggers is not None else {}
        self.hooks = hooks 
        self.epoch = 0
        self.step = 0
        self._setup_fabric()
        self._setup_checkpoint()
        self._setup_model()
        self._setup_data()
        self._setup_optimization()
        self._restore_states()
        self._setup_logger()
        self._setup_fabric_wrapping()

    def _setup_fabric(self):
        if self.fabric is None:
            self.fabric_loggers = {
                name: fl for name in self.configs.training.logger
                if (fl := registry.create("logger", name).build_fabric_logger(self.configs.training)) is not None
            }
            if self.configs.training.parallel_strategy == 'ddp':
                parallel_strategy = DDPStrategy(gradient_as_bucket_view=env_variables.ddp_gradient_as_bucket, find_unused_parameters=env_variables.ddp_find_unused_parameters)
            else:
                raise NotImplementedError
            self.fabric = Fabric(
                accelerator=self.device, devices=self.configs.training.device_count, num_nodes=self.configs.training.num_nodes,
                precision=self.configs.training.precision,
                strategy=parallel_strategy,
                loggers=list(self.fabric_loggers.values())
            )
            self.fabric.launch() # From now on, everything belongs to all ranks
            self.fabric.seed_everything(self.configs.training.get('seed', 27))

    def _setup_checkpoint(self):
        self.ckpt_manager = CheckpointDirectoryManager(
            self.configs.training.save_dir, self.configs.training.checkpoint_name, self.configs.training.version)
        self.load_path, _, reason = self.ckpt_manager.decide_checkpoint(
            self.configs.training.checkpoint, self.configs.training.start_from_scratch, self.configs.training.start_from_step)
        self.fabric.print(f"======== STARTING TRAINING ========")
        self.fabric.print(f"Checkpoint: {reason}")
        if self.load_path:
            self.checkpoint_holder = registry.create("checkpoint_holder", "matformer", path=self.load_path,
                                    what_to_reset=self.configs.training.what_to_reset)
            if self.configs.model.base_model:
                self.fabric.print(f"Checkpoint is loaded instead of the base model.")
        elif getattr(self.configs.model,"base_model",None):
            self.fabric.print(f"Training is inheriting weights and config from the base model: {self.configs.model.base_model}.")
            self.checkpoint_holder = make_holder(self.configs.model.base_model)
        else:
            self.checkpoint_holder = None

    def _setup_model(self):
        # Cheap, pre-build alignment: when label2id is declared in the data section, the dataset is
        # the source of truth for num_classes (heads are about to be built from it)
        label2id = next((c.label2id for c in self.configs.instances() if getattr(c, "label2id", None)), None)
        if label2id:
            for cfg in self.configs.instances():
                if hasattr(cfg, "num_classes") and cfg.num_classes != len(label2id):
                    self.fabric.print(f"WARNING: '{cfg.name}' num_classes={cfg.num_classes} overridden "
                                      f"with {len(label2id)} labels from the dataset's label2id")
                    cfg.num_classes = len(label2id)

        model_dict = load_model(
            checkpoint_holder=self.checkpoint_holder,
            configs=self.configs,
            inference=False,
            map_location=self.device
                    )
        self.configs = model_dict['configs'] # Configs are returned by the model and overwrite the original
        self.model = model_dict['model']

        self.fabric.print(f"MISSING: {model_dict['missing']}")
        self.fabric.print(f"UNEXPECTED: {model_dict['unexpected']}")
        if isinstance(model_dict['additional_weights_results'], list):
            for r in model_dict['additional_weights_results']:
                print(f"Additional weights: path {r['path']}, MISSING: {r['missing']}, UNEXPECTED: {r['unexpected']}")

        if self.configs.training.compile:
            raise NotImplementedError
            #self.model = compile_model(self.model)

    def _setup_data(self):
        # Configure datamodule
        self.datamodule = self.configs.training.data.instantiate()
        self.datamodule.setup() # It's done here to correctly count the total training steps
        # Compute the number of training steps
        self.total_steps, self.total_batches = self.count_training_steps()
        self.configs.set_override(f"training.total_steps={self.total_steps}")
        self.configs.set_override(f"training.num_batches={self.total_batches}")
        if getattr(self.datamodule.config, "max_seq_len", None) and getattr(self.datamodule.config, "batch_size", None):
            self.configs.set_override(f"training.token_per_batch={self.datamodule.config.max_seq_len * self.datamodule.config.batch_size * self.configs.training.accumulate_grad_batches}")
        self.fabric.print(f"token_per_batch: {self.configs.training.token_per_batch}")
        if self.total_steps is None:
            self.fabric.print("The Datamodule is not returning the length. Thus, LR scheduling is disabled")
            self.configs.set_override("training.lr_scheduling=False")
        # Hard clash check: labels are now fully resolved (a full scan ran if label2id was absent).
        # Anything still inconsistent with the dataset is a user error: stop instead of letting
        # cross_entropy crash later with an opaque CUDA device-side assert.
        if getattr(getattr(self.datamodule, "config", None), "task", None) != "regression":
            try:
                num_labels = self.datamodule.get_num_labels()
            except Exception:
                num_labels = None  # No label column (e.g. language modeling): nothing to check
            if num_labels:
                for cfg in self.configs.instances():
                    if getattr(cfg, "num_classes", None) not in (None, num_labels):
                        raise ValueError(f"Config '{cfg.name}' declares num_classes={cfg.num_classes} but the "
                                         f"dataset has {num_labels} labels: fix the yaml or set label2id.")
                if hasattr(self.configs.model, "num_classes") and self.configs.model.num_classes is None and num_labels != 2:
                    raise ValueError(f"The dataset has {num_labels} labels but model num_classes is unset "
                                     f"(classification heads would silently default to 2 classes): "
                                     f"set 'num_classes: {num_labels}' in the model section.")

    def _setup_optimization(self):
        # Configure optimizer(s)
        self.optimizers = configure_optimizers(self.model, self.configs.training) #(!) According to torch doc, it's important to initialize scheduler before optimizer to avoid losing states
        # Configure scheduler(s)
        self.schedulers = configure_lr_schedulers(self.configs.training, self.optimizers)

    def _restore_states(self):
        if self.load_path is not None and self.checkpoint_holder is not None:
            state_holder = self.checkpoint_holder.get_states()
            if state_holder:
                self._restore_state(self.optimizers, state_holder.optimizer, "optimizer",
                                    on_load=lambda o: self._optimizer_state_to_device(o, self.fabric.device))
                self._restore_state(self.schedulers, state_holder.lr_scheduler, "lr_scheduler")
                self._restore_state(self.datamodule, state_holder.datamodule, "datamodule")
                self.step = state_holder.step
                self.epoch = state_holder.epoch

    def _setup_logger(self):
        # Setup logger
        self.step_ref = [self.step]

        if self.logger is not None:
            # Sandbox mode: use the scoped logger provided by the caller.
            # It already prefixes keys and delegates to the parent's loggers.
            pass
        else:
            # Normal training: create a fresh logger group and add adapters.
            self.logger = MatformerLogger()
            for name in self.configs.training.logger:
                self.logger.add(
                    registry.create(
                        "logger",
                        name,
                        fabric=self.fabric,
                        fabric_logger=self.fabric_loggers.get(name),
                        total=self.total_steps,
                        initial=self.step_ref[0],
                        configs=self.configs,
                    )
                )

        self.model.setup_logging(self.logger, self.step_ref)

    def _setup_fabric_wrapping(self):
        # 9. Fabric setup
        self.dataloader = self.fabric.setup_dataloaders(self.datamodule.train_dataloader())
        self.model, *self.optimizers = self.fabric.setup(self.model, *self.optimizers)
        # Vedere se serve davvero self.model.mark_forward_method("training_step")
    def count_training_steps(self):
        """
        Return the training steps count for each epoch
        Expects the datamodule to return the *per rank* length (already divided by rank size)
        Will divide the training steps by the number of nodes
        """
        batch_size = self.configs.training.batch_size
        accumulate_grad_batches = self.configs.training.accumulate_grad_batches
        num_nodes = self.configs.training.num_nodes
        max_steps = self.configs.training.max_steps
        if hasattr(self.datamodule, '__len__'):
            per_rank_data_len = len(self.datamodule)
            print(f"PER RANK: {per_rank_data_len}")
            print(f"BATCH SIZE: {batch_size}")
            if per_rank_data_len and per_rank_data_len > 0:
                num_batches = math.ceil(per_rank_data_len / batch_size)
                print(f"NUM BATCHES: {num_batches}")
                total_steps = (num_batches // accumulate_grad_batches) // num_nodes # Dividi anche per i nodi
                print(f"Accumulate: {accumulate_grad_batches}, nodi: {num_nodes}, total: {total_steps}")
                if max_steps:
                    total_steps = max_steps

                return total_steps, num_batches
            else:
                return None, None
        else:
            return None, None

    @staticmethod
    def _optimizer_state_to_device(optimizer, device):
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    @staticmethod
    def _restore_state(targets, states, label, tolerant=False, on_load=None):
        if targets is None or states is None:
            return
        as_list = lambda x: x if isinstance(x, list) else [x]
        try:
            for t, s in zip(as_list(targets), as_list(states)):
                t.load_state_dict(s)
                if on_load:
                    on_load(t)
        except Exception as e:
            if not tolerant:
                raise
            print(f"WARNING! Impossible to resume {label}'s state: {e}")

    def save_plugins(self, step, epoch, save_path=None):
        use_manager = save_path is None
        if use_manager:
            assert isinstance(self.ckpt_manager, CheckpointDirectoryManager)
            save_path = self.ckpt_manager.plugins_dir(step) 
        save_path = Path(save_path)            
       
        for plugin_name, plugin in tqdm(self.model.plugins.items()):
            plugin_save_path=save_path.joinpath(f"{plugin_name}_e{epoch}_s{step}.pt")
            plugin.save(plugin_save_path)
            print(f"Saved {plugin_name} at {plugin_save_path}")
            
    @staticmethod
    def as_state(obj,name):
        if isinstance(obj,list):
            return [x.state_dict() for x in obj]
        elif hasattr(obj,"state_dict"):
            return obj.state_dict
        else:
            print(f"Impossible to save {name}")
            return None
    def save(self, step, epoch, what="model", model_path=None, plugins_path=None):
        """Unified save entry point: periodic saves (every n steps / n epochs) and run()'s
        final save all go through here, so they share the same model/plugins setup.
        `what`: 'model', 'plugins' ('plugin' accepted too) or 'both'."""
        what = {"plugin": "plugins"}.get(what, what)
        assert what in ("model", "plugins", "both"), f"Unknown save target: {what}"
        if what in ("model", "both"):
            self.save_checkpoint(step=step, epoch=epoch, save_path=model_path)
        if what in ("plugins", "both"):
            self.save_plugins(step=step, epoch=epoch, save_path=plugins_path)

    def save_checkpoint(self, step, epoch, save_path=None, exclude=None):
        use_manager = save_path is None
        if use_manager:
            assert isinstance(self.ckpt_manager, CheckpointDirectoryManager)
            save_path = self.ckpt_manager.return_save_path(step)
        save_path = Path(save_path)

        #as_state = lambda x: [o.state_dict() for o in x] if isinstance(x, list) else x.state_dict()
        checkpoint = {
            "state_dict": self.model.stable_state_dict(),
            "optimizer":  self.as_state(obj=self.optimizers, name="Optimizer"),
            "scheduler":  self.as_state(obj=self.schedulers, name="Scheduler"),
            "datamodule": self.datamodule.state_dict(),
            "step":       step,
            "epoch":      epoch,
            "configs":    self.model.configs.save_config(exclude=exclude, fmt='dict'),
        }

        tmp_path = save_path.with_name(save_path.name + ".tmp")
        if self.fabric:  # Fabric writes on rank 0 only
            self.fabric.save(tmp_path, checkpoint)
            self.fabric.barrier()
            if self.fabric.is_global_zero:
                os.replace(tmp_path, save_path)
            self.fabric.barrier()  # no rank moves on before the rename is done
        else:
            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, save_path)

        if use_manager and (self.fabric is None or self.fabric.is_global_zero):
            self.ckpt_manager.mark_saved(step, save_path)

    def train(self):
        train_config = self.configs.training
        validation_group = self.configs.validation if hasattr(self.configs, 'validation') else None
        val_every = train_config.validate_every_n_steps

        total_steps = getattr(train_config, 'total_steps', None)
        if validation_group:
            for vname, vcfg in validation_group.items():
                if vcfg.every_percent_steps is not None:
                    if total_steps is None:
                        raise ValueError(f"Validation '{vname}' sets every_percent_steps={vcfg.every_percent_steps} "
                                        f"but training.total_steps is None.")
                    self.configs.set_override(f"validation.{vname}.every_n_steps={int(total_steps * vcfg.every_percent_steps)}")

        for i, epoch in enumerate(range(train_config.max_epochs)):
            print(f"Epoch {i}/{train_config.max_epochs}")
            getattr(self.model, 'on_epoch_start', lambda: None)()  
            for batch_idx, batch in enumerate(self.dataloader):
                if batch_idx>self.total_steps:
                    print("Total steps: {self.total_steps} reached.")
                    break
                is_accumulating = (batch_idx + 1) % train_config.get('accumulate_grad_batches',1) != 0

                with self.fabric.no_backward_sync(self.model, enabled=is_accumulating):
                    loss = self.model.training_step(batch, batch_idx)
                    self.fabric.backward(loss / train_config.get('accumulate_grad_batches',1))

                if not is_accumulating:
                    self.fabric.clip_gradients(self.model, self.optimizers[0], max_norm=train_config.gradient_clip_val)
                    #if self.hooks.on_before_optimizer: self.hooks.on_before_optimizer(self.model, self.optimizers)
                    for optimizer in self.optimizers: optimizer.step()
                    #if self.hooks.on_after_optimizer:  self.hooks.on_after_optimizer(self.model, self.optimizers)
                    if self.schedulers:
                        # The LR log is now inside the loop (it used to be after it, so only the last scheduler
                        # was logged), and the index is renamed from `i` to `s_idx` so it doesn't shadow the epoch index
                        for s_idx, scheduler in enumerate(self.schedulers):
                            scheduler.step()
                            self.logger.log(f"lr{s_idx}", scheduler.get_last_lr()[0], step=self.step, prog_bar=True, on_step=True, on_epoch=False)
                    for optimizer in self.optimizers: optimizer.zero_grad()
                    self.step += 1
                    self.step_ref[0] = self.step
                    self.logger.advance()
                    #if self.hooks.on_step_end: self.hooks.on_step_end(self.model, self.step, loss)
                    #for every_n, fn in self.hooks.periodic.items():
                    #    if self.step % every_n == 0: fn(self.model, self.step)

                    if validation_group:
                        for vname, vcfg in validation_group.items():
                            if vcfg.every_n_steps and self.step % vcfg.every_n_steps == 0:
                                vcfg.task.instantiate(model=self.model, logger=self.logger, trainer=self)()
                        if self.model.runtime_mode=='virtual':
                            self.model.activation_store.switch_databank(self.configs.training.data.name, rewind=False)
                    #if self.hooks.on_validation and val_every > 0 and self.step % val_every == 0:
                    #    self.hooks.on_validation(self.model, self.step)

                    if train_config.save_every_n_steps > 0 and self.step % train_config.save_every_n_steps == 0:
                        self.save(step=self.step, epoch=self.epoch, what=train_config.save_what)

            self.epoch += 1
            if train_config.save_every_n_epochs > 0 and self.epoch % train_config.save_every_n_epochs == 0:
                self.save(step=self.step, epoch=self.epoch, what=train_config.save_what)
            if validation_group:
                for vname, vcfg in validation_group.items():
                    if vcfg.every_n_epochs and self.epoch % vcfg.every_n_epochs == 0:
                        vcfg.task.instantiate(model=self.model, logger=self.logger, trainer=self)()
                if self.model.runtime_mode=='virtual':
                    self.model.activation_store.switch_databank(self.configs.training.data.name, rewind=True)
            #if self.hooks.on_epoch_end: self.hooks.on_epoch_end(self.model, self.epoch)
            self.logger.reset()
        # training finished

    def run(self, save_at_end=True, save_model=True, save_plugins=False, force_model_save_dir=None, force_plugins_save_dir=None):
        self.train()
        if save_at_end:
            print("Training completed. Saving model.")
            what = "both" if save_model and save_plugins else "plugins" if save_plugins else "model" if save_model else None
            if what:
                self.save(step=self.configs.training.total_steps, epoch=self.epoch, what=what,
                          model_path=force_model_save_dir, plugins_path=force_plugins_save_dir)
        self.logger.close()
#matformer/utils.py
"""
Generic utils to load config, detect best device, compile a model...

import json
from matformer.dataclasses.model_dataclasses import ModelConfig
import torch
from importlib import import_module
from datetime import datetime



def detect_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'
    
def compile_model(model):
    try:
        if tok_cfg['varlen_strategy']:
            print("Trying compilation: dynamic sequence length, normal")
            try:
                model=torch.compile(model, dynamic=True)
            except:
                print("Trying compilation: dynamic sequence length, normal autotune")
                model=torch.compile(model, dynamic=True)
        else:
            print("Trying compilation: fixed sequence length, max autotune")
            try:
                model=torch.compile(model, mode="max-autotune")
            except:
                print("Trying compilation: fixed sequence length, normale autotune")
                model=torch.compile(model)
    except:
        print("Compilation failed! Running non-compiled model")
    return model
"""
