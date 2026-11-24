#matformer/training/matformer_trainer.py
import json, math, os, time, torch
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
from matformer.matformer_registry import registry
from matformer.matformer_wrapper import MatformerModelWrapper
from matformer.training import control
from matformer.training.ckpt_manager import CheckpointDirectoryManager
from matformer.matformer_wrapper import load_model
from matformer.training.optimizers import configure_optimizers
from matformer.training.lr_scheduler import configure_lr_schedulers
from matformer.training.logger import MatformerLogger
from matformer.training.sqlite_logger import run_dir_for
from matformer.utils import env_variables
from matformer.utils.table import Table
from matformer.training.utils import precision_dtype, matmul_parameter_count, peak_tflops, step_flops
from matformer.loading.utils import make_holder
from pathlib import Path
from tqdm import tqdm


def _batch_tokens(batch):
    # TODO2: where is this used? Is it still relevant or deprecated?
    """(tokens, sequences) of a batch, read from its shape alone: no reduction, hence no device
    synchronisation. The padded path counts its nominal width, padding included, which is what its
    kernels really pay for."""
    try:
        if getattr(batch, "isUnpadded", False):
            return int(batch.tensor.shape[0]), int(batch.batch_size)
        return int(batch.tensor.numel()), int(batch.tensor.shape[0])
    except (AttributeError, TypeError, IndexError):
        return 0, 0


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
        self.control = control.channel()
        self.control.install_signals()

    def _setup_fabric(self):
        """ Setups torch.fabric for distributed training; currently hardcoded to DDP"""
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
        """ Uses the CheckpointDirectoryManager to decide if to resume from a checkpoint or start from scratch
        	Return the checkpoint_holder instance that will be used for training (either wrapping an existing checkpoint or a base model to be further trained) or None if the training starts/restarts from scratch.
        """
        self.ckpt_manager = CheckpointDirectoryManager(
            self.configs.training.save_dir, self.configs.training.checkpoint_name, self.configs.training.version)
        self.load_path, _, reason = self.ckpt_manager.decide_checkpoint(
            self.configs.training.checkpoint, self.configs.training.start_from_scratch, self.configs.training.start_from_step)
        self.fabric.print(f"======== STARTING TRAINING ========")
        self.fabric.print(f"Checkpoint: {reason}")
        if self.load_path:
            self.checkpoint_holder = registry.create("checkpoint_holder", "matformer", path=self.load_path,
                                    what_to_reset=self.configs.training.what_to_reset)
            if getattr(self.configs.model, "base_model", None):
                self.fabric.print(f"Checkpoint is loaded instead of the base model.")
        elif getattr(self.configs.model,"base_model",None):
            self.fabric.print(f"Training is inheriting weights and config from the base model: {self.configs.model.base_model}.")
            self.checkpoint_holder = make_holder(self.configs.model.base_model)
        else:
            self.checkpoint_holder = None

    def _setup_model(self):
        # Start of AI slop 
        # Those lines must be rewritten, they are sloppish. Ok to overwrite num_classes from the dataset configs, but this is defnitely not the idiomatically correct way to do it!!!
        # Just check "configs.data"!!! Why "any config that has label2id inside", it's ugly, sloppish and longer than it could be. Or perhaps there are other dataclasses with num classes? I don't remember that
        # Cheap, pre-build alignment: when label2id is declared in the data section, the dataset is
        # the source of truth for num_classes (heads are about to be built from it)
        label2id = next((c.label2id for c in self.configs.instances() if getattr(c, "label2id", None)), None)
        if label2id:
            for cfg in self.configs.instances():
                if hasattr(cfg, "num_classes") and cfg.num_classes != len(label2id):
                    self.fabric.print(f"WARNING: '{cfg.name}' num_classes={cfg.num_classes} overridden "
                                      f"with {len(label2id)} labels from the dataset's label2id")
                    cfg.num_classes = len(label2id)
        # End of AI slop
        model_dict = load_model(
            checkpoint_holder=self.checkpoint_holder,
            configs=self.configs,
            inference=False,
            dtype=precision_dtype(self.configs.training.precision),
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
        # AI SLOP start
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
        # AI SLOP end
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
            save_path = self.save_checkpoint(step=step, epoch=epoch, save_path=model_path)
        if what in ("plugins", "both"):
            self.save_plugins(step=step, epoch=epoch, save_path=plugins_path)
        return save_path

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
        return save_path

    def train(self):
        train_config = self.configs.training
        validation_group = self.configs.validation if hasattr(self.configs, 'validation') else None
        val_every = train_config.validate_every_n_steps
        # Throughput rarely moves during a run, so it is measured over a window of steps and logged as
        # rarely as asked: [clock when the window opened, step it opened at, tokens, sequences].
        every = int(getattr(train_config, 'log_speed_every_n_steps', 0) or 0)
        window = [time.perf_counter(), self.step, 0, 0]

        total_steps = getattr(train_config, 'total_steps', None)
        if validation_group:
            for vname, vcfg in validation_group.items():
                if vcfg.every_percent_steps is not None:
                    if total_steps is None:
                        raise ValueError(f"Validation '{vname}' sets every_percent_steps={vcfg.every_percent_steps} "
                                        f"but training.total_steps is None.")
                    self.configs.set_override(f"validation.{vname}.every_n_steps={int(total_steps * vcfg.every_percent_steps)}")
        # Start of the training steps
        for i, epoch in enumerate(range(train_config.max_epochs)):
            if self.control.stop:
                break
            print(f"Epoch {i}/{train_config.max_epochs}")
            getattr(self.model, 'on_epoch_start', lambda: None)()  
            for batch_idx, batch in enumerate(self.dataloader):
                if batch_idx>self.total_steps:
                    print(f"Total steps: {self.total_steps} reached.", flush=True)
                    break
                if every: #TODO2/AI SLOP: a bit ugly... different variable name? It's not immediately clear what this code does
                    tokens, sequences = _batch_tokens(batch)
                    window[2] += tokens
                    window[3] += sequences
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
                        for s_idx, scheduler in enumerate(self.schedulers):
                            scheduler.step()
                            self.logger.log(f"lr{s_idx}", scheduler.get_last_lr()[0], step=self.step, prog_bar=True, on_step=True, on_epoch=False)
                    for optimizer in self.optimizers: optimizer.zero_grad()
                    self.step += 1
                    self.step_ref[0] = self.step
                    self.logger.advance()
                    if every and self.step - window[1] >= every:
                        self._report_speed(time.perf_counter() - window[0], window[2], window[3],
                                           self.step - window[1])
                        window[:] = [time.perf_counter(), self.step, 0, 0]
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

                    if control.poll(self):
                        print(f"[control] ending the loop at step {self.step}", flush=True)
                        break

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

#TODO2: _report_speed, _mock_batch, _probe_step, probe_memory makes the code unpleasant to read, consider moving somewhere else (ex. in utils) and subclass/import

    def _report_speed(self, seconds, tokens, sequences, steps):
        """Throughput of the last `steps` steps in the units trainings are compared by: tokens per
        second, sequences per second, the FLOPs actually performed, and MFU — their share of the
        device's peak. Every number comes from counters the loop already holds plus one clock read per
        window, so nothing is synchronised in order to measure it. Figures are per rank; the cluster
        ones multiply by the world size, and MFU is identical either way."""
        environment = getattr(self, "_speed_environment", None)
        if environment is None:
            name = ""
            try:
                if self.fabric.device.type == "cuda":
                    name = torch.cuda.get_device_name(self.fabric.device)
            except Exception:
                name = ""
            model = getattr(self.configs, "model", None)
            signals = [str(one) for one in (getattr(self.configs.training, "learning_signals", None) or [])]
            environment = {"device": name,
                           "parameters": matmul_parameter_count(self.model),
                           "layers": int(getattr(model, "num_hidden_layers", 0) or 0),
                           "hidden": int(getattr(model, "hidden_size", 0) or 0),
                           "peak": peak_tflops(name, self.configs.training.precision),
                           # A masked objective attends in both directions: twice the quadratic term.
                           "causal": not any("mask" in one.lower() for one in signals)}
            self._speed_environment = environment
        seconds, steps = max(float(seconds), 1e-9), max(int(steps), 1)
        world = max(int(getattr(self.fabric, "world_size", 1) or 1), 1)
        step_seconds = seconds / steps
        per_step_tokens, per_step_sequences = tokens / steps, sequences / steps
        rate = step_flops(per_step_tokens, environment["parameters"], environment["layers"],
                          environment["hidden"], per_step_tokens / max(per_step_sequences, 1.0),
                          causal=environment["causal"]) / step_seconds
        speed = {"tokens_per_second": per_step_tokens / step_seconds,
                 "tokens_per_second_cluster": world * per_step_tokens / step_seconds,
                 "sequences_per_second": per_step_sequences / step_seconds,
                 "steps_per_second": 1.0 / step_seconds,
                 "step_ms": 1000.0 * step_seconds,
                 "tflops": rate / 1e12,
                 "tflops_cluster": world * rate / 1e12,
                 "mfu_percent": (100.0 * rate / (environment["peak"] * 1e12)
                                 if environment["peak"] else None),
                 "peak_tflops": environment["peak"], "device": environment["device"]}
        for key, value in speed.items():
            if isinstance(value, (int, float)):
                self.logger.log(f"speed/{key}", value, prog_bar=(key == "tokens_per_second"))
        return speed

    def _mock_batch(self, rows, seq_len):
        """A batch of mock tokens, built by the dataloader's own `collate_fn`: same type, same packing,
        same attributes, no document read, because the only thing at stake is how many tokens a step
        carries. Every row is as long as the config allows and continues its own document, which is the
        shape a step costs the most in."""
        import numpy as np
        pad = int(getattr(self.configs.tokenizer, "pad_token_id", 0) or 0)
        vocab = max(int(getattr(self.configs.model, "vocab_size", 0) or 0), pad + 2, 2)
        ids = np.random.randint(1, vocab, size=(rows, seq_len))
        if pad:
            ids[ids == pad] = 1
        task = str(getattr(self.datamodule.config, "task", "lm") or "lm")
        classes = next((one.num_classes for one in self.configs.instances()
                        if getattr(one, "num_classes", None)), 2)
        labels = (np.random.randint(0, max(int(classes), 1), size=(rows, seq_len))
                  if task == "token-level" else
                  np.random.randint(0, max(int(classes), 1), size=rows)
                  if task in ("sentence-level", "regression") else None)
        items = []
        for row in range(rows):
            item = {"object": ids[row].tolist(), "is_same_document": True}
            if labels is not None:
                item["labels"] = labels[row].tolist() if task == "token-level" else int(labels[row])
            items.append(item)
        return self.datamodule.collate_fn(items).to(self.fabric.device)

    def _probe_step(self, rows, seq_len, optimize=False, sync_gradients=False):
        """One mock forward and backward on the live model, and the memory it peaks at: `(fits, bytes)`.
        Gradient reduction is off while the search runs — a rank that dies of OOM halfway through a step
        would otherwise leave its partners waiting in a collective forever — and the reduction buffers
        exist anyway, because the warm-up step does synchronise. Anything that is not a full GPU is
        re-raised: a probe is no place to hide a real bug behind a shrunken batch size."""
        cuda = self.fabric.device.type == "cuda"
        batch, floor, allocated, reserved = None, 0, 0, 0
        if cuda:
            floor = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        try:
            batch = self._mock_batch(rows, seq_len)
            with self.fabric.no_backward_sync(self.model, enabled=not sync_gradients):
                loss = self.model.training_step(batch, 0)
                self.fabric.backward(loss)
                if optimize:
                    self.fabric.clip_gradients(self.model, self.optimizers[0],
                                               max_norm=self.configs.training.gradient_clip_val)
                    for optimizer in self.optimizers:
                        optimizer.step()
            fits = True
        except Exception as raised:
            if not (isinstance(raised, torch.cuda.OutOfMemoryError)
                    or any(word in str(raised).lower() for word in ("memory", "cublas", "nccl"))):
                raise
            fits = False      # keep no reference to it: an exception holds every frame of the failed
                              # step, and with them the activations it died on — the next attempt would
                              # inherit a full GPU and report an out-of-memory it never caused
        # Read before the clean-up, so it is this attempt's own high-water mark. Allocated peak is what
        # the search extrapolates on: it always comes back down between attempts, while reserved is a
        # mark the allocator may hold on to for a while.
        if cuda:
            allocated, reserved = (torch.cuda.max_memory_allocated(),
                                   torch.cuda.max_memory_reserved())
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)
        cache = getattr(self.model, "cache", None)
        if getattr(cache, "storage", None):
            cache.storage.clear()
        if getattr(cache, "clear_cache", None):        # alibi/mask/rotary caches are keyed by shape:
            cache.clear_cache()                        # without this, each new batch size in the search
                                                       # leaves another matrix behind and ratchets the
                                                       # floor up until everything after OOMs
        batch = None
        if cuda and not fits:
            import gc
            gc.collect()
            torch.cuda.empty_cache()        # let the next attempt start from a clean allocator
        return fits, floor, allocated, reserved

    def probe_memory(self, fill=0.94, max_rows=0, steps=3, seq_len=None, apply=False,
                     keep_weights=True, report_path=None):
        """How big a batch this run really fits, and how fast it then goes.

        The trainer is the one place where everything that costs memory is assembled at once —
        parameters, gradients, the optimizer's moments, DDP buckets, mounted plugins, the layer schedule
        with its loops — so the question is answered here, on the live model, without rebuilding anything
        between attempts. Mock tokens stand in for the dataset: the MDAT is never read, because only the
        token count is at stake.

        The worst case is what gets measured, since that is what a long run eventually hits: every row at
        `max_seq_len`, the masking rate pinned to the highest the config allows (predicting more positions
        means gathering more rows out of the vocabulary), and every configured learning signal walked in
        turn, so a run alternating CLM and MLM is bounded by its dearer objective. What no probe can
        promise is that an arbitrary plugin never allocates on a path this walk does not take.

        The search is the cheap part: two measured points are enough, because activation memory is linear
        in tokens, so `peak = intercept + slope * rows` predicts the ceiling directly. The prediction is
        then verified, and the answer bisected if it was wrong — five or six steps forward and back, a
        handful of seconds.

        `steps` real steps at the winning size confirm the answer is stable rather than lucky (a model
        that keeps activations from one step to the next does not say so in a single step) and time them.
        Returns the full report; `apply` also hands the found batch size to this trainer's dataloader, so
        that a probe and the training that follows it can share one job. Probing takes real steps, so the
        weights and the optimizer's state are put back afterwards unless `keep_weights=False`."""
        training, data = self.configs.training, self.datamodule.config
        seq_len = int(seq_len or getattr(data, "max_seq_len", None)
                      or getattr(self.configs.model, "max_position_embeddings", 0) or 0)
        cuda = self.fabric.device.type == "cuda"
        total = torch.cuda.get_device_properties(self.fabric.device).total_memory if cuda else 0
        budget, world = int(total * fill), max(int(self.fabric.world_size or 1), 1)
        accumulate = int(getattr(training, "accumulate_grad_batches", 1) or 1)
        # Fabric wraps the model, and the DDP strategy wraps it once more: the object whose attributes
        # the training step actually reads is the Matformer wrapper underneath both, and that is where a
        # probe has to install itself. Anything else would be a flag nobody looks at, and an out-of-memory
        # would be swallowed exactly as it is during training.
        target = self.model
        for _ in range(3):
            if isinstance(target, MatformerModelWrapper):
                break
            target = getattr(target, "module", None) or getattr(target, "_forward_module", None) or target
        if not isinstance(target, MatformerModelWrapper):
            target = self.model
        target.strict_steps = True        # a probe needs the failure, not the wrapper's skipped batch
        selector = getattr(target, "learning_signal_selector", None)
        signals = list(getattr(selector, "learning_signals", []) or [])
        names = [type(one).__name__.replace("LearningSignal", "") for one in signals]
        attempts, peak, fits, slope, low = [], 0, True, 0, max(int(training.batch_size or 1), 1)
        started, tokens, sequences, done = time.perf_counter(), 0, 0, 0

        def to_cpu(obj):
            if torch.is_tensor(obj):
                return obj.detach().to("cpu")
            if isinstance(obj, dict):
                return {key: to_cpu(value) for key, value in obj.items()}
            if isinstance(obj, (list, tuple)):
                return type(obj)(to_cpu(value) for value in obj)
            return obj

        # Worst case masking: the gather a masked loss performs over the vocabulary is sized by how many
        # positions are predicted, so the *average* rate would undershoot the real peak.
        rates = {}
        for signal in signals:
            maskerator = getattr(signal, "maskerator", None)
            if maskerator is None:
                continue
            configured = getattr(getattr(self.configs, "masking", None), "masked_substitution_rate", None)
            worst = (max(configured) if isinstance(configured, (list, tuple)) and configured
                     else configured if isinstance(configured, (int, float)) and configured else 0.15)
            rates[id(maskerator)] = (maskerator.substitution_rate, maskerator.substitution_rate_upper,
                                     maskerator.external_schedule)
            maskerator.substitution_rate, maskerator.substitution_rate_upper = float(worst), None
            maskerator.external_schedule = False
        # One signal at a time, each of them, so that both halves of a hybrid run get bounded. A
        # multitask selector already sums them into one step, which is heavier still: leave it alone.
        cycle = None
        if selector is not None and len(signals) > 1 and \
                "multitask" not in str(getattr(training, "learning_signal_selector", "")):
            import itertools
            cycle = itertools.cycle(signals)
            selector.pick = lambda batch, batch_idx=None: [next(cycle)]

        def probe(rows, optimize=False, sync_gradients=False):
            ok, floor, measured, held = self._probe_step(rows, seq_len, optimize=optimize,
                                                         sync_gradients=sync_gradients)
            if world > 1:
                ok = bool(self.fabric.all_reduce(torch.tensor([1.0 if ok else 0.0],
                                                              device=self.fabric.device),
                                                 reduce_op="min").item())
            attempts.append({"rows": rows, "tokens": rows * seq_len, "fits": ok,
                             "floor_mib": round(floor / 2 ** 20),
                             "peak_mib": round(measured / 2 ** 20),
                             "reserved_mib": round(held / 2 ** 20)})
            self.fabric.print(f"[probe] {rows:>7} rows  {rows * seq_len:>10} tokens  "
                              f"{'fits' if ok else 'OOM'}  peak {measured / 2 ** 20:>7.0f} MiB "
                              f"(+{(measured - floor) / 2 ** 20:>6.0f})  held {held / 2 ** 20:>7.0f}",
                              flush=True)
            return ok, measured

        try:
            # A first complete step — reduced, clipped, applied — so that everything a steady step needs
            # exists before anything is measured: DDP buckets, the optimizer's moments, cuDNN and
            # flash-attention workspaces.
            self._probe_step(1, seq_len, optimize=True, sync_gradients=True)
            snapshot = [parameter.detach().to("cpu") for parameter in self.model.parameters()]
            moments = [to_cpu(optimizer.state_dict()) for optimizer in self.optimizers]
            base = max(torch.cuda.memory_reserved(), torch.cuda.memory_allocated()) if cuda else 0
            if cuda:
                fits, peak_one = probe(1)
                low, points, high = (1, [(1, peak_one)], None) if fits else (0, [], 1)

                def fit():
                    """peak(rows) from the two largest measured successes. Activation memory is linear
                    in tokens, so two points near the answer place the ceiling; whatever curvature they
                    miss is caught by verifying the prediction instead of trusting it."""
                    if len(points) < 2:
                        return max(peak_one - base, 1), base
                    (rows_1, p1), (rows_2, p2) = points[-2], points[-1]
                    slope = max((p2 - p1) / max(rows_2 - rows_1, 1), 1.0)
                    return slope, p2 - slope * rows_2

                # Doubling ramp, then the fit. Extrapolating the ceiling from a single row is how a probe
                # ends up throwing a 100x batch at the GPU: the fixed cost of a step hides the per-token
                # one, and an out-of-memory, however well cleaned up, is slow and poisons what follows.
                # So each attempt at most doubles what is known to work, until the two measured points
                # are close enough to the ceiling for the line between them to be the answer.
                candidate = 2
                for _ in range(12):
                    if max_rows:
                        candidate = min(candidate, int(max_rows))
                    candidate = max(int(candidate), low + 1)
                    if candidate <= low or (high is not None and candidate >= high):
                        break
                    fits, peak = probe(candidate)
                    if not fits:
                        high = candidate
                        candidate = (low + high) // 2
                        continue
                    low, peak = candidate, peak
                    points.append((candidate, peak))
                    slope, intercept = fit()
                    if peak >= budget * 0.85:
                        break           # the line is now measured next to the ceiling: good enough
                    predicted = int((budget - intercept) / slope)
                    candidate = min(2 * low, predicted) if len(points) < 3 else predicted
                slope, _ = fit()
                low = min(low, int(max_rows)) if max_rows else low
                if peak > budget and low > 1:
                    # The step that fit went over the requested ceiling: give back the rows the measured
                    # line says do not belong there, so `fill` keeps meaning what it says.
                    slope, intercept = fit()
                    low = max(min(low, int((budget - intercept) / slope)), 1)
                self.fabric.print(f"[probe] ceiling {budget / 2 ** 20:.0f} MiB of "
                                  f"{total / 2 ** 20:.0f} MiB, {slope / max(seq_len, 1) / 2 ** 10:.0f}"
                                  f" KiB per token, {low} rows/rank verified", flush=True)
            # Confirm with consecutive steps (and time them); halve if the steady state turns out to be
            # hungrier than a single step looked like.
            for _ in range(3):
                if low < 1:
                    break
                started, tokens, sequences, done = time.perf_counter(), 0, 0, 0
                for _ in range(max(int(steps), 0)):
                    fits, peak = probe(low, optimize=True)
                    if not fits:
                        break
                    tokens, sequences, done = tokens + low * seq_len, sequences + low, done + 1
                if fits:
                    break
                low = max(low // 2, 1)
            speed = self._report_speed(time.perf_counter() - started, tokens, sequences, done) \
                if done else {}
            if keep_weights:
                with torch.no_grad():
                    for parameter, saved in zip(self.model.parameters(), snapshot):
                        parameter.copy_(saved.to(parameter.device))
                    for optimizer, state in zip(self.optimizers, moments):
                        optimizer.load_state_dict(state)
            snapshot = moments = None
            if cuda:
                torch.cuda.empty_cache()
        finally:
            target.strict_steps = False
            if cycle is not None:
                del selector.pick
            for signal in signals:
                maskerator = getattr(signal, "maskerator", None)
                saved = rates.get(id(maskerator)) if maskerator is not None else None
                if saved is not None:
                    (maskerator.substitution_rate, maskerator.substitution_rate_upper,
                     maskerator.external_schedule) = saved

        environment = getattr(self, "_speed_environment", None) or {}
        report = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                  "gpu": environment.get("device", ""), "precision": training.precision,
                  "world_size": world, "accumulate_grad_batches": accumulate,
                  "max_seq_len": seq_len, "fill": fill,
                  "vram_mib": round(total / 2 ** 20), "base_mib": round(base / 2 ** 20),
                  "budget_mib": round(budget / 2 ** 20),
                  "rows_per_rank": low, "tokens_per_rank": low * seq_len,
                  "rows_per_optimizer_step": low * world * accumulate,
                  "token_per_batch": low * seq_len * accumulate,
                  "token_per_batch_global": low * seq_len * world * accumulate,
                  "peak_mib": round(peak / 2 ** 20),
                  "headroom_mib": round((budget - peak) / 2 ** 20),
                  "bytes_per_token": round(slope / seq_len) if (slope and seq_len) else 0,
                  "bytes_per_row": round(slope), "signals_tested": names,
                  "probes": attempts, "fits": bool(fits and low >= 1),
                  "recommendation": f"training.batch_size={low}"}
        report.update({f"speed_{key}": value for key, value in speed.items() if value is not None})
        for key in ("rows_per_rank", "tokens_per_rank", "token_per_batch", "peak_mib",
                    "headroom_mib", "bytes_per_token"):
            self.logger.log(f"probe/{key}", report[key])
        self.fabric.print("\n[probe] search trace\n" + str(
            Table(["rows", "tokens", "floor MiB", "peak MiB", "held MiB", "fits"],
                  [{"rows": one["rows"], "tokens": one["tokens"], "floor MiB": one.get("floor_mib"),
                    "peak MiB": one["peak_mib"], "held MiB": one.get("reserved_mib"),
                    "fits": "yes" if one["fits"] else "no"} for one in attempts])))
        self.fabric.print(f"[probe] safe batch {low} rows/rank = {low * seq_len} tokens/rank · "
                          f"{report['token_per_batch_global']} tokens per optimizer step · "
                          f"peak {report['peak_mib']}/{report['vram_mib']} MiB "
                          f"(headroom {report['headroom_mib']}) · "
                          f"{report.get('speed_tokens_per_second', 0):,.0f} tokens/s/rank · "
                          f"{report.get('speed_tflops', 0):.1f} TFLOP/s · "
                          f"MFU {report.get('speed_mfu_percent') or 0:.1f}%", flush=True)
        if apply and low >= 1:
            training.batch_size = data.batch_size = low
            self.configs.set_override(f"training.batch_size={low}")
            self.configs.set_override(f"training.token_per_batch={low * seq_len * accumulate}")
            self.dataloader = self.fabric.setup_dataloaders(self.datamodule.train_dataloader())
            self.fabric.print(f"[probe] batch size applied: {low} rows per rank", flush=True)
        if self.fabric.is_global_zero:
            # Loose run artifacts (the memory probe) live next to the checkpoints: each point of a
            # sweep already carries its own run_name, so each report is its own too.
            path = Path(report_path) if report_path else run_dir_for(training)
            report.setdefault("run", training.run_name)
            if getattr(self.configs, "active_sweep", None) is not None:
                report["sweep"] = self.configs.active_sweep.asdict()
            path.mkdir(parents=True, exist_ok=True)
            (path / "memory_probe.json").write_text(json.dumps(report, indent=1, default=str))
        return report

    def run(self, save_at_end=True, save_model=True, save_plugins=False, force_model_save_dir=None, force_plugins_save_dir=None):
        self.train()
        # A stop requested through the control channel already wrote its checkpoint in the loop.
        if save_at_end and not self.control.stop:
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
