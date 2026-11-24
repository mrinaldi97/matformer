---
title: MatformerTrainer
sources:
  - matformer/training/matformer_trainer.py
---
# MatformerTrainer
The single entry point for anything related to models' training is the MatformerTrainer class. Its role is to correctly set up the training procedure, including handling distributed training, iterating through the batches of data performing the training steps, launch evaluations or generic tasks whenever certain conditions are met and finally save the trained models. 
Its init is divided in eight steps, that we will briefly enumerate; more details can be understood by directly looking at the code. 
1. "_setup_fabric()" => Currently, Matformer is hardcoded to rely on torch fabric, which is a good choice to simplify the code regarding distributed training and mixed precision strategies. In th future, this function could became an optional step. Its scope it's very simple: it sets up the parallelism strategy (currently hardcoded to "ddp", see the [[intro/limitations | Limitations]] page and, if training runs on more than one rank, distributed the work on all the requested accelerator.
2. _setup_checkpoint() => This function works with [[training/checkpointing | CheckpointDirectoryManager]] class to understand if the present training run should be started from scratch or resume (either from the latest trained step or for a specific step/version specified by the user). If a checkpoint is resumed, a [[loading/checkpoint-holder | CheckpointHolder]] (of native matformer type) is created; if the training starts from a pre-trained model, a [[loading/checkpoint-holder | CheckpointHolder]] of the correct type is created (for example, an huggingface's checkpoint holder to fine-tune a model coming from HF); it returns None if the training starts/restarts from scratch.
3. _setup_model() => Creates or load the model using the [[training/wrapper | MatformerModelWrapper]]. NOTE: this steps expands the original MatformerConfigs object with the configuration of the loaded model: this allows, for example, to have a ClassificationModel agnostic to stuff such as hidden_size or layer number, because those fields depend on the base model loaded and are populated in this step, inside the [[training/wrapper | MatformerModelWrapper]]. After, missing/unexpected keys, both from the base model/checkpoint as well as from the [[loading/checkpoint-holder#additional-weights | AdditionalWeights]], is reported. Finally, model is eventually compiled respecting the [[components/configuration | configuration]] configuration.
4. _setup_data() => here a [[data/data-module | MatformerDataModule]] is instantiated, and the total number of training steps, batches and token per batch is computed (see [[file:matformer/training/matformer_trainer.py:count_training_steps()]]
5. setup_optimization() => [[this.optimizers | Optimizers]] and [[this.lr_schedulers]] are configured;
6. _restore_states() => If training is resuming from a checkpoint, states of optimisers, lr_scheduler, datamodule, step and epoch are restored from the [[loading/checkpoint-holder | CheckpointHolder]] 
7. _setup_logger()
8. _setup_fabric_wrapping()

After the trainer is initialized, run() launch the training, which is handled by the train() function, which wraps the trainig loop and, if requested, saves model and/or pugins in the checkpoint's folder at the training's end. 

## The training loop
Before starting the training loop, validation groups are normalised so that, if they are set in the config with a fraction of total steps, the config is overridden with their actual value according to the total training steps.
After this, the training starts, optionally controlled by [[training/control | Control]] which allow for dynamically stop the training or perform an arbitrary [[tasks/index | task]] during the training, useful for scenarios like booked Slurm clusters or to do model's diagnosis without interrupting the training. The actual loop proceeds through all the epochs and steps, performs evaluations (actually, any [[tasks/index | MatformerTasks]]) at the requested step/epochs and save intermediate checkpoints. It exposes useful generic callbacks:
- self.model.on_epoch_start at each epoch start 
- currently other callbacks are commented out

When the model runs as a [[components/virtual-models | VirtualModel]], the validation step replays a databank. At the end of a step-based validation the trainer calls `activation_store.switch_databank(training.data.name, rewind=False)`, so the next pass resumes from its position. After an epoch-based validation it calls it with `rewind=True`, so the next epoch starts from the beginning.


## Optimizers

Optimizers are configured by `configure_optimizers(model, training_config)` in `matformer/training/optimizers.py`. The field is `training.optimizer`, and the current options are:

| Name | What it builds |
|------|----------------|
| `adamw` | `torch.optim.AdamW(model.parameters(), lr, weight_decay)` |
| `adam` | `torch.optim.Adam(model.parameters(), lr, weight_decay)` |
| `muon` | Matformer's `Muon`: 2-D matrices go to Muon, embeddings, the LM head, convolutions and 1-D parameters go to AdamW |
| `muonclip` | the external `muon` package's `MuonClip`, with the model geometry derived from `num_attention_heads` and `hidden_size` |
| `muonflash` | the experimental `flash_muon` path: a flash Muon for matrices plus AdamW for the rest |

The function returns a list, including the single-optimizer case, because Fabric is set up with lists and the Muon paths can return more than one optimizer. `no_decay_for_embedding` is implemented for Muon and moves the LM head and the token embeddings to the no-decay group.

## LR schedulers

`configure_lr_schedulers(training_config, optimizers)` builds one `LambdaLR` per optimizer, or returns `None` when `lr_scheduling` is false or when the data module could not report a length (the trainer disables scheduling in that case).

The schedule has three phases:

1. **Warmup**: `warmup_steps_fraction * total_steps` steps, linear from 0 to 1.
2. **Hold**: `hold_steps_fraction * total_steps` steps at the peak learning rate.
3. **Decay**: the rest, from 1 to `final_lr / lr` using the `scheduler` function:
   * `cosine` (and the deprecated `custom` alias): cosine decay;
   * `linear`: linear decay;
   * `polynomial`: `(1 - p) ** polynomial_lr_power`;
   * `constant`: stays at the peak.

`annealing_start_step` and `annealing_lr` add an optional final annealing phase: from that step on, the schedule decays from `annealing_lr` to `final_lr` instead of using the three phases above. The fractions are fractions of `total_steps`, not absolute step counts, so a config does not have to know its own length.

## Saving

`save(step, epoch, what)` is the single entry point for periodic saves and for the final save. `what` is `model`, `plugins` or `both`; the training config's `save_what` decides what periodic saves write, while `run()` chooses at the end. A model save contains the stable state dict, the optimizer and scheduler states, the datamodule state, step, epoch and the configuration, so a resume is exact. The write goes to a `.tmp` file and is renamed on rank 0 after a barrier, so a crash never leaves a half-written checkpoint under the real name.

`CheckpointDirectoryManager` owns the naming (`{name}_v{version}_s{step}.ckpt`), the resume log (`matformer_resume.csv`) and the decision of what to resume from. See [[training/checkpointing | checkpointing]] for the details.

## The memory probe

`probe_memory(...)` measures the batch size that fits in the available memory. It builds mock batches with the dataloader's `collate_fn`, measures two points, extrapolates linearly in tokens, verifies the prediction and bisects when the prediction was wrong. It runs each configured learning signal in turn and uses the highest configured masking rate. The weights and optimizer state are restored afterwards. With `apply=True` it writes the measured batch size into the config and rebuilds the dataloader. The report is written to `memory_probe.json` in the run's log folder.
