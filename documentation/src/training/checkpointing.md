---
title: Checkpointing
sources:
  - matformer/training/ckpt_manager.py
  - matformer/dataclasses/checkpoint_dataclasses.py
  - matformer/loading/matformer_holder.py
---
# Checkpointing

A Matformer checkpoint is a `.ckpt` file with the stable state dict, the optimizer and scheduler states, the datamodule state, the step, the epoch and the configuration. A run can resume from it, and a new run can start from its weights with the states reset.

## Naming

`CheckpointDirectoryManager` writes:

```
{save_dir}/{checkpoint_name}_v{version}_s{step}.ckpt
```

A resumed run keeps its version and adds steps. `start_from_scratch`, `start_from_step` and an explicit checkpoint start a new version. `matformer_resume.csv` records each saved checkpoint. Plugin checkpoints are written to `{name}_v{version}_s{step}_plugins/`.

## Start mode

`decide_checkpoint(checkpoint, start_from_scratch, start_from_step)` returns `(load_path, step, reason)`:

* `checkpoint`: an explicit `.ckpt`. The run continues its state in a new version. To fine-tune, set `what_to_reset` to `optimizer,lr_scheduler,datamodule,steps`.
* `start_from_scratch`: a new version with nothing loaded.
* `start_from_step`: the closest checkpoint at or after that step in the current version, then a new version.
* otherwise: the latest checkpoint of the current version, or scratch if none exists.

## Checkpoint section

A config can name the model to load:

```yaml
checkpoint:
  - name: finale
    path: "$CHECKPOINT_ASSETS/FINALE_32768.ckpt"
```

A checkpoint entry can represent a series:

```yaml
checkpoint:
  - name: probes
    path: "$CHECKPOINT_ASSETS/sequence_classification/ironita"
    every_n_steps: 500
  - name: donors
    path: "$CHECKPOINT_ASSETS/a.ckpt | $CHECKPOINT_ASSETS/b.ckpt"
```

A folder is read as a checkpoint series, a path with top-level pipes is a sweep dimension, and the two combine. `CheckpointDirectoryManager.select` implements the selection: `model_name` selects one run, `every_n_steps` selects steps that are a multiple of it, `versions` selects versions, and `last` keeps the N newest. A location without Matformer checkpoints is returned unchanged, so a HuggingFace repo or a loose folder still works.

## Saving

`MatformerTrainer.save(step, epoch, what)` writes periodic and final checkpoints. `what` is `model`, `plugins` or `both`. `save_what` controls periodic saves, and `run()` controls the final save. The checkpoint is written to a `.tmp` file and renamed on rank 0 after a barrier.

## Resuming

`_setup_checkpoint` creates a `MatformerCheckpointHolder` for a Matformer checkpoint or the appropriate holder for a base model. `_restore_states` restores the optimizer, the scheduler, the datamodule, the step and the epoch. `what_to_reset` lists the parts that are not restored.

The holder is described in [[loading/checkpoint-holder | CheckpointHolder]].
