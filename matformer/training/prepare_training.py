"""Wire model, data, optimizer, logging and DDP without hiding resume decisions."""
from copy import deepcopy
from datetime import timedelta
import math
import os
from pathlib import Path
import uuid
import torch
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
from matformer.models import PL_ModelWrapper
from matformer.data.matformer_data_module import MatformerDataModule
from matformer.training.ckpt_manager import CheckpointDirectoryManager
from matformer.training.checkpointing import restore_checkpoint, is_snapshot
from matformer.training.logger import MatformerLogger, JSONLLogger, WandbAdapter, TQDMBar
from matformer.training.optimizers import configure_optimizers
from matformer.training.lr_scheduler import configure_lr_schedulers
from matformer.training.utils import detect_device, get_model_class, compile_model


def decide_checkpoint(save_dir, model_name, version=None, checkpoint=None,
                      start_from_scratch=False, start_from_step=None,
                      print_info=True, fabric=None):
    manager = CheckpointDirectoryManager(save_dir, model_name, version)
    choice, error = None, None
    if fabric is None or fabric.global_rank == 0:
        try:
            if checkpoint:
                path = Path(checkpoint)
                if not path.is_file():
                    raise FileNotFoundError(path)
                match = [(v, s) for v, s, p in manager._scan() if p.resolve() == path.resolve()]
                if match:
                    manager.current_version = match[0][0]
                    if any(s > match[0][1] for s, _ in manager._version_ckpts()):
                        manager.current_version = manager._latest_version() + 1
                else:
                    manager.start_from_scratch()
            elif start_from_scratch:
                path, _ = manager.start_from_scratch()
            elif start_from_step is not None:
                path, _ = manager.restart_from_step(start_from_step)
            else:
                path, _ = manager.resume()
            choice = (str(path) if path else None, manager.current_version)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
    if fabric is not None:
        choice, error = fabric.broadcast((choice, error), src=0)
    if error:
        raise RuntimeError(f'Checkpoint selection failed: {error}')
    load_path, manager.current_version = choice
    if fabric and print_info:
        fabric.print(f'Resuming {load_path}' if load_path else 'Starting a new training run')
    return load_path, manager


def count_training_steps(datamodule, batch_size=1, accumulate_grad_batches=1,
                         num_nodes=1, max_steps=None, max_epochs=1):
    """Dataset length already includes its global-rank partition."""
    if min(batch_size, accumulate_grad_batches, max_epochs) < 1:
        raise ValueError('Batch size, accumulation and epochs must be positive')
    num_batches = math.ceil(len(datamodule) / batch_size)
    if not num_batches:
        raise ValueError('Training dataset is empty')
    total = math.ceil(num_batches / accumulate_grad_batches) * max_epochs
    if max_steps is not None:
        if max_steps < 1:
            raise ValueError('max_steps must be positive')
        total = min(total, max_steps)
    return total, num_batches


def validate_training_config(config):
    for key in ('save_every_n_steps', 'validate_every_n_steps'):
        value = config.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f'{key} must be an integer; <= 0 disables it')
    for key in ('accumulate_grad_batches', 'max_epochs', 'max_steps', 'max_tokens',
                'stage_max_tokens', 'schedule_steps', 'keep_last_checkpoints',
                'snapshot_every_n_steps'):
        value = config.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError(f'{key} must be a positive integer or null')
    for key in ('max_time_seconds', 'save_every_n_seconds'):
        if config.get(key) is not None and config[key] <= 0:
            raise ValueError(f'{key} must be positive')
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 1
           for x in config.get('checkpoint_steps', [])):
        raise ValueError('checkpoint_steps must contain positive integer steps')
    is_snapshot(1, config)


def check_resume_config(saved, current, next_data_stage):
    """Operational stopping/logging settings may change; math and data may not."""
    previous = saved['raw_config']
    for section in ('model_class', 'model_config', 'tokenizer', 'precision'):
        if previous.get(section) != current.get(section):
            raise ValueError(f'Checkpoint {section} differs; use --weights-only for a new experiment')
    keys = ('optimizer', 'lr', 'final_lr', 'weight_decay', 'betas', 'eps',
            'lr_scheduling', 'scheduler', 'warmup_steps', 'hold_steps', 'poly_power',
            'annealing_start_step', 'annealing_lr', 'accumulate_grad_batches',
            'gradient_clip_val', 'muon_momentum', 'muon_nesterov', 'muon_ns_steps',
            'no_decay_for_embedding', 'seed')
    for key in keys:
        if previous['training'].get(key) != current['training'].get(key):
            raise ValueError(f'Exact resume cannot change training.{key}')
    requested = current['training'].get('schedule_steps')
    if requested is not None and requested != saved['train_config']['total_steps']:
        raise ValueError('Cannot change the saved LR schedule horizon')
    if not next_data_stage and previous['data'] != current['data']:
        raise ValueError('Data config changed: finish the stage and use --next-data-stage')
    if not next_data_stage and previous['training'].get('max_epochs', 1) != current['training'].get('max_epochs', 1):
        raise ValueError('Use a new data stage rather than changing max_epochs on resume')
    if next_data_stage:
        for key in ('batch_size', 'wanted_from_strategy'):
            if previous['data'].get(key) != current['data'].get(key):
                raise ValueError(f'Data stages must keep data.{key}')


def instantiate_training(model_config, train_config, data_config, tokenizer_config, raw_config,
                         device='auto', precision='bf16-mixed', num_nodes=1, device_count=1,
                         checkpoint=None, start_from_scratch=False, what_to_reset=None,
                         max_steps=None, version=None, _compile=False,
                         next_data_stage=False, weights_only=False):
    if what_to_reset:
        raise ValueError('Partial --reset is unsafe for exact resume; use --weights-only or --next-data-stage')
    if next_data_stage and (weights_only or start_from_scratch):
        raise ValueError('--next-data-stage cannot be combined with a new run')
    if weights_only and not checkpoint:
        raise ValueError('--weights-only requires --checkpoint')
    if max_steps is not None:
        train_config['max_steps'] = max_steps
    validate_training_config(train_config)
    if data_config.get('num_workers', 0) != 0:
        raise ValueError('MDAT uses its own prefetcher: set data.num_workers=0')
    if model_config.training_objective != 'autoregressive':
        raise ValueError('This pretraining runner currently supports autoregressive models')
    if model_config.tie_word_embeddings and os.environ.get('ALLOW_WEIGHT_TYING', '').lower() != 'true':
        raise ValueError('This library gates legacy weight tying: set ALLOW_WEIGHT_TYING=true '
                         'when tie_word_embeddings=true, or explicitly disable tying in the model config')
    if data_config.get('wanted_from_strategy', 'chunked_tokens') != 'chunked_tokens':
        raise ValueError('Use chunked_tokens for resumable pretraining; depth loops use model_config.num_loops')
    # Cross-batch memory hooks need additional cache serialization. Depth loops
    # share weights inside a forward pass and do not require such state.
    if 'recurrence' in str(raw_config['model_config']) or 'previous_state' in str(raw_config['model_config']):
        raise ValueError('Cross-batch recurrence hooks are unsupported for exact resume; use num_loops')
    accelerator, _ = detect_device() if device == 'auto' else ('gpu' if device == 'cuda' else device, device)
    strategy = (DDPStrategy(find_unused_parameters=False, broadcast_buffers=False,
                            timeout=timedelta(minutes=30))
                if num_nodes * device_count > 1 else 'auto')
    fabric = Fabric(accelerator=accelerator, devices=device_count, num_nodes=num_nodes,
                    precision=precision, strategy=strategy)
    fabric.launch()
    fabric.seed_everything(train_config.get('seed', 27))
    raw_config = deepcopy(raw_config)
    raw_config['precision'] = precision
    raw_config['training'] = deepcopy(train_config)
    load_path, manager = decide_checkpoint(raw_config.get('save_dir', './checkpoints'),
        train_config.get('checkpoint_name', 'matformer_model'), version, checkpoint,
        start_from_scratch, fabric=fabric)
    saved = torch.load(load_path, map_location='cpu', weights_only=False) if load_path else None
    if next_data_stage and saved is None:
        raise ValueError('--next-data-stage requires a completed-stage checkpoint')
    if saved is not None and not weights_only:
        if saved.get('format_version') != 2:
            raise ValueError('Legacy checkpoint: use --weights-only; complete resume state is unavailable')
        if saved['world_size'] != fabric.world_size:
            raise ValueError('Exact resume requires the same total GPU/process count')
        stage_id = data_config.get('stage_id', 'default')
        if next_data_stage and (not saved['stage_complete'] or stage_id == saved['stage_id']
                                or stage_id in saved['completed_stages']):
            raise ValueError('The previous data stage must be complete and the new stage_id unused')
        check_resume_config(saved, raw_config, next_data_stage)
    if weights_only:
        manager.current_version = manager._latest_version() + 1
    model = PL_ModelWrapper(get_model_class(raw_config['model_class']), model_config,
                            None, str(fabric.device), batch_size=data_config['batch_size'],
                            train_config=train_config)
    if weights_only:
        result = model.load_checkpoint(checkpoint=saved)
        if result['missing'] or result['unexpected']:
            raise ValueError('Checkpoint weights do not match this model')

    def build_data(path, validation=False):
        module = MatformerDataModule(
            mdat_path=path, iteration_modality='chunked_tokens',
            pad_token_id=model_config.pad_token_id,
            varlen_strategy=tokenizer_config.get('varlen_strategy', 'padding'),
            mdat_strategy=data_config['mdat_strategy'],
            mdat_view=data_config.get('validation_mdat_view' if validation else 'mdat_view', 'default'),
            mdat_selection=None if validation else data_config.get('mdat_selection'),
            prefetch_buffer=data_config.get('prefetch_buffer', 0),
            max_seq_len=model_config.max_position_embeddings,
            batch_size=data_config.get('validation_batch_size', data_config['batch_size']) if validation else data_config['batch_size'],
            num_devices=fabric.world_size)
        module.setup()
        return module

    datamodule = build_data(data_config['data_root'])
    val_data = build_data(data_config['validation_data_root'], True) if data_config.get('validation_data_root') else None
    total, batches = count_training_steps(datamodule, data_config['batch_size'],
        train_config.get('accumulate_grad_batches', 1), max_epochs=train_config.get('max_epochs', 1))
    train_config['total_steps'] = (saved['train_config']['total_steps'] if saved and not weights_only
                                   else train_config.get('schedule_steps', total))
    train_config['num_batches'] = batches
    optimizers = configure_optimizers(model, train_config)
    schedulers = configure_lr_schedulers(train_config, optimizers)
    raw_model = model
    if _compile:
        model = compile_model(model)
    model, *optimizers = fabric.setup(model, *optimizers)
    # Fabric must route training_step through DDP.forward, not bypass its reducer.
    model.mark_forward_method('training_step')
    model.mark_forward_method('validation_step')
    loader = fabric.setup_dataloaders(datamodule.train_dataloader(),
                                      use_distributed_sampler=False, move_to_device=False)
    val_loader = (fabric.setup_dataloaders(val_data.train_dataloader(),
                  use_distributed_sampler=False, move_to_device=False) if val_data else None)
    run_id = saved['run_id'] if saved and not weights_only else fabric.broadcast(uuid.uuid4().hex[:12], src=0)
    step = saved['step'] if saved and not weights_only else 0
    logger, step_ref = MatformerLogger(), [step]
    if fabric.global_rank == 0:
        log_dir = Path(raw_config.get('save_dir', './checkpoints')) / 'logs' / run_id
        logger.add(JSONLLogger(log_dir / 'metrics.jsonl'))
        mode = os.environ.get('WANDB_MODE', raw_config.get('wandb_mode', 'offline'))
        if mode not in ('online', 'offline', 'disabled'):
            raise ValueError('wandb_mode must be online, offline, or disabled')
        if mode != 'disabled':
            from lightning.pytorch.loggers import WandbLogger
            # Offline segments get distinct IDs; group links restarts. Offline
            # W&B does not implement online resume semantics.
            segment_id = run_id if mode == 'online' else f'{run_id}-{uuid.uuid4().hex[:8]}'
            wandb_logger = WandbLogger(name=raw_config.get('wandb_run_name', 'matformer'),
                project=raw_config.get('wandb_project', 'matformer'), id=segment_id,
                group=run_id, resume='allow' if mode == 'online' else None,
                mode=mode, save_dir=str(log_dir), config=raw_config)
            logger.add(WandbAdapter(wandb_logger))
            _ = wandb_logger.experiment
        logger.add(TQDMBar(total=train_config.get('max_steps') or train_config['total_steps'], initial=step))
    raw_model.setup_logging(logger, step_ref)
    state = dict(model=model, raw_model=raw_model, optimizer=optimizers,
        lr_scheduler=schedulers, datamodule=datamodule, dataloader=loader,
        val_datamodule=val_data, val_dataloader=val_loader, ckpt_manager=manager,
        train_config=train_config, raw_config=raw_config, logger=logger, step_ref=step_ref,
        epoch=0, step=0, batch_in_epoch=0, tokens_seen=0, stage_tokens=0,
        stage_id=data_config.get('stage_id', 'default'), stage_complete=False,
        completed_stages=[], run_id=run_id, token_weighted=True)
    fabric.seed_everything(train_config.get('seed', 27) + fabric.global_rank)
    if saved is not None and not weights_only:
        restore_checkpoint(fabric, state, saved, next_data_stage=next_data_stage)
    fabric.print(f'world_size={fabric.world_size}; effective sequences/update='
                 f'{fabric.world_size * data_config["batch_size"] * train_config.get("accumulate_grad_batches", 1)}; '
                 f'LR horizon={train_config["total_steps"]}')
    return fabric, state
