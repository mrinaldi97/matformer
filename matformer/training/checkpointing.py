"""Complete, atomic DDP checkpoints at optimizer boundaries (trusted files only)."""
import os
import random
import numpy as np
import torch
import torch.distributed as dist


def capture_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'].cpu())


def gather_rank_state(value):
    if not dist.is_initialized():
        return [value]
    states = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
    dist.gather_object(value, states, dst=0)
    return states


def is_snapshot(step, config):
    interval = config.get('snapshot_every_n_steps', 0)
    if (interval and step % interval == 0) or step in config.get('checkpoint_steps', []):
        return True
    base = config.get('checkpoint_log_base')
    if base is not None:
        if isinstance(base, bool) or not isinstance(base, int) or base < 2:
            raise ValueError('checkpoint_log_base must be an integer >= 2')
        value = 1
        while value < step:
            value *= base
        return value == step
    return False


def checkpoint_due(step, config):
    every = config.get('save_every_n_steps', 0)
    return (every > 0 and step % every == 0) or is_snapshot(step, config)


def save_checkpoint(fabric, state):
    """All ranks must call; only rank zero writes and publishes the file."""
    manager = state['ckpt_manager']
    rank_states = gather_rank_state(capture_rng())
    data_state = state['datamodule'].state_dict()
    path = manager._ckpt_path(state['step'])
    error = None
    if fabric.global_rank == 0:
        raw_model = state['raw_model']
        payload = {
            'format_version': 2,
            'model_state_dict': raw_model.state_dict(),
            'state_dict': raw_model.stable_state_dict(),
            'config': raw_model.config,
            'optimizer': [o.state_dict() for o in state['optimizer']],
            'scheduler': [s.state_dict() for s in (state['lr_scheduler'] or [])],
            'precision': fabric.strategy.precision.state_dict(),
            'rng_per_rank': rank_states,
            'world_size': fabric.world_size,
            'datamodule': data_state,
            'raw_config': state['raw_config'],
            'train_config': state['train_config'],
            'run_id': state['run_id'],
            **{key: state[key] for key in ('step', 'epoch', 'batch_in_epoch',
                 'tokens_seen', 'stage_tokens', 'stage_id', 'completed_stages', 'stage_complete')},
        }
        temporary = path.with_name(path.name + '.tmp')
        try:
            with temporary.open('wb') as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            manager.write_resume(state['step'], path)
            # Deletion is opt-in: absent/null preserves every checkpoint.
            keep = state['train_config'].get('keep_last_checkpoints')
            if keep is not None:
                candidates = sorted((s, p) for s, p in manager._version_ckpts()
                                    if not is_snapshot(s, state['train_config']))
                for _, old in candidates[:-keep]:
                    old.unlink()
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
    error = fabric.broadcast(error, src=0)
    if error:
        raise RuntimeError(f'Checkpoint write failed: {error}')
    fabric.barrier()
    fabric.print(f'Checkpoint saved: {path} (targets={state["tokens_seen"]})')
    return path


def restore_checkpoint(fabric, state, checkpoint, *, next_data_stage=False):
    if checkpoint.get('format_version') != 2:
        raise ValueError('Legacy checkpoints lack complete RNG/precision state. '
                         'Use --weights-only to start a new run from legacy weights.')
    if checkpoint['world_size'] != fabric.world_size:
        raise ValueError('Exact resume requires the same total GPU/process count')
    state['raw_model'].load_state_dict(checkpoint['model_state_dict'], strict=True)
    for key in ('optimizer', 'lr_scheduler'):
        saved = checkpoint['scheduler' if key == 'lr_scheduler' else key]
        current = state[key] or []
        if len(saved) != len(current):
            raise ValueError(f'Checkpoint {key} count differs')
        for obj, saved_state in zip(current, saved):
            obj.load_state_dict(saved_state)
    fabric.strategy.precision.load_state_dict(checkpoint['precision'])
    for key in ('step', 'tokens_seen', 'run_id', 'completed_stages'):
        state[key] = checkpoint[key]
    if next_data_stage:
        if not checkpoint['stage_complete']:
            raise ValueError('Finish the previous data stage before --next-data-stage')
        if state['stage_id'] == checkpoint['stage_id'] or state['stage_id'] in state['completed_stages']:
            raise ValueError('A new, unused data.stage_id is required')
        state['completed_stages'] = [*state['completed_stages'], checkpoint['stage_id']]
    else:
        if state['stage_id'] != checkpoint['stage_id']:
            raise ValueError('Changed data.stage_id requires --next-data-stage')
        for key in ('epoch', 'batch_in_epoch', 'stage_tokens', 'stage_complete'):
            state[key] = checkpoint[key]
        state['datamodule'].load_state_dict(checkpoint['datamodule'])
        if state['batch_in_epoch'] == 0:
            state['datamodule'].mdat.__dict__.pop('_skip_reinit', None)
    state['step_ref'][0] = state['step']
    restore_rng(checkpoint['rng_per_rank'][fabric.global_rank])
