"""Token-weighted pretraining with coordinated stops at optimizer boundaries."""
import signal
import time
import torch
from matformer.training.causal_loss import target_mask
from matformer.training.checkpointing import checkpoint_due, save_checkpoint
from matformer.training.training_hooks import TrainingHooks
from matformer.training.validation import validate


def save_training_checkpoint(fabric, state):
    return save_checkpoint(fabric, state)


def coordinated_stop(fabric, state):
    limit = state['train_config'].get('max_time_seconds')
    expired = limit is not None and time.monotonic() - state['_started_at'] >= limit
    flag = torch.tensor(int(expired or state.get('stop_requested', False)), device=fabric.device)
    stop = bool(fabric.all_reduce(flag, reduce_op='max').item())
    if stop:
        state['stop_requested'] = True
    return stop


def budget_targets(fabric, batch, remaining):
    """Allocate an exact remaining target budget in global-rank order."""
    sequence = batch['sequence']
    mask = target_mask(sequence)
    world = getattr(fabric, 'world_size', 1)
    counts = torch.zeros(world, dtype=torch.long, device=fabric.device)
    counts[fabric.global_rank] = int(mask.sum())
    counts = fabric.all_reduce(counts, reduce_op='sum')
    global_count = int(counts.sum())
    local_count = int(counts[fabric.global_rank])
    if remaining is not None and global_count > remaining:
        allowance = max(0, remaining - int(counts[:fabric.global_rank].sum()))
        local_count = min(local_count, allowance)
        positions = mask.flatten().nonzero().flatten()
        mask.flatten()[positions[local_count:]] = False
        sequence.extra_attributes['loss_mask'] = mask
        if 'loss_mask' not in sequence.extra_follow_keys:
            sequence.extra_follow_keys.append('loss_mask')
        global_count = remaining
    return local_count, global_count


def training_loop(fabric, training_dict):
    state = training_dict
    model, optimizers = state['model'], state['optimizer']
    schedulers, loader = state['lr_scheduler'], state['dataloader']
    config, logger = state['train_config'], state['logger']
    hooks = state.get('hooks', TrainingHooks())
    accumulation = config.get('accumulate_grad_batches', 1)
    epochs = config.get('max_epochs', 1)
    validation_every = config.get('validate_every_n_steps', -1)
    if accumulation < 1:
        raise ValueError('accumulate_grad_batches must be positive')
    state.setdefault('batch_in_epoch', 0)
    state.setdefault('tokens_seen', 0)
    state.setdefault('stage_tokens', 0)
    state.setdefault('stage_complete', False)
    state.setdefault('_started_at', time.monotonic())
    last_save = time.monotonic()
    state['last_validation_step'] = None
    if state['stage_complete']:
        return
    model.train()
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    for epoch in range(state['epoch'], epochs):
        num_batches, offset = len(loader), state['batch_in_epoch']
        if not 0 <= offset <= num_batches:
            raise ValueError('Checkpoint batch offset is incompatible with this dataloader')
        if config.get('max_steps') is not None and state['step'] >= config['max_steps']:
            return
        if config.get('max_tokens') is not None and state['tokens_seen'] >= config['max_tokens']:
            return
        if coordinated_stop(fabric, state):
            return
        pending, window_tokens, window_loss = 0, 0, 0.0
        for batch_idx, batch in enumerate(loader, start=offset):
            if pending == 0:
                window_size = min(accumulation, num_batches - batch_idx)
            weighted = isinstance(batch, dict) and 'sequence' in batch
            local_count = global_count = 0
            hit_budget = False
            if weighted:
                limits = []
                if config.get('max_tokens') is not None:
                    limits.append(config['max_tokens'] - state['tokens_seen'] - window_tokens)
                if config.get('stage_max_tokens') is not None:
                    limits.append(config['stage_max_tokens'] - state['stage_tokens'] - window_tokens)
                remaining = max(0, min(limits)) if limits else None
                local_count, global_count = budget_targets(fabric, batch, remaining)
                hit_budget = remaining is not None and global_count >= remaining
                window_tokens += global_count
            pending += 1
            accumulating = pending < window_size and not hit_budget
            with fabric.no_backward_sync(model, enabled=accumulating):
                with fabric.autocast():
                    loss = model.training_step(batch, batch_idx)
                finite = torch.tensor(int(torch.isfinite(loss.detach()).all()), device=fabric.device)
                if not fabric.all_reduce(finite, reduce_op='min').item():
                    raise FloatingPointError(f'Non-finite training loss at batch {batch_idx}')
                # DDP averages gradients across ranks. Normalize the summed
                # token losses once, over the entire accumulation window.
                fabric.backward(loss * local_count if weighted else loss / window_size)
            window_loss += float(loss.detach()) * local_count
            state['batch_in_epoch'] = batch_idx + 1
            if accumulating:
                continue
            if weighted:
                if window_tokens == 0:
                    raise ValueError('Accumulation window contains no next-token targets')
                factor = getattr(fabric, 'world_size', 1) / window_tokens
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(factor)
            for optimizer in optimizers:
                fabric.clip_gradients(model, optimizer, max_norm=config.get('gradient_clip_val', 1.0))
            if hooks.on_before_optimizer:
                hooks.on_before_optimizer(model, optimizers)
            for optimizer in optimizers:
                optimizer.step()
            if hooks.on_after_optimizer:
                hooks.on_after_optimizer(model, optimizers)
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            pending = 0
            state['step'] += 1
            state['tokens_seen'] += window_tokens
            state['stage_tokens'] += window_tokens
            step = state['step']
            state['step_ref'][0] = step
            if schedulers:
                for index, scheduler in enumerate(schedulers):
                    scheduler.step()
                    logger.log(f'lr{index}', scheduler.get_last_lr()[0], step=step)
            if weighted:
                total_loss = fabric.all_reduce(torch.tensor(window_loss, dtype=torch.float64,
                                               device=fabric.device), reduce_op='sum').item()
                logger.log('train/token_loss', total_loss / window_tokens, step=step, prog_bar=True)
                logger.log('train/tokens_seen', state['tokens_seen'], step=step)
            logger.advance()
            window_tokens, window_loss = 0, 0.0
            if hooks.on_step_end:
                hooks.on_step_end(model, step, loss)
            for every_n, function in hooks.periodic.items():
                if step % every_n == 0:
                    function(model, step)
            if batch_idx + 1 == num_batches:
                state['epoch'], state['batch_in_epoch'] = epoch + 1, 0
                state['stage_complete'] = epoch + 1 == epochs
                if hooks.on_epoch_end:
                    hooks.on_epoch_end(model, epoch + 1)
            if config.get('stage_max_tokens') is not None and state['stage_tokens'] >= config['stage_max_tokens']:
                state['stage_complete'] = True
            if coordinated_stop(fabric, state):
                fabric.print('Stop requested; saving at an optimizer boundary.')
                return
            # Validation happens before periodic saves, so a restart sees the
            # same RNG state. Final saves still precede final validation.
            if validation_every > 0 and step % validation_every == 0:
                if state.get('val_dataloader') is not None:
                    validate(fabric, model, state['val_dataloader'], logger, step,
                             config.get('validation_max_batches'),
                             should_stop=lambda: coordinated_stop(fabric, state))
                    state['last_validation_step'] = step
                if hooks.on_validation:
                    hooks.on_validation(model, step)
            seconds = config.get('save_every_n_seconds')
            timed = seconds is not None and time.monotonic() - last_save >= seconds
            due = torch.tensor(int(timed or checkpoint_due(step, config)), device=fabric.device)
            if fabric.all_reduce(due, reduce_op='max').item():
                save_training_checkpoint(fabric, state)
                last_save = time.monotonic()
            if (state.get('stop_requested') or hit_budget or state['stage_complete']
                    or (config.get('max_steps') is not None and step >= config['max_steps'])):
                return
        if pending:
            raise RuntimeError('Dataloader ended before its reported length; unfinished accumulation')
        if state['epoch'] != epoch + 1:
            raise RuntimeError('Dataloader length differs from its reported length')


def launch_training_loop(fabric, training_dict, save_at_end=True):
    def request_stop(signum, frame):
        training_dict['stop_requested'] = True
    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGUSR1, signal.SIGTERM)}
    training_dict['_started_at'] = time.monotonic()
    try:
        training_loop(fabric, training_dict)
        if save_at_end:
            save_training_checkpoint(fabric, training_dict)
        if (training_dict.get('val_dataloader') is not None
                and not coordinated_stop(fabric, training_dict)
                and training_dict.get('last_validation_step') != training_dict['step']):
            validate(fabric, training_dict['model'], training_dict['val_dataloader'],
                     training_dict['logger'], training_dict['step'],
                     training_dict['train_config'].get('validation_max_batches'),
                     should_stop=lambda: coordinated_stop(fabric, training_dict))
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        for key in ('datamodule', 'val_datamodule'):
            if training_dict.get(key) is not None:
                training_dict[key].teardown()
        training_dict['logger'].close()
