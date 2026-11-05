"""Run with torchrun --standalone --nproc_per_node=2 (small CPU/Gloo test)."""
from copy import deepcopy
from pathlib import Path
import os
import signal
import sys
import torch
import torch.distributed as dist
from matformer.data.collator import TokenCollator
from matformer.model_config import ModelConfig
from matformer.models import PL_ModelWrapper
from matformer.transformer_blocks import Autoregressive_Model
from matformer.training.checkpointing import save_checkpoint
from matformer.training.optimizers import configure_optimizers
from matformer.training.lr_scheduler import configure_lr_schedulers
from matformer.training.training_loop import training_loop, launch_training_loop
from matformer.training.training_hooks import TrainingHooks
from test_mdat_selection import make_dataset
from test_pretraining_runtime import tiny_config, build, close


def main():
    torch.set_num_threads(1)
    dist.init_process_group('gloo')
    rank, world = dist.get_rank(), dist.get_world_size()
    root = Path(sys.argv[1])
    if rank == 0:
        make_dataset(root / 'data')
    dist.barrier()
    cfg = tiny_config(root / 'data', root / 'checkpoints')
    cfg['training'].update(max_epochs=2, max_steps=1, accumulate_grad_batches=100)
    fabric, state = build(cfg, device_count=world)
    rows = [row.tolist() for batch in state['datamodule'].train_dataloader()
            for row in batch['sequence'].detach() if len(row) > 1]
    all_rows = [None] * world
    dist.all_gather_object(all_rows, rows)
    initial = deepcopy(state['raw_model'].state_dict())
    gradients = {}
    def capture_gradients(model, optimizers):
        gradients.update({name: p.grad.detach().clone()
                          for name, p in state['raw_model'].named_parameters()})
    state['hooks'] = TrainingHooks(on_before_optimizer=capture_gradients)
    training_loop(fabric, state)
    assert state['epoch'] == 1 and state['batch_in_epoch'] == 0
    # An uneven DDP epoch must match one update over all real targets.
    if rank == 0:
        reference = PL_ModelWrapper(Autoregressive_Model, ModelConfig(**cfg['model_config']), None, 'cpu')
        reference.load_state_dict(initial)
        optimizers = configure_optimizers(reference, cfg['training'])
        schedule_config = {**cfg['training'], 'total_steps': 20}
        configure_lr_schedulers(schedule_config, optimizers)
        batch = TokenCollator(0, 6)([row for rows in all_rows for row in rows])
        reference.training_step(batch).backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
        for name, parameter in reference.named_parameters():
            torch.testing.assert_close(parameter.grad, gradients[name], rtol=2e-5, atol=1e-7)
        for optimizer in optimizers:
            optimizer.step()
        for name, value in reference.state_dict().items():
            # Summation order differs; Adam magnifies tiny near-zero gradient
            # differences. Resume below still requires bit-for-bit equality.
            torch.testing.assert_close(value, state['raw_model'].state_dict()[name], rtol=2e-5, atol=2e-6)
    dist.barrier()
    path = save_checkpoint(fabric, state)
    state['train_config']['max_steps'] = 2
    training_loop(fabric, state)
    expected = deepcopy(state['raw_model'].state_dict())
    expected_tokens = state['tokens_seen']
    close(state)
    cfg['training']['max_steps'] = 2
    fabric, resumed = build(cfg, device_count=world, checkpoint=str(path))
    training_loop(fabric, resumed)
    for name, value in expected.items():
        torch.testing.assert_close(value, resumed['raw_model'].state_dict()[name], rtol=0, atol=0)
    assert resumed['tokens_seen'] == expected_tokens
    # Every rank must end with identical weights and counters.
    parameters = torch.cat([p.detach().flatten() for p in resumed['raw_model'].parameters()])
    gathered = [torch.empty_like(parameters) for _ in range(world)]
    dist.all_gather(gathered, parameters)
    for tensor in gathered:
        torch.testing.assert_close(tensor, parameters, rtol=0, atol=0)
    close(resumed)
    dist.barrier()
    cfg['save_dir'] = str(root / 'budget')
    cfg['training'].update(max_epochs=1, max_tokens=5, accumulate_grad_batches=2)
    fabric, bounded = build(cfg, device_count=world)
    training_loop(fabric, bounded)
    assert bounded['tokens_seen'] == 5
    close(bounded)
    cfg['save_dir'] = str(root / 'signal')
    cfg['training'].pop('max_tokens')
    cfg['training'].pop('max_steps')
    fabric, signaled = build(cfg, device_count=world)
    def signal_one_rank(model, step, loss):
        if rank == world - 1:
            os.kill(os.getpid(), signal.SIGUSR1)
    signaled['hooks'] = TrainingHooks(on_step_end=signal_one_rank)
    launch_training_loop(fabric, signaled)
    assert signaled['step'] == 1 and signaled['stop_requested']
    assert signaled['ckpt_manager'].resume()[1] == 1
    dist.barrier()
    if rank == 0:
        saved = torch.load(path, weights_only=False)
        assert len(saved['rng_per_rank']) == world
        assert len(saved['datamodule']['per_rank']) == world
        print('PASS: DDP global mean, exact resume, per-rank state, exact token cap, coordinated SIGUSR1 save')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
