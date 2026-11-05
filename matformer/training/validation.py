"""Token-weighted held-out autoregressive evaluation."""
from itertools import islice
import torch
from matformer.training.checkpointing import capture_rng, restore_rng


@torch.no_grad()
def validate(fabric, model, dataloader, logger, step, max_batches=None, should_stop=None):
    if max_batches is not None and max_batches < 1:
        raise ValueError('validation_max_batches must be positive')
    was_training = model.training
    rng = capture_rng()
    model.eval()
    totals = torch.zeros(2, dtype=torch.float64, device=fabric.device)
    try:
        batches = iter(dataloader)
        if max_batches is not None:
            batches = islice(batches, max_batches)
        for batch in batches:
            if should_stop is not None and should_stop():
                return None
            if batch.get('worker_has_finished', False):
                continue
            with fabric.autocast():
                result = model.validation_step(batch)
            totals[0] += result['loss_sum'].double()
            totals[1] += result['tokens']
        totals = fabric.all_reduce(totals, reduce_op='sum')
        if totals[1].item() == 0:
            raise RuntimeError('Validation produced no next-token targets')
        loss = (totals[0] / totals[1]).item()
        if not torch.isfinite(totals).all():
            raise FloatingPointError('Non-finite validation loss')
        logger.log('val/loss', loss, step=step, prog_bar=True)
        logger.log('val/tokens', totals[1].item(), step=step)
        fabric.print(f'Validation step {step}: loss={loss:.6f}, targets={int(totals[1].item())}')
        return loss
    finally:
        restore_rng(rng)
        model.train(was_training)
        if hasattr(model, 'cache'):
            model.cache.storage.pop('aux_losses', None)
