"""Token-budgeted, single-epoch MLM annealing over a frozen mixture."""
from contextlib import nullcontext
from datetime import timedelta
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from matformer.tensors_dataclasses import UnpaddedTensor
from matformer.training.optimizers import configure_optimizers
from matformer.training.wandb_logging import WandbLogger
from scripts.benchmarks.benchmark_albertone_mlm import source, construct, masker, masked_loss, digest, publish, ddp_weighting_check

DOMAINS = ('wikipedia', 'academic', 'books', 'gazzetta_ufficiale', 'subtitles')


def verify_training_data(root, receipt):
    """Verify frozen cache bytes and warm sequential reads before GPU iteration."""
    if receipt.get('fixture') or receipt.get('status') != 'passed':
        raise ValueError('Production requires a validated real corpus')
    for split, domains in receipt['splits'].items():
        for domain, result in domains.items():
            for name, expected in result['sha256'].items():
                path = Path(root) / 'data' / domain / split / name
                if digest(path) != expected:
                    raise ValueError(f'Prepared data changed: {path}')


class TokenMixture:
    def __init__(self, root, split):
        self.root, self.split = Path(root), split
        self.indices, self.tokens = [], []
        for name in DOMAINS:
            directory = self.root / 'data' / name / split
            self.indices.append(np.load(directory / 'index.npy', allow_pickle=False))
            self.tokens.append(np.memmap(directory / 'tokens.bin', mode='r', dtype='<u2'))
        self.ends = np.cumsum([len(x) for x in self.indices])
        self.starts = np.r_[0, self.ends[:-1]]
        self.lengths = np.concatenate([x['length'] for x in self.indices]).astype(np.int64)

    def __len__(self):
        return int(self.ends[-1])

    def rows(self, numbers):
        rows = []
        for number in numbers:
            domain = int(np.searchsorted(self.ends, number, side='right'))
            entry = self.indices[domain][number - self.starts[domain]]
            start, length = int(entry['offset']), int(entry['length'])
            rows.append(self.tokens[domain][start:start + length])
        return rows


def pack(rows, device):
    """Direct equivalent of TokenCollator(..., 'unpadding'), without padding slots."""
    lengths = np.asarray([len(row) for row in rows], dtype=np.int32)
    if not len(lengths) or int(lengths.min()) < 2 or int(lengths.max()) > 1024:
        raise ValueError('Invalid independent sequence lengths')
    values = torch.from_numpy(np.concatenate(rows).astype(np.int64))
    indices = torch.from_numpy(np.concatenate([np.arange(n, dtype=np.int64) + i * 1024
                                               for i, n in enumerate(lengths)]))
    cumulative = torch.from_numpy(np.r_[np.int32(0), np.cumsum(lengths, dtype=np.int32)])
    return UnpaddedTensor(tensor=values, indices=indices, cu_seqlens=cumulative,
                         max_seq_len=int(lengths.max()), original_seq_len=1024,
                         batch_size=len(rows)).to(device)


def microbatches(order, lengths, budget=16384):
    """Each real sequence occurs once; keep document tails and the partial batch."""
    batches, begin, tokens = [], 0, 0
    for i, number in enumerate(order):
        length = int(lengths[number])
        if length > budget:
            raise ValueError('Sequence exceeds the microbatch token budget')
        if tokens + length > budget:
            batches.append(order[begin:i])
            begin, tokens = i, 0
        tokens += length
    if begin < len(order):
        batches.append(order[begin:])
    return batches


def learning_rate(update, total, peak=1e-4, warmup_fraction=.01):
    warmup = max(1, math.ceil(total * warmup_fraction))
    if total < 2:
        raise ValueError('Annealing needs at least two optimizer updates')
    if update <= warmup:
        return peak * update / warmup
    progress = (update - warmup) / (total - warmup)
    return peak * .5 * (1 + math.cos(math.pi * progress))


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def parameter_digest(model):
    h = hashlib.sha256()
    for name, value in nn.Module.state_dict(model).items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def save_checkpoint(path, raw, optimizer, run, update, counts, rank, world):
    rng = dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(),
               numpy=np.random.get_state(), python=random.getstate())
    states = [None] * world if rank == 0 else None
    dist.gather_object(rng, states, dst=0)
    if rank == 0:
        temporary = path.with_suffix('.pending')
        weights = cpu_tree(nn.Module.state_dict(raw))
        inverse = {actual: stable for stable, actual in raw._mappings.items()}
        canonical = {inverse.get(key, key): value for key, value in weights.items()}
        torch.save(dict(format_version=1, model=weights, state_dict=canonical, config=raw.config,
                        optimizer=cpu_tree(optimizer.state_dict()), run=run,
                        update=update, counts=counts, rng_by_rank=states), temporary)
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        temporary.replace(path)
    dist.barrier()


@torch.no_grad()
def validate(raw, cfg, data, rank, world, budget):
    raw.eval()
    totals, metrics = np.zeros(3, dtype=np.float64), {}
    for domain, name in enumerate(DOMAINS):
        order = np.arange(data.starts[domain], data.ends[domain], dtype=np.int64)[rank::world]
        corruption = masker(cfg, .28, 9127 + domain * 101 + rank)
        sums = torch.zeros(3, dtype=torch.float64, device='cuda')
        for numbers in microbatches(order, data.lengths, budget):
            original = pack(data.rows(numbers), 'cuda')
            masked, _ = corruption(original)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = raw(masked, is_causal=False)
                loss = masked_loss(output, original, masked)
            sums[0] += loss.detach().double()
            sums[1] += masked.cloze_mask.sum()
            sums[2] += original.tensor.numel()
        dist.all_reduce(sums)
        values = sums.cpu().numpy()
        if values[1] <= 0 or not np.isfinite(values).all():
            raise ValueError('Invalid validation targets/loss')
        totals += values
        metrics[f'val/{name}/loss'] = float(values[0] / values[1])
        metrics[f'val/{name}/targets'] = int(values[1])
    metrics.update({'val/loss': float(totals[0] / totals[1]),
                    'val/perplexity': math.exp(min(float(totals[0] / totals[1]), 80)),
                    'val/targets': int(totals[1]), 'val/input_tokens': int(totals[2]),
                    'val/masking_rate': .28})
    raw.train()
    return metrics


def train(args):
    rank, local_rank, world = (int(os.environ[k]) for k in ('SLURM_PROCID', 'SLURM_LOCALID', 'SLURM_NTASKS'))
    assert world == 4, 'This experiment requires one full four-GPU node'
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl', rank=rank, world_size=world,
                            timeout=timedelta(minutes=10), device_id=torch.device('cuda', local_rank))
    began = time.monotonic()
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, stop)
    lock = logger = None
    try:
        output_error = [None]
        if rank == 0:
            try:
                args.output.mkdir(parents=True, exist_ok=True)
                lock = (args.output / '.training.lock').open('a')
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if not args.resume and (args.output / 'run.json').exists():
                    raise FileExistsError('Use a fresh output root or an explicit checkpoint resume')
            except OSError as error:
                output_error[0] = str(error)
        # Only the rank holding the lock checks the directory. Other ranks may
        # reach this point after rank zero has published run.json.
        dist.broadcast_object_list(output_error, src=0)
        if output_error[0] is not None:
            raise RuntimeError(output_error[0])
        receipt_path = args.data / 'data-validation.json'
        receipt = json.loads(receipt_path.read_text())
        assert receipt['status'] == 'passed'
        if not args.smoke:
            assert json.loads(args.preflight.read_text())['status'] == 'passed'
        data_hash = digest(receipt_path)
        cfg, weights, optimizer_config, provenance = source(args.checkpoint)
        optimizer_config = {k: v for k, v in optimizer_config.items()
                            if k in ('optimizer', 'lr', 'weight_decay', 'betas', 'eps', 'no_decay_for_embedding')
                            or k.startswith('muon_')}
        optimizer_config['lr'] = args.lr
        assert args.lr == 1e-4 and args.rate in (.15, .28, .40)
        assert cfg.cloze_probability == 1 and not cfg.random_probability and not cfg.same_probability
        if rank == 0:
            assert digest(args.checkpoint) == '5bbc6630a430daf405579cf793bc9918b846ed9a68594820ee4bff6ecd2c110c'
            if not args.smoke:
                verify_training_data(args.data, receipt)
        dist.barrier()
        if args.smoke:
            ddp_weighting_check(local_rank, rank, world)
        torch.manual_seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        random.seed(args.seed + rank)
        raw = construct(cfg, weights, 'flash').cuda().train()
        from matformer.modules.attention_implementations.flash import FlashAttentionKernel
        kernels = [module for module in raw.modules() if isinstance(module, FlashAttentionKernel)]
        assert kernels
        for kernel in kernels:
            kernel.deterministic = True
        del weights
        optimizer = configure_optimizers(raw, optimizer_config)[0]
        model = DDP(raw, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False)
        # DDP changes its bucket layout after the first backward. Prime that
        # layout on every invocation so a resumed first update uses the same
        # reduction grouping as an uninterrupted later update. No optimizer
        # step is taken, and RNG state is preserved.
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        dummy = pack([[cfg.bos_token_id, cfg.eos_token_id]], 'cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            zero = model(dummy, is_causal=False).tensor.sum() * 0
        zero.backward()
        optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        del dummy, zero
        training, validation = TokenMixture(args.data, 'train'), TokenMixture(args.data, 'validation')
        order = np.random.default_rng(args.seed).permutation(len(training))
        batches = microbatches(order, training.lengths, args.micro_tokens)
        stride = world * args.accumulation
        full_updates = math.ceil(len(batches) / stride)
        total_updates = min(full_updates, args.smoke) if args.smoke else full_updates
        run = dict(run_id=args.run_id, data_validation_sha256=data_hash,
                   source_checkpoint=str(args.checkpoint.resolve()),
                   source_checkpoint_sha256='5bbc6630a430daf405579cf793bc9918b846ed9a68594820ee4bff6ecd2c110c',
                   source_provenance=provenance, optimizer=optimizer_config,
                   mask_rate=args.rate, validation_mask_rate=.28, validation_mask_seed=9127,
                   sequence_length=1024, world_size=world, seed=args.seed,
                   micro_input_token_budget=args.micro_tokens, accumulation=args.accumulation,
                   maximum_global_input_tokens=args.micro_tokens * stride,
                   total_updates=total_updates, warmup_updates=max(1, math.ceil(total_updates * .01)),
                   schedule='1% linear warmup, cosine decay to zero',
                   deterministic_flash_backward=True,
                   ddp_bucket_priming=True,
                   input_content_tokens=receipt.get('content_tokens'),
                   masking_policy='Original 100% MASK substitution; PAD excluded, BOS/EOS eligible',
                   wandb_project='UniTO-LLM', wandb_entity='team-jerik',
                   wandb_run_name=args.run_id, smoke=bool(args.smoke))
        counts = dict(input_tokens=0, content_tokens=0, targets=0, sequences=0)
        start, restored_rng = 0, None
        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            assert checkpoint['run'] == run, 'Resume cannot change data, mask rate, optimizer or schedule'
            nn.Module.load_state_dict(raw, checkpoint['model'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            counts, start = checkpoint['counts'], checkpoint['update']
            restored_rng = checkpoint['rng_by_rank'][rank]
            del checkpoint
        if rank == 0:
            publish(args.output / 'run.json', run)
            logger = WandbLogger(args.output / 'logs' / args.run_id, args.run_id, run,
                                 mode='disabled' if args.smoke else 'offline', step=start,
                                 tokens_seen=counts['targets'], resumed=bool(args.resume))
            logger.run.define_metric('train/input_tokens_seen')
            logger.run.define_metric('compare/val_loss_vs_input_tokens', step_metric='train/input_tokens_seen')
            logger.run.define_metric('compare/train_loss_vs_input_tokens', step_metric='train/input_tokens_seen')
        if restored_rng:
            torch.set_rng_state(restored_rng['cpu'])
            torch.cuda.set_rng_state(restored_rng['cuda'])
            np.random.set_state(restored_rng['numpy'])
            random.setstate(restored_rng['python'])
        dist.barrier()

        def log(metrics, update, training_update=False):
            if rank != 0:
                return
            values = {**metrics, 'train/tokens_seen': counts['targets'],
                      'train/input_tokens_seen': counts['input_tokens'],
                      'train/content_tokens_seen': counts['content_tokens']}
            if 'val/loss' in values:
                values['compare/val_loss_vs_input_tokens'] = values['val/loss']
            if 'train/token_loss' in values:
                values['compare/train_loss_vs_input_tokens'] = values['train/token_loss']
            logger.log(values, update)
            if training_update:
                logger.advance()
            else:
                logger.flush()
            row = dict(step=update, segment_id=logger.segment_id, **values)
            with (logger.directory / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)

        if not args.resume:
            if logger:
                logger.update_status(phase='validation')
            baseline = validate(raw, cfg, validation, rank, world, args.micro_tokens)
            log(baseline, 0)
        for update in range(start + 1, total_updates + 1):
            torch.cuda.synchronize()
            step_start = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            lr = learning_rate(update, total_updates, args.lr)
            for group in optimizer.param_groups:
                group['lr'] = lr
            corruption = masker(cfg, args.rate, args.seed + update * world + rank)
            prepared = []
            local_counts = torch.zeros(4, dtype=torch.int64, device='cuda')
            for micro in range(args.accumulation):
                index = (update - 1) * stride + micro * world + rank
                real = index < len(batches)
                rows = training.rows(batches[index]) if real else [np.asarray([5, 6], dtype=np.uint16)]
                original = pack(rows, 'cuda')
                masked, _ = corruption(original)
                if not real:
                    masked.cloze_mask.zero_()
                else:
                    local_counts += torch.tensor([original.tensor.numel(), original.tensor.numel() - 2 * len(rows),
                                                  int(masked.cloze_mask.sum()), len(rows)], device='cuda')
                prepared.append((original, masked))
            dist.all_reduce(local_counts)
            targets = local_counts[2]
            assert targets > 0
            loss_total = torch.zeros((), dtype=torch.float64, device='cuda')
            for micro, (original, masked) in enumerate(prepared):
                with model.no_sync() if micro + 1 < len(prepared) else nullcontext():
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        output = model(masked, is_causal=False)
                        loss = masked_loss(output, original, masked)
                    (loss * (world / targets)).backward()
                loss_total += loss.detach().double()
                del output, loss
            norm = torch.nn.utils.clip_grad_norm_(raw.parameters(), 1., error_if_nonfinite=True)
            if update == 1 and any(p.grad is None for p in raw.parameters() if p.requires_grad):
                raise ValueError('Missing trainable parameter gradients')
            optimizer.step()
            dist.all_reduce(loss_total)
            value = float((loss_total / targets).item())
            assert math.isfinite(value)
            for key, number in zip(counts, local_counts.cpu().tolist()):
                counts[key] += number
            seconds = torch.tensor(time.monotonic() - step_start, device='cuda', dtype=torch.float64)
            dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
            log({'train/token_loss': value, 'train/loss': value, 'train/lr': lr,
                 'train/masked_targets': int(targets), 'train/update_input_tokens': int(local_counts[0]),
                 'train/gradient_norm': float(norm), 'train/update_seconds': float(seconds),
                 'train/input_tokens_per_second': float(local_counts[0] / seconds)}, update, True)
            del prepared
            if args.smoke and args.rate == .28 and update == 1 and not args.resume:
                save_checkpoint(args.output / 'resume-test.pt', raw, optimizer, run, update, counts, rank, world)
            if update % args.validate_every == 0 or update == total_updates:
                if logger:
                    logger.update_status(phase='validation')
                result = validate(raw, cfg, validation, rank, world, args.micro_tokens)
                log(result, update)
            stopping = torch.tensor(int(stopped or time.monotonic() - began > args.max_seconds), device='cuda')
            dist.all_reduce(stopping, op=dist.ReduceOp.MAX)
            if not args.smoke and (update % args.save_every == 0 or update == total_updates or stopping.item()):
                if logger:
                    logger.update_status(phase='checkpointing')
                save_checkpoint(args.output / 'last.pt', raw, optimizer, run, update, counts, rank, world)
            if stopping.item() and update != total_updates:
                raise TimeoutError('Bounded allocation stopped safely; checkpoint retained, no automatic GPU successor')
        sample = next(raw.parameters()).detach().flatten()[:1024].clone()
        reference = sample.clone()
        dist.broadcast(reference, src=0)
        torch.testing.assert_close(sample, reference, rtol=0, atol=0)
        if not args.smoke:
            assert counts['input_tokens'] == receipt['input_tokens']['train']
            assert counts['content_tokens'] == receipt['content_tokens']['train']
            assert counts['sequences'] == len(training)
        if rank == 0:
            resume_max_difference = None
            if args.smoke and args.rate == .28:
                reference_path = args.output / 'uninterrupted-final-weights.pt'
                final_weights = cpu_tree(nn.Module.state_dict(raw))
                if args.resume:
                    expected = torch.load(reference_path, map_location='cpu', weights_only=True)
                    differences = []
                    for key, value in final_weights.items():
                        torch.testing.assert_close(value, expected[key], rtol=1e-5, atol=2e-7)
                        differences.append(float((value.float() - expected[key].float()).abs().max()))
                    resume_max_difference = max(differences)
                    del expected
                else:
                    torch.save(final_weights, reference_path)
                del final_weights
            final = dict(status='completed', completed=True, job_id=os.environ['SLURM_JOB_ID'],
                         run=run, counts=counts, updates=total_updates, final_lr=lr,
                         validation=result, elapsed_seconds=time.monotonic() - began,
                         parameter_sha256=parameter_digest(raw), rank_parameter_agreement=True,
                         resumed_max_parameter_difference=resume_max_difference,
                         peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
            publish(args.output / ('resumed-result.json' if args.resume else 'result.json'), final)
            logger.update_status(phase='completed')
    except BaseException:
        if logger:
            logger.update_status(phase='failed')
        raise
    finally:
        if logger:
            logger.close()
        if lock:
            lock.close()
        dist.destroy_process_group()
