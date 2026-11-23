"""Bounded, allocation-only AlBERTone MLM throughput measurement.

Preparation checks the complete pretrained encoder/head and stages a small book
sample. GPU measurements update disposable in-memory models only. Production
training, dataset preparation, and checkpoint writing are deliberately separate.
"""
import argparse
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import fields
from datetime import timedelta
import gc
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import time
import zlib

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from matformer.data.collator import TokenCollator
from matformer.model_config import ModelConfig
from matformer.matformer_registry import registry
from matformer.training.attention_backends import portable_state, validate_flash_attention
from matformer.training.optimizers import configure_optimizers
from matformer.transformer_blocks import TransformerWithLMHead
from matformer.utils.masked_models import Maskerator
from matformer.utils.matformer_cache import CachedStuff


def publish(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


def source(path):
    # The source is the already pinned, trusted local project checkpoint.
    checkpoint = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    saved = checkpoint['hyper_parameters']['config']
    cfg = ModelConfig(**{f.name: deepcopy(getattr(saved, f.name))
                         for f in fields(ModelConfig) if hasattr(saved, f.name)})
    weights, ignored_legacy_aliases = {}, []
    saved_weights = checkpoint['state_dict']
    for key, value in saved_weights.items():
        canonical = key.removeprefix('model.')
        if key != canonical and canonical in saved_weights and not key.endswith('_extra_state'):
            equal = torch.equal(value, saved_weights[canonical])
            if not equal and canonical not in ('lm_head.weight', 'lm_head.bias'):
                raise ValueError(f'Conflicting encoder alias: {key}')
            # PL_ModelWrapper._load_stable_state_dict consumes lm_head.* and
            # ignores model.lm_head.*: those prefixed names are not in _mappings.
            # Preserve that explicit native convention, recording both entries.
            ignored_legacy_aliases.append(dict(key=key, canonical=canonical, equal=equal))
            continue
        key = canonical
        if key.endswith('_extra_state'):
            continue
        if key in weights and not torch.equal(weights[key], value):
            raise ValueError(f'Conflicting legacy duplicate: {key}')
        weights[key] = value
    head, embedding = weights['lm_head.weight'], weights['encoder.embed_tokens.weight']
    shared = (head.untyped_storage().data_ptr() == embedding.untyped_storage().data_ptr()
              and head.storage_offset() == embedding.storage_offset())
    # Historical configs can claim tying even when the implementation did not tie.
    provenance = dict(configured_weight_tying=cfg.tie_word_embeddings,
                      stored_head_embedding_shared=shared,
                      stored_head_embedding_equal=torch.equal(head, embedding),
                      ignored_legacy_aliases=ignored_legacy_aliases,
                      head_selection='Canonical lm_head.* stable keys, matching PL_ModelWrapper._load_stable_state_dict')
    cfg.tie_word_embeddings = shared
    cfg.lm_head_bias = 'lm_head.bias' in weights
    assert cfg.training_objective == 'masked' and not cfg.is_causal
    assert cfg.max_position_embeddings == 1024 and cfg.vocab_size == 32768
    train = deepcopy(checkpoint['hyper_parameters']['train_config'])
    assert train['optimizer'] == 'muon'
    train['lr'] = 1e-4
    return cfg, weights, train, provenance


def construct(cfg, weights, attention):
    cfg = deepcopy(cfg)
    cfg.default_layer.attn_impl = attention
    for layer in cfg.custom_layers.values():
        layer.attn_impl = attention
    os.environ['ALLOW_WEIGHT_TYING'] = str(cfg.tie_word_embeddings).lower()
    registry.load_modules()
    cache = CachedStuff()
    cache.registry = deepcopy(registry)
    cache.registry.reset_selections()
    model = TransformerWithLMHead(cfg, cache=cache)
    mapped = {model._mappings.get(key, key): value for key, value in weights.items()}
    mapped = portable_state(model, mapped)
    # Bypass permissive legacy loading; the MLM head is required, too.
    nn.Module.load_state_dict(model, mapped, strict=True)
    for key, value in nn.Module.state_dict(model).items():
        if not torch.equal(value, mapped[key]):
            raise AssertionError(f'Checkpoint restore changed {key}')
    if attention == 'flash':
        validate_flash_attention(model, unpadding=True)
    return model


def masker(cfg, rate, seed):
    return Maskerator(mask_token=cfg.mask_token_id, substitution_rate=float(rate),
                      pad_token_id=cfg.pad_token_id, cloze_prob=cfg.cloze_probability,
                      random_prob=cfg.random_probability, same_prob=cfg.same_probability,
                      vocab_size=cfg.vocab_size, random_seed=seed)


def masked_loss(output, original, masked):
    selected = masked.cloze_mask
    return F.cross_entropy(output.tensor[selected].float(), original.tensor[selected], reduction='sum')


def prepare(args):
    import lmdb
    import zstandard
    from transformers import AutoTokenizer

    started = time.perf_counter()
    cfg, weights, train, provenance = source(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    assert len(tokenizer) == cfg.vocab_size
    for key in ('pad_token_id', 'bos_token_id', 'eos_token_id', 'mask_token_id'):
        assert getattr(tokenizer, key) == getattr(cfg, key), key
    before = args.books.stat()
    env = lmdb.open(str(args.books), subdir=False, readonly=True, lock=False,
                    readahead=False, create=False, max_readers=1)
    chunks, document_ids, sampled = [], [], []
    try:
        with env.begin(write=False) as txn:
            stored_length = txn.get(b'__len__')
            documents = int(stored_length) if stored_length else None
            for number in range(min(documents or 512, 512)):
                data = txn.get(str(number).encode())
                if data is None:
                    continue
                if data.startswith(b'\x28\xb5\x2f\xfd'):
                    data = zstandard.ZstdDecompressor().decompress(data, max_output_size=128 << 20)
                elif data[:1] == b'x':
                    data = zlib.decompress(data)
                text = data.decode('utf-8')
                try:
                    value = json.loads(text)
                except json.JSONDecodeError:
                    value = text
                if isinstance(value, str):
                    text = value
                elif isinstance(value, dict):
                    text = next((value[k] for k in ('text', 'content', 'document')
                                 if isinstance(value.get(k), str)), None)
                    if text is None:
                        raise ValueError(f'Unknown book record fields: {list(value)}')
                else:
                    raise ValueError(f'Unexpected book record type: {type(value)}')
                # Bounded benchmark sampling, not full-corpus tokenization/counting.
                ids = tokenizer.encode(text[:262144], add_special_tokens=False)
                sampled.append(dict(document_id=number, prefix_tokens=len(ids)))
                for offset in range(0, len(ids) - 1021, 1022):
                    chunks.append([cfg.bos_token_id, *ids[offset:offset + 1022], cfg.eos_token_id])
                    document_ids.append(number)
                    if len(chunks) == 2048:
                        break
                if len(chunks) == 2048:
                    break
    finally:
        env.close()
    after = args.books.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'Books changed while sampling'
    assert len(chunks) == 2048, f'Insufficient full-length sample: {len(chunks)}'
    order = np.random.default_rng(27).permutation(len(chunks))
    tokens = np.asarray(chunks, dtype=np.int64)[order]
    np.save(args.output / 'books-full1024.npy', tokens, allow_pickle=False)

    model = construct(cfg, weights, 'sdpa').eval()
    collator = TokenCollator(cfg.pad_token_id, 1024, 'unpadding')
    batch = collator([tokens[0, :19].tolist(), tokens[1, :31].tolist()])['sequence']
    original = batch.tensor.clone()
    corruption = masker(cfg, .28, 27)
    masked, _ = corruption(batch)
    assert torch.equal(batch.tensor, original)
    assert torch.equal(masked.cu_seqlens, batch.cu_seqlens)
    assert torch.equal(masked.tensor[~masked.cloze_mask], original[~masked.cloze_mask])
    assert (masked.tensor[masked.cloze_mask] == cfg.mask_token_id).all()
    repeated, _ = masker(cfg, .28, 27)(batch)
    assert torch.equal(masked.tensor, repeated.tensor)
    with torch.no_grad():
        output = model(masked, is_causal=False)
        loss = masked_loss(output, batch, masked)
        assert torch.isfinite(loss) and loss > 0
        labels = original.clone()
        labels[~masked.cloze_mask] = -100
        reference = F.cross_entropy(output.tensor.float(), labels, ignore_index=-100, reduction='sum')
        torch.testing.assert_close(loss, reference)
    report = dict(status='passed', job_id=os.environ['SLURM_JOB_ID'],
                  elapsed_seconds=time.perf_counter() - started,
                  checkpoint=str(args.checkpoint), checkpoint_sha256=digest(args.checkpoint),
                  source_book_file=str(args.books), source_book_bytes=before.st_size,
                  source_book_mtime_ns=before.st_mtime_ns, stored_book_documents=documents,
                  sample_documents=sampled, sampled_unique_documents=len(set(document_ids)),
                  sample_chunks=len(tokens), sample_input_tokens=int(tokens.size),
                  sample_sha256=digest(args.output / 'books-full1024.npy'),
                  model_parameters=sum(p.numel() for p in model.parameters()),
                  checkpoint_metadata=provenance, optimizer=train,
                  tests=['tokenizer identity', 'strict complete encoder and MLM-head restore',
                         'source values exactly restored', 'immutable books source',
                         'deterministic masking', 'original input preserved',
                         'unmasked tokens and document boundaries preserved',
                         'finite actual-model MLM loss', 'same-position masked loss equals ignore-index reference'],
                  limitations=['First book prefixes; no academic file access.',
                               'Full 1024-token chunks only; not the production 40/40/20 mixture.',
                               'This sample does not establish the full books token count.'])
    publish(args.output / 'preparation.json', report)
    print(json.dumps(report), flush=True)


def maximum(values):
    tensor = torch.tensor(values, device='cuda', dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.tolist()


def ddp_weighting_check(local_rank, rank, world):
    torch.manual_seed(71)
    raw = nn.Linear(4, 7, bias=False).cuda()
    reference = deepcopy(raw)
    model = DDP(raw, device_ids=[local_rank])
    x = torch.arange(world * 3 * 4, device='cuda', dtype=torch.float32).reshape(world * 3, 4) / 50
    targets = torch.arange(world * 3, device='cuda') % 7
    # Unequal prediction counts across ranks and two accumulation microbatches.
    selected = torch.arange(world * 3, device='cuda') % 4 != 0
    count = selected.sum()
    for micro, indices in enumerate((torch.tensor([rank * 3], device='cuda'),
                                     torch.tensor([rank * 3 + 1, rank * 3 + 2], device='cuda'))):
        with model.no_sync() if micro == 0 else nullcontext():
            logits = model(x[indices])
            take = selected[indices]
            loss = F.cross_entropy(logits[take], targets[indices][take], reduction='sum')
            (loss * world / count).backward()
    F.cross_entropy(reference(x)[selected], targets[selected]).backward()
    torch.testing.assert_close(raw.weight.grad, reference.weight.grad, atol=2e-7, rtol=2e-6)
    del model, raw, reference
    dist.barrier()


def batch_list(tokens, cfg, microbatch, rank, world):
    rows = tokens[rank * 512:(rank + 1) * 512]
    assert len(rows) == 512 and 512 % microbatch == 0 and world == 4
    collator = TokenCollator(cfg.pad_token_id, 1024, 'unpadding')
    return [collator(rows[i:i + microbatch].tolist())['sequence'].to('cuda')
            for i in range(0, 512, microbatch)]


def local_probe(raw, cfg, train, batch):
    optimizer = configure_optimizers(raw, train)[0]
    corruption = masker(cfg, .40, 27)
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        masked, _ = corruption(batch)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = raw(masked, is_causal=False)
            loss = masked_loss(output, batch, masked) / masked.cloze_mask.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(raw.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return dict(seconds=times, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)


def gpu(args):
    rank = int(os.environ['SLURM_PROCID'])
    world = int(os.environ['SLURM_NTASKS'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    assert world == 4
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl', rank=rank, world_size=world,
                            timeout=timedelta(minutes=4), device_id=torch.device('cuda', local_rank))
    begun = time.perf_counter()
    preparation = json.loads((args.output / 'preparation.json').read_text())
    assert preparation['status'] == 'passed'
    cfg, weights, train, provenance = source(args.checkpoint)
    tokens = np.load(args.output / 'books-full1024.npy', allow_pickle=False)
    ddp_weighting_check(local_rank, rank, world)
    report = dict(status='running', job_id=os.environ['SLURM_JOB_ID'],
                  node=os.environ['SLURM_JOB_NODELIST'], rank_zero_host=socket.gethostname(),
                  world_size=world, gpu=torch.cuda.get_device_name(),
                  gpu_total_gib=torch.cuda.get_device_properties(local_rank).total_memory / 2**30,
                  torch_version=torch.__version__, sequence_length=1024, precision='bf16-mixed',
                  attention='FlashAttention-2 unpadded, bidirectional ALiBi',
                  head='Legacy full-token vocabulary projection; loss only at masked positions',
                  optimizer='Original Muon implementation, fresh state per case, LR 1e-4',
                  checkpoint_sha256=preparation['checkpoint_sha256'],
                  checkpoint_metadata=provenance, global_input_tokens_per_update=2097152,
                  warmup_updates=args.warmup, measured_updates=args.steps,
                  correctness=['Strict full model/head restore', 'DDP accumulated global target-weighted gradients match single-process reference'],
                  probes=[], results=[], limitations=[
                      'Small completed-books sample; no production mixture or document-tail distribution.',
                      'Fixed distinct full-length batches replayed from GPU memory with fresh dynamic masks.',
                      'Measured update includes masking, full forward/loss, backward, accumulated DDP sync, clipping and Muon step.',
                      'Data preparation/loading/transfers, validation, checkpoint writes and queue time excluded.',
                      'Original masking policy retained: PAD excluded, BOS/EOS eligible.',
                      'Weights updated only in memory; no trained checkpoint is exported.',
                      'No activation checkpointing or torch.compile.'])

    microbatch = None
    for size in (8, 16):
        if rank == 0:
            print(f'Memory probe: {size} sequences/GPU at 40% masking', flush=True)
        raw = construct(cfg, weights, 'flash').cuda().train()
        batches = batch_list(tokens, cfg, size, rank, world)
        try:
            result = local_probe(raw, cfg, train, batches[0])
            ok = result['peak_reserved_gib'] < report['gpu_total_gib'] * .88
            status = 'passed' if ok else 'insufficient_memory_margin'
        except torch.cuda.OutOfMemoryError:
            result, ok, status = {}, False, 'out_of_memory'
        raw.zero_grad(set_to_none=True)
        del batches, raw
        gc.collect()
        torch.cuda.empty_cache()
        failure = maximum([not ok])[0]
        peaks = maximum([result.get('peak_allocated_gib', 0), result.get('peak_reserved_gib', 0)])
        report['probes'].append(dict(microbatch=size, all_ranks_passed=not bool(failure),
                                    rank_zero_status=status, maximum_peak_allocated_gib=peaks[0],
                                    maximum_peak_reserved_gib=peaks[1], rank_zero_details=result))
        if not failure:
            microbatch = size
        if rank == 0:
            publish(args.output / 'result.json', report)
    if microbatch is None:
        raise RuntimeError('Neither microbatch candidate fits with a safe memory margin')

    for rate in (.15, .28, .40):
        if time.perf_counter() - begun > args.max_seconds:
            raise TimeoutError('Bounded benchmark time exhausted')
        if rank == 0:
            print(f'Fresh AlBERTone101: mask={rate}, microbatch={microbatch}, accumulation={512 // microbatch}', flush=True)
        raw = construct(cfg, weights, 'flash').cuda().train()
        optimizer = configure_optimizers(raw, train)[0]
        model = DDP(raw, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False)
        batches = batch_list(tokens, cfg, microbatch, rank, world)
        corruption = masker(cfg, rate, 27000 + rank)
        measurements = []
        for step in range(args.warmup + args.steps):
            if time.perf_counter() - begun > args.max_seconds:
                raise TimeoutError('Bounded benchmark time exhausted')
            optimizer.zero_grad(set_to_none=True)
            dist.barrier()
            torch.cuda.synchronize()
            if step == args.warmup:
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            masked_batches = [corruption(batch)[0] for batch in batches]
            global_targets = torch.stack([batch.cloze_mask.sum() for batch in masked_batches]).sum()
            dist.all_reduce(global_targets)
            total_loss = torch.zeros((), device='cuda')
            for micro, (original, masked) in enumerate(zip(batches, masked_batches)):
                with model.no_sync() if micro + 1 < len(batches) else nullcontext():
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        output = model(masked, is_causal=False)
                        loss = masked_loss(output, original, masked)
                    (loss * (world / global_targets)).backward()
                total_loss += loss.detach()
                del output, loss
            norm = torch.nn.utils.clip_grad_norm_(raw.parameters(), 1., error_if_nonfinite=True)
            if step == 0 and any(p.grad is None for p in raw.parameters() if p.requires_grad):
                raise RuntimeError('Trainable parameter missing a gradient')
            optimizer.step()
            torch.cuda.synchronize()
            seconds = maximum([time.perf_counter() - started])[0]
            dist.all_reduce(total_loss)
            loss_value = (total_loss / global_targets).item()
            assert np.isfinite(loss_value)
            row = dict(update=step, seconds=seconds, input_tokens=2097152,
                       masked_targets=global_targets.item(), loss=loss_value, gradient_norm=norm.item())
            if step >= args.warmup:
                measurements.append(row)
            if rank == 0:
                print(json.dumps(dict(masking_rate=rate, warmup=step < args.warmup, **row)), flush=True)
        # A deterministic parameter slice must agree after all accumulated updates.
        check = next(raw.parameters()).detach().flatten()[:256].clone()
        reference = check.clone()
        dist.broadcast(reference, src=0)
        torch.testing.assert_close(check, reference, atol=0, rtol=0)
        memory = maximum([torch.cuda.max_memory_allocated() / 2**30,
                          torch.cuda.max_memory_reserved() / 2**30])
        mean = statistics.mean(row['seconds'] for row in measurements)
        result = dict(masking_rate=rate, microbatch_per_gpu=microbatch,
                      accumulation=512 // microbatch, parameters=sum(p.numel() for p in raw.parameters()),
                      mean_update_seconds=mean, input_tokens_per_second=2097152 / mean,
                      hours_per_billion_input_tokens=1e9 * mean / 2097152 / 3600,
                      peak_allocated_gib=memory[0], peak_reserved_gib=memory[1],
                      fresh_source_and_optimizer=True, rank_parameter_agreement=True, measurements=measurements)
        report['results'].append(result)
        if rank == 0:
            publish(args.output / 'result.json', report)
            print(json.dumps(result), flush=True)
        del optimizer, model, raw, batches, masked_batches, check, reference
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
    report['status'] = 'completed'
    report['elapsed_seconds'] = time.perf_counter() - begun
    if rank == 0:
        publish(args.output / 'result.json', report)
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--checkpoint', type=Path, default=Path('checkpoints/retrieval/sources/AlBERTone101/checkpoint.ckpt'))
    parser.add_argument('--tokenizer', type=Path, default=Path('tokenizers/gettone-retrieval'))
    parser.add_argument('--books', type=Path, default=Path('/leonardo_work/IscrC_ALBERTIN/books/data.dat'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--max-seconds', type=int, default=1450)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Preparation and measurements require a Slurm allocation')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', 2)))
    (prepare if args.prepare else gpu)(args)


if __name__ == '__main__':
    main()
