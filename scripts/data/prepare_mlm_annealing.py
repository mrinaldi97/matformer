"""Prepare one immutable, document-disjoint Gettone mixture inside Slurm.

Non-Wikipedia LMDBs are read-only. Wikipedia uses a separately completed
extraction. Sampling is without replacement, with quotas measured in content
tokens. Complete books are retained; only the last sampled document of another
source may be shortened to its quota, and its unused part is never reused.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np
from tokenizers import Tokenizer
import lmdb

from scripts.data.count_lmdb_tokens import read_text, sha256, publish, mixture

ROOT = Path('/leonardo_work/IscrC_ALBERTIN')
NAMES = ('wikipedia', 'academic', 'books', 'gazzetta_ufficiale', 'subtitles')
INDEX = np.dtype([('offset', '<u8'), ('length', '<u2'), ('document', '<u8')])
WORKER_SOURCE = WORKER_TOKENIZER = None


def open_source(path):
    # Full-corpus preparation benefits from read-ahead on the shared filesystem.
    # Unlike sparse counting probes, it will touch many adjacent leaf/value pages.
    return lmdb.open(str(path), subdir=False, readonly=True, lock=False,
                     readahead=True, create=False, max_readers=1)


def worker_init(source, tokenizer, kind):
    global WORKER_SOURCE, WORKER_TOKENIZER
    WORKER_SOURCE = (open_source(source) if kind == 'lmdb' else open(source, 'rb'), kind)
    WORKER_TOKENIZER = Tokenizer.from_file(str(tokenizer))
    WORKER_TOKENIZER.no_padding()
    WORKER_TOKENIZER.no_truncation()


def tokenize_one(number):
    source, kind = WORKER_SOURCE
    if kind == 'lmdb':
        text = read_text(source, int(number))
        identity = str(number)
    else:
        source.seek(int(number))
        row = json.loads(source.readline())
        text, identity = row['text'], row['id']
    encoded = WORKER_TOKENIZER.encode(text, add_special_tokens=False).ids
    return (int(number), identity, np.asarray(encoded, dtype='<u2'),
            hashlib.sha256(text.encode()).hexdigest(),
            hashlib.sha256(' '.join(text.split()).encode()).hexdigest())


def identity(path):
    stat = path.stat()
    return dict(path=str(path.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                sha256=sha256(path))


def book_plan(args):
    path = args.output / 'plan.json'
    if path.exists():
        return json.loads(path.read_text())
    receipt = json.loads((args.books_count / 'result.json').read_text())
    ledger = args.books_count / 'document-counts.jsonl'
    assert receipt['status'] == 'completed' and sha256(ledger) == receipt['document_counts_sha256']
    rows = sorted((json.loads(line) for line in ledger.read_text().splitlines()),
                  key=lambda row: row['document_id'])
    assert len(rows) == receipt['documents']
    order = np.random.default_rng(args.seed).permutation(len(rows))
    target = round(receipt['content_tokens'] * .005)
    cumulative = np.cumsum([rows[i]['content_tokens'] for i in order], dtype=np.int64)
    count = int(np.argmin(np.abs(cumulative - target))) + 1
    heldout = sorted(int(i) for i in order[:count])
    actual = sum(rows[i]['content_tokens'] for i in heldout)
    plan = dict(seed=args.seed, books_validation_fraction_requested=.005,
                books_validation_target=target, books_validation_documents=heldout,
                books_validation_content_tokens=actual,
                books_validation_fraction_actual=actual / receipt['content_tokens'],
                books_source_identity=receipt['identity'],
                tokenizer_sha256=sha256(args.tokenizer / 'tokenizer.json'),
                train_quotas=mixture(receipt['content_tokens'] - actual),
                validation_quotas=mixture(actual), sequence_length=1024,
                chunking='Non-overlapping, within-document 1022 content tokens + BOS/EOS; keep tails',
                validation_mask_rate=.28, validation_mask_seed=9127,
                source_sampling='Seeded document permutation without replacement',
                scope='Held out from this annealing campaign; original pretraining coverage unknown')
    publish(path, plan)
    return plan


def candidates(source, kind):
    if kind == 'lmdb':
        env = open_source(source)
        with env.begin() as txn:
            declared = int(txn.get(b'__len__'))
            keys = sorted(int(k) for k in txn.cursor().iternext(values=False) if k.isdigit())
        env.close()
        if keys != list(range(declared)):
            raise ValueError(f'Incomplete LMDB upload: {source}')
        return np.arange(declared, dtype=np.int64)
    offsets = []
    with source.open('rb') as stream:
        while True:
            offset = stream.tell()
            if not stream.readline():
                break
            offsets.append(offset)
    return np.asarray(offsets, dtype=np.int64)


def prepare_split(args, plan, domain, split, source, kind, choices, excluded):
    directory = args.output / 'data' / domain / split
    result_path = directory / 'result.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        assert result['status'] == 'completed' and result['plan_sha256'] == sha256(args.output / 'plan.json')
        assert result['source'] == identity(source)
        assert all(sha256(directory / name) == value for name, value in result['sha256'].items())
        return result
    if directory.exists():
        raise RuntimeError(f'Unpublished partial cache needs explicit recovery: {directory}')
    directory.mkdir(parents=True)
    before = identity(source)
    quota = plan[f'{split}_quotas'][domain]
    indices, used, total, input_total, selected_docs, rejected = [], set(excluded), 0, 0, 0, 0
    began = time.monotonic()
    last_progress = began
    with (directory / 'tokens.bin').open('wb') as output, \
            (directory / 'documents.jsonl').open('w') as records, \
            ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                initializer=worker_init, initargs=(source, args.tokenizer / 'tokenizer.json', kind)) as pool:
        # Bound both submitted work and returned token arrays, including very long books.
        for start in range(0, len(choices), args.workers * 2):
            if time.monotonic() > args.deadline:
                raise TimeoutError('Bounded CPU preparation expired; no automatic successor')
            batch = choices[start:start + args.workers * 2]
            for number, doc_id, tokens, text_hash, normalized_hash in pool.map(tokenize_one, batch):
                if total == quota:
                    break
                if not len(tokens) or normalized_hash in used:
                    rejected += 1
                    if domain == 'books':
                        raise ValueError('A book would be lost to an empty/duplicate/excluded text')
                    continue
                used.add(normalized_hash)
                take = min(len(tokens), quota - total)
                if domain == 'books' and take != len(tokens):
                    raise ValueError('Book count no longer agrees with pinned count receipt')
                for offset in range(0, take, 1022):
                    content = tokens[offset:min(offset + 1022, take)]
                    chunk = np.concatenate((np.asarray([5], dtype='<u2'), content,
                                            np.asarray([6], dtype='<u2')))
                    indices.append((input_total, len(chunk), number))
                    output.write(chunk.tobytes())
                    input_total += len(chunk)
                records.write(json.dumps(dict(document=number, source_id=doc_id,
                    text_sha256=text_hash, normalized_sha256=normalized_hash,
                    full_content_tokens=len(tokens), used_content_tokens=take)) + '\n')
                total += take
                selected_docs += 1
            if time.monotonic() - last_progress >= 30 or total == quota:
                progress = dict(domain=domain, split=split, content_tokens=total, quota=quota,
                                documents=selected_docs, seconds=time.monotonic() - began)
                publish(directory / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
                last_progress = time.monotonic()
            if total == quota:
                break
        output.flush()
        os.fsync(output.fileno())
        records.flush()
        os.fsync(records.fileno())
    if total != quota:
        raise ValueError(f'Insufficient unique {domain}/{split} tokens: {total} < {quota}')
    np.save(directory / 'index.npy', np.asarray(indices, dtype=INDEX), allow_pickle=False)
    after = identity(source)
    assert before == after, 'Source changed while tokenizing'
    result = dict(status='completed', job_id=os.environ['SLURM_JOB_ID'], source=before,
                  plan_sha256=sha256(args.output / 'plan.json'), domain=domain, split=split,
                  content_tokens=total, input_tokens=input_total, chunks=len(indices),
                  documents=selected_docs, rejected_empty_or_duplicate=rejected,
                  elapsed_seconds=time.monotonic() - began,
                  sha256={name: sha256(directory / name)
                          for name in ('tokens.bin', 'index.npy', 'documents.jsonl')})
    publish(result_path, result)
    return result


def document_hashes(args, split):
    hashes = set()
    for domain in NAMES:
        file = args.output / 'data' / domain / split / 'documents.jsonl'
        if file.exists():
            with file.open() as stream:
                hashes.update(json.loads(line)['normalized_sha256'] for line in stream)
    return hashes


def nonwiki(args, plan):
    sources = dict(books=ROOT / 'books/data.dat', academic=ROOT / 'academic/data.dat',
                   gazzetta_ufficiale=ROOT / 'gazzetta_ufficiale/data.dat',
                   subtitles=ROOT / 'subtitles_subtitles/data.dat')
    expected = plan['books_source_identity']
    assert sha256(sources['books']) == expected['source_sha256']
    heldout = set(plan['books_validation_documents'])
    orders = {}
    for i, (domain, source) in enumerate(sources.items()):
        began = time.monotonic()
        fingerprint = identity(source)
        publish(args.output / f'{domain}-source-identity.json', fingerprint)
        print(f'{domain}: sequential source hash/warm read in {time.monotonic() - began:.1f}s', flush=True)
        ids = candidates(source, 'lmdb')
        orders[domain] = np.random.default_rng(args.seed + i + 1).permutation(ids)
        print(f'{domain}: complete numeric key coverage, {len(ids)} documents', flush=True)
    for split in ('validation', 'train'):
        for domain, source in sources.items():
            ids = orders[domain]
            if domain == 'books':
                ids = np.asarray([i for i in ids if (int(i) in heldout) == (split == 'validation')])
            # Exact normalized-text duplicates can never cross the holdout boundary.
            excluded = document_hashes(args, 'validation') if split == 'train' else set()
            prepare_split(args, plan, domain, split, source, 'lmdb', ids, excluded)
    publish(args.output / 'nonwiki-completed.json', dict(status='completed', job_id=os.environ['SLURM_JOB_ID']))


def wikipedia(args, plan):
    split_dir = args.output / 'wikipedia/split'
    extraction = json.loads((split_dir / 'manifest.json').read_text())
    assert extraction['specification']['seed'] == 27
    for i, split in enumerate(('validation', 'train')):
        path = split_dir / f'{split}.jsonl'
        ids = candidates(path, 'jsonl')
        assert len(ids) == extraction['counts'][split]['documents']
        ids = np.random.default_rng(args.seed + 101 + i).permutation(ids)
        excluded = document_hashes(args, 'validation') if split == 'train' else set()
        prepare_split(args, plan, 'wikipedia', split, path, 'jsonl', ids, excluded)
    publish(args.output / 'wikipedia-completed.json', dict(status='completed', job_id=os.environ['SLURM_JOB_ID']))


def finalize(args, plan):
    manifests, doc_ids, hashes = {}, {}, {}
    for split in ('validation', 'train'):
        manifests[split], doc_ids[split], hashes[split] = {}, set(), set()
        for domain in NAMES:
            directory = args.output / 'data' / domain / split
            result = json.loads((directory / 'result.json').read_text())
            assert result['status'] == 'completed'
            assert result['plan_sha256'] == sha256(args.output / 'plan.json')
            assert result['content_tokens'] == plan[f'{split}_quotas'][domain]
            for name, expected in result['sha256'].items():
                assert sha256(directory / name) == expected, directory / name
            index = np.load(directory / 'index.npy', allow_pickle=False)
            tokens = np.memmap(directory / 'tokens.bin', dtype='<u2', mode='r')
            assert len(index) == result['chunks'] and len(tokens) == result['input_tokens']
            assert np.all((index['length'] >= 3) & (index['length'] <= 1024))
            assert np.array_equal(index['offset'], np.r_[0, np.cumsum(index['length'].astype(np.int64))[:-1]])
            assert np.all(tokens[index['offset'].astype(np.int64)] == 5)
            assert np.all(tokens[(index['offset'] + index['length'] - 1).astype(np.int64)] == 6)
            assert int(index['length'].astype(np.int64).sum()) - 2 * len(index) == result['content_tokens']
            assert int(tokens.max()) < 32768
            with (directory / 'documents.jsonl').open() as stream:
                rows = [json.loads(line) for line in stream]
            assert len(rows) == result['documents']
            assert sum(row['used_content_tokens'] for row in rows) == result['content_tokens']
            for row in rows:
                key = (domain, row['source_id'])
                assert key not in doc_ids[split]
                doc_ids[split].add(key)
                hashes[split].add(row['normalized_sha256'])
            manifests[split][domain] = result
    assert not doc_ids['train'] & doc_ids['validation'], 'A whole document crosses the split'
    assert not hashes['train'] & hashes['validation'], 'An exact normalized duplicate crosses the split'
    assert sha256(args.tokenizer / 'tokenizer.json') == plan['tokenizer_sha256']
    receipt = dict(status='passed', job_id=os.environ['SLURM_JOB_ID'], plan=plan, splits=manifests,
                   content_tokens={s: sum(r['content_tokens'] for r in m.values()) for s, m in manifests.items()},
                   input_tokens={s: sum(r['input_tokens'] for r in m.values()) for s, m in manifests.items()},
                   validation=['Exact quotas', 'Whole-document separation', 'Normalized text duplicate separation',
                               'Token/index/provenance hashes', 'Chunk boundaries and retained tails', 'Tokenizer identity'],
                   limitation='No semantic near-duplicate screening or original-pretraining decontamination')
    publish(args.output / 'data-validation.json', receipt)
    print(json.dumps({k: receipt[k] for k in ('status', 'content_tokens', 'input_tokens')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['nonwiki', 'wikipedia', 'finalize'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, default=Path('tokenizers/gettone-retrieval'))
    parser.add_argument('--books-count', type=Path, default=Path('artifacts/mlm_annealing/books-token-count-20261118'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=271118)
    parser.add_argument('--max-seconds', type=int, default=13200)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Corpus processing requires Slurm')
    assert args.workers <= int(os.environ['SLURM_CPUS_PER_TASK'])
    args.deadline = time.monotonic() + args.max_seconds
    args.output.mkdir(parents=True, exist_ok=True)
    plan = book_plan(args)
    assert plan['seed'] == args.seed and plan['tokenizer_sha256'] == sha256(args.tokenizer / 'tokenizer.json')
    globals()[args.stage](args, plan)


if __name__ == '__main__':
    main()
