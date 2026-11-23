"""Count complete LMDB documents with a pinned tokenizer, within a Slurm job.

Read-only sources; no truncation, padding, text filtering, or token-cache writes.
Per-document counts and hashes are retained for audit and bounded resumption.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import signal
import time
import zlib

import lmdb
from tokenizers import Tokenizer
import zstandard

ENV = None
TOKENIZER = None
STOP = False


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def publish(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def open_source(path):
    return lmdb.open(str(path), subdir=False, readonly=True, lock=False,
                     readahead=False, create=False, max_readers=1)


def read_text(env, number):
    with env.begin(write=False) as txn:
        payload = txn.get(str(number).encode())
    if payload is None:
        raise ValueError(f'Missing document {number}')
    if payload.startswith(b'\x28\xb5\x2f\xfd'):
        payload = zstandard.ZstdDecompressor().decompress(payload, max_output_size=512 << 20)
    elif len(payload) >= 2 and payload[0] & 15 == 8 and int.from_bytes(payload[:2], 'big') % 31 == 0:
        payload = zlib.decompress(payload)
    text = payload.decode('utf-8', errors='strict')
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = text
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ('text', 'content', 'document'):
            if isinstance(value.get(key), str):
                return value[key]
    raise ValueError(f'Unsupported record type/fields at document {number}')


def worker_init(source, tokenizer):
    global ENV, TOKENIZER
    ENV = open_source(source)
    TOKENIZER = Tokenizer.from_file(str(tokenizer))
    TOKENIZER.no_padding()
    TOKENIZER.no_truncation()


def count_one(number):
    text = read_text(ENV, number)
    encoding = TOKENIZER.encode(text, add_special_tokens=False)
    count = len(encoding)
    return dict(document_id=number, content_tokens=count, characters=len(text),
                text_utf8_bytes=len(text.encode('utf-8')),
                text_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest(),
                fixed_1022_content_chunks=(count + 1021) // 1022,
                empty_text=not text.strip())


def mixture(total):
    names = ['wikipedia', 'academic', 'books', 'gazzetta_ufficiale', 'subtitles']
    numerators = [6, 6, 4, 3, 1]
    quotas = [total * n // 4 for n in numerators]
    remaining = total * 5 - sum(quotas)
    ranked = sorted(range(5), key=lambda i: -(total * numerators[i] % 4))
    for i in ranked[:remaining]:
        quotas[i] += 1
    assert quotas[2] == total and sum(quotas) == total * 5
    return dict(zip(names, quotas))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('/leonardo_work/IscrC_ALBERTIN/books/data.dat'))
    parser.add_argument('--tokenizer', type=Path, default=Path('tokenizers/gettone-retrieval'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-seconds', type=int, default=13200)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Counting requires a Slurm allocation')
    if args.workers < 1 or args.workers > int(os.environ['SLURM_CPUS_PER_TASK']):
        raise ValueError('Worker count must fit the CPU allocation')
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    token_file = args.tokenizer / 'tokenizer.json'
    before = args.source.stat()
    identity = dict(source=str(args.source.resolve()), source_bytes=before.st_size,
                    source_mtime_ns=before.st_mtime_ns, source_sha256=sha256(args.source),
                    tokenizer_sha256={p.name: sha256(p) for p in sorted(args.tokenizer.glob('*.json'))},
                    add_special_tokens=False, truncation=False, padding=False)
    identity_file = args.output / 'identity.json'
    if identity_file.exists():
        if json.loads(identity_file.read_text()) != identity:
            raise ValueError('Source/tokenizer changed; use a new output directory')
    else:
        publish(identity_file, identity)

    env = open_source(args.source)
    try:
        with env.begin(write=False) as txn:
            declared = int(txn.get(b'__len__'))
            keys = [key for key in txn.cursor().iternext(values=False)]
        numeric = sorted(int(k) for k in keys if k.isdigit())
        assert numeric == list(range(declared)), 'Document keys do not match declared coverage'
        metadata_keys = [k.decode('utf-8') for k in keys if not k.isdigit()]
        # Validate the lightweight backend against the same HF interface used by
        # the throughput sample, with complete documents and Unicode edge cases.
        from transformers import AutoTokenizer
        reference = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        backend = Tokenizer.from_file(str(token_file))
        backend.no_padding()
        backend.no_truncation()
        assert backend.get_vocab_size() == len(reference) == 32768
        checks = ['', 'È un testo italiano: perché?\nCittà, università. [P] [T]Titolo[/T]',
                  read_text(env, 0), read_text(env, declared - 1)]
        for text in checks:
            assert backend.encode(text, add_special_tokens=False).ids == reference.encode(text, add_special_tokens=False, truncation=False)
        del checks, reference, backend
    finally:
        env.close()
    records_file = args.output / 'document-counts.jsonl'
    records = {}
    if records_file.exists():
        for line in records_file.read_text().splitlines():
            row = json.loads(line)
            number = row['document_id']
            assert number not in records and 0 <= number < declared
            records[number] = row
    pending_ids = iter(n for n in range(declared) if n not in records)
    total_tokens = sum(x['content_tokens'] for x in records.values())

    def stopped(signum, frame):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGUSR1, stopped)
    print(json.dumps(dict(stage='counting', documents=declared, resumed_documents=len(records),
                          workers=args.workers, full_document_reference_checks='passed')), flush=True)
    last_progress = 0
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=worker_init, initargs=(args.source, token_file)) as pool:
        active = {}
        with records_file.open('a') as stream:
            while True:
                bounded_stop = STOP or time.monotonic() - started >= args.max_seconds
                while not bounded_stop and len(active) < args.workers * 2:
                    number = next(pending_ids, None)
                    if number is None:
                        break
                    active[pool.submit(count_one, number)] = number
                if not active:
                    break
                done, _ = wait(active, timeout=5, return_when=FIRST_COMPLETED)
                for future in done:
                    number = active.pop(future)
                    row = future.result()
                    assert row['document_id'] == number and number not in records
                    records[number] = row
                    total_tokens += row['content_tokens']
                    stream.write(json.dumps(row) + '\n')
                elapsed = time.monotonic() - started
                if elapsed - last_progress >= 30 or len(records) == declared:
                    stream.flush()
                    os.fsync(stream.fileno())
                    progress = dict(status='counting', job_id=os.environ['SLURM_JOB_ID'],
                                    documents_counted=len(records), documents_total=declared,
                                    content_tokens=total_tokens, elapsed_seconds=elapsed)
                    publish(args.output / 'progress.json', progress)
                    print(json.dumps(progress), flush=True)
                    last_progress = elapsed
            stream.flush()
            os.fsync(stream.fileno())
    after = args.source.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    assert sha256(args.source) == identity['source_sha256'], 'Source changed during count'
    assert {p.name: sha256(p) for p in sorted(args.tokenizer.glob('*.json'))} == identity['tokenizer_sha256']
    if len(records) != declared:
        publish(args.output / 'partial.json', dict(status='bounded_partial', documents=len(records),
                                                   content_tokens=total_tokens, job_id=os.environ['SLURM_JOB_ID']))
        raise SystemExit(75)
    by_hash = {}
    duplicate_tokens = duplicate_documents = 0
    for number in sorted(records):
        row = records[number]
        if row['text_sha256'] in by_hash:
            assert row['content_tokens'] == by_hash[row['text_sha256']]['content_tokens']
            duplicate_documents += 1
            duplicate_tokens += row['content_tokens']
        else:
            by_hash[row['text_sha256']] = row
    chunks = sum(x['fixed_1022_content_chunks'] for x in records.values())
    report = dict(status='completed', job_id=os.environ['SLURM_JOB_ID'],
                  identity=identity, documents=declared, metadata_keys=metadata_keys,
                  full_document_reference_checks='passed', full_coverage_verified=True,
                  content_tokens=total_tokens, exact_duplicate_documents=duplicate_documents,
                  exact_duplicate_content_tokens=duplicate_tokens,
                  exact_unique_text_content_tokens=total_tokens - duplicate_tokens,
                  empty_text_documents=sum(x['empty_text'] for x in records.values()),
                  text_characters=sum(x['characters'] for x in records.values()),
                  text_utf8_bytes=sum(x['text_utf8_bytes'] for x in records.values()),
                  longest_document_tokens=max(x['content_tokens'] for x in records.values()),
                  fixed_1022_content_chunks=chunks,
                  hypothetical_input_tokens_with_bos_eos_per_fixed_chunk=total_tokens + 2 * chunks,
                  mixture_using_all_raw_books=mixture(total_tokens),
                  mixture_total_content_tokens=5 * total_tokens,
                  elapsed_seconds=time.monotonic() - started,
                  workers=args.workers, document_counts_sha256=sha256(records_file),
                  limitations=['Raw full-corpus count before deduplication, quality/language filtering or validation holdout.',
                               'Content tokens exclude injected BOS/EOS and padding; tokenizer normalization is preserved.',
                               'The BOS/EOS estimate assumes fixed 1022-content-token chunks; sentence-aware packing can differ.',
                               'Exact duplicate totals use decoded text hashes; near-duplicates were not assessed.',
                               'Other-domain mixture quotas are targets, not measured available corpus sizes.'])
    publish(args.output / 'result.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
