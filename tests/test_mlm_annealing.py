"""Allocation-only regression checks for the standalone annealing runner."""
import json
import os
from pathlib import Path
import unittest
import tempfile
import hashlib
import numpy as np
import torch

from matformer.data.collator import TokenCollator
from matformer.training.mlm_annealing import pack, microbatches, learning_rate, TokenMixture, DOMAINS, verify_training_data
from scripts.data.prepare_mlm_annealing import INDEX
from scripts.data.count_lmdb_tokens import mixture, publish


class AnnealingTests(unittest.TestCase):
    def test_modified_cache_is_rejected_before_training(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as temporary:
            root = Path(temporary)
            path = root / 'data/books/train/tokens.bin'
            path.parent.mkdir(parents=True)
            path.write_bytes(b'original')
            receipt = dict(status='passed', splits={'train': {'books': {
                'sha256': {'tokens.bin': hashlib.sha256(b'original').hexdigest()}}}})
            verify_training_data(root, receipt)
            path.write_bytes(b'changed!')
            with self.assertRaisesRegex(ValueError, 'Prepared data changed'):
                verify_training_data(root, receipt)
            with self.assertRaisesRegex(ValueError, 'real corpus'):
                verify_training_data(root, {'status': 'passed', 'fixture': True})

    def test_direct_unpadding_matches_existing_collator(self):
        rows = [[5, 13, 87, 6], [5, 23, 6], [5, 90, 21, 2, 4, 6]]
        expected = TokenCollator(0, 1024, 'unpadding')(rows)['sequence']
        actual = pack(rows, 'cpu')
        for key in ('tensor', 'indices', 'cu_seqlens'):
            torch.testing.assert_close(getattr(actual, key), getattr(expected, key), rtol=0, atol=0)
        self.assertEqual(actual.max_seq_len, expected.max_seq_len)
        torch.testing.assert_close(actual.pad().tensor, expected.pad().tensor)

    def test_token_budget_preserves_all_sequences_and_tail(self):
        lengths = np.array([1024, 7, 1024, 213, 512, 1024, 3])
        order = np.array([3, 2, 0, 6, 1, 5, 4])
        batches = microbatches(order, lengths, 2048)
        np.testing.assert_array_equal(np.concatenate(batches), order)
        self.assertTrue(all(lengths[b].sum() <= 2048 for b in batches))
        self.assertEqual(sum(int(lengths[b].sum()) for b in batches), int(lengths.sum()))

    def test_cosine_has_warmup_peak_and_exact_zero_endpoint(self):
        rates = [learning_rate(i, 1000) for i in range(1, 1001)]
        self.assertEqual(rates[9], 1e-4)
        self.assertEqual(rates[-1], 0)
        self.assertTrue(all(a <= b for a, b in zip(rates[:9], rates[1:10])))
        self.assertTrue(all(a >= b for a, b in zip(rates[9:-1], rates[10:])))

    def test_quotas_preserve_book_count_and_total(self):
        for books in (1, 7, 2077917, 413505418):
            quotas = mixture(books)
            self.assertEqual(quotas['books'], books)
            self.assertEqual(sum(quotas.values()), 5 * books)
            for domain, numerator in zip(DOMAINS, [6, 6, 4, 3, 1]):
                self.assertLessEqual(abs(quotas[domain] - books * numerator / 4), 1)


def fixture(root):
    """Small replay fixture from the measured book sample; never a production corpus."""
    root.mkdir(parents=True, exist_ok=False)
    sample = np.load('artifacts/mlm_annealing/throughput-20261117/books-full1024.npy')
    totals = {'train': 0, 'validation': 0}
    content = {'train': 0, 'validation': 0}
    for split, count in [('train', 1600), ('validation', 24)]:
        for d, domain in enumerate(DOMAINS):
            directory = root / 'data' / domain / split
            directory.mkdir(parents=True)
            indices, offset = [], 0
            with (directory / 'tokens.bin').open('wb') as output:
                for i in range(count):
                    row = sample[(d * 317 + i) % len(sample)].astype('<u2')
                    # Exercise unpadded document tails and partial microbatches.
                    if i % 11 == 0:
                        row = np.r_[row[:37], np.uint16(6)]
                    output.write(row.tobytes())
                    indices.append((offset, len(row), i))
                    offset += len(row)
                    content[split] += len(row) - 2
            np.save(directory / 'index.npy', np.array(indices, dtype=INDEX), allow_pickle=False)
            totals[split] += offset
    publish(root / 'data-validation.json', dict(status='passed', fixture=True,
            input_tokens=totals, content_tokens=content,
            limitation='Replayed book throughput sample with artificial tails; engineering check only'))
    train = TokenMixture(root, 'train')
    assert len(train) == 8000 and sum(train.lengths) == totals['train']
    for domain, start in enumerate(train.starts):
        row = train.rows([int(start)])[0]
        assert row[0] == 5 and row[-1] == 6 and len(row) == 38


if __name__ == '__main__':
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run these checks in Slurm')
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AnnealingTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    fixture(Path(os.environ['MLM_CHECK_ROOT']) / 'fixture')
    publish(Path(os.environ['MLM_CHECK_ROOT']) / 'cpu-validation.json',
            dict(status='passed', tests=result.testsRun, job_id=os.environ['SLURM_JOB_ID']))
