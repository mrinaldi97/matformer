"""Offline integration tests using real SQLite and LMDB files, without models."""
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import sqlite3
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from matformer.data.mdat import DatasetSelection, MatformerDataset
from matformer.data.mdat.exceptions import SubMdatMissesStrategy
from matformer.data.mdat.user_interface import build_cli
from matformer.data.matformer_data_module import MatformerDataModule


LENGTHS = {'wiki': [6, 4, 8], 'books': [7, 3, 5], 'web': [2, 9]}


def make_dataset(root):
    """Create a tiny existing-format MDAT with cached, known token counts."""
    dataset = MatformerDataset.create_new(str(root))
    strategy = {
        'strategy_name': 'tiny', 'tokenizer_type': 'hf', 'tokenizer_name': 'unused',
        'splitter_class': 'SmallSplitter', 'splitter_init': {'language': 'english'},
        'modality': 'text', 'chunk_size': 6, 'wants_from_db': ['data'],
        'returns': ['tokens', 'chunks'], 'required_databases': ['tokens', 'chunks'],
        'bos_token_id': 101, 'eos_token_id': 102, 'mask_token_id': 103, 'vocab_size': 1000,
    }
    dataset.register_strategy('tiny', strategy)
    for name, lengths in LENGTHS.items():
        source = root.parent / f'{name}.jsonl'
        source.write_text(''.join(json.dumps({'text': f'Document number {index} in domain {name}.',
                                              'source_id': index}) + '\n'
                                  for index in range(len(lengths))))
        sub = dataset.add_submdat(name)
        sub.convert_to_submdat('jsonl', str(source), progress_bar=False)
        sub.set_strategy('tiny', add=True, map_size=1 << 20)
        chunks_total = 0
        divided = [0] * 4
        for index, length in enumerate(lengths):
            chunks = [(start, min(start + 4, length)) for start in range(0, length, 4)]
            chunks_total += len(chunks)
            for multiplier in range(1, 4):
                divided[multiplier] += math.ceil(len(chunks) / multiplier)
            sub.pretok_db['tokens'].write_batch_with_keys([
                (index, sub.current_strategy.prepare_tokens_for_storage(list(range(1, length + 1))))])
            sub.pretok_db['chunks'].write_batch_with_keys([
                (index, sub.current_strategy.prepare_chunks_for_storage(chunks))])
        sub.add_strategy_end('tiny', {'total_tokens': sum(lengths), 'total_chunks': chunks_total,
                                     'processed_docs': len(lengths), 'precomputed_lengths': divided})
    dataset.set_view('default')
    # A saved split containing two specific documents, not domain prefixes.
    dataset.db.add_view('validation')
    ids = dataset.db.get_dsmap('name')
    for name in LENGTHS:
        dataset.db.link_submdat_view(ids[name], 'validation', name == 'web',
                                    name != 'web', False, int(name != 'web'), 0)
    with (root / 'shuffling' / 'validation.mdat').open('wb') as stream:
        for name, index in [('books', 2), ('wiki', 1)]:
            stream.write(struct.pack('<QQ', ids[name], index))
    dataset.db.update_manifest({'shuffle_struct_format': '<QQ'}, view='validation')
    dataset.close()


def snapshot(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob('*') if path.is_file()}


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'corpus'
        make_dataset(self.root)

    def load(self, **kwargs):
        dataset = MatformerDataset.load_dataset(str(self.root), readonly=True,
            prefetch_buffer=kwargs.pop('prefetch_buffer', 0), **kwargs)
        self.addCleanup(dataset.close)
        return dataset

    def documents(self, dataset):
        return [(item['object']['submdat_name'], item['object']['key']) for item in dataset]

    def test_budget_skips_oversized_documents_and_keeps_whole_documents(self):
        dataset = self.load(strategy='tiny', selection={'domains': ['wiki', 'books'], 'max_tokens': 9})
        self.assertEqual(self.documents(dataset), [('wiki', 0), ('books', 1)])
        self.assertEqual(dataset.total_tokens, 9)
        self.assertEqual(dataset.total_documents, 2)

    def test_domain_caps_and_total_cap(self):
        dataset = self.load(strategy='tiny', selection={'domain_tokens': {'wiki': 8, 'books': 7},
                                                       'max_tokens': 12})
        self.assertEqual(self.documents(dataset), [('wiki', 0), ('books', 1)])
        self.assertEqual(dataset.selection_summary['domains']['books']['tokens'], 3)
        self.assertLessEqual(dataset.total_tokens, 12)

    def test_saved_split_membership_is_preserved(self):
        dataset = self.load(ds_view='validation', strategy='tiny', selection={'max_tokens': 9})
        self.assertEqual(self.documents(dataset), [('books', 2), ('wiki', 1)])
        self.assertEqual(dataset.total_tokens, 9)

    def test_raw_domain_selection_needs_no_tokenizer(self):
        dataset = self.load(selection={'domains': ['books']})
        self.assertEqual(self.documents(dataset), [('books', index) for index in range(3)])
        self.assertIsNone(dataset.total_tokens)

    def test_zero_insufficient_and_excessive_budgets(self):
        for budget, count in [(0, 0), (1, 0), (10000, 8)]:
            with self.subTest(budget=budget):
                with MatformerDataset.load_dataset(str(self.root), readonly=True, strategy='tiny',
                        prefetch_buffer=0, selection={'max_tokens': budget}) as dataset:
                    self.assertEqual(len(dataset), count)
                    self.assertEqual(len(self.documents(dataset)), count)
                    self.assertLessEqual(dataset.total_tokens, budget)

    def test_seed_is_reproducible_and_does_not_change_global_rng(self):
        import random
        state = random.getstate()
        dataset = self.load(strategy='tiny', selection={'max_tokens': 20, 'seed': 31})
        first = self.documents(dataset)
        fingerprint = dataset.selection.fingerprint
        dataset.select({'max_tokens': 20, 'seed': 31})
        self.assertEqual(first, self.documents(dataset))
        self.assertEqual(fingerprint, dataset.selection.fingerprint)
        self.assertEqual(state, random.getstate())

    def test_readonly_selection_does_not_modify_source(self):
        before = snapshot(self.root)
        dataset = self.load(strategy='tiny', selection={'max_tokens': 20})
        self.documents(dataset)
        with self.assertRaises(sqlite3.OperationalError):
            dataset.db.connect().execute('DELETE FROM view')
        directory = dataset.selection.pointer_path.parent
        dataset.close()
        self.assertFalse(directory.exists())
        self.assertEqual(before, snapshot(self.root))

    def test_chunk_lengths_and_repeated_epochs_with_prefetch(self):
        dataset = self.load(strategy='tiny', max_seq_len=12, prefetch_buffer=2,
                            selection={'domains': ['wiki']})
        dataset.set_iteration_modality('chunked_tokens')
        first = list(dataset)
        self.assertEqual(len(first), len(dataset))
        self.assertEqual(len(first), 3)
        self.assertEqual(first, list(dataset))
        self.assertTrue(all(len(item['object']) <= 12 for item in first))

    def test_resume_preserves_partial_document_and_checks_selection(self):
        dataset = self.load(strategy='tiny', selection={'domains': ['wiki']})
        dataset.set_iteration_modality('chunked_tokens')
        iterator = iter(dataset)
        next(iterator)
        state = dataset.get_state_dict()
        remainder = []
        while True:
            try:
                remainder.append(next(iterator))
            except StopIteration:
                break
        dataset.restore_state_dict(state)
        self.assertEqual(list(dataset), remainder)
        self.assertEqual(len(list(dataset)), len(dataset))
        dataset.select({'domains': ['books']})
        with self.assertRaisesRegex(ValueError, 'selection differs'):
            dataset.restore_state_dict(state)

    def test_view_switch_clears_selection_and_stale_domains(self):
        dataset = self.load(strategy='tiny', selection={'domains': ['web'], 'max_tokens': 2})
        self.assertEqual(len(dataset), 1)
        dataset.set_view('validation')
        self.assertIsNone(dataset.selection)
        self.assertEqual(set(dataset.loaded_submdats), {'wiki', 'books'})
        self.assertEqual(self.documents(dataset), [('books', 2), ('wiki', 1)])
        self.assertEqual(dataset.total_tokens, 9)

    def test_distributed_selection_lengths_and_membership(self):
        dataset = self.load(strategy='tiny', selection={'max_tokens': 20})
        expected = set(self.documents(dataset))
        dataset.dist = True
        dataset.world_size = 2
        rank_documents = []
        for rank in range(2):
            dataset.rank_size = rank
            rank_documents.append({(item['object']['submdat_name'], item['object']['key'])
                                   for item in dataset if not item.get('worker_has_finished')})
        self.assertFalse(rank_documents[0] & rank_documents[1])
        self.assertEqual(set.union(*rank_documents), expected)
        dataset.set_iteration_modality('chunked_tokens')
        self.assertEqual(len(dataset), dataset.get_distributed_length_before_training(2))
        for rank in range(2):
            dataset.rank_size = rank
            self.assertEqual(len(list(dataset)), len(dataset))

    def test_datamodule_passes_selection_to_real_dataloader(self):
        module = MatformerDataModule(str(self.root), 'chunked_tokens', 0,
            max_seq_len=6, mdat_strategy='tiny', batch_size=2, distributed=False,
            mdat_selection={'domains': ['wiki'], 'max_tokens': 10}, prefetch_buffer=0)
        self.addCleanup(module.teardown)
        module.setup()
        batches = list(module.train_dataloader())
        self.assertEqual(module.mdat.total_tokens, 10)
        self.assertEqual(len(module), 3)
        self.assertEqual(len(batches), 2)
        self.assertEqual(sum(batch['sequence'].batch_size for batch in batches), 3)
        module.setup()  # Repeated setup must close the previous LMDB environments.
        self.assertEqual(len(list(module.train_dataloader())), 2)

    def test_validation_errors_are_actionable(self):
        for config in [{'max_tokens': -1}, {'max_tokens': 1.5}, {'max_tokens': True},
                       {'domains': 'wiki'}, {'domains': []}, {'seed': '27'},
                       {'domains': ['wiki'], 'domain_tokens': {'web': 2}}]:
            with self.subTest(config=config), self.assertRaises(ValueError):
                DatasetSelection(**config)
        dataset = self.load()
        with self.assertRaisesRegex(ValueError, 'require a pretokenization strategy'):
            dataset.select({'max_tokens': 5})
        with self.assertRaisesRegex(ValueError, 'absent from the source view'):
            dataset.select({'domains': ['unknown']})
        with self.assertRaisesRegex(ValueError, 'positive multiple'):
            dataset.set_strategy('tiny', max_seq_len=5)

    def test_old_imports_and_cli_parser_still_work(self):
        from matformer.data.mdat.mdat import LMDBDataset, DatabaseManager, SmallSplitter
        from matformer.data.mdat.lmdb import LMDBDataset as Storage
        self.assertIs(LMDBDataset, Storage)
        self.assertTrue(callable(DatabaseManager))
        self.assertTrue(callable(SmallSplitter))
        parser = build_cli()
        args = parser.parse_args(['register-strategy', '--path', 'corpus', '--name', 'tiny', '--json', 's.json'])
        self.assertEqual(args.json_path, 's.json')
        args = parser.parse_args(['convert', '--path', 'corpus', '--submdat', 'wiki', '--type', 'jsonl', '--source', 'x.jsonl'])
        self.assertEqual(args._type, 'jsonl')

    def test_existing_full_view_loading_without_selection(self):
        dataset = self.load(strategy='tiny', prefetch_buffer=2)
        self.assertEqual(len(self.documents(dataset)), 8)
        dataset.set_iteration_modality('chunked_tokens')
        self.assertEqual(len(list(dataset)), len(dataset))
        self.assertEqual(len(list(dataset)), 14)

    def test_excluded_domains_do_not_need_the_requested_strategy(self):
        with sqlite3.connect(self.root / 'mdat.db') as connection:
            domain_id = connection.execute("SELECT ds_map_id FROM submdat WHERE name='web'").fetchone()[0]
            connection.execute("DELETE FROM pretok_strategy_submdat WHERE submdat_id=?", (domain_id,))
        dataset = self.load(strategy='tiny', selection={'domains': ['wiki'], 'max_tokens': 10})
        self.assertEqual(dataset.total_tokens, 10)
        with self.assertRaisesRegex(SubMdatMissesStrategy, 'web'):
            dataset.select({'domains': ['web']}, strategy='tiny')

    def test_prefetch_errors_reach_the_caller(self):
        dataset = self.load(strategy='tiny', selection={'domains': ['wiki']}, prefetch_buffer=1)
        with patch.object(dataset, '_load_next_document_inner', side_effect=ValueError('damaged cache')):
            with self.assertRaisesRegex(ValueError, 'damaged cache'):
                list(dataset)

    def test_changed_chunk_size_cannot_resume_old_state(self):
        dataset = self.load(strategy='tiny', selection={'domains': ['wiki']})
        dataset.set_iteration_modality('chunked_tokens')
        next(iter(dataset))
        state = dataset.get_state_dict()
        dataset.set_strategy('tiny', max_seq_len=12)
        with self.assertRaisesRegex(ValueError, 'sequence length'):
            dataset.restore_state_dict(state)

    def test_recurrent_selection_fails_clearly_instead_of_dropping_chunks(self):
        dataset = self.load(strategy='tiny', selection={'domains': ['wiki']}, batch_size=2)
        with self.assertRaisesRegex(ValueError, 'recurrent loader'):
            dataset.set_iteration_modality('chunked_for_recurrence')
        dataset.set_view('default')
        dataset.set_iteration_modality('chunked_for_recurrence')
        with self.assertRaisesRegex(ValueError, 'recurrent loader'):
            dataset.select({'domains': ['wiki']})

    def test_full_view_distributed_counts_do_not_write_readonly_metadata(self):
        before = snapshot(self.root)
        dataset = self.load(strategy='tiny')
        dataset.set_iteration_modality('chunked_tokens')
        expected = dataset.get_distributed_length_before_training(3)
        dataset.dist = True
        dataset.world_size = 3
        for rank in range(3):
            dataset.rank_size = rank
            self.assertEqual(len(list(dataset)), expected)
        dataset.close()
        self.assertEqual(before, snapshot(self.root))

    def test_cli_load_reports_selection_and_keeps_corpus_unchanged(self):
        from matformer.data.mdat.user_interface import cmd_load
        before = snapshot(self.root)
        config = self.root.parent / 'selection.json'
        config.write_text(json.dumps({'domains': ['books'], 'max_tokens': 10}))
        result = cmd_load(str(self.root), readonly=True, strategy='tiny', selection_json=str(config))
        try:
            self.assertEqual(result['selection']['tokens'], 10)
            self.assertEqual(result['documents'], 2)
        finally:
            result['ds_object'].close()
        self.assertEqual(before, snapshot(self.root))

    def test_existing_view_export_unwraps_document_records(self):
        dataset = self.load()
        target = self.root.parent / 'export.jsonl'
        self.assertEqual(dataset.export_view(str(target), view='validation', limiter=1), 1)
        rows = [json.loads(line) for line in target.read_text().splitlines()]
        self.assertEqual(rows[0]['group'], 'books')
        self.assertIn('Document number 2', rows[0]['text'])
        self.assertEqual(dataset.export_view(str(target), limiter=0), 0)
        self.assertEqual(target.read_text(), '')

    def test_legacy_shuffle_and_singleton_permutation(self):
        from matformer.data.mdat.utils import RandomPermutator
        self.assertEqual(RandomPermutator(1)(0), 0)
        with self.assertRaises(Exception):
            RandomPermutator(0)(0)
        with MatformerDataset.load_dataset(str(self.root), prefetch_buffer=0) as dataset:
            dataset.shuffle()
            dataset.set_view('default')
            self.assertEqual(set(self.documents(dataset)),
                             {(name, index) for name, lengths in LENGTHS.items() for index in range(len(lengths))})

    def test_direct_script_entry_point_reads_real_lmdb_data(self):
        script = Path(__file__).resolve().parents[1] / 'matformer/data/mdat/mdat.py'
        config = self.root.parent / 'selection.json'
        config.write_text(json.dumps({'domains': ['wiki'], 'max_tokens': 10}))
        result = subprocess.run([sys.executable, str(script), 'load', '--path', str(self.root),
            '--readonly', '--strategy', 'tiny', '--selection-json', str(config)],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"tokens": 10', result.stdout)


if __name__ == '__main__':
    unittest.main()
