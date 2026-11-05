"""Correctness tests for pretraining boundaries, budgets and complete resume."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import contextlib
import io
import random
import tempfile
import unittest
import numpy as np
import torch
from matformer.data.collator import TokenCollator
from matformer.data.mdat.preprocessing import SmallSplitter
from matformer.data.mdat.selection import DatasetSelection
from matformer.model_config import ModelConfig, LayerConfig
from matformer.models import PL_ModelWrapper
from matformer.transformer_blocks import Autoregressive_Model
from matformer.training.causal_loss import target_mask
from matformer.training.checkpointing import capture_rng, restore_rng, is_snapshot, save_checkpoint, restore_checkpoint
from matformer.training.prepare_training import instantiate_training
from matformer.training.training_loop import training_loop
from matformer.tokenizer_evaluation import evaluate_tokenizer
from test_mdat_selection import make_dataset


def tiny_config(root, save_dir):
    return dict(model_class='Autoregressive_Model',
        model_config=asdict(ModelConfig(hidden_size=8, ffn_factor=2., num_hidden_layers=1,
            num_loops=2, num_attention_heads=2, vocab_size=1000, pad_token_id=0,
            bos_token_id=101, eos_token_id=102, mask_token_id=103,
            tie_word_embeddings=False, rms_norm_eps=1e-6, max_position_embeddings=6,
            bias=False, training_objective='autoregressive', is_causal=True,
            default_layer=LayerConfig(attn_impl='sdpa', positional_encoding='alibi'))),
        training=dict(optimizer='adamw', lr=0.001, lr_scheduling=True,
            scheduler='cosine', warmup_steps=0, schedule_steps=20, max_epochs=1,
            accumulate_grad_batches=2, seed=27, save_every_n_steps=-1),
        data=dict(data_root=str(root), mdat_strategy='tiny', batch_size=2,
                  num_workers=0, prefetch_buffer=2, stage_id='part-0'),
        tokenizer=dict(type='huggingface', pretrained_name='unused', varlen_strategy='padding'),
        save_dir=str(save_dir), wandb_mode='disabled')


def build(config, **kwargs):
    cfg = deepcopy(config)
    return instantiate_training(ModelConfig(**cfg['model_config']), cfg['training'],
        cfg['data'], cfg['tokenizer'], cfg, device='cpu', precision='32-true', **kwargs)


def close(state):
    state['datamodule'].teardown()
    state['logger'].close()


class BatchTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_eos_equal_pad_and_exhausted_rank_preserve_real_targets(self):
        for strategy in ('padding', 'unpadding'):
            batch = TokenCollator(6, 4, strategy)([
                {'object': [5, 7, 6]}, {'object': [0], 'worker_has_finished': True}])
            self.assertFalse(batch['worker_has_finished'])
            seq = batch['sequence']
            self.assertEqual(seq.tensor[target_mask(seq)].tolist(), [7, 6])
        with self.assertRaises(ValueError):
            TokenCollator(0, 4)([[1, 2, 3, 4, 5]])
        with self.assertRaises(ValueError):
            TokenCollator(0, 4)([[]])

    def test_loop_shares_parameters_and_backpropagates_through_every_pass(self):
        cfg = tiny_config('unused', 'unused')['model_config']
        model = PL_ModelWrapper(Autoregressive_Model, ModelConfig(**cfg), None, 'cpu')
        count = sum(p.numel() for p in model.parameters())
        calls = []
        layer = model.model.encoder.blocks.layers[0]
        handle = layer.register_forward_hook(lambda *args: calls.append(1))
        loss = model.training_step(TokenCollator(0, 6)([[101, 7, 102]]))
        loss.backward()
        handle.remove()
        self.assertEqual(len(calls), 2)
        cfg['num_loops'] = 1
        other = PL_ModelWrapper(Autoregressive_Model, ModelConfig(**cfg), None, 'cpu')
        self.assertEqual(count, sum(p.numel() for p in other.parameters()))
        self.assertTrue(all(p.grad is not None for p in layer.parameters()))
        with self.assertRaises(ValueError):
            ModelConfig(num_loops=0)

    def test_sdpa_masks_padding_and_honors_no_alibi(self):
        from matformer.modules.attention_implementations.sdpa import SDPAKernel
        from matformer.utils.matformer_cache import CachedStuff
        kernel = SDPAKernel(1, 2, False, None, ['nope'], CachedStuff())
        sequence = TokenCollator(0, 4)([[5, 6]])['sequence']
        q, k, v = [torch.randn(1, 1, 4, 2) for _ in range(3)]
        expected = torch.nn.functional.scaled_dot_product_attention(q, k[:, :, :2], v[:, :, :2])
        actual = kernel(q=q, k=k, v=v, key_input=sequence)
        torch.testing.assert_close(actual, expected)

    def test_packing_keeps_all_token_ids_including_trailing_whitespace(self):
        splitter = SmallSplitter.__new__(SmallSplitter)
        splitter.chunk_size = 3
        splitter.punkt = SimpleNamespace(span_tokenize=lambda text: [(0, 2)])
        encoding = {'input_ids': [10, 11, 12, 13, 14],
                    'offset_mapping': [(i, i+1) for i in range(5)]}
        result = splitter._process('Ciao ', encoding)
        self.assertEqual(result['tokens'], encoding['input_ids'])
        self.assertTrue(all(0 < end - start <= 3 for start, end in result['chunks']))

    def test_composition_is_explicit_and_rounds_to_total(self):
        selection = DatasetSelection(max_tokens=7, composition={'wiki': '50%', 'books': '50%', 'web': '0%'})
        self.assertEqual(sum(selection.domain_tokens.values()), 7)
        self.assertEqual(selection.domain_tokens['web'], 0)
        for shares in ({'wiki': '70%'}, {'wiki': 100}, {'wiki': '-1%', 'web': '101%'}):
            with self.assertRaises(ValueError):
                DatasetSelection(max_tokens=7, composition=shares)

    def test_rng_and_snapshot_schedule(self):
        state = capture_rng()
        first = (random.random(), np.random.rand(), torch.rand(2))
        restore_rng(state)
        second = (random.random(), np.random.rand(), torch.rand(2))
        self.assertEqual(first[:2], second[:2])
        torch.testing.assert_close(first[2], second[2])
        cfg = dict(checkpoint_log_base=2, snapshot_every_n_steps=10, checkpoint_steps=[7])
        self.assertTrue(all(is_snapshot(step, cfg) for step in (1, 2, 4, 7, 8, 10, 16)))
        self.assertFalse(is_snapshot(9, cfg))

    def test_tokenizer_report_runs_without_network(self):
        class Tokenizer:
            unk_token_id = None
            def encode(self, text, **kwargs): return list(text.encode('utf-8'))
            def decode(self, ids, **kwargs): return bytes(ids).decode('utf-8')
            def __len__(self): return 256
        report = evaluate_tokenizer(Tokenizer(), ['Caffè', '', '👩🏽‍💻'])
        self.assertEqual(report['roundtrip_failures'], 0)
        self.assertEqual(report['utf8_bytes_per_token'], 1)


class ResumeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        with contextlib.redirect_stdout(io.StringIO()):
            make_dataset(self.root / 'data')
        self.cfg = tiny_config(self.root / 'data', self.root / 'checkpoints')

    def tearDown(self):
        self.folder.cleanup()

    def test_complete_resume_matches_next_update_and_rng(self):
        cfg = deepcopy(self.cfg)
        cfg['training']['max_steps'] = 1
        fabric, state = build(cfg)
        training_loop(fabric, state)
        path = save_checkpoint(fabric, state)
        rng = capture_rng()
        state['train_config']['max_steps'] = 2
        # A live dataloader iterator would continue; preserve its cursor when
        # entering a second training_loop for this comparison.
        data_state = state['datamodule'].state_dict()
        state['datamodule'].load_state_dict(data_state)
        training_loop(fabric, state)
        expected = deepcopy(state['raw_model'].state_dict())
        expected_scheduler = deepcopy(state['lr_scheduler'][0].state_dict())
        close(state)
        cfg['training']['max_steps'] = 2
        resumed_fabric, resumed = build(cfg, checkpoint=str(path))
        torch.testing.assert_close(torch.get_rng_state(), rng['torch'])
        training_loop(resumed_fabric, resumed)
        for name, value in expected.items():
            torch.testing.assert_close(value, resumed['raw_model'].state_dict()[name], rtol=0, atol=0)
        self.assertEqual(expected_scheduler, resumed['lr_scheduler'][0].state_dict())
        self.assertEqual(resumed['step'], 2)
        close(resumed)

    def test_exact_target_limit_and_new_stage_keep_optimizer_schedule(self):
        cfg = deepcopy(self.cfg)
        cfg['training']['stage_max_tokens'] = 7
        fabric, state = build(cfg)
        training_loop(fabric, state)
        self.assertEqual((state['tokens_seen'], state['stage_tokens']), (7, 7))
        self.assertTrue(state['stage_complete'])
        path = save_checkpoint(fabric, state)
        step = state['step']
        close(state)
        cfg['data']['stage_id'] = 'part-1'
        fabric, state = build(cfg, checkpoint=str(path), next_data_stage=True)
        self.assertEqual(state['step'], step)
        self.assertEqual(state['epoch'], 0)
        self.assertEqual(state['completed_stages'], ['part-0'])
        training_loop(fabric, state)
        self.assertEqual(state['tokens_seen'], 14)
        self.assertEqual(state['train_config']['total_steps'], 20)
        close(state)

    def test_global_target_limit_is_exact(self):
        cfg = deepcopy(self.cfg)
        cfg['training']['max_tokens'] = 9
        fabric, state = build(cfg)
        training_loop(fabric, state)
        self.assertEqual(state['tokens_seen'], 9)
        self.assertFalse(state['stage_complete'])
        close(state)

    def test_atomic_failure_preserves_checkpoint_and_strict_resume_guards(self):
        fabric, state = build(self.cfg)
        state['train_config']['max_steps'] = 1
        training_loop(fabric, state)
        path = save_checkpoint(fabric, state)
        saved = torch.load(path, weights_only=False)
        changed = {**saved, 'world_size': 2}
        with self.assertRaisesRegex(ValueError, 'same total'):
            restore_checkpoint(fabric, state, changed)
        altered_data = deepcopy(saved['datamodule'])
        altered_data['source_identity'] = [('changed', 1, 1)]
        with self.assertRaisesRegex(ValueError, 'Dataset files changed'):
            state['datamodule'].load_state_dict(altered_data)
        state['step'] = 2
        with patch('matformer.training.checkpointing.torch.save', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(RuntimeError, 'disk full'):
                save_checkpoint(fabric, state)
        self.assertEqual(state['ckpt_manager'].resume()[0], path)
        self.assertEqual(torch.load(path, weights_only=False)['step'], 1)
        close(state)

    def test_real_composition_and_merged_chunk_preserve_content(self):
        from matformer.data.mdat import MatformerDataset
        with MatformerDataset.load_dataset(str(self.root / 'data'), readonly=True,
                strategy='tiny', max_seq_len=12, prefetch_buffer=0,
                selection={'max_tokens': 100, 'composition': {'wiki': '100%', 'books': '0%'}}) as data:
            self.assertEqual(data.selection_summary['domains']['wiki']['tokens'], 18)
            self.assertEqual(data.selection_summary['domains']['books']['tokens'], 0)
            data.set_iteration_modality('chunked_tokens')
            rows = [record['object'] for record in data]
            self.assertEqual([len(row) for row in rows], [8, 6, 10])
            self.assertTrue(all(row[0] == 101 and row[-1] == 102 for row in rows))
            self.assertEqual(sum(len(row)-2 for row in rows), 18)


if __name__ == '__main__':
    unittest.main()
