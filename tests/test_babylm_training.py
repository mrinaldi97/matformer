"""CPU regression tests; no corpus training, network, or accelerator required."""
import contextlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from matformer.tensors_dataclasses import PaddedTensor
from matformer.training.causal_loss import causal_pairs
from matformer.training.training_loop import training_loop
from matformer.training.validation import validate
from matformer.training.prepare_training import count_training_steps
from matformer.utils.matformer_cache import CachedStuff
from matformer.models import PL_ModelWrapper
from matformer.model_config import ModelConfig, LayerConfig
from matformer.transformer_blocks import Autoregressive_Model
from matformer.data.matformer_data_module import MatformerDataModule

class Logger:
    def __init__(self): self.values = {}
    def log(self, name, value, **kwargs): self.values[name] = value
    def advance(self): pass
    def close(self): pass

class Fabric:
    device = torch.device('cpu')
    global_rank = 0
    def autocast(self): return contextlib.nullcontext()
    def no_backward_sync(self, model, enabled): return contextlib.nullcontext()
    def backward(self, loss): loss.backward()
    def clip_gradients(self, model, optimizer, max_norm): pass
    def all_reduce(self, value, reduce_op): return value
    def print(self, *args): pass

class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
    def training_step(self, batch, batch_idx): return self.weight * batch
    def stable_state_dict(self): return self.state_dict()

class Loader:
    def __init__(self, offset=0): self.offset = offset
    def __len__(self): return 3
    def __iter__(self): return iter([1., 3., 9.][self.offset:])

class TrainingTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1)
    def padded(self):
        ids = torch.tensor([[5, 7, 6, 0], [5, 8, 9, 6]])
        return PaddedTensor(tensor=ids, padding_mask=ids == 0)

    def test_padded_and_unpadded_have_identical_pairs(self):
        seq = self.padded()
        logits = torch.arange(2 * 4 * 12).reshape(2, 4, 12).float()
        out = PaddedTensor(tensor=logits, padding_mask=seq.padding_mask)
        a, targets = causal_pairs(out, seq, 0)
        b, other = causal_pairs(out.unpad(), seq.unpad(), 0)
        self.assertEqual(targets.tolist(), [7, 6, 8, 9, 6])
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(targets, other)
        c, third = causal_pairs(out, seq.unpad(), 0)
        torch.testing.assert_close(a, c)
        torch.testing.assert_close(targets, third)

    def test_mask_cache_handles_smaller_final_batch(self):
        cache = CachedStuff()
        for size in (3, 1, 3):
            q = torch.zeros(size, 2, 4, 8)
            mask = cache.get_attention_mask(q, q, 2, causal=True, add_alibi=True)
            self.assertEqual(mask.shape, (size, 2, 4, 4))
            self.assertTrue(torch.isneginf(mask[..., 0, 1]).all())

    def state(self, offset=0, step=0):
        model = Toy()
        return dict(model=model, optimizer=[torch.optim.SGD(model.parameters(), lr=1.)],
                    lr_scheduler=None, dataloader=Loader(offset),
                    train_config={'accumulate_grad_batches': 2, 'max_epochs': 1},
                    epoch=0, step=step, batch_in_epoch=offset, step_ref=[step], logger=Logger())

    def test_final_accumulation_window_is_stepped_and_normalized(self):
        state = self.state()
        training_loop(Fabric(), state)
        self.assertEqual(state['model'].weight.item(), -11.)
        self.assertEqual((state['step'], state['epoch'], state['batch_in_epoch']), (2, 1, 0))
        self.assertIsNone(state['model'].weight.grad)

    def test_max_steps_stops_and_resume_matches_uninterrupted(self):
        state = self.state()
        state['train_config']['max_steps'] = 1
        training_loop(Fabric(), state)
        self.assertEqual(state['model'].weight.item(), -2.)
        self.assertEqual((state['step'], state['epoch'], state['batch_in_epoch']), (1, 0, 2))
        state['dataloader'] = Loader(2)
        state['train_config']['max_steps'] = 2
        training_loop(Fabric(), state)
        self.assertEqual(state['model'].weight.item(), -11.)
        self.assertEqual(state['epoch'], 1)

    def test_completed_epoch_does_not_repeat_on_resume(self):
        state = self.state()
        training_loop(Fabric(), state)
        training_loop(Fabric(), state)
        self.assertEqual(state['model'].weight.item(), -11.)
        self.assertEqual(state['step'], 2)

    def test_steps_include_partial_window_and_do_not_divide_by_nodes(self):
        data = list(range(21))
        self.assertEqual(count_training_steps(data, 4, 4, num_nodes=2, max_epochs=2), (4, 6))
        self.assertEqual(count_training_steps(data, 4, 4, max_steps=1), (1, 6))

    def test_validation_weights_by_targets_and_restores_training(self):
        class Model(torch.nn.Module):
            def validation_step(self, batch): return batch
        model, logger = Model(), Logger()
        batches = [{'loss_sum': torch.tensor(2.), 'tokens': 1},
                   {'loss_sum': torch.tensor(12.), 'tokens': 3}]
        loss = validate(Fabric(), model, batches, logger, 4)
        self.assertEqual(loss, 3.5)
        self.assertTrue(model.training)
        self.assertEqual(logger.values['val/tokens'], 4.)

    def test_checkpoint_without_scheduler_is_atomic_and_records_progress(self):
        model = Toy()
        model.config = {'test': True}
        optimizer = torch.optim.SGD(model.parameters(), lr=1.)
        data = SimpleNamespace(state_dict=lambda: {'cursor': 2})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'test.ckpt'
            PL_ModelWrapper.save_checkpoint(model, [optimizer], None, 7, 1, data,
                                            save_path=path, batch_in_epoch=4)
            checkpoint = torch.load(path, weights_only=False)
            self.assertEqual((checkpoint['step'], checkpoint['epoch'], checkpoint['batch_in_epoch']), (7, 1, 4))
            self.assertIsNone(checkpoint['scheduler'])
            self.assertFalse(path.with_name('test.ckpt.tmp').exists())
            restored = Toy()
            restored.load_state_dict(checkpoint['state_dict'])
            torch.testing.assert_close(restored.weight, model.weight)

    def test_real_tiny_model_short_batches_and_validation(self):
        config = ModelConfig(hidden_size=8, ffn_factor=2., num_hidden_layers=1,
            num_attention_heads=2, vocab_size=16, pad_token_id=0, bos_token_id=5,
            eos_token_id=6, mask_token_id=4, tie_word_embeddings=False,
            rms_norm_eps=1e-6, max_position_embeddings=4, bias=False,
            training_objective='autoregressive', is_causal=True,
            default_layer=LayerConfig(attn_impl='sdpa', positional_encoding='alibi',
                                     normalization_position='post'))
        model = PL_ModelWrapper(Autoregressive_Model, config, None, 'cpu', batch_size=2)
        data = MatformerDataModule('unused', 'chunked_tokens', 0, varlen_strategy='padding', max_seq_len=4)
        for rows in ([[5, 7, 6], [5, 8, 9, 6]], [[5, 7, 6]]):
            batch = data.collate_fn(rows)
            loss = model.training_step(batch)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            model.zero_grad(set_to_none=True)
            result = model.validation_step(batch)
            expected = sum(len(row) - 1 for row in rows)
            self.assertEqual(result['tokens'], expected)
            self.assertTrue(torch.isfinite(result['loss_sum']))
        with patch.object(model, 'forward', side_effect=RuntimeError('test failure')):
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                model.training_step(data.collate_fn([[5, 7, 6]]))

    def test_real_model_muon_checkpoint_resume_matches_next_update(self):
        from matformer.training.optimizers import configure_optimizers
        from matformer.training.lr_scheduler import configure_lr_schedulers
        config = ModelConfig(hidden_size=8, ffn_factor=2., num_hidden_layers=1,
            num_attention_heads=2, vocab_size=16, pad_token_id=0, bos_token_id=5,
            eos_token_id=6, mask_token_id=4, tie_word_embeddings=False,
            rms_norm_eps=1e-6, max_position_embeddings=4, bias=False,
            training_objective='autoregressive', is_causal=True,
            default_layer=LayerConfig(attn_impl='sdpa', positional_encoding='alibi',
                                     normalization_position='post'))
        training = dict(optimizer='muon', lr=.01, lr_scheduling=True,
                        scheduler='cosine', total_steps=3, warmup_steps=0.)
        def build():
            model = PL_ModelWrapper(Autoregressive_Model, config, None, 'cpu', batch_size=2)
            optimizers = configure_optimizers(model, training)
            schedulers = configure_lr_schedulers(training, optimizers)
            return model, optimizers, schedulers
        model, optimizers, schedulers = build()
        data = MatformerDataModule('unused', 'chunked_tokens', 0, varlen_strategy='padding', max_seq_len=4)
        batch = data.collate_fn([[5, 7, 6], [5, 8, 9, 6]])
        def update(model, optimizers, schedulers):
            loss = model.training_step(batch)
            loss.backward()
            for opt in optimizers:
                opt.step()
                opt.zero_grad(set_to_none=True)
            for scheduler in schedulers: scheduler.step()
        update(model, optimizers, schedulers)
        state = SimpleNamespace(state_dict=lambda: {'cursor': 2})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'real.ckpt'
            PL_ModelWrapper.save_checkpoint(model, optimizers, schedulers, 1, 0, state,
                                            save_path=path, batch_in_epoch=1)
            other, other_opts, other_sched = build()
            loaded = other.load_checkpoint(str(path), map_location='cpu',
                optimizer=other_opts, scheduler=other_sched)
            self.assertEqual(loaded['missing'], [])
            self.assertEqual(loaded['unexpected'], [])
            self.assertEqual(loaded['batch_in_epoch'], 1)
            update(model, optimizers, schedulers)
            update(other, other_opts, other_sched)
            for left, right in zip(model.parameters(), other.parameters()):
                torch.testing.assert_close(left, right)
            self.assertEqual(schedulers[0].state_dict(), other_sched[0].state_dict())

if __name__ == '__main__': unittest.main()
