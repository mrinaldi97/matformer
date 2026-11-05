"""Public CLI integration: CPU training/resume and completely offline tools."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast
from train_model import main as train_main
from matformer.tokenizer_evaluation import main as tokenizer_main, ITALIAN_SAMPLES
from test_mdat_selection import make_dataset
from test_pretraining_runtime import tiny_config


class CLITests(unittest.TestCase):
    def test_offline_training_resume_and_local_tokenizer(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix='matformer-cli-') as folder:
            root = Path(folder)
            make_dataset(root / 'data')
            cfg = tiny_config(root / 'data', root / 'checkpoints')
            cfg['wandb_mode'] = 'offline'
            cfg['wandb_project'] = 'matformer-local-verification'
            config = root / 'train.json'
            config.write_text(json.dumps(cfg))
            with patch.dict(os.environ, {'WANDB_MODE': 'offline', 'WANDB_SILENT': 'true',
                                        'WANDB_DIR': str(root)}):
                for steps in (1, 2):
                    with patch('sys.argv', ['train_model.py', '--config', str(config),
                            '--device', 'cpu', '--precision', '32-true', '--max-steps', str(steps)]):
                        train_main()
            paths = sorted((root / 'checkpoints').glob('*.ckpt'))
            self.assertEqual(len(paths), 2)
            saved = torch.load(paths[-1], weights_only=False)
            self.assertEqual((saved['step'], saved['format_version']), (2, 2))
            self.assertEqual(len(list(root.glob('**/offline-run-*'))), 2)
            inner = Tokenizer(models.BPE(unk_token='<unk>'))
            inner.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            inner.decoder = decoders.ByteLevel()
            inner.train_from_iterator(ITALIAN_SAMPLES, trainers.BpeTrainer(vocab_size=300,
                show_progress=False, special_tokens=['<pad>', '<bos>', '<eos>', '<unk>'],
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=inner, pad_token='<pad>',
                                               bos_token='<bos>', eos_token='<eos>', unk_token='<unk>')
            tokenizer.save_pretrained(root / 'tokenizer')
            with patch('sys.argv', ['tokenizer_evaluation', '--tokenizer', str(root / 'tokenizer'),
                                   '--output', str(root / 'report.json')]):
                tokenizer_main()
            report = json.loads((root / 'report.json').read_text())
            self.assertEqual(report['roundtrip_failures'], 0)
            self.assertEqual(report['documents'], len(ITALIAN_SAMPLES))


if __name__ == '__main__':
    unittest.main()
