import torch
import json
import random
import torch.utils.data
from pathlib import Path
from typing import Tuple, Optional, List, Union
from tokenizers import models, trainers, decoders, pre_tokenizers, normalizers
from tokenizers import Tokenizer, Regex
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from pytorch_lightning import seed_everything
from matformer.mdat import MatformerDataset
import argparse
import sys

class UnicodeRangeHelper:
    UNICODE_RANGES = {
        'basic_latin': r'\u0000-\u007F',
        'latin': r'\u0000-\u024F',  # basic + supplement + extended
        'ipa_extensions': r'\u0250-\u02AF',
        'spacing_modifiers': r'\u02B0-\u02FF',
        'combining_diacriticals': r'\u0300-\u036F',
        'greek': r'\u0370-\u03FF',
        'greek_extended': r'\u1F00-\u1FFF',
        'cyrillic': r'\u0400-\u04FF',
        'cyrillic_supplement': r'\u0500-\u052F',
        'cyrillic_extended': r'\u2DE0-\u2DFF\uA640-\uA69F',
        'armenian': r'\u0530-\u058F',
        'hebrew': r'\u0590-\u05FF',
        'arabic': r'\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF',
        'general_punctuation': r'\u2000-\u206F',
        'currency_symbols': r'\u20A0-\u20CF',
        'letterlike_symbols': r'\u2100-\u214F',
        'number_forms': r'\u2150-\u218F',
        'arrows': r'\u2190-\u21FF',
        'math_operators': r'\u2200-\u22FF',
        'misc_technical': r'\u2300-\u23FF',
        'box_drawing': r'\u2500-\u257F',
        'block_elements': r'\u2580-\u259F',
        'geometric_shapes': r'\u25A0-\u25FF',
        'misc_symbols': r'\u2600-\u26FF',
        'dingbats': r'\u2700-\u27BF',
        'braille': r'\u2800-\u28FF',
        'cjk_symbols': r'\u3000-\u303F',
        'hiragana': r'\u3040-\u309F',
        'katakana': r'\u30A0-\u30FF',
        'cjk_unified': r'\u4E00-\u9FFF',
        'private_use': r'\uE000-\uF8FF',
        'emoticons': (
            r'\U0001F300-\U0001F5FF'
            r'\U0001F600-\U0001F64F'
            r'\U0001F680-\U0001F6FF'
            r'\U0001F700-\U0001F77F'
            r'\U0001F780-\U0001F7FF'
            r'\U0001F800-\U0001F8FF'
            r'\U0001F900-\U0001F9FF'
            r'\U0001FA00-\U0001FA6F'
            r'\U0001FA70-\U0001FAFF'
            r'\u2600-\u26FF'
            r'\u2700-\u27BF'
        )
    }
    
    @classmethod
    def build_unicode_pattern(cls, ranges: List[Union[str, tuple]], include_whitespace: bool = True, 
                             include_digits: bool = True, include_punctuation: bool = True) -> str:
        pattern_parts = []
        for range_spec in ranges:
            if isinstance(range_spec, str):
                if range_spec in cls.UNICODE_RANGES:
                    pattern_parts.append(cls.UNICODE_RANGES[range_spec])
                else:
                    raise ValueError(f"Unknown Unicode range: {range_spec}")
            elif isinstance(range_spec, tuple) and len(range_spec) == 2:
                start, end = range_spec
                pattern_parts.append(f"\\u{start:04X}-\\u{end:04X}")
            else:
                raise ValueError(f"Invalid range specification: {range_spec}")
        if include_whitespace:
            pattern_parts.append(r'\s')
        if include_digits:
            pattern_parts.append(r'\p{N}')
        if include_punctuation:
            pattern_parts.append(r'\p{P}')
        return ''.join(pattern_parts)

    @classmethod
    def collect_range_characters(cls, ranges: List[str]) -> dict:
        """
        Materialize the characters for the given ranges.
        Returns a dict {range_name: [chars]}.
        """
        result = {}
        for name in ranges:
            if name not in cls.UNICODE_RANGES:
                continue
            spans = cls.UNICODE_RANGES[name]
            chars = []
            # spans may contain multiple segments concatenated
            parts = spans.split("\\u") if "\\u" in spans else spans.split("\\U")
            # More robust: parse pairs like r'\u0370-\u03FF'
            import re
            for match in re.finditer(r'\\u([0-9A-Fa-f]{4})-\\u([0-9A-Fa-f]{4})|\\U([0-9A-Fa-f]{8})-\\U([0-9A-Fa-f]{8})', spans):
                if match.group(1) and match.group(2):
                    start, end = int(match.group(1), 16), int(match.group(2), 16)
                elif match.group(3) and match.group(4):
                    start, end = int(match.group(3), 16), int(match.group(4), 16)
                else:
                    continue
                chars.extend([chr(cp) for cp in range(start, end + 1)])
            result[name] = chars
        return result


def train_tokenizer(
              config: str | Path,
              save_path: str | Path,
              seed: int = 27,
              mdat: str | Path = '',
              mdat_view: Optional[str] = None,
              initialize_vocab: bool = False,
              test_mode: bool = False
              ) -> None:
        seed_everything(seed)
        cfg = json.loads(Path(config).read_text())
        
        unicode_cfg = cfg.get('unicode_filtering', {})
        selected_ranges = unicode_cfg.get('ranges', []) + [tuple(r) for r in unicode_cfg.get('custom_ranges', [])]

        # collect forced characters
        forced_chars = {}
        if initialize_vocab or test_mode:
            # only named ranges are handled
            range_names = [r for r in selected_ranges if isinstance(r, str)]
            forced_chars = UnicodeRangeHelper.collect_range_characters(range_names)

            if test_mode:
                print("=== Expected vocabulary occupation per group ===")
                for k, chars in forced_chars.items():
                    print(f"{k}: {len(chars)} tokens")
                return

        ds = MatformerDataset.load_dataset(Path(mdat))
        if mdat_view:
            ds.set_view('gettone_train_view')
            print(f"View {mdat_view} set.")
        ds.set_iteration_modality(modality='document', with_meta=False, return_raw=True)
        
        tokenizer = Tokenizer(models.Unigram())
        
        normalizer_list = []
        
        if unicode_cfg.get('enabled', False):
            allowed_pattern = UnicodeRangeHelper.build_unicode_pattern(
                selected_ranges,
                include_whitespace=unicode_cfg.get('include_whitespace', True),
                include_digits=unicode_cfg.get('include_digits', True),
                include_punctuation=unicode_cfg.get('include_punctuation', True)
            )
            normalizer_list.append(
                normalizers.Replace(Regex(f'[^{allowed_pattern}]'), '')
            )
        
        if cfg.get('normalization', {}).get('nmt', False):
            normalizer_list.append(normalizers.Nmt())
        
        if cfg.get('normalization', {}).get('nfc', True):
            normalizer_list.append(normalizers.NFC())
        elif cfg.get('normalization', {}).get('nfkc', False):
            normalizer_list.append(normalizers.NFKC())
        
        case_mode = cfg.get('normalization', {}).get('case', 'preserve')
        if case_mode == 'lowercase':
            normalizer_list.append(normalizers.Lowercase())
        elif case_mode == 'uppercase':
            normalizer_list.append(normalizers.Uppercase())
        
        if cfg.get('normalization', {}).get('normalize_whitespace', True):
            normalizer_list.append(normalizers.Replace(Regex(r'\s+'), ' '))
            normalizer_list.append(normalizers.Strip())
        
        if cfg.get('replace_url', False):
            url_pattern = r"(?:https?://)?(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:/\S*)?"
            repl_url = normalizers.Replace(Regex(url_pattern), cfg['url_token'])
            normalizer_list.append(repl_url)
        
        if cfg.get('replace_email', False):
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            repl_email = normalizers.Replace(Regex(email_pattern), cfg.get('email_token', '[EMAIL]'))
            normalizer_list.append(repl_email)
        
        tokenizer.normalizer = normalizers.Sequence(normalizer_list)
        
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement='▁', prepend_scheme='always')
        tokenizer.decoder = decoders.Metaspace(replacement='▁', prepend_scheme='always')
        
        trainer = trainers.UnigramTrainer(
            vocab_size=cfg['vocab_size'],
            show_progress=True,
            special_tokens=cfg['special_tokens'],
            unk_token=cfg['unk_token'],
            initial_alphabet=[c for chars in forced_chars.values() for c in chars] if initialize_vocab else None
        )
        
        tokenizer.train_from_iterator(ds, trainer, length=len(ds))
        
        tt = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token=cfg['unk_token'],
            pad_token=cfg['pad_token'],
            cls_token=cfg['cls_token'],
            sep_token=cfg['sep_token'],
            mask_token=cfg['mask_token']
        )
        
        tt.save_pretrained(save_path)
        

def main():
    parser = argparse.ArgumentParser(description='Train a tokenizer from configuration file')
    parser.add_argument('config', type=str, help='Path to the JSON configuration file')
    parser.add_argument('--save-path', type=str, help='Directory to save the trained tokenizer')
    parser.add_argument('--mdat', type=str, help='Path to the Mdat dataset')
    parser.add_argument('--mdat-view', type=str, help='Mdat view specification')
    parser.add_argument('--seed', type=int, default=27, help='Random seed (default: 27)')
    parser.add_argument('--initialize-vocab', action='store_true', help='Force inclusion of selected unicode ranges in vocab')
    parser.add_argument('--test', action='store_true', help='Print expected vocabulary occupation per group and exit')
    
    args = parser.parse_args()
    
    if not args.save_path:
        args.save_path = "./tokenizer"
    
    train_tokenizer(
        config=args.config,
        save_path=args.save_path,
        mdat=args.mdat,
        mdat_view=args.mdat_view,
        seed=args.seed,
        initialize_vocab=args.initialize_vocab,
        test_mode=args.test
    )

if __name__ == "__main__":
    main()
