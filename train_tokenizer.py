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
        'latin_supplement': r'\u0080-\u00FF', 
        'latin_extended_a': r'\u0100-\u017F',
        'latin_extended_b': r'\u0180-\u024F',
        'latin': r'\u0000-\u024F',  # basic + extended
        'ipa_extensions': r'\u0250-\u02AF',
        'spacing_modifiers': r'\u02B0-\u02FF',
        'combining_diacriticals': r'\u0300-\u036F',
        'greek': r'\u0370-\u03FF',
        'cyrillic': r'\u0400-\u04FF',
        'cyrillic_supplement': r'\u0500-\u052F',
        'armenian': r'\u0530-\u058F',
        'hebrew': r'\u0590-\u05FF',
        'arabic': r'\u0600-\u06FF',
        'general_punctuation': r'\u2000-\u206F',
        'superscripts_subscripts': r'\u2070-\u209F',
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
        'emoticons': r'\U0001F600-\U0001F64F',
        'misc_symbols_pictographs': r'\U0001F300-\U0001F5FF',
        'transport_symbols': r'\U0001F680-\U0001F6FF'
    }
    
    @classmethod
    def build_unicode_pattern(cls, ranges: List[Union[str, tuple]], include_whitespace: bool = True, 
                             include_digits: bool = True, include_punctuation: bool = True) -> str:
        """
        Build a regex pattern from Unicode range specifications.
        
        Args:
            ranges: List of range names or custom ranges as tuples (start, end)
            include_whitespace: Whether to include whitespace characters
            include_digits: Whether to include digit characters (\p{N})
            include_punctuation: Whether to include punctuation (\p{P})
        
        Returns:
            Regex pattern string for keeping specified characters
        """
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
        
        # Add standard character classes
        if include_whitespace:
            pattern_parts.append(r'\s')
        if include_digits:
            pattern_parts.append(r'\p{N}')
        if include_punctuation:
            pattern_parts.append(r'\p{P}')
            
        return ''.join(pattern_parts)


    
def train_tokenizer(
              config: str | Path,
              save_path: str | Path,
              seed: int = 27,
              mdat: str | Path = '',
              mdat_view: Optional[str] = None
              ) -> None:
        """
        Trains a new sentencepiece-like tokenizer compatible with huggingface.
        
        :param config: path to the json config file for tokenizer training.
        :param save_path: folder in which to store the trained tokenizer.
        :param seed: seed for random number generation.
        :param mdat: path of the Mdat dataset.
        :param mdat_view: Mdat's view
        """
        seed_everything(seed)
        cfg = json.loads(Path(config).read_text())
        
        ds = MatformerDataset.load_dataset(Path(mdat))
        ds.set_iteration_modality(modality='document', with_meta=False, return_raw=True)
        
        tokenizer = Tokenizer(models.Unigram())
        
        normalizer_list = []
        
        if cfg.get('unicode_filtering', {}).get('enabled', False):
            unicode_cfg = cfg['unicode_filtering']
            ranges = unicode_cfg.get('ranges', [])
            custom_ranges = unicode_cfg.get('custom_ranges', [])
            
            converted_ranges = ranges + [tuple(r) for r in custom_ranges]
            
            allowed_pattern = UnicodeRangeHelper.build_unicode_pattern(
                converted_ranges,
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
        # 'preserve' means no case normalization
        
        # Whitespace normalization
        if cfg.get('normalization', {}).get('normalize_whitespace', True):
            normalizer_list.append(normalizers.Replace(Regex(r'\s+'), ' '))
            normalizer_list.append(normalizers.Strip())
        
        # URL replacement
        if cfg.get('replace_url', False):
            url_pattern = r"(?:https?://)?(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:/\S*)?"
            repl_url = normalizers.Replace(Regex(url_pattern), cfg['url_token'])
            normalizer_list.append(repl_url)
        
        # Email replacement
        if cfg.get('replace_email', False):
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            repl_email = normalizers.Replace(Regex(email_pattern), cfg.get('email_token', '[EMAIL]'))
            normalizer_list.append(repl_email)
        
        tokenizer.normalizer = normalizers.Sequence(normalizer_list)
        
        # Set pre-tokenizer and decoder
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement='▁', prepend_scheme='always')
        tokenizer.decoder = decoders.Metaspace(replacement='▁', prepend_scheme='always')
        
        trainer = trainers.UnigramTrainer(
            vocab_size=cfg['vocab_size'],
            show_progress=True,
            special_tokens=cfg['special_tokens'],
            unk_token=cfg['unk_token']
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
    
    args = parser.parse_args()
    
    if not args.save_path:
        config_path = Path(args.config)
        args.save_path ="./tokenizer"
    
    train_tokenizer(
        config=args.config,
        save_path=args.save_path,
        mdat=args.mdat,
        mdat_view=args.mdat_view,
        seed=args.seed
    )

if __name__ == "__main__":
    main()
