import json
import argparse
from pathlib import Path
from typing import Optional, Union, List
import torch
from pytorch_lightning import seed_everything
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, decoders, Regex
from transformers import PreTrainedTokenizerFast
from matformer.mdat import MatformerDataset
import sentencepiece as spm
import re
from tqdm import tqdm


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
            parts = spans.split("\\u") if "\\u" in spans else spans.split("\\U")
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
def train_hf_tokenizer(cfg: dict, dataset: MatformerDataset, save_path: Path, initialize_vocab: bool):
    tokenizer_type = cfg['tokenizer_type'].lower()
    model = models.BPE() if tokenizer_type == "bpe" else models.Unigram()

    tokenizer = Tokenizer(model)
    normalizer_list = []

    unicode_cfg = cfg.get('unicode_filtering', {})
    if unicode_cfg.get('enabled', False):
        pattern = UnicodeRangeHelper.build_unicode_pattern(
            unicode_cfg.get('ranges', []),
            unicode_cfg.get('include_whitespace', True),
            unicode_cfg.get('include_digits', True),
            unicode_cfg.get('include_punctuation', True)
        )
        normalizer_list.append(normalizers.Replace(Regex(f'[^{pattern}]'), ''))
    if cfg.get('normalization', {}).get('nfc', True):
        normalizer_list.append(normalizers.NFC())

    tokenizer.normalizer = normalizers.Sequence(normalizer_list)

    specials = cfg.get("special_extra_tokens", [])
    if specials:
        escaped = [re.escape(t) for t in specials]
        pattern = "(" + "|".join(escaped) + ")"
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(pattern), behavior="isolated"),
            pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
        ])
    else:
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement='▁', prepend_scheme='always')

    tokenizer.decoder = decoders.Metaspace(replacement='▁', prepend_scheme='always')

    initial_alphabet = None
    if initialize_vocab:
        range_names = [r for r in unicode_cfg.get('ranges', []) if isinstance(r, str)]
        forced_chars = UnicodeRangeHelper.collect_range_characters(range_names)
        initial_alphabet = [c for chars in forced_chars.values() for c in chars]

    trainer_cls = trainers.BpeTrainer if tokenizer_type == "bpe" else trainers.UnigramTrainer
    trainer = trainer_cls(
        vocab_size=cfg['vocab_size'],
        show_progress=True,
        special_tokens=cfg['special_tokens'] + specials,
        unk_token=cfg['unk_token'],
        initial_alphabet=initial_alphabet
    )

    tokenizer.train_from_iterator(dataset, trainer, length=len(dataset))
    tokenizer.save(str(save_path / "tokenizer.json"))

    fast_tok = PreTrainedTokenizerFast(
        tokenizer_file=str(save_path / "tokenizer.json"),
        **{k: cfg[k] for k in ['unk_token', 'pad_token', 'cls_token', 'sep_token', 'mask_token'] if k in cfg}
    )
    fast_tok.save_pretrained(save_path)


def train_sentencepiece(cfg: dict, dataset: MatformerDataset, save_path: Path):
    tokenizer_type = cfg['tokenizer_type'].lower()
    model_type = 'bpe' if tokenizer_type == 'bpe' else 'unigram'
    input_file = save_path / "sp_input.txt"

    with open(input_file, "w", encoding="utf-8") as f:
        for doc in dataset:
            text = doc if isinstance(doc, str) else str(doc)
            f.write(text.replace("\n", " ") + "\n")

    spm.SentencePieceTrainer.Train(
        sentence_iterator=(
            doc if isinstance(doc, str) else str(doc)
            for doc in tqdm(dataset, desc="Training SentencePiece", total=len(dataset))
        ),
        model_prefix=str(save_path / "spm"),
        vocab_size=cfg["vocab_size"],
        model_type=model_type,
        character_coverage=cfg.get("character_coverage", 0.9995),
        user_defined_symbols=cfg.get("special_extra_tokens", []),
        unk_id=0,
        pad_id=1,
        bos_id=2,
        eos_id=3,
        train_extremely_large_corpus=True,
        input_sentence_size=cfg.get("input_sentence_size", 10000000),
        shuffle_input_sentence=True,
        verbose=True
    )

    fast_tok = PreTrainedTokenizerFast(tokenizer_file=str(save_path / "spm.model"))
    fast_tok.save_pretrained(save_path)


def train_tokenizer(config: Union[str, Path], save_path: Union[str, Path], seed: int = 27,
                    mdat: Union[str, Path] = '', mdat_view: Optional[str] = None, initialize_vocab: bool = False):
    seed_everything(seed)
    cfg = json.loads(Path(config).read_text())
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    dataset = MatformerDataset.load_dataset(Path(mdat))
    if mdat_view:
        dataset.set_view(mdat_view)
    dataset.set_iteration_modality(modality='document', with_meta=False, return_raw=True)

    backend = cfg.get('tokenizer_backend', 'huggingface').lower()
    if backend == 'huggingface':
        train_hf_tokenizer(cfg, dataset, save_path, initialize_vocab)
    elif backend == 'sentencepiece':
        train_sentencepiece(cfg, dataset, save_path)
    else:
        raise ValueError(f"Unknown tokenizer backend: {backend}")


def main():
    p = argparse.ArgumentParser(description='Train tokenizer (HF or SentencePiece)')
    p.add_argument('config', type=str)
    p.add_argument('--save-path', type=str, default='./tokenizer')
    p.add_argument('--mdat', type=str, required=True)
    p.add_argument('--mdat-view', type=str)
    p.add_argument('--seed', type=int, default=27)
    p.add_argument('--initialize-vocab', action='store_true', help='Force inclusion of Unicode ranges in vocab')
    args = p.parse_args()
    train_tokenizer(args.config, args.save_path, args.seed, args.mdat, args.mdat_view, args.initialize_vocab)


if __name__ == "__main__":
    main()
