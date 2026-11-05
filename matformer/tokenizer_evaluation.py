"""Offline tokenizer diagnostics, independent of model training.

Run ``python -m matformer.tokenizer_evaluation --tokenizer LOCAL_DIRECTORY``.
"""
import argparse
import json
from pathlib import Path


ITALIAN_SAMPLES = [
    "L'intelligenza artificiale è già qui: perché non studiarla?",
    "Un po’ d’acqua, l’Università di Torino e il caffè: €3,50.",
    'à è é ì ò ù À È É Ì Ò Ù — «virgolette» …',
    'Prima riga.\n\nSeconda\triga.  Due spazi. ',
    '2026-11-06: 1.234,56 + 42 = ? https://esempio.it/perché',
    'Unicode: caffè caffe\u0300 👩🏽‍💻 🇮🇹 漢字 العربية',
    '',
]


def evaluate_tokenizer(tokenizer, texts, *, max_examples=10):
    """Measure fertility/byte efficiency and report lossy normalization.

    Round-trip failures are diagnostics, not automatically tokenizer defects:
    some tokenizers deliberately normalize case, accents, or whitespace.
    """
    counts = dict(documents=0, characters=0, utf8_bytes=0, words=0, tokens=0,
                  unknown_tokens=0, roundtrip_failures=0, max_token_id=-1)
    examples = []
    unknown = getattr(tokenizer, 'unk_token_id', None)
    for text in texts:
        if not isinstance(text, str):
            raise ValueError('Every tokenizer test record must contain a string')
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        counts['documents'] += 1
        counts['characters'] += len(text)
        counts['utf8_bytes'] += len(text.encode('utf-8'))
        counts['words'] += len(text.split())
        counts['tokens'] += len(ids)
        counts['unknown_tokens'] += sum(token == unknown for token in ids) if unknown is not None else 0
        counts['max_token_id'] = max(counts['max_token_id'], max(ids, default=-1))
        if text != decoded:
            counts['roundtrip_failures'] += 1
            if len(examples) < max_examples:
                examples.append({'text': text, 'decoded': decoded, 'token_ids': ids})
    tokens = counts['tokens']
    special = {name: getattr(tokenizer, f'{name}_token_id', None)
               for name in ('bos', 'eos', 'pad', 'unk', 'mask')}
    return {**counts, 'vocabulary_size_including_added_tokens': len(tokenizer),
            'tokens_per_word': tokens / counts['words'] if counts['words'] else None,
            'utf8_bytes_per_token': counts['utf8_bytes'] / tokens if tokens else None,
            'unknown_fraction': counts['unknown_tokens'] / tokens if tokens else 0,
            'special_token_ids': special,
            'pad_equals_eos': special['pad'] is not None and special['pad'] == special['eos'],
            'roundtrip_examples': examples}


def read_texts(paths, field='text'):
    for filename in paths:
        path = Path(filename)
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                yield json.loads(line)[field] if path.suffix == '.jsonl' else line.rstrip('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer', required=True, help='Local Hugging Face tokenizer directory')
    parser.add_argument('--input', nargs='*', default=[], help='UTF-8 TXT or JSONL files; default: Italian probes')
    parser.add_argument('--text-field', default='text')
    parser.add_argument('--output', help='Optional JSON report')
    parser.add_argument('--model-config', help='Combined training JSON: verify vocabulary and BOS/EOS/PAD IDs')
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    report = evaluate_tokenizer(tokenizer, read_texts(args.input, args.text_field) if args.input else ITALIAN_SAMPLES)
    if args.model_config:
        config = json.loads(Path(args.model_config).read_text())['model_config']
        errors = []
        if config['vocab_size'] < max(tokenizer.get_vocab().values(), default=-1) + 1:
            errors.append('model vocab_size cannot represent every tokenizer ID')
        for name in ('bos', 'eos', 'pad'):
            if config.get(f'{name}_token_id') != report['special_token_ids'][name]:
                errors.append(f'{name}_token_id differs between tokenizer and model')
        report['model_config_errors'] = errors
    output = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(output + '\n', encoding='utf-8')
    print(output)
    if report.get('model_config_errors'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
