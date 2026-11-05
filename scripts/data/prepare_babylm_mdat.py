"""Convert saved HF splits into separate, verified MDAT caches."""
import argparse
import json
import math
import tempfile
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer

from matformer.data.mdat import MatformerDataset


def make_strategy(tokenizer_path, chunk_size):
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=True
    )
    return {
        "strategy_name": f"gettone_{chunk_size}",
        "tokenizer_type": "huggingface",
        "tokenizer_name": str(tokenizer_path),
        "tokenizer_args": {"local_files_only": True},
        "splitter_class": "SmallSplitter",
        "splitter_init": {"language": "italian"},
        "modality": "text",
        "chunk_size": chunk_size,
        "wants_from_db": ["data"],
        "wants_raw": True,
        "returns": ["tokens", "chunks"],
        "required_databases": ["tokens", "chunks"],
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "mask_token_id": tokenizer.mask_token_id,
        "vocab_size": len(tokenizer),
    }


def build_split(rows, destination, strategy, batch_size):
    name = strategy["strategy_name"]
    input_documents = len(rows)
    rows = rows.filter(
        lambda row: len(row["text"].encode("utf-8")) > 10,
        load_from_cache_file=False,
        desc="Exclude texts of 10 bytes or fewer",
    )
    skipped_short_documents = input_documents - len(rows)
    print(f"{destination.name}: skipped {skipped_short_documents} short documents")
    if len(rows) == 0:
        raise ValueError("No documents remain after filtering")
    raw_bytes = 0
    words = 0

    with tempfile.TemporaryDirectory(prefix="babylm_import_") as temporary:
        source = Path(temporary) / "documents.jsonl"
        with source.open("w", encoding="utf-8") as stream:
            for index, row in enumerate(rows):
                text = row["text"]
                if not isinstance(text, str):
                    raise ValueError(f"Document {index}: text is not a string")
                size = len(text.encode("utf-8"))
                if size <= 10:
                    raise ValueError(
                        f"Document {index}: only {size} bytes. "
                        "The current MDAT importer would discard it; "
                        "stopping instead of silently changing the split."
                    )
                raw_bytes += size
                words += len(text.split())
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

        dataset = MatformerDataset.create_new(str(destination))
        try:
            dataset.register_strategy(name, strategy)
            sub = dataset.add_submdat("corpus")
            imported = sub.convert_to_submdat("jsonl", str(source))
            if imported["document_number"] != len(rows):
                raise RuntimeError(f"Incomplete import: {imported}")

            # Refresh document counts and reopen the importer's closed handles.
            dataset.set_view("default")
            sub = dataset.loaded_submdats["corpus"]
            multiplier = max(4, math.ceil((16 << 20) / raw_bytes))
            sub.pretokenize_submdat(
                name,
                parallel=False,
                batch_size=batch_size,
                chunking_strict_checks=True,
                disk_size_multiplier=multiplier,
            )
        finally:
            dataset.close()

    # Reopen the cache as training will: read-only, from disk.
    cached = MatformerDataset.load_dataset(
        str(destination),
        readonly=True,
        prefetch_buffer=0,
        strategy=name,
        max_seq_len=strategy["chunk_size"],
    )
    try:
        if cached.total_documents != len(rows):
            raise RuntimeError("Cached document count differs from input")

        cached.set_iteration_modality("tokens")
        documents = 0
        for item in cached:
            if item.get("worker_has_finished"):
                break
            if len(item["object"]) == 0:
                raise RuntimeError("A document produced no tokens")
            documents += 1
        if documents != len(rows):
            raise RuntimeError("Token-cache document count differs from input")

        cached.set_iteration_modality("chunked_tokens")
        expected_chunks = len(cached)
        chunks = 0
        maximum_length = 0
        for item in cached:
            if item.get("worker_has_finished"):
                break
            tokens = item["object"]
            if not 3 <= len(tokens) <= strategy["chunk_size"]:
                raise RuntimeError(f"Invalid chunk length: {len(tokens)}")
            if tokens[0] != strategy["bos_token_id"]:
                raise RuntimeError("Missing BOS token")
            if tokens[-1] != strategy["eos_token_id"]:
                raise RuntimeError("Missing EOS token")
            if any(t < 0 or t >= strategy["vocab_size"] for t in tokens):
                raise RuntimeError("Token ID outside vocabulary")
            chunks += 1
            maximum_length = max(maximum_length, len(tokens))

        if chunks == 0 or chunks != expected_chunks:
            raise RuntimeError(
                f"Chunk count mismatch: read {chunks}, expected {expected_chunks}"
            )
    finally:
        cached.close()

    return {
        "input_documents": input_documents,
        "skipped_short_documents": skipped_short_documents,
        "documents": documents,
        "whitespace_words": words,
        "chunks": chunks,
        "maximum_sequence_length": maximum_length,
    }


def main():
    base = Path("datasets/babybabellm_ita_gettone")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=base / "split/dataset")
    parser.add_argument("--tokenizer", type=Path, default=base / "tokenizer")
    parser.add_argument("--output", type=Path, default=base / "mdat")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.chunk_size < 3 or args.batch_size < 1:
        parser.error("chunk-size must be >= 3 and batch-size >= 1")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}; choose another path")

    splits = load_from_disk(str(args.source))
    strategy = make_strategy(args.tokenizer, args.chunk_size)
    for split in ("train", "validation"):
        if split not in splits or len(splits[split]) == 0:
            raise ValueError(f"Missing or empty split: {split}")

    args.output.mkdir(parents=True)
    report = {
        "source": str(args.source),
        "strategy": strategy,
        "view": "default",
        "limit_per_split": args.limit,
        "splits": {},
    }
    for split in ("train", "validation"):
        rows = splits[split]
        if args.limit is not None:
            rows = rows.select(range(min(args.limit, len(rows))))
        report["splits"][split] = build_split(
            rows, args.output / split, strategy, args.batch_size
        )

    source_manifest = args.source.parent / "manifest.json"
    if source_manifest.exists():
        report["source_manifest"] = json.loads(source_manifest.read_text())

    (args.output / "manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["splits"], indent=2))
    print(f"Verified MDAT caches saved to: {args.output}")


if __name__ == "__main__":
    main()
