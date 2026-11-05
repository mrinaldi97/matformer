"""Create a reproducible document-level train/validation split."""

import argparse
import json
from pathlib import Path

from datasets import DatasetDict, load_from_disk


def split_statistics(dataset):
    return {
        "documents": len(dataset),
        "whitespace_words": sum(len(text.split()) for text in dataset["text"]),
    }


def prepare_split(source, output, validation_fraction=0.02, seed=27):
    source = Path(source)
    output = Path(output)

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    original = load_from_disk(str(source))
    if not isinstance(original, DatasetDict) or "train" not in original:
        raise ValueError("Expected a saved DatasetDict containing 'train'.")

    # Split whole documents, preserving every column and all original records.
    partition = original["train"].train_test_split(
        test_size=validation_fraction,
        seed=seed,
        shuffle=True,
    )
    result = DatasetDict({
        "train": partition["train"],
        "validation": partition["test"],
    })

    manifest = {
        "source": str(source.resolve()),
        "source_split": "train",
        "seed": seed,
        "validation_fraction_by_document": validation_fraction,
        "deduplication": False,
        "splits": {
            name: split_statistics(dataset)
            for name, dataset in result.items()
        },
    }

    provenance = source.parent / "sources.json"
    if provenance.exists():
        manifest["provenance"] = json.loads(provenance.read_text())

    output.mkdir(parents=True)
    result.save_to_disk(str(output / "dataset"))
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nSaved split: {(output / 'dataset').resolve()}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="datasets/babybabellm_ita_gettone/original",
    )
    parser.add_argument(
        "--output",
        default="datasets/babybabellm_ita_gettone/split",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=27)
    args = parser.parse_args()

    prepare_split(
        args.source,
        args.output,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
