#!/usr/bin/env python3
"""Copy the manual documentation into the Markdown sources.

This is a one-off migration helper. It reads documentation/backup/documentazione.txt (the
author's original text) and writes the base version of each page, so no line is retyped by
hand. Existing pages are left alone unless --force is passed.
"""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

DOC_ROOT = Path(__file__).resolve().parents[1]
BACKUP = DOC_ROOT / "backup" / "documentazione.txt"
TODO = DOC_ROOT / "backup" / "todo_documentazione.txt"
SRC = DOC_ROOT / "src"

SECTIONS = {
    "tasks/index.md": ("MatformerTask", ["matformer/tasks/task.py"], (99, 110)),
    "components/configuration.md": ("Configuration", ["matformer/matformer_configs.py",
                                                      "matformer/dataclasses/matformer_config.py"], (112, 147)),
    "training/learning-signals.md": ("Learning signals", [
        "matformer/training/learning_signals/learning_signals.py",
        "matformer/training/learning_signals_selectors/learning_signals_selectors.py"], (150, 170)),
    "training/wrapper.md": ("MatformerWrapper", ["matformer/matformer_wrapper.py"], (172, 205)),
    "components/tensors.md": ("TensorsDC", ["matformer/dataclasses/tensors_dataclasses.py"], (208, 240)),
    "training/trainer.md": ("MatformerTrainer", ["matformer/training/matformer_trainer.py"], (243, 268)),
    "components/registry.md": ("The Matformer Registry", ["matformer/utils/matformer_registry_implementation.py",
                                                          "matformer/matformer_registry.py"], (274, 419)),
    "components/transformer.md": ("The neural network", [
        "matformer/transformer_blocks/transformer.py",
        "matformer/transformer_blocks/heads.py"], (426, 452)),
    "components/sockets-and-stream.md": ("Sockets, plugins and the stream", [
        "matformer/plugins/socket_mixin.py", "matformer/modules/stream.py"], (454, 459)),
}


def block(path: Path, start: int, stop: int) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    text = "\n".join(lines[start - 1:stop])
    return textwrap.dedent(text.replace("\t", "    ")).strip("\n")


def front_matter(title: str, sources: list[str]) -> str:
    lines = ["---", f"title: {title}", "sources:"]
    lines += [f"  - {source}" for source in sources]
    lines += ["---", ""]
    return "\n".join(lines)


def write(page: str, title: str, sources: list[str], body: str, force: bool) -> None:
    target = SRC / page
    if target.exists() and not force:
        print(f"skip {page} (already exists)")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(front_matter(title, sources) + body + "\n", encoding="utf-8")
    print(f"wrote {page}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    for page, (title, sources, (start, stop)) in SECTIONS.items():
        write(page, title, sources, block(BACKUP, start, stop), arguments.force)

    write(
        "components/matformer-module.md",
        "MatformerModule",
        ["matformer/matformer_module.py"],
        block(TODO, 101, 126),
        arguments.force,
    )
    write(
        "components/neural-networks.md",
        "Neural network components",
        ["matformer/transformer_blocks/__init__.py"],
        block(TODO, 80, 99),
        arguments.force,
    )


if __name__ == "__main__":
    main()
