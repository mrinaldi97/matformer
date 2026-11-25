#!/usr/bin/env python3
"""Qwen3-style model: HuggingFace Trainer vs Matformer, 4 GPUs.

The script builds every config from scratch (no repository/legacy config is read), tokenizes a
real corpus once, and runs the two trainers with the same architecture, tokenizer, sequence
length and batch size. It reports wall time, steps/second and the per-rank CUDA high-water mark.

    python scripts/training/benchmark_hf_matformer.py --steps 200 --batch-size 48
    tail -f /tmp/benchmark_hf_matformer.log       # live progress while it runs

Both trainers are launched with torchrun on 4 GPUs, so the subprocess lifecycle is the same.
The two runs are sequential, so they see the same free GPUs.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import socket
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
TORCHRUN = shutil.which("torchrun") or str(Path(PYTHON).with_name("torchrun"))

HF_SCRIPT = r'''
import json, os, time
import torch
from torch.utils.data import Dataset
from transformers import Qwen3Config, Qwen3ForCausalLM, Trainer, TrainingArguments

import json
length = int(os.environ["BENCH_SEQ"])
chunks = [json.loads(line)["input_ids"] for line in open(os.environ["BENCH_TOKENS"], encoding="utf-8")]

class Chunks(Dataset):
    def __len__(self): return len(chunks)
    def __getitem__(self, i):
        row = torch.tensor(chunks[i], dtype=torch.long)
        return {"input_ids": row, "labels": row}

config = Qwen3Config(
    vocab_size=int(os.environ["BENCH_VOCAB"]), hidden_size=int(os.environ["BENCH_HIDDEN"]),
    intermediate_size=int(os.environ["BENCH_INTERMEDIATE"]),
    num_hidden_layers=int(os.environ["BENCH_LAYERS"]),
    num_attention_heads=int(os.environ["BENCH_HEADS"]),
    num_key_value_heads=int(os.environ["BENCH_KV_HEADS"]),
    head_dim=int(os.environ["BENCH_HIDDEN"]) // int(os.environ["BENCH_HEADS"]),
    max_position_embeddings=length, tie_word_embeddings=True, attention_bias=False,
    rms_norm_eps=1e-6, hidden_act="silu", pad_token_id=int(os.environ["BENCH_PAD"]),
)
model = Qwen3ForCausalLM(config)
args = TrainingArguments(
    output_dir=os.environ["BENCH_OUT"], per_device_train_batch_size=int(os.environ["BENCH_BATCH"]),
    max_steps=int(os.environ["BENCH_STEPS"]), learning_rate=1e-4, bf16=True,
    logging_steps=10, save_strategy="no", report_to=[], dataloader_num_workers=0,
    remove_unused_columns=False, seed=27, ddp_find_unused_parameters=False,
)
started = time.perf_counter()
Trainer(model=model, args=args, train_dataset=Chunks()).train()
seconds = time.perf_counter() - started
rank = int(os.environ.get("RANK", "0"))
payload = {"rank": rank, "seconds": seconds,
           "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
           "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
           "device": torch.cuda.get_device_name()}
prefix = os.environ["BENCH_MEM"]
open(f"{prefix}.rank{rank}.json", "w").write(json.dumps(payload))
'''


def write_corpus(path: Path, repeats: int = 4) -> None:
    base = [
        "Il giardino dei fiori infelici era un luogo silenzioso e antico, dove il vento parlava alle pietre.",
        "Ogni mattina la città si svegliava lentamente, mentre i treni attraversavano la pianura.",
        "La ricerca scientifica richiede pazienza, curiosità e la capacità di accettare i risultati inattesi.",
        "Un modello linguistico impara dalle regolarità del testo, non dalle intenzioni di chi lo ha scritto.",
        "Nel laboratorio si misurava il tempo, la memoria e la velocità con la stessa attenzione.",
        "Le montagne conservano la memoria dell'acqua e delle stagioni che si sono succedute nei secoli.",
    ]
    with path.open("w", encoding="utf-8") as handle:
        for _ in range(repeats):
            for line in base:
                handle.write(line + "\n")


def matformer_yaml(path: Path, tokens: Path, tokenizer: str, vocab: int, pad: int,
                   eos: int, seq: int, batch: int, steps: int, rows: int,
                   hidden: int, layers: int, heads: int, kv_heads: int, intermediate: int) -> None:
    path.write_text(f'''tokenizer:
  type: huggingface
  pretrained_name: {tokenizer}
  pad_token_id: {pad}
  eos_token_id: {eos}
model:
  type: default
  model_class: language_model
  name: tiny_qwen3_bench
  hidden_size: {hidden}
  num_hidden_layers: {layers}
  num_attention_heads: {heads}
  ffn_factor: {intermediate / hidden:.4f}
  vocab_size: {vocab}
  max_position_embeddings: {seq}
  tie_word_embeddings: true
  rms_norm_eps: 1e-6
  bias: false
  is_causal: true
  training_objective: autoregressive
  default_layer:
    normalization: rmsnorm
    normalization_position: pre
    ffn_activation: swiglu
    sequence_mixer: attention
    sequence_mixer_config:
      attn_impl: sdpa
      num_attention_heads: {heads}
      num_key_value_heads: {kv_heads}
      head_dim: {hidden // heads}
      positional_encoding: [rope]
      rope_theta: 10000.0
data:
  - name: train
    type: lazy
    file_format: jsonl
    data_root: {tokens}
    text_column: text
    input_ids_column: input_ids
    tokenize: false
    add_bos: false
    add_eos: false
    batch_size: {batch}
    max_seq_len: {seq}
    pad_token_id: {pad}
    varlen_strategy: unpadding
    head_samples: {rows}
training:
  ref_data: train
  learning_signal_selector: single
  learning_signals: [autoregressive]
  optimizer: adamw
  lr: 0.0001
  batch_size: {batch}
  accumulate_grad_batches: 1
  max_steps: {steps}
  max_epochs: 1
  save_every_n_steps: 0
  save_every_n_epochs: 0
  logger: [tqdm]
  project_name: benchmark
  run_name: tiny_qwen3_matformer
  checkpoint_name: tiny_qwen3_matformer
  save_dir: /tmp/matformer_benchmark
  precision: bf16-mixed
  seed: 27
''', encoding="utf-8")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def read_mem(prefix: str) -> list:
    rows = []
    prefix = Path(prefix)
    for path in sorted(prefix.parent.glob(prefix.name + ".rank*.json")):
        try:
            rows.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            pass
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--rows", type=int, default=8192, help="max documents read from the corpus")
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--intermediate", type=int, default=4096)
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--corpus", default="/home/matteo/dataset_4100775.jsonl",
                        help="jsonl with a text column or a plain-text file; missing -> synthetic corpus")
    parser.add_argument("--log", default="/tmp/benchmark_hf_matformer.log")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    work = Path(tempfile.mkdtemp(prefix="matformer_bench_"))
    corpus = Path(args.corpus)
    if not corpus.is_file():
        corpus = work / "corpus.jsonl"
        write_corpus(corpus)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    vocab = max(len(tokenizer), 1)
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    eos = tokenizer.eos_token_id or pad
    tokens_path = work / "tokens.jsonl"
    written = 0
    with tokens_path.open("w", encoding="utf-8") as out:
        for line in corpus.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                text = json.loads(line).get("text", line)
            except json.JSONDecodeError:
                text = line
            ids = tokenizer(text, add_special_tokens=False, truncation=True,
                            max_length=args.seq_len - 1)["input_ids"] + [eos]
            ids = ids[:args.seq_len] + [pad] * max(0, args.seq_len - len(ids))
            out.write(json.dumps({"input_ids": ids}) + "\n")
            written += 1
            if written >= args.rows:
                break
    yaml_path = work / "matformer.yaml"
    matformer_yaml(yaml_path, tokens_path, args.tokenizer, vocab, pad, eos,
                   args.seq_len, args.batch_size, args.steps, written,
                   args.hidden, args.layers, args.heads, args.kv_heads, args.intermediate)
    hf_script = work / "hf_train.py"
    hf_script.write_text(HF_SCRIPT)

    env = dict(os.environ, MATFORMER_SERVER="0", BENCH_CORPUS=str(corpus),
               BENCH_TOKENS=str(tokens_path),
               BENCH_TOKENIZER=args.tokenizer, BENCH_SEQ=str(args.seq_len), BENCH_PAD=str(pad),
               BENCH_VOCAB=str(vocab), BENCH_BATCH=str(args.batch_size),
               BENCH_STEPS=str(args.steps), BENCH_ROWS=str(args.rows),
               BENCH_HIDDEN=str(args.hidden), BENCH_LAYERS=str(args.layers),
               BENCH_HEADS=str(args.heads), BENCH_KV_HEADS=str(args.kv_heads),
               BENCH_INTERMEDIATE=str(args.intermediate),
               BENCH_OUT=str(work / "hf_out"), BENCH_MEM=str(work / "hf_mem"))
    log_path = Path(args.log)
    print(f"log        tail -f {log_path}", flush=True)
    results = {}
    runs = {
        "HF": lambda: [TORCHRUN, "--nproc_per_node", "4", "--master_port", str(free_port()),
                       str(hf_script)],
        "Matformer": lambda: [TORCHRUN, "--nproc_per_node", "4", "--master_port", str(free_port()),
                              "-m", "matformer.training.launch", "--cfg", str(yaml_path),
                              "--gpu", "4", "--max-steps", str(args.steps), "--no-save-model"],
    }
    for name in ("HF", "Matformer"):
        command = runs[name]()
        env["BENCH_MEM"] = str(work / ("hf_mem" if name == "HF" else "mat_mem"))
        env["MATFORMER_MEMORY_REPORT"] = str(work / "mat_mem")
        for old in Path("/tmp").glob(Path(env["BENCH_MEM"]).name + ".rank*.json"):
            old.unlink()
        started = time.perf_counter()
        print(f"\n=== {name} ===", flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n=== {name} ===\n")
            log.flush()
            completed = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        elapsed = time.perf_counter() - started
        if completed.returncode != 0:
            print(f"{name}: failed with exit code {completed.returncode}", file=sys.stderr)
            return 1
        memory = read_mem(env["BENCH_MEM"])
        results[name] = {"wall_seconds": elapsed, "memory": memory}

    print("\n=== comparison ===")
    print(f"{'trainer':10} {'wall s':>8} {'steps/s':>8} {'tokens/s/rank':>14} {'peak MiB/rank':>16}")
    for name, result in results.items():
        memory = result["memory"]
        peak = ", ".join(f"{row['rank']}:{row.get('peak_allocated_mib', 0):.0f}"
                         for row in sorted(memory, key=lambda one: one["rank"]))
        tokens_s = args.batch_size * args.seq_len * args.steps / max(result["wall_seconds"], 1e-9)
        print(f"{name:10} {result['wall_seconds']:8.2f} {args.steps / result['wall_seconds']:8.2f} "
              f"{tokens_s:14.0f} {peak:>16}")
    print(f"\nworkdir {work}  (use --keep to inspect)")
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
