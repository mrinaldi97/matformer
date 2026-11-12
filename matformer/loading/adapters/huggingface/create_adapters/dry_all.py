#!/usr/bin/env python3
"""Build fake HF checkpoints (real config.json + tiny safetensors with the exact
tensor NAMES only) for every adapter family, then run check_adapter.py in dry mode
on each. CPU-safe: no tensor data, no downloads of weights.

Usage:  MATFORMER_ATTENTION_NORMAL=sdpa .venv/bin/python matformer/loading/adapters/huggingface/create_adapters/dry_all.py
"""
import json, os, subprocess, sys
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file
import torch

OUT = "/tmp/fakeck"
CHECK = str(__import__("pathlib").Path(__file__).parent / "check_adapter.py")

def hf_cfg(repo):
    return json.load(open(hf_hub_download(repo, "config.json")))

def build(name, cfg, names):
    d = os.path.join(OUT, name); os.makedirs(d, exist_ok=True)
    json.dump(cfg, open(f"{d}/config.json", "w"), indent=1)
    save_file({n: torch.zeros(1) for n in names}, f"{d}/model.safetensors")
    return d

def bert_names(prefix, n, head=True, lm_prefix="cls.predictions"):
    ns = []
    if head:
        ns += [f"{lm_prefix}.decoder.weight" if False else None]  # placeholder, unused
    return ns

def enc_bert(prefix, n, layer_path="{p}.encoder.layer.{i}", tie_head=None):
    """BERT-skeleton names under `prefix`; head names returned separately."""
    ns = [f"{prefix}.embeddings.word_embeddings.weight",
          f"{prefix}.embeddings.position_embeddings.weight",
          f"{prefix}.embeddings.token_type_embeddings.weight",
          f"{prefix}.embeddings.LayerNorm.weight",
          f"{prefix}.embeddings.LayerNorm.bias"]
    for i in range(n):
        b = layer_path.format(p=prefix, i=i)
        for q in ("query", "key", "value"):
            ns += [f"{b}.attention.self.{q}.weight", f"{b}.attention.self.{q}.bias"]
        ns += [f"{b}.attention.output.dense.weight", f"{b}.attention.output.dense.bias",
               f"{b}.attention.output.LayerNorm.weight", f"{b}.attention.output.LayerNorm.bias",
               f"{b}.intermediate.dense.weight", f"{b}.intermediate.dense.bias",
               f"{b}.output.dense.weight", f"{b}.output.dense.bias",
               f"{b}.output.LayerNorm.weight", f"{b}.output.LayerNorm.bias"]
    return ns

def llama_names(n, *, lm_head=True, prefix="model"):
    ns = [f"{prefix}.embed_tokens.weight", f"{prefix}.norm.weight"]
    if lm_head: ns.append("lm_head.weight")
    for i in range(n):
        b = f"{prefix}.layers.{i}"
        ns += [f"{b}.input_layernorm.weight", f"{b}.post_attention_layernorm.weight"]
        ns += [f"{b}.self_attn.{p}_proj.weight" for p in "qkvo"]
        ns += [f"{b}.mlp.{m}_proj.weight" for m in ("gate", "up", "down")]
    return ns

# ------------------------------------------------------------------ build fakes
fakes = {}

cfg = hf_cfg("FacebookAI/roberta-base")
n = cfg["num_hidden_layers"]
names = enc_bert("roberta", n) + [
    "roberta.lm_head.dense.weight", "roberta.lm_head.dense.bias",
    "roberta.lm_head.layer_norm.weight", "roberta.lm_head.layer_norm.bias",
    "roberta.lm_head.bias"]           # decoder.weight tied -> absent
fakes["roberta"] = build("roberta", cfg, names)

cfg = hf_cfg("FacebookAI/xlm-roberta-base")
fakes["xlmr"] = build("xlmr", cfg, enc_bert("roberta", cfg["num_hidden_layers"]) + [
    "roberta.lm_head.dense.weight", "roberta.lm_head.dense.bias",
    "roberta.lm_head.layer_norm.weight", "roberta.lm_head.layer_norm.bias",
    "roberta.lm_head.bias"])

cfg = hf_cfg("distilbert/distilbert-base-cased")
n = cfg["n_layers"]
names = ["distilbert.embeddings.word_embeddings.weight",
         "distilbert.embeddings.position_embeddings.weight",
         "distilbert.embeddings.LayerNorm.weight", "distilbert.embeddings.LayerNorm.bias",
         "vocab_transform.weight", "vocab_transform.bias",
         "vocab_layer_norm.weight", "vocab_layer_norm.bias", "vocab_projector.bias"]
for i in range(n):
    b = f"distilbert.transformer.layer.{i}"
    for q in ("q", "k", "v"):
        names += [f"{b}.attention.{q}_lin.weight", f"{b}.attention.{q}_lin.bias"]
    names += [f"{b}.attention.out_lin.weight", f"{b}.attention.out_lin.bias",
              f"{b}.sa_layer_norm.weight", f"{b}.sa_layer_norm.bias",
              f"{b}.lin1.weight", f"{b}.lin1.bias", f"{b}.lin2.weight", f"{b}.lin2.bias",
              f"{b}.output_layer_norm.weight", f"{b}.output_layer_norm.bias"]
fakes["distilbert"] = build("distilbert", cfg, names)

cfg = hf_cfg("google/electra-base-discriminator")
fakes["electra_disc"] = build("electra_disc", cfg,
                              enc_bert("electra", cfg["num_hidden_layers"]))

cfg = hf_cfg("google/electra-small-generator")
fakes["electra_gen"] = build("electra_gen", cfg,
    enc_bert("electra", cfg["num_hidden_layers"]) + [
        "generator_predictions.dense.weight", "generator_predictions.dense.bias",
        "generator_predictions.LayerNorm.weight", "generator_predictions.LayerNorm.bias",
        "generator_lm_head.weight", "generator_lm_head.bias"])

cfg = hf_cfg("EuroBERT/EuroBERT-210m")
fakes["eurobert"] = build("eurobert", cfg, llama_names(cfg["num_hidden_layers"]))

cfg = hf_cfg("chandar-lab/NeoBERT")
n = cfg["num_hidden_layers"]
names = ["model.encoder.weight", "model.layer_norm.weight",
         "decoder.weight", "decoder.bias"]
for i in range(n):
    b = f"model.transformer_encoder.{i}"
    names += [f"{b}.qkv.weight", f"{b}.wo.weight",
              f"{b}.attention_norm.weight", f"{b}.ffn_norm.weight",
              f"{b}.ffn.w1.weight", f"{b}.ffn.w2.weight", f"{b}.ffn.w3.weight"]
fakes["neobert"] = build("neobert", cfg, names)

# previously-built regression fakes (from the earlier session scripts)
for name in ("ministral8b", "smollm3", "stablelm", "gemma2", "llama3rope",
             "mistral_nemo", "mistral_v01", "exaone", "olmo1"):
    if os.path.isdir(os.path.join(OUT, name)):
        fakes[name] = os.path.join(OUT, name)

# olmo2 regression fake, rebuilt with the 0425 release's QK-norm tensors
if os.path.isdir(os.path.join(OUT, "olmo2")):
    cfg = json.load(open(f"{OUT}/olmo2/config.json"))
    n = cfg["num_hidden_layers"]
    names = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
    for i in range(n):
        b = f"model.layers.{i}"
        names += [f"{b}.post_attention_layernorm.weight", f"{b}.post_feedforward_layernorm.weight"]
        names += [f"{b}.self_attn.{p}_proj.weight" for p in "qkvo"]
        names += [f"{b}.self_attn.q_norm.weight", f"{b}.self_attn.k_norm.weight"]
        names += [f"{b}.mlp.{m}_proj.weight" for m in ("gate", "up", "down")]
    fakes["olmo2"] = build("olmo2", cfg, names)

# ------------------------------------------------------------------ run dry checks
env = dict(os.environ, MATFORMER_ATTENTION_NORMAL="sdpa")
fails = []
for name, path in fakes.items():
    r = subprocess.run([sys.executable, CHECK, path, "--mode", "dry", "--json"],
                       capture_output=True, text=True, env=env)
    try:
        rep = json.loads(r.stdout)["reports"][0]
        ok = rep["ok"]
        bad = [f"{c['name']}: {c['detail'][:140]}" for c in rep["checks"] if not c["ok"]]
    except Exception:
        ok, bad = False, [f"exit {r.returncode}: {r.stdout[-200:]} {r.stderr[-300:]}"]
    print(f"{'PASS' if ok else 'FAIL':4}  {name:14} {'' if ok else bad}")
    if not ok:
        fails.append(name)

print("\nSummary:", f"{len(fakes)-len(fails)}/{len(fakes)} passed",
      ("FAILURES: " + ", ".join(fails)) if fails else "")
sys.exit(1 if fails else 0)
