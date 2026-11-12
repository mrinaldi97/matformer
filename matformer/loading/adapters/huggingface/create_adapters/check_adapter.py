#!/usr/bin/env python3
"""
check_adapter.py — validate a Matformer HuggingFace adapter against a real HF model.

Modes
-----
dry : CPU-only, no tensor data loaded. Verifies that
      (1) an adapter exists for the model_type and loads as YAML,
      (2) the HF config converts into a Matformer ModelConfig and every layer resolves,
      (3) the model can be instantiated on the meta device,
      (4) the adapter's stable weight names exactly match the model's expected stable
          names (catches renames/typos/fused-projection mismatches without any download
          of tensor data),
      (5) every tensor actually present in the checkpoint is consumed by the adapter.
gpu : full comparison — loads the model with both transformers and Matformer on the GPU
      (sequentially, to halve peak memory) and compares the logits of a forward pass.
both: dry then gpu.
auto: dry, plus gpu if CUDA is available (default).

Usage
-----
  python check_adapter.py Qwen/Qwen3-1.7B
  python check_adapter.py Qwen/Qwen3-1.7B --mode gpu --prompt "The quick brown fox"
  python check_adapter.py meta-llama/Meta-Llama-3-8B --mode both --dtype bfloat16 --json
  python check_adapter.py ./local_hf_dir                       # local HF-format folder

The argument is a HF hub repo id or a local HF-format folder. Exit code: 0 all checks
passed, 1 at least one check failed, 2 the script itself errored out.
With --json the report is printed as a single JSON object (agent-friendly); the human
format is the default.
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

import torch


def resolve_path(path: str) -> str:
    """Make a holder path: hf:// scheme for hub ids / local HF folders."""
    return path if "://" in path else f"hf://{path}"


# ----------------------------------------------------------------- dry checks
def _checkpoint_tensor_names(folder: Path) -> set:
    """All tensor names in the checkpoint, without loading tensor data."""
    from safetensors import safe_open
    index = next(folder.glob("*.index.json"), None)
    if index is not None:
        files = sorted({folder / f for f in json.loads(index.read_text())["weight_map"].values()})
    else:
        files = sorted(folder.glob("*.safetensors"))
    names = set()
    for f in files:
        with safe_open(f, framework="pt", device="cpu") as sf:
            names |= set(sf.keys())
    return names


def dry_checks(path: str) -> dict:
    """No tensor data, no GPU. Returns a report dict with 'ok' and per-check details."""
    from matformer.loading.utils import make_holder
    from matformer.matformer_wrapper import load_model

    report, state = {"mode": "dry", "checks": []}, {}

    def check(name, fn):
        try:
            detail = fn()
            report["checks"].append({"name": name, "ok": True, "detail": detail or ""})
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            report["checks"].append({"name": name, "ok": False, "detail": str(e)})

    def adapter_load():
        state["holder"] = make_holder(resolve_path(path))
        return f"model_type adapter loaded from {state['holder']._DEFS}"

    def config_conversion():
        cfgs = state["cfgs"] = state["holder"].get_configs()
        m = cfgs.model
        missing = [f for f in ("hidden_size", "num_hidden_layers", "vocab_size", "ffn_factor")
                   if getattr(m, f, None) is None]
        assert not missing, f"ModelConfig fields left unset: {missing}"
        return (f"hidden={m.hidden_size} layers={m.num_hidden_layers} vocab={m.vocab_size} "
                f"ffn_factor={m.ffn_factor:.3g} causal={m.is_causal} objective={m.training_objective}")

    def layer_resolution():
        m = state["cfgs"].model
        mixers = {}
        for i in range(m.num_hidden_layers):
            lc = m.get_layer_config(i)
            mixers[str(lc.sequence_mixer)] = mixers.get(str(lc.sequence_mixer), 0) + 1
        return f"all {m.num_hidden_layers} layers resolve; mixers: {mixers}"

    def meta_instantiation():
        # load_model with configs only (no holder) builds the model without touching any
        # tensor data — meta device keeps it allocation-free.
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            result = load_model(configs=state["cfgs"], map_location="meta", inference=True)
        finally:
            torch.set_default_dtype(prev)
        state["expected_keys"] = set(result["model"].model._mappings)
        return f"model builds on meta device; {len(state['expected_keys'])} expected stable weights"

    def stable_name_coverage():
        rename, rules, _aliases = state["holder"]._expand_definition(state["holder"].definition)
        present = _checkpoint_tensor_names(Path(state["holder"].path))
        # mirror the loader's _stream: a rename whose source (or an aliased candidate)
        # is absent from the checkpoint is skipped, so it provides nothing — that lets
        # one adapter serve both flavoured checkpoints (e.g. ELECTRA generator/discriminator)
        converted = {stable for src, stable in rename.items()
                     if any(c in present for c in
                            state["holder"]._candidate_source_keys(src, _aliases))}
        # a transform rule whose source is absent from the checkpoint is skipped too
        # (e.g. ModernBERT layer 0, whose HF attn_norm is an Identity with no tensor)
        runnable = [r for r in rules
                    if any(any(c in present for c in
                               state["holder"]._candidate_source_keys(s, _aliases))
                           for s in r["sources"])]
        converted |= {o for r in runnable for o in r["outputs"]}
        missing = state["expected_keys"] - converted
        unexpected = sorted(converted - state["expected_keys"])
        assert not unexpected, (
            f"unexpected (adapter provides, model doesn't expect): {unexpected}")
        # a tied lm_head may legitimately have no tensor of its own (the loader aliases
        # it to the token embeddings) — the only tolerated missing stable name
        tied = getattr(state["cfgs"].model, "tie_word_embeddings", False)
        fatal = sorted(missing - ({"lm_head.weight"} if tied else set()))
        assert not fatal, (
            f"missing (model expects, adapter doesn't provide): {fatal}")
        return f"{len(converted)} stable names match the model exactly"

    def checkpoint_coverage():
        rename, rules, _aliases = state["holder"]._expand_definition(state["holder"].definition)
        consumed = set(rename) | {s for r in rules for s in r["sources"]}
        present = _checkpoint_tensor_names(Path(state["holder"].path))
        # account for source_key_aliases: a rename key resolves as-is, prefix-stripped
        # or prefix-added (see checkpoint_holder._candidate_source_keys)
        candidates = {c for k in consumed for c in
                      state["holder"]._candidate_source_keys(k, _aliases)}
        unconsumed = sorted(k for k in present - candidates if "_extra_state" not in k)
        return (f"{len(candidates & present)}/{len(present)} checkpoint tensors consumed; "
                f"unconsumed: {unconsumed or 'none'}")

    check("adapter-load", adapter_load)
    check("config-conversion", config_conversion)
    check("layer-resolution", layer_resolution)
    check("meta-instantiation", meta_instantiation)
    check("stable-name-coverage", stable_name_coverage)
    check("checkpoint-coverage", checkpoint_coverage)
    report["ok"] = all(c["ok"] for c in report["checks"])
    return report

def _is_bare_model(local: str) -> bool:
    """True when the checkpoint's architecture is a bare text model (e.g. an embedding
    model like Gemma3TextModel) with no lm_head: AutoModel output = last hidden state."""
    import json as _json
    cfg = _json.loads((Path(local) / "config.json").read_text())
    archs = cfg.get("architectures", [])
    return bool(archs) and not any("ForCausalLM" in a or "ForMaskedLM" in a for a in archs)


def _install_hf_component_hooks(hf_model, store):
    """Register hooks on a decoder-LM transformers model, storing float cpu copies.
    Returns (handles, pairs) — pairs are (label, hf_key, matformer_socket_suffix) in
    execution order — or None when the architecture doesn't match the decoder pattern."""
    handles = []

    def _save(key, from_input=False):
        def hook(module, inputs, output):
            t = inputs[0] if from_input else output
            if isinstance(t, tuple):
                t = t[0]
            store[key] = t.detach().float().cpu()
        return hook

    try:
        inner, layers = hf_model.model, hf_model.model.layers
    except AttributeError:
        return None
    pairs = []
    handles.append(inner.embed_tokens.register_forward_hook(_save("embed_tokens")))
    pairs.append(("token embeddings", "embed_tokens", ".embed_tokens"))
    for i in range(len(layers)):
        L = layers[i]
        p = f"layer_{i}"
        handles += [
            L.input_layernorm.register_forward_hook(_save(f"{p}.attn_normed")),
            L.self_attn.register_forward_hook(_save(f"{p}.attn_out")),
            L.post_attention_layernorm.register_forward_hook(_save(f"{p}.post_attn", from_input=True)),
            L.post_attention_layernorm.register_forward_hook(_save(f"{p}.mlp_in")),
            L.mlp.down_proj.register_forward_hook(_save(f"{p}.mlp_hidden", from_input=True)),
            L.mlp.register_forward_hook(_save(f"{p}.mlp_out")),
            L.register_forward_hook(_save(f"{p}.block_out")),
        ]
        pairs += [
            (f"L{i} attn-norm out",       f"{p}.attn_normed", f".layers.{i}.pre_attn"),
            (f"L{i} attn out (o_proj)",   f"{p}.attn_out",    f".layers.{i}.self_attn"),
            (f"L{i} post-attn residual",  f"{p}.post_attn",   f".layers.{i}.post_add_attn"),
            (f"L{i} mlp-norm out",        f"{p}.mlp_in",      f".layers.{i}.pre_mlp"),
            (f"L{i} mlp hidden (pre-dn)", f"{p}.mlp_hidden",  f".layers.{i}.mlp.hidden_act"),
            (f"L{i} mlp out",             f"{p}.mlp_out",     f".layers.{i}.post_mlp"),
            (f"L{i} block out",           f"{p}.block_out",   f".layers.{i}.layer_output"),
        ]
    handles.append(inner.norm.register_forward_hook(_save("final_norm")))
    pairs.append(("final norm out", "final_norm", ".final_norm"))
    return handles, pairs


def _mf_capture_plugin():
    """A Plugin that records the tensor flowing through the socket it is mounted on
    (under its socket address) and returns None, leaving the stream untouched."""
    from matformer.plugins.plugin_classes import Plugin
    from matformer.dataclasses.tensors_dataclasses import TensorDC

    class CapturePlugin(Plugin):
        name = "capture"

        def forward(self, x, **kw):
            if x is not None:
                t = x.tensor if isinstance(x, TensorDC) else x
                self.cache.storage.setdefault("capture", {})[self.address] = t.detach().float().cpu()
            return None

    return CapturePlugin


def _compare_component_pairs(pairs, hf_acts, mf_acts, sockets, atol):
    """Pairwise HF-vs-Matformer comparison of captured intermediates, execution order."""
    def _addr(suffix):
        name = suffix.lstrip(".")
        return next((a for a in sockets if a == name or a.endswith(suffix)), None)

    rows, first_diverged = [], None
    for label, hf_key, suffix in pairs:
        hf_t, addr = hf_acts.get(hf_key), _addr(suffix)
        mf_t = mf_acts.get(addr) if addr else None
        row = {"point": label, "hf": hf_key, "matformer": addr}
        if hf_t is None or mf_t is None:
            row.update(status="NOT CAPTURED")
        elif hf_t.shape != mf_t.shape:
            row.update(status=f"SHAPE {tuple(hf_t.shape)} vs {tuple(mf_t.shape)}")
            first_diverged = first_diverged or label
        else:
            d = (hf_t - mf_t).abs()
            ref = max(hf_t.abs().max().item(), 1e-6)
            row.update(max_abs=d.max().item(), mean_abs=d.mean().item(),
                       rel=d.max().item() / ref)
            ok = row["max_abs"] <= atol or row["rel"] <= 0.02
            row["status"] = "ok" if ok else "DIVERGED"
            if not ok:
                first_diverged = first_diverged or label
        rows.append(row)
    return rows, first_diverged


def gpu_checks(path: str, device: str, dtype: torch.dtype, seq_len: int, prompt: str | None,
               atol: float, components: bool = False) -> dict:
    """Forward-pass logits comparison: transformers vs Matformer.
    With components=True, additionally captures every socket/layer intermediate of both
    runs (one pass each) and reports where the two computations first diverge."""
    from matformer.loading.utils import make_holder
    from matformer.matformer_wrapper import load_model
    from transformers import (AutoModelForCausalLM, AutoModelForMaskedLM,
                              AutoTokenizer)

    holder = make_holder(resolve_path(path))
    local = str(holder.path)
    objective = holder.get_configs().model.training_objective
    masked = objective == "masked"
    bare = _is_bare_model(local)

    # ---- input ids: tokenize the prompt, or random ids seeded for reproducibility
    if prompt is not None:
        ids = torch.tensor(AutoTokenizer.from_pretrained(local)(prompt)["input_ids"])[:seq_len]
        ids = ids.unsqueeze(0)
    else:
        vocab = holder.get_configs().model.vocab_size
        torch.manual_seed(0)
        ids = torch.randint(0, vocab, (1, seq_len))
    note = ("masked-LM objective: comparison assumes Matformer's plain (unmasked) forward "
            "matches the HF model without masking") if masked else ""

    # ---- HF side first, then free it before building the Matformer (halves peak memory)
    hf_acts, hf_handles, component_pairs = {}, None, None
    if bare:
        from transformers import AutoModel
        hf_model = AutoModel.from_pretrained(local, torch_dtype=dtype).to(device).eval()
        with torch.no_grad():
            hf_out = hf_model(input_ids=ids.to(device)).last_hidden_state.float().cpu()
        note = ("bare model (embedding, no lm_head): comparing last hidden states; " + note).strip()
    else:
        hf_cls = AutoModelForMaskedLM if masked else AutoModelForCausalLM
        hf_model = hf_cls.from_pretrained(local, torch_dtype=dtype).to(device).eval()
        if components:
            got = _install_hf_component_hooks(hf_model, hf_acts)
            if got:
                hf_handles, component_pairs = got
            else:
                note = (note + " | component hooks unsupported on this architecture").strip(" |")
        with torch.no_grad():
            hf_out = hf_model(input_ids=ids.to(device)).logits.float().cpu()
    if hf_handles:
        for h in hf_handles:
            h.remove()
    del hf_model
    torch.cuda.empty_cache() if device.startswith("cuda") else None

    # ---- Matformer side
    result = load_model(path=resolve_path(path), map_location=device, dtype=dtype,
                        inference=True)
    missing, unexpected = result["missing"], result["unexpected"]
    model = result["model"].eval()
    if components and component_pairs:
        model.attach("*", _mf_capture_plugin())   # one passive capture plugin per socket
    with torch.no_grad():
        if bare:
            out = model(ids.to(device), return_type="hidden")
        else:
            out = model(ids.to(device))
        mf_out = (out.tensor if hasattr(out, "tensor") else out).float().cpu()
    components_report = None
    if components and component_pairs:
        rows, first_diverged = _compare_component_pairs(
            component_pairs, hf_acts, model.cache.storage.get("capture", {}),
            model.sockets, atol)
        components_report = {"pairs": rows, "first_diverged": first_diverged}

    diff = (hf_out - mf_out).abs()
    report = {
        "mode": "gpu",
        "missing_weights": sorted(missing or []),
        "unexpected_weights": sorted(unexpected or []),
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "atol": atol,
        "input": prompt if prompt is not None else f"random ids (1 x {ids.shape[1]}), seed 0",
        "note": note,
        "components": components_report,
    }
    missing_ok = (not report["missing_weights"]
                  or report["missing_weights"] == ["lm_head.weight"])   # tied lm_head: benign
    report["ok"] = (missing_ok and not report["unexpected_weights"]
                    and report["max_abs_diff"] <= atol)
    return report


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        "check_adapter",
        description="Validate a Matformer HF adapter for a given HF model.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("model", help="HF hub repo id or local HF-format folder")
    ap.add_argument("--mode", default="auto", choices=["dry", "gpu", "both", "auto"],
                    help="auto = dry, plus gpu when CUDA is available (default)")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"],
                    help="dtype for the GPU comparison (default float32, for exactness)")
    ap.add_argument("--prompt", default=None,
                    help="text input for the GPU forward pass (default: seeded random ids)")
    ap.add_argument("--seq-len", type=int, default=64, help="sequence length for random ids")
    ap.add_argument("--device", default=None, help="override device for the GPU mode")
    ap.add_argument("--atol", type=float, default=1e-4, help="max-abs-diff tolerance")
    ap.add_argument("--components", action="store_true",
                    help="in gpu mode, also capture every intermediate (HF hooks + Matformer "
                         "socket plugins) and report where the two runs first diverge")
    ap.add_argument("--json", action="store_true", help="print a single JSON report")
    args = ap.parse_args()

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
             "float16": torch.float16}[args.dtype]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    want_gpu = args.mode in ("gpu", "both") or (args.mode == "auto" and torch.cuda.is_available())
    if args.mode == "gpu" and not torch.cuda.is_available() and not args.device:
        print("WARNING: CUDA not available, running GPU check on CPU", file=sys.stderr)

    reports = []
    import io
    real_stdout = sys.stdout
    if args.json:
        sys.stdout = io.StringIO()    # framework prints (registry warnings, holders...) to stdout
    try:
        reports.append(dry_checks(args.model))
        if want_gpu:
            if reports[0]["mode"] == "dry" and not reports[0]["ok"]:
                print("Dry checks failed, skipping GPU comparison.", file=sys.stderr)
            else:
                reports.append(gpu_checks(args.model, device, dtype, args.seq_len,
                                          args.prompt, args.atol, components=args.components))
    except Exception:
        traceback.print_exc()
        sys.stdout = real_stdout
        if args.json:
            print(json.dumps({"ok": False, "error": traceback.format_exc()}, indent=2))
        sys.exit(2)
    finally:
        sys.stdout = real_stdout

    if args.json:
        final = {"ok": all(r["ok"] for r in reports), "reports": reports}
        print(json.dumps(final, indent=2))
    else:
        for r in reports:
            print(f"\n=== {r['mode'].upper()} ===")
            for c in r.get("checks", []):
                print(f"[{'PASS' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail']}")
            for k in ("missing_weights", "unexpected_weights", "max_abs_diff", "mean_abs_diff", "note"):
                if r.get(k) is not None and r[k] != "":
                    print(f"  {k}: {r[k]}")
            comps = r.get("components")
            if comps:
                print("  component parity (execution order):")
                for c in comps["pairs"]:
                    nums = (f"  max_abs={c['max_abs']:.3e} mean_abs={c['mean_abs']:.3e} rel={c['rel']:.3e}"
                            if "max_abs" in c else "")
                    print(f"    [{c['status']:>28s}] {c['point']}{nums}")
                print(f"  first divergence: {comps['first_diverged'] or 'none'}")
            print(f"-> {'PASS' if r['ok'] else 'FAIL'}")
    sys.exit(0 if all(r["ok"] for r in reports) else 1)


if __name__ == "__main__":
    main()
