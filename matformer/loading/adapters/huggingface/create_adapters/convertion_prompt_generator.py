#!/usr/bin/env python3
"""
prompt_generator.py — builds a ready-to-send conversion prompt for a HF model.

Usage:
    python prompt_generator.py Qwen/Qwen3-1.7B
    python prompt_generator.py --file-list models.txt   # one HF repo id per line
"""
import re, json, argparse, requests
from pathlib import Path
from huggingface_hub import hf_hub_download, get_safetensors_metadata

PROMPT_TEMPLATE = Path(__file__).with_name("convertion_generation_prompt.txt").read_text()
ADAPTERS_DIR = Path(__file__).parent.parent
GH_RAW = "https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/{t}/modeling_{t}.py"


def collapse(names):
    return sorted({".".join(re.sub(r"\d+", "N", p) for p in n.split(".")) for n in names})


def build_prompt(model_id: str):
    # 1. config.json — tiny file, cheap to fetch, gives us model_type to check the skip condition.
    config = json.loads(Path(hf_hub_download(model_id, "config.json")).read_text())
    model_type = config.get("model_type", "unknown")

    out_yaml = ADAPTERS_DIR / f"{model_type}.yaml"
    if out_yaml.exists():
        print(f"[skip] {model_id} -> {out_yaml} already exists")
        return

    # 2. Weight names WITHOUT downloading any tensor data: get_safetensors_metadata does a
    #    range-request for just the safetensors header (works for sharded and single-file repos).
    weight_names = list(get_safetensors_metadata(model_id).weight_map.keys())
    collapsed = collapse(weight_names)

    # 3. Modeling file, best-effort from GitHub.
    modeling_src = ""
    resp = requests.get(GH_RAW.format(t=model_type), timeout=15)
    if resp.ok:
        modeling_src = resp.text
    else:
        print(f"[warn] no modeling file for '{model_type}' (HTTP {resp.status_code})")

    parts = [
        PROMPT_TEMPLATE,
        f"\n\n=== HF CONFIG.JSON ({model_id}) ===\n{json.dumps(config, indent=2)}",
        f"\n\n=== COLLAPSED WEIGHT NAME LIST ===\n" + "\n".join(collapsed),
    ]
    if modeling_src:
        parts.append(f"\n\n=== HF MODELING FILE (modeling_{model_type}.py) ===\n{modeling_src}")

    out_file = Path(f"{model_type}_prompt.txt")
    out_file.write_text("".join(parts))
    print(f"[ok] {model_id} -> {out_file}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id", nargs="?", help="e.g. Qwen/Qwen3-1.7B")
    ap.add_argument("--file-list", help="text file, one HF repo id per line")
    args = ap.parse_args()

    ids = []
    if args.file_list:
        ids = [l.strip() for l in Path(args.file_list).read_text().splitlines() if l.strip() and not l.startswith("#")]
    elif args.model_id:
        ids = [args.model_id]
    else:
        ap.error("provide a model_id or --file-list")

    for mid in ids:
        try:
            build_prompt(mid)
        except Exception as e:
            print(f"[error] {mid}: {e}")


if __name__ == "__main__":
    main()