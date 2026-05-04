import sys
sys.path.append('../')
from matformer.matformer_registry import registry
from matformer.transformer_blocks import BERTModel, Autoregressive_Model
from matformer.models import PL_ModelWrapper
import torch
import re
import json
import argparse
from pathlib import Path
from tqdm import tqdm


def load_model(checkpoint_path, ModelClass=BERTModel, map_location='cpu', tokenizer=None, overrides=None):
    model, cfg, ckpt = PL_ModelWrapper.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        ModelClass=ModelClass,
        map_location=map_location,
        tokenizer=tokenizer,
        overrides=overrides,
        return_checkpoint=True
    )
    model = model.to(map_location).to(torch.bfloat16).eval()
    for module in model.modules():
        if hasattr(module, "alibi_slopes") and module.alibi_slopes is not None:
            module.alibi_slopes = module.alibi_slopes.to(dtype=torch.float32)
    return model, cfg, ckpt


def extract_parameter_metadata(param_name):
    match = re.search(r'\.layers\.(\d+)', param_name)
    layer = int(match.group(1)) if match else -1
    if 'attn' in param_name:
        family = 'attention'
    elif 'mlp' in param_name:
        family = 'mlp'
    elif 'lm_head' in param_name:
        family = 'lm_head'
    elif 'embed' in param_name:
        family = 'embedding'
    else:
        family = 'other'
    return {
        "param_name": param_name,
        "layer": layer,
        "family": family,
        "isNorm": 'norm' in param_name,
        "isBias": param_name.split('.')[-1] != 'weight',
    }


def compute_static_stats(array: torch.Tensor) -> dict:
    array = array.to(torch.float32)

    # --- Distributional (valid for any shape) ---
    mean     = array.mean()
    diffs    = array - mean
    var      = (diffs ** 2).mean()
    std      = var.sqrt()
    zscores  = diffs / std
    skewness = (zscores ** 3).mean()
    kurtosis = (zscores ** 4).mean() - 3.0

    # --- Spectral (undefined for 1D tensors, e.g. LayerNorm weights) ---
    if array.ndim >= 2:
        svdvals        = torch.linalg.svdvals(array)    # descending, shape (min(m,n),)
        frobenius      = svdvals.norm()                  # == sqrt(Σσ_i²) == ||W||_F
        spectral_norm  = svdvals[0]                      # σ_1 = max amplification factor
        stable_rank    = (frobenius / spectral_norm)**2  # ||W||_F² / σ_1²
        p              = svdvals / svdvals.sum()
        effective_rank = torch.exp(-(p * p.clamp(min=1e-10).log()).sum())  # Roy & Vetterli 2007
    else:
        frobenius = spectral_norm = stable_rank = effective_rank = None

    return {
        "mean":           float(mean),
        "var":            float(var),
        "std":            float(std),
        "skewness":       float(skewness),
        "kurtosis":       float(kurtosis),
        "frobenius":      float(frobenius)      if frobenius      is not None else None,
        "spectral_norm":  float(spectral_norm)  if spectral_norm  is not None else None,
        "stable_rank":    float(stable_rank)    if stable_rank    is not None else None,
        "effective_rank": float(effective_rank) if effective_rank is not None else None,
    }


def process_checkpoint(checkpoint_path, ModelClass, device, tokenizer, output_file, include_step):
    model, cfg, ckpt = load_model(checkpoint_path, ModelClass=ModelClass, map_location=device, tokenizer=tokenizer)
    global_step = ckpt['global_step']

    with torch.no_grad():
        for param in tqdm(model._mappings.keys(), desc=f"step={global_step}"):
            if 'alibi_slopes' in param:
                continue
            stats = extract_parameter_metadata(param)
            stats.update(compute_static_stats(model._get_weight_pointer(param)))
            if include_step:
                stats['global_step'] = global_step
            with open(output_file, "a") as f:
                f.write(json.dumps(stats) + "\n")

    del model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoints_dir", type=str, help="Directory containing multiple checkpoints")
    group.add_argument("--checkpoint",      type=str, help="Single checkpoint file")
    parser.add_argument("--output_file",    type=str, required=True)
    parser.add_argument("--arch",           type=str, default="bert", choices=["bert", "gpt"])
    parser.add_argument("--device",         type=str, default="cpu")
    parser.add_argument("--tokenizer",      type=str, default="mrinaldi/Gettone")
    args = parser.parse_args()

    ModelClass = BERTModel if args.arch == "bert" else Autoregressive_Model

    if args.checkpoints_dir:
        checkpoints = sorted(Path(args.checkpoints_dir).glob("*.ckpt"))
        for ckpt_path in tqdm(checkpoints):
            process_checkpoint(ckpt_path, ModelClass, args.device, args.tokenizer, args.output_file, include_step=True)
    else:
        process_checkpoint(args.checkpoint, ModelClass, args.device, args.tokenizer, args.output_file, include_step=False)
