"""Small training-side helpers that don't belong to the trainer class itself."""
from typing import Optional

import torch

# Peak dense throughput per GPU in TFLOP/s, as the vendor publishes it. torch cannot know this, and
# it is the denominator of MFU, so it is declared here rather than benchmarked. Listing bf16
# separately is the point: bf16 tensor cores do not exist before Ampere, so a bf16 run on a V100 is
# served by the fp32 cores at a tenth of its fp16 rate. That is the kind of thing worth making visible.
PEAK_TFLOPS = {
    "V100":     {"fp16": 125.0, "bf16": 15.7,   "fp32": 15.7},
    "T4":       {"fp16": 65.0,  "bf16": 8.1,    "fp32": 8.1},
    "A30":      {"fp16": 165.0, "bf16": 165.0,  "fp32": 19.5},
    "A40":      {"fp16": 150.0, "bf16": 150.0,  "fp32": 19.0},
    "A100":     {"fp16": 312.0, "bf16": 312.0,  "fp32": 19.5},
    "L4":       {"fp16": 121.0, "bf16": 121.0,  "fp32": 30.3},
    "L40":      {"fp16": 181.0, "bf16": 181.0,  "fp32": 90.5},
    "H100":     {"fp16": 989.0, "bf16": 989.0,  "fp32": 67.0},
    "H200":     {"fp16": 989.0, "bf16": 989.0,  "fp32": 67.0},
    "GH200":    {"fp16": 989.0, "bf16": 989.0,  "fp32": 67.0},
    "B200":     {"fp16": 2250.0, "bf16": 2250.0, "fp32": 80.0},
    "3090":     {"fp16": 71.0,  "bf16": 71.0,   "fp32": 35.6},
    "4090":     {"fp16": 165.0, "bf16": 165.0,  "fp32": 82.6},
    "6000 Ada": {"fp16": 182.5, "bf16": 182.5,  "fp32": 91.1},
}


def detect_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def precision_dtype(precision):
    """The parameter dtype a precision setting implies. A model built in a dtype the target GPU
    cannot compute with fails in obscure places (the AMP unscale kernel, for one), so this comes from
    the config rather than from a default that happens to suit one architecture.

    bf16 needs no loss scaler, so its weights live in bf16 directly. Half precision does: `16-mixed`
    keeps fp32 weights and lets autocast produce fp16 activations, because a scaler cannot unscale
    fp16 gradients. Only the explicit `-true` settings put the weights in the reduced dtype."""
    text = str(precision or "").lower()
    if text.endswith("-true"):
        if "bf16" in text or "bfloat" in text:
            return torch.bfloat16
        if "16" in text or "half" in text:
            return torch.float16
        return torch.float64 if "64" in text else torch.float32
    if "bf16" in text or "bfloat" in text:
        return torch.bfloat16
    if "64" in text:
        return torch.float64
    return torch.float32


def compute_dtype_name(precision) -> str:
    """Which dtype the matmuls of a run actually happen in, as `PEAK_TFLOPS` spells it."""
    text = str(precision or "").lower()
    if "bf16" in text:
        return "bf16"
    if "16" in text or "half" in text:
        return "fp16"
    return "fp32"


def peak_tflops(device_name: str = "", precision: str = "bf16-mixed") -> Optional[float]:
    """The TFLOP/s a device reaches at this precision, None when the device is unknown. Matched on the
    marketing name (`nvidia-smi -L` gives the same string), longest key first so that '6000 Ada' wins
    over a bare digit."""
    if not device_name:
        return None
    key = next((one for one in sorted(PEAK_TFLOPS, key=len, reverse=True)
                if one.lower() in device_name.lower()), "")
    table = PEAK_TFLOPS.get(key)
    return table.get(compute_dtype_name(precision)) if table else None


def matmul_parameter_count(model) -> int:
    """Parameters that take part in a multiplication: embedding *lookups* cost no FLOPs, so they are
    left out, while the projection over the vocabulary is kept (it is the single largest matmul of a
    language model step, and forgetting it deflates MFU badly)."""
    configs = getattr(model, "configs", None)
    model_config = getattr(configs, "model", None) if configs is not None else None
    tied = bool(getattr(model_config, "tie_word_embeddings", False))
    hidden, vocab = (getattr(model_config, "hidden_size", None),
                     getattr(model_config, "vocab_size", None))
    total, shared = 0, set()
    for name, parameter in model.named_parameters():
        if id(parameter) in shared:
            continue
        shared.add(id(parameter))
        lowered = name.lower()
        if "embed" in lowered or "wte" in lowered or "wpe" in lowered:
            # A tied table is also the output projection: count it once, as a matmul.
            if tied and hidden and vocab and "token" in lowered:
                total += int(hidden) * int(vocab)
            continue
        total += parameter.numel()
    return total


def step_flops(tokens: int, parameters: int, layers: int = 0, hidden: int = 0,
               doc_len: float = 0.0, causal: bool = False) -> float:
    """FLOPs of one forward *and* backward step, the standard estimate: six multiply-adds per
    parameter per token, plus the attention's quadratic term, which the sequence mixer pays on top of
    the projections.

    `doc_len` is the mean document length in the batch. The exact term needs the sum of the squared
    lengths, which on the GPU would cost a synchronisation per step just to know it: with packed
    batches of comparable chunks the mean is both free and close, and it is the number that matters
    when comparing two implementations of the same workload."""
    quadratic = 0.0
    if layers and hidden and doc_len and tokens:
        sequences = max(tokens / max(doc_len, 1.0), 1.0)
        # QK^T and A·V are two matmuls of 2·d·s² FLOPs per layer, halved when causal, tripled backward.
        quadratic = (6.0 if causal else 12.0) * layers * hidden * sequences * doc_len ** 2
    return 6.0 * parameters * tokens + quadratic
