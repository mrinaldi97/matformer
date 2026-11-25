"""Queue dialog fields without importing torch.

The server can answer "which switches does a training take?" before a model or a CUDA context
exists. Keeping this schema separate from the launcher is what makes the queue dialog open on a
slow login node instead of waiting for torch.
"""
RESET_CHOICES = ("lr_scheduler", "optimizer", "datamodule", "steps")
DEVICES = ("auto", "cuda", "cpu", "mps", "tpu")
CLI_TO_TRAINING_FIELD = {
    "gpu": "device_count", "nodes": "num_nodes", "reset": "what_to_reset",
    "ckpt_version": "version", "checkpoint": "checkpoint",
    "start_from_scratch": "start_from_scratch", "max_steps": "max_steps", "compile": "compile",
}

_EXTRA_FIELDS = [
    {"name": "device", "label": "Accelerator", "kind": "choice", "options": list(DEVICES),
     "default": "auto", "doc": "What the job asks torch for"},
    {"name": "gpus", "label": "Pin to GPUs", "kind": "text", "default": None,
     "doc": "CUDA_VISIBLE_DEVICES for this job only, e.g. 2,3"},
    {"name": "sweep", "label": "Sweep points", "kind": "text", "default": None,
     "doc": "Only these points of the config's sweep (numbers or name globs); empty runs them all"},
    {"name": "save_model", "label": "Save model", "kind": "bool", "default": True, "doc": ""},
    {"name": "save_plugins", "label": "Save plugins", "kind": "bool", "default": False, "doc": ""},
]



_TRAINING_FIELDS = [
    {"name": "device_count", "label": "GPUs", "kind": "int", "default": 1, "min": 0,
     "doc": "GPU count for this job"},
    {"name": "num_nodes", "label": "Nodes", "kind": "int", "default": 1, "min": 1,
     "doc": "SLURM nodes for this job"},
    {"name": "what_to_reset", "label": "Reset", "kind": "text", "default": None,
     "doc": "Comma separated: lr_scheduler, optimizer, datamodule, steps"},
    {"name": "version", "label": "Checkpoint version", "kind": "int", "default": 1, "min": 1},
    {"name": "checkpoint", "label": "Checkpoint", "kind": "path", "default": None,
     "doc": "Resume or start from this checkpoint"},
    {"name": "start_from_scratch", "label": "Start from scratch", "kind": "bool", "default": False},
    {"name": "max_steps", "label": "Max steps", "kind": "int", "default": None, "min": 0},
    {"name": "compile", "label": "torch.compile", "kind": "bool", "default": False},
]


def spec_fields() -> list:
    """The launch switches, static so a slow registry import cannot block the queue dialog."""
    return list(_EXTRA_FIELDS) + [dict(field) for field in _TRAINING_FIELDS]


