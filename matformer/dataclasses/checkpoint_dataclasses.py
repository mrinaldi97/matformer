"""The `checkpoint` section: which saved model a run loads, declared in the config instead of
typed on the command line.

    checkpoint:
      - name: finale
        path: "$CHECKPOINT_ASSETS/FINALE_32768.ckpt"

It is the YAML answer to `run_task.py --model`, and it does one thing the flag cannot: one entry
can stand for a *series* of models, and each of them becomes a run.

    checkpoint:
      - name: probes
        path: "$CHECKPOINT_ASSETS/sequence_classification/ironita"   # a folder of checkpoints
        every_n_steps: 500        # ... one run per 500 steps of it
      - name: donors
        path: "$CHECKPOINT_ASSETS/a.ckpt | $CHECKPOINT_ASSETS/b.ckpt" # ... and both of these

A `path` that names a folder is read as a checkpoint series (see
CheckpointDirectoryManager.select), a `path` holding pipes is a sweep dimension like any other
field, and the two combine: every folder, at every cadence, crossed with every other swept field.
Anything the loaders already accept -- one .ckpt, a `hf://repo`, a safetensors folder -- still comes
back unchanged, so a `checkpoint` entry never guesses at what a path means.

Each run of a sweep gets its own name from the point that produced it, so evaluating a dozen
checkpoints writes a dozen sets of results rather than one overwritten file (see
matformer/sweeps.py).
"""
from dataclasses import dataclass
from typing import ClassVar, List, Optional

from matformer.dataclasses.matformer_config import MatformerMultiConfig
from matformer.matformer_registry import registry


@registry.register_dataclass("checkpoint", "matformer", default=True)
@dataclass
class CheckpointConfig(MatformerMultiConfig):
    path: Optional[str] = None            # one checkpoint, a folder of them, a hf:// repo, or a|b|c
    model_name: Optional[str] = None      # inside a folder, the run whose checkpoints to pick
    every_n_steps: int = 0                # 0: the path alone; N: one run per checkpoint at *N steps
    versions: Optional[List[int]] = None  # checkpoint versions to consider (default: the newest)
    last: int = 0                         # keep only the N newest of what the rule selected

    def expand(self, path=None) -> list:
        """Every concrete checkpoint this entry stands for. `path` overrides the declared one, so a
        sweep can expand each of a `a | b | c` declaration in turn; a list of paths is every one of
        them, which is how a folder of models and a list of files are the same declaration."""
        from matformer.training.ckpt_manager import CheckpointDirectoryManager
        path = path if path is not None else self.path
        if not path:
            return []
        if isinstance(path, (list, tuple)):
            return [one for item in path for one in self.expand(item)]
        return CheckpointDirectoryManager.select(str(path), model_name=self.model_name,
                                                 every_n_steps=self.every_n_steps,
                                                 versions=self.versions, last=self.last)
