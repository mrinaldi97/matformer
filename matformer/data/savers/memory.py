import torch.distributed as dist
from matformer.data.savers import MatformerDataSaver
from matformer.matformer_registry import registry

SUMMARIES = {}
"""Process-global store of eval summaries: key -> list of summary dicts (one per save_summary call).
Lets programmatic drivers (experiments/) collect summaries without touching the filesystem."""


def pop_summaries(key):
    """Return (and forget) every summary accumulated under `key`."""
    return SUMMARIES.pop(key, [])


@registry.register("saver", "memory", "default")
class MemorySaver(MatformerDataSaver):
    """Saver that never writes files: summaries land in the module-level SUMMARIES store
    (keyed by config.key), batch records are dropped. Intended for in-process experiments
    that need the metrics back as Python objects, e.g. MDL coding inside a K-sweep."""

    def __init__(self, config):
        self.config = config
        self.key = config.key or config.name

    def save_batch(self, records: list, rank: int = 0) -> None:
        pass

    def save_summary(self, summary, rank: int = 0) -> None:
        if rank != 0:
            return
        SUMMARIES.setdefault(self.key, []).append(dict(summary))

    def end(self) -> None:
        pass
