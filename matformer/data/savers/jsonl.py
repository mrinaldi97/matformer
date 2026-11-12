import torch.distributed as dist
from matformer.data.savers import MatformerDataSaver
from matformer.matformer_registry import registry
import torch
import numpy as np
def _to_serializable(obj):
        # tensors / ndarrays -> nested lists; dicts/lists/tuples recursed
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):        # np.float32, np.int64, ...
            return obj.item()
        if isinstance(obj, dict):
            return {k: str(_to_serializable(v)) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [_to_serializable(v) for v in obj]
        return obj
@registry.register("saver", "jsonl", "default")
class JSONLSaver(MatformerDataSaver):
    """Writes JSON-Lines using orjson bytes mode when available.

    Args:
        filename: output file path.
        mode: open mode; 'b' is appended automatically.
    """

    def __init__(self, config):
        self.config=config
        if config.mode is None:
            config.mode = 'a'
        try:
            import orjson
            self._dumps = orjson.dumps
        except ImportError:
            import json
            print("Install orjson for better performance.")
            self._dumps = lambda d: json.dumps(d, ensure_ascii=False).encode()
        if 'b' not in config.mode:
            config.mode += 'b'
        self.file = open(config.filename, config.mode)
        self.summary_file = None

    def save_batch(self, records: list[dict], rank: int = 0) -> None:
        if dist.is_available() and dist.is_initialized():
            gathered = [None] * dist.get_world_size() if rank == 0 else None
            dist.gather_object(records, gathered, dst=0)
            if rank != 0:
                return
            records = [r for batch in gathered for r in batch]
        for rec in records:
            self.file.write(self._dumps(_to_serializable(rec)) + b'\n')            
        self.file.flush()

    def save_summary(self, summary, rank: int = 0) -> None:
        if rank != 0:
            return
        if self.summary_file is None:
            mode = self.config.summary_mode or self.config.mode
            if 'b' not in mode:
                mode += 'b'
            self.summary_file = open(self.config.summary_file, mode)
        self.summary_file.write(self._dumps(_to_serializable(summary)) + b'\n')
        self.summary_file.flush()

    def end(self) -> None:
        self.file.close()
        if self.summary_file is not None:
            self.summary_file.close()
