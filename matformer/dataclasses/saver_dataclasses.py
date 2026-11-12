from dataclasses import dataclass, asdict
from typing import ClassVar, Optional
from matformer.dataclasses.matformer_config import MatformerMultiConfig
from matformer.matformer_registry import registry


@dataclass
class SaverConfig(MatformerMultiConfig):
    def build(self):
        from matformer.utils.load_config import get_saver_class
        cls    = get_saver_class(self.saver_class)
        kwargs = {k: v for k, v in asdict(self).items() if k != "saver_class"}
        return cls(**kwargs)

@registry.register_dataclass("saver", "jsonl", default=True)
@dataclass
class JSONLSaverConfig(SaverConfig):
    filename:     str  = "output.jsonl"
    summary_file: str = "summary.jsonl"
    mode:       str = "a"
    summary_mode: Optional[str] = None

@registry.register_dataclass("saver", "NullSaver", default=True)
@dataclass
class NullSaverConfig(SaverConfig):
    pass

@registry.register_dataclass("saver", "memory")
@dataclass
class MemorySaverConfig(SaverConfig):
    key: Optional[str] = None   # bucket in the in-process store; defaults to the section's name
