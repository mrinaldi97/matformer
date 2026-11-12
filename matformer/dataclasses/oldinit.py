# matformer/dataclasses/__init__.py

import importlib
import pkgutil
from pathlib import Path

from .matformer_config import MatformerConfig, MatformerMultiConfig
from .tensors_dataclasses        import TensorDC, PaddedTensor, UnpaddedTensor, NormalTensor
from .model_dataclasses          import LayerConfig, TokenizerConfig, MaskingConfig, ModelConfig, SequenceClassificationConfig, TokenClassificationConfig
from .data_dataclasses           import DataConfig
from .training_dataclasses       import TrainingConfig
from .tasks_dataclasses          import TaskConfig, GenerationTaskConfig
from .saver_dataclasses          import SaverConfig, JSONLSaverConfig, NullSaverConfig
# Auto-import: any module added to this directory self-registers automatically
for _mod in pkgutil.iter_modules([str(Path(__file__).parent)]):
    if not _mod.name.startswith("_"):
        importlib.import_module(f".{_mod.name}", package=__name__)
