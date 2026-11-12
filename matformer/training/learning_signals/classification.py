import torch
import torch.nn.functional as F
from matformer.matformer_registry import registry
from matformer.training import LearningSignal
from matformer.dataclasses.tensors_dataclasses import TensorDC, NormalTensor, PaddedTensor, UnpaddedTensor
class BaseClassificationLearningSignal(LearningSignal):
    def prepare_inputs(self, batch):
        return batch, {}

    def model_kwargs(self, metadata):
        return {}

    def loss_function(self, predictions, targets):
        w = getattr(self.configs.training, 'class_weights', None)
        weight = torch.tensor(w, device=predictions.device) if w else None
        return F.cross_entropy(predictions, targets, weight=weight)

    def post_loss(self, loss, predictions, targets):
        accuracy = (predictions.argmax(-1) == targets).float().mean()
        self.log('train/accuracy', accuracy, prog_bar=True)

@registry.register("learning_signal", "sequence_classification", "default")
class ClassificationLearningSignal(BaseClassificationLearningSignal):
    """Sentence-level classification: one label per sequence"""
    def select_targets(self, predictions, batch, metadata):
        if hasattr(predictions,'tensor'):
            predictions=predictions.tensor
        return predictions, batch.attributes['labels']

@registry.register("learning_signal", "token_classification", "default")
class TokenClassificationLearningSignal(BaseClassificationLearningSignal):
    """Token-level classification: one label per token"""
    def select_targets(self, predictions, batch, metadata):
        preds, targets, mask = self._flatten(
            NormalTensor(tensor=predictions), 
            NormalTensor(tensor=batch.following_attributes['labels'])
        )
        return preds[mask], targets[mask]
