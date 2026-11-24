import torch
import torch.nn.functional as F
from matformer.matformer_registry import registry
from matformer.training import LearningSignal
from matformer.dataclasses.data_dataclasses import IGNORE_INDEX
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
    """Sentence-level classification: one label per sequence.
    Targets are None on unlabelled data, so the same signal also drives plain inference."""
    def select_targets(self, predictions, batch, metadata):
        if hasattr(predictions,'tensor'):
            predictions=predictions.tensor
        return predictions, batch.attributes.get('labels')

@registry.register("learning_signal", "token_classification", "default")
class TokenClassificationLearningSignal(BaseClassificationLearningSignal):
    """Token-level classification: one label per token.

    Labels travel inside the sequence's following_attributes, so they have already been padded or
    unpadded exactly like the tokens they belong to. Positions marked with IGNORE_INDEX (subword
    continuations, special tokens, padding) are dropped before the loss: masking on them, and not
    on the pad id, is what keeps class 0 a trainable class.
    """
    def select_targets(self, predictions, batch, metadata):
        if isinstance(predictions, TensorDC):
            predictions = predictions.tensor
        targets = batch.following_attributes.get('labels')
        if targets is None:      # unlabelled data: inference only, nothing to select
            return predictions, None
        if isinstance(targets, TensorDC):
            targets = targets.tensor
        if predictions.dim() == 3:  # padded layout: (B,S,C) against (B,S) labels
            predictions = predictions.reshape(-1, predictions.shape[-1])
            targets = targets.reshape(-1)
        mask = targets != IGNORE_INDEX
        return predictions[mask], targets[mask]

    def loss_function(self, predictions, targets):
        w = getattr(self.configs.training, 'class_weights', None)
        weight = torch.tensor(w, device=predictions.device, dtype=predictions.dtype) if w else None
        return F.cross_entropy(predictions, targets, weight=weight, ignore_index=IGNORE_INDEX)
