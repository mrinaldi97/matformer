from abc import ABC, abstractmethod
from matformer.dataclasses.tensors_dataclasses import TensorDC, NormalTensor, PaddedTensor, UnpaddedTensor
import torch

class LearningSignal(ABC):
    """
    LearningSignal is the part of the training step that handles parts such as input preparation (ex. masking in a masked model),
    the selection of targets for the loss and the loss function itself.
    Users can easily write new LearningSignal for their purposes and they will be called during the training step. 
    Each LearningSignal should use the [[MatformerRegistry]] in order to be seen by the training step code.
    """    
    def __init__(self, wrapper=None, config=None, **kwargs) -> None:
        self.wrapper = wrapper or config
        if self.wrapper is not None:
            self.configs = getattr(self.wrapper, "configs", None)
            self.log = getattr(self.wrapper, "log", None)


    @abstractmethod
    def prepare_inputs(self, batch) -> tuple:
        """Returns (input_sequence, metadata). metadata is a dict carrying anything select_targets will need"""

    def model_kwargs(self, metadata: dict) -> dict:
        """Extra kwargs forwarded to the model forward call (e.g. is_causal)."""
        return {}

    @abstractmethod
    def select_targets(self, predictions, batch, metadata) -> tuple:
        """Returns (predictions, targets), flat tensors ready for self.loss_function."""

    def post_loss(self, loss, predictions, targets) -> None:
        """
        Optional hook, useful for things such as logging (ex.accuracy).
        predictions and targets are the already-selected flat tensors from select_targets.
        """

    def _flatten(self, predictions, targets, masks=None):
        """
        Flattens predictions (B,S,H)->(B*S,H) and targets (B,S)->(B*S,),
        then returns the AND of the base padding mask with any extra masks provided.
        When predictions is an UnpaddedTensor everything is already flat
        and all positions are valid (no padding to mask).
        """
        pad_id = getattr(self.configs.tokenizer, "pad_token_id", 0)
        if pad_id is None:
            pad_id = 0
        if masks is None:
            masks = []
        is_unpadded = isinstance(predictions, UnpaddedTensor)
        if is_unpadded:
            predictions_flat = predictions.tensor
            targets_flat = targets.tensor
            base_mask = torch.ones_like(targets_flat, dtype=torch.bool)
            masks_flat = masks
        else:
            predictions_flat = predictions.tensor.view(-1, predictions.tensor.size(-1))
            targets_flat = targets.tensor.view(-1)
            base_mask = (targets_flat != pad_id)
            masks_flat = [m.view(-1) for m in masks]
        combined_mask = base_mask
        for m in masks_flat:
            combined_mask = combined_mask & m
        return predictions_flat, targets_flat, combined_mask
