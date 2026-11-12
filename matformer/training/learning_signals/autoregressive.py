from matformer.matformer_registry import registry
from matformer.training import LearningSignal

@registry.register("learning_signal", "autoregressive", "default")
class AutoregressiveLearningSignal(LearningSignal):
    def __init__(self, wrapper=None, config=None, **kwargs):
        super().__init__(wrapper=wrapper, config=config, **kwargs)
        self.loss_function = registry.create("loss", "cross_entropy_loss")
        
    def prepare_inputs(self, batch):
        return batch, {}

    def model_kwargs(self, metadata):
        return {"is_causal": True} # Useful for scenario when is_causal is not in the model's config (ex. hybrid model)

    def select_targets(self, predictions, batch, metadata):
        targets = batch
        predictions_flat, targets_flat, base_mask = self._flatten(predictions, targets)
        # Shift: predict token t+1 at position t; exclude positions where t+1 is padding.
        mask = base_mask[1:]
        return predictions_flat[:-1][mask], targets_flat[1:][mask]
