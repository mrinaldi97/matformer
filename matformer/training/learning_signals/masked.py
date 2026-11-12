from matformer.matformer_registry import registry
from matformer.training import LearningSignal
from matformer.utils.masked_models import Maskerator
@registry.register("learning_signal", "masked","default")
class MaskedLearningSignal(LearningSignal):
    def __init__(self, wrapper=None, config=None, **kwargs):
        super().__init__(wrapper=wrapper, config=config, **kwargs)
        assert 'masking' in self.configs.list_configs(),"You asked for a MaskedLearningSignal without providing a 'masking' section in the config"
        assert 'tokenizer' in self.configs.list_configs(),"Cannot initialize a MaskedLearningSignal without a 'tokenizer' section in the config with a mask_token_id"
        assert hasattr(self.configs.tokenizer,"mask_token_id"),"Cannot initialize a MaskedLearningSignal without a 'tokenizer' section in the config with a mask_token_id"
        self.maskerator = Maskerator(
            mask_token=self.configs.tokenizer.mask_token_id,
            substitution_rate=self.configs.masking.masked_substitution_rate,
            pad_token_id=self.configs.tokenizer.pad_token_id,
            cloze_prob=self.configs.masking.cloze_probability,
            random_prob=self.configs.masking.random_probability,
            same_prob=self.configs.masking.same_probability,
            vocab_size=self.configs.model.vocab_size
        )
        self.loss_function = registry.create("loss", "cross_entropy_loss")

    def prepare_inputs(self, batch):
        if hasattr(self.wrapper, "device"):
            batch = batch.to(self.wrapper.device)
        input_sequence, masking_ratio = self.maskerator(batch)
        return input_sequence, {'cloze_mask': input_sequence.following_attributes['cloze_mask']} # Useful for scenario when is_causal is not in the model's config (ex. hybrid model)

    def model_kwargs(self, metadata):
        return {"is_causal": False}

    def select_targets(self, predictions, batch, metadata):
        targets = batch
        predictions_flat, targets_flat, mask = self._flatten(
            predictions, targets, [metadata['cloze_mask']]
        )
        return predictions_flat[mask], targets_flat[mask]

    def post_loss(self, loss, predictions, targets):
        acc = (predictions.argmax(dim=-1) == targets).float().mean()
        self.log("train/accuracy", acc, prog_bar=True, on_step=True, on_epoch=True,
                 batch_size=self.configs.training.batch_size)