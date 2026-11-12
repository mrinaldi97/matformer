from matformer.matformer_registry import registry
from matformer.training import LearningSignal
import torch.nn.functional as F
@registry.register("learning_signal", "multiprobe_ls", "default")
class MultiProbeLearningSignal(LearningSignal):
    def prepare_inputs(self, batch):
        return batch, {}

    def model_kwargs(self, metadata):
        return {}

    def select_targets(self, predictions, batch, metadata):
        assert isinstance(predictions,dict)
        return predictions, batch.attributes['labels']

    def loss_function(self, predictions, targets):
        get_t = (lambda n: targets[n]) if isinstance(targets, dict) else (lambda n: targets)   
        self.losses={}
        for n, logits in predictions.items():
              self.losses[n]=F.cross_entropy(logits, get_t(n))
        return sum(self.losses.values())
        #return sum(F.cross_entropy(logits, get_t(n)) for n, logits in predictions.items())

    def post_loss(self, loss, predictions, targets):
        get_t = (lambda n: targets[n]) if isinstance(targets, dict) else (lambda n: targets)
        for name, logits in predictions.items():
            t = get_t(name)
            self.log(f'train/{name}_loss', self.losses[name].float(), prog_bar=False)
            self.log(
                f'train/{name}_accuracy',
                (logits.argmax(-1) == t).float().mean(),
                prog_bar=False
            )
