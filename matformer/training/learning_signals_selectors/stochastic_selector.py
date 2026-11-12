from matformer.matformer_registry import registry
from matformer.training.learning_signals_selectors import LearningSignalSelector
import torch
@registry.register("learning_signal_selector", "stochastic", "default")
class StochasticSelector(LearningSignalSelector):
    """
    Randomly picks one signal per step with configurable weights.
    Supports a linear curriculum with a optional anti-forgetting final phase.

    selector_config keys:
        weights (List[float]): sampling weights, one per signal. Default: uniform.
        curriculum (bool): linearly shift weights during training. Default: False.
        weights_start (List[float]): weights at the start of training.
        weights_end (List[float]): weights at the end of the curriculum phase.
        final_steps (int): number of terminal steps where curriculum is suspended.
        final_weights (List[float]): fixed weights applied during the final phase,
            independently of where the curriculum ended up. Intended to keep all
            objectives active and prevent catastrophic forgetting of any one signal.
    """
    def populate_learning_signals(self):
        for signal_type in self.configs.training.learning_signals:
            self.learning_signals.append(self._create_signal(signal_type))

    def pick(self, batch, batch_idx):
        weights = self._resolve_weights()
        self.wrapper.log('train/signal_weight_0', weights[0], prog_bar=False)
        idx = torch.multinomial(torch.tensor(weights, dtype=torch.float), 1).item()
        return [self.learning_signals[idx]]

    def _resolve_weights(self):
        n = len(self.learning_signals)
        weights = self.selector_config.get('weights', [1.0 / n] * n)

        if not self.selector_config.get('curriculum', False):
            return weights

        total_steps = self.configs.training.total_steps
        final_steps = self.selector_config.get('final_steps', 0)
        current_step = self.wrapper.global_step
        curriculum_steps = max(1, total_steps - final_steps)

        if current_step >= curriculum_steps:
            # Final phase: curriculum is suspended; use fixed weights to prevent catastrophic forgetting.
            return self.selector_config.get('final_weights', weights)

        # Curriculum phase: linear interpolation between start and end weights.
        progress = current_step / curriculum_steps
        start = self.selector_config.get('weights_start', weights)
        end = self.selector_config.get('weights_end', weights)
        return [s + progress * (e - s) for s, e in zip(start, end)]