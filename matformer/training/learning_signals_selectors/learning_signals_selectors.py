from matformer.matformer_registry import registry
from abc import ABC, abstractmethod

class LearningSignalSelector(ABC):
    """
    Decides which LearningSignal(s) are active at each training step.
    In the most common and simple case, this class is a no-op: there is just
    one active LearningSingle (ex. autoregressive). However, this class adds flexibility
    for complex scenario such as hybrid autoregressive + masked models.
    """
    def __init__(self, wrapper):
        self.learning_signals = []
        self.wrapper = wrapper
        self.configs = wrapper.configs
        self.selector_config = getattr(self.configs.training, 'selector_config', {})
        self.populate_learning_signals()
        if not isinstance(self.configs.training.learning_signals,list):
            self.configs.training.learning_signals=[self.configs.training.learning_signals]
    @abstractmethod
    def populate_learning_signals(self):
        pass

    @abstractmethod
    def pick(self, batch, batch_idx) -> list:
        """Returns the list of LearningSignals active for this training step."""

    def pre_step(self, batch) -> None:
        """
        Optional hook called once per step before the signal loop.
        """

    def _create_signal(self, signal_type):
        return registry.create("learning_signal", signal_type, wrapper=self.wrapper)

@registry.register("learning_signal_selector", "single", "default")
class SingleSelector(LearningSignalSelector):

    def populate_learning_signals(self):
        self.learning_signals.append(
            self._create_signal(self.configs.training.learning_signals[0])
        )

    def pick(self, batch, batch_idx):
        return [self.learning_signals[0]]


@registry.register("learning_signal_selector", "multitask", "default")
class MultitaskSelector(LearningSignalSelector):
    """Activates all signals simultaneously; losses are summed in the training step."""

    def populate_learning_signals(self):
        for signal_type in self.configs.training.learning_signals:
            self.learning_signals.append(self._create_signal(signal_type))

    def pick(self, batch, batch_idx):
        return self.learning_signals
