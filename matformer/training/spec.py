"""Compatibility import: the launch switches live in matformer.spec_fields so importing them
does not drag in the training package's learning-signal imports."""
from matformer.spec_fields import CLI_TO_TRAINING_FIELD, DEVICES, RESET_CHOICES, spec_fields

__all__ = ["CLI_TO_TRAINING_FIELD", "DEVICES", "RESET_CHOICES", "spec_fields"]
