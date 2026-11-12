"""Small training-side helpers that don't belong to the trainer class itself."""
import torch


def detect_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'
