"""Loss functions for classification tasks."""
from matformer.matformer_registry import registry
import torch
import torch.nn as nn
import torch.nn.functional as F


@registry.register(
    "loss_fn",
    "cross_entropy",
    "torch",
    requires=["torch"],
    priority=10
)
class CrossEntropyLoss(nn.Module):
    """
    Standard cross-entropy loss.
    Args at init: class_weights, label_smoothing, ignore_index
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._args = args
        self._kwargs = kwargs

    def forward(self, logits, labels, **extra_kwargs):
        kw = dict(self._kwargs)
        kw.update(extra_kwargs)
        
        class_weights = kw.get('class_weights', None)
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, device=logits.device, dtype=logits.dtype)
        
        label_smoothing = kw.get('label_smoothing', 0.0)
        ignore_index = kw.get('ignore_index', -100)
        
        return F.cross_entropy(
            logits, labels,
            weight=class_weights,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )

@registry.register(
    "loss_fn",
    "focal",
    "torch",
    requires=["torch"],
    priority=10
)
class FocalLoss(nn.Module):
    """
    Focal loss for severe class imbalance.
    Args at init: focal_alpha, focal_gamma, class_weights, ignore_index
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._args = args
        self._kwargs = kwargs

    def forward(self, logits, labels, **extra_kwargs):
        kw = dict(self._kwargs)
        kw.update(extra_kwargs)
        
        alpha = kw.get('focal_alpha', 0.25)
        gamma = kw.get('focal_gamma', 2.0)
        ignore_index = kw.get('ignore_index', -100)
        class_weights = kw.get('class_weights', None)
        
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, device=logits.device, dtype=logits.dtype)
        
        # Compute base CE loss
        ce_loss = F.cross_entropy(
            logits, labels, 
            reduction='none', 
            weight=class_weights,
            ignore_index=ignore_index
        )
        
        # Apply focal term
        pt = torch.exp(-ce_loss)
        focal = alpha * (1 - pt) ** gamma * ce_loss
        
        # Handle ignore_index
        if ignore_index is not None:
            mask = labels != ignore_index
            if mask.any():
                return focal[mask].mean()
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        return focal.mean()


@registry.register(
    "loss_fn",
    "bce",
    "torch",
    requires=["torch"],
    priority=10
)
class BCELoss(nn.Module):
    """
    Binary cross-entropy for binary classification.
    Args at init: pos_weight, ignore_index
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._args = args
        self._kwargs = kwargs

    def forward(self, logits, labels, **extra_kwargs):
        kw = dict(self._kwargs)
        kw.update(extra_kwargs)
        
        pos_weight = kw.get('pos_weight', None)
        ignore_index = kw.get('ignore_index', -100)
        
        if pos_weight is not None:
            pos_weight = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
        
        # Assume single output or two-class logits
        if logits.shape[-1] == 2:
            logits = logits[:, 1]
        else:
            logits = logits.squeeze(-1)
        
        loss = F.binary_cross_entropy_with_logits(
            logits, 
            labels.float(),
            pos_weight=pos_weight,
            reduction='none'
        )
        
        # Handle ignore_index
        if ignore_index is not None:
            mask = labels != ignore_index
            if mask.any():
                return loss[mask].mean()
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        return loss.mean()