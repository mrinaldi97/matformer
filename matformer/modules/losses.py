"""Loss functions for classification tasks."""
import torch
import torch.nn.functional as F


def register_loss_functions(registry):
    """Register all available loss functions."""
    
    @registry.register("loss_fn", "cross_entropy", "torch", requires=["torch"], priority=10)
    def cross_entropy_loss(logits, labels, config, ignore_index=-100):
        """
        Standard cross-entropy loss.
        Config: class_weights (list), label_smoothing (float)
        """
        class_weights = config.get('class_weights', None)
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, device=logits.device, dtype=logits.dtype)
        
        label_smoothing = config.get('label_smoothing', 0.0)
        
        return F.cross_entropy(
            logits, 
            labels,
            weight=class_weights,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )
    
    @registry.register("loss_fn", "focal", "torch", requires=["torch"], priority=10)
    def focal_loss(logits, labels, config, ignore_index=-100):
        """
        Focal loss for severe class imbalance.
        Config: focal_alpha (float), focal_gamma (float), class_weights (list)
        """
        alpha = config.get('focal_alpha', 0.25)
        gamma = config.get('focal_gamma', 2.0)
        class_weights = config.get('class_weights', None)
        
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
    
    @registry.register("loss_fn", "bce", "torch", requires=["torch"], priority=10)
    def bce_loss(logits, labels, config, ignore_index=-100):
        """
        Binary cross-entropy for binary classification.
        Config: pos_weight (float)
        """
        pos_weight = config.get('pos_weight', None)
        if pos_weight is not None:
            pos_weight = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
        
        # Assume single output or two-class logits
        if logits.shape[-1] == 2:
            logits = logits[:, 1]  # Take positive class logit
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