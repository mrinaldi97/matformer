"""Loss functions for classification tasks."""
import torch
import torch.nn.functional as F


def register_loss_functions(registry):
    
    @registry.register("loss_fn", "cross_entropy")
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
    
    @registry.register("loss_fn", "focal")
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
    
    @registry.register("loss_fn", "dice")
    def dice_loss(logits, labels, config, ignore_index=-100):
        """
        Dice loss for segmentation/token classification.
        Config: smooth (float)
        """
        smooth = config.get('dice_smooth', 1.0)
        num_classes = logits.shape[-1]
        
        # Get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # One-hot encode labels
        labels_one_hot = F.one_hot(labels.clamp(min=0), num_classes).float()
        
        # Handle ignore_index
        if ignore_index is not None:
            mask = (labels != ignore_index).float().unsqueeze(-1)
            probs = probs * mask
            labels_one_hot = labels_one_hot * mask
        
        # Compute dice coefficient per class
        intersection = (probs * labels_one_hot).sum(dim=0)
        cardinality = (probs + labels_one_hot).sum(dim=0)
        dice_score = (2.0 * intersection + smooth) / (cardinality + smooth)
        
        return 1.0 - dice_score.mean()
    
    @registry.register("loss_fn", "mse")
    def mse_loss(logits, labels, config, ignore_index=-100):
        """
        MSE loss for regression tasks.
        Config: none
        """
        # Assume single output for regression
        if logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        
        loss = F.mse_loss(logits, labels.float(), reduction='none')
        
        # Handle ignore_index
        if ignore_index is not None:
            mask = labels != ignore_index
            if mask.any():
                return loss[mask].mean()
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        return loss.mean()
    
    @registry.register("loss_fn", "bce")
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