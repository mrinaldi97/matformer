from matformer.matformer_registry import registry
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ClassificationLossBase(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._kwargs = kwargs
        self._cached_weights = {}

    def _merge_kwargs(self, extra_kwargs):
        return {**self._kwargs, **extra_kwargs}

    def _get_class_weights(self, weights, device, dtype):
        if weights is None:
            return None

        key = (tuple(weights) if isinstance(weights, list) else weights, device, dtype)
        if key not in self._cached_weights:
            self._cached_weights[key] = torch.tensor(
                weights, device=device, dtype=dtype
            )
        return self._cached_weights[key]

    def _apply_ignore_mask(self, loss, labels, ignore_index):
        """Apply ignore_index masking to unreduced loss."""
        if ignore_index is None:
            return loss.mean()

        mask = labels != ignore_index
        if not mask.any():
            return torch.tensor(0.0, device=loss.device, requires_grad=True)
        return loss[mask].mean()


@registry.register("loss_fn", "cross_entropy", "torch", requires=["torch"], priority=10)
class CrossEntropyLoss(_ClassificationLossBase):
    """
    Args: class_weights, label_smoothing, ignore_index
    """

    def forward(self, logits, labels, **extra_kwargs):
        kw = self._merge_kwargs(extra_kwargs)

        weights = self._get_class_weights(
            kw.get("class_weights"), logits.device, logits.dtype
        )

        return F.cross_entropy(
            logits,
            labels,
            weight=weights,
            ignore_index=kw.get("ignore_index", -100),
            label_smoothing=kw.get("label_smoothing", 0.0),
        )


@registry.register("loss_fn", "focal", "torch", requires=["torch"], priority=10)
class FocalLoss(_ClassificationLossBase):
    """
    Args: focal_alpha, focal_gamma, class_weights, ignore_index
    """

    def forward(self, logits, labels, **extra_kwargs):
        kw = self._merge_kwargs(extra_kwargs)

        alpha = kw.get("focal_alpha", 0.25)
        gamma = kw.get("focal_gamma", 2.0)
        ignore_index = kw.get("ignore_index", -100)

        weights = self._get_class_weights(
            kw.get("class_weights"), logits.device, logits.dtype
        )

        ce_loss = F.cross_entropy(logits, labels, weight=weights, reduction="none")

        pt = torch.exp(-ce_loss)
        focal = alpha * (1 - pt) ** gamma * ce_loss

        return self._apply_ignore_mask(focal, labels, ignore_index)


@registry.register("loss_fn", "bce", "torch", requires=["torch"], priority=10)
class BCELoss(_ClassificationLossBase):
    """
    Args: pos_weight, ignore_index
    """

    def forward(self, logits, labels, **extra_kwargs):
        kw = self._merge_kwargs(extra_kwargs)

        pos_weight = kw.get("pos_weight")
        if pos_weight is not None:
            pos_weight = torch.tensor(
                [pos_weight], device=logits.device, dtype=logits.dtype
            )

        # Normalize shape
        if logits.shape[-1] == 2:
            logits = logits[:, 1]
        else:
            logits = logits.squeeze(-1)

        if logits.shape != labels.shape:
            raise ValueError(
                f"Shape mismatch after normalization: "
                f"logits {logits.shape} vs labels {labels.shape}"
            )

        loss = F.binary_cross_entropy_with_logits(
            logits, labels.float(), pos_weight=pos_weight, reduction="none"
        )

        return self._apply_ignore_mask(loss, labels, kw.get("ignore_index", -100))


class _RegressionLossBase(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._kwargs = kwargs

    def _compute_loss(self, logits, labels):
        """Override in subclasses."""
        raise NotImplementedError

    def forward(self, logits, labels, **extra_kwargs):
        kw = {**self._kwargs, **extra_kwargs}

        if logits.shape[-1] == 1:
            logits = logits.squeeze(-1)

        if logits.shape != labels.shape:
            raise ValueError(
                f"Shape mismatch: logits {logits.shape} vs labels {labels.shape}"
            )

        # Use NaN masking for regression
        use_nan_masking = kw.get("use_nan_masking", False)

        loss = self._compute_loss(logits, labels.float())

        if use_nan_masking:
            mask = ~torch.isnan(labels)
            if mask.any():
                return loss[mask].mean()
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        return loss.mean()


@registry.register("loss_fn", "mse", "torch", requires=["torch"], priority=10)
class MSELoss(_RegressionLossBase):
    def _compute_loss(self, logits, labels):
        return F.mse_loss(logits, labels, reduction="none")


@registry.register("loss_fn", "mae", "torch", requires=["torch"], priority=10)
class MAELoss(_RegressionLossBase):
    def _compute_loss(self, logits, labels):
        return F.l1_loss(logits, labels, reduction="none")
