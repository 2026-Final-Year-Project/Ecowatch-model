"""Losses and overlap metrics for binary semantic segmentation."""

import torch
from torch import nn


def _flatten_predictions(logits: torch.Tensor, threshold: float) -> torch.Tensor:
    return (torch.sigmoid(logits) >= threshold).float().flatten(1)


def dice_score_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    epsilon: float = 1e-7,
) -> torch.Tensor:
    """Return mean Dice overlap for a batch of binary logits."""
    predictions = _flatten_predictions(logits, threshold)
    targets = targets.flatten(1)
    intersection = (predictions * targets).sum(dim=1)
    denominator = predictions.sum(dim=1) + targets.sum(dim=1)
    return ((2 * intersection + epsilon) / (denominator + epsilon)).mean()


def iou_score_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    epsilon: float = 1e-7,
) -> torch.Tensor:
    """Return mean intersection-over-union for a batch of binary logits."""
    predictions = _flatten_predictions(logits, threshold)
    targets = targets.flatten(1)
    intersection = (predictions * targets).sum(dim=1)
    union = predictions.sum(dim=1) + targets.sum(dim=1) - intersection
    return ((intersection + epsilon) / (union + epsilon)).mean()


class BCEDiceLoss(nn.Module):
    """Weighted binary cross-entropy and soft Dice loss."""

    def __init__(self, bce_weight: float = 0.5, epsilon: float = 1e-7):
        super().__init__()
        if not 0 <= bce_weight <= 1:
            raise ValueError("bce_weight must be between zero and one.")
        self.bce_weight = bce_weight
        self.epsilon = epsilon
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        probabilities = torch.sigmoid(logits).flatten(1)
        targets = targets.flatten(1)
        intersection = (probabilities * targets).sum(dim=1)
        overlap = (2 * intersection + self.epsilon) / (
            probabilities.sum(dim=1) + targets.sum(dim=1) + self.epsilon
        )
        dice_loss = 1 - overlap.mean()
        return self.bce_weight * bce + (1 - self.bce_weight) * dice_loss
