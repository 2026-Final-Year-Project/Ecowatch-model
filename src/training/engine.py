"""Single-epoch training and evaluation loop."""

from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader

from src.evaluation.metrics import BCEDiceLoss, dice_score_from_logits, iou_score_from_logits


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device | str,
    training: bool,
    optimizer: torch.optim.Optimizer | None = None,
    criterion: torch.nn.Module | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Run one train or evaluation epoch and aggregate segmentation metrics."""
    if training and optimizer is None:
        raise ValueError("An optimizer is required for a training epoch.")
    if len(loader) == 0:
        raise ValueError("Cannot run an epoch with an empty data loader.")

    device = torch.device(device)
    criterion = criterion or BCEDiceLoss()
    model.train(training)
    total_loss = total_dice = total_iou = 0.0
    gradient_context = nullcontext if training else torch.no_grad

    with gradient_context():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, masks)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item()
            total_dice += dice_score_from_logits(logits.detach(), masks, threshold).item()
            total_iou += iou_score_from_logits(logits.detach(), masks, threshold).item()

    batches = len(loader)
    return {
        "loss": total_loss / batches,
        "dice": total_dice / batches,
        "iou": total_iou / batches,
    }
