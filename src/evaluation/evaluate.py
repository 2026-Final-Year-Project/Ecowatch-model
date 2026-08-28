"""Evaluate a saved U-Net checkpoint on a fixed dataset split."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.models.unet import UNet
from src.training.engine import run_epoch


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    loader: DataLoader,
    device: torch.device | str = "cpu",
    threshold: float = 0.5,
    output_path: str | Path | None = None,
) -> dict[str, float | int]:
    """Load a checkpoint and compute loss, Dice, and IoU without gradients."""
    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model = UNet(
        in_channels=config.get("in_channels", 13),
        features=tuple(config.get("features", (32, 64, 128, 256))),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = run_epoch(model, loader, device=device, training=False, threshold=threshold)
    results = {
        **metrics,
        "best_epoch": checkpoint.get("epoch", -1),
        "best_validation_dice": checkpoint.get("val_dice", float("nan")),
        "threshold": threshold,
    }
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))
    return results
