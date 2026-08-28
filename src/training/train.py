"""High-level U-Net training workflow."""

import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .engine import run_epoch


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    output_dir: str | Path,
    epochs: int = 30,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    threshold: float = 0.5,
    device: torch.device | str | None = None,
) -> tuple[Path, list[dict[str, float | int]]]:
    """Train, select the best validation checkpoint, and save metric history."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    history: list[dict[str, float | int]] = []
    best_validation_dice = -float("inf")
    features = tuple(block.layers[0].out_channels for block in model.down_blocks)
    in_channels = model.down_blocks[0].layers[0].in_channels

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, training=True, optimizer=optimizer, threshold=threshold)
        validation_metrics = run_epoch(model, validation_loader, device, training=False, threshold=threshold)
        scheduler.step(validation_metrics["dice"])
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(row)

        if validation_metrics["dice"] > best_validation_dice:
            best_validation_dice = validation_metrics["dice"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": best_validation_dice,
                    "config": {
                        "in_channels": in_channels,
                        "features": features,
                        "threshold": threshold,
                    },
                },
                checkpoint_path,
            )

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss {train_metrics['loss']:.4f}, Dice {train_metrics['dice']:.4f} | "
            f"val loss {validation_metrics['loss']:.4f}, "
            f"Dice {validation_metrics['dice']:.4f}, IoU {validation_metrics['iou']:.4f}"
        )

    history_path = output_dir / "training_history.csv"
    with history_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    return checkpoint_path, history
