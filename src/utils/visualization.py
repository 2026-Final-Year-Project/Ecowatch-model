"""Visualize dataset samples, training history, and model predictions."""

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset


def sentinel2_rgb(image: torch.Tensor) -> np.ndarray:
    """Convert a channels-first normalized image to the notebook's RGB composite."""
    if image.ndim != 3 or image.shape[0] < 3:
        raise ValueError("Expected an image with at least three spectral bands.")
    return np.clip(image[[2, 1, 0]].permute(1, 2, 0).cpu().numpy(), 0, 1)


def plot_sample(dataset: Dataset, index: int = 0) -> plt.Figure:
    """Plot a normalized satellite image beside its binary mask."""
    image, mask = dataset[index]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(sentinel2_rgb(image))
    axes[0].set_title("Satellite image")
    axes[1].imshow(mask.squeeze().cpu(), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground-truth mask")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    return figure


def plot_training_history(history: Sequence[dict[str, float | int]]) -> plt.Figure:
    """Plot train/validation loss and overlap metrics."""
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="BCEDiceLoss")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_dice"] for row in history], label="train Dice")
    axes[1].plot(epochs, [row["val_dice"] for row in history], label="validation Dice")
    axes[1].plot(epochs, [row["val_iou"] for row in history], label="validation IoU")
    axes[1].set(title="Segmentation metrics", xlabel="Epoch", ylabel="Score")
    axes[1].legend()
    figure.tight_layout()
    return figure


def plot_predictions(
    model: torch.nn.Module,
    dataset: Dataset,
    device: torch.device | str = "cpu",
    count: int = 4,
    threshold: float = 0.5,
    seed: int | None = None,
) -> plt.Figure:
    """Plot satellite, label, and prediction panels for random dataset samples."""
    if len(dataset) == 0:
        raise ValueError("Cannot visualize an empty dataset.")
    device = torch.device(device)
    generator = np.random.default_rng(seed)
    indices = generator.choice(len(dataset), size=min(count, len(dataset)), replace=False)
    figure, axes = plt.subplots(len(indices), 3, figsize=(12, 4 * len(indices)), squeeze=False)
    model.eval()
    for row, index in enumerate(indices):
        image, target = dataset[int(index)]
        with torch.inference_mode():
            probability = torch.sigmoid(model(image.unsqueeze(0).to(device))).squeeze().cpu().numpy()
        axes[row, 0].imshow(sentinel2_rgb(image))
        axes[row, 0].set_title("Satellite image")
        axes[row, 1].imshow(target.squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("Ground truth")
        axes[row, 2].imshow(probability >= threshold, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title(f"Prediction (threshold {threshold})")
        for axis in axes[row]:
            axis.axis("off")
    figure.tight_layout()
    return figure
