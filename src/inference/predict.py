"""Run U-Net inference on already-loaded Sentinel-2 arrays."""

from pathlib import Path

import numpy as np
import torch

from src.data.preprocessing import normalize_bands
from src.models.unet import UNet


def load_model(checkpoint_path: str | Path, device: torch.device | str = "cpu") -> UNet:
    """Restore the best notebook-compatible U-Net checkpoint."""
    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model = UNet(
        in_channels=config.get("in_channels", 13),
        features=tuple(config.get("features", (32, 64, 128, 256))),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_mask(
    model: torch.nn.Module,
    image: np.ndarray,
    threshold: float = 0.5,
    device: torch.device | str = "cpu",
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the probability map and thresholded mask for one channels-first image."""
    if normalize:
        image = normalize_bands(image.astype(np.float32, copy=False))
    tensor = torch.from_numpy(image.astype(np.float32, copy=False)).unsqueeze(0).to(device)
    with torch.inference_mode():
        probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    return probability, (probability >= threshold).astype(np.uint8)
