"""Discover and preprocess SmallMinesDS Sentinel-2 TIFF files."""

from pathlib import Path

import numpy as np
import rasterio


def sample_id(path: str | Path) -> str:
    """Return the shared identifier from an IMG_* or MASK_* filename."""
    name = Path(path).stem
    for prefix in ("IMG_", "MASK_"):
        if name.startswith(prefix):
            return name.removeprefix(prefix)
    return name


def collect_pairs(dataset_root: str | Path) -> list[tuple[Path, Path]]:
    """Pair images and masks recursively by their shared filename identifier."""
    root = Path(dataset_root)
    image_paths = sorted(root.glob("**/IMAGE/*.tif"))
    mask_paths = sorted(root.glob("**/MASK/*.tif"))
    images = {sample_id(path): path for path in image_paths}
    masks = {sample_id(path): path for path in mask_paths}
    return [(images[key], masks[key]) for key in sorted(images.keys() & masks.keys())]


def normalize_bands(image: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Min-max normalize each spectral band independently to [0, 1]."""
    if image.ndim != 3:
        raise ValueError(f"Expected an array shaped (bands, height, width), got {image.shape}.")
    minimum = image.min(axis=(1, 2), keepdims=True)
    maximum = image.max(axis=(1, 2), keepdims=True)
    return (image - minimum) / np.maximum(maximum - minimum, epsilon)


def read_image(path: str | Path, normalize: bool = True) -> np.ndarray:
    """Read a multi-band Sentinel-2 TIFF as float32."""
    with rasterio.open(path) as source:
        image = source.read().astype(np.float32)
    return normalize_bands(image) if normalize else image


def read_mask(path: str | Path) -> np.ndarray:
    """Read the first TIFF band and return a binary (1, H, W) mask."""
    with rasterio.open(path) as source:
        mask = source.read(1).astype(np.float32)
    return (mask > 0).astype(np.float32)[None, ...]
