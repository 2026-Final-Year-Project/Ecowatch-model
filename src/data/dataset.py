"""PyTorch dataset for paired SmallMinesDS imagery and masks."""

from pathlib import Path
from typing import Callable, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from .preprocessing import read_image, read_mask
from .splits import load_pairs

Pair = tuple[str | Path, str | Path]
Transform = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


class SmallMinesDataset(Dataset):
    """Load normalized 13-band images and corresponding binary masks."""

    def __init__(self, samples: str | Path | Sequence[Pair], transform: Transform | None = None):
        self.samples = load_pairs(samples) if isinstance(samples, (str, Path)) else list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.samples[index]
        image = torch.from_numpy(read_image(image_path))
        mask = torch.from_numpy(read_mask(mask_path))
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        return image, mask


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 2,
    pin_memory: bool | None = None,
) -> DataLoader:
    """Create a loader with the settings used by the training notebook."""
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
