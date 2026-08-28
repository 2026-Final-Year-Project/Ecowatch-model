"""Create and persist reproducible train/validation/test splits."""

import csv
from pathlib import Path
from typing import Iterable, Sequence

from sklearn.model_selection import train_test_split

Pair = tuple[str | Path, str | Path]


def create_splits(
    pairs: Sequence[Pair],
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, list[Pair]]:
    """Create deterministic splits using the notebook's two-stage procedure."""
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1:
        raise ValueError("Split fractions must be between zero and one.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must leave a non-empty test fraction.")
    test_fraction = 1 - train_fraction - validation_fraction
    train, remainder = train_test_split(
        list(pairs), test_size=1 - train_fraction, random_state=seed
    )
    relative_test_fraction = test_fraction / (validation_fraction + test_fraction)
    validation, test = train_test_split(
        remainder, test_size=relative_test_fraction, random_state=seed
    )
    return {
        "train": train,
        "val": validation,
        "test": test,
    }


def save_pairs(pairs: Iterable[Pair], destination: str | Path) -> Path:
    """Write image/mask pairs to the CSV format consumed by the dataset class."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("image_path", "mask_path"))
        writer.writeheader()
        writer.writerows(
            {"image_path": str(image), "mask_path": str(mask)} for image, mask in pairs
        )
    return destination


def load_pairs(source: str | Path) -> list[tuple[Path, Path]]:
    """Read image/mask paths from a split CSV."""
    source = Path(source)
    with source.open(newline="") as file:
        reader = csv.DictReader(file)
        if not {"image_path", "mask_path"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{source} must contain image_path and mask_path columns.")
        return [(Path(row["image_path"]), Path(row["mask_path"])) for row in reader]


def replace_path_root(csv_path: str | Path, old_root: str | Path, new_root: str | Path) -> None:
    """Relocate paths in an existing split CSV, matching the Colab workflow."""
    old_root, new_root = str(old_root), str(new_root)
    pairs = [
        (str(image).replace(old_root, new_root, 1), str(mask).replace(old_root, new_root, 1))
        for image, mask in load_pairs(csv_path)
    ]
    save_pairs(pairs, csv_path)
