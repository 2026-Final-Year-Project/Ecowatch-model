"""Discover SmallMinesDS pairs and create reproducible split CSV files."""

import argparse
from pathlib import Path

from src.data.preprocessing import collect_pairs
from src.data.splits import create_splits, save_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()

    pairs = collect_pairs(arguments.dataset_root)
    if not pairs:
        raise SystemExit(f"No IMAGE/MASK TIFF pairs found below {arguments.dataset_root}.")
    splits = create_splits(pairs, seed=arguments.seed)
    save_pairs(pairs, arguments.output_dir / "paired_dataset.csv")
    for name, samples in splits.items():
        destination = save_pairs(samples, arguments.output_dir / f"{name}_dataset.csv")
        print(f"{name}: {len(samples)} samples -> {destination}")


if __name__ == "__main__":
    main()
