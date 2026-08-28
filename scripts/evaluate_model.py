"""Evaluate a trained EcoWatch checkpoint on the fixed test split."""

import argparse
from pathlib import Path

import torch

from src.data.dataset import SmallMinesDataset, create_dataloader
from src.evaluation.evaluate import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--test-csv", required=True, type=Path)
    parser.add_argument("--output", default=Path("experiments/unet_v1/test_metrics.json"), type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    arguments = parser.parse_args()

    device = arguments.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SmallMinesDataset(arguments.test_csv)
    loader = create_dataloader(dataset, arguments.batch_size, num_workers=arguments.num_workers)
    results = evaluate_checkpoint(
        arguments.checkpoint, loader, device, arguments.threshold, arguments.output
    )
    print("Held-out test results")
    for name, value in results.items():
        print(f"{name}: {value:.4f}" if isinstance(value, float) else f"{name}: {value}")


if __name__ == "__main__":
    main()
