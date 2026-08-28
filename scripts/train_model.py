"""Train the EcoWatch U-Net from prepared SmallMinesDS split CSVs."""

import argparse
import json
from pathlib import Path

import torch

from src.data.dataset import SmallMinesDataset, create_dataloader
from src.models.unet import UNet
from src.training.train import train_model
from src.training.utils import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--val-csv", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("experiments/unet_v1"), type=Path)
    parser.add_argument("--config", default=Path("configs/unet.json"), type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    arguments = parser.parse_args()

    config = json.loads(arguments.config.read_text())
    seed_everything(config["seed"])
    device = arguments.device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = SmallMinesDataset(arguments.train_csv)
    validation_dataset = SmallMinesDataset(arguments.val_csv)
    train_loader = create_dataloader(
        train_dataset, config["batch_size"], shuffle=True, num_workers=config["num_workers"]
    )
    validation_loader = create_dataloader(
        validation_dataset, config["batch_size"], num_workers=config["num_workers"]
    )
    model = UNet(in_channels=config["in_channels"], features=tuple(config["features"]))
    checkpoint, _ = train_model(
        model,
        train_loader,
        validation_loader,
        arguments.output_dir,
        epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        threshold=config["threshold"],
        device=device,
    )
    print(f"Best checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
