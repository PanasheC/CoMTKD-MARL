"""Evaluate a CIFAR-100 checkpoint."""
from __future__ import annotations

import argparse
import json

import torch

from dataset.cifar100 import get_cifar100_dataloaders, get_fake_cifar100_dataloaders
from helper import load_model_checkpoint
from models import build_model
from train_loops import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--arch", default="ShuffleV2")
    parser.add_argument("--data", default="./data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dummy-data", action="store_true")
    args = parser.parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    )
    model = build_model(args.arch, num_classes=100).to(device)
    load_model_checkpoint(model, args.checkpoint, device)
    if args.dummy_data:
        _, loader = get_fake_cifar100_dataloaders(batch_size=args.batch_size)
    else:
        _, loader = get_cifar100_dataloaders(
            args.data, args.batch_size, args.workers, download=True
        )
    metrics = evaluate(loader, model, device)
    print(json.dumps(metrics.__dict__, indent=2))


if __name__ == "__main__":
    main()
