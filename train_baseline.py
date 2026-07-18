"""Train a CIFAR-100 teacher or student baseline."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch

from dataset.cifar100 import get_cifar100_dataloaders, get_fake_cifar100_dataloaders
from helper import append_jsonl, save_checkpoint, seed_everything, set_logger
from models import build_model
from train_loops import evaluate, train_baseline_epoch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a CIFAR-100 baseline model")
    parser.add_argument("--model", default="RegNetY_400MF")
    parser.add_argument("--data-folder", default="./data")
    parser.add_argument("--checkpoint-dir", default="./checkpoints/teachers")
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr-type", choices=["cosine", "multistep"], default="cosine")
    parser.add_argument("--milestones", type=int, nargs="+", default=[150, 180, 210])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--autoaugment", action="store_true")
    parser.add_argument("--cutout", type=int, default=0)
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dummy-data", action="store_true")
    parser.add_argument("--dummy-train-size", type=int, default=64)
    parser.add_argument("--dummy-test-size", type=int, default=32)
    return parser


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed, args.deterministic)
    device = resolve_device(args.device)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.checkpoint_dir) / f"{args.model}-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    args.logger = set_logger(run_dir / "train.log", name=f"teacher-{timestamp}")
    args.logger.info("device=%s model=%s", device, args.model)

    if args.dummy_data:
        train_loader, validation_loader = get_fake_cifar100_dataloaders(
            batch_size=args.batch_size,
            train_size=args.dummy_train_size,
            test_size=args.dummy_test_size,
            num_workers=0,
            seed=args.seed,
        )
    else:
        train_loader, validation_loader = get_cifar100_dataloaders(
            data_folder=args.data_folder,
            batch_size=args.batch_size,
            num_workers=args.workers,
            download=args.download,
            seed=args.seed,
            autoaugment=args.autoaugment,
            cutout=args.cutout,
        )

    model = build_model(args.model, num_classes=100).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    best_top1 = float("-inf")
    for epoch in range(args.epochs):
        train_result = train_baseline_epoch(
            train_loader, model, optimizer, device, epoch, args, scaler
        )
        validation = evaluate(validation_loader, model, device)
        metrics = {
            "epoch": epoch,
            "train_loss": train_result.loss,
            "train_top1": train_result.top1,
            "val_loss": validation.loss,
            "val_top1": validation.top1,
            "val_top5": validation.top5,
            "val_ece": validation.ece,
            "val_brier": validation.brier,
        }
        append_jsonl(run_dir / "metrics.jsonl", metrics)
        save_checkpoint(run_dir / f"{args.model}.pth.tar", model, epoch, metrics, optimizer)
        if validation.top1 > best_top1:
            best_top1 = validation.top1
            save_checkpoint(
                run_dir / f"{args.model}_best.pth.tar", model, epoch, metrics, optimizer
            )
        args.logger.info(
            "epoch=%d train_top1=%.2f val_top1=%.2f best=%.2f",
            epoch,
            train_result.top1,
            validation.top1,
            best_top1,
        )


if __name__ == "__main__":
    main()
