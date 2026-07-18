"""Equal-weight multi-teacher knowledge distillation on CIFAR-100."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch

from dataset.cifar100 import get_cifar100_dataloaders, get_fake_cifar100_dataloaders
from helper import append_jsonl, load_model_checkpoint, save_checkpoint, seed_everything, set_logger
from models import FeatureAdapterBank, build_model, infer_feature_dimensions
from setting import teacher_model_path_dict
from train_loops import evaluate, train_avg_epoch


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Equal-weight multi-teacher KD")
    p.add_argument("--data", default="./data")
    p.add_argument("--arch", default="ShuffleV2")
    p.add_argument("--teacher-name-list", nargs="+", default=list(teacher_model_path_dict))
    p.add_argument("--teacher-checkpoint", action="append", default=[], metavar="MODEL=PATH")
    p.add_argument("--checkpoint-dir", default="./checkpoints/average")
    p.add_argument("--epochs", type=int, default=240)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--lr-type", choices=["cosine", "multistep"], default="cosine")
    p.add_argument("--milestones", type=int, nargs="+", default=[150, 180, 210])
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--ce-weight", type=float, default=1.0)
    p.add_argument("--kd-weight", type=float, default=1.0)
    p.add_argument("--feat-weight", type=float, default=5.0)
    p.add_argument("--kd-T", dest="kd_T", type=float, default=4.0)
    p.add_argument("--common-dim", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--print-freq", type=int, default=50)
    p.add_argument("--device", default="auto")
    p.add_argument("--dummy-data", action="store_true")
    p.add_argument("--allow-random-teachers", action="store_true")
    p.add_argument("--dummy-train-size", type=int, default=64)
    p.add_argument("--dummy-test-size", type=int, default=32)
    return p


def device_from(name: str) -> torch.device:
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)


def checkpoint_mapping(entries: list[str]) -> dict[str, str]:
    mapping = dict(teacher_model_path_dict)
    for entry in entries:
        if "=" not in entry:
            raise ValueError("--teacher-checkpoint must use MODEL=PATH")
        name, path = entry.split("=", 1)
        mapping[name] = path
    return mapping


def load_teachers(args, device: torch.device):
    mapping = checkpoint_mapping(args.teacher_checkpoint)
    teachers = []
    for name in args.teacher_name_list:
        model = build_model(name, num_classes=100).to(device)
        path = Path(mapping.get(name, ""))
        if path.is_file():
            load_model_checkpoint(model, path, device)
        elif not args.allow_random_teachers:
            raise FileNotFoundError(
                f"Checkpoint for {name} was not found at '{path}'. Train it or pass --allow-random-teachers for a smoke test."
            )
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        teachers.append(model)
    return teachers


def main() -> None:
    args = parser().parse_args()
    seed_everything(args.seed, args.deterministic)
    device = device_from(args.device)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.checkpoint_dir) / f"{args.arch}-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    args.logger = set_logger(run_dir / "train.log", name=f"avg-{timestamp}")
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
            args.data, args.batch_size, args.workers, args.download, seed=args.seed
        )
    teachers = load_teachers(args, device)
    student = build_model(args.arch, num_classes=100).to(device)
    dimensions = infer_feature_dimensions(teachers, student, device)
    adapters = FeatureAdapterBank(
        dimensions.teacher_dims, dimensions.student_dim, args.common_dim
    ).to(device)
    optimizer = torch.optim.SGD(
        list(student.parameters()) + list(adapters.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    best = float("-inf")
    for epoch in range(args.epochs):
        train_result = train_avg_epoch(
            train_loader, student, teachers, adapters, optimizer, device, epoch, args, scaler
        )
        validation = evaluate(validation_loader, student, device)
        record = {
            "epoch": epoch,
            "train_loss": train_result.loss,
            "train_top1": train_result.top1,
            "val_top1": validation.top1,
            "val_top5": validation.top5,
            "val_ece": validation.ece,
            "val_brier": validation.brier,
        }
        append_jsonl(run_dir / "metrics.jsonl", record)
        extra = {"adapters": adapters.state_dict(), "teacher_names": args.teacher_name_list}
        save_checkpoint(run_dir / f"{args.arch}.pth.tar", student, epoch, record, optimizer, extra)
        if validation.top1 > best:
            best = validation.top1
            save_checkpoint(run_dir / f"{args.arch}_best.pth.tar", student, epoch, record, optimizer, extra)
        args.logger.info("epoch=%d val_top1=%.2f best=%.2f", epoch, validation.top1, best)


if __name__ == "__main__":
    main()
