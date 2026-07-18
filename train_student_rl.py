"""Train CoMTKD-MARL on CIFAR-100.

The filename matches the reference MTKD-RL repository, while the implementation
adds cooperative teacher actors, a centralized coherence critic, a graph
synchronization oracle, and a learned teacher-cardinality policy.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import warnings

import torch

from dataset.cifar100 import get_cifar100_dataloaders, get_fake_cifar100_dataloaders
from distiller_zoo import CoherentDistillationObjective
from helper import append_jsonl, load_model_checkpoint, save_checkpoint, seed_everything, set_logger
from models import FeatureAdapterBank, build_model, infer_feature_dimensions
from models.comtkd_marl import CoMTKDMARL, MAPPOTrainer, OBSERVATION_NAMES, RolloutBuffer
from setting import teacher_model_path_dict
from train_loops import CoMTKDTrainState, evaluate, train_comtkd_epoch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Coherent Multi-Teacher KD with MARL")
    parser.add_argument("--data", default="./data")
    parser.add_argument("--arch", default="ShuffleV2")
    parser.add_argument("--teacher-name-list", nargs="+", default=list(teacher_model_path_dict))
    parser.add_argument("--teacher-checkpoint", action="append", default=[], metavar="MODEL=PATH")
    parser.add_argument("--checkpoint-dir", default="./checkpoints/comtkd_marl")
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--lr-type", choices=["cosine", "multistep"], default="cosine")
    parser.add_argument("--milestones", type=int, nargs="+", default=[150, 180, 210])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--student-grad-clip", type=float, default=5.0)
    parser.add_argument("--kd-T", dest="kd_T", type=float, default=4.0)
    parser.add_argument("--logit-weight", type=float, default=1.0)
    parser.add_argument("--feat-weight", type=float, default=5.0)
    parser.add_argument("--relational-weight", type=float, default=0.1)
    parser.add_argument("--uncertainty-weight", type=float, default=0.05)
    parser.add_argument("--common-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--sync-rounds", type=int, default=3)
    parser.add_argument("--min-temperature", type=float, default=1.0)
    parser.add_argument("--max-temperature", type=float, default=8.0)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-minibatch-size", type=int, default=256)
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--policy-grad-clip", type=float, default=1.0)
    parser.add_argument("--rollout-size", type=int, default=1024)
    parser.add_argument("--forced-cardinality", type=int, default=None)
    parser.add_argument("--reward-mode", choices=["loss_delta", "negative_loss"], default="loss_delta")
    parser.add_argument("--reward-novelty", type=float, default=0.05)
    parser.add_argument("--reward-coherence", type=float, default=0.10)
    parser.add_argument("--reward-redundancy", type=float, default=0.05)
    parser.add_argument("--reward-conflict", type=float, default=0.05)
    parser.add_argument("--reward-cost", type=float, default=0.02)
    parser.add_argument("--reward-stability", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trial", default="1")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dummy-data", action="store_true")
    parser.add_argument("--allow-random-teachers", action="store_true")
    parser.add_argument("--dummy-train-size", type=int, default=64)
    parser.add_argument("--dummy-test-size", type=int, default=32)
    parser.add_argument("--resume", default="")
    # Compatibility arguments from MTKD-RL. Multi-process execution is left to
    # torchrun in a future distributed extension.
    parser.add_argument("--dynamic", action="store_true", help="Retained for command compatibility")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--dist-backend", default="nccl")
    return parser


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def parse_checkpoint_mapping(entries: list[str]) -> dict[str, str]:
    mapping = dict(teacher_model_path_dict)
    for entry in entries:
        if "=" not in entry:
            raise ValueError("--teacher-checkpoint must use MODEL=PATH")
        name, path = entry.split("=", 1)
        mapping[name] = path
    return mapping


def load_teachers(args, device: torch.device) -> list[torch.nn.Module]:
    mapping = parse_checkpoint_mapping(args.teacher_checkpoint)
    teachers: list[torch.nn.Module] = []
    for name in args.teacher_name_list:
        teacher = build_model(name, num_classes=100).to(device)
        checkpoint_path = Path(mapping.get(name, ""))
        if checkpoint_path.is_file():
            load_model_checkpoint(teacher, checkpoint_path, device)
        elif not args.allow_random_teachers:
            raise FileNotFoundError(
                f"Checkpoint for {name} was not found at '{checkpoint_path}'. "
                "Train the teacher, update setting.py, or pass --allow-random-teachers for a smoke test."
            )
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        teachers.append(teacher)
    return teachers


def teacher_cost_vector(teachers: list[torch.nn.Module], device: torch.device) -> torch.Tensor:
    counts = [sum(parameter.numel() for parameter in teacher.parameters()) for teacher in teachers]
    return torch.tensor(counts, device=device, dtype=torch.float32)


def main() -> None:
    args = build_parser().parse_args()
    if args.world_size > 1:
        warnings.warn(
            "This release validates single-process training. Use one process per experiment, then aggregate seeds."
        )
    seed_everything(args.seed, args.deterministic)
    device = resolve_device(args.device)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    teacher_string = "_".join(args.teacher_name_list)
    run_name = f"{args.arch}-cifar100-comtkd-{args.trial}-{len(args.teacher_name_list)}-{teacher_string}-{timestamp}"
    run_dir = Path(args.checkpoint_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    args.logger = set_logger(run_dir / "train.log", name=f"comtkd-{timestamp}")
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args) | {"logger": None}, handle, indent=2, default=str)
    args.logger.info("device=%s run=%s", device, run_name)
    args.logger.info("teacher observations=%s", ", ".join(OBSERVATION_NAMES))

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
            data_folder=args.data,
            batch_size=args.batch_size,
            num_workers=args.workers,
            download=args.download,
            seed=args.seed,
        )

    teachers = load_teachers(args, device)
    student = build_model(args.arch, num_classes=100).to(device)
    dimensions = infer_feature_dimensions(teachers, student, device)
    adapters = FeatureAdapterBank(
        dimensions.teacher_dims, dimensions.student_dim, args.common_dim
    ).to(device)
    policy = CoMTKDMARL(
        teacher_count=len(teachers),
        observation_dim=len(OBSERVATION_NAMES),
        channel_count=4,
        hidden_dim=args.hidden_dim,
        attention_heads=args.attention_heads,
        sync_rounds=args.sync_rounds,
        min_temperature=args.min_temperature,
        max_temperature=args.max_temperature,
    ).to(device)
    objective = CoherentDistillationObjective(
        temperature=args.kd_T,
        logit_weight=args.logit_weight,
        feature_weight=args.feat_weight,
        relational_weight=args.relational_weight,
        uncertainty_weight=args.uncertainty_weight,
    ).to(device)

    student_optimizer = torch.optim.SGD(
        list(student.parameters()) + list(adapters.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    actor_optimizer = torch.optim.AdamW(
        policy.actor_parameters(), lr=args.actor_lr, weight_decay=1e-4
    )
    critic_optimizer = torch.optim.AdamW(
        policy.critic.parameters(), lr=args.critic_lr, weight_decay=1e-4
    )
    ppo_trainer = MAPPOTrainer(
        policy=policy,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        clip_ratio=args.ppo_clip,
        entropy_coefficient=args.entropy_coefficient,
        value_coefficient=args.value_coefficient,
        max_grad_norm=args.policy_grad_clip,
        epochs=args.ppo_epochs,
        minibatch_size=args.ppo_minibatch_size,
    )
    rollout_buffer = RolloutBuffer()
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    state = CoMTKDTrainState()
    start_epoch = 0
    best_top1 = float("-inf")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        student.load_state_dict(checkpoint["model"])
        adapters.load_state_dict(checkpoint["adapters"])
        policy.load_state_dict(checkpoint["policy"])
        student_optimizer.load_state_dict(checkpoint["optimizer"])
        actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_top1 = float(checkpoint.get("best_top1", best_top1))

    costs = teacher_cost_vector(teachers, device)
    for epoch in range(start_epoch, args.epochs):
        train_result, state = train_comtkd_epoch(
            loader=train_loader,
            student=student,
            teachers=teachers,
            adapters=adapters,
            policy=policy,
            objective=objective,
            optimizer=student_optimizer,
            ppo_trainer=ppo_trainer,
            rollout_buffer=rollout_buffer,
            device=device,
            epoch=epoch,
            args=args,
            state=state,
            teacher_costs=costs,
            scaler=scaler,
        )
        validation = evaluate(validation_loader, student, device)
        record = {
            "epoch": epoch,
            "train_loss": train_result.loss,
            "train_top1": train_result.top1,
            "val_loss": validation.loss,
            "val_top1": validation.top1,
            "val_top5": validation.top5,
            "val_ece": validation.ece,
            "val_brier": validation.brier,
            "policy_updates": state.policy_updates,
            **{f"train_{key}": value for key, value in train_result.metrics.items()},
        }
        append_jsonl(run_dir / "metrics.jsonl", record)
        extra = {
            "adapters": adapters.state_dict(),
            "policy": policy.state_dict(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "teacher_names": args.teacher_name_list,
            "best_top1": max(best_top1, validation.top1),
            "config": {key: value for key, value in vars(args).items() if key != "logger"},
        }
        save_checkpoint(
            run_dir / f"{args.arch}.pth.tar",
            student,
            epoch,
            record,
            student_optimizer,
            extra,
        )
        if validation.top1 > best_top1:
            best_top1 = validation.top1
            save_checkpoint(
                run_dir / f"{args.arch}_best.pth.tar",
                student,
                epoch,
                record,
                student_optimizer,
                extra,
            )
        args.logger.info(
            "epoch=%d val_top1=%.2f val_ece=%.4f best=%.2f active=%.2f coherence=%.5f",
            epoch,
            validation.top1,
            validation.ece,
            best_top1,
            train_result.metrics.get("active", float("nan")),
            train_result.metrics.get("coherence", float("nan")),
        )


if __name__ == "__main__":
    main()
