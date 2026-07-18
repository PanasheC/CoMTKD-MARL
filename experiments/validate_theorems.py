"""Empirical checks for the three central CoMTKD-MARL theorems.

The script does not claim to prove the theorems empirically. It computes the
quantities that the paper states should be measured: teacher error covariance,
coalition risk, synchronization contraction, and a cost-adjusted saturation
curve.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from dataset.cifar100 import get_cifar100_dataloaders, get_fake_cifar100_dataloaders
from helper import load_model_checkpoint
from helper.theorem_metrics import (
    estimate_multi_teacher_advantage,
    optimal_simplex_weights,
    saturation_cardinality,
    teacher_error_covariance,
)
from models import build_model
from models.comtkd_marl.synchronization import KnowledgeSynchronizationOracle
from setting import teacher_model_path_dict


def parse_mapping(entries: list[str]) -> dict[str, str]:
    mapping = dict(teacher_model_path_dict)
    for entry in entries:
        name, path = entry.split("=", 1)
        mapping[name] = path
    return mapping


@torch.no_grad()
def collect(
    teachers: list[torch.nn.Module], loader, device: torch.device, max_batches: int | None
):
    probabilities, labels = [], []
    for batch_index, (inputs, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs = inputs.to(device)
        batch_probs = []
        for teacher in teachers:
            logits = teacher(inputs)
            batch_probs.append(torch.softmax(logits, dim=-1))
        probabilities.append(torch.stack(batch_probs, dim=1).cpu())
        labels.append(targets.cpu())
    return torch.cat(probabilities), torch.cat(labels)


def subset_curve(covariance: np.ndarray, per_teacher_cost: float):
    teacher_count = covariance.shape[0]
    selected: list[int] = []
    remaining = list(range(teacher_count))
    rows = []
    best_single = float(np.diag(covariance).min())
    for cardinality in range(1, teacher_count + 1):
        best_candidate = None
        best_risk = float("inf")
        best_weights = None
        for candidate in remaining:
            subset = selected + [candidate]
            matrix = covariance[np.ix_(subset, subset)]
            weights = optimal_simplex_weights(matrix)
            risk = float(weights @ matrix @ weights)
            if risk < best_risk:
                best_candidate = candidate
                best_risk = risk
                best_weights = weights
        assert best_candidate is not None and best_weights is not None
        selected.append(best_candidate)
        remaining.remove(best_candidate)
        gross_gain = best_single - best_risk
        net_value = gross_gain - per_teacher_cost * cardinality
        rows.append(
            {
                "cardinality": cardinality,
                "selected_teacher_index": best_candidate,
                "coalition": selected.copy(),
                "risk": best_risk,
                "gross_gain": gross_gain,
                "net_value": net_value,
                "weights": best_weights.tolist(),
            }
        )
    return rows


def coherence_curve(probabilities: torch.Tensor, rounds: int):
    device = probabilities.device
    batch, teachers, classes = probabilities.shape
    features = probabilities
    active = torch.ones(batch, teachers, dtype=torch.bool, device=device)
    weights = torch.full((batch, teachers), 1.0 / teachers, device=device)
    rows = []
    for round_count in range(rounds + 1):
        oracle = KnowledgeSynchronizationOracle(rounds=round_count).to(device)
        oracle.eval()
        output = oracle(probabilities, features, weights, active)
        rows.append(
            {
                "round": round_count,
                "coherence": float(output.final_coherence.mean()),
                "spectral_rho": float(output.spectral_rho.mean()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-name-list", nargs="+", default=list(teacher_model_path_dict))
    parser.add_argument("--teacher-checkpoint", action="append", default=[])
    parser.add_argument("--data", default="./data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="./results/theorem_validation")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--sync-rounds", type=int, default=8)
    parser.add_argument("--per-teacher-cost", type=float, default=0.001)
    parser.add_argument("--dummy-data", action="store_true")
    parser.add_argument("--allow-random-teachers", action="store_true")
    args = parser.parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    )
    mapping = parse_mapping(args.teacher_checkpoint)
    teachers = []
    for name in args.teacher_name_list:
        model = build_model(name, num_classes=100).to(device)
        path = Path(mapping.get(name, ""))
        if path.is_file():
            load_model_checkpoint(model, path, device)
        elif not args.allow_random_teachers:
            raise FileNotFoundError(path)
        model.eval()
        teachers.append(model)
    if args.dummy_data:
        _, loader = get_fake_cifar100_dataloaders(batch_size=args.batch_size)
    else:
        _, loader = get_cifar100_dataloaders(
            args.data, args.batch_size, args.workers, download=True
        )
    probabilities, labels = collect(teachers, loader, device, args.max_batches)
    covariance = teacher_error_covariance(probabilities, labels)
    advantage = estimate_multi_teacher_advantage(covariance)
    subset_rows = subset_curve(advantage.covariance, args.per_teacher_cost)
    optimal_cardinality = saturation_cardinality([row["net_value"] for row in subset_rows])
    coherence_rows = coherence_curve(probabilities.to(device), args.sync_rounds)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "teacher_names": args.teacher_name_list,
        "samples": int(labels.numel()),
        "covariance": advantage.covariance.tolist(),
        "optimal_weights": advantage.optimal_weights.tolist(),
        "aggregate_risk": advantage.aggregate_risk,
        "best_single_risk": advantage.best_single_risk,
        "risk_margin": advantage.risk_margin,
        "estimated_optimal_cardinality": optimal_cardinality,
        "per_teacher_cost": args.per_teacher_cost,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output / "cardinality_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=subset_rows[0].keys())
        writer.writeheader()
        writer.writerows(subset_rows)
    with (output / "coherence_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=coherence_rows[0].keys())
        writer.writeheader()
        writer.writerows(coherence_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
