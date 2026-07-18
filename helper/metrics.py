"""Training and calibration metrics."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


class AverageMeter:
    def __init__(self, name: str = "meter", fmt: str = ":.4f") -> None:
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.value = 0.0
        self.average = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, count: int = 1) -> None:
        self.value = float(value)
        self.sum += float(value) * count
        self.count += count
        self.average = self.sum / max(self.count, 1)

    @property
    def avg(self) -> float:
        return self.average


def topk_correct(logits: Tensor, targets: Tensor, topk: tuple[int, ...] = (1,)) -> list[Tensor]:
    max_k = min(max(topk), logits.shape[1])
    predictions = logits.topk(max_k, dim=1).indices.t()
    correct = predictions.eq(targets.view(1, -1).expand_as(predictions))
    return [correct[: min(k, max_k)].reshape(-1).float().sum() for k in topk]


def accuracy(logits: Tensor, targets: Tensor, topk: tuple[int, ...] = (1,)) -> list[Tensor]:
    counts = topk_correct(logits, targets, topk)
    return [count * 100.0 / targets.shape[0] for count in counts]


def expected_calibration_error(logits: Tensor, targets: Tensor, bins: int = 15) -> Tensor:
    probabilities = torch.softmax(logits, dim=-1)
    confidence, prediction = probabilities.max(dim=-1)
    correct = prediction.eq(targets)
    boundaries = torch.linspace(0, 1, bins + 1, device=logits.device)
    ece = logits.new_zeros(())
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        in_bin = (confidence > lower) & (confidence <= upper)
        if in_bin.any():
            accuracy_bin = correct[in_bin].float().mean()
            confidence_bin = confidence[in_bin].mean()
            ece = ece + in_bin.float().mean() * (accuracy_bin - confidence_bin).abs()
    return ece


def brier_score(logits: Tensor, targets: Tensor) -> Tensor:
    probabilities = torch.softmax(logits, dim=-1)
    labels = torch.nn.functional.one_hot(targets, logits.shape[-1]).to(probabilities.dtype)
    return ((probabilities - labels) ** 2).sum(dim=-1).mean()


@dataclass(frozen=True)
class EvaluationMetrics:
    loss: float
    top1: float
    top5: float
    ece: float
    brier: float
