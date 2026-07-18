"""Teacher coalition selection and theorem-oriented cardinality utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class SaturationEstimate:
    optimal_cardinality: int
    increments: tuple[float, ...]
    net_values: tuple[float, ...]


def select_topk_mask(scores: Tensor, counts: Tensor) -> Tensor:
    """Select a different top-k teacher set for every batch item."""
    if scores.ndim != 2:
        raise ValueError("scores must have shape [batch, teachers]")
    batch, teacher_count = scores.shape
    counts = counts.to(device=scores.device, dtype=torch.long).clamp(1, teacher_count)
    order = scores.argsort(dim=1, descending=True)
    ranks = torch.empty_like(order)
    rank_values = torch.arange(teacher_count, device=scores.device).expand(batch, -1)
    ranks.scatter_(1, order, rank_values)
    return ranks < counts.unsqueeze(1)


def normalized_active_weights(gates: Tensor, importance: Tensor, active_mask: Tensor) -> Tensor:
    positive = gates.clamp_min(1e-6) * torch.exp(importance.clamp(-12.0, 12.0))
    positive = positive * active_mask.to(positive.dtype)
    return positive / positive.sum(dim=1, keepdim=True).clamp_min(1e-8)


def teacher_marginal_utility(
    relevance: Tensor,
    novelty: Tensor,
    redundancy: Tensor,
    conflict: Tensor,
    cost: Tensor,
    novelty_weight: float = 1.0,
    redundancy_weight: float = 1.0,
    conflict_weight: float = 1.0,
    cost_weight: float = 1.0,
) -> Tensor:
    return (
        relevance
        + novelty_weight * novelty
        - redundancy_weight * redundancy
        - conflict_weight * conflict
        - cost_weight * cost
    )


def estimate_optimal_cardinality(
    conditional_gains: Sequence[float],
    incremental_costs: Sequence[float],
    student_capacity: float,
) -> SaturationEstimate:
    """Apply the Teacher Saturation stopping rule to an ordered teacher sequence.

    The incremental net value is
    min(g_{m+1}, [B_S - G_m]_+) - d_{m+1}.
    """
    if len(conditional_gains) != len(incremental_costs):
        raise ValueError("conditional_gains and incremental_costs must have equal length")
    gross = 0.0
    net = 0.0
    increments: list[float] = []
    values: list[float] = []
    optimum = 0
    best_value = float("-inf")
    for idx, (gain, cost) in enumerate(zip(conditional_gains, incremental_costs), start=1):
        usable_gain = min(float(gain), max(float(student_capacity) - gross, 0.0))
        increment = usable_gain - float(cost)
        increments.append(increment)
        gross += float(gain)
        net += increment
        values.append(net)
        if net > best_value:
            best_value = net
            optimum = idx
        if increment <= 0.0:
            break
    return SaturationEstimate(optimum, tuple(increments), tuple(values))
