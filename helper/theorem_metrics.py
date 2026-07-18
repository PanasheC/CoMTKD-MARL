"""Empirical estimators for the three central theorems."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class AdvantageEstimate:
    covariance: np.ndarray
    optimal_weights: np.ndarray
    aggregate_risk: float
    best_single_risk: float
    risk_margin: float


def teacher_error_covariance(teacher_probabilities: Tensor, targets: Tensor) -> Tensor:
    """Estimate C_mn = E <e_m, e_n> using one-hot labels as posterior samples."""
    labels = torch.nn.functional.one_hot(
        targets, teacher_probabilities.shape[-1]
    ).to(teacher_probabilities.dtype)
    errors = teacher_probabilities - labels.unsqueeze(1)
    return torch.einsum("bmc,bnc->mn", errors, errors) / teacher_probabilities.shape[0]


def _project_simplex(vector: np.ndarray) -> np.ndarray:
    sorted_vector = np.sort(vector)[::-1]
    cumulative = np.cumsum(sorted_vector)
    rho_candidates = sorted_vector - (cumulative - 1.0) / np.arange(1, len(vector) + 1) > 0
    rho = np.nonzero(rho_candidates)[0][-1]
    theta = (cumulative[rho] - 1.0) / (rho + 1)
    return np.maximum(vector - theta, 0.0)


def optimal_simplex_weights(covariance: np.ndarray, iterations: int = 5000) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=np.float64)
    teacher_count = covariance.shape[0]
    eigen_max = max(float(np.linalg.eigvalsh(covariance).max()), 1e-8)
    step = 0.5 / eigen_max
    weights = np.full(teacher_count, 1.0 / teacher_count)
    for _ in range(iterations):
        updated = _project_simplex(weights - step * (2.0 * covariance @ weights))
        if np.linalg.norm(updated - weights) < 1e-11:
            weights = updated
            break
        weights = updated
    return weights


def estimate_multi_teacher_advantage(covariance: Tensor) -> AdvantageEstimate:
    c = covariance.detach().cpu().double().numpy()
    weights = optimal_simplex_weights(c)
    aggregate = float(weights @ c @ weights)
    best = float(np.diag(c).min())
    return AdvantageEstimate(c, weights, aggregate, best, best - aggregate)


def observed_contraction_ratios(coherence: Sequence[float]) -> list[float]:
    values = np.asarray(coherence, dtype=np.float64)
    ratios = []
    for previous, current in zip(values[:-1], values[1:]):
        ratios.append(float(current / max(previous, 1e-12)))
    return ratios


def saturation_cardinality(net_values: Sequence[float]) -> int:
    if not net_values:
        raise ValueError("net_values cannot be empty")
    return int(np.argmax(np.asarray(net_values))) + 1
