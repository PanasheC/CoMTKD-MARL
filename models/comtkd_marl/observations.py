"""Construction of local teacher observations for cooperative distillation."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F

from .synchronization import pairwise_js_divergence


OBSERVATION_NAMES = (
    "teacher_ce",
    "teacher_entropy",
    "teacher_confidence",
    "teacher_correct",
    "teacher_student_kl",
    "feature_similarity",
    "gradient_alignment",
    "novelty",
    "redundancy",
    "conflict",
    "capacity_mismatch",
    "teacher_cost",
)


@dataclass
class ObservationOutput:
    observations: Tensor
    novelty: Tensor
    redundancy: Tensor
    conflict: Tensor
    teacher_student_kl: Tensor
    gradient_alignment: Tensor


def _normalize_feature(feature: Tensor) -> Tensor:
    return F.normalize(feature, dim=-1, eps=1e-8)


def build_teacher_observations(
    teacher_logits: Tensor,
    teacher_features: Tensor,
    student_logits: Tensor,
    student_feature: Tensor,
    targets: Tensor,
    teacher_costs: Tensor | None = None,
    base_temperature: float = 4.0,
) -> ObservationOutput:
    """Construct [B, M, 12] teacher observations.

    Gradient alignment is computed in logit space. For cross entropy, the
    supervised gradient direction is q - y. For teacher m, the distillation
    direction is q - p_m.
    """
    batch, teacher_count, classes = teacher_logits.shape
    teacher_probs = torch.softmax(teacher_logits / base_temperature, dim=-1)
    student_probs = torch.softmax(student_logits / base_temperature, dim=-1)
    expanded_targets = targets[:, None].expand(-1, teacher_count)
    teacher_ce = F.cross_entropy(
        teacher_logits.reshape(batch * teacher_count, classes),
        expanded_targets.reshape(-1),
        reduction="none",
    ).reshape(batch, teacher_count)
    teacher_entropy = -(teacher_probs * teacher_probs.clamp_min(1e-8).log()).sum(dim=-1)
    teacher_confidence, teacher_predictions = teacher_probs.max(dim=-1)
    teacher_correct = teacher_predictions.eq(expanded_targets).to(teacher_logits.dtype)
    student_expanded = student_probs.unsqueeze(1).expand_as(teacher_probs)
    teacher_student_kl = (
        teacher_probs
        * (teacher_probs.clamp_min(1e-8).log() - student_expanded.clamp_min(1e-8).log())
    ).sum(dim=-1)

    teacher_features_n = _normalize_feature(teacher_features)
    student_feature_n = _normalize_feature(student_feature).unsqueeze(1)
    feature_similarity = (teacher_features_n * student_feature_n).sum(dim=-1)

    one_hot = F.one_hot(targets, classes).to(student_probs.dtype)
    supervised_direction = student_probs - one_hot
    distillation_direction = student_expanded - teacher_probs
    gradient_alignment = F.cosine_similarity(
        distillation_direction,
        supervised_direction.unsqueeze(1),
        dim=-1,
        eps=1e-8,
    )

    pairwise_js = pairwise_js_divergence(teacher_probs)
    pairwise_cos = torch.bmm(teacher_features_n, teacher_features_n.transpose(1, 2))
    eye = torch.eye(teacher_count, device=teacher_logits.device, dtype=torch.bool).unsqueeze(0)
    peer_count = max(teacher_count - 1, 1)
    conflict = pairwise_js.masked_fill(eye, 0.0).sum(dim=-1) / peer_count
    redundancy = pairwise_cos.masked_fill(eye, 0.0).sum(dim=-1) / peer_count
    novelty = 0.5 * conflict + 0.5 * (1.0 - redundancy)

    capacity_mismatch = (
        teacher_features.norm(dim=-1) - student_feature.unsqueeze(1).norm(dim=-1)
    ).abs()
    capacity_mismatch = capacity_mismatch / teacher_features.norm(dim=-1).clamp_min(1e-6)

    if teacher_costs is None:
        teacher_costs = teacher_logits.new_ones(teacher_count)
    teacher_costs = teacher_costs.to(device=teacher_logits.device, dtype=teacher_logits.dtype)
    teacher_costs = teacher_costs / teacher_costs.max().clamp_min(1e-8)
    teacher_costs = teacher_costs.unsqueeze(0).expand(batch, -1)

    observations = torch.stack(
        [
            teacher_ce,
            teacher_entropy,
            teacher_confidence,
            teacher_correct,
            teacher_student_kl,
            feature_similarity,
            gradient_alignment,
            novelty,
            redundancy,
            conflict,
            capacity_mismatch,
            teacher_costs,
        ],
        dim=-1,
    )
    # Stable scale for the policy. Per-batch normalization is intentionally
    # detached because observations are environmental state, not model targets.
    mean = observations.mean(dim=(0, 1), keepdim=True)
    std = observations.std(dim=(0, 1), keepdim=True, unbiased=False).clamp_min(1e-5)
    observations = ((observations - mean) / std).detach()
    return ObservationOutput(
        observations=observations,
        novelty=novelty.detach(),
        redundancy=redundancy.detach(),
        conflict=conflict.detach(),
        teacher_student_kl=teacher_student_kl.detach(),
        gradient_alignment=gradient_alignment.detach(),
    )
