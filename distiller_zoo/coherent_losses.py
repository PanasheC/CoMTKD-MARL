"""Losses for synchronized multi-teacher knowledge distillation."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DistillKL(nn.Module):
    """Teacher-to-student KL divergence with optional sample reduction."""

    def __init__(self, temperature: float = 4.0) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(
        self,
        student_logits: Tensor,
        teacher_logits_or_probabilities: Tensor,
        unreduce: bool = False,
        teacher_is_probability: bool = False,
    ) -> Tensor:
        temperature = self.temperature
        student_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
        if teacher_is_probability:
            teacher_probability = teacher_logits_or_probabilities.clamp_min(1e-8)
            teacher_probability = teacher_probability / teacher_probability.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
        else:
            teacher_probability = F.softmax(
                teacher_logits_or_probabilities / temperature, dim=-1
            )
        loss = F.kl_div(
            student_log_prob, teacher_probability, reduction="none"
        ).sum(dim=-1) * temperature**2
        return loss if unreduce else loss.mean()


class ProbabilityDistillationLoss(nn.Module):
    def __init__(self, temperature: float = 4.0) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, student_logits: Tensor, teacher_probability: Tensor) -> Tensor:
        student_log_prob = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probability = teacher_probability.clamp_min(1e-8)
        teacher_probability = teacher_probability / teacher_probability.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        return F.kl_div(
            student_log_prob,
            teacher_probability,
            reduction="none",
        ).sum(dim=-1) * self.temperature**2


class RelationalGramLoss(nn.Module):
    """Match within-minibatch relational geometry."""

    @staticmethod
    def gram(features: Tensor) -> Tensor:
        features = F.normalize(features, dim=-1, eps=1e-8)
        return features @ features.transpose(0, 1)

    def forward(self, student_features: Tensor, teacher_features: Tensor) -> Tensor:
        return F.mse_loss(self.gram(student_features), self.gram(teacher_features))


class UncertaintyMatchingLoss(nn.Module):
    def forward(self, student_logits: Tensor, teacher_probability: Tensor) -> Tensor:
        student_probability = torch.softmax(student_logits, dim=-1)
        student_entropy = -(
            student_probability * student_probability.clamp_min(1e-8).log()
        ).sum(dim=-1)
        teacher_entropy = -(
            teacher_probability * teacher_probability.clamp_min(1e-8).log()
        ).sum(dim=-1)
        return F.mse_loss(student_entropy, teacher_entropy)


@dataclass
class LossBreakdown:
    total: Tensor
    supervised: Tensor
    logit: Tensor
    feature: Tensor
    relational: Tensor
    uncertainty: Tensor
    per_sample_total: Tensor
    channel_weights: Tensor


class CoherentDistillationObjective(nn.Module):
    """Complete student objective from the CoMTKD-MARL paper."""

    def __init__(
        self,
        temperature: float = 4.0,
        logit_weight: float = 1.0,
        feature_weight: float = 5.0,
        relational_weight: float = 0.1,
        uncertainty_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.logit_loss = ProbabilityDistillationLoss(temperature)
        self.relational_loss = RelationalGramLoss()
        self.uncertainty_loss = UncertaintyMatchingLoss()
        self.base_weights = torch.tensor(
            [logit_weight, feature_weight, relational_weight, uncertainty_weight],
            dtype=torch.float32,
        )

    def forward(
        self,
        student_logits: Tensor,
        student_feature: Tensor,
        targets: Tensor,
        aggregate_probability: Tensor,
        aggregate_feature: Tensor,
        teacher_weights: Tensor,
        channel_allocations: Tensor,
    ) -> LossBreakdown:
        supervised_per_sample = F.cross_entropy(student_logits, targets, reduction="none")
        logit_per_sample = self.logit_loss(student_logits, aggregate_probability)
        feature_per_sample = ((student_feature - aggregate_feature) ** 2).mean(dim=-1)
        relational = self.relational_loss(student_feature, aggregate_feature)
        uncertainty = self.uncertainty_loss(student_logits, aggregate_probability)
        channel_weights = (
            teacher_weights.unsqueeze(-1) * channel_allocations
        ).sum(dim=1)
        base = self.base_weights.to(student_logits.device, student_logits.dtype)
        effective = channel_weights * base.unsqueeze(0)
        per_sample_total = (
            supervised_per_sample
            + effective[:, 0] * logit_per_sample
            + effective[:, 1] * feature_per_sample
            + effective[:, 2] * relational
            + effective[:, 3] * uncertainty
        )
        return LossBreakdown(
            total=per_sample_total.mean(),
            supervised=supervised_per_sample.mean(),
            logit=logit_per_sample.mean(),
            feature=feature_per_sample.mean(),
            relational=relational,
            uncertainty=uncertainty,
            per_sample_total=per_sample_total,
            channel_weights=channel_weights,
        )
