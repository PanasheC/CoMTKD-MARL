"""Feature distillation losses compatible with the MTKD-RL interface."""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class FeatureMSELoss(nn.Module):
    def forward(self, student_feature: Tensor, teacher_feature: Tensor) -> Tensor:
        if student_feature.shape != teacher_feature.shape:
            if student_feature.ndim == 4 and teacher_feature.ndim == 4:
                teacher_feature = F.adaptive_avg_pool2d(
                    teacher_feature, student_feature.shape[-2:]
                )
            else:
                raise ValueError(
                    "FeatureMSELoss requires matching vector dimensions or compatible feature maps"
                )
        difference = (student_feature - teacher_feature) ** 2
        return difference.flatten(1).mean(dim=1)


class FeatureKLLoss(nn.Module):
    def __init__(self, temperature: float = 4.0) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, student_feature: Tensor, teacher_feature: Tensor) -> Tensor:
        student = student_feature.flatten(1)
        teacher = teacher_feature.flatten(1)
        if student.shape[1] != teacher.shape[1]:
            raise ValueError("FeatureKLLoss requires matching flattened dimensions")
        temperature = self.temperature
        return (
            F.kl_div(
                F.log_softmax(student / temperature, dim=-1),
                F.softmax(teacher / temperature, dim=-1),
                reduction="none",
            ).sum(dim=-1)
            * temperature**2
        )
