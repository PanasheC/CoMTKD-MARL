"""Feature adapters and shape inference utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def pooled_dim(feature: Tensor) -> int:
    if feature.ndim == 2:
        return feature.shape[1]
    if feature.ndim == 4:
        return feature.shape[1]
    raise ValueError(f"Expected a 2D or 4D feature tensor, got shape {tuple(feature.shape)}")


def pool_feature(feature: Tensor) -> Tensor:
    if feature.ndim == 2:
        return feature
    if feature.ndim == 4:
        return F.adaptive_avg_pool2d(feature, 1).flatten(1)
    raise ValueError(f"Expected a 2D or 4D feature tensor, got shape {tuple(feature.shape)}")


class VectorAdapter(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, feature: Tensor) -> Tensor:
        return self.proj(pool_feature(feature))


class FeatureAdapterBank(nn.Module):
    """Maps heterogeneous teacher and student features into one knowledge space."""

    def __init__(
        self,
        teacher_dims: Sequence[int],
        student_dim: int,
        common_dim: int = 256,
    ) -> None:
        super().__init__()
        self.teacher_adapters = nn.ModuleList(
            VectorAdapter(int(dim), common_dim) for dim in teacher_dims
        )
        self.student_adapter = VectorAdapter(int(student_dim), common_dim)
        self.common_dim = common_dim

    def teachers(self, teacher_features: Sequence[Tensor]) -> Tensor:
        if len(teacher_features) != len(self.teacher_adapters):
            raise ValueError(
                f"Expected {len(self.teacher_adapters)} teacher features, got {len(teacher_features)}"
            )
        return torch.stack(
            [adapter(feature) for adapter, feature in zip(self.teacher_adapters, teacher_features)],
            dim=1,
        )

    def student(self, student_feature: Tensor) -> Tensor:
        return self.student_adapter(student_feature)


@dataclass(frozen=True)
class FeatureDimensions:
    teacher_dims: list[int]
    student_dim: int


@torch.no_grad()
def infer_feature_dimensions(
    teacher_models: Sequence[nn.Module],
    student_model: nn.Module,
    device: torch.device,
    image_size: int = 32,
) -> FeatureDimensions:
    dummy = torch.zeros(2, 3, image_size, image_size, device=device)
    teacher_dims: list[int] = []
    for model in teacher_models:
        features, _ = model(dummy, is_feat=True)
        teacher_dims.append(pooled_dim(features[-2]))
    student_features, _ = student_model(dummy, is_feat=True)
    return FeatureDimensions(teacher_dims, pooled_dim(student_features[-2]))


# Compatibility names used in the reference implementation.
class Regress(VectorAdapter):
    """Compatibility alias for the student-to-teacher feature regressor."""


class TransFeat(FeatureAdapterBank):
    """Compatibility alias for the multi-teacher feature adapter bank."""
