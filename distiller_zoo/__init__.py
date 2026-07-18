"""Distillation loss zoo."""
from .coherent_losses import (
    CoherentDistillationObjective,
    DistillKL,
    LossBreakdown,
    ProbabilityDistillationLoss,
    RelationalGramLoss,
    UncertaintyMatchingLoss,
)
from .feature_mse_mtkd_rl import FeatureKLLoss, FeatureMSELoss

__all__ = [
    "CoherentDistillationObjective",
    "DistillKL",
    "LossBreakdown",
    "ProbabilityDistillationLoss",
    "RelationalGramLoss",
    "UncertaintyMatchingLoss",
    "FeatureKLLoss",
    "FeatureMSELoss",
]
