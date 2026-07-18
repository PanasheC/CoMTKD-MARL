"""Model registry compatible with the MTKD-RL command line names."""
from __future__ import annotations

from typing import Callable

from torch import nn

from .cifar_resnet import resnet20, resnet32, resnet56, resnet110, resnet32x4
from .wide_resnet import wrn_28_4, wrn_40_2
from .torchvision_wrappers import ShuffleV2, MobileNetV2, RegNetX_400MF, RegNetY_400MF

ModelBuilder = Callable[..., nn.Module]

model_dict: dict[str, ModelBuilder] = {
    "resnet20": resnet20,
    "resnet32": resnet32,
    "resnet56": resnet56,
    "resnet110": resnet110,
    "resnet32x4": resnet32x4,
    "wrn_28_4": wrn_28_4,
    "wrn_40_2": wrn_40_2,
    "ShuffleV2": ShuffleV2,
    "MobileNetV2": MobileNetV2,
    "RegNetX_400MF": RegNetX_400MF,
    "RegNetY_400MF": RegNetY_400MF,
}


def build_model(name: str, num_classes: int = 100, **kwargs: object) -> nn.Module:
    if name not in model_dict:
        choices = ", ".join(sorted(model_dict))
        raise KeyError(f"Unknown model '{name}'. Available models: {choices}")
    return model_dict[name](num_classes=num_classes, **kwargs)
