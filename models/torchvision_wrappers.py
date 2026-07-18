"""Torchvision backbones adapted to CIFAR resolution and feature distillation."""
from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision import models as tvm


class ShuffleNetV2CIFAR(nn.Module):
    def __init__(self, num_classes: int = 100, width: float = 0.5) -> None:
        super().__init__()
        builders: dict[float, Callable[..., nn.Module]] = {
            0.5: tvm.shufflenet_v2_x0_5,
            1.0: tvm.shufflenet_v2_x1_0,
        }
        if width not in builders:
            raise ValueError(f"Supported ShuffleNet widths are {tuple(builders)}")
        net = builders[width](weights=None, num_classes=num_classes)
        net.conv1[0] = nn.Conv2d(3, net.conv1[0].out_channels, 3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()
        self.conv1 = net.conv1
        self.maxpool = net.maxpool
        self.stage2 = net.stage2
        self.stage3 = net.stage3
        self.stage4 = net.stage4
        self.conv5 = net.conv5
        self.fc = net.fc
        self.feature_dim = self.fc.in_features

    def forward(self, x: Tensor, is_feat: bool = False):
        f0 = self.maxpool(self.conv1(x))
        f1 = self.stage2(f0)
        f2 = self.stage3(f1)
        f3 = self.stage4(f2)
        f4 = self.conv5(f3)
        pooled = f4.mean([2, 3])
        logits = self.fc(pooled)
        if is_feat:
            return [f0, f1, f2, f3, f4, pooled], logits
        return logits


class MobileNetV2CIFAR(nn.Module):
    def __init__(self, num_classes: int = 100, width_mult: float = 1.0) -> None:
        super().__init__()
        net = tvm.mobilenet_v2(weights=None, num_classes=num_classes, width_mult=width_mult)
        first_conv = net.features[0][0]
        net.features[0][0] = nn.Conv2d(
            3,
            first_conv.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.features = net.features
        self.classifier = net.classifier
        self.feature_dim = net.last_channel
        self.capture_indices = {1, 3, 6, 13, len(self.features) - 1}

    def forward(self, x: Tensor, is_feat: bool = False):
        feats: list[Tensor] = []
        out = x
        for idx, layer in enumerate(self.features):
            out = layer(out)
            if idx in self.capture_indices:
                feats.append(out)
        pooled = F.adaptive_avg_pool2d(out, 1).flatten(1)
        logits = self.classifier(pooled)
        if is_feat:
            return [*feats, pooled], logits
        return logits


class RegNetCIFAR(nn.Module):
    def __init__(self, variant: str, num_classes: int = 100) -> None:
        super().__init__()
        builders = {
            "x_400mf": tvm.regnet_x_400mf,
            "y_400mf": tvm.regnet_y_400mf,
        }
        if variant not in builders:
            raise ValueError(f"Unsupported RegNet variant: {variant}")
        net = builders[variant](weights=None, num_classes=num_classes)
        # CIFAR stem, retaining the RegNet stage design.
        stem_out = net.stem[0].out_channels
        net.stem[0] = nn.Conv2d(3, stem_out, 3, stride=1, padding=1, bias=False)
        self.stem = net.stem
        self.trunk_output = net.trunk_output
        self.avgpool = net.avgpool
        self.fc = net.fc
        self.feature_dim = self.fc.in_features

    def forward(self, x: Tensor, is_feat: bool = False):
        f0 = self.stem(x)
        feats: list[Tensor] = [f0]
        out = f0
        for stage in self.trunk_output:
            out = stage(out)
            feats.append(out)
        pooled = self.avgpool(out).flatten(1)
        logits = self.fc(pooled)
        if is_feat:
            return [*feats, pooled], logits
        return logits


def ShuffleV2(num_classes: int = 100, **_: object) -> ShuffleNetV2CIFAR:
    return ShuffleNetV2CIFAR(num_classes=num_classes, width=0.5)


def MobileNetV2(num_classes: int = 100, **_: object) -> MobileNetV2CIFAR:
    return MobileNetV2CIFAR(num_classes=num_classes)


def RegNetX_400MF(num_classes: int = 100, **_: object) -> RegNetCIFAR:
    return RegNetCIFAR("x_400mf", num_classes=num_classes)


def RegNetY_400MF(num_classes: int = 100, **_: object) -> RegNetCIFAR:
    return RegNetCIFAR("y_400mf", num_classes=num_classes)
