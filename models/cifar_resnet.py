"""CIFAR ResNet backbones with feature-return interfaces.

The module follows the interface used by MTKD research repositories:
``model(x, is_feat=True) -> (feature_list, logits)``.
"""
from __future__ import annotations

from typing import Callable, Type

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + self.shortcut(x), inplace=True)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        out = F.relu(out + self.shortcut(x), inplace=True)
        return out


class CIFARResNet(nn.Module):
    """ResNet designed for 32 by 32 images."""

    def __init__(
        self,
        block: Type[BasicBlock] | Type[Bottleneck],
        blocks_per_stage: list[int],
        num_classes: int = 100,
        width: int = 1,
    ) -> None:
        super().__init__()
        stem_width = 16 * width
        self.in_planes = stem_width
        self.conv1 = nn.Conv2d(3, stem_width, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(stem_width)
        self.layer1 = self._make_layer(block, 16 * width, blocks_per_stage[0], stride=1)
        self.layer2 = self._make_layer(block, 32 * width, blocks_per_stage[1], stride=2)
        self.layer3 = self._make_layer(block, 64 * width, blocks_per_stage[2], stride=2)
        self.feature_dim = 64 * width * block.expansion
        self.fc = nn.Linear(self.feature_dim, num_classes)
        self._initialize_weights()

    def _make_layer(
        self,
        block: Type[BasicBlock] | Type[Bottleneck],
        planes: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (blocks - 1)
        layers: list[nn.Module] = []
        for block_stride in strides:
            layers.append(block(self.in_planes, planes, block_stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor, is_feat: bool = False):
        f0 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        f1 = self.layer1(f0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        pooled = F.adaptive_avg_pool2d(f3, 1).flatten(1)
        logits = self.fc(pooled)
        if is_feat:
            return [f0, f1, f2, f3, pooled], logits
        return logits


def _basic(depth: int, width: int = 1, num_classes: int = 100) -> CIFARResNet:
    if (depth - 2) % 6 != 0:
        raise ValueError(f"Basic CIFAR ResNet depth must satisfy (depth - 2) % 6 == 0, got {depth}")
    n = (depth - 2) // 6
    return CIFARResNet(BasicBlock, [n, n, n], num_classes=num_classes, width=width)


def resnet20(num_classes: int = 100, **_: object) -> CIFARResNet:
    return _basic(20, num_classes=num_classes)


def resnet32(num_classes: int = 100, **_: object) -> CIFARResNet:
    return _basic(32, num_classes=num_classes)


def resnet56(num_classes: int = 100, **_: object) -> CIFARResNet:
    return _basic(56, num_classes=num_classes)


def resnet110(num_classes: int = 100, **_: object) -> CIFARResNet:
    return _basic(110, num_classes=num_classes)


def resnet32x4(num_classes: int = 100, **_: object) -> CIFARResNet:
    return _basic(32, width=4, num_classes=num_classes)
