"""Wide ResNet for CIFAR-100."""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class WideBasicBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, dropout: float, stride: int) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, stride=stride, padding=1, bias=False)
        self.shortcut = (
            nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)
            if stride != 1 or in_planes != out_planes
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        return out + self.shortcut(x)


class WideResNet(nn.Module):
    def __init__(
        self,
        depth: int = 28,
        widen_factor: int = 4,
        dropout: float = 0.0,
        num_classes: int = 100,
    ) -> None:
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError("WideResNet depth must satisfy (depth - 4) % 6 == 0")
        n = (depth - 4) // 6
        widths = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        self.conv1 = nn.Conv2d(3, widths[0], 3, padding=1, bias=False)
        self.block1 = self._make_group(n, widths[0], widths[1], dropout, stride=1)
        self.block2 = self._make_group(n, widths[1], widths[2], dropout, stride=2)
        self.block3 = self._make_group(n, widths[2], widths[3], dropout, stride=2)
        self.bn = nn.BatchNorm2d(widths[3])
        self.feature_dim = widths[3]
        self.fc = nn.Linear(widths[3], num_classes)
        self._initialize_weights()

    @staticmethod
    def _make_group(
        n: int, in_planes: int, out_planes: int, dropout: float, stride: int
    ) -> nn.Sequential:
        layers: list[nn.Module] = [WideBasicBlock(in_planes, out_planes, dropout, stride)]
        layers.extend(WideBasicBlock(out_planes, out_planes, dropout, 1) for _ in range(n - 1))
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor, is_feat: bool = False):
        f0 = self.conv1(x)
        f1 = self.block1(f0)
        f2 = self.block2(f1)
        f3 = F.relu(self.bn(self.block3(f2)), inplace=True)
        pooled = F.adaptive_avg_pool2d(f3, 1).flatten(1)
        logits = self.fc(pooled)
        if is_feat:
            return [f0, f1, f2, f3, pooled], logits
        return logits


def wrn_28_4(num_classes: int = 100, **_: object) -> WideResNet:
    return WideResNet(depth=28, widen_factor=4, num_classes=num_classes)


def wrn_40_2(num_classes: int = 100, **_: object) -> WideResNet:
    return WideResNet(depth=40, widen_factor=2, num_classes=num_classes)
