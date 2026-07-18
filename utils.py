"""Compatibility utilities used by the root training scripts."""
from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
from torch import Tensor, nn

from distiller_zoo import DistillKL
from helper.logger import set_logger
from helper.metrics import AverageMeter, topk_correct


def cal_param_size(model: nn.Module) -> float:
    return sum(parameter.numel() for parameter in model.parameters()) / 1e6


def cal_multi_adds(model: nn.Module, input_size: tuple[int, ...] = (1, 3, 32, 32)) -> float:
    """Return a conservative placeholder when a FLOP profiler is unavailable.

    The function is retained for compatibility. Use ptflops or fvcore for a
    publication-grade operation count.
    """
    del model, input_size
    return float("nan")


def correct_num(logits: Tensor, targets: Tensor, topk: tuple[int, ...] = (1, 5)):
    return topk_correct(logits, targets, topk)


def adjust_lr(optimizer: torch.optim.Optimizer, epoch: int, args) -> float:
    base_lr = float(getattr(args, "lr", getattr(args, "init_lr", 0.1)))
    lr_type = getattr(args, "lr_type", "cosine")
    epochs = max(int(getattr(args, "epochs", 1)), 1)
    if lr_type == "cosine":
        lr = 0.5 * base_lr * (1.0 + math.cos(math.pi * epoch / epochs))
    else:
        milestones = getattr(args, "milestones", [150, 180, 210])
        decay_count = sum(epoch >= milestone for milestone in milestones)
        lr = base_lr * (0.1**decay_count)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


__all__ = [
    "AverageMeter",
    "DistillKL",
    "adjust_lr",
    "cal_multi_adds",
    "cal_param_size",
    "correct_num",
    "set_logger",
]
