"""Checkpoint input and output helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    epoch: int,
    metrics: dict[str, float],
    optimizer: torch.optim.Optimizer | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "epoch": epoch,
        "model": model.state_dict(),
        "metrics": metrics,
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if extra:
        state.update(extra)
    torch.save(state, path)


def load_model_checkpoint(
    model: nn.Module,
    path: str | Path,
    device: torch.device,
    strict: bool = True,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=strict)
    return checkpoint
