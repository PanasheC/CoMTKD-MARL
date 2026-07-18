"""Default checkpoint registry.

Populate this mapping after training teachers, or pass one or more
``--teacher-checkpoint MODEL=PATH`` arguments to the student scripts.
"""
from __future__ import annotations

teacher_model_path_dict: dict[str, str] = {
    "RegNetY_400MF": "./checkpoints/teachers/RegNetY_400MF_best.pth.tar",
    "RegNetX_400MF": "./checkpoints/teachers/RegNetX_400MF_best.pth.tar",
    "resnet32x4": "./checkpoints/teachers/resnet32x4_best.pth.tar",
    "wrn_28_4": "./checkpoints/teachers/wrn_28_4_best.pth.tar",
}
