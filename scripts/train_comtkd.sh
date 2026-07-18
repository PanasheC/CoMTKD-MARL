#!/usr/bin/env bash
set -euo pipefail
python train_student_rl.py \
  --data "${1:-./data}" \
  --arch ShuffleV2 \
  --teacher-name-list RegNetY_400MF RegNetX_400MF resnet32x4 wrn_28_4 \
  --checkpoint-dir "${2:-./checkpoints/comtkd_marl}" \
  --dynamic \
  --amp
