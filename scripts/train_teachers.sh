#!/usr/bin/env bash
set -euo pipefail
DATA=${1:-./data}
OUT=${2:-./checkpoints/teachers}
for MODEL in RegNetY_400MF RegNetX_400MF resnet32x4 wrn_28_4; do
  python train_baseline.py \
    --model "$MODEL" \
    --data-folder "$DATA" \
    --checkpoint-dir "$OUT" \
    --amp
done
