#!/usr/bin/env bash
set -euo pipefail
python experiments/cardinality_sweep.py \
  --max-teachers 4 \
  --seeds 11 22 33 44 55 \
  --output-root ./checkpoints/cardinality_sweep \
  --extra --data ./data --arch ShuffleV2 --amp
