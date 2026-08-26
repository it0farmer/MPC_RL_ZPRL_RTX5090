#!/usr/bin/env bash
set -euo pipefail
python -m experiments.quick_diagnostic \
  --config configs/rtx5090/halfcheetah.yaml \
  --steps 10000 \
  --seeds 0
