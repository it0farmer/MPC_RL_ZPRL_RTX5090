#!/usr/bin/env bash
set -euo pipefail
python scripts/check_rtx5090.py
python -m unittest discover -s tests -v
python -m experiments.train --config configs/rtx5090/halfcheetah.yaml --method planning_residual --steps 2000 --run-name rtx5090_smoke
