#!/usr/bin/env bash
set -e
for seed in 0 1 2 3 4; do
  python -m experiments.ablation --config configs/halfcheetah.yaml --steps 100000 --seed "$seed"
done
