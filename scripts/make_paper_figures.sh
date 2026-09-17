#!/usr/bin/env bash
set -euo pipefail
python -m experiments.aggregate --root "${RUNS_ROOT:-runs}" --out "${SUMMARY:-results/summary.csv}"
python -m experiments.paper_figures \
  --summary "${SUMMARY_PER_SEED:-results/summary_per_seed.csv}" \
  --runs "${RUNS_ROOT:-runs}" \
  --out "${FIG_OUT:-results/paper_figures}" \
  --smooth-episodes "${SMOOTH_EPISODES:-20}"
