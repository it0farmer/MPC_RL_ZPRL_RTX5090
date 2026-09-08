#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export MPLBACKEND=Agg

TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"
START_JOB="${START_JOB:-1}"
SUITE="${SUITE:-configs/rtx5090/ablation_suite.yaml}"
RUN_ROOT="runs/zprl_ablation_${TAG}"
RESULT_ROOT="results/zprl_ablation_${TAG}"
LOG="${RESULT_ROOT}/ablation.log"

mkdir -p "$RUN_ROOT" "$RESULT_ROOT"

{
  echo "TAG=$TAG"
  echo "GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "SUITE=$SUITE"
  echo "START_JOB=$START_JOB"
  python --version 2>&1
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
} > "${RESULT_ROOT}/environment.txt"

python -m experiments.run_zprl_ablation \
  --suite "$SUITE" \
  --run-root "$RUN_ROOT" \
  --result-root "$RESULT_ROOT" \
  --start-job "$START_JOB" 2>&1 | tee "$LOG"

python -m experiments.plot_ablation \
  --root "$RUN_ROOT" \
  --outdir "$RESULT_ROOT/figures" \
  --expected-seeds 0 1 2 \
  --require-complete 2>&1 | tee -a "$LOG"

echo
echo "ZPRL ablations completed."
echo "Runs:    $RUN_ROOT"
echo "Results: $RESULT_ROOT"
echo "Summary: $RESULT_ROOT/figures/ablation_summary.csv"
echo "Figures: $RESULT_ROOT/figures"
