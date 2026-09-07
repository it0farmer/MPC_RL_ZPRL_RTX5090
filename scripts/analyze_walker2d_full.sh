#!/usr/bin/env bash
set -euo pipefail

# Final Walker2d paper analysis from completed 100k runs.
# Usage:
#   bash scripts/analyze_walker2d_full.sh
# Optional:
#   ROOT=runs OUT=results/walker2d_final MIN_STEPS=90000 \
#   bash scripts/analyze_walker2d_full.sh

cd "$(dirname "$0")/.."
export MPLBACKEND=Agg

ROOT="${ROOT:-runs}"
OUT="${OUT:-results/walker2d_final}"
MIN_STEPS="${MIN_STEPS:-90000}"

mkdir -p "$OUT"

python -m experiments.aggregate \
  --root "$ROOT" \
  --out "$OUT/summary.csv" \
  --env Walker2d-v5 \
  --methods mpc_only action_residual planning_residual zprl_style \
  --min-steps "$MIN_STEPS" \
  --require-eval \
  --expected-seeds 0 1 2 3 4 \
  --require-complete \
  --tail 20

python -m experiments.summarize_eval \
  --root "$ROOT" \
  --out "$OUT/eval_summary" \
  --env Walker2d-v5 \
  --methods mpc_only action_residual planning_residual zprl_style \
  --min-steps "$MIN_STEPS" \
  --expected-seeds 0 1 2 3 4 \
  --require-complete

python -m experiments.plot_results \
  --root "$ROOT" \
  --outdir "$OUT/figures" \
  --env Walker2d-v5 \
  --methods mpc_only action_residual planning_residual zprl_style \
  --min-steps "$MIN_STEPS" \
  --expected-methods mpc_only action_residual planning_residual zprl_style \
  --expected-seeds 0 1 2 3 4 \
  --require-complete \
  --smooth-window 10 \
  --points 400 \
  --error-band std

echo
echo "Walker2d final analysis completed."
echo "Summary:      $OUT/summary.csv"
echo "Per seed:     $OUT/summary_per_seed.csv"
echo "Completeness: $OUT/summary_completeness.csv"
echo "Eval summary: $OUT/eval_summary/comparison_eval.csv"
echo "Eval coverage:$OUT/eval_summary/eval_coverage.csv"
echo "Figures:      $OUT/figures"
