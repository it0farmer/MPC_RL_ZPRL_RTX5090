#!/usr/bin/env bash
set -euo pipefail

python -m experiments.run_suite --suite configs/rtx5090/paper_suite.yaml "$@"
python -m experiments.aggregate --root runs --out results/summary.csv --tail 20
python -m experiments.plot_results --summary results/summary.csv --outdir results/figures
