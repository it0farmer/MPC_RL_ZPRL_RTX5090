#!/usr/bin/env bash
set -e
python -m experiments.run_suite --suite configs/paper_suite.yaml
python -m experiments.aggregate --root runs --out results/summary.csv
python -m experiments.plot_results --root runs --outdir results/figures
