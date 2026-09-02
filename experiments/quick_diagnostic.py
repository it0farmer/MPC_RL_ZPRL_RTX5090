from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mpcrl.config import load_yaml
from mpcrl.plotting import outside_legend, set_paper_style


METHODS = ('mpc_only', 'action_residual', 'planning_residual')


def run(cmd):
    print('RUN', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def numeric_mean(g, col):
    return float(pd.to_numeric(g[col], errors='coerce').mean()) if col in g else np.nan


def numeric_last(g, col):
    if col not in g:
        return np.nan
    s = pd.to_numeric(g[col], errors='coerce').dropna()
    return float(s.iloc[-1]) if len(s) else np.nan


def build_summary(df, requested_steps):
    rows = []
    for (method, seed), g in df.groupby(['method', 'seed'], dropna=False):
        g = g.sort_values('global_step')
        ret = pd.to_numeric(g['episode_return'], errors='coerce')
        final_logged_step = int(pd.to_numeric(g['global_step'], errors='coerce').max())
        rows.append({
            'method': method,
            'seed': int(seed),
            'episodes': int(len(g)),
            'requested_steps': int(requested_steps),
            'last_completed_episode_step': final_logged_step,
            'completion_pct': 100.0,
            'return_mean': float(ret.mean()),
            'return_last5_mean': float(ret.tail(5).mean()),
            'return_best': float(ret.max()),
            'mpc_ms_mean': numeric_mean(g, 'mpc_ms'),
            'prediction_mse_last': numeric_last(g, 'prediction_mse'),
            'action_d1_mean': numeric_mean(g, 'action_d1'),
            'action_d2_mean': numeric_mean(g, 'action_d2'),
            'residual_norm_mean': numeric_mean(g, 'residual_norm'),
            'effective_residual_norm_mean': numeric_mean(g, 'effective_residual_norm'),
            'gate_mean': numeric_mean(g, 'gate'),
            'adaptive_gate_mean': numeric_mean(g, 'adaptive_gate'),
            'residual_ramp_mean': numeric_mean(g, 'residual_ramp'),
            'mpc_cache_hit_rate_mean': numeric_mean(g, 'mpc_cache_hit_rate'),
            'wm_loss_last': numeric_last(g, 'wm_loss'),
        })
    return pd.DataFrame(rows)


def build_comparison(per_seed):
    metric = 'return_last5_mean'
    rows = []
    base = per_seed[per_seed['method'] == 'mpc_only'][metric]
    base_mean = float(base.mean()) if len(base) else np.nan
    for method, g in per_seed.groupby('method'):
        vals = pd.to_numeric(g[metric], errors='coerce')
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
        rel = (
            (mean - base_mean) / max(abs(base_mean), 1e-9) * 100.0
            if np.isfinite(base_mean) else np.nan
        )
        rows.append({
            'method': method,
            'seeds': int(len(vals)),
            'last5_return_mean': mean,
            'last5_return_std': std,
            'vs_mpc_pct': rel,
        })
    return pd.DataFrame(rows).sort_values('last5_return_mean', ascending=False)


def plot_metrics(df, outdir):
    set_paper_style()
    outdir.mkdir(parents=True, exist_ok=True)
    # Use English labels on headless Ubuntu to avoid repeated CJK glyph warnings
    # when Songti/Noto CJK is not installed on the server.
    metrics = [
        ('episode_return', 'Episode return'),
        ('mpc_ms', 'MPC planning time / ms'),
        ('prediction_mse', 'One-step prediction MSE'),
        ('action_d1', 'First-order action variation'),
        ('effective_residual_norm', 'Effective residual norm'),
        ('gate', 'Effective gate'),
    ]
    for metric, ylabel in metrics:
        if metric not in df:
            continue
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for method, g in df.groupby('method'):
            q = g[['global_step', metric]].copy()
            q['global_step'] = pd.to_numeric(q['global_step'], errors='coerce')
            q[metric] = pd.to_numeric(q[metric], errors='coerce')
            q = (
                q.dropna()
                .groupby('global_step', as_index=False)[metric]
                .mean()
                .sort_values('global_step')
            )
            if len(q):
                ax.plot(q['global_step'], q[metric], marker='o', label=method)
        ax.set_xlabel('Environment steps')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.25)
        outside_legend(ax)
        fig.tight_layout()
        fig.savefig(outdir / f'{metric}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/rtx5090/halfcheetah.yaml')
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--tag')
    a = p.parse_args()

    cfg = load_yaml(a.config)
    root = Path(cfg['logging'].get('root', 'runs'))
    env_id = cfg['env']['id']
    tag = a.tag or datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = Path('results') / 'diagnostic_10k' / tag
    outdir.mkdir(parents=True, exist_ok=True)
    run_dirs = []

    for method in METHODS:
        for seed in a.seeds:
            name = f'diagnostic10k__{env_id}__{method}__seed{seed}__{tag}'
            run([
                sys.executable, '-m', 'experiments.train',
                '--config', a.config,
                '--method', method,
                '--steps', str(a.steps),
                '--seed', str(seed),
                '--run-name', name,
            ])
            run_dirs.append(root / name)

    frames = []
    for d in run_dirs:
        f = d / 'episodes.csv'
        q = pd.read_csv(f)
        q['run_dir'] = str(d)
        frames.append(q)

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(outdir / 'episodes_all.csv', index=False)

    per_seed = build_summary(df, a.steps)
    per_seed.to_csv(outdir / 'summary_per_seed.csv', index=False)

    comparison = build_comparison(per_seed)
    comparison.to_csv(outdir / 'comparison.csv', index=False)

    numeric_cols = per_seed.select_dtypes(include=[np.number]).columns
    aggregate = per_seed.groupby('method')[numeric_cols].agg(['mean', 'std'])
    aggregate.to_csv(outdir / 'summary_mean_std.csv')

    plot_metrics(df, outdir / 'figures')

    print('\nPer-seed summary:')
    print(per_seed.to_string(index=False))
    print('\nFinal-window comparison (last 5 completed episodes per seed):')
    print(comparison.to_string(index=False))
    print('Saved:', outdir)


if __name__ == '__main__':
    main()
