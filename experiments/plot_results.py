from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

from mpcrl.plotting import outside_legend


METRICS = [
    ('episode_return', 'Episode return'),
    ('mpc_ms', 'MPC planning time / ms'),
    ('action_d1', 'First-order action variation'),
    ('prediction_mse', 'One-step prediction MSE'),
    ('effective_residual_norm', 'Effective residual norm'),
    ('gate', 'Effective gate'),
]


def set_english_paper_style():
    """Paper plotting style without CJK glyph warnings on headless Ubuntu."""
    names = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams['font.family'] = 'Times New Roman' if 'Times New Roman' in names else 'DejaVu Serif'
    plt.rcParams['font.size'] = 10.5
    plt.rcParams['axes.unicode_minus'] = False


def load_best_runs(root):
    """Keep only the longest/newest run for each (env, method, seed)."""
    best = {}
    for f in glob.glob(os.path.join(root, '**', 'episodes.csv'), recursive=True):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        required = {'env', 'method', 'seed', 'global_step'}
        if df.empty or not required.issubset(df.columns):
            continue
        df = df.copy()
        df['global_step'] = pd.to_numeric(df['global_step'], errors='coerce')
        df = df.dropna(subset=['env', 'method', 'seed', 'global_step'])
        if df.empty:
            continue
        last = df.iloc[-1]
        key = (str(last['env']), str(last['method']), int(float(last['seed'])))
        score = (float(df['global_step'].max()), Path(f).stat().st_mtime)
        if key not in best or score > best[key]['score']:
            best[key] = {
                'env': key[0],
                'method': key[1],
                'seed': key[2],
                'run_dir': Path(f).parent,
                'df': df,
                'score': score,
            }
    return list(best.values())


def smooth_run(df, metric, window):
    if metric not in df.columns:
        return None
    q = df[['global_step', metric]].copy()
    q['global_step'] = pd.to_numeric(q['global_step'], errors='coerce')
    q[metric] = pd.to_numeric(q[metric], errors='coerce')
    q = q.dropna().groupby('global_step', as_index=False)[metric].mean().sort_values('global_step')
    if len(q) < 2:
        return None
    q[metric] = q[metric].rolling(max(1, int(window)), min_periods=1, center=True).mean()
    return q


def align_seed_curves(runs, metric, points, window):
    prepared = []
    for r in runs:
        q = smooth_run(r['df'], metric, window)
        if q is not None:
            prepared.append((r, q))
    if not prepared:
        return None

    xmax = max(float(q['global_step'].max()) for _, q in prepared)
    xmin = min(float(q['global_step'].min()) for _, q in prepared)
    grid = np.linspace(xmin, xmax, max(20, int(points)))
    values = []
    seeds = []
    for r, q in prepared:
        x = q['global_step'].to_numpy(dtype=float)
        y = q[metric].to_numpy(dtype=float)
        yi = np.interp(grid, x, y)
        yi[(grid < x.min()) | (grid > x.max())] = np.nan
        values.append(yi)
        seeds.append(r['seed'])
    return grid, np.asarray(values, dtype=float), seeds


def save_curve_data(path, grid, values, seeds):
    data = {'global_step': grid}
    for seed, y in zip(seeds, values):
        data[f'seed_{seed}'] = y
    n = np.sum(np.isfinite(values), axis=0)
    data['n_seeds'] = n
    data['mean'] = np.nanmean(values, axis=0)
    if values.shape[0] > 1:
        data['std'] = np.nanstd(values, axis=0, ddof=1)
    else:
        data['std'] = np.full_like(grid, np.nan, dtype=float)
    pd.DataFrame(data).to_csv(path, index=False)


def plot_training_curves(records, outdir, points, window, min_steps):
    data_dir = Path(outdir) / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    envs = sorted({r['env'] for r in records})
    for env in envs:
        env_runs = [r for r in records if r['env'] == env and r['score'][0] >= min_steps]
        if not env_runs:
            continue
        safe_env = env.replace('/', '_')
        for metric, ylabel in METRICS:
            fig, ax = plt.subplots(figsize=(6.4, 4.2))
            plotted = False
            for method in sorted({r['method'] for r in env_runs}):
                mruns = [r for r in env_runs if r['method'] == method]
                aligned = align_seed_curves(mruns, metric, points, window)
                if aligned is None:
                    continue
                grid, values, seeds = aligned
                mean = np.nanmean(values, axis=0)
                count = np.sum(np.isfinite(values), axis=0)
                std = np.full_like(mean, np.nan)
                valid_multi = count >= 2
                if values.shape[0] >= 2:
                    with np.errstate(invalid='ignore', divide='ignore'):
                        std_all = np.nanstd(values, axis=0, ddof=1)
                    std[valid_multi] = std_all[valid_multi]

                line, = ax.plot(grid, mean, label=f'{method} (n={len(seeds)})')
                if np.any(valid_multi):
                    ax.fill_between(
                        grid,
                        mean - std,
                        mean + std,
                        where=valid_multi,
                        alpha=0.18,
                        color=line.get_color(),
                    )
                save_curve_data(
                    data_dir / f'{safe_env}__{method}__{metric}.csv',
                    grid,
                    values,
                    seeds,
                )
                plotted = True

            if not plotted:
                plt.close(fig)
                continue
            ax.set_xlabel('Environment steps')
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            outside_legend(ax)
            fig.tight_layout()
            stem = Path(outdir) / f'{safe_env}__{metric}'
            fig.savefig(str(stem) + '.png', dpi=300, bbox_inches='tight')
            fig.savefig(str(stem) + '.pdf', bbox_inches='tight')
            plt.close(fig)


def plot_final_eval(records, outdir, min_steps):
    rows = []
    for r in records:
        if r['score'][0] < min_steps:
            continue
        f = r['run_dir'] / 'eval.csv'
        if not f.exists():
            continue
        try:
            e = pd.read_csv(f)
        except Exception:
            continue
        vals = pd.to_numeric(e.get('episode_return'), errors='coerce').dropna()
        if len(vals):
            rows.append({
                'env': r['env'],
                'method': r['method'],
                'seed': r['seed'],
                'eval_return': float(vals.mean()),
                'eval_episode_std': float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                'eval_episodes': int(len(vals)),
            })
    if not rows:
        return

    q = pd.DataFrame(rows)
    q.to_csv(Path(outdir) / 'final_eval_per_seed.csv', index=False)
    stats = q.groupby(['env', 'method'])['eval_return'].agg(['mean', 'std', 'count']).reset_index()
    stats.to_csv(Path(outdir) / 'final_eval_mean_std.csv', index=False)

    for env, g in stats.groupby('env'):
        g = g.sort_values('method').reset_index(drop=True)
        x = np.arange(len(g))
        yerr = g['std'].fillna(0.0).to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.bar(x, g['mean'].to_numpy(dtype=float), yerr=yerr, capsize=4)
        labels = [f"{m}\n(n={int(n)})" for m, n in zip(g['method'], g['count'])]
        ax.set_xticks(x, labels)
        ax.set_ylabel('Deterministic final evaluation return')
        ax.grid(axis='y', alpha=0.25)
        fig.tight_layout()
        safe_env = env.replace('/', '_')
        stem = Path(outdir) / f'{safe_env}__final_eval_return'
        fig.savefig(str(stem) + '.png', dpi=300, bbox_inches='tight')
        fig.savefig(str(stem) + '.pdf', bbox_inches='tight')
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='runs')
    p.add_argument('--summary', help='kept for backward compatibility; plots are built from run files')
    p.add_argument('--outdir', default='results/figures')
    p.add_argument('--smooth-window', type=int, default=10)
    p.add_argument('--points', type=int, default=300)
    p.add_argument(
        '--min-steps',
        type=int,
        default=0,
        help='only plot selected runs whose last completed episode reaches at least this step',
    )
    a = p.parse_args()

    set_english_paper_style()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    records = load_best_runs(a.root)
    if not records:
        raise SystemExit('No valid episodes.csv found')

    plot_training_curves(records, a.outdir, a.points, a.smooth_window, a.min_steps)
    plot_final_eval(records, a.outdir, a.min_steps)
    print(f'Paper figures saved to: {a.outdir}')


if __name__ == '__main__':
    main()
