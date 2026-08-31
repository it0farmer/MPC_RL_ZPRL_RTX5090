from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd


def first_success_step(g):
    if 'success' not in g:
        return np.nan
    s = pd.to_numeric(g['success'], errors='coerce')
    idx = s[s > 0.5].index
    return float(g.loc[idx[0], 'global_step']) if len(idx) else np.nan


def load_runs(root):
    records = []
    for f in glob.glob(os.path.join(root, '**', 'episodes.csv'), recursive=True):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if df.empty or not {'env', 'method', 'seed', 'global_step'}.issubset(df.columns):
            continue
        df = df[df['env'].notna() & df['method'].notna() & df['seed'].notna()].copy()
        if df.empty:
            continue
        df['global_step'] = pd.to_numeric(df['global_step'], errors='coerce')
        df = df[df['global_step'].notna()]
        if df.empty:
            continue
        last = df.iloc[-1]
        records.append({
            'path': f,
            'env': str(last['env']),
            'method': str(last['method']),
            'seed': int(float(last['seed'])),
            'max_step': float(df['global_step'].max()),
            'mtime': Path(f).stat().st_mtime,
            'df': df,
        })
    return records


def choose_one_run_per_seed(records):
    """Avoid double-counting repeated/restarted runs.

    Prefer the run with the largest completed global_step. If two runs reached
    the same step, keep the newest file.
    """
    best = {}
    for r in records:
        key = (r['env'], r['method'], r['seed'])
        score = (r['max_step'], r['mtime'])
        if key not in best or score > (best[key]['max_step'], best[key]['mtime']):
            best[key] = r
    return list(best.values())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='runs')
    p.add_argument('--out', default='results/summary.csv')
    p.add_argument('--tail', type=int, default=20,
                   help='aggregate the final N episodes of each selected run')
    a = p.parse_args()

    records = choose_one_run_per_seed(load_runs(a.root))
    if not records:
        raise SystemExit('No valid episodes.csv found')

    cols = [
        'episode_return', 'success', 'mpc_ms', 'prediction_mse', 'action_d1',
        'action_d2', 'residual_norm', 'effective_residual_norm', 'gate',
        'adaptive_gate', 'gate_z', 'residual_ramp', 'mpc_cache_hit_rate',
        'episode_length',
    ]
    rows = []
    for r in records:
        g = r['df'].tail(max(1, int(a.tail))).reset_index(drop=True)
        row = {
            'env': r['env'],
            'method': r['method'],
            'seed': r['seed'],
            'global_step': r['max_step'],
            'episodes_used': len(g),
            'run_path': r['path'],
        }
        for col in cols:
            row[col] = (
                pd.to_numeric(g[col], errors='coerce').mean()
                if col in g else np.nan
            )
        row['sample_efficiency_step'] = first_success_step(r['df'].reset_index(drop=True))
        rows.append(row)

    per_seed = pd.DataFrame(rows).sort_values(['env', 'method', 'seed'])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    per_seed_path = out.with_name(out.stem + '_per_seed.csv')
    per_seed.to_csv(per_seed_path, index=False)

    numeric = [c for c in per_seed.columns if c not in {'env', 'method', 'run_path'}]
    summary = per_seed.groupby(['env', 'method'], dropna=False)[numeric].agg(['mean', 'std']).reset_index()
    summary.to_csv(out, index=False)
    print(per_seed[['env', 'method', 'seed', 'global_step', 'episode_return', 'episode_length']].to_string(index=False))
    print('\nmean±std summary:\n', summary.to_string(index=False))
    print('saved', out)
    print('saved', per_seed_path)


if __name__ == '__main__':
    main()
