from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--glob', dest='pattern', required=True, help='Quoted run-directory glob')
    p.add_argument('--out')
    a = p.parse_args()

    dirs = [Path(x) for x in sorted(glob.glob(a.pattern)) if Path(x).is_dir()]
    frames = []
    for d in dirs:
        f = d / 'eval.csv'
        if not f.exists():
            continue
        q = pd.read_csv(f)
        q['run_dir'] = str(d)
        frames.append(q)

    if not frames:
        raise SystemExit('No eval.csv found for the supplied glob')

    df = pd.concat(frames, ignore_index=True)
    per_seed = (
        df.groupby(['method', 'train_seed'], as_index=False)
        .agg(
            eval_return_mean=('episode_return', 'mean'),
            eval_return_std=('episode_return', 'std'),
            eval_episodes=('episode_return', 'size'),
            eval_length_mean=('episode_length', 'mean'),
            eval_action_d1_mean=('action_d1', 'mean'),
            eval_gate_mean=('gate', 'mean'),
        )
    )

    base = per_seed.loc[per_seed.method == 'mpc_only', 'eval_return_mean']
    base_mean = float(base.mean()) if len(base) else np.nan
    rows = []
    for method, g in per_seed.groupby('method'):
        vals = g.eval_return_mean.astype(float)
        mean = float(vals.mean())
        rows.append({
            'method': method,
            'seeds': len(vals),
            'eval_return_mean': mean,
            'eval_return_std_across_seeds': float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            'vs_mpc_pct': (
                (mean - base_mean) / max(abs(base_mean), 1e-9) * 100.0
                if np.isfinite(base_mean) else np.nan
            ),
        })
    comparison = pd.DataFrame(rows).sort_values('eval_return_mean', ascending=False)

    print('\nPer-seed deterministic evaluation:')
    print(per_seed.to_string(index=False))
    print('\nComparison:')
    print(comparison.to_string(index=False))

    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / 'eval_all.csv', index=False)
        per_seed.to_csv(out / 'eval_per_seed.csv', index=False)
        comparison.to_csv(out / 'comparison_eval.csv', index=False)
        print('saved', out)


if __name__ == '__main__':
    main()
