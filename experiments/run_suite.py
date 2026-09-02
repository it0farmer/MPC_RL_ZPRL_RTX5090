from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from tqdm.auto import tqdm

from mpcrl.config import load_yaml


def build_jobs(suite, include_lewm=False):
    jobs = []
    eval_episodes = int(suite.get('eval_episodes', 5))
    for cfg in suite['tasks']:
        env_name = Path(cfg).stem
        for method in suite['methods']:
            for seed in suite['seeds']:
                if method == 'zprl_style':
                    cmd = [
                        sys.executable, '-m', 'experiments.train_zprl_style',
                        '--config', cfg,
                        '--seed', str(seed),
                        '--steps', str(suite['steps']),
                        '--eval-episodes', str(eval_episodes),
                    ]
                else:
                    cmd = [
                        sys.executable, '-m', 'experiments.train',
                        '--config', cfg,
                        '--method', method,
                        '--seed', str(seed),
                        '--steps', str(suite['steps']),
                        '--eval-episodes', str(eval_episodes),
                    ]
                jobs.append((env_name, method, seed, cmd))

    if include_lewm or suite.get('run_lewm', False):
        for cfg in suite.get('lewm_configs', []):
            env_name = Path(cfg).stem.replace('lewm_', '')
            for seed in suite['seeds']:
                cmd = [
                    sys.executable, '-m', 'experiments.train_lewm',
                    '--config', cfg,
                    '--seed', str(seed),
                ]
                jobs.append((env_name, 'lewm_mpc', seed, cmd))
    return jobs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--suite', default='configs/paper_suite.yaml')
    p.add_argument('--include-lewm', action='store_true')
    p.add_argument(
        '--start-job',
        type=int,
        default=1,
        help='1-based job index; useful after an interrupted suite',
    )
    a = p.parse_args()

    suite = load_yaml(a.suite)
    jobs = build_jobs(suite, include_lewm=a.include_lewm)
    total_jobs = len(jobs)
    start = max(1, int(a.start_job))
    if start > total_jobs:
        raise SystemExit(f'--start-job={start} exceeds total jobs={total_jobs}')

    print(
        f"Paper suite: {total_jobs} jobs, starting from job {start}, "
        f"final eval={int(suite.get('eval_episodes', 5))} episodes/run"
    )
    bar = tqdm(
        total=total_jobs,
        initial=start - 1,
        desc='paper suite',
        unit='job',
        dynamic_ncols=True,
    )

    for job_idx, (env_name, method, seed, cmd) in enumerate(jobs, start=1):
        if job_idx < start:
            continue
        bar.set_postfix_str(f'#{job_idx} {env_name}/{method}/seed{seed}')
        tqdm.write(
            f'\n=== JOB {job_idx}/{total_jobs}: '
            f'{env_name} | {method} | seed={seed} ==='
        )
        subprocess.run(cmd, check=True)
        bar.update(1)

    bar.close()
    print('Paper suite completed.')


if __name__ == '__main__':
    main()
