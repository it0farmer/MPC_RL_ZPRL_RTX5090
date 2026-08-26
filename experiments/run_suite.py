from __future__ import annotations
import argparse, subprocess, sys
from mpcrl.config import load_yaml


def run(cmd):
    print('RUN',' '.join(cmd),flush=True); subprocess.run(cmd,check=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--suite',default='configs/paper_suite.yaml'); p.add_argument('--include-lewm',action='store_true'); a=p.parse_args(); s=load_yaml(a.suite)
    for cfg in s['tasks']:
        for method in s['methods']:
            for seed in s['seeds']:
                if method=='zprl_style': cmd=[sys.executable,'-m','experiments.train_zprl_style','--config',cfg,'--seed',str(seed),'--steps',str(s['steps'])]
                else: cmd=[sys.executable,'-m','experiments.train','--config',cfg,'--method',method,'--seed',str(seed),'--steps',str(s['steps'])]
                run(cmd)
    if a.include_lewm or s.get('run_lewm',False):
        for cfg in s.get('lewm_configs',[]):
            for seed in s['seeds']: run([sys.executable,'-m','experiments.train_lewm','--config',cfg,'--seed',str(seed)])
if __name__=='__main__': main()
