from __future__ import annotations
import argparse,glob,os
import numpy as np, pandas as pd


def first_success_step(g):
    s=g['success'].dropna()
    if len(s)==0: return np.nan
    idx=s[s>0.5].index
    return float(g.loc[idx[0],'global_step']) if len(idx) else np.nan


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',default='runs'); p.add_argument('--out',default='results/summary.csv'); a=p.parse_args(); fs=glob.glob(os.path.join(a.root,'**','episodes.csv'),recursive=True)
    if not fs: raise SystemExit('No episodes.csv found')
    df=pd.concat([pd.read_csv(f) for f in fs],ignore_index=True); keys=['env','method','seed']
    rows=[]
    for k,g in df.groupby(keys,dropna=False):
        row=dict(zip(keys,k));
        for col in ['episode_return','success','mpc_ms','prediction_mse','action_d1','action_d2','residual_norm','effective_residual_norm','gate','adaptive_gate','gate_z','residual_ramp','mpc_cache_hit_rate']:
            row[col]=pd.to_numeric(g[col],errors='coerce').mean() if col in g else np.nan
        row['sample_efficiency_step']=first_success_step(g.reset_index(drop=True)); rows.append(row)
    per_seed=pd.DataFrame(rows); summary=per_seed.groupby(['env','method'],dropna=False).agg(['mean','std']).reset_index(); os.makedirs(os.path.dirname(a.out) or '.',exist_ok=True); summary.to_csv(a.out,index=False); print(summary.to_string()); print('saved',a.out)
if __name__=='__main__': main()
