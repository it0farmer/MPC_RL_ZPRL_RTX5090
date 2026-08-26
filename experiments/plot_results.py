from __future__ import annotations
import argparse,glob,os
import pandas as pd
import matplotlib.pyplot as plt
from mpcrl.plotting import set_paper_style,outside_legend

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',default='runs'); p.add_argument('--summary'); p.add_argument('--outdir',default='results/figures'); a=p.parse_args(); set_paper_style(); os.makedirs(a.outdir,exist_ok=True)
    fs=glob.glob(os.path.join(a.root,'**','episodes.csv'),recursive=True); df=pd.concat([pd.read_csv(f) for f in fs],ignore_index=True)
    for metric,ylabel in [('episode_return','累计回报'),('mpc_ms','MPC规划时间 / ms'),('action_d1','动作一阶平滑度'),('prediction_mse','模型一步预测MSE'),('effective_residual_norm','有效残差范数'),('gate','有效门控系数')]:
        if metric not in df.columns: continue
        fig,ax=plt.subplots(figsize=(6.4,4.2))
        for (env,method),g in df.groupby(['env','method']):
            g=g.sort_values('global_step'); sm=g[metric].rolling(10,min_periods=1).mean(); ax.plot(g['global_step'],sm,label=f'{env}-{method}')
        ax.set_xlabel('环境交互步'); ax.set_ylabel(ylabel); ax.grid(alpha=.25); outside_legend(ax); fig.tight_layout(); fig.savefig(os.path.join(a.outdir,f'{metric}.png'),dpi=300,bbox_inches='tight'); plt.close(fig)
if __name__=='__main__': main()
