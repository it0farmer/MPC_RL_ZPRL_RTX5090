from __future__ import annotations
import csv,os,numpy as np
def infer_success(info,episode_return,threshold=None):
    for key in ('success','is_success'):
        if key in info:return float(bool(info[key]))
    if threshold is not None:return float(episode_return>=float(threshold))
    return float('nan')
class EpisodeMetrics:
    def __init__(self):self.reset()
    def reset(self): self.ret=0.0; self.n=0; self.mpc_ms=[]; self.pred_err=[]; self.unc=[]; self.resnorm=[]; self.gates=[]; self.adaptive_gates=[]; self.gate_z=[]; self.ramps=[]; self.effective_resnorm=[]; self.cache_hits=[]; self.actions=[]
    def step(self,reward,action,mpc_ms=np.nan,pred_err=np.nan,unc=np.nan,resnorm=np.nan,gate=np.nan,adaptive_gate=np.nan,gate_z=np.nan,ramp=np.nan,effective_resnorm=np.nan,cache_hit=np.nan):
        self.ret+=float(reward); self.n+=1; self.actions.append(np.asarray(action,np.float32).copy()); self.mpc_ms.append(mpc_ms); self.pred_err.append(pred_err); self.unc.append(unc); self.resnorm.append(resnorm); self.gates.append(gate); self.adaptive_gates.append(adaptive_gate); self.gate_z.append(gate_z); self.ramps.append(ramp); self.effective_resnorm.append(effective_resnorm); self.cache_hits.append(float(cache_hit) if cache_hit is not None else np.nan)
    def finish(self,info,success_threshold=None):
        a=np.asarray(self.actions); d1=np.linalg.norm(np.diff(a,axis=0),axis=1).mean() if len(a)>1 else np.nan; d2=np.linalg.norm(np.diff(a,n=2,axis=0),axis=1).mean() if len(a)>2 else np.nan; nm=lambda x:float(np.nanmean(x)) if np.any(np.isfinite(x)) else float('nan'); return {'episode_return':self.ret,'episode_length':self.n,'success':infer_success(info,self.ret,success_threshold),'mpc_ms':nm(np.asarray(self.mpc_ms,float)),'prediction_mse':nm(np.asarray(self.pred_err,float)),'uncertainty':nm(np.asarray(self.unc,float)),'residual_norm':nm(np.asarray(self.resnorm,float)),'gate':nm(np.asarray(self.gates,float)),'adaptive_gate':nm(np.asarray(self.adaptive_gates,float)),'gate_z':nm(np.asarray(self.gate_z,float)),'residual_ramp':nm(np.asarray(self.ramps,float)),'effective_residual_norm':nm(np.asarray(self.effective_resnorm,float)),'mpc_cache_hit_rate':nm(np.asarray(self.cache_hits,float)),'action_d1':float(d1),'action_d2':float(d2)}
class CSVLogger:
    def __init__(self,path,fieldnames): self.path=path; self.fieldnames=fieldnames; os.makedirs(os.path.dirname(path),exist_ok=True); (not os.path.exists(path)) and open(path,'w',encoding='utf-8').close()
    def write(self,row):
        exists=os.path.getsize(self.path)>0
        with open(self.path,'a',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=self.fieldnames)
            if not exists:w.writeheader()
            w.writerow({k:row.get(k,'') for k in self.fieldnames})
