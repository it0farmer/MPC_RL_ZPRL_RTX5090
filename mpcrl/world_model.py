from __future__ import annotations
from dataclasses import dataclass
import numpy as np,torch,torch.nn as nn,torch.nn.functional as F
class MLPDynamics(nn.Module):
    def __init__(self,obs_dim,action_dim,hidden_dim=256): super().__init__(); self.net=nn.Sequential(nn.Linear(obs_dim+action_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,obs_dim+1))
    def forward(self,obs,action): return self.net(torch.cat([obs,action],dim=-1))
@dataclass
class WMPrediction: next_obs: torch.Tensor; reward: torch.Tensor; uncertainty: torch.Tensor
class EnsembleWorldModel(nn.Module):
    def __init__(self,obs_dim,action_dim,ensemble_size=5,hidden_dim=256,lr=1e-3,weight_decay=1e-5,device='cpu',precision='fp32'):
        super().__init__(); self.obs_dim=obs_dim; self.action_dim=action_dim; self.ensemble_size=ensemble_size; self.device=torch.device(device); self.amp_dtype=torch.bfloat16 if precision=='bf16' else None; self.members=nn.ModuleList([MLPDynamics(obs_dim,action_dim,hidden_dim) for _ in range(ensemble_size)]).to(self.device); self.optimizers=[torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=weight_decay) for m in self.members]; self.register_buffer('obs_mean',torch.zeros(obs_dim)); self.register_buffer('obs_std',torch.ones(obs_dim)); self.register_buffer('act_mean',torch.zeros(action_dim)); self.register_buffer('act_std',torch.ones(action_dim)); self.register_buffer('delta_mean',torch.zeros(obs_dim)); self.register_buffer('delta_std',torch.ones(obs_dim)); self.register_buffer('rew_mean',torch.zeros(1)); self.register_buffer('rew_std',torch.ones(1)); self.to(self.device)
    @torch.no_grad()
    def update_stats(self,batch):
        def st(x): t=torch.as_tensor(x,dtype=torch.float32,device=self.device); return t.mean(0),t.std(0).clamp_min(1e-4)
        o,a,no,r=batch['obs'],batch['action'],batch['next_obs'],batch['reward']; self.obs_mean[:],self.obs_std[:]=st(o); self.act_mean[:],self.act_std[:]=st(a); self.delta_mean[:],self.delta_std[:]=st(no-o); self.rew_mean[:],self.rew_std[:]=st(r)
    def _norm_inputs(self,obs,action): return (obs-self.obs_mean)/self.obs_std,(action-self.act_mean)/self.act_std
    def _decode(self,obs,y): delta=y[...,:self.obs_dim]*self.delta_std+self.delta_mean; reward=y[...,self.obs_dim:]*self.rew_std+self.rew_mean; return obs+delta,reward
    def fit_batch(self,batch,updates=50,batch_size=256):
        self.update_stats(batch); obs=torch.as_tensor(batch['obs'],dtype=torch.float32,device=self.device); action=torch.as_tensor(batch['action'],dtype=torch.float32,device=self.device); next_obs=torch.as_tensor(batch['next_obs'],dtype=torch.float32,device=self.device); reward=torch.as_tensor(batch['reward'],dtype=torch.float32,device=self.device); n=len(obs); losses=[]
        for _ in range(updates):
            for member,opt in zip(self.members,self.optimizers):
                idx=torch.randint(0,n,(min(batch_size,n),),device=self.device); o,a,no,r=obs[idx],action[idx],next_obs[idx],reward[idx]; on,an=self._norm_inputs(o,a); target=torch.cat([((no-o)-self.delta_mean)/self.delta_std,(r-self.rew_mean)/self.rew_std],-1); pred=member(on,an); loss=F.mse_loss(pred.float(),target.float()); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(float(loss.detach()))
        return float(np.mean(losses))
    def predict(self,obs,action):
        if not torch.is_tensor(obs): obs=torch.as_tensor(obs,dtype=torch.float32,device=self.device)
        if not torch.is_tensor(action): action=torch.as_tensor(action,dtype=torch.float32,device=self.device)
        on,an=self._norm_inputs(obs,action); nexts=[]; rews=[]
        for m in self.members: no,r=self._decode(obs,m(on,an).float()); nexts.append(no); rews.append(r)
        nexts=torch.stack(nexts,0); rews=torch.stack(rews,0); return WMPrediction(nexts.mean(0),rews.mean(0),nexts.var(0,unbiased=False).mean(-1,keepdim=True)+rews.var(0,unbiased=False))
    @torch.no_grad()
    def rollout_return(self,start_obs,action_sequences,discount=.99,uncertainty_penalty=0.):
        acts=torch.as_tensor(action_sequences,dtype=torch.float32,device=self.device); n,h,_=acts.shape; obs=torch.as_tensor(start_obs,dtype=torch.float32,device=self.device).reshape(1,-1).expand(n,-1); ret=torch.zeros((n,1),device=self.device); unc=torch.zeros_like(ret); g=1.
        for t in range(h): p=self.predict(obs,acts[:,t]); ret+=g*(p.reward-uncertainty_penalty*p.uncertainty); unc+=p.uncertainty; obs=p.next_obs; g*=discount
        return ret.squeeze(-1),(unc/h).squeeze(-1)
