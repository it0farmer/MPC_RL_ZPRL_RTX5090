from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


def mlp(inp,out,hidden):
    return nn.Sequential(nn.Linear(inp,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,out))


class GaussianActor(nn.Module):
    def __init__(self, context_dim, action_dim, hidden=256):
        super().__init__(); self.net=mlp(context_dim, action_dim*2, hidden); self.action_dim=action_dim
    def forward(self, x):
        mean, log_std = self.net(x).chunk(2,-1); log_std=log_std.clamp(LOG_STD_MIN,LOG_STD_MAX)
        return mean, log_std
    def sample(self,x):
        mean, log_std=self(x); std=log_std.exp(); dist=torch.distributions.Normal(mean,std); z=dist.rsample(); a=torch.tanh(z)
        logp=dist.log_prob(z)-torch.log(1-a.pow(2)+1e-6); return a,logp.sum(-1,keepdim=True),torch.tanh(mean)


class Critic(nn.Module):
    def __init__(self, context_dim, action_dim, hidden=256): super().__init__(); self.net=mlp(context_dim+action_dim,1,hidden)
    def forward(self,c,a): return self.net(torch.cat([c,a],-1))


class ResidualSAC:
    def __init__(self, context_dim, residual_dim, hidden=256, lr=3e-4, gamma=0.99, tau=0.005,
                 init_alpha=0.2, target_entropy_scale=1.0, consistency_coef=0.0, device='cpu', precision='fp32'):
        self.device=torch.device(device); self.gamma=gamma; self.tau=tau; self.consistency_coef=consistency_coef
        self.precision=precision; self.amp_dtype=torch.bfloat16 if precision == 'bf16' else (torch.float16 if precision == 'fp16' else None)
        self.actor=GaussianActor(context_dim,residual_dim,hidden).to(self.device)
        self.q1=Critic(context_dim,residual_dim,hidden).to(self.device); self.q2=Critic(context_dim,residual_dim,hidden).to(self.device)
        self.tq1=Critic(context_dim,residual_dim,hidden).to(self.device); self.tq2=Critic(context_dim,residual_dim,hidden).to(self.device)
        self.tq1.load_state_dict(self.q1.state_dict()); self.tq2.load_state_dict(self.q2.state_dict())
        self.actor_opt=torch.optim.Adam(self.actor.parameters(),lr=lr)
        self.q_opt=torch.optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()),lr=lr)
        self.log_alpha=torch.tensor(np.log(init_alpha),device=self.device,requires_grad=True)
        self.alpha_opt=torch.optim.Adam([self.log_alpha],lr=lr)
        self.target_entropy=-float(residual_dim)*target_entropy_scale

    @property
    def alpha(self): return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, context, deterministic=False):
        c=torch.as_tensor(context,dtype=torch.float32,device=self.device).reshape(1,-1)
        a,_,mean=self.actor.sample(c); return (mean if deterministic else a).squeeze(0).cpu().numpy()

    def update(self,batch):
        b={k:torch.as_tensor(v,dtype=torch.float32,device=self.device) for k,v in batch.items()}
        c,a,r,nc,d=b['context'],b['residual'],b['reward'],b['next_context'],b['done']
        amp_enabled=self.device.type == 'cuda' and self.amp_dtype is not None
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=amp_enabled):
            na,nlogp,_=self.actor.sample(nc); tq=torch.min(self.tq1(nc,na),self.tq2(nc,na))-self.alpha.detach()*nlogp
            target=r+self.gamma*(1-d)*tq
        with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=amp_enabled):
            q1,q2=self.q1(c,a),self.q2(c,a)
            qloss=F.mse_loss(q1.float(),target.float())+F.mse_loss(q2.float(),target.float())
        self.q_opt.zero_grad(set_to_none=True); qloss.backward(); self.q_opt.step()
        with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=amp_enabled):
            pa,logp,_=self.actor.sample(c); q=torch.min(self.q1(c,pa),self.q2(c,pa))
            consistency=(pa.float().pow(2).mean())
            actor_loss=(self.alpha.detach()*logp.float()-q.float()).mean()+self.consistency_coef*consistency
        self.actor_opt.zero_grad(set_to_none=True); actor_loss.backward(); self.actor_opt.step()
        alpha_loss=-(self.log_alpha*(logp.detach()+self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True); alpha_loss.backward(); self.alpha_opt.step()
        with torch.no_grad():
            for p,tp in zip(self.q1.parameters(),self.tq1.parameters()): tp.data.lerp_(p.data,self.tau)
            for p,tp in zip(self.q2.parameters(),self.tq2.parameters()): tp.data.lerp_(p.data,self.tau)
        return {'q_loss':float(qloss.detach()),'actor_loss':float(actor_loss.detach()),'alpha':float(self.alpha.detach()),'consistency':float(consistency.detach())}

    def state_dict(self):
        return {'actor':self.actor.state_dict(),'q1':self.q1.state_dict(),'q2':self.q2.state_dict(),'tq1':self.tq1.state_dict(),'tq2':self.tq2.state_dict(),'log_alpha':self.log_alpha.detach().cpu()}

    def load_bundle(self, bundle):
        self.actor.load_state_dict(bundle['actor']); self.q1.load_state_dict(bundle['q1']); self.q2.load_state_dict(bundle['q2'])
        self.tq1.load_state_dict(bundle['tq1']); self.tq2.load_state_dict(bundle['tq2'])
        with torch.no_grad(): self.log_alpha.copy_(bundle['log_alpha'].to(self.device))
