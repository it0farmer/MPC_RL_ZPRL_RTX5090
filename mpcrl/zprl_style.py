from __future__ import annotations
import numpy as np,torch,torch.nn as nn,torch.nn.functional as F
class BottleneckBasePolicy(nn.Module):
    def __init__(self,obs_dim,action_dim,latent_dim=16,hidden=256):
        super().__init__(); self.encoder=nn.Sequential(nn.Linear(obs_dim,hidden),nn.ReLU(),nn.Linear(hidden,latent_dim)); self.decoder=nn.Sequential(nn.Linear(latent_dim,hidden),nn.ReLU(),nn.Linear(hidden,action_dim),nn.Tanh())
    def encode(self,obs): return self.encoder(obs)
    def decode(self,z): return self.decoder(z)
    def forward(self,obs): return self.decode(self.encode(obs))
def fit_behavior_clone(policy,obs,target_action_norm,epochs=100,batch_size=256,lr=1e-3,device='cpu'):
    policy.to(device); opt=torch.optim.Adam(policy.parameters(),lr=lr); o=torch.as_tensor(obs,dtype=torch.float32,device=device); a=torch.as_tensor(target_action_norm,dtype=torch.float32,device=device); losses=[]
    for _ in range(epochs):
        idx=torch.randint(0,len(o),(min(batch_size,len(o)),),device=device); loss=F.mse_loss(policy(o[idx]),a[idx]); opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.detach()))
    for p in policy.parameters(): p.requires_grad_(False)
    policy.eval(); return float(np.mean(losses[-10:]))
