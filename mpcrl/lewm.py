from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelEncoder(nn.Module):
    def __init__(self,in_channels=3,latent_dim=128):
        super().__init__(); self.conv=nn.Sequential(nn.Conv2d(in_channels,32,5,2,2),nn.BatchNorm2d(32),nn.GELU(),nn.Conv2d(32,64,5,2,2),nn.BatchNorm2d(64),nn.GELU(),nn.Conv2d(64,128,3,2,1),nn.BatchNorm2d(128),nn.GELU(),nn.AdaptiveAvgPool2d(1)); self.proj=nn.Sequential(nn.Flatten(),nn.Linear(128,latent_dim),nn.BatchNorm1d(latent_dim,affine=False))
    def forward(self,x): return self.proj(self.conv(x))
class ActionPredictor(nn.Module):
    def __init__(self,latent_dim,action_dim,hidden=256): super().__init__(); self.net=nn.Sequential(nn.Linear(latent_dim+action_dim,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,latent_dim))
    def forward(self,z,a): return self.net(torch.cat([z,a],-1))
class SIGReg(nn.Module):
    def __init__(self,knots=17,num_proj=256,t_max=3.0):
        super().__init__(); self.num_proj=num_proj; t=torch.linspace(0.0,t_max,knots); dx=t[1]-t[0] if knots>1 else torch.tensor(1.0); trap=torch.ones_like(t); trap[[0,-1]]=0.5; window=torch.exp(-0.5*t.square()); self.register_buffer('t',t); self.register_buffer('weights',trap*window*dx); self.register_buffer('phi',torch.exp(-0.5*t.square()))
    def forward(self,z):
        z=z.reshape(-1,z.shape[-1]); d=z.shape[-1]; dirs=torch.randn(d,self.num_proj,device=z.device,dtype=z.dtype); dirs=dirs/(dirs.norm(dim=0,keepdim=True)+1e-8); h=z@dirs; angle=h.unsqueeze(-1)*self.t; c=angle.cos().mean(0); s=angle.sin().mean(0); err=(c-self.phi).square()+s.square(); return (err*self.weights).sum(-1).mean()
class LeWorldModel(nn.Module):
    def __init__(self,action_dim,latent_dim=128,hidden=256,sigreg_lambda=0.1,sigreg_projections=256,sigreg_knots=17): super().__init__(); self.encoder=PixelEncoder(3,latent_dim); self.predictor=ActionPredictor(latent_dim,action_dim,hidden); self.sigreg=SIGReg(sigreg_knots,sigreg_projections); self.sigreg_lambda=sigreg_lambda
    def encode(self,frames): return self.encoder(frames.float()/255.0 if frames.dtype==torch.uint8 else frames)
    def loss(self,frame_t,action_t,frame_tp1):
        z=self.encode(frame_t); target=self.encode(frame_tp1); pred=self.predictor(z,action_t); pred_loss=F.mse_loss(pred.float(),target.float()); sr=self.sigreg(torch.cat([z,target],0).float()); total=pred_loss+self.sigreg_lambda*sr; return total,{'total':float(total.detach()),'pred':float(pred_loss.detach()),'sigreg':float(sr.detach())}
    def latent_step(self,z,a): return self.predictor(z,a)
class RewardProbe(nn.Module):
    def __init__(self,latent_dim,action_dim,hidden=128): super().__init__(); self.net=nn.Sequential(nn.Linear(latent_dim+action_dim,hidden),nn.ReLU(),nn.Linear(hidden,1))
    def forward(self,z,a): return self.net(torch.cat([z,a],-1))
