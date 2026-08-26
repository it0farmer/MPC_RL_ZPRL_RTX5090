from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np, torch
from mpcrl.config import load_yaml
from mpcrl.envs import make_mujoco_env,dims,action_bounds
from mpcrl.replay import TransitionReplay,ResidualReplay
from mpcrl.world_model import EnsembleWorldModel
from mpcrl.cem import CEMPlanner
from mpcrl.zprl_style import BottleneckBasePolicy,fit_behavior_clone
from mpcrl.sac import ResidualSAC
from mpcrl.metrics import EpisodeMetrics,CSVLogger
from mpcrl.utils import set_seed,configure_accelerator,accelerator_summary


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/halfcheetah.yaml'); ap.add_argument('--steps',type=int,default=30000); ap.add_argument('--seed',type=int); ap.add_argument('--latent-dim',type=int,default=16); args=ap.parse_args()
    cfg=load_yaml(args.config); seed=args.seed if args.seed is not None else cfg['env']['seed']; set_seed(seed); hw=cfg.get('hardware',{}); precision=str(hw.get('precision','fp32')).lower(); device=configure_accelerator(hw); print(accelerator_summary(device,precision)); env,obs,_=make_mujoco_env(cfg['env']['id'],seed); od,ad=dims(env); low,high=action_bounds(env); mid=(high+low)/2; span=(high-low)/2
    buf=TransitionReplay(100000,od,ad); warm=cfg['train']['warmup_steps']
    for _ in range(warm):
        a=env.action_space.sample(); no,r,te,tr,_=env.step(a); buf.add(obs,a,r,no,te or tr); obs=no
        if te or tr: obs,_=env.reset()
    w=cfg['world_model']; wm=EnsembleWorldModel(od,ad,w['ensemble_size'],w['hidden_dim'],w['lr'],w['weight_decay'],device,precision); wm.fit_batch(buf.sample(min(buf.size,10000)),updates=cfg['train']['wm_updates'])
    m=cfg['mpc']; planner=CEMPlanner(wm,low,high,m['horizon'],m['candidates'],m['elites'],m['iterations'],m['alpha'],m['init_std'],m['min_std'],m['discount'],w.get('uncertainty_penalty',0.0))
    sample=buf.sample(min(2048,buf.size)); states=sample['obs']; targets=[]
    for s in states: targets.append(planner.plan(s).actions[0])
    targets=np.asarray(targets); target_norm=np.clip((targets-mid)/np.maximum(span,1e-6),-1,1)
    base=BottleneckBasePolicy(od,ad,args.latent_dim).to(device); bc=fit_behavior_clone(base,states,target_norm,epochs=120,batch_size=int(cfg['train']['batch_size']),device=device)
    ctx_dim=od+args.latent_dim; agent=ResidualSAC(ctx_dim,args.latent_dim,hidden=cfg['sac']['hidden_dim'],lr=cfg['sac']['lr'],gamma=cfg['sac']['gamma'],tau=cfg['sac']['tau'],init_alpha=cfg['sac']['init_alpha'],target_entropy_scale=cfg['sac']['target_entropy_scale'],consistency_coef=cfg['residual'].get('consistency_coef',0.05),device=device,precision=precision); rb=ResidualReplay(100000,ctx_dim,args.latent_dim)
    out=Path(cfg['logging']['root'])/f"{cfg['env']['id']}__zprl_style__seed{seed}__{int(time.time())}"; out.mkdir(parents=True,exist_ok=True); fields=['global_step','episode','method','env','seed','episode_return','episode_length','success','mpc_ms','prediction_mse','uncertainty','residual_norm','gate','action_d1','action_d2','bc_loss']; log=CSVLogger(str(out/'episodes.csv'),fields)
    obs,_=env.reset(); em=EpisodeMetrics(); ep=0
    for step in range(1,args.steps+1):
        ot=torch.as_tensor(obs,dtype=torch.float32,device=device).unsqueeze(0); z=base.encode(ot).squeeze(0).detach().cpu().numpy(); c=np.concatenate([obs,z]); dz=agent.act(c); z2=torch.as_tensor(z+cfg['residual']['residual_scale']*dz,dtype=torch.float32,device=device).unsqueeze(0)
        with torch.no_grad(): an=base.decode(z2).squeeze(0).cpu().numpy(); action=np.clip(mid+span*an,low,high)
        no,r,te,tr,info=env.step(action); done=te or tr
        if done: nc=np.zeros(ctx_dim,np.float32)
        else:
            with torch.no_grad(): nz=base.encode(torch.as_tensor(no,dtype=torch.float32,device=device).unsqueeze(0)).squeeze(0).cpu().numpy(); nc=np.concatenate([no,nz])
        rb.add(c,dz,r,nc,done); em.step(r,action,resnorm=np.linalg.norm(dz),gate=1.0)
        if rb.size>=int(cfg['train']['batch_size']): agent.update(rb.sample(int(cfg['train']['batch_size'])))
        if done:
            row=em.finish(info,cfg['env'].get('success_return_threshold')); row.update(global_step=step,episode=ep,method='zprl_style',env=cfg['env']['id'],seed=seed,bc_loss=bc); log.write(row); print(row); ep+=1; em.reset(); obs,_=env.reset()
        else: obs=no
    torch.save({'base_policy':base.state_dict(),'agent':agent.state_dict(),'latent_dim':args.latent_dim,'config':cfg},out/'final.pt'); env.close(); print('output:',out)
if __name__=='__main__': main()
