from __future__ import annotations
import argparse, random, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from mpcrl.config import load_yaml, save_yaml
from mpcrl.envs import make_mujoco_env,dims,action_bounds
from mpcrl.lewm import LeWorldModel,RewardProbe
from mpcrl.lewm_planner import LeWMCEMPlanner
from mpcrl.metrics import EpisodeMetrics,CSVLogger
from mpcrl.utils import set_seed,configure_accelerator,accelerator_summary,autocast_context


def resize_frame(img,size):
    x=torch.as_tensor(img).permute(2,0,1).unsqueeze(0).float()
    return F.interpolate(x,(size,size),mode='bilinear',align_corners=False).squeeze(0).permute(1,2,0).byte().numpy()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/lewm_halfcheetah.yaml'); ap.add_argument('--seed',type=int); ap.add_argument('--episodes',type=int); args=ap.parse_args()
    cfg=load_yaml(args.config); seed=args.seed if args.seed is not None else cfg['env']['seed']; cfg['env']['seed']=seed; set_seed(seed); hw=cfg.get('hardware',{}); precision=str(hw.get('precision','fp32')).lower(); device=configure_accelerator(hw); print(accelerator_summary(device,precision)); size=cfg['env']['render_size']
    env,_,_=make_mujoco_env(cfg['env']['id'],seed,render_mode='rgb_array'); _,ad=dims(env); low,high=action_bounds(env); data=[]; frame=resize_frame(env.render(),size)
    for _ in range(cfg['collect']['transitions']):
        a=env.action_space.sample(); _,r,te,tr,_=env.step(a); nf=resize_frame(env.render(),size); data.append((frame,a,nf,r)); frame=nf
        if te or tr: env.reset(); frame=resize_frame(env.render(),size)
    lc=cfg['lewm']; model=LeWorldModel(ad,lc['latent_dim'],lc['hidden_dim'],lc['sigreg_lambda'],lc['sigreg_projections'],lc['sigreg_knots']).to(device); opt=torch.optim.AdamW(model.parameters(),lr=lc['lr']); bs=lc['batch_size']
    for ep in range(lc['epochs']):
        random.shuffle(data); logs=[]
        for i in range(0,len(data)-bs+1,bs):
            b=data[i:i+bs]; f=torch.as_tensor(np.stack([x[0] for x in b]),device=device).permute(0,3,1,2); a=torch.as_tensor(np.stack([x[1] for x in b]),dtype=torch.float32,device=device); nf=torch.as_tensor(np.stack([x[2] for x in b]),device=device).permute(0,3,1,2)
            with autocast_context(device,precision): loss,info=model.loss(f,a,nf)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),10); opt.step(); logs.append(info)
        print(f"LeWM epoch {ep+1}: total={np.mean([x['total'] for x in logs]):.5f} pred={np.mean([x['pred'] for x in logs]):.5f} sigreg={np.mean([x['sigreg'] for x in logs]):.5f}")
    for p in model.parameters(): p.requires_grad_(False)
    model.eval(); probe=RewardProbe(lc['latent_dim'],ad).to(device); po=torch.optim.Adam(probe.parameters(),lr=cfg['reward_probe']['lr'])
    for ep in range(cfg['reward_probe']['epochs']):
        random.shuffle(data); ls=[]
        for i in range(0,len(data)-bs+1,bs):
            b=data[i:i+bs]; f=torch.as_tensor(np.stack([x[0] for x in b]),device=device).permute(0,3,1,2); a=torch.as_tensor(np.stack([x[1] for x in b]),dtype=torch.float32,device=device); r=torch.as_tensor([[x[3]] for x in b],dtype=torch.float32,device=device)
            with torch.no_grad(): z=model.encode(f)
            with autocast_context(device,precision): pred_r=probe(z,a)
            loss=F.mse_loss(pred_r.float(),r.float()); po.zero_grad(set_to_none=True); loss.backward(); po.step(); ls.append(float(loss.detach()))
        print(f'Reward probe epoch {ep+1}: loss={np.mean(ls):.5f}')
    mc=cfg['mpc']; planner=LeWMCEMPlanner(model,probe,low,high,mc['horizon'],mc['candidates'],mc['elites'],mc['iterations'],mc['init_std'],mc['min_std'],mc['discount'],device,precision)
    out=Path('runs')/f"{cfg['env']['id']}__lewm_mpc__seed{seed}__{int(time.time())}"; out.mkdir(parents=True,exist_ok=True); save_yaml(cfg,out/'config.yaml')
    fields=['global_step','episode','method','env','seed','episode_return','episode_length','success','mpc_ms','prediction_mse','uncertainty','residual_norm','gate','action_d1','action_d2']; log=CSVLogger(str(out/'episodes.csv'),fields)
    total_steps=0; n_eval=args.episodes or cfg['eval']['episodes']
    for e in range(n_eval):
        env.reset(seed=seed+100+e); em=EpisodeMetrics()
        while True:
            fr=resize_frame(env.render(),size); plan=planner.plan(fr); _,r,te,tr,info=env.step(plan.actions[0]); total_steps+=1; em.step(r,plan.actions[0],mpc_ms=plan.planning_ms)
            if te or tr: break
        row=em.finish(info,None); row.update(global_step=total_steps,episode=e,method='lewm_mpc',env=cfg['env']['id'],seed=seed); log.write(row); print(row)
    torch.save({'lewm':model.state_dict(),'reward_probe':probe.state_dict(),'config':cfg},out/'final.pt'); env.close(); print('output:',out)
if __name__=='__main__': main()
