from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from mpcrl.cem import CEMPlanner
from mpcrl.config import load_yaml
from mpcrl.envs import action_bounds, dims, make_mujoco_env
from mpcrl.gate import residual_ramp
from mpcrl.metrics import CSVLogger, EpisodeMetrics
from mpcrl.replay import ResidualReplay, TransitionReplay
from mpcrl.sac import ResidualSAC
from mpcrl.utils import accelerator_summary, configure_accelerator, set_seed
from mpcrl.world_model import EnsembleWorldModel
from mpcrl.zprl_style import BottleneckBasePolicy, fit_behavior_clone


def _pbar(iterable, enabled=True, **kwargs):
    return tqdm(iterable, disable=not enabled, dynamic_ncols=True, mininterval=0.5, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/halfcheetah.yaml')
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--seed', type=int)
    ap.add_argument('--latent-dim', type=int)
    ap.add_argument('--bc-collect-steps', type=int)
    ap.add_argument('--bc-epochs', type=int)
    ap.add_argument('--no-progress', action='store_true')
    args = ap.parse_args()

    show_progress = not args.no_progress
    cfg = load_yaml(args.config)
    seed = args.seed if args.seed is not None else cfg['env']['seed']
    set_seed(seed)

    hw = cfg.get('hardware', {})
    precision = str(hw.get('precision', 'fp32')).lower()
    device = configure_accelerator(hw)
    print(accelerator_summary(device, precision))

    env, obs, _ = make_mujoco_env(cfg['env']['id'], seed)
    od, ad = dims(env)
    low, high = action_bounds(env)
    mid = (high + low) / 2
    span = (high - low) / 2

    tc = cfg['train']
    zc = cfg.get('zprl_style', {})
    latent_dim = int(args.latent_dim or zc.get('latent_dim', 16))
    bc_collect_steps = int(args.bc_collect_steps or zc.get('bc_collect_steps', 6000))
    bc_epochs = int(args.bc_epochs or zc.get('bc_epochs', 80))
    latent_residual_scale = float(zc.get('residual_scale', 0.05))
    latent_ramp_steps = int(zc.get('residual_ramp_steps', 5000))

    buf = TransitionReplay(max(100000, int(args.steps) + bc_collect_steps + 10000), od, ad)
    warm = int(tc['warmup_steps'])
    for _ in _pbar(
        range(warm),
        show_progress,
        total=warm,
        desc=f"ZPRL warmup {cfg['env']['id']} seed={seed}",
        leave=False,
    ):
        a = env.action_space.sample()
        no, r, te, tr, _ = env.step(a)
        done = te or tr
        buf.add(obs, a, r, no, done)
        obs = no
        if done:
            obs, _ = env.reset()

    w = cfg['world_model']
    wm = EnsembleWorldModel(
        od, ad, w['ensemble_size'], w['hidden_dim'], w['lr'], w['weight_decay'], device, precision
    )
    wm_loss = wm.fit_batch(
        buf.sample(min(buf.size, 10000)),
        updates=int(tc['wm_updates']),
        batch_size=int(tc['batch_size']),
    )

    m = cfg['mpc']
    planner = CEMPlanner(
        wm, low, high, m['horizon'], m['candidates'], m['elites'], m['iterations'],
        m['alpha'], m['init_std'], m['min_std'], m['discount'], w.get('uncertainty_penalty', 0.0)
    )

    # Build the frozen base policy from states actually visited by MPC rather
    # than random warmup states. This is the critical prerequisite for a fair
    # bottleneck-residual baseline: RL should refine a competent base policy.
    bc_states = []
    bc_targets = []
    obs, _ = env.reset(seed=seed + 101)
    planner.reset()
    collect = _pbar(
        range(1, bc_collect_steps + 1),
        show_progress,
        total=bc_collect_steps,
        desc=f"ZPRL MPC-BC collect {cfg['env']['id']} seed={seed}",
        leave=False,
    )
    for i in collect:
        plan = planner.plan(obs)
        action = np.clip(plan.actions[0], low, high)
        bc_states.append(obs.copy())
        bc_targets.append(action.copy())

        no, r, te, tr, _ = env.step(action)
        done = te or tr
        buf.add(obs, action, r, no, done)

        if i % int(tc['wm_update_interval']) == 0:
            wm_loss = wm.fit_batch(
                buf.sample(min(buf.size, 10000)),
                updates=int(tc['wm_updates']),
                batch_size=int(tc['batch_size']),
            )
            if show_progress:
                collect.set_postfix(wm=f'{wm_loss:.4f}')

        if done:
            obs, _ = env.reset()
            planner.reset()
        else:
            obs = no

    states = np.asarray(bc_states, dtype=np.float32)
    targets = np.asarray(bc_targets, dtype=np.float32)
    target_norm = np.clip((targets - mid) / np.maximum(span, 1e-6), -1, 1)

    base = BottleneckBasePolicy(od, ad, latent_dim).to(device)
    bc = fit_behavior_clone(
        base,
        states,
        target_norm,
        epochs=bc_epochs,
        batch_size=int(tc['batch_size']),
        device=device,
        show_progress=show_progress,
    )

    with torch.no_grad():
        pred = base(torch.as_tensor(states, dtype=torch.float32, device=device)).cpu().numpy()
    bc_dataset_mse = float(np.mean((pred - target_norm) ** 2))
    print(f'ZPRL base-policy BC: samples={len(states)} final_mse={bc_dataset_mse:.6f}')

    ctx_dim = od + latent_dim
    agent = ResidualSAC(
        ctx_dim,
        latent_dim,
        hidden=cfg['sac']['hidden_dim'],
        lr=cfg['sac']['lr'],
        gamma=cfg['sac']['gamma'],
        tau=cfg['sac']['tau'],
        init_alpha=cfg['sac']['init_alpha'],
        target_entropy_scale=cfg['sac']['target_entropy_scale'],
        consistency_coef=cfg['residual'].get('consistency_coef', 0.05),
        device=device,
        precision=precision,
    )
    rb = ResidualReplay(max(100000, int(args.steps) + 10000), ctx_dim, latent_dim)

    out = Path(cfg['logging']['root']) / f"{cfg['env']['id']}__zprl_style__seed{seed}__{int(time.time())}"
    out.mkdir(parents=True, exist_ok=True)
    fields = [
        'global_step', 'episode', 'method', 'env', 'seed', 'episode_return',
        'episode_length', 'success', 'mpc_ms', 'prediction_mse', 'uncertainty',
        'residual_norm', 'effective_residual_norm', 'gate', 'residual_ramp',
        'action_d1', 'action_d2', 'bc_loss', 'bc_dataset_mse',
    ]
    log = CSVLogger(str(out / 'episodes.csv'), fields)

    obs, _ = env.reset(seed=seed + 202)
    em = EpisodeMetrics()
    ep = 0
    pbar = _pbar(
        range(1, args.steps + 1),
        show_progress,
        total=args.steps,
        desc=f"{cfg['env']['id']} | zprl_style | seed={seed}",
    )

    for step in pbar:
        ot = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        z = base.encode(ot).squeeze(0).detach().cpu().numpy()
        c = np.concatenate([obs, z])
        dz = agent.act(c)
        ramp = residual_ramp(step, 1, latent_ramp_steps)
        effective_dz = ramp * latent_residual_scale * dz
        z2 = torch.as_tensor(z + effective_dz, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            an = base.decode(z2).squeeze(0).cpu().numpy()
        action = np.clip(mid + span * an, low, high)

        no, r, te, tr, info = env.step(action)
        done = te or tr
        if done:
            nc = np.zeros(ctx_dim, np.float32)
        else:
            with torch.no_grad():
                nz = base.encode(
                    torch.as_tensor(no, dtype=torch.float32, device=device).unsqueeze(0)
                ).squeeze(0).cpu().numpy()
            nc = np.concatenate([no, nz])

        rb.add(c, dz, r, nc, done)
        em.step(
            r,
            action,
            resnorm=float(np.linalg.norm(dz)),
            effective_resnorm=float(np.linalg.norm(effective_dz)),
            gate=ramp,
            ramp=ramp,
        )

        if rb.size >= int(tc['batch_size']):
            agent.update(rb.sample(int(tc['batch_size'])))

        if done:
            row = em.finish(info, cfg['env'].get('success_return_threshold'))
            row.update(
                global_step=step,
                episode=ep,
                method='zprl_style',
                env=cfg['env']['id'],
                seed=seed,
                bc_loss=bc,
                bc_dataset_mse=bc_dataset_mse,
            )
            log.write(row)
            ep += 1
            em.reset()
            obs, _ = env.reset()
            if show_progress:
                pbar.set_postfix(
                    ep=ep,
                    ret=f"{row['episode_return']:.1f}",
                    length=int(row['episode_length']),
                    ramp=f'{ramp:.2f}',
                    bc=f'{bc_dataset_mse:.4f}',
                )
                if ep % 10 == 0:
                    tqdm.write(json.dumps(row, ensure_ascii=False))
            else:
                print(json.dumps(row, ensure_ascii=False))
        else:
            obs = no

    torch.save(
        {
            'base_policy': base.state_dict(),
            'agent': agent.state_dict(),
            'latent_dim': latent_dim,
            'config': cfg,
            'bc_dataset_mse': bc_dataset_mse,
        },
        out / 'final.pt',
    )
    env.close()
    print('output:', out)


if __name__ == '__main__':
    main()
