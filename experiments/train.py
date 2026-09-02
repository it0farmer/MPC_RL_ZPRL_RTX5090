from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from mpcrl.cem import CEMPlanner
from mpcrl.config import load_yaml, save_yaml
from mpcrl.envs import action_bounds, dims, make_mujoco_env
from mpcrl.gate import adaptive_uncertainty_gate, make_uncertainty_gate, residual_ramp
from mpcrl.metrics import CSVLogger, EpisodeMetrics
from mpcrl.plan_cache import PlanCache
from mpcrl.planning_residual import expand_temporal_residual
from mpcrl.replay import ResidualReplay, TransitionReplay
from mpcrl.residual_context import gated_strength, residual_context
from mpcrl.sac import ResidualSAC
from mpcrl.utils import accelerator_summary, configure_accelerator, set_seed
from mpcrl.world_model import EnsembleWorldModel


METHODS = {'mpc_only', 'action_residual', 'planning_residual'}


def plan_chunk(plan_actions, k):
    p = plan_actions[:k]
    if len(p) < k:
        p = np.concatenate([p, np.repeat(p[-1:], k - len(p), axis=0)], axis=0)
    return p


def _progress(iterable, enabled: bool, **kwargs):
    return tqdm(iterable, disable=not enabled, dynamic_ncols=True, mininterval=0.5, **kwargs)


def _adaptive_gate_for_plan(method, rc, normalized_gate, uncertainty, update=True):
    if method != 'planning_residual':
        return float(rc.get('fixed_gate', 1.0)), np.nan
    if not rc.get('adaptive_gate', True):
        return float(rc.get('fixed_gate', 1.0)), np.nan
    if normalized_gate is not None:
        gout = normalized_gate(uncertainty, update=update)
        return gout.value, gout.normalized_uncertainty
    return adaptive_uncertainty_gate(
        uncertainty,
        rc['gate_threshold'],
        rc['gate_temperature'],
    ), np.nan


def train(cfg, method='planning_residual', steps=None, run_name=None, show_progress=True):
    assert method in METHODS

    hw = cfg.get('hardware', {})
    precision = str(hw.get('precision', 'fp32')).lower()
    device = configure_accelerator(hw)
    print(accelerator_summary(device, precision))

    seed = int(cfg['env'].get('seed', 0))
    set_seed(seed)
    env, obs, _ = make_mujoco_env(cfg['env']['id'], seed)
    obs_dim, act_dim = dims(env)
    low, high = action_bounds(env)
    span = (high - low) / 2

    tc = cfg['train']
    total = int(steps or tc['total_steps'])
    wm_cfg = cfg['world_model']
    mp = cfg['mpc']
    rc = cfg['residual']
    sc = cfg['sac']

    wm = EnsembleWorldModel(
        obs_dim,
        act_dim,
        wm_cfg['ensemble_size'],
        wm_cfg['hidden_dim'],
        wm_cfg['lr'],
        wm_cfg['weight_decay'],
        device,
        precision,
    )
    wm_buf = TransitionReplay(max(total + 10000, 100000), obs_dim, act_dim)

    warmup_steps = int(tc['warmup_steps'])
    for _ in _progress(
        range(warmup_steps),
        show_progress,
        total=warmup_steps,
        desc=f"warmup {cfg['env']['id']} seed={seed}",
        leave=False,
    ):
        a = env.action_space.sample()
        no, r, term, trunc, _ = env.step(a)
        done = term or trunc
        wm_buf.add(obs, a, r, no, done)
        obs = no
        if done:
            obs, _ = env.reset()

    wm_loss = wm.fit_batch(
        wm_buf.sample(min(wm_buf.size, 10000)),
        updates=int(tc['wm_updates']),
        batch_size=int(tc['batch_size']),
    )

    planner = CEMPlanner(
        wm,
        low,
        high,
        mp['horizon'],
        mp['candidates'],
        mp['elites'],
        mp['iterations'],
        mp['alpha'],
        mp['init_std'],
        mp['min_std'],
        mp['discount'],
        wm_cfg.get('uncertainty_penalty', 0.0),
    )

    # Receding-horizon MPC executes only the first action. Planning Residual
    # conditions on the full short MPC chunk, learns one action-space residual,
    # and expands it through the chunk with temporal decay.
    k = 1 if method == 'action_residual' else int(rc['chunk_len'])
    residual_dim = act_dim
    # The gate changes how the raw residual maps to the executed action, so the
    # critic must observe it. Otherwise early ramp and late training create
    # inconsistent transitions for the same (state, residual) tuple.
    context_dim = obs_dim + k * act_dim + 1
    temporal_decay = float(rc.get('temporal_decay', 0.8))
    gate_power = float(rc.get('gate_power', 1.0)) if method == 'planning_residual' else 1.0

    agent = None
    rb = None
    if method != 'mpc_only':
        agent = ResidualSAC(
            context_dim,
            residual_dim,
            sc['hidden_dim'],
            sc['lr'],
            sc['gamma'],
            sc['tau'],
            sc['init_alpha'],
            sc['target_entropy_scale'],
            rc.get('consistency_coef', 0.0) if method == 'planning_residual' else 0.0,
            device,
            precision,
        )
        rb = ResidualReplay(max(total + 10000, 100000), context_dim, residual_dim)

    normalized_gate = (
        make_uncertainty_gate(rc)
        if method == 'planning_residual' and rc.get('adaptive_gate', True)
        else None
    )
    rl_start_local = max(1, int(tc['rl_start_steps']) - warmup_steps)
    ramp_steps = int(rc.get('ramp_steps', 1))
    cache = PlanCache()

    root = Path(cfg['logging'].get('root', 'runs'))
    name = run_name or f"{cfg['env']['id']}__{method}__seed{seed}__{int(time.time())}"
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, out / 'config.yaml')

    fields = [
        'global_step', 'episode', 'method', 'env', 'seed',
        'episode_return', 'episode_length', 'success', 'mpc_ms',
        'prediction_mse', 'uncertainty', 'residual_norm',
        'effective_residual_norm', 'gate', 'adaptive_gate', 'gate_z',
        'residual_ramp', 'mpc_cache_hit_rate', 'action_d1', 'action_d2',
        'wm_loss',
    ]
    logger = CSVLogger(str(out / 'episodes.csv'), fields)
    metrics = EpisodeMetrics()
    episode = 0
    obs, _ = env.reset(seed=seed + 1)
    planner.reset()
    cache.clear()
    console_every = max(1, int(tc.get('console_log_interval_episodes', 10)))

    pbar = _progress(
        range(1, total + 1),
        show_progress,
        total=total,
        desc=f"{cfg['env']['id']} | {method} | seed={seed}",
    )

    for step in pbar:
        plan, cache_hit = cache.get_or_plan(obs, planner)
        base = plan_chunk(plan.actions, k)

        if agent is None:
            adaptive_gate = 0.0
            gate_z = np.nan
            ramp = 0.0
            effective_gate = 0.0
            residual = np.zeros(act_dim, dtype=np.float32)
            corrected = base
            effective_delta = np.zeros_like(base)
            c = None
        else:
            adaptive_gate, gate_z = _adaptive_gate_for_plan(
                method, rc, normalized_gate, plan.uncertainty, update=True
            )
            ramp = residual_ramp(step, rl_start_local, ramp_steps)
            effective_gate = gated_strength(adaptive_gate, ramp, gate_power)
            c = residual_context(obs, base, effective_gate)

            residual = agent.act(c).reshape(act_dim)
            if method == 'planning_residual':
                residual_chunk = expand_temporal_residual(
                    residual, k, act_dim, decay=temporal_decay
                )
            else:
                residual_chunk = residual.reshape(1, act_dim)

            effective_delta = (
                effective_gate
                * float(rc['residual_scale'])
                * residual_chunk
                * span[None, :]
            )
            corrected = base + effective_delta

        action = np.clip(corrected[0], low, high)
        no, r, term, trunc, info = env.step(action)
        done = term or trunc

        with torch.no_grad():
            pred = wm.predict(obs[None], action[None])
            pe = float(torch.mean(
                (pred.next_obs.squeeze(0) - torch.as_tensor(no, device=device)) ** 2
            ).cpu())

        wm_buf.add(obs, action, r, no, done)
        if step % int(tc['wm_update_interval']) == 0:
            wm_loss = wm.fit_batch(
                wm_buf.sample(min(wm_buf.size, 10000)),
                updates=int(tc['wm_updates']),
                batch_size=int(tc['batch_size']),
            )

        if agent is not None:
            if done:
                nc = np.zeros(context_dim, np.float32)
                cache.clear()
            else:
                nplan = planner.plan(no)
                nbase = plan_chunk(nplan.actions, k)
                next_adaptive, _ = _adaptive_gate_for_plan(
                    method, rc, normalized_gate, nplan.uncertainty, update=False
                )
                next_ramp = residual_ramp(step + 1, rl_start_local, ramp_steps)
                next_gate = gated_strength(next_adaptive, next_ramp, gate_power)
                nc = residual_context(no, nbase, next_gate)
                cache.put(no, nplan)
            rb.add(c, residual, r, nc, done)

        metrics.step(
            r,
            action,
            plan.planning_ms,
            pe,
            plan.uncertainty,
            float(np.linalg.norm(residual)),
            effective_gate,
            adaptive_gate=adaptive_gate,
            gate_z=gate_z,
            ramp=ramp,
            effective_resnorm=float(np.linalg.norm(effective_delta)),
            cache_hit=cache_hit,
        )

        if agent is not None and rb.size >= int(tc['batch_size']) and step >= rl_start_local:
            for _ in range(int(tc['rl_updates_per_step'])):
                agent.update(rb.sample(int(tc['batch_size'])))

        if done:
            row = metrics.finish(info, cfg['env'].get('success_return_threshold'))
            row.update({
                'global_step': step,
                'episode': episode,
                'method': method,
                'env': cfg['env']['id'],
                'seed': seed,
                'wm_loss': wm_loss,
            })
            logger.write(row)
            episode += 1
            metrics.reset()
            obs, _ = env.reset()
            planner.reset()
            cache.clear()

            if show_progress:
                pbar.set_postfix({
                    'ep': episode,
                    'ret': f"{row['episode_return']:.1f}",
                    'len': int(row['episode_length']),
                    'wm': f"{wm_loss:.4f}",
                    'gate': f"{row['gate']:.2f}" if np.isfinite(row['gate']) else 'nan',
                    'cache': f"{row['mpc_cache_hit_rate']:.2f}" if np.isfinite(row['mpc_cache_hit_rate']) else 'nan',
                })
                if episode % console_every == 0:
                    tqdm.write(json.dumps(row, ensure_ascii=False))
            else:
                print(json.dumps(row, ensure_ascii=False))
        else:
            obs = no

        if step % int(tc.get('checkpoint_interval', 5000)) == 0:
            ck = {'world_model': wm.state_dict(), 'step': step, 'method': method}
            if agent is not None:
                ck['agent'] = agent.state_dict()
            if normalized_gate is not None:
                ck['gate_state'] = normalized_gate.state_dict()
            torch.save(ck, out / f'checkpoint_{step}.pt')

    final = {'world_model': wm.state_dict(), 'step': total, 'method': method}
    if agent is not None:
        final['agent'] = agent.state_dict()
    if normalized_gate is not None:
        final['gate_state'] = normalized_gate.state_dict()
    torch.save(final, out / 'final.pt')
    env.close()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/halfcheetah.yaml')
    p.add_argument('--method', choices=sorted(METHODS), default='planning_residual')
    p.add_argument('--steps', type=int)
    p.add_argument('--seed', type=int)
    p.add_argument('--run-name')
    p.add_argument('--no-progress', action='store_true')
    a = p.parse_args()

    cfg = load_yaml(a.config)
    if a.seed is not None:
        cfg['env']['seed'] = a.seed
    print('output:', train(cfg, a.method, a.steps, a.run_name, show_progress=not a.no_progress))


if __name__ == '__main__':
    main()
