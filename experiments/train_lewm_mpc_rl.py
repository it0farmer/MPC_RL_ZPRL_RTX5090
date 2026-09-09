from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mpcrl.config import load_yaml, save_yaml
from mpcrl.envs import action_bounds, dims, make_mujoco_env
from mpcrl.lewm import LeWorldModel, RewardProbe
from mpcrl.lewm_planner import LeWMCEMPlanner
from mpcrl.lewm_rl import (
    apply_action_residual,
    effective_rank,
    normalize_action,
    residual_ramp,
)
from mpcrl.metrics import CSVLogger, EpisodeMetrics
from mpcrl.replay import ResidualReplay
from mpcrl.sac import ResidualSAC
from mpcrl.utils import (
    accelerator_summary,
    autocast_context,
    configure_accelerator,
    set_seed,
)


def resize_frame(img, size):
    x = torch.as_tensor(img).permute(2, 0, 1).unsqueeze(0).float()
    return (
        F.interpolate(x, (size, size), mode='bilinear', align_corners=False)
        .squeeze(0)
        .permute(1, 2, 0)
        .byte()
        .numpy()
    )


@torch.no_grad()
def encode_one(model, frame, device):
    x = torch.as_tensor(frame, device=device).permute(2, 0, 1).unsqueeze(0)
    return model.encode(x).float().squeeze(0).cpu().numpy().astype(np.float32)


def collect_dataset(env, size, transitions):
    data = []
    env.reset()
    frame = resize_frame(env.render(), size)
    for _ in range(int(transitions)):
        action = env.action_space.sample()
        _, reward, terminated, truncated, _ = env.step(action)
        next_frame = resize_frame(env.render(), size)
        data.append((frame, action.astype(np.float32), next_frame, float(reward)))
        frame = next_frame
        if terminated or truncated:
            env.reset()
            frame = resize_frame(env.render(), size)
    return data


def train_lewm_and_probe(cfg, data, action_dim, device, precision):
    lc = cfg['lewm']
    model = LeWorldModel(
        action_dim,
        lc['latent_dim'],
        lc['hidden_dim'],
        lc['sigreg_lambda'],
        lc['sigreg_projections'],
        lc['sigreg_knots'],
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lc['lr']))
    batch_size = int(lc['batch_size'])

    for epoch in range(int(lc['epochs'])):
        random.shuffle(data)
        logs = []
        for i in range(0, len(data) - batch_size + 1, batch_size):
            batch = data[i:i + batch_size]
            frames = torch.as_tensor(
                np.stack([x[0] for x in batch]), device=device
            ).permute(0, 3, 1, 2)
            actions = torch.as_tensor(
                np.stack([x[1] for x in batch]), dtype=torch.float32, device=device
            )
            next_frames = torch.as_tensor(
                np.stack([x[2] for x in batch]), device=device
            ).permute(0, 3, 1, 2)
            with autocast_context(device, precision):
                loss, info = model.loss(frames, actions, next_frames)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            logs.append(info)
        if not logs:
            raise RuntimeError('LeWM training dataset is smaller than one batch.')
        print(
            f"LeWM epoch {epoch + 1}/{int(lc['epochs'])}: "
            f"total={np.mean([x['total'] for x in logs]):.6f} "
            f"pred={np.mean([x['pred'] for x in logs]):.6f} "
            f"sigreg={np.mean([x['sigreg'] for x in logs]):.6f}"
        )

    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    probe = RewardProbe(lc['latent_dim'], action_dim).to(device)
    probe_opt = torch.optim.Adam(
        probe.parameters(), lr=float(cfg['reward_probe']['lr'])
    )
    probe_losses = []
    for epoch in range(int(cfg['reward_probe']['epochs'])):
        random.shuffle(data)
        losses = []
        for i in range(0, len(data) - batch_size + 1, batch_size):
            batch = data[i:i + batch_size]
            frames = torch.as_tensor(
                np.stack([x[0] for x in batch]), device=device
            ).permute(0, 3, 1, 2)
            actions = torch.as_tensor(
                np.stack([x[1] for x in batch]), dtype=torch.float32, device=device
            )
            rewards = torch.as_tensor(
                [[x[3]] for x in batch], dtype=torch.float32, device=device
            )
            with torch.no_grad():
                z = model.encode(frames)
            with autocast_context(device, precision):
                pred_reward = probe(z, actions)
            loss = F.mse_loss(pred_reward.float(), rewards.float())
            probe_opt.zero_grad(set_to_none=True)
            loss.backward()
            probe_opt.step()
            losses.append(float(loss.detach()))
        mean_loss = float(np.mean(losses))
        probe_losses.append(mean_loss)
        print(
            f"Reward probe epoch {epoch + 1}/{int(cfg['reward_probe']['epochs'])}: "
            f"loss={mean_loss:.6f}"
        )

    probe.eval()
    for p in probe.parameters():
        p.requires_grad_(False)

    return model, probe, float(probe_losses[-1])


@torch.no_grad()
def representation_diagnostics(model, data, device, max_samples=4096):
    n = min(int(max_samples), len(data))
    if n < 2:
        return {'effective_rank': float('nan'), 'latent_dim': float('nan')}
    idx = np.linspace(0, len(data) - 1, n).astype(int)
    chunks = []
    for start in range(0, n, 256):
        ids = idx[start:start + 256]
        frames = torch.as_tensor(
            np.stack([data[i][0] for i in ids]), device=device
        ).permute(0, 3, 1, 2)
        chunks.append(model.encode(frames).float().cpu())
    z = torch.cat(chunks, dim=0)
    return {
        'effective_rank': effective_rank(z),
        'latent_dim': int(z.shape[-1]),
        'samples': int(z.shape[0]),
    }


def build_planner(cfg, model, probe, low, high, device, precision):
    mc = cfg['mpc']
    return LeWMCEMPlanner(
        model,
        probe,
        low,
        high,
        int(mc['horizon']),
        int(mc['candidates']),
        int(mc['elites']),
        int(mc['iterations']),
        float(mc['init_std']),
        float(mc['min_std']),
        float(mc['discount']),
        device,
        precision,
    )


def make_eval_logger(out_dir):
    fields = [
        'eval_episode', 'eval_seed', 'train_seed', 'method', 'env',
        'episode_return', 'episode_length', 'success', 'mpc_ms',
        'prediction_mse', 'uncertainty', 'residual_norm',
        'effective_residual_norm', 'gate', 'residual_ramp',
        'action_d1', 'action_d2', 'effective_rank',
    ]
    return CSVLogger(str(Path(out_dir) / 'eval.csv'), fields)


def evaluate_controller(
    env,
    cfg,
    planner,
    model,
    device,
    low,
    high,
    train_seed,
    n_eval,
    method,
    out_dir,
    effective_rank_value,
    agent=None,
):
    eval_log = make_eval_logger(out_dir)
    size = int(cfg['env']['render_size'])
    rlcfg = cfg.get('rl_residual', {})
    residual_scale = float(rlcfg.get('residual_scale', 0.0))
    returns = []
    successes = []

    for episode in range(int(n_eval)):
        eval_seed = 700000 + int(train_seed) * 1000 + episode
        np.random.seed(eval_seed)
        torch.manual_seed(eval_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(eval_seed)
        env.reset(seed=eval_seed)
        em = EpisodeMetrics()
        info = {}

        while True:
            frame = resize_frame(env.render(), size)
            plan = planner.plan(frame)
            base_action = plan.actions[0].astype(np.float32)

            if agent is None:
                raw_residual = np.zeros_like(base_action, dtype=np.float32)
                action = base_action
                effective = np.zeros_like(base_action, dtype=np.float32)
                ramp = 0.0
            else:
                z = encode_one(model, frame, device)
                context = np.concatenate(
                    [z, normalize_action(base_action, low, high)]
                ).astype(np.float32)
                raw_residual = agent.act(context, deterministic=True).astype(np.float32)
                ramp = 1.0
                action, effective = apply_action_residual(
                    base_action,
                    raw_residual,
                    low,
                    high,
                    residual_scale,
                    ramp=1.0,
                )

            _, reward, terminated, truncated, info = env.step(action)
            em.step(
                reward,
                action,
                mpc_ms=plan.planning_ms,
                resnorm=float(np.linalg.norm(raw_residual)),
                gate=ramp,
                ramp=ramp,
                effective_resnorm=float(np.linalg.norm(effective)),
            )
            if terminated or truncated:
                break

        metrics = em.finish(info, cfg['env'].get('success_return_threshold'))
        returns.append(float(metrics['episode_return']))
        if np.isfinite(metrics['success']):
            successes.append(float(metrics['success']))
        row = dict(metrics)
        row.update(
            eval_episode=episode,
            eval_seed=eval_seed,
            train_seed=train_seed,
            method=method,
            env=cfg['env']['id'],
            effective_rank=effective_rank_value,
        )
        eval_log.write(row)
        print(row)

    result = {
        'return_mean': float(np.mean(returns)),
        'return_std': float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0,
        'success_rate': (
            100.0 * float(np.mean(successes)) if successes else float('nan')
        ),
    }
    print(
        f"FINAL_EVAL method={method} seed={train_seed} episodes={len(returns)} "
        f"return={result['return_mean']:.3f}±{result['return_std']:.3f} "
        f"success_rate={result['success_rate']:.2f}%"
    )
    return result


def train_residual_controller(
    env,
    cfg,
    planner,
    model,
    device,
    precision,
    low,
    high,
    seed,
    out_dir,
    steps_override=None,
):
    rlcfg = cfg['rl_residual']
    latent_dim = int(cfg['lewm']['latent_dim'])
    action_dim = int(len(low))
    context_dim = latent_dim + action_dim
    batch_size = int(rlcfg.get('batch_size', 512))
    total_steps = int(steps_override or rlcfg.get('total_steps', 50000))
    random_steps = int(rlcfg.get('random_steps', 3000))
    start_steps = int(rlcfg.get('start_steps', random_steps))
    ramp_steps = int(rlcfg.get('ramp_steps', 10000))
    residual_scale = float(rlcfg.get('residual_scale', 0.1))
    updates_per_step = int(rlcfg.get('updates_per_step', 1))

    agent = ResidualSAC(
        context_dim=context_dim,
        residual_dim=action_dim,
        hidden=int(rlcfg.get('hidden_dim', 512)),
        lr=float(rlcfg.get('lr', 3e-4)),
        gamma=float(rlcfg.get('gamma', 0.99)),
        tau=float(rlcfg.get('tau', 0.005)),
        init_alpha=float(rlcfg.get('init_alpha', 0.1)),
        target_entropy_scale=float(rlcfg.get('target_entropy_scale', 0.7)),
        consistency_coef=float(rlcfg.get('consistency_coef', 0.05)),
        device=device,
        precision=precision,
    )
    replay = ResidualReplay(
        capacity=int(rlcfg.get('replay_capacity', max(100000, total_steps + 1))),
        context_dim=context_dim,
        residual_dim=action_dim,
    )

    episode_fields = [
        'global_step', 'episode', 'method', 'env', 'seed', 'episode_return',
        'episode_length', 'success', 'mpc_ms', 'prediction_mse', 'uncertainty',
        'residual_norm', 'effective_residual_norm', 'gate', 'residual_ramp',
        'action_d1', 'action_d2', 'q_loss', 'actor_loss', 'alpha',
    ]
    episode_log = CSVLogger(str(Path(out_dir) / 'episodes.csv'), episode_fields)

    size = int(cfg['env']['render_size'])
    global_step = 0
    episode = 0
    last_update = {'q_loss': np.nan, 'actor_loss': np.nan, 'alpha': np.nan}

    while global_step < total_steps:
        env.reset(seed=seed * 100000 + episode)
        frame = resize_frame(env.render(), size)
        plan = planner.plan(frame)
        em = EpisodeMetrics()
        info = {}

        while global_step < total_steps:
            base_action = plan.actions[0].astype(np.float32)
            z = encode_one(model, frame, device)
            context = np.concatenate(
                [z, normalize_action(base_action, low, high)]
            ).astype(np.float32)

            if global_step < random_steps:
                raw_residual = np.random.uniform(
                    -1.0, 1.0, size=action_dim
                ).astype(np.float32)
            else:
                raw_residual = agent.act(context, deterministic=False).astype(np.float32)

            ramp = residual_ramp(global_step, start_steps, ramp_steps)
            action, effective = apply_action_residual(
                base_action,
                raw_residual,
                low,
                high,
                residual_scale,
                ramp=ramp,
            )

            _, reward, terminated, truncated, info = env.step(action)
            global_step += 1
            done = bool(terminated or truncated)

            em.step(
                reward,
                action,
                mpc_ms=plan.planning_ms,
                resnorm=float(np.linalg.norm(raw_residual)),
                gate=ramp,
                ramp=ramp,
                effective_resnorm=float(np.linalg.norm(effective)),
            )

            if done:
                next_context = np.zeros(context_dim, dtype=np.float32)
            else:
                next_frame = resize_frame(env.render(), size)
                next_plan = planner.plan(next_frame)
                next_base = next_plan.actions[0].astype(np.float32)
                next_z = encode_one(model, next_frame, device)
                next_context = np.concatenate(
                    [next_z, normalize_action(next_base, low, high)]
                ).astype(np.float32)

            replay.add(
                context,
                raw_residual,
                float(reward),
                next_context,
                float(done),
            )

            if global_step >= start_steps and replay.size >= batch_size:
                for _ in range(updates_per_step):
                    last_update = agent.update(replay.sample(batch_size))

            if done or global_step >= total_steps:
                metrics = em.finish(
                    info, cfg['env'].get('success_return_threshold')
                )
                row = dict(metrics)
                row.update(
                    global_step=global_step,
                    episode=episode,
                    method='lewm_mpc_rl',
                    env=cfg['env']['id'],
                    seed=seed,
                    q_loss=last_update.get('q_loss', np.nan),
                    actor_loss=last_update.get('actor_loss', np.nan),
                    alpha=last_update.get('alpha', np.nan),
                )
                episode_log.write(row)
                print(
                    f"RL step={global_step}/{total_steps} episode={episode} "
                    f"return={metrics['episode_return']:.3f} "
                    f"ramp={ramp:.3f} eff_res={metrics['effective_residual_norm']:.4f}"
                )
                episode += 1
                break

            frame = next_frame
            plan = next_plan

    return agent


def write_representation_csv(path, cfg, seed, diag, probe_mse):
    fields = ['env', 'seed', 'latent_dim', 'effective_rank', 'samples', 'reward_probe_mse']
    log = CSVLogger(str(path), fields)
    row = {
        'env': cfg['env']['id'],
        'seed': seed,
        'latent_dim': diag['latent_dim'],
        'effective_rank': diag['effective_rank'],
        'samples': diag['samples'],
        'reward_probe_mse': probe_mse,
    }
    log.write(row)


def main():
    ap = argparse.ArgumentParser(
        description='Paired LeWM-MPC vs LeWM-MPC+RL experiment using one frozen LeWM.'
    )
    ap.add_argument('--config', required=True)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--rl-steps', type=int, default=None)
    ap.add_argument('--eval-episodes', type=int, default=None)
    ap.add_argument('--run-root', default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed = int(args.seed if args.seed is not None else cfg['env'].get('seed', 0))
    cfg['env']['seed'] = seed
    if 'rl_residual' not in cfg:
        raise SystemExit(
            'Config is missing rl_residual. Pull the latest RTX5090 LeWM configs.'
        )

    set_seed(seed)
    hw = cfg.get('hardware', {})
    precision = str(hw.get('precision', 'fp32')).lower()
    device = configure_accelerator(hw)
    print(accelerator_summary(device, precision))

    env, _, _ = make_mujoco_env(
        cfg['env']['id'], seed, render_mode='rgb_array'
    )
    _, action_dim = dims(env)
    low, high = action_bounds(env)
    size = int(cfg['env']['render_size'])

    print('\n=== Stage 1/5: collect one shared LeWM dataset ===')
    data = collect_dataset(env, size, int(cfg['collect']['transitions']))

    print('\n=== Stage 2/5: train one shared LeWM + reward probe ===')
    model, probe, probe_mse = train_lewm_and_probe(
        cfg, data, action_dim, device, precision
    )
    diag = representation_diagnostics(model, data, device)
    print(
        f"REPRESENTATION effective_rank={diag['effective_rank']:.3f} "
        f"latent_dim={diag['latent_dim']} samples={diag['samples']}"
    )

    planner = build_planner(cfg, model, probe, low, high, device, precision)
    n_eval = int(args.eval_episodes or cfg.get('eval', {}).get('episodes', 10))

    run_root = Path(
        args.run_root or cfg.get('logging', {}).get('root', 'runs')
    )
    tag = int(time.time())
    base_out = run_root / (
        f"{cfg['env']['id']}__lewm_mpc__seed{seed}__pair{tag}"
    )
    rl_out = run_root / (
        f"{cfg['env']['id']}__lewm_mpc_rl__seed{seed}__pair{tag}"
    )
    base_out.mkdir(parents=True, exist_ok=True)
    rl_out.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, base_out / 'config.yaml')
    save_yaml(cfg, rl_out / 'config.yaml')
    write_representation_csv(
        base_out / 'representation.csv', cfg, seed, diag, probe_mse
    )
    write_representation_csv(
        rl_out / 'representation.csv', cfg, seed, diag, probe_mse
    )

    # A minimal episodes.csv is useful for generic run discovery. Baseline has
    # no online learning, so global_step=0 is intentional.
    base_episode_log = CSVLogger(
        str(base_out / 'episodes.csv'),
        ['global_step', 'episode', 'method', 'env', 'seed', 'episode_return', 'episode_length'],
    )
    base_episode_log.write({
        'global_step': 0,
        'episode': 0,
        'method': 'lewm_mpc',
        'env': cfg['env']['id'],
        'seed': seed,
        'episode_return': np.nan,
        'episode_length': np.nan,
    })

    print('\n=== Stage 3/5: evaluate frozen LeWM-MPC baseline ===')
    base_eval = evaluate_controller(
        env,
        cfg,
        planner,
        model,
        device,
        low,
        high,
        seed,
        n_eval,
        'lewm_mpc',
        base_out,
        diag['effective_rank'],
        agent=None,
    )

    print('\n=== Stage 4/5: train RL residual at the MPC action stage ===')
    # Re-seed before controller learning. LeWM/probe remain frozen.
    set_seed(seed + 12345)
    agent = train_residual_controller(
        env,
        cfg,
        planner,
        model,
        device,
        precision,
        low,
        high,
        seed,
        rl_out,
        steps_override=args.rl_steps,
    )

    print('\n=== Stage 5/5: evaluate LeWM-MPC+RL on identical eval seeds ===')
    rl_eval = evaluate_controller(
        env,
        cfg,
        planner,
        model,
        device,
        low,
        high,
        seed,
        n_eval,
        'lewm_mpc_rl',
        rl_out,
        diag['effective_rank'],
        agent=agent,
    )

    delta = rl_eval['return_mean'] - base_eval['return_mean']
    pct = delta / max(abs(base_eval['return_mean']), 1e-9) * 100.0
    success_delta = (
        rl_eval['success_rate'] - base_eval['success_rate']
        if np.isfinite(base_eval['success_rate']) and np.isfinite(rl_eval['success_rate'])
        else float('nan')
    )
    pair_summary = {
        'env': cfg['env']['id'],
        'seed': seed,
        'baseline_return_mean': base_eval['return_mean'],
        'rl_return_mean': rl_eval['return_mean'],
        'delta_return': delta,
        'improvement_pct': pct,
        'baseline_success_rate': base_eval['success_rate'],
        'rl_success_rate': rl_eval['success_rate'],
        'success_rate_delta': success_delta,
        'effective_rank': diag['effective_rank'],
        'shared_world_model': True,
        'rl_steps': int(args.rl_steps or cfg['rl_residual']['total_steps']),
    }
    with open(rl_out / 'pair_summary.json', 'w', encoding='utf-8') as f:
        json.dump(pair_summary, f, indent=2, ensure_ascii=False)

    torch.save(
        {
            'lewm': model.state_dict(),
            'reward_probe': probe.state_dict(),
            'residual_sac': agent.state_dict(),
            'config': cfg,
            'pair_summary': pair_summary,
        },
        rl_out / 'final.pt',
    )
    torch.save(
        {
            'lewm': model.state_dict(),
            'reward_probe': probe.state_dict(),
            'config': cfg,
            'pair_summary': pair_summary,
        },
        base_out / 'final.pt',
    )
    env.close()

    print('\n=== PAIRED RESULT ===')
    print(json.dumps(pair_summary, indent=2, ensure_ascii=False))
    print('baseline output:', base_out)
    print('rl output:', rl_out)


if __name__ == '__main__':
    main()
