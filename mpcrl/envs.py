from __future__ import annotations
import numpy as np


def make_mujoco_env(env_id: str, seed: int, render_mode=None):
    try:
        import gymnasium as gym
    except ImportError as e:
        raise RuntimeError('缺少 gymnasium。请执行: pip install "gymnasium[mujoco]" mujoco') from e
    env = gym.make(env_id, render_mode=render_mode)
    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)
    return env, np.asarray(obs, dtype=np.float32), info


def dims(env):
    if len(env.observation_space.shape) != 1 or len(env.action_space.shape) != 1:
        raise ValueError('当前训练入口要求一维连续 observation/action space。')
    return int(env.observation_space.shape[0]), int(env.action_space.shape[0])


def action_bounds(env):
    return env.action_space.low.astype(np.float32), env.action_space.high.astype(np.float32)
