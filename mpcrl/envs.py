from __future__ import annotations
import numpy as np


def _contiguous_rgb_frame(frame):
    """Return an RGB frame with positive C-contiguous strides.

    Some MuJoCo/Gymnasium render paths return vertically flipped NumPy views
    with a negative stride. PyTorch cannot convert those views with
    ``torch.as_tensor``. Normal contiguous frames are returned without a copy;
    negative/non-contiguous views are copied once at the environment boundary.
    """
    if frame is None:
        return None
    arr = np.asarray(frame)
    return arr if arr.flags.c_contiguous else np.ascontiguousarray(arr)


def make_mujoco_env(env_id: str, seed: int, render_mode=None):
    try:
        import gymnasium as gym
    except ImportError as e:
        raise RuntimeError('缺少 gymnasium。请执行: pip install "gymnasium[mujoco]" mujoco') from e

    env = gym.make(env_id, render_mode=render_mode)
    if render_mode == 'rgb_array':
        class _ContiguousRenderWrapper(gym.Wrapper):
            def render(self):
                return _contiguous_rgb_frame(self.env.render())

        env = _ContiguousRenderWrapper(env)

    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)
    return env, np.asarray(obs, dtype=np.float32), info


def dims(env):
    if len(env.observation_space.shape) != 1 or len(env.action_space.shape) != 1:
        raise ValueError('当前训练入口要求一维连续 observation/action space。')
    return int(env.observation_space.shape[0]), int(env.action_space.shape[0])


def action_bounds(env):
    return env.action_space.low.astype(np.float32), env.action_space.high.astype(np.float32)
