from __future__ import annotations

import numpy as np
import torch


def normalize_action(action, low, high):
    """Map an environment action to approximately [-1, 1]."""
    action = np.asarray(action, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    half = np.maximum((high - low) * 0.5, 1e-6)
    mid = (high + low) * 0.5
    return np.clip((action - mid) / half, -1.0, 1.0).astype(np.float32)


def apply_action_residual(base_action, residual, low, high, scale, ramp=1.0):
    """Apply a bounded RL residual to the first MPC action.

    ``residual`` is expected in [-1, 1]. ``scale=1`` would allow the
    correction to span one half action range. Paper experiments use a much
    smaller scale so MPC remains the dominant controller.
    """
    base = np.asarray(base_action, dtype=np.float32)
    residual = np.asarray(residual, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    half = (high - low) * 0.5
    effective = np.clip(float(ramp), 0.0, 1.0) * float(scale) * half * residual
    action = np.clip(base + effective, low, high).astype(np.float32)
    return action, effective.astype(np.float32)


def effective_rank(latents, eps=1e-12):
    """Entropy effective rank of a batch of latent representations.

    The representation is centered, singular values are normalized into a
    probability vector p_i, and exp(-sum p_i log p_i) is returned. This is a
    representation diagnostic; when LeWM is frozen, MPC and MPC+RL should
    have the same effective rank by construction.
    """
    if torch.is_tensor(latents):
        x = latents.detach().float().cpu().numpy()
    else:
        x = np.asarray(latents, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 2:
        return float('nan')
    x = x - np.mean(x, axis=0, keepdims=True)
    s = np.linalg.svd(x, compute_uv=False)
    s = np.maximum(s, 0.0)
    total = float(s.sum())
    if not np.isfinite(total) or total <= eps:
        return 0.0
    p = s / total
    p = p[p > eps]
    return float(np.exp(-(p * np.log(p)).sum()))


def residual_ramp(step, start_steps, ramp_steps):
    if step < int(start_steps):
        return 0.0
    if int(ramp_steps) <= 0:
        return 1.0
    return float(np.clip((step - int(start_steps)) / float(ramp_steps), 0.0, 1.0))
