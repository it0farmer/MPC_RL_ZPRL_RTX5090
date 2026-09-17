from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SafeguardResult:
    scale: float
    predicted_gain: float
    corrected_plan: np.ndarray


def expand_temporal_residual(residual, chunk_len: int, action_dim: int, decay: float = 0.8) -> np.ndarray:
    """Expand one action-space residual into a smooth MPC-plan correction."""
    k = max(int(chunk_len), 1)
    ad = int(action_dim)
    d = float(np.clip(decay, 0.0, 1.0))
    r = np.asarray(residual, dtype=np.float32).reshape(ad)
    scales = np.power(d, np.arange(k, dtype=np.float32))[:, None]
    return scales * r[None, :]


def safeguard_residual_plan(
    world_model,
    obs,
    base_plan,
    delta_chunk,
    action_low,
    action_high,
    discount=0.99,
    uncertainty_penalty=0.0,
    scales=(1.0, 0.5, 0.25, 0.0),
    min_improvement=0.0,
) -> SafeguardResult:
    """Select the strongest model-predicted non-degrading residual correction.

    The same learned residual is evaluated at several shrinkage factors.  The
    unmodified MPC plan (scale=0) is always included.  A non-zero correction is
    accepted only when its world-model return is at least ``min_improvement``
    above the re-scored MPC plan.  This acts as a lightweight action shield;
    it does not alter the actor or the baseline MPC optimization problem.
    """
    base = np.asarray(base_plan, dtype=np.float32)
    delta = np.asarray(delta_chunk, dtype=np.float32)
    low = np.asarray(action_low, dtype=np.float32)
    high = np.asarray(action_high, dtype=np.float32)
    if base.ndim != 2:
        raise ValueError(f'base_plan must be [H,A], got {base.shape}')
    if delta.ndim != 2 or delta.shape[1] != base.shape[1]:
        raise ValueError(
            f'delta_chunk must be [K,{base.shape[1]}], got {delta.shape}'
        )
    k = min(len(delta), len(base))
    if k <= 0:
        return SafeguardResult(0.0, 0.0, base.copy())

    candidates = []
    clean_scales = []
    for value in tuple(scales) + (0.0,):
        s = float(np.clip(value, 0.0, 1.0))
        if any(abs(s - old) < 1e-12 for old in clean_scales):
            continue
        plan = base.copy()
        plan[:k] = np.clip(base[:k] + s * delta[:k], low, high)
        candidates.append(plan)
        clean_scales.append(s)

    with torch.no_grad():
        scores, _ = world_model.rollout_return(
            obs,
            np.stack(candidates, axis=0),
            float(discount),
            float(uncertainty_penalty),
        )
    score_values = scores.detach().float().cpu().numpy().reshape(-1)
    baseline_idx = clean_scales.index(0.0)
    baseline_score = float(score_values[baseline_idx])

    best_idx = int(np.nanargmax(score_values))
    best_scale = clean_scales[best_idx]
    best_gain = float(score_values[best_idx] - baseline_score)
    if best_scale == 0.0 or best_gain < float(min_improvement):
        best_idx = baseline_idx
        best_scale = 0.0
        best_gain = 0.0

    return SafeguardResult(
        scale=float(best_scale),
        predicted_gain=float(best_gain),
        corrected_plan=np.asarray(candidates[best_idx], dtype=np.float32),
    )
