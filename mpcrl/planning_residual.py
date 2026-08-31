from __future__ import annotations
import numpy as np


def expand_temporal_residual(residual, chunk_len: int, action_dim: int, decay: float = 0.8) -> np.ndarray:
    """Expand one action-space residual into a smooth MPC-plan correction.

    SAC learns only ``action_dim`` residual variables. The vector is applied to
    the current MPC action and propagated through the remaining planning chunk
    with geometric decay. This keeps every learned residual dimension causally
    connected to the action that is actually executed under receding-horizon
    MPC, while still producing a coherent correction of the short plan.
    """
    k = max(int(chunk_len), 1)
    ad = int(action_dim)
    d = float(np.clip(decay, 0.0, 1.0))
    r = np.asarray(residual, dtype=np.float32).reshape(ad)
    scales = np.power(d, np.arange(k, dtype=np.float32))[:, None]
    return scales * r[None, :]
