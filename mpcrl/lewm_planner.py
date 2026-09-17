from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class LatentPlan:
    actions: np.ndarray
    score: float
    planning_ms: float


class LeWMCEMPlanner:
    def __init__(
        self, model, reward_probe, low, high, horizon=12, candidates=256,
        elites=32, iterations=4, init_std=.7, min_std=.05, discount=.99,
        device='cpu', precision='fp32',
    ):
        self.model = model
        self.probe = reward_probe
        self.low = np.asarray(low, np.float32)
        self.high = np.asarray(high, np.float32)
        self.h = horizon
        self.n = candidates
        self.k = elites
        self.it = iterations
        self.init_std = init_std
        self.min_std = min_std
        self.discount = discount
        self.device = torch.device(device)
        self.amp_dtype = torch.bfloat16 if precision == 'bf16' else None

    def _encode_frame(self, frame, count):
        x = torch.as_tensor(frame, device=self.device).permute(2, 0, 1).unsqueeze(0)
        return self.model.encode(x).expand(int(count), -1)

    def _score_tensor(self, z, seq):
        ret = torch.zeros((len(seq), 1), device=self.device)
        g = 1.0
        for t in range(seq.shape[1]):
            ret += g * self.probe(z, seq[:, t]).float()
            z = self.model.latent_step(z, seq[:, t])
            g *= self.discount
        return ret.squeeze(-1)

    @torch.no_grad()
    def score_sequences(self, frame, action_sequences):
        """Score fixed action sequences with the same frozen LeWM/reward probe."""
        seq = torch.as_tensor(
            action_sequences, dtype=torch.float32, device=self.device
        )
        if seq.ndim != 3 or seq.shape[-1] != len(self.low):
            raise ValueError(
                f'action_sequences must be [N,H,{len(self.low)}], got {tuple(seq.shape)}'
            )
        z = self._encode_frame(frame, len(seq))
        return self._score_tensor(z, seq).float().cpu().numpy()

    @torch.no_grad()
    def plan(self, frame):
        t0 = time.perf_counter()
        ad = len(self.low)
        low = torch.tensor(self.low, device=self.device)
        high = torch.tensor(self.high, device=self.device)
        mean = ((low + high) / 2).repeat(self.h, 1)
        std = ((high - low) / 2 * self.init_std).repeat(self.h, 1)
        z0 = self._encode_frame(frame, self.n)
        best = None
        best_score = -1e30

        for _ in range(self.it):
            seq = (
                mean[None]
                + torch.randn(self.n, self.h, ad, device=self.device) * std[None]
            ).clamp(low, high)
            score = self._score_tensor(z0.clone(), seq)
            vals, idx = torch.topk(score, self.k)
            elite = seq[idx]
            mean = elite.mean(0)
            std = elite.std(0, unbiased=False).clamp_min(self.min_std)
            if float(vals[0]) > best_score:
                best_score = float(vals[0])
                best = seq[idx[0]].cpu().numpy()

        return LatentPlan(
            best,
            best_score,
            (time.perf_counter() - t0) * 1000,
        )
