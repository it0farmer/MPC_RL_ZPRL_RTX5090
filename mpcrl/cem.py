from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
import torch


@dataclass
class PlanResult:
    actions: np.ndarray
    score: float
    uncertainty: float
    planning_ms: float


class CEMPlanner:
    def __init__(self, world_model, action_low, action_high, horizon=15, candidates=512,
                 elites=64, iterations=5, alpha=0.2, init_std=0.6, min_std=0.05,
                 discount=0.99, uncertainty_penalty=0.0):
        self.model = world_model
        self.low = np.asarray(action_low, np.float32); self.high = np.asarray(action_high, np.float32)
        self.horizon, self.candidates, self.elites, self.iterations = horizon, candidates, elites, iterations
        self.alpha, self.init_std, self.min_std, self.discount = alpha, init_std, min_std, discount
        self.uncertainty_penalty = uncertainty_penalty
        self.action_dim = len(self.low); self.prev_mean = None

    def reset(self): self.prev_mean = None

    def _init_mean(self):
        if self.prev_mean is None: return np.tile((self.low+self.high)/2, (self.horizon,1))
        shifted = np.concatenate([self.prev_mean[1:], self.prev_mean[-1:]], axis=0)
        return shifted.copy()

    def plan(self, obs) -> PlanResult:
        t0=time.perf_counter(); device=self.model.device
        mean = torch.as_tensor(self._init_mean(), dtype=torch.float32, device=device)
        span = torch.as_tensor((self.high-self.low)/2, dtype=torch.float32, device=device)
        std = span * self.init_std
        low=torch.as_tensor(self.low,device=device); high=torch.as_tensor(self.high,device=device)
        best_score=-float('inf'); best_seq=None; best_unc=0.0
        for _ in range(self.iterations):
            noise=torch.randn(self.candidates,self.horizon,self.action_dim,device=device)
            seq=(mean.unsqueeze(0)+noise*std.unsqueeze(0)).clamp(low,high)
            scores, unc = self.model.rollout_return(obs, seq, self.discount, self.uncertainty_penalty)
            vals, idx = torch.topk(scores, k=min(self.elites,self.candidates), largest=True)
            elite=seq[idx]; new_mean=elite.mean(0); new_std=elite.std(0,unbiased=False).clamp_min(self.min_std)
            mean=self.alpha*mean+(1-self.alpha)*new_mean
            std=self.alpha*std+(1-self.alpha)*new_std
            if float(vals[0])>best_score:
                best_score=float(vals[0]); best_seq=seq[idx[0]].detach().cpu().numpy(); best_unc=float(unc[idx[0]])
        self.prev_mean = mean.detach().cpu().numpy()
        ms=(time.perf_counter()-t0)*1000.0
        return PlanResult(best_seq, best_score, best_unc, ms)
