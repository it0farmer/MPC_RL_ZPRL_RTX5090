from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


class BottleneckBasePolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, latent_dim=16, hidden=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh(),
        )

    def encode(self, obs):
        return self.encoder(obs)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, obs):
        return self.decode(self.encode(obs))


def fit_behavior_clone(
    policy,
    obs,
    target_action_norm,
    epochs=80,
    batch_size=256,
    lr=1e-3,
    device='cpu',
    show_progress=True,
):
    """Train the frozen ZPRL-style base policy on an MPC rollout dataset.

    ``epochs`` is a true full-dataset epoch count (the previous implementation
    performed only one random mini-batch update per epoch, which severely
    underfit the base policy). The returned value is the final epoch MSE.
    """
    policy.to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    o = torch.as_tensor(obs, dtype=torch.float32, device=device)
    a = torch.as_tensor(target_action_norm, dtype=torch.float32, device=device)
    n = len(o)
    if n == 0:
        raise ValueError('behavior-clone dataset is empty')

    batch_size = max(1, min(int(batch_size), n))
    losses = []
    iterator = tqdm(
        range(int(epochs)),
        desc='ZPRL base-policy BC',
        disable=not show_progress,
        dynamic_ncols=True,
        mininterval=0.5,
        leave=False,
    )
    for _ in iterator:
        perm = torch.randperm(n, device=device)
        epoch_losses = []
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            loss = F.mse_loss(policy(o[idx]), a[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach()))
        epoch_loss = float(np.mean(epoch_losses))
        losses.append(epoch_loss)
        if show_progress:
            iterator.set_postfix(mse=f'{epoch_loss:.5f}')

    for p in policy.parameters():
        p.requires_grad_(False)
    policy.eval()
    return float(np.mean(losses[-min(5, len(losses)):]))
