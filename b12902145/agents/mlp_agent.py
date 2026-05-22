"""
MLP Policy Network for OracleGambit (Phase 1 Baseline).

Architecture
------------
  obs (flat vector)
    → Linear(obs_dim, h1) + Tanh
    → Linear(h1, h2)      + Tanh
    → Linear(h2, D)       → logits
    → Categorical(logits) → action (door index 0 .. D-1)

Notes
-----
* One shared MlpAgent instance is used for ALL players (parameter sharing).
  The host has its own separate MlpAgent instance.
* Input is the FULL observation from OracleGambitEnv (history + mask + current).
  This mirrors the Transformer agent in Phase 2 so results are directly comparable.
* No replay buffer is needed — REINFORCE is on-policy (batch is collected fresh
  each update and then discarded).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class MlpAgent(nn.Module):
    """Two-hidden-layer MLP policy outputting a Categorical distribution over doors."""

    def __init__(
        self,
        obs_dim: int,
        num_doors: int,
        hidden_dims: tuple[int, ...] = (256, 128),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.Tanh()])
            in_dim = h
        layers.append(nn.Linear(in_dim, num_doors))
        self.net = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, obs: torch.Tensor) -> Categorical:
        """Return a Categorical distribution given a batch of observations."""
        return Categorical(logits=self.net(obs))

    # ------------------------------------------------------------------
    # Inference (no grad)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> tuple[int, float]:
        """
        Sample one action from the policy.

        Parameters
        ----------
        obs : 1-D numpy array (the raw env observation)

        Returns
        -------
        door_index : int   in [0, num_doors)
        log_prob   : float log π(a | o)  — stored for the REINFORCE update
        """
        t = torch.FloatTensor(obs).unsqueeze(0)
        dist = self(t)
        a = dist.sample()
        return int(a.item()), float(dist.log_prob(a).item())

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def evaluate(
        self,
        obs_batch: torch.Tensor,    # (B, obs_dim)
        action_batch: torch.Tensor,  # (B,)  LongTensor of door indices
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log-probs and entropy for a batch.
        Used by ReinforceRunner during the update step.

        Returns
        -------
        log_probs : (B,)
        entropy   : (B,)
        """
        dist = self(obs_batch)
        return dist.log_prob(action_batch), dist.entropy()
