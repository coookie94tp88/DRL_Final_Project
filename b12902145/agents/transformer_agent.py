"""
Transformer Actor-Critic for OracleGambit (Phase 2: PPO comparison).

Observation structure (flat vector from OracleGambitEnv)
---------------------------------------------------------
  Player: [history(50 × 8), mask(50), current(5)]  → 455-dim
  Host:   [history(50 × 4), mask(50), current(5)]  → 255-dim

Architecture
------------
1. History tokens (window × hist_feat) → Linear embedding + positional encoding
   → TransformerEncoder (2 layers, 4 heads) → masked-mean pooling  → h_pool
2. Current-round features (5-dim) → Linear → c_emb
3. trunk = concat(h_pool, c_emb)                      (2 × d_model)
4. Actor  head: Linear(128) → Tanh → Linear(num_doors) → Categorical
5. Critic head: Linear(128) → Tanh → Linear(1)          → V(s)

Key interface differences vs. MlpAgent
---------------------------------------
* act()      returns (action, log_prob, **value**)  ← 3-tuple (PPO needs value)
* evaluate() returns (log_probs, entropy, **values**)  ← 3-tuple
* forward()  expects (B, obs_dim) — always batched; call unsqueeze(0) for single obs
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class TransformerAgent(nn.Module):
    """
    Transformer-based Actor-Critic for OracleGambit.

    Parameters
    ----------
    obs_dim        : total flat observation dimension (455 for player, 255 for host)
    num_doors      : number of action choices (4)
    hist_feat_size : features per history timestep (8 for player, 4 for host)
    history_window : number of history steps stored by the env (50)
    d_model        : transformer embedding dimension (default 64)
    nhead          : number of attention heads (default 4)
    num_enc_layers : number of TransformerEncoder layers (default 2)
    ff_dim         : feedforward hidden dim inside transformer (default 256)
    hidden_dim     : MLP hidden dim for actor / critic heads (default 128)
    """

    def __init__(
        self,
        obs_dim: int,
        num_doors: int,
        hist_feat_size: int,
        history_window: int = 50,
        d_model: int = 64,
        nhead: int = 4,
        num_enc_layers: int = 2,
        ff_dim: int = 256,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.history_window = history_window
        self.hist_feat_size = hist_feat_size
        self.current_feat_size = obs_dim - history_window * hist_feat_size - history_window
        if self.current_feat_size <= 0:
            raise ValueError(
                f"Computed current_feat_size={self.current_feat_size} ≤ 0 "
                f"for obs_dim={obs_dim}, history_window={history_window}, "
                f"hist_feat_size={hist_feat_size}."
            )

        # ── History encoder ──────────────────────────────────────────────
        self.hist_proj = nn.Linear(hist_feat_size, d_model)
        self.pos_emb   = nn.Embedding(history_window, d_model)  # learnable positional enc

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=0.0,       # no dropout in RL for stability
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_enc_layers)

        # ── Current-round features ──────────────────────────────────────
        self.curr_proj = nn.Linear(self.current_feat_size, d_model)

        trunk_dim = d_model * 2  # h_pool ‖ c_emb

        # ── Actor head ──────────────────────────────────────────────────
        self.actor = nn.Sequential(
            nn.Linear(trunk_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_doors),
        )

        # ── Critic head ─────────────────────────────────────────────────
        self.critic = nn.Sequential(
            nn.Linear(trunk_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Small orthogonal init for the MLP heads; Xavier for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01 if m.out_features <= 4 else 1.0)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_obs(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split a batched flat observation into (history, bool_mask, current).

        Parameters
        ----------
        obs : (B, obs_dim)

        Returns
        -------
        history : (B, window, hist_feat)   — per-step features
        mask    : (B, window)   bool       — True = valid (non-padding) timestep
        current : (B, curr_feat)           — current-round context
        """
        W = self.history_window
        F = self.hist_feat_size

        hist_flat = obs[:, : W * F]
        mask_flat = obs[:, W * F : W * F + W]
        current   = obs[:, W * F + W :]

        history = hist_flat.view(obs.shape[0], W, F)
        return history, mask_flat.bool(), current

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[Categorical, torch.Tensor]:
        """
        Parameters
        ----------
        obs : (B, obs_dim)  — **always batched** (use unsqueeze(0) for single samples)

        Returns
        -------
        dist  : Categorical over doors  (batch size B)
        value : (B,) — V(s) from the critic head
        """
        history, mask, current = self._parse_obs(obs)
        B, W, _ = history.shape
        device = obs.device

        # ── History encoding ─────────────────────────────────────────────
        pos = torch.arange(W, device=device).unsqueeze(0).expand(B, -1)  # (B, W)
        h   = self.hist_proj(history) + self.pos_emb(pos)                 # (B, W, d_model)

        # src_key_padding_mask: True → IGNORE token (PyTorch convention)
        # Our mask: True = valid → invert for transformer
        all_padding = ~mask.any(dim=1)   # (B,) rows where every token is padding
        # Replace fully-padding rows with a dummy valid mask to avoid NaN from softmax
        safe_mask = mask.clone()
        safe_mask[all_padding, 0] = True  # force at least one valid token

        h = self.transformer(h, src_key_padding_mask=~safe_mask)           # (B, W, d_model)

        # Masked mean pooling (only valid timesteps contribute)
        valid = safe_mask.float().unsqueeze(-1)                   # (B, W, 1)
        h_pool = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)  # (B, d_model)

        # ── Current-round features ────────────────────────────────────────
        c_emb = self.curr_proj(current)                            # (B, d_model)

        # ── Combined trunk ────────────────────────────────────────────────
        trunk = torch.cat([h_pool, c_emb], dim=-1)                # (B, 2*d_model)

        dist  = Categorical(logits=self.actor(trunk))
        value = self.critic(trunk).squeeze(-1)                    # (B,)

        return dist, value

    # ------------------------------------------------------------------
    # Inference (no gradient)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> tuple[int, float, float]:
        """
        Sample one action from the policy.

        Parameters
        ----------
        obs : 1-D numpy array (raw env observation)

        Returns
        -------
        door_index : int   in [0, num_doors)
        log_prob   : float log π(a | s)   — for PPO importance sampling ratio
        value      : float V(s)            — for GAE advantage estimation
        """
        t    = torch.FloatTensor(obs).unsqueeze(0)  # (1, obs_dim)
        dist, value = self.forward(t)
        a    = dist.sample()                         # (1,)
        return (
            int(a.item()),
            float(dist.log_prob(a).item()),
            float(value.item()),
        )

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def evaluate(
        self,
        obs_batch: torch.Tensor,
        action_batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Recompute log-probs, per-sample entropy, and values for a batch.

        Parameters
        ----------
        obs_batch    : (B, obs_dim)
        action_batch : (B,) LongTensor

        Returns
        -------
        log_probs : (B,)
        entropy   : (B,)
        values    : (B,)
        """
        dist, values = self.forward(obs_batch)
        return dist.log_prob(action_batch), dist.entropy(), values

    @torch.no_grad()
    def value(self, obs: np.ndarray) -> float:
        """Evaluate V(s) for a single numpy observation (used for GAE bootstrap)."""
        t = torch.FloatTensor(obs).unsqueeze(0)
        _, v = self.forward(t)
        return float(v.item())
