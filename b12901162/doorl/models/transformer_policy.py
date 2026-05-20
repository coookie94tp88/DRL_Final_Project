"""Player Transformer policy for DooRL.

Supports three ``parameter_sharing`` modes:

* ``encoder`` (default): a single shared Transformer trunk; per-player heads + a
  per-player identity embedding select among policies.
* ``none``: an array of ``num_players`` independent Transformer policies.
* ``full``: a single Transformer with a single set of heads; identity is injected
  as an extra feature.

Action heads:

* Phase I — ``bribe_pct`` ~ Beta(alpha, beta) on ``[0, 1]``.
* Phase II — ``door`` ~ Categorical(4) and ``bet_pct`` ~ Beta(alpha, beta).

The value head outputs a scalar baseline used by HAPPO/PPO.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Categorical

from doorl.env.doorl_env import PHASE_BET, PHASE_BRIBE


@dataclass
class PlayerPolicyConfig:
    obs_dim: int
    num_players: int
    num_doors: int = 4
    d_model: int = 128
    n_layers: int = 2
    nhead: int = 8
    dim_ff: int = 256
    dropout: float = 0.1
    parameter_sharing: str = "encoder"  # {encoder, none, full}

    def validate(self) -> None:
        if self.parameter_sharing not in {"encoder", "none", "full"}:
            raise ValueError(
                f"parameter_sharing must be one of {{encoder, none, full}}, "
                f"got {self.parameter_sharing!r}"
            )


class _TransformerTrunk(nn.Module):
    def __init__(self, in_dim: int, cfg: PlayerPolicyConfig) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_dim, cfg.d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_ff,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim). Treat as a single-token sequence + CLS.
        x = self.input_proj(x).unsqueeze(1)  # (B, 1, d)
        b = x.size(0)
        cls = self.cls.expand(b, -1, -1)
        h = torch.cat([cls, x], dim=1)
        h = self.encoder(h)
        return h[:, 0]  # CLS


class _Heads(nn.Module):
    def __init__(self, cfg: PlayerPolicyConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.bribe = nn.Linear(d, 2)  # alpha, beta logits
        self.door = nn.Linear(d, cfg.num_doors)
        self.bet = nn.Linear(d, 2)
        self.value = nn.Linear(d, 1)

    def forward(
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.bribe(h), self.door(h), self.bet(h), self.value(h)


def _beta_params(
    raw: torch.Tensor,
    max_concentration: float = 100.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map head outputs to valid Beta concentrations (always finite, bounded)."""
    raw = torch.nan_to_num(raw, nan=0.0, posinf=10.0, neginf=-10.0)
    raw = raw.clamp(-10.0, 10.0)
    alpha = F.softplus(raw[..., 0]) + 1.0
    beta = F.softplus(raw[..., 1]) + 1.0
    cap = max(2.0, float(max_concentration))
    alpha = alpha.clamp(1.0 + 1e-4, cap)
    beta = beta.clamp(1.0 + 1e-4, cap)
    return alpha, beta


class PlayerPolicy(nn.Module):
    """Player policy supporting three parameter-sharing modes."""

    def __init__(
        self, cfg: PlayerPolicyConfig, max_beta_concentration: float = 100.0
    ) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.max_beta_concentration = max_beta_concentration
        n = cfg.num_players

        # The identity embedding is appended to the obs for `encoder` and `full`.
        identity_dim = n if cfg.parameter_sharing in {"encoder", "full"} else 0
        in_dim = cfg.obs_dim + identity_dim

        if cfg.parameter_sharing == "none":
            self.trunks = nn.ModuleList(
                [_TransformerTrunk(cfg.obs_dim, cfg) for _ in range(n)]
            )
            self.heads = nn.ModuleList([_Heads(cfg) for _ in range(n)])
        elif cfg.parameter_sharing == "encoder":
            self.trunk = _TransformerTrunk(in_dim, cfg)
            self.heads = nn.ModuleList([_Heads(cfg) for _ in range(n)])
        else:  # full
            self.trunk = _TransformerTrunk(in_dim, cfg)
            self.head = _Heads(cfg)

    def _id_one_hot(self, agent_idx: torch.Tensor, batch_size: int) -> torch.Tensor:
        n = self.cfg.num_players
        return F.one_hot(agent_idx, num_classes=n).float()

    def _trunk_forward(
        self, obs: torch.Tensor, agent_idx: torch.Tensor
    ) -> torch.Tensor:
        cfg = self.cfg
        if cfg.parameter_sharing == "none":
            outs = []
            for b in range(obs.size(0)):
                idx = int(agent_idx[b].item())
                outs.append(self.trunks[idx](obs[b : b + 1]))
            return torch.cat(outs, dim=0)
        # encoder / full both use the identity-feature trunk
        id_oh = self._id_one_hot(agent_idx, obs.size(0))
        x = torch.cat([obs, id_oh], dim=-1)
        return self.trunk(x)

    def _heads_forward(
        self, h: torch.Tensor, agent_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        if cfg.parameter_sharing == "full":
            return self.head(h)
        # `encoder` or `none`: per-agent head
        bribe_out, door_out, bet_out, value_out = [], [], [], []
        for b in range(h.size(0)):
            idx = int(agent_idx[b].item())
            bribe, door, bet, value = self.heads[idx](h[b : b + 1])
            bribe_out.append(bribe)
            door_out.append(door)
            bet_out.append(bet)
            value_out.append(value)
        return (
            torch.cat(bribe_out, dim=0),
            torch.cat(door_out, dim=0),
            torch.cat(bet_out, dim=0),
            torch.cat(value_out, dim=0),
        )

    def forward(
        self, obs: torch.Tensor, agent_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self._trunk_forward(obs, agent_idx)
        return self._heads_forward(h, agent_idx)

    # ---------- sampling helpers ----------

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        agent_idx: torch.Tensor,
        phase: int,
        deterministic: bool = False,
    ) -> Tuple[dict, dict]:
        """Sample an action conditioned on the env phase.

        Returns a tuple (action_dict, info_dict) where ``info_dict`` contains
        per-component log probs and the value estimate.
        """
        bribe, door, bet, value = self.forward(obs, agent_idx)
        cap = self.max_beta_concentration
        b_alpha, b_beta = _beta_params(bribe, cap)
        bet_alpha, bet_beta = _beta_params(bet, cap)
        door_logits = torch.nan_to_num(door, nan=0.0, posinf=20.0, neginf=-20.0)

        bribe_dist = Beta(b_alpha, b_beta)
        bet_dist = Beta(bet_alpha, bet_beta)
        door_dist = Categorical(logits=door_logits)

        device = obs.device
        batch = obs.size(0)
        if deterministic:
            bribe_sample = b_alpha / (b_alpha + b_beta)
            bet_sample = bet_alpha / (bet_alpha + bet_beta)
            door_sample = door_dist.probs.argmax(dim=-1)
        else:
            bribe_sample = bribe_dist.rsample()
            bet_sample = bet_dist.rsample()
            door_sample = door_dist.sample()

        bribe_sample = bribe_sample.clamp(1e-4, 1.0 - 1e-4)
        bet_sample = bet_sample.clamp(1e-4, 1.0 - 1e-4)

        # Env phase: only bribe in Phase I; door+bet in Phase II (obs has public/private).
        if phase == PHASE_BRIBE:
            bribe_a = bribe_sample
            door_a = torch.zeros(batch, dtype=torch.long, device=device)
            bet_a = torch.full((batch,), 0.5, device=device)
            logp = bribe_dist.log_prob(bribe_a)
        elif phase == PHASE_BET:
            bribe_a = torch.zeros(batch, device=device)
            door_a = door_sample
            bet_a = bet_sample
            logp = door_dist.log_prob(door_a) + bet_dist.log_prob(bet_a)
        else:
            bribe_a = torch.zeros(batch, device=device)
            door_a = torch.zeros(batch, dtype=torch.long, device=device)
            bet_a = torch.full((batch,), 0.5, device=device)
            logp = torch.zeros(batch, device=device)

        action = {
            "bribe_pct": bribe_a.detach().cpu().numpy(),
            "door": door_a.detach().cpu().numpy(),
            "bet_pct": bet_a.detach().cpu().numpy(),
        }
        info = {
            "logp": logp.detach(),
            "logp_bribe": bribe_dist.log_prob(bribe_a).detach(),
            "logp_door": door_dist.log_prob(door_a).detach(),
            "logp_bet": bet_dist.log_prob(bet_a).detach(),
            "value": value.squeeze(-1).detach(),
        }
        return action, info

    def evaluate(
        self,
        obs: torch.Tensor,
        agent_idx: torch.Tensor,
        actions: dict,
        phases: Optional[torch.Tensor] = None,
    ) -> dict:
        """Recompute log-probs, entropies, and value for PPO/HAPPO updates."""
        bribe, door, bet, value = self.forward(obs, agent_idx)
        cap = self.max_beta_concentration
        b_alpha, b_beta = _beta_params(bribe, cap)
        bet_alpha, bet_beta = _beta_params(bet, cap)
        door_logits = torch.nan_to_num(door, nan=0.0, posinf=20.0, neginf=-20.0)
        value = torch.nan_to_num(value.squeeze(-1), nan=0.0, posinf=0.0, neginf=0.0)
        bribe_dist = Beta(b_alpha, b_beta)
        bet_dist = Beta(bet_alpha, bet_beta)
        door_dist = Categorical(logits=door_logits)

        bribe_a = actions["bribe_pct"].clamp(1e-4, 1.0 - 1e-4)
        bet_a = actions["bet_pct"].clamp(1e-4, 1.0 - 1e-4)
        door_a = actions["door"].long()

        logp_bribe = bribe_dist.log_prob(bribe_a)
        logp_door = door_dist.log_prob(door_a)
        logp_bet = bet_dist.log_prob(bet_a)
        if phases is None:
            logp = logp_bribe + logp_door + logp_bet
            ent = (
                bribe_dist.entropy()
                + door_dist.entropy()
                + bet_dist.entropy()
            )
        else:
            phases = phases.long().to(obs.device)
            logp = torch.zeros_like(logp_bribe)
            logp = torch.where(phases == PHASE_BRIBE, logp_bribe, logp)
            logp = torch.where(
                phases == PHASE_BET, logp_door + logp_bet, logp
            )
            ent = torch.zeros_like(logp_bribe)
            ent = torch.where(phases == PHASE_BRIBE, bribe_dist.entropy(), ent)
            ent = torch.where(
                phases == PHASE_BET,
                door_dist.entropy() + bet_dist.entropy(),
                ent,
            )

        return {
            "logp": logp,
            "logp_bribe": logp_bribe,
            "logp_door": logp_door,
            "logp_bet": logp_bet,
            "entropy": ent,
            "entropy_bribe": bribe_dist.entropy(),
            "entropy_door": door_dist.entropy(),
            "entropy_bet": bet_dist.entropy(),
            "value": value,
        }
