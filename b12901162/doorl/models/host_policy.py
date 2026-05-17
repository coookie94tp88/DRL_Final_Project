"""Host policy for DooRL.

Architecture:

* A Transformer encoder operates on per-player tokens (each token mixes the
  player's bribe and balance) plus a CLS token.
* The CLS token feeds a Categorical(4) head for ``public_door`` and a scalar
  value head.
* Each per-player token feeds a **shared per-player head** producing the
  4-way logits for the private signal of that player. Sharing across players
  keeps the action space tractable as N grows (R-T7 mitigation).

The full Host action is

    public_door ~ Categorical(4)
    private_logits[i] = head_shared(token_i)   # (N, 4)

with the env sampling ``private_door_i ~ softmax(private_logits[i])``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


@dataclass
class HostPolicyConfig:
    obs_dim: int
    num_players: int
    num_doors: int = 4
    d_model: int = 128
    n_layers: int = 2
    nhead: int = 8
    dim_ff: int = 256
    dropout: float = 0.1


class HostPolicy(nn.Module):
    def __init__(self, cfg: HostPolicyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        n = cfg.num_players

        # tokenize: project the (huge) flat obs vector into a single "global" token,
        # plus per-player tokens built from a sliced view of the obs concatenated
        # with the player's index embedding.
        self.global_proj = nn.Linear(cfg.obs_dim, d)
        self.player_proj = nn.Linear(cfg.obs_dim, d)
        self.player_embed = nn.Embedding(n, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.cls, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_ff,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)

        self.public_head = nn.Linear(d, cfg.num_doors)
        self.private_head = nn.Linear(d, cfg.num_doors)  # SHARED across players
        self.value_head = nn.Linear(d, 1)

    def _tokens(self, obs: torch.Tensor) -> torch.Tensor:
        """Return (B, 1 + N, d) sequence: CLS + N per-player tokens.

        We currently use the same obs vector for all per-player tokens and rely
        on the player_embed lookup to distinguish them. This is intentionally
        simple — extending to slice-based per-player tokens is left for v2.
        """
        b = obs.size(0)
        n = self.cfg.num_players
        d = self.cfg.d_model
        global_tok = self.global_proj(obs).unsqueeze(1)  # (B, 1, d)
        player_proj = self.player_proj(obs).unsqueeze(1).expand(b, n, d)  # (B, N, d)
        idx = torch.arange(n, device=obs.device).unsqueeze(0).expand(b, n)
        player_tok = player_proj + self.player_embed(idx)  # (B, N, d)
        cls = self.cls.expand(b, -1, -1)
        return torch.cat([cls, global_tok, player_tok], dim=1)  # (B, 2+N, d)

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self._tokens(obs)
        h = self.encoder(tokens)
        cls = h[:, 0]
        player_tokens = h[:, 2:]  # (B, N, d)

        public_logits = self.public_head(cls)
        private_logits = self.private_head(player_tokens)  # (B, N, 4)
        value = self.value_head(cls)
        return public_logits, private_logits, value

    @torch.no_grad()
    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[dict, dict]:
        public_logits, private_logits, value = self.forward(obs)
        pub_dist = Categorical(logits=public_logits)
        if deterministic:
            pub = pub_dist.probs.argmax(dim=-1)
        else:
            pub = pub_dist.sample()
        action = {
            "public_door": pub.detach().cpu().numpy(),
            "private_logits": private_logits.detach().cpu().numpy(),
        }
        info = {
            "logp_public": pub_dist.log_prob(pub).detach(),
            "private_logits": private_logits.detach(),
            "value": value.squeeze(-1).detach(),
        }
        return action, info

    def evaluate(self, obs: torch.Tensor, actions: dict) -> dict:
        public_logits, private_logits, value = self.forward(obs)
        pub_dist = Categorical(logits=public_logits)
        pub_a = actions["public_door"].long()

        return {
            "logp_public": pub_dist.log_prob(pub_a),
            "entropy_public": pub_dist.entropy(),
            "private_logits": private_logits,
            "value": value.squeeze(-1),
        }
