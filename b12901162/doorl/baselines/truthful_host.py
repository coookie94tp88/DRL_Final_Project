"""Truthful Host baseline: public + private signals always reveal the true door."""

from __future__ import annotations

import numpy as np


class TruthfulHost:
    def __init__(self, num_players: int, seed: int = 0) -> None:
        self.num_players = num_players
        self.rng = np.random.default_rng(seed)

    def act(self, env, obs):
        true_door = env._round_true_door
        logits = np.full((self.num_players, 4), -5.0, dtype=np.float32)
        logits[:, true_door] = 5.0
        return {
            "public_door": int(true_door),
            "private_logits": logits,
        }
