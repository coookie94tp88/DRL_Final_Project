"""Random Host baseline: uniform softmax private logits + uniform public door."""

from __future__ import annotations

import numpy as np


class RandomHost:
    def __init__(self, num_players: int, seed: int = 0) -> None:
        self.num_players = num_players
        self.rng = np.random.default_rng(seed)

    def act(self, env, obs):
        return {
            "public_door": int(self.rng.integers(0, 4)),
            "private_logits": self.rng.normal(
                0.0, 1.0, size=(self.num_players, 4)
            ).astype(np.float32),
        }
