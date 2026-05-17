"""Noisy-truthful Host: public always true; private truthfulness scales with bribe."""

from __future__ import annotations

import numpy as np


class NoisyTruthfulHost:
    def __init__(self, num_players: int, seed: int = 0) -> None:
        self.num_players = num_players
        self.rng = np.random.default_rng(seed)

    def act(self, env, obs):
        true_door = env._round_true_door
        logits = np.zeros((self.num_players, 4), dtype=np.float32)
        bribes = env._round_bribe_pcts  # (N,)
        for i in range(self.num_players):
            # truth strength scales with bribe; baseline near-uniform when bribe=0
            strength = 1.0 + 8.0 * float(bribes[i])
            logits[i, :] = -strength / 3.0
            logits[i, true_door] = strength
        return {
            "public_door": int(true_door),
            "private_logits": logits,
        }
