"""Greedy player baseline: always bribe a small fixed pct, follow public signal, bet max."""

from __future__ import annotations

import numpy as np


class GreedyPlayer:
    def __init__(self, bribe_pct: float = 0.05, bet_pct: float = 1.0) -> None:
        self.bribe_pct = float(bribe_pct)
        self.bet_pct = float(bet_pct)

    def act(self, env, obs, i: int, phase: int):
        if phase == 0:
            return {
                "bribe_pct": np.array([self.bribe_pct], dtype=np.float32),
                "door": 0,
                "bet_pct": np.array([0.0], dtype=np.float32),
            }
        return {
            "bribe_pct": np.array([0.0], dtype=np.float32),
            "door": int(env._round_public_signal),
            "bet_pct": np.array([self.bet_pct], dtype=np.float32),
        }
