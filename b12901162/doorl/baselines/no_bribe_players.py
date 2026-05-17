"""No-bribe baseline: never bribe; follow private signal anyway (uniform if bribe=0)."""

from __future__ import annotations

import numpy as np


class NoBribePlayer:
    def __init__(self, bet_pct: float = 0.5) -> None:
        self.bet_pct = float(bet_pct)

    def act(self, env, obs, i: int, phase: int):
        if phase == 0:
            return {
                "bribe_pct": np.array([0.0], dtype=np.float32),
                "door": 0,
                "bet_pct": np.array([0.0], dtype=np.float32),
            }
        return {
            "bribe_pct": np.array([0.0], dtype=np.float32),
            "door": int(env._round_private_signals[i]),
            "bet_pct": np.array([self.bet_pct], dtype=np.float32),
        }
