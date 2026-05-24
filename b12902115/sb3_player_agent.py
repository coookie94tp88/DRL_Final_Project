"""Load SB3 PPO player trained with train_sb3.py (belief actions)."""

from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO

from train_sb3 import PlayerExtractor


class SB3PlayerAgent:
    def __init__(self, model_path: str, num_players: int, device: str = "cpu"):
        self.num_players = num_players
        self.model = PPO.load(
            model_path,
            device=device,
            custom_objects={
                "policy_kwargs": {
                    "features_extractor_class": PlayerExtractor,
                    "features_extractor_kwargs": {},
                }
            },
        )

    def _predict(self, player_obs: dict, deterministic: bool = True) -> np.ndarray:
        action, _ = self.model.predict(player_obs, deterministic=deterministic)
        return np.asarray(action, dtype=np.float32)

    def get_bribe_action(self, player_obs: dict, deterministic: bool = True) -> np.ndarray:
        action = self._predict(player_obs, deterministic)
        n = self.num_players
        return np.clip(action[:n], 0.0, 1.0)

    def get_bet_action(
        self, player_obs: dict, deterministic: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        action = self._predict(player_obs, deterministic)
        n = self.num_players
        bet_frac = np.clip(action[n : 2 * n], 0.0, 1.0)
        beliefs = np.rint(np.clip(action[2 * n :], 0.0, 2.0)).astype(np.int32)
        return beliefs, bet_frac
