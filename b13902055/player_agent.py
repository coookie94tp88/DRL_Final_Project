import numpy as np
import torch
from stable_baselines3 import PPO


class TrainedPlayerAgent:
    def __init__(self, model_path: str, num_players: int, num_doors: int, device: str = "cpu"):
        self.model = PPO.load(model_path, device=device)
        self.num_players = num_players
        self.num_doors = num_doors

    def _predict_flat_action(self, player_obs, deterministic: bool = True):
        action, _ = self.model.predict(player_obs, deterministic=deterministic)
        action = np.asarray(action, dtype=np.float32)
        return action

    def get_bribe_action(self, player_obs, deterministic: bool = True):
        action = self._predict_flat_action(player_obs, deterministic)
        n = self.num_players
        bribe = np.clip(action[:n], 0.0, 1.0)
        return bribe

    def get_bet_action(self, player_obs, deterministic: bool = True):
        action = self._predict_flat_action(player_obs, deterministic)
        n = self.num_players

        bet_frac = np.clip(action[n:2 * n], 0.0, 1.0)
        door_vals = np.clip(action[2 * n:], 0.0, self.num_doors - 1)
        doors = np.rint(door_vals).astype(np.int32)

        return doors, bet_frac