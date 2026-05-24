import numpy as np
import torch

from train_both_CTDE import PlayerActor, flatten_local_player_obs


class TrainedCTDEPlayerAgent:
    def __init__(
        self,
        checkpoint_path: str,
        num_players: int,
        num_doors: int,
        history_window: int = 50,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.num_players = num_players
        self.num_doors = num_doors
        self.history_window = history_window

        local_obs_dim = 5 + history_window * (7 + num_doors)
        self.actor = PlayerActor(local_obs_dim=local_obs_dim, num_doors=num_doors).to(self.device)

        state = torch.load(checkpoint_path, map_location=self.device)
        self.actor.load_state_dict(state["player_actor"])
        self.actor.eval()

    def _build_local_batch(self, player_obs: dict[str, np.ndarray]) -> torch.Tensor:
        local_obs = [flatten_local_player_obs(player_obs, i) for i in range(self.num_players)]
        return torch.as_tensor(np.stack(local_obs, axis=0), dtype=torch.float32, device=self.device)

    def get_bribe_action(self, player_obs: dict[str, np.ndarray], deterministic: bool = True) -> np.ndarray:
        local_batch = self._build_local_batch(player_obs)
        with torch.no_grad():
            if deterministic:
                feat = self.actor.shared(local_batch)
                bribe = torch.sigmoid(self.actor.bribe_mu(feat)).squeeze(-1)
            else:
                bribe, _ = self.actor.sample_bribe(local_batch)
                bribe = bribe.squeeze(-1)
        return np.clip(bribe.cpu().numpy(), 0.0, 1.0).astype(np.float32)

    def get_bet_action(
        self,
        player_obs: dict[str, np.ndarray],
        deterministic: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        local_batch = self._build_local_batch(player_obs)
        with torch.no_grad():
            if deterministic:
                feat = self.actor.shared(local_batch)
                door_logits = self.actor.door_logits(feat)
                doors = torch.argmax(door_logits, dim=-1)
                bet_frac = torch.sigmoid(self.actor.bet_mu(feat)).squeeze(-1)
            else:
                doors, bet_frac, _ = self.actor.sample_bet(local_batch)
                bet_frac = bet_frac.squeeze(-1)

        doors_np = np.clip(doors.cpu().numpy(), 0, self.num_doors - 1).astype(np.int32)
        bet_np = np.clip(bet_frac.cpu().numpy(), 0.0, 1.0).astype(np.float32)
        return doors_np, bet_np
