import numpy as np
import torch

from train_both_CTDE import HostPolicy, flatten_host_obs


class TrainedCTDEHostAgent:
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

        host_obs_dim = (2 + num_doors) + (num_players * 3) + (history_window * (2 + 2 * num_doors))
        self.policy = HostPolicy(
            host_obs_dim=host_obs_dim,
            num_players=num_players,
            num_doors=num_doors,
        ).to(self.device)

        state = torch.load(checkpoint_path, map_location=self.device)
        self.policy.load_state_dict(state["host_policy"])
        self.policy.eval()

    def get_action(self, host_obs: dict[str, np.ndarray], deterministic: bool = True) -> tuple[int, np.ndarray]:
        host_obs_tensor = torch.as_tensor(flatten_host_obs(host_obs), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            feat = self.policy.net(host_obs_tensor)
            public_logits = self.policy.public_head(feat)
            private_logits = self.policy.private_head(feat).view(self.num_players, self.num_doors)

            if deterministic:
                public_signal = int(torch.argmax(public_logits).item())
                private_signals = torch.argmax(private_logits, dim=1).cpu().numpy().astype(np.int32)
            else:
                public_dist = torch.distributions.Categorical(logits=public_logits)
                private_dist = torch.distributions.Categorical(logits=private_logits)
                public_signal = int(public_dist.sample().item())
                private_signals = private_dist.sample().cpu().numpy().astype(np.int32)

        public_signal = int(np.clip(public_signal, 0, self.num_doors - 1))
        private_signals = np.clip(private_signals, 0, self.num_doors - 1).astype(np.int32)
        return public_signal, private_signals
