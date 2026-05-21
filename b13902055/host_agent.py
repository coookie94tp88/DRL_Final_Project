import numpy as np
import torch
import torch.nn as nn
from env import OracleGambitConfig


class HostPolicy(nn.Module):
    def __init__(self, config: OracleGambitConfig):
        super().__init__()
        curr_dim = config.current_host_dim
        players_dim = config.num_players * config.host_player_state_dim
        hist_dim = config.history_window * config.hist_host_dim

        self.net = nn.Sequential(
            nn.Linear(curr_dim + players_dim + hist_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.num_doors = config.num_doors
        self.num_players = config.num_players

        self.public_head = nn.Linear(128, self.num_doors)
        self.private_head = nn.Linear(128, self.num_players * self.num_doors)

    def forward(self, obs):
        cur = obs["current"].flatten()
        players = obs["players"].flatten()
        hist = obs["history"].flatten()
        x = torch.cat([cur, players, hist], dim=0)
        x = self.net(x)

        public_logits = self.public_head(x)
        private_logits = self.private_head(x).view(self.num_players, self.num_doors)

        return public_logits, private_logits


class TrainedHostAgent:
    def __init__(self, model_path: str, config: OracleGambitConfig, device: str = "cpu"):
        self.device = torch.device(device)
        self.policy = HostPolicy(config).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        self.policy.load_state_dict(state)
        self.policy.eval()

    def get_action(self, env, deterministic: bool = True):
        host_obs_np = env._get_host_obs()
        host_obs = {
            "current": torch.tensor(host_obs_np["current"], dtype=torch.float32, device=self.device),
            "players": torch.tensor(host_obs_np["players"], dtype=torch.float32, device=self.device),
            "history": torch.tensor(host_obs_np["history"], dtype=torch.float32, device=self.device),
        }

        with torch.no_grad():
            public_logits, private_logits = self.policy(host_obs)

            if deterministic:
                public_signal = torch.argmax(public_logits).item()
                private_signals = torch.argmax(private_logits, dim=1).cpu().numpy()
            else:
                public_dist = torch.distributions.Categorical(logits=public_logits)
                private_dist = torch.distributions.Categorical(logits=private_logits)

                public_signal = public_dist.sample().item()
                private_signals = private_dist.sample().cpu().numpy()

        return public_signal, private_signals