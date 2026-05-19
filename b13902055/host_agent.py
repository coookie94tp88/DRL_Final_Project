import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env import OracleGambitConfig


class HostRDQN(nn.Module):
    def __init__(self, num_players=10, num_doors=4, hist_dim=10, d_model=128):
        super().__init__()
        self.num_players = num_players
        self.num_doors = num_doors

        self.lstm = nn.LSTM(input_size=hist_dim, hidden_size=d_model, batch_first=True)
        self.fc_curr = nn.Linear(num_doors + num_players, d_model)
        self.fc_fusion = nn.Sequential(nn.Linear(d_model * 2, 256), nn.ReLU())
        self.q_pub = nn.Linear(256, num_doors)
        self.q_privs = nn.ModuleList([nn.Linear(256, num_doors) for _ in range(num_players)])

    def forward(self, curr, hist):
        _, (h_n, _) = self.lstm(hist)
        hist_feat = h_n[-1]
        curr_feat = F.relu(self.fc_curr(curr))
        merged = self.fc_fusion(torch.cat([hist_feat, curr_feat], dim=-1))
        q_pub_vals = self.q_pub(merged)
        q_priv_vals = [head(merged) for head in self.q_privs]
        return q_pub_vals, q_priv_vals


class TrainedHostAgent:
    def __init__(self, checkpoint_path: str, config: OracleGambitConfig, hist_dim: int | None = None, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = config

        ckpt_num_doors = checkpoint.get("num_doors")
        if ckpt_num_doors is None:
            print("⚠️ host checkpoint has no 'num_doors' metadata; falling back to config.num_doors.")
            self.num_doors = int(config.num_doors)
        else:
            self.num_doors = int(ckpt_num_doors)
        effective_hist_dim = int(hist_dim if hist_dim is not None else config.hist_host_dim)

        self.rdqn = HostRDQN(
            num_players=config.num_players,
            num_doors=self.num_doors,
            hist_dim=effective_hist_dim,
        ).to(self.device)

        state_dict = checkpoint["host_rdqn"] if "host_rdqn" in checkpoint else checkpoint
        self.rdqn.load_state_dict(state_dict)
        self.rdqn.eval()

        print(
            f"✅ HostAgent loaded (Episode {checkpoint.get('episode', 'Unknown')}, Doors={self.num_doors}, Device={self.device})"
        )

    def _process_obs(self, env):
        c = self.config
        if c.num_doors != self.num_doors:
            raise ValueError(
                f"Env doors ({c.num_doors}) and host model doors ({self.num_doors}) mismatch."
            )

        winning_door = env.current_winning_door
        bribes = env.current_bribes

        win_door_oh = np.zeros(self.num_doors, dtype=np.float32)
        if winning_door >= 0:
            win_door_oh[winning_door] = 1.0

        curr_processed = np.concatenate([win_door_oh, bribes])
        curr_tensor = torch.as_tensor(curr_processed, dtype=torch.float32, device=self.device).unsqueeze(0)

        hist = env._get_observations()["host"]["history"]
        hist_tensor = torch.as_tensor(hist, dtype=torch.float32, device=self.device).unsqueeze(0)
        return curr_tensor, hist_tensor

    def get_action(self, env):
        curr_tensor, hist_tensor = self._process_obs(env)
        with torch.no_grad():
            q_pub, q_privs = self.rdqn(curr_tensor, hist_tensor)
            pub_act = q_pub.argmax(dim=-1).item()
            priv_acts = [q.argmax(dim=-1).item() for q in q_privs]
        return pub_act, np.array(priv_acts, dtype=np.int32)
