import os
os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from env import OracleGambitEnv, OracleGambitConfig, Phase


# -----------------------------
# Player feature extractor
# -----------------------------
class PlayerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        curr_shape = observation_space["current"].shape
        hist_shape = observation_space["history"].shape
        super().__init__(observation_space, features_dim=128)

        curr_dim = int(np.prod(curr_shape))
        hist_dim = int(np.prod(hist_shape))

        self.net = nn.Sequential(
            nn.Linear(curr_dim + hist_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

    def forward(self, obs):
        current = obs["current"].flatten(1)
        history = obs["history"].flatten(1)
        x = torch.cat([current, history], dim=1)
        return self.net(x)


# -----------------------------
# Host policy network
# -----------------------------
class HostPolicy(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()
        curr_dim = int(np.prod(obs_space["current"].shape))
        players_dim = int(np.prod(obs_space["players"].shape))
        hist_dim = int(np.prod(obs_space["history"].shape))

        self.net = nn.Sequential(
            nn.Linear(curr_dim + players_dim + hist_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.num_doors = action_space["public_signal"].n
        self.num_players = action_space["private_signals"].nvec.shape[0]

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


# -----------------------------
# Host action sampler (returns log-prob)
# -----------------------------
def sample_host_action(host_policy, host_obs):
    public_logits, private_logits = host_policy(host_obs)

    public_dist = torch.distributions.Categorical(logits=public_logits)
    private_dist = torch.distributions.Categorical(logits=private_logits)

    public_signal = public_dist.sample()
    private_signals = private_dist.sample()

    logprob = public_dist.log_prob(public_signal) + private_dist.log_prob(private_signals).sum()
    return public_signal.item(), private_signals.cpu().numpy(), logprob


# -----------------------------
# Player env wrapper (BRIBE+BET in one step)
# Action space is Box to satisfy SB3
# -----------------------------
class PlayerEnvWrapper(gym.Env):
    def __init__(self, base_env, host_policy):
        super().__init__()
        self.base = base_env
        self.host_policy = host_policy

        self.num_players = base_env.cfg.num_players
        self.num_doors = base_env.cfg.num_doors

        self.observation_space = base_env.player_observation_space

        # Action layout (flat Box):
        # [bribe_fractions (N), bet_fractions (N), door_values (N)]
        low = np.array([0.0] * (3 * self.num_players), dtype=np.float32)
        high = np.array(
            [1.0] * (2 * self.num_players) + [self.num_doors - 1] * self.num_players,
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        self.last_host_logprob = None

    def reset(self, *, seed=None, options=None):
        obs, info = self.base.reset(seed=seed, options=options)
        return obs["players"], info

    def step(self, action):
        if self.base.phase != Phase.BRIBE:
            raise RuntimeError("Expected Phase.BRIBE at start of step()")

        action = np.asarray(action, dtype=np.float32)
        n = self.num_players

        bribe = np.clip(action[:n], 0.0, 1.0)
        bet_frac = np.clip(action[n:2 * n], 0.0, 1.0)
        door_vals = np.clip(action[2 * n:], 0.0, self.num_doors - 1)
        doors = np.rint(door_vals).astype(np.int32)

        # BRIBE
        obs, _, _, _, _ = self.base.step_bribe(bribe)

        # SIGNAL (host acts)
        if self.base.phase != Phase.SIGNAL:
            raise RuntimeError("Expected Phase.SIGNAL after bribe")

        host_obs = {
            "current": torch.tensor(obs["host"]["current"], dtype=torch.float32),
            "players": torch.tensor(obs["host"]["players"], dtype=torch.float32),
            "history": torch.tensor(obs["host"]["history"], dtype=torch.float32),
        }
        public_signal, private_signals, logprob = sample_host_action(self.host_policy, host_obs)
        self.last_host_logprob = logprob

        obs, _, _, _, _ = self.base.step_signal(public_signal, private_signals)

        # BET
        if self.base.phase != Phase.BET:
            raise RuntimeError("Expected Phase.BET after signal")

        obs, rewards, terminated, truncated, info = self.base.step_bet(
            doors, bet_frac
        )

        reward_scalar = float(np.mean(rewards["players"]))
        return obs["players"], reward_scalar, terminated, truncated, info


# -----------------------------
# Training loop (same loop)
# -----------------------------
def train(
    total_rounds=10000,
    save_every=50,
    ckpt_dir="./checkpoints",
    player_resume_path="./checkpoints/player_model_1450.zip",
    host_resume_path=None,
):
    import os

    os.makedirs(ckpt_dir, exist_ok=True)
    cfg = OracleGambitConfig(num_doors=3, num_players=10, max_rounds=20)
    base_env = OracleGambitEnv(cfg)

    host_policy = HostPolicy(base_env.host_observation_space, base_env.host_action_space)
    host_policy.train()
    host_optimizer = optim.Adam(host_policy.parameters(), lr=1e-4)
    player_env = PlayerEnvWrapper(base_env, host_policy)

    # --------- MODEL LOADING FROM CUSTOM PATH ----------
    if player_resume_path is not None and os.path.isfile(player_resume_path):
        print(f"Loading player policy from {player_resume_path}")
        player_policy = PPO.load(player_resume_path, env=player_env, device="auto")
    else:
        print(f"Training player policy from scratch.")
        player_policy = PPO(
            "MultiInputPolicy",
            player_env,
            policy_kwargs=dict(features_extractor_class=PlayerExtractor),
            verbose=1,
        )

    if host_resume_path is not None and os.path.isfile(host_resume_path):
        print(f"Loading host model from {host_resume_path}")
        host_policy.load_state_dict(torch.load(host_resume_path, map_location="cpu"))
        print("Loaded host model.")
    else:
        print(f"Training host model from scratch.")

    base_env.reset()
    round_count = 0  # force start from 0

    while round_count < total_rounds:
        player_policy.learn(total_timesteps=1, reset_num_timesteps=False)

        if base_env.phase == Phase.BRIBE and player_env.last_host_logprob is not None:
            host_reward = base_env.hist_host_profit[-1]
            host_loss = -player_env.last_host_logprob * torch.tensor(host_reward, dtype=torch.float32)

            host_optimizer.zero_grad()
            host_loss.backward()
            host_optimizer.step()

            round_count += 1

        if round_count % save_every == 0 and round_count > 0:
            player_save_path = f"{ckpt_dir}/player_model_{round_count}.zip"
            host_save_path = f"{ckpt_dir}/host_model_{round_count}.pt"
            player_policy.save(player_save_path)
            torch.save(host_policy.state_dict(), host_save_path)
            print(f"[Saved @ round {round_count}]")

    player_save_path = f"{ckpt_dir}/player_model_{round_count}.zip"
    host_save_path = f"{ckpt_dir}/host_model_{round_count}.pt"
    player_policy.save(player_save_path)
    torch.save(host_policy.state_dict(), host_save_path)
    print(f"Training complete. Final checkpoint saved at round {round_count}")


if __name__ == "__main__":
    # Usage: optionally pass the checkpoint paths manually here
    player_ckpt = input("Enter player model path to resume (or leave blank): ").strip() or None
    host_ckpt = input("Enter host model path to resume (or leave blank): ").strip() or None
    train(
        total_rounds=10000,
        save_every=50,
        ckpt_dir="./checkpoints",
        player_resume_path=player_ckpt,
        host_resume_path=host_ckpt,
    )