#!/usr/bin/env python3
"""Co-train Player (PPO / Stable-Baselines3) + Host (policy gradient) on b12902115 env.

Mirrors b13902055/train_both.py but uses belief actions (pub/priv/rnd) and b129 host obs
(including private_honesty_hist).

Usage (conda activate doorl):
  python train_sb3.py --total-rounds 500 --save-every 50
  python train_sb3.py --total-rounds 10000 --resume-player checkpoints_sb3/player_model_500.zip
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp2/b12902115/tmp/mpl")

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from env import OracleGambitConfig, OracleGambitEnv, Phase


# ── Player feature extractor (flatten current + history per SB3 batch) ──
class PlayerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=128)
        curr_dim = int(np.prod(observation_space["current"].shape))
        hist_dim = int(np.prod(observation_space["history"].shape))
        self.net = nn.Sequential(
            nn.Linear(curr_dim + hist_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

    def forward(self, obs: dict) -> torch.Tensor:
        current = obs["current"].flatten(1)
        history = obs["history"].flatten(1)
        return self.net(torch.cat([current, history], dim=1))


# ── Host policy (MLP over full host dict obs) ──
class HostPolicy(nn.Module):
    def __init__(self, obs_space: gym.spaces.Dict, action_space: gym.spaces.Dict):
        super().__init__()
        in_dim = sum(int(np.prod(obs_space[k].shape)) for k in obs_space.spaces)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.num_doors = action_space["public_signal"].n
        self.num_players = action_space["private_signals"].nvec.shape[0]
        self.public_head = nn.Linear(128, self.num_doors)
        self.private_head = nn.Linear(128, self.num_players * self.num_doors)

    def forward(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        parts = [obs[k].flatten() for k in ("current", "players", "history", "private_honesty_hist")]
        x = self.net(torch.cat(parts, dim=0))
        public_logits = self.public_head(x)
        private_logits = self.private_head(x).view(self.num_players, self.num_doors)
        return public_logits, private_logits


def sample_host_action(host_policy: HostPolicy, host_obs: dict) -> tuple[int, np.ndarray, torch.Tensor]:
    public_logits, private_logits = host_policy(host_obs)
    public_dist = torch.distributions.Categorical(logits=public_logits)
    private_dist = torch.distributions.Categorical(logits=private_logits)
    public_signal = public_dist.sample()
    private_signals = private_dist.sample()
    logprob = public_dist.log_prob(public_signal) + private_dist.log_prob(private_signals).sum()
    return int(public_signal.item()), private_signals.cpu().numpy().astype(np.int32), logprob


def host_obs_to_tensors(host_obs: dict) -> dict:
    return {k: torch.as_tensor(host_obs[k], dtype=torch.float32) for k in host_obs}


# ── Gym wrapper: two SB3 steps per game round (matches eval_sb3 timing) ──
class PlayerEnvWrapper(gym.Env):
    """Two PPO steps per round (same timing as eval / train_both SAC):

    Step A (base phase BRIBE): action uses ``action[:N]`` only — bribe fractions.
        Then host SIGNAL. Returns BET-phase obs, reward=0.

    Step B (base phase BET): action uses ``action[N:2N]`` bet, ``action[2N:3N]`` belief.
        Beliefs chosen **after** public/private signals are visible in obs.
        Returns next-round BRIBE obs (or terminal), env mean player reward.

    Full action vector shape ``3×N`` is kept so SB3 ``Box`` size stays fixed; unused
    slots are ignored on each sub-step.
    """

    def __init__(self, base_env: OracleGambitEnv, host_policy: HostPolicy):
        super().__init__()
        self.base = base_env
        self.host_policy = host_policy
        self.num_players = base_env.cfg.num_players
        self.observation_space = base_env.player_observation_space
        n = self.num_players
        low = np.array([0.0] * (2 * n) + [0.0] * n, dtype=np.float32)
        high = np.array([1.0] * (2 * n) + [2.0] * n, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.last_host_logprob: torch.Tensor | None = None
        self._pending_host_logprob: torch.Tensor | None = None

    def reset(self, *, seed=None, options=None):
        obs, info = self.base.reset(seed=seed, options=options)
        self.last_host_logprob = None
        self._pending_host_logprob = None
        return obs["players"], info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        n = self.num_players

        if self.base.phase == Phase.BRIBE:
            bribe = np.clip(action[:n], 0.0, 1.0)
            obs, _, _, _, info = self.base.step({"player_bribe_fractions": bribe})
            if self.base.phase != Phase.SIGNAL:
                raise RuntimeError("Expected Phase.SIGNAL after bribe")

            host_obs = host_obs_to_tensors(obs["host"])
            pub, priv, logprob = sample_host_action(self.host_policy, host_obs)
            self._pending_host_logprob = logprob
            obs, _, _, _, info = self.base.step({"public_signal": pub, "private_signals": priv})
            if self.base.phase != Phase.BET:
                raise RuntimeError("Expected Phase.BET after signal")
            return obs["players"], 0.0, False, False, info

        if self.base.phase == Phase.BET:
            bet_frac = np.clip(action[n : 2 * n], 0.0, 1.0)
            belief_vals = np.clip(action[2 * n :], 0.0, 2.0)
            beliefs = np.rint(belief_vals).astype(np.int32)
            obs, rewards, terminated, truncated, info = self.base.step(
                {"player_beliefs": beliefs, "bet_fractions": bet_frac}
            )
            self.last_host_logprob = self._pending_host_logprob
            self._pending_host_logprob = None
            reward_scalar = float(np.mean(rewards["players"]))
            return obs["players"], reward_scalar, terminated, truncated, info

        raise RuntimeError(f"Unexpected phase in wrapper step: {self.base.phase.name}")


def _unwrap_player_wrapper(vec_env) -> PlayerEnvWrapper:
    env = vec_env.envs[0]
    while hasattr(env, "env"):
        env = env.env
    if not isinstance(env, PlayerEnvWrapper):
        raise TypeError(f"Expected PlayerEnvWrapper, got {type(env)}")
    return env


class HostUpdateCallback(BaseCallback):
    """Policy-gradient host update after each wrapper step (one game round)."""

    def __init__(
        self,
        host_optimizer: optim.Optimizer,
        player_policy: PPO,
        host_policy: HostPolicy,
        ckpt_dir: str,
        save_every: int,
    ):
        super().__init__()
        self.host_optimizer = host_optimizer
        self.player_policy = player_policy
        self.host_policy = host_policy
        self.ckpt_dir = ckpt_dir
        self.save_every = save_every
        self.round_count = 0

    def _on_step(self) -> bool:
        wrapper = _unwrap_player_wrapper(self.training_env)
        if wrapper.last_host_logprob is None:
            return True
        host_reward = float(wrapper.base.hist_host_profit[-1])
        host_loss = -wrapper.last_host_logprob * torch.tensor(host_reward, dtype=torch.float32)
        self.host_optimizer.zero_grad()
        host_loss.backward()
        self.host_optimizer.step()
        wrapper.last_host_logprob = None
        self.round_count += 1

        if self.round_count % 50 == 0:
            print(
                f"[round {self.round_count}] host_step={host_reward:+.2f} "
                f"host_cum={wrapper.base.host_cumulative_profit:+.1f}"
            )
        if self.save_every > 0 and self.round_count % self.save_every == 0:
            p_path = os.path.join(self.ckpt_dir, f"player_model_{self.round_count}.zip")
            h_path = os.path.join(self.ckpt_dir, f"host_model_{self.round_count}.pt")
            self.player_policy.save(p_path)
            torch.save(self.host_policy.state_dict(), h_path)
            print(f"[Saved @ round {self.round_count}] {p_path}")
        return True


def train(
    *,
    total_rounds: int,
    save_every: int,
    ckpt_dir: str,
    num_doors: int,
    max_rounds: int,
    seed: int,
    player_resume_path: str | None,
    host_resume_path: str | None,
    device: str,
) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    cfg = OracleGambitConfig(
        num_players=10,
        num_doors=num_doors,
        max_rounds=max_rounds,
        initial_balance=1000.0,
    )
    base_env = OracleGambitEnv(config=cfg, seed=seed)
    host_policy = HostPolicy(base_env.host_observation_space, base_env.host_action_space)
    host_policy.train()
    host_optimizer = optim.Adam(host_policy.parameters(), lr=1e-4)

    if host_resume_path and os.path.isfile(host_resume_path):
        host_policy.load_state_dict(torch.load(host_resume_path, map_location="cpu", weights_only=True))
        print(f"Loaded host: {host_resume_path}")

    player_env = PlayerEnvWrapper(base_env, host_policy)

    policy_kwargs = dict(features_extractor_class=PlayerExtractor, features_extractor_kwargs={})
    if player_resume_path and os.path.isfile(player_resume_path):
        print(f"Loading player PPO: {player_resume_path}")
        player_policy = PPO.load(
            player_resume_path,
            env=player_env,
            device=device,
            custom_objects={"policy_kwargs": policy_kwargs},
        )
    else:
        print("Training player PPO from scratch")
        n_steps = min(64, max(8, max_rounds * 4))
        player_policy = PPO(
            "MultiInputPolicy",
            player_env,
            policy_kwargs=policy_kwargs,
            n_steps=n_steps,
            batch_size=n_steps,
            n_epochs=4,
            learning_rate=3e-4,
            gamma=0.99,
            verbose=1,
            device=device,
        )

    base_env.reset(seed=seed)
    host_cb = HostUpdateCallback(
        host_optimizer, player_policy, host_policy, ckpt_dir, save_every
    )
    # 2 SB3 steps per game round (bribe sub-step + bet/belief sub-step)
    env_steps = total_rounds * 2
    print(f"Training {total_rounds} game rounds ({env_steps} PPO env steps)")
    player_policy.learn(
        total_timesteps=env_steps,
        callback=host_cb,
        reset_num_timesteps=player_resume_path is None,
    )

    n = host_cb.round_count
    p_final = os.path.join(ckpt_dir, f"player_model_{n}.zip")
    h_final = os.path.join(ckpt_dir, f"host_model_{n}.pt")
    player_policy.save(p_final)
    torch.save(host_policy.state_dict(), h_final)
    print(f"Done. rounds={n} player={p_final} host={h_final}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SB3 PPO + host PG on b12902115 OracleGambit")
    parser.add_argument("--total-rounds", type=int, default=100000)
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--ckpt-dir", default="./checkpoints_sb3")
    parser.add_argument("--num-doors", type=int, default=4)
    parser.add_argument("--max-rounds", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume-player", default=None)
    parser.add_argument("--resume-host", default=None)
    args = parser.parse_args()

    train(
        total_rounds=args.total_rounds,
        save_every=args.save_every,
        ckpt_dir=args.ckpt_dir,
        num_doors=args.num_doors,
        max_rounds=args.max_rounds,
        seed=args.seed,
        player_resume_path=args.resume_player,
        host_resume_path=args.resume_host,
        device=args.device,
    )


if __name__ == "__main__":
    main()
