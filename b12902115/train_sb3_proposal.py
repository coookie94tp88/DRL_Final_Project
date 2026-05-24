#!/usr/bin/env python3
"""Co-train Players (SB3 PPO) + Host (policy gradient) on ProposalVoteEnv.

One PPO env step = one game round: host proposes (PG), then players vote (PPO).
Player obs at decision time is pre-vote (proposal hidden); action is accept/reject per player.

Usage:
  python train_sb3_proposal.py --total-rounds 5000 --save-every 500
  python train_sb3_proposal.py --resume-player checkpoints_sb3_proposal/player_model_1000.zip
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

from proposal_vote_env import Phase, ProposalVoteConfig, ProposalVoteEnv, Vote


class PlayerExtractor(BaseFeaturesExtractor):
    """Flatten player ``current`` + ``history`` (same pattern as train_sb3.py)."""

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


class ProposalHostPolicy(nn.Module):
    """Host proposes A/B from full host dict observation."""

    def __init__(self, obs_space: gym.spaces.Dict):
        super().__init__()
        in_dim = sum(int(np.prod(obs_space[k].shape)) for k in obs_space.spaces)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, obs: dict) -> torch.Tensor:
        parts = [obs[k].flatten() for k in ("current", "players", "history", "vote_history")]
        return self.net(torch.cat(parts, dim=0))


def host_obs_to_tensors(host_obs: dict) -> dict:
    return {k: torch.as_tensor(host_obs[k], dtype=torch.float32) for k in host_obs}


def sample_host_proposal(
    host_policy: ProposalHostPolicy, host_obs: dict
) -> tuple[int, torch.Tensor]:
    logits = host_policy(host_obs)
    dist = torch.distributions.Categorical(logits=logits)
    proposal = dist.sample()
    return int(proposal.item()), dist.log_prob(proposal)


class ProposalPlayerEnvWrapper(gym.Env):
    """Single SB3 step per round: host propose → player votes → settlement.

    Player receives obs at round start (``Phase.PROPOSE``); proposal is hidden.
    Action: shape ``(num_players,)`` in [0, 1], rounded to reject/accept votes.
    """

    def __init__(self, base_env: ProposalVoteEnv, host_policy: ProposalHostPolicy):
        super().__init__()
        self.base = base_env
        self.host_policy = host_policy
        self.num_players = base_env.cfg.num_players
        self.observation_space = base_env.player_observation_space
        self.action_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_players,),
            dtype=np.float32,
        )
        self.last_host_logprob: torch.Tensor | None = None

    def reset(self, *, seed=None, options=None):
        obs, info = self.base.reset(seed=seed, options=options)
        self.last_host_logprob = None
        return obs["players"], info

    def step(self, action):
        if self.base.phase != Phase.PROPOSE:
            raise RuntimeError(f"Expected Phase.PROPOSE, got {self.base.phase.name}")

        obs_dict = self.base._get_observations()
        host_obs = host_obs_to_tensors(obs_dict["host"])
        proposal, logprob = sample_host_proposal(self.host_policy, host_obs)
        self.last_host_logprob = logprob

        self.base.step_propose(proposal)
        votes = np.rint(np.clip(np.asarray(action, dtype=np.float32), 0.0, 1.0)).astype(
            np.int32
        )
        obs, rewards, terminated, truncated, info = self.base.step_vote(votes)
        reward_scalar = float(np.mean(rewards["players"]))
        return obs["players"], reward_scalar, terminated, truncated, info


def _unwrap_player_wrapper(vec_env) -> ProposalPlayerEnvWrapper:
    env = vec_env.envs[0]
    while hasattr(env, "env"):
        env = env.env
    if not isinstance(env, ProposalPlayerEnvWrapper):
        raise TypeError(f"Expected ProposalPlayerEnvWrapper, got {type(env)}")
    return env


class HostUpdateCallback(BaseCallback):
    """REINFORCE host update after each completed round."""

    def __init__(
        self,
        host_optimizer: optim.Optimizer,
        player_policy: PPO,
        host_policy: ProposalHostPolicy,
        ckpt_dir: str,
        save_every: int,
        log_every: int = 50,
    ):
        super().__init__()
        self.host_optimizer = host_optimizer
        self.player_policy = player_policy
        self.host_policy = host_policy
        self.ckpt_dir = ckpt_dir
        self.save_every = save_every
        self.log_every = log_every
        self.round_count = 0
        self._prop_a = 0
        self._prop_b = 0
        self._accept = 0

    def _on_step(self) -> bool:
        wrapper = _unwrap_player_wrapper(self.training_env)
        if wrapper.last_host_logprob is None:
            return True

        host_reward = float(wrapper.base.hist_host_reward[-1])
        host_loss = -wrapper.last_host_logprob * torch.tensor(host_reward, dtype=torch.float32)
        self.host_optimizer.zero_grad()
        host_loss.backward()
        self.host_optimizer.step()
        wrapper.last_host_logprob = None

        info_last = wrapper.base.hist_proposal[-1]
        if info_last == 0:
            self._prop_a += 1
        else:
            self._prop_b += 1
        if wrapper.base.hist_passed[-1] > 0.5:
            self._accept += 1

        self.round_count += 1
        if self.round_count % self.log_every == 0:
            n = self.round_count
            print(
                f"[round {n}] host_r={host_reward:+.1f} "
                f"host_cum={wrapper.base.host_cumulative_reward:+.1f} "
                f"prop_A={100*self._prop_a/n:.0f}% accept={100*self._accept/n:.0f}%"
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
    num_players: int,
    max_rounds: int,
    history_window: int,
    seed: int,
    player_resume_path: str | None,
    host_resume_path: str | None,
    device: str,
    host_lr: float,
    player_lr: float,
) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    cfg = ProposalVoteConfig(
        num_players=num_players,
        max_rounds=max_rounds,
        history_window=history_window,
    )
    base_env = ProposalVoteEnv(config=cfg, seed=seed)
    host_policy = ProposalHostPolicy(base_env.host_observation_space)
    host_policy.train()
    host_optimizer = optim.Adam(host_policy.parameters(), lr=host_lr)

    if host_resume_path and os.path.isfile(host_resume_path):
        host_policy.load_state_dict(
            torch.load(host_resume_path, map_location="cpu", weights_only=True)
        )
        print(f"Loaded host: {host_resume_path}")

    player_env = ProposalPlayerEnvWrapper(base_env, host_policy)
    policy_kwargs = dict(
        features_extractor_class=PlayerExtractor,
        features_extractor_kwargs={},
    )

    n_steps = min(128, max(16, max_rounds * 2))
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
        player_policy = PPO(
            "MultiInputPolicy",
            player_env,
            policy_kwargs=policy_kwargs,
            n_steps=n_steps,
            batch_size=n_steps,
            n_epochs=4,
            learning_rate=player_lr,
            gamma=0.99,
            verbose=1,
            device=device,
        )

    base_env.reset(seed=seed)
    host_cb = HostUpdateCallback(
        host_optimizer,
        player_policy,
        host_policy,
        ckpt_dir,
        save_every,
    )
    print(
        f"Training {total_rounds} rounds on ProposalVoteEnv "
        f"(N={num_players}, n={cfg.n}, max_rounds/episode={max_rounds})"
    )
    player_policy.learn(
        total_timesteps=total_rounds,
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
    parser = argparse.ArgumentParser(
        description="SB3 PPO (players) + PG (host) on ProposalVoteEnv"
    )
    parser.add_argument("--total-rounds", type=int, default=100_000)
    parser.add_argument("--save-every", type=int, default=10_000)
    parser.add_argument("--ckpt-dir", default="./checkpoints_sb3_proposal")
    parser.add_argument("--num-players", type=int, default=10)
    parser.add_argument("--max-rounds", type=int, default=50)
    parser.add_argument("--history-window", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--host-lr", type=float, default=1e-4)
    parser.add_argument("--player-lr", type=float, default=3e-4)
    parser.add_argument("--resume-player", default=None)
    parser.add_argument("--resume-host", default=None)
    args = parser.parse_args()

    train(
        total_rounds=args.total_rounds,
        save_every=args.save_every,
        ckpt_dir=args.ckpt_dir,
        num_players=args.num_players,
        max_rounds=args.max_rounds,
        history_window=args.history_window,
        seed=args.seed,
        player_resume_path=args.resume_player,
        host_resume_path=args.resume_host,
        device=args.device,
        host_lr=args.host_lr,
        player_lr=args.player_lr,
    )


if __name__ == "__main__":
    main()
