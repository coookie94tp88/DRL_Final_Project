import argparse
import os
import random
import csv
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal

from env import OracleGambitConfig, OracleGambitEnv, Phase


METRIC_COLUMNS = [
    "step",
    "avg_bet",
    "avg_bribe",
    "host_final_reward",
    "player_final_reward",
    "host_true_private_signal_rate",
    "player_follow_private_signal_rate",
]


def _save_metrics_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _print_metric_row(prefix, row):
    print(
        f"{prefix} | step={int(row['step'])} | avg_bet={row['avg_bet']:.3f} | "
        f"avg_bribe={row['avg_bribe']:.3f} | host_final_reward={row['host_final_reward']:+.3f} | "
        f"player_final_reward={row['player_final_reward']:+.3f} | "
        f"host_true_private_signal_rate={row['host_true_private_signal_rate']:.3f} | "
        f"player_follow_private_signal_rate={row['player_follow_private_signal_rate']:.3f}"
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flatten_local_player_obs(players_obs: dict[str, np.ndarray], player_idx: int) -> np.ndarray:
    current = players_obs["current"][player_idx]
    history = players_obs["history"][player_idx].reshape(-1)
    return np.concatenate([current, history], axis=0).astype(np.float32)


def flatten_global_obs(obs: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
    p_current = obs["players"]["current"].reshape(-1)
    p_history = obs["players"]["history"].reshape(-1)
    h_current = obs["host"]["current"].reshape(-1)
    h_players = obs["host"]["players"].reshape(-1)
    h_history = obs["host"]["history"].reshape(-1)
    return np.concatenate([p_current, p_history, h_current, h_players, h_history], axis=0).astype(np.float32)


class PlayerActor(nn.Module):
    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 1.0
    EPSILON = 1e-6

    def __init__(self, local_obs_dim: int, num_doors: int):
        super().__init__()
        self.num_doors = num_doors
        self.shared = nn.Sequential(
            nn.Linear(local_obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.bribe_mu = nn.Linear(128, 1)
        # One shared log-std per action head (broadcast across players/batch).
        self.bribe_log_std = nn.Parameter(torch.tensor(0.0))
        self.bet_mu = nn.Linear(128, 1)
        # One shared log-std per action head (broadcast across players/batch).
        self.bet_log_std = nn.Parameter(torch.tensor(0.0))
        self.door_logits = nn.Linear(128, num_doors)

    def _sample_fraction(self, mu: torch.Tensor, log_std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        std = log_std.exp()
        dist = Normal(mu, std)
        raw = dist.rsample()
        frac = torch.sigmoid(raw)
        log_prob = dist.log_prob(raw) - torch.log(frac * (1.0 - frac) + self.EPSILON)
        return frac, log_prob.sum(dim=-1)

    def sample_bribe(self, local_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.shared(local_obs)
        mu = self.bribe_mu(feat)
        log_std = self.bribe_log_std.expand_as(mu).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return self._sample_fraction(mu, log_std)

    def sample_bet(self, local_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.shared(local_obs)
        door_dist = Categorical(logits=self.door_logits(feat))
        door = door_dist.sample()
        door_log_prob = door_dist.log_prob(door)

        mu = self.bet_mu(feat)
        log_std = self.bet_log_std.expand_as(mu).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        bet_frac, bet_log_prob = self._sample_fraction(mu, log_std)
        return door, bet_frac, door_log_prob + bet_log_prob


class CentralCritic(nn.Module):
    def __init__(self, global_obs_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        return self.net(global_obs).squeeze(-1)


class HostPolicy(nn.Module):
    def __init__(self, host_obs_dim: int, num_players: int, num_doors: int):
        super().__init__()
        self.num_players = num_players
        self.num_doors = num_doors
        self.net = nn.Sequential(
            nn.Linear(host_obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.public_head = nn.Linear(128, num_doors)
        self.private_head = nn.Linear(128, num_players * num_doors)

    def sample_action(self, host_obs: torch.Tensor) -> tuple[int, np.ndarray, torch.Tensor]:
        """Return (public_signal, private_signals_np, log_prob).

        private_signals are returned as numpy because env.step_signal expects numpy/int arrays.
        """
        feat = self.net(host_obs)
        public_logits = self.public_head(feat)
        private_logits = self.private_head(feat).view(self.num_players, self.num_doors)

        public_dist = Categorical(logits=public_logits)
        private_dist = Categorical(logits=private_logits)

        public_signal = public_dist.sample()
        private_signals = private_dist.sample()
        log_prob = public_dist.log_prob(public_signal) + private_dist.log_prob(private_signals).sum()
        return int(public_signal.item()), private_signals.detach().cpu().numpy(), log_prob


def flatten_host_obs(obs: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [
            obs["current"].reshape(-1),
            obs["players"].reshape(-1),
            obs["history"].reshape(-1),
        ],
        axis=0,
    ).astype(np.float32)


@dataclass
class TrainConfig:
    episodes: int = 500
    save_every: int = 50
    gamma: float = 0.99
    lr_player: float = 3e-4
    lr_critic: float = 3e-4
    lr_host: float = 3e-4
    grad_clip_norm: float = 1.0
    num_players: int = 10
    num_doors: int = 4
    max_rounds: int = 20
    initial_balance: float = 1000.0
    history_window: int = 50
    seed: int = 42
    out_dir: str = "./checkpoints_ctde"


def discounted_returns(rewards: list[float], gamma: float, device: torch.device) -> torch.Tensor:
    out = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        out.append(running)
    out.reverse()
    return torch.as_tensor(out, dtype=torch.float32, device=device)


def train_both_ctde(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.out_dir, exist_ok=True)

    env_cfg = OracleGambitConfig(
        num_players=cfg.num_players,
        num_doors=cfg.num_doors,
        max_rounds=cfg.max_rounds,
        initial_balance=cfg.initial_balance,
        history_window=cfg.history_window,
    )
    env = OracleGambitEnv(config=env_cfg, seed=cfg.seed)
    obs, _ = env.reset()

    local_obs_dim = flatten_local_player_obs(obs["players"], 0).shape[0]
    global_obs_dim = flatten_global_obs(obs).shape[0]
    host_obs_dim = flatten_host_obs(obs["host"]).shape[0]

    player_actor = PlayerActor(local_obs_dim=local_obs_dim, num_doors=cfg.num_doors).to(device)
    player_critic = CentralCritic(global_obs_dim=global_obs_dim).to(device)
    host_policy = HostPolicy(host_obs_dim=host_obs_dim, num_players=cfg.num_players, num_doors=cfg.num_doors).to(device)

    opt_player = optim.Adam(player_actor.parameters(), lr=cfg.lr_player)
    opt_critic = optim.Adam(player_critic.parameters(), lr=cfg.lr_critic)
    opt_host = optim.Adam(host_policy.parameters(), lr=cfg.lr_host)
    metrics_path = os.path.join(cfg.out_dir, "training_metrics.csv")
    metrics_rows = []

    for ep in range(1, cfg.episodes + 1):
        obs, _ = env.reset()
        player_log_probs: list[torch.Tensor] = []
        player_values: list[torch.Tensor] = []
        player_rewards: list[float] = []
        host_log_probs: list[torch.Tensor] = []
        host_rewards: list[float] = []
        ep_avg_bet_sum = 0.0
        ep_avg_bribe_sum = 0.0
        ep_truth_rate_sum = 0.0
        ep_follow_rate_sum = 0.0
        ep_rounds = 0
        ep_truth_rate_rounds = 0

        while True:
            if env.phase != Phase.BRIBE:
                raise RuntimeError(
                    f"Expected BRIBE phase at loop start, but got {env.phase}. "
                    "This usually means a prior phase step did not complete correctly."
                )

            global_state = torch.as_tensor(flatten_global_obs(obs), dtype=torch.float32, device=device)
            player_values.append(player_critic(global_state))

            bribe_log_prob_sum = torch.zeros((), dtype=torch.float32, device=device)
            bribe_fracs = np.zeros(cfg.num_players, dtype=np.float32)
            for i in range(cfg.num_players):
                local_obs_np = flatten_local_player_obs(obs["players"], i)
                local_obs = torch.as_tensor(local_obs_np, dtype=torch.float32, device=device).unsqueeze(0)
                bribe_frac, logp = player_actor.sample_bribe(local_obs)
                bribe_fracs[i] = float(bribe_frac.squeeze(0).item())
                bribe_log_prob_sum = bribe_log_prob_sum + logp.squeeze(0)

            obs, _, _, _, _ = env.step({"player_bribe_fractions": bribe_fracs})

            if env.phase != Phase.SIGNAL:
                raise RuntimeError(
                    f"Expected SIGNAL phase after bribe, but got {env.phase}. "
                    "Check BRIBE actions and environment transitions."
                )
            host_obs_tensor = torch.as_tensor(flatten_host_obs(obs["host"]), dtype=torch.float32, device=device)
            pub, priv, host_logp = host_policy.sample_action(host_obs_tensor)
            host_log_probs.append(host_logp)
            obs, _, _, _, _ = env.step({"public_signal": pub, "private_signals": priv})

            if env.phase != Phase.BET:
                raise RuntimeError(
                    f"Expected BET phase after signal, but got {env.phase}. "
                    "Check host SIGNAL actions and environment transitions."
                )

            bet_log_prob_sum = torch.zeros((), dtype=torch.float32, device=device)
            doors = np.zeros(cfg.num_players, dtype=np.int32)
            bet_fracs = np.zeros(cfg.num_players, dtype=np.float32)
            for i in range(cfg.num_players):
                local_obs_np = flatten_local_player_obs(obs["players"], i)
                local_obs = torch.as_tensor(local_obs_np, dtype=torch.float32, device=device).unsqueeze(0)
                door, bet_frac, logp = player_actor.sample_bet(local_obs)
                doors[i] = int(door.squeeze(0).item())
                bet_fracs[i] = float(bet_frac.squeeze(0).item())
                bet_log_prob_sum = bet_log_prob_sum + logp.squeeze(0)

            obs, rewards, terminated, truncated, info = env.step(
                {"player_doors": doors, "bet_fractions": bet_fracs}
            )

            round_player_reward = float(np.mean(rewards["players"]))
            player_rewards.append(round_player_reward)
            # Shared player actor update (collective objective): sum log-prob across all players in a round.
            player_log_probs.append(bribe_log_prob_sum + bet_log_prob_sum)

            round_host_reward = float(rewards["host"])
            host_rewards.append(round_host_reward)
            winning_door = int(info.get("winning_door", -1))
            private_signals = env.hist_private_signals[-1].astype(np.int32)
            if winning_door >= 0:
                ep_truth_rate_sum += float(np.mean(private_signals == winning_door))
                ep_truth_rate_rounds += 1
            ep_follow_rate_sum += float(np.mean(doors == private_signals))
            ep_avg_bet_sum += float(np.mean(env.hist_bets[-1]))
            ep_avg_bribe_sum += float(np.mean(env.hist_bribes[-1]))
            ep_rounds += 1
            if terminated or truncated:
                break

        returns_player = discounted_returns(player_rewards, cfg.gamma, device)
        values = torch.stack(player_values)
        # Detach baseline values so actor update does not backprop through critic.
        advantages = returns_player - values.detach()

        critic_loss = F.mse_loss(values, returns_player)
        # CTDE choice here is team-level: centralized critic provides one advantage per round.
        player_log_probs_tensor = torch.stack(player_log_probs)
        assert player_log_probs_tensor.shape == advantages.shape
        player_loss = -(advantages * player_log_probs_tensor).mean()

        returns_host = discounted_returns(host_rewards, cfg.gamma, device)
        host_loss = -(returns_host * torch.stack(host_log_probs)).mean()

        opt_critic.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(player_critic.parameters(), cfg.grad_clip_norm)
        opt_critic.step()

        opt_player.zero_grad()
        player_loss.backward()
        nn.utils.clip_grad_norm_(player_actor.parameters(), cfg.grad_clip_norm)
        opt_player.step()

        opt_host.zero_grad()
        host_loss.backward()
        nn.utils.clip_grad_norm_(host_policy.parameters(), cfg.grad_clip_norm)
        opt_host.step()

        if ep % 10 == 0 or ep == 1:
            print(
                f"Ep {ep:4d} | rounds={len(player_rewards):3d} | "
                f"player_return={returns_player[0].item():+.2f} | "
                f"host_return={returns_host[0].item():+.2f}"
            )

        metric_row = {
            "step": ep,
            "avg_bet": ep_avg_bet_sum / max(1, ep_rounds),
            "avg_bribe": ep_avg_bribe_sum / max(1, ep_rounds),
            "host_final_reward": float(np.sum(host_rewards)),
            "player_final_reward": float(np.sum(player_rewards)),
            "host_true_private_signal_rate": ep_truth_rate_sum / max(1, ep_truth_rate_rounds),
            "player_follow_private_signal_rate": ep_follow_rate_sum / max(1, ep_rounds),
        }
        metrics_rows.append(metric_row)
        if ep % 10 == 0 or ep == 1:
            _print_metric_row("[Metrics]", metric_row)

        if ep % cfg.save_every == 0:
            torch.save(
                {
                    "episode": ep,
                    "num_players": cfg.num_players,
                    "num_doors": cfg.num_doors,
                    "player_actor": player_actor.state_dict(),
                    "player_critic": player_critic.state_dict(),
                    "host_policy": host_policy.state_dict(),
                },
                os.path.join(cfg.out_dir, f"ctde_ep_{ep}.pt"),
            )
            _save_metrics_csv(metrics_path, metrics_rows)

    final_path = os.path.join(cfg.out_dir, "ctde_final.pt")
    torch.save(
        {
            "episode": cfg.episodes,
            "num_players": cfg.num_players,
            "num_doors": cfg.num_doors,
            "player_actor": player_actor.state_dict(),
            "player_critic": player_critic.state_dict(),
            "host_policy": host_policy.state_dict(),
        },
        final_path,
    )
    _save_metrics_csv(metrics_path, metrics_rows)
    if metrics_rows:
        _print_metric_row("[Metrics]", metrics_rows[-1])
    print(f"Metrics saved to {metrics_path}")
    print(f"Training complete. Saved: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50000)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--num-players", type=int, default=10)
    parser.add_argument("--num-doors", type=int, default=3)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--history-window", type=int, default=50)
    parser.add_argument("--initial-balance", type=float, default=1000.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr-player", type=float, default=3e-4)
    parser.add_argument("--lr-critic", type=float, default=3e-4)
    parser.add_argument("--lr-host", type=float, default=3e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="./checkpoints_ctde")
    args = parser.parse_args()

    train_cfg = TrainConfig(
        episodes=args.episodes,
        save_every=args.save_every,
        gamma=args.gamma,
        lr_player=args.lr_player,
        lr_critic=args.lr_critic,
        lr_host=args.lr_host,
        grad_clip_norm=args.grad_clip_norm,
        num_players=args.num_players,
        num_doors=args.num_doors,
        max_rounds=args.max_rounds,
        initial_balance=args.initial_balance,
        history_window=args.history_window,
        seed=args.seed,
        out_dir=args.out_dir,
    )
    train_both_ctde(train_cfg)