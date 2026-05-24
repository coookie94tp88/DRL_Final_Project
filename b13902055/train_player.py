import os
import argparse
import csv
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from env import OracleGambitEnv, OracleGambitConfig, Phase

os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"


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


class PlayerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        curr_shape = observation_space["current"].shape
        hist_shape = observation_space["history"].shape
        super().__init__(observation_space, features_dim=128)
        curr_dim = int(np.prod(curr_shape))
        hist_dim = int(np.prod(hist_shape))
        self.net = torch.nn.Sequential(
            torch.nn.Linear(curr_dim + hist_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
        )
    def forward(self, obs):
        current = obs["current"].flatten(1)
        history = obs["history"].flatten(1)
        x = torch.cat([current, history], dim=1)
        return self.net(x)


class PlayerEnvWrapper(gym.Env):
    def __init__(self, base_env):
        super().__init__()
        self.base = base_env
        self.cfg = base_env.cfg
        self.num_players = self.cfg.num_players
        self.num_doors = self.cfg.num_doors
        self.observation_space = base_env.player_observation_space

        # Action space: [bribe_fractions (N), bet_fractions (N), door_values (N)]
        low = np.array([0.0] * (3 * self.num_players), dtype=np.float32)
        high = np.array(
            [1.0] * (2 * self.num_players) + [self.num_doors - 1] * self.num_players,
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.last_round_metrics = {
            "avg_bet": 0.0,
            "avg_bribe": 0.0,
            "host_final_reward": 0.0,
            "player_final_reward": 0.0,
            "host_true_private_signal_rate": 1.0,
            "player_follow_private_signal_rate": 0.0,
        }

    def reset(self, *, seed=None, options=None):
        obs, info = self.base.reset(seed=seed, options=options)
        return obs["players"], info

    def step(self, action):
        if self.base.phase != Phase.BRIBE:
            raise RuntimeError("Expected Phase.BRIBE at start of step().")
        action = np.asarray(action, dtype=np.float32)
        n = self.num_players

        bribe = np.clip(action[:n], 0.0, 1.0)
        bet_frac = np.clip(action[n:2 * n], 0.0, 1.0)
        door_vals = np.clip(action[2 * n:], 0, self.num_doors - 1)
        doors = np.rint(door_vals).astype(np.int32)

        # BRIBE
        obs, _, _, _, _ = self.base.step_bribe(bribe)

        # SIGNAL (host: random public, true private)
        if self.base.phase != Phase.SIGNAL:
            raise RuntimeError("Expected Phase.SIGNAL after bribe.")
        winning_door = self.base.current_winning_door
        public_signal = self.base.host_action_space["public_signal"].sample()
        private_signals = np.full(self.num_players, winning_door, dtype=np.int32)
        obs, _, _, _, _ = self.base.step_signal(public_signal, private_signals)

        # BET
        if self.base.phase != Phase.BET:
            raise RuntimeError("Expected Phase.BET after signal.")
        obs, rewards, terminated, truncated, info = self.base.step_bet(doors, bet_frac)
        winning_door = int(info.get("winning_door", -1))
        if winning_door >= 0:
            host_truth_rate = float(np.mean(private_signals == winning_door))
        else:
            host_truth_rate = 0.0
        follow_private_rate = float(np.mean(doors == private_signals))
        self.last_round_metrics = {
            "avg_bet": float(np.mean(self.base.hist_bets[-1])),
            "avg_bribe": float(np.mean(self.base.hist_bribes[-1])),
            "host_final_reward": float(self.base.hist_host_profit[-1]),
            "player_final_reward": float(np.mean(self.base.hist_player_rewards[-1])),
            "host_true_private_signal_rate": host_truth_rate,
            "player_follow_private_signal_rate": follow_private_rate,
        }
        reward_scalar = float(np.mean(rewards["players"]))
        return obs["players"], reward_scalar, terminated, truncated, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_rounds", type=int, default=10000)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--ent_coef", type=float, default=0.00)
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume_path", type=str, default=None, help="Player checkpoint (zip) file")
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    metrics_path = os.path.join(args.ckpt_dir, "training_metrics.csv")
    metrics_rows = []
    cfg = OracleGambitConfig(num_doors=3, num_players=10, max_rounds=20)
    base_env = OracleGambitEnv(cfg)
    player_env = PlayerEnvWrapper(base_env)

    round_count = 0
    if args.resume_path and os.path.isfile(args.resume_path):
        print(f"Resuming player PPO from {args.resume_path}")
        player_policy = PPO.load(args.resume_path, env=player_env, device="auto", ent_coef=args.ent_coef)
        try:
            base = os.path.basename(args.resume_path)
            name = os.path.splitext(base)[0]
            round_count = int(name.split("_")[-1])
        except Exception:
            round_count = 0
    else:
        print("Training player PPO from scratch.")
        player_policy = PPO(
            "MultiInputPolicy",
            player_env,
            policy_kwargs=dict(features_extractor_class=PlayerExtractor),
            ent_coef=args.ent_coef,
            verbose=1,
        )

    base_env.reset()
    while round_count < args.total_rounds:
        player_policy.learn(total_timesteps=1, reset_num_timesteps=False)
        if base_env.phase == Phase.BRIBE:
            round_count += 1
            row = {"step": round_count, **player_env.last_round_metrics}
            metrics_rows.append(row)
            if round_count % args.save_every == 0 and round_count > 0:
                fname = f"{args.ckpt_dir}/player_model_honesthost_{round_count}.zip"
                player_policy.save(fname)
                _save_metrics_csv(metrics_path, metrics_rows)
                if metrics_rows:
                    _print_metric_row("[Metrics]", metrics_rows[-1])
                print(f"[Saved @ round {round_count}]")

    player_policy.save(f"{args.ckpt_dir}/player_model_honesthost_{round_count}.zip")
    _save_metrics_csv(metrics_path, metrics_rows)
    if metrics_rows:
        _print_metric_row("[Metrics]", metrics_rows[-1])
    print(f"Metrics saved to {metrics_path}")
    print(f"Training complete. Final checkpoint saved at round {round_count}")

if __name__ == "__main__":
    main()