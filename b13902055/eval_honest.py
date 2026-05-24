import os
import argparse
import numpy as np
from stable_baselines3 import PPO
from rich.console import Console
from rich.panel import Panel

from env import OracleGambitEnv, OracleGambitConfig, Phase

os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"

def get_door_name(door_idx: int) -> str:
    if door_idx < 0:
        return "-"
    return chr(65 + int(door_idx))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to trained player_model_*.zip")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--num_players", type=int, default=10)
    parser.add_argument("--num_doors", type=int, default=3)
    parser.add_argument("--max_rounds", type=int, default=20)
    parser.add_argument("--init_balance", type=float, default=1000)
    args = parser.parse_args()

    console = Console()
    cfg = OracleGambitConfig(
        num_players=args.num_players,
        num_doors=args.num_doors,
        max_rounds=args.max_rounds,
        initial_balance=args.init_balance,
    )
    env = OracleGambitEnv(cfg, seed=42)
    from stable_baselines3.common.env_util import make_vec_env

    player_policy = PPO.load(args.model_path, device="auto")

    scores = []
    win_counts = np.zeros(cfg.num_players, dtype=int)

    for ep in range(args.episodes):
        obs, info = env.reset()
        player_obs = obs["players"]
        round_cnt = 0
        done = False
        episode_reward = 0

        while not done and round_cnt < cfg.max_rounds:
            if env.phase == Phase.BRIBE:
                # PPO expects flattened box
                action, _ = player_policy.predict(player_obs, deterministic=True)
                n = cfg.num_players
                bribe = np.clip(action[:n], 0.0, 1.0)
                # step bribe
                obs, _, _, _, info = env.step({"player_bribe_fractions": bribe})
            elif env.phase == Phase.SIGNAL:
                # Host: honest (true private for each), random public
                winning_door = env.current_winning_door
                public_signal = env.host_action_space["public_signal"].sample()
                private_signals = np.full(cfg.num_players, winning_door, dtype=np.int32)
                obs, _, _, _, info = env.step({
                    "public_signal": public_signal,
                    "private_signals": private_signals,
                })
            elif env.phase == Phase.BET:
                action, _ = player_policy.predict(obs["players"], deterministic=True)
                n = cfg.num_players
                bet_frac = np.clip(action[n:2*n], 0.0, 1.0)
                door_vals = np.clip(action[2*n:], 0, cfg.num_doors-1)
                doors = np.rint(door_vals).astype(np.int32)
                obs, rewards, terminated, truncated, info = env.step({
                    "player_doors": doors,
                    "bet_fractions": bet_frac
                })
                player_obs = obs["players"]
                episode_reward += np.sum(rewards["players"])
                done = terminated or truncated
                round_cnt += 1

                # Track wins for stats
                true_door = info["winning_door"]
                for i in range(cfg.num_players):
                    if doors[i] == true_door and bet_frac[i] > 0:
                        win_counts[i] += 1

        scores.append(episode_reward)
        console.print(
            f"[blue]Episode {ep+1}: Total reward {episode_reward:.1f}, Wins: {np.sum(win_counts)}"
        )

    # Summary
    mean_score = np.mean(scores)
    total_wins = np.sum(win_counts)
    console.print(
        Panel(
            f"Mean episode reward: {mean_score:.2f}\nTotal player wins (all games): {total_wins}",
            title="Eval Results", style="green"
        )
    )


if __name__ == "__main__":
    main()