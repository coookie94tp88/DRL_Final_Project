import argparse
import os

import numpy as np
from rich.console import Console
from rich.panel import Panel

from env import OracleGambitConfig, OracleGambitEnv, Phase
from host_CTDE_agent import TrainedCTDEHostAgent
from player_CTDE_agent import TrainedCTDEPlayerAgent

os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"


def get_door_name(door_idx: int) -> str:
    if door_idx < 0:
        return "-"
    return chr(65 + int(door_idx))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-path",
        "--checkpoint_path",
        dest="checkpoint_path",
        required=True,
        help="Path to CTDE checkpoint (.pt)",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--num_players", type=int, default=10)
    parser.add_argument("--num_doors", type=int, default=4)
    parser.add_argument("--max_rounds", type=int, default=20)
    parser.add_argument("--history_window", type=int, default=50)
    parser.add_argument("--initial_balance", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    deterministic = not args.stochastic
    console = Console()

    cfg = OracleGambitConfig(
        num_players=args.num_players,
        num_doors=args.num_doors,
        max_rounds=args.max_rounds,
        history_window=args.history_window,
        initial_balance=args.initial_balance,
    )

    env = OracleGambitEnv(config=cfg, seed=args.seed)
    player_agent = TrainedCTDEPlayerAgent(
        checkpoint_path=args.checkpoint_path,
        num_players=cfg.num_players,
        num_doors=cfg.num_doors,
        history_window=cfg.history_window,
        device=args.device,
    )
    host_agent = TrainedCTDEHostAgent(
        checkpoint_path=args.checkpoint_path,
        num_players=cfg.num_players,
        num_doors=cfg.num_doors,
        history_window=cfg.history_window,
        device=args.device,
    )

    episode_player_rewards = []
    episode_host_rewards = []
    total_rounds = 0

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        ep_player_reward = 0.0
        ep_host_reward = 0.0
        rounds = 0

        while True:
            if env.phase == Phase.BRIBE:
                bribe_fractions = player_agent.get_bribe_action(obs["players"], deterministic=deterministic)
                obs, _, _, _, _ = env.step({"player_bribe_fractions": bribe_fractions})
            elif env.phase == Phase.SIGNAL:
                pub_sig, priv_sigs = host_agent.get_action(obs["host"], deterministic=deterministic)
                obs, _, _, _, _ = env.step({"public_signal": pub_sig, "private_signals": priv_sigs})
            elif env.phase == Phase.BET:
                doors, bet_fracs = player_agent.get_bet_action(obs["players"], deterministic=deterministic)
                obs, rewards, terminated, truncated, info = env.step(
                    {"player_doors": doors, "bet_fractions": bet_fracs}
                )
                ep_player_reward += float(np.mean(rewards["players"]))
                ep_host_reward += float(rewards["host"])
                rounds += 1

                if terminated or truncated:
                    console.print(
                        f"[cyan]Ep {ep}[/cyan] rounds={rounds} "
                        f"player_return={ep_player_reward:+.2f} "
                        f"host_return={ep_host_reward:+.2f} "
                        f"last_winning_door={get_door_name(info['winning_door'])}"
                    )
                    break
            else:
                raise RuntimeError(f"Unknown phase {env.phase}")

        episode_player_rewards.append(ep_player_reward)
        episode_host_rewards.append(ep_host_reward)
        total_rounds += rounds

    console.print(
        Panel(
            f"Episodes: {args.episodes}\n"
            f"Total rounds: {total_rounds}\n"
            f"Mean player return: {np.mean(episode_player_rewards):+.2f}\n"
            f"Mean host return: {np.mean(episode_host_rewards):+.2f}\n"
            f"Mode: {'deterministic' if deterministic else 'stochastic'}",
            title="CTDE Eval Summary",
            style="green",
        )
    )


if __name__ == "__main__":
    main()
