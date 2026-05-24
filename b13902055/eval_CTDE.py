import argparse
import os

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from env import OracleGambitConfig, OracleGambitEnv, Phase
from host_CTDE_agent import TrainedCTDEHostAgent
from player_CTDE_agent import TrainedCTDEPlayerAgent

os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"


def get_door_name(door_idx: int) -> str:
    if door_idx < 0:
        return "-"
    return chr(65 + int(door_idx))


def render_round_log(
    console: Console,
    env: OracleGambitEnv,
    rewards: dict,
    info: dict,
    episode_num: int,
    round_num: int,
) -> None:
    c = env.cfg

    correct_door = int(info["winning_door"])
    pub_sig = int(env.hist_public_signal[-1])
    priv_sigs = env.hist_private_signals[-1].astype(int)
    choices = env.hist_choices[-1].astype(int)
    bribes = env.hist_bribes[-1]
    bets = env.hist_bets[-1]
    player_rewards = env.hist_player_rewards[-1]
    host_profit = float(rewards["host"])

    total_pool = float(np.sum(bets))
    door_totals = np.zeros(c.num_doors, dtype=np.float32)
    for i in range(c.num_players):
        if bets[i] > 0:
            door_totals[choices[i]] += bets[i]

    door_ratios = np.zeros(c.num_doors, dtype=np.float32)
    if total_pool > 0:
        door_ratios = door_totals / total_pool

    console.print(f"\n[bold cyan]=== Episode {episode_num} / Round {round_num} ===[/bold cyan]")

    summary_text = Text()
    summary_text.append(f"Winning Door: {get_door_name(correct_door)}\n", style="bold green")
    summary_text.append(f"Host Public Signal: {get_door_name(pub_sig)}\n", style="bold yellow")
    summary_text.append(f"Total Pool: {total_pool:.2f}\n", style="bold magenta")
    summary_text.append(f"Host Profit: {host_profit:.2f}\n", style="bold red" if host_profit < 0 else "bold green")
    door_ratio_str = " | ".join([f"{get_door_name(d)}: {door_ratios[d] * 100:.1f}%" for d in range(c.num_doors)])
    summary_text.append(f"Door Distribution: [{door_ratio_str}]", style="bold blue")
    console.print(Panel(summary_text, title="Round Summary", expand=False))

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Player", justify="right")
    table.add_column("Bal (Start)", justify="right")
    table.add_column("Bribe", justify="right")
    table.add_column("Priv Sig", justify="center")
    table.add_column("Bet", justify="right")
    table.add_column("Choice", justify="center")
    table.add_column("Reward", justify="right")
    table.add_column("Bal (End)", justify="right", style="bold cyan")

    for i in range(c.num_players):
        bal_start = float(env.balances[i] - player_rewards[i])
        bal_end = float(env.balances[i])
        b_val = float(bribes[i])
        bet_val = float(bets[i])
        p_sig = get_door_name(priv_sigs[i])
        choice_str = get_door_name(choices[i])
        r_val = float(player_rewards[i])

        if choices[i] == correct_door and bet_val > 0:
            choice_str = f"[green]{choice_str} ✓[/green]"
        elif bet_val > 0:
            choice_str = f"[red]{choice_str} ✗[/red]"

        if priv_sigs[i] == correct_door:
            p_sig = f"[green]{p_sig}[/green]"

        table.add_row(
            f"P{i}",
            f"{bal_start:.1f}",
            f"{b_val:.1f}",
            p_sig,
            f"{bet_val:.1f}",
            choice_str,
            f"[green]+{r_val:.1f}[/green]" if r_val > 0 else f"[red]{r_val:.1f}[/red]",
            f"{bal_end:.1f}",
        )

    console.print(table)


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

    console.print(
        Panel(
            "OracleGambit - CTDE Evaluation\n"
            f"Players={cfg.num_players}  Doors={cfg.num_doors}  Rounds={cfg.max_rounds}  Episodes={args.episodes}",
            style="bold yellow",
        )
    )

    episode_player_rewards = []
    episode_host_rewards = []
    total_rounds = 0

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        ep_player_reward = 0.0
        ep_host_reward = 0.0
        rounds = 0
        console.print(f"\n[bold yellow]Starting Episode {ep}/{args.episodes}[/bold yellow]")

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
                render_round_log(
                    console=console,
                    env=env,
                    rewards=rewards,
                    info=info,
                    episode_num=ep,
                    round_num=rounds,
                )

                if terminated or truncated:
                    console.print(
                        f"\n[bold yellow]Episode {ep} Over at Round {rounds}![/bold yellow]\n"
                        f"[cyan]Episode return[/cyan] "
                        f"player={ep_player_reward:+.2f} host={ep_host_reward:+.2f}"
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