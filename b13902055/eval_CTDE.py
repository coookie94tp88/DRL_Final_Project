import argparse
import csv
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from env import OracleGambitConfig, OracleGambitEnv, Phase
from host_CTDE_agent import TrainedCTDEHostAgent
from player_CTDE_agent import TrainedCTDEPlayerAgent

os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"


METRIC_COLUMNS = [
    "checkpoint",
    "checkpoint_episode",
    "episodes",
    "total_rounds",
    "avg_bet",
    "avg_bribe",
    "host_final_reward",
    "player_final_reward",
    "host_true_private_signal_rate",
    "player_follow_private_signal_rate",
]


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


def parse_checkpoint_episode(path: str) -> int | None:
    m = re.search(r"ctde_ep_(\d+)\.pt$", os.path.basename(path))
    if not m:
        return None
    return int(m.group(1))


def find_checkpoints(
    checkpoint_dir: str,
    start_ep: int | None,
    end_ep: int | None,
    ep_step: int,
) -> list[str]:
    root = Path(os.path.abspath(os.path.expanduser(checkpoint_dir)))
    if not root.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    candidates: list[tuple[int, str]] = []
    for p in root.glob("ctde_ep_*.pt"):
        ep = parse_checkpoint_episode(str(p))
        if ep is None:
            continue
        if start_ep is not None and ep < start_ep:
            continue
        if end_ep is not None and ep > end_ep:
            continue
        candidates.append((ep, str(p)))
    candidates.sort(key=lambda x: x[0])
    if not candidates:
        return []

    if ep_step <= 0:
        raise ValueError("--ep-step must be > 0")

    if start_ep is None:
        start_ep = candidates[0][0]
    selected = [(ep, p) for ep, p in candidates if (ep - start_ep) % ep_step == 0]
    return [p for _, p in selected]


def evaluate_checkpoint(
    checkpoint_path: str,
    args: argparse.Namespace,
    console: Console,
    show_round_log: bool,
) -> dict:
    deterministic = not args.stochastic
    cfg = OracleGambitConfig(
        num_players=args.num_players,
        num_doors=args.num_doors,
        max_rounds=args.max_rounds,
        history_window=args.history_window,
        initial_balance=args.initial_balance,
    )
    env = OracleGambitEnv(config=cfg, seed=args.seed)
    player_agent = TrainedCTDEPlayerAgent(
        checkpoint_path=checkpoint_path,
        num_players=cfg.num_players,
        num_doors=cfg.num_doors,
        history_window=cfg.history_window,
        device=args.device,
    )
    host_agent = TrainedCTDEHostAgent(
        checkpoint_path=checkpoint_path,
        num_players=cfg.num_players,
        num_doors=cfg.num_doors,
        history_window=cfg.history_window,
        device=args.device,
    )

    episode_player_rewards: list[float] = []
    episode_host_rewards: list[float] = []
    total_rounds = 0
    total_avg_bet_sum = 0.0
    total_avg_bribe_sum = 0.0
    total_truth_rate_sum = 0.0
    total_follow_rate_sum = 0.0

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        ep_player_reward = 0.0
        ep_host_reward = 0.0
        rounds = 0
        if show_round_log:
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
                total_avg_bet_sum += float(np.mean(env.hist_bets[-1]))
                total_avg_bribe_sum += float(np.mean(env.hist_bribes[-1]))
                winning_door = int(info.get("winning_door", -1))
                private_signals = env.hist_private_signals[-1].astype(np.int32)
                if winning_door >= 0:
                    total_truth_rate_sum += float(np.mean(private_signals == winning_door))
                total_follow_rate_sum += float(np.mean(doors == private_signals))
                if show_round_log:
                    render_round_log(
                        console=console,
                        env=env,
                        rewards=rewards,
                        info=info,
                        episode_num=ep,
                        round_num=rounds,
                    )

                if terminated or truncated:
                    if show_round_log:
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

    checkpoint_episode = parse_checkpoint_episode(checkpoint_path)
    return {
        "checkpoint": checkpoint_path,
        "checkpoint_episode": checkpoint_episode,
        "episodes": args.episodes,
        "total_rounds": total_rounds,
        "avg_bet": total_avg_bet_sum / max(1, total_rounds),
        "avg_bribe": total_avg_bribe_sum / max(1, total_rounds),
        "host_final_reward": float(np.mean(episode_host_rewards)),
        "player_final_reward": float(np.mean(episode_player_rewards)),
        "host_true_private_signal_rate": total_truth_rate_sum / max(1, total_rounds),
        "player_follow_private_signal_rate": total_follow_rate_sum / max(1, total_rounds),
        "mode": "deterministic" if deterministic else "stochastic",
    }


def print_summary_table(console: Console, rows: list[dict]) -> None:
    table = Table(title="CTDE Checkpoint Evaluation Summary", show_header=True, header_style="bold magenta")
    table.add_column("ep", justify="right")
    table.add_column("checkpoint", justify="left")
    table.add_column("episodes", justify="right")
    table.add_column("rounds", justify="right")
    table.add_column("avg_bet", justify="right")
    table.add_column("avg_bribe", justify="right")
    table.add_column("host_final_reward", justify="right")
    table.add_column("player_final_reward", justify="right")
    table.add_column("host_true_priv_rate", justify="right")
    table.add_column("player_follow_priv_rate", justify="right")

    for row in rows:
        episode_label = "-" if row["checkpoint_episode"] is None else str(int(row["checkpoint_episode"]))
        table.add_row(
            episode_label,
            os.path.basename(row["checkpoint"]),
            str(int(row["episodes"])),
            str(int(row["total_rounds"])),
            f"{row['avg_bet']:.3f}",
            f"{row['avg_bribe']:.3f}",
            f"{row['host_final_reward']:+.3f}",
            f"{row['player_final_reward']:+.3f}",
            f"{row['host_true_private_signal_rate']:.3f}",
            f"{row['player_follow_private_signal_rate']:.3f}",
        )
    console.print(table)


def write_summary_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in METRIC_COLUMNS})


def plot_summary_figure(rows: list[dict], output_path: str, title: str) -> None:
    episodes = [r["checkpoint_episode"] for r in rows]
    has_missing_episode = any(ep is None for ep in episodes)
    if has_missing_episode:
        x = np.arange(1, len(rows) + 1, dtype=np.int32)
        x_label = "Checkpoint Index"
    else:
        assert all(ep is not None for ep in episodes)
        x = np.asarray([int(ep) for ep in episodes], dtype=np.int32)
        x_label = "Checkpoint Episode"
    host_reward = np.asarray([float(r["host_final_reward"]) for r in rows], dtype=np.float32)
    player_reward = np.asarray([float(r["player_final_reward"]) for r in rows], dtype=np.float32)
    avg_bribe = np.asarray([float(r["avg_bribe"]) for r in rows], dtype=np.float32)
    avg_bet = np.asarray([float(r["avg_bet"]) for r in rows], dtype=np.float32)
    host_true_priv = np.asarray([float(r["host_true_private_signal_rate"]) for r in rows], dtype=np.float32)
    player_follow_priv = np.asarray([float(r["player_follow_private_signal_rate"]) for r in rows], dtype=np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(x, host_reward, marker="o", linewidth=2, label="Host Final Reward")
    ax.plot(x, player_reward, marker="o", linewidth=2, label="Player Final Reward")
    ax.axhline(0.0, color="gray", linewidth=1, linestyle="--")
    ax.set_title("Final Rewards vs Checkpoint")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(x, avg_bribe, marker="o", linewidth=2, label="Avg Bribe")
    ax.plot(x, avg_bet, marker="o", linewidth=2, label="Avg Bet")
    ax.set_title("Average Bribe / Bet vs Checkpoint")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Fraction")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(x, host_true_priv, marker="o", linewidth=2, label="Host True Private Signal Rate")
    ax.set_title("Host True Private Signal Rate")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(x, player_follow_priv, marker="o", linewidth=2, color="tab:green", label="Player Follow Private Signal Rate")
    ax.set_title("Player Follow Private Signal Rate")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle(title)
    output_file = os.path.abspath(os.path.expanduser(output_path))
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-path",
        "--checkpoint_path",
        dest="checkpoint_path",
        default=None,
        help="Optional single CTDE checkpoint (.pt). If omitted, checkpoints are discovered from --checkpoint-dir.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints_ctde",
        help="Directory to discover checkpoints named ctde_ep_xx.pt",
    )
    parser.add_argument("--start-ep", type=int, default=None, help="Start checkpoint episode (inclusive)")
    parser.add_argument("--end-ep", type=int, default=None, help="End checkpoint episode (inclusive)")
    parser.add_argument("--ep-step", type=int, default=50, help="Episode interval for checkpoint selection")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--num_players", type=int, default=10)
    parser.add_argument("--num_doors", type=int, default=4)
    parser.add_argument("--max_rounds", type=int, default=20)
    parser.add_argument("--history_window", type=int, default=50)
    parser.add_argument("--initial_balance", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--render-round-log", action="store_true", help="Print per-round details while evaluating")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional CSV path for summary table")
    parser.add_argument("--no-figure", action="store_true", help="Disable figure generation")
    parser.add_argument("--figure-path", type=str, default=None, help="Optional output path for summary figure (.png)")
    parser.add_argument("--figure-title", type=str, default="CTDE Checkpoint Evaluation", help="Title for generated figure")
    args = parser.parse_args()

    console = Console()
    deterministic = not args.stochastic

    if args.checkpoint_path:
        checkpoint_path = os.path.abspath(os.path.expanduser(args.checkpoint_path))
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoints = [checkpoint_path]
    else:
        checkpoints = find_checkpoints(
            checkpoint_dir=args.checkpoint_dir,
            start_ep=args.start_ep,
            end_ep=args.end_ep,
            ep_step=args.ep_step,
        )
        if not checkpoints:
            raise FileNotFoundError(
                "No checkpoints found for the given filter. "
                f"dir={args.checkpoint_dir}, start_ep={args.start_ep}, end_ep={args.end_ep}, step={args.ep_step}"
            )

    console.print(
        Panel(
            "OracleGambit - CTDE Evaluation\n"
            f"Checkpoints={len(checkpoints)}  Episodes/ckpt={args.episodes}  "
            f"Players={args.num_players}  Doors={args.num_doors}  Rounds={args.max_rounds}\n"
            f"Mode: {'deterministic' if deterministic else 'stochastic'}",
            style="bold yellow",
        )
    )

    rows: list[dict] = []
    for idx, ckpt in enumerate(checkpoints, start=1):
        console.print(f"[cyan]({idx}/{len(checkpoints)}) Evaluating {ckpt}[/cyan]")
        row = evaluate_checkpoint(
            checkpoint_path=ckpt,
            args=args,
            console=console,
            show_round_log=args.render_round_log and len(checkpoints) == 1,
        )
        rows.append(row)

    # Put checkpoints without parsable `ctde_ep_<n>.pt` episode numbers at the end.
    rows.sort(key=lambda x: (x["checkpoint_episode"] is None, x["checkpoint_episode"] or 0))
    print_summary_table(console, rows)
    if args.output_csv:
        write_summary_csv(args.output_csv, rows)
        console.print(f"[green]Summary CSV saved to {args.output_csv}[/green]")
    if not args.no_figure:
        default_figure_path = os.path.join(
            os.path.abspath(os.path.expanduser(args.checkpoint_dir)),
            "eval_ctde_summary.png",
        )
        figure_path = args.figure_path if args.figure_path else default_figure_path
        plot_summary_figure(rows=rows, output_path=figure_path, title=args.figure_title)
        console.print(f"[green]Summary figure saved to {os.path.abspath(os.path.expanduser(figure_path))}[/green]")


if __name__ == "__main__":
    main()