import os
os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel

from env import OracleGambitEnv, OracleGambitConfig, Phase
from player_agent import TrainedPlayerAgent
from host_agent import TrainedHostAgent


def get_door_name(door_idx: int) -> str:
    if door_idx < 0:
        return "-"
    return chr(65 + int(door_idx))


def render_round_log(console: Console, env: OracleGambitEnv, rewards: dict, info: dict, round_num: int):
    c = env.cfg

    correct_door = info["winning_door"]
    pub_sig = int(env.hist_public_signal[-1])
    priv_sigs = env.hist_private_signals[-1].astype(int)
    choices = env.hist_choices[-1].astype(int)
    bribes = env.hist_bribes[-1]
    bets = env.hist_bets[-1]
    player_rewards = env.hist_player_rewards[-1]
    host_profit = rewards["host"]

    total_pool = np.sum(bets)
    door_totals = np.zeros(c.num_doors)
    for i in range(c.num_players):
        if bets[i] > 0:
            door_totals[choices[i]] += bets[i]

    door_ratios = np.zeros(c.num_doors)
    if total_pool > 0:
        door_ratios = door_totals / total_pool

    console.print(f"\n[bold cyan]=== Round {round_num} ==-[/bold cyan]")

    summary_text = Text()
    summary_text.append(f"Winning Door: {get_door_name(correct_door)}\n", style="bold green")
    summary_text.append(f"Host Public Signal: {get_door_name(pub_sig)}\n", style="bold yellow")
    summary_text.append(f"Total Pool: {total_pool:.2f}\n", style="bold magenta")
    summary_text.append(f"Host Profit: {host_profit:.2f}\n", style="bold red" if host_profit < 0 else "bold green")

    door_ratio_str = " | ".join([f"{get_door_name(d)}: {door_ratios[d]*100:.1f}%" for d in range(c.num_doors)])
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
        bal_start = env.balances[i] - player_rewards[i]
        bal_end = env.balances[i]
        b_val = bribes[i]
        bet_val = bets[i]
        p_sig = get_door_name(priv_sigs[i])
        choice_str = get_door_name(choices[i])
        r_val = player_rewards[i]

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


if __name__ == "__main__":
    console = Console()

    PLAYER_PATH = "player_model.zip"
    HOST_PATH = "host_model.pt"

    config = OracleGambitConfig(
        num_players=10,
        num_doors=3,
        max_rounds=20,
        initial_balance=1000.0,
    )
    env = OracleGambitEnv(config=config, seed=42)
    obs, info = env.reset()

    console.print("[bold yellow]載入 MARL Agents...[/bold yellow]")
    player_agent = TrainedPlayerAgent(PLAYER_PATH, num_players=config.num_players, num_doors=config.num_doors, device="cuda")
    host_agent = TrainedHostAgent(HOST_PATH, config, device="cuda")

    console.print(Panel(
        "OracleGambit - AI vs AI Evaluation\n"
        f"Players={config.num_players}  Doors={config.num_doors}  Rounds={config.max_rounds}",
        style="bold yellow"
    ))

    round_count = 1

    while True:
        if env.phase == Phase.BRIBE:
            bribe_fractions = player_agent.get_bribe_action(obs["players"], deterministic=True)
            action = {"player_bribe_fractions": bribe_fractions}
            obs, _, _, _, info = env.step(action)

        elif env.phase == Phase.SIGNAL:
            pub_sig, priv_sigs = host_agent.get_action(env, deterministic=True)
            action = {
                "public_signal": pub_sig,
                "private_signals": priv_sigs
            }
            obs, _, _, _, info = env.step(action)

        elif env.phase == Phase.BET:
            doors, bet_fracs = player_agent.get_bet_action(obs["players"], deterministic=True)
            action = {
                "player_doors": doors,
                "bet_fractions": bet_fracs
            }
            obs, rewards, terminated, truncated, info = env.step(action)

            render_round_log(console, env, rewards, info, round_count)

            if terminated or truncated:
                console.print(f"\n[bold yellow]Game Over at Round {round_count}![/bold yellow]")
                break

            round_count += 1