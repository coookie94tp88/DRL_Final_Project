import numpy as np
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel

from env import OracleGambitEnv, OracleGambitConfig, Phase, PlayerBelief

BELIEF_LABELS = {
    PlayerBelief.BELIEVE_PUBLIC: "信public",
    PlayerBelief.BELIEVE_PRIVATE: "信private",
    PlayerBelief.RANDOM: "隨機",
}


def get_door_name(door_idx: int) -> str:
    if door_idx < 0:
        return "-"
    return chr(65 + int(door_idx))


def render_round_log(console: Console, env: OracleGambitEnv, rewards: dict, info: dict, round_num: int):
    c = env.cfg

    correct_door = info["winning_door"]
    pub_sig = int(env.hist_public_signal[-1])
    priv_sigs = env.hist_private_signals[-1].astype(int)
    beliefs = env.hist_beliefs[-1].astype(int)
    chosen = env.hist_chosen_doors[-1].astype(int)
    bribes = env.hist_bribes[-1]
    bets = env.hist_bets[-1]
    player_rewards = env.hist_player_rewards[-1]
    host_profit = rewards["host"]

    frac_correct = float(env.hist_frac_correct[-1])
    frac_pub = float(env.hist_frac_believe_public[-1])
    host_honest = float(env.hist_host_public_honest[-1])

    header = Text()
    header.append(f"Round   {round_num:2d}/{c.max_rounds}  ", style="cyan")
    header.append("Correct Door: ")
    header.append(f"🚪{get_door_name(correct_door)}  ", style="bold yellow")
    header.append("Public Signal: ")
    pub_style = "bold green" if pub_sig == correct_door else "bold red"
    pub_icon = "✔" if pub_sig == correct_door else "✖"
    header.append(f"🚪{get_door_name(pub_sig)}{pub_icon}  ", style=pub_style)
    header.append(f"Host誠實(public): {'是' if host_honest else '否'}  ", style="bold cyan")
    header.append(f"選對比例: {frac_correct*100:.0f}%  ", style="grey74")
    header.append(f"信public比例: {frac_pub*100:.0f}%  ", style="grey74")
    host_color = "green" if host_profit >= 0 else "red"
    header.append(f"Host Reward: {host_profit:+.1f}", style=f"bold {host_color}")

    console.print(header)

    table = Table(show_header=True, header_style="grey74", box=None, padding=(0, 2))
    table.add_column("Player")
    table.add_column("Belief")
    table.add_column("Door")
    table.add_column("Bribe", justify="right")
    table.add_column("Bet", justify="right")
    table.add_column("Balance", justify="right")
    table.add_column("Reward", justify="right")

    for i in range(c.num_players):
        belief = beliefs[i]
        belief_str = BELIEF_LABELS.get(PlayerBelief(belief), str(belief)) if belief >= 0 else "-"
        door = chosen[i]
        door_str = f"🚪{get_door_name(door)}"
        if door == correct_door:
            door_str += "✔"

        rew = player_rewards[i]
        rew_str = f"[{'green' if rew > 0 else 'red'}]{rew:+.1f}[/]" if rew != 0 else "0.0"
        bal_str = f"[green]{env.balances[i]:.1f}[/green]"

        table.add_row(
            f"player_{i}",
            belief_str,
            door_str,
            f"{bribes[i]:.1f}",
            f"{bets[i]:.1f}",
            bal_str,
            rew_str,
        )

    console.print(table)
    console.print(f"[grey74]Bribes total: {np.sum(bribes):.1f}[/grey74]")
    console.print("[cyan]" + "─" * 70 + "[/cyan]\n")


def main():
    console = Console()

    config = OracleGambitConfig(
        num_players=10,
        num_doors=4,
        max_rounds=10,
        initial_balance=1000.0,
        max_bribe_fraction=1,
    )
    env = OracleGambitEnv(config=config, seed=42)
    obs, info = env.reset()

    console.print(
        Panel(
            "OracleGambit - Random Agent Simulation\n"
            f"Players={config.num_players}  Doors={config.num_doors}  Rounds={config.max_rounds}  Seed=42",
            style="bold yellow",
        )
    )

    round_count = 1

    while True:
        if env.phase == Phase.BRIBE:
            action = {"player_bribe_fractions": env.player_bribe_action_space.sample()}
            env.step(action)

        elif env.phase == Phase.SIGNAL:
            action = env.host_action_space.sample()
            env.step(action)

        elif env.phase == Phase.BET:
            action = env.player_bet_action_space.sample()
            obs, rewards, terminated, truncated, info = env.step(action)
            render_round_log(console, env, rewards, info, round_count)
            round_count += 1

            if terminated or truncated:
                console.print("[bold red]=== Game Over ===[/bold red]")
                break


if __name__ == "__main__":
    main()
