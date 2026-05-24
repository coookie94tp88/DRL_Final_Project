import numpy as np
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel

# 載入你的環境與雙方 Agent
from env import OracleGambitEnv, OracleGambitConfig, Phase, PlayerBelief
from player_agent import TrainedPlayerAgent
from host_agent import TrainedHostAgent

def get_door_name(door_idx: int) -> str:
    """將數字 0, 1, 2, 3 轉換為 A, B, C, D"""
    if door_idx < 0: return "-"
    return chr(65 + int(door_idx))

def render_round_log(console: Console, env: OracleGambitEnv, rewards: dict, info: dict, round_num: int):
    """繪製終端機視覺化表格 (沿用你原本精美的 Rich 設計)"""
    c = env.cfg
    
    correct_door = info["winning_door"]
    pub_sig = int(env.hist_public_signal[-1])
    priv_sigs = env.hist_private_signals[-1].astype(int)
    raw_beliefs = env.hist_beliefs[-1].astype(int)
    bribes = env.hist_bribes[-1]
    # BELIEVE_PRIVATE without bribe is coerced to RANDOM in env._map_beliefs_to_doors
    beliefs = raw_beliefs.copy()
    beliefs[(bribes <= 0) & (beliefs == PlayerBelief.BELIEVE_PRIVATE)] = PlayerBelief.RANDOM
    choices = env.hist_chosen_doors[-1].astype(int)
    bets = env.hist_bets[-1]
    player_rewards = env.hist_player_rewards[-1]
    host_profit = rewards["host"]
    
    total_pool = np.sum(bets)
    door_totals = np.zeros(c.num_doors)
    for i in range(c.num_players):
        if bets[i] > 0:
            door_totals[choices[i]] += bets[i]
            
    frac_correct = float(env.hist_frac_correct[-1])
    frac_pub = float(env.hist_frac_believe_public[-1])
    host_honest = float(env.hist_host_public_honest[-1])

    console.print(f"\n[bold cyan]=== Round {round_num} ===[/bold cyan]")

    summary_text = Text()
    summary_text.append(f"Winning Door: {get_door_name(correct_door)}\n", style="bold green")
    summary_text.append(f"Host Public Signal: {get_door_name(pub_sig)}\n", style="bold yellow")
    summary_text.append(f"Host honest (public): {'yes' if host_honest else 'no'}\n", style="bold cyan")
    summary_text.append(f"Frac correct picks: {frac_correct*100:.1f}%\n", style="bold blue")
    summary_text.append(f"Frac believe public: {frac_pub*100:.1f}%\n", style="bold blue")
    summary_text.append(f"Total Pool: {total_pool:.2f}\n", style="bold magenta")
    summary_text.append(f"Host Profit: {host_profit:.2f}\n", style="bold red" if host_profit < 0 else "bold green")
    
    console.print(Panel(summary_text, title="Round Summary", expand=False))

    # --- 玩家詳細資訊 ---
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Player", justify="right")
    table.add_column("Bal (Start)", justify="right")
    table.add_column("Bribe", justify="right")
    table.add_column("Priv Sig", justify="center")
    table.add_column("Bet", justify="right")
    table.add_column("Belief", justify="center")
    table.add_column("Door", justify="center")
    table.add_column("Reward", justify="right")
    table.add_column("Bal (End)", justify="right", style="bold cyan")

    for i in range(c.num_players):
        bal_start = env.balances[i] - player_rewards[i]
        bal_end = env.balances[i]
        b_val = bribes[i]
        bet_val = bets[i]
        p_sig = get_door_name(priv_sigs[i])
        belief_str = TrainedPlayerAgent.BELIEF_NAMES[int(beliefs[i])] if beliefs[i] >= 0 else "-"
        choice_str = get_door_name(choices[i])
        r_val = player_rewards[i]

        if choices[i] == correct_door and bet_val > 0:
            choice_str = f"[green]{choice_str} ✓[/green]"
        elif bet_val > 0:
            choice_str = f"[red]{choice_str} ✗[/red]"

        # 標記真實內線
        if priv_sigs[i] == correct_door:
            p_sig = f"[green]{p_sig}[/green]"

        table.add_row(
            f"P{i}",
            f"{bal_start:.1f}",
            f"{b_val:.1f}",
            p_sig,
            f"{bet_val:.1f}",
            belief_str,
            choice_str,
            f"[green]+{r_val:.1f}[/green]" if r_val > 0 else f"[red]{r_val:.1f}[/red]",
            f"{bal_end:.1f}"
        )

    console.print(table)


class EvalPlayerStats:
    """Per-player aggregates over all completed rounds in one episode."""

    def __init__(self, num_players: int) -> None:
        self.n = num_players
        self.alive_rounds = np.zeros(num_players, dtype=np.int32)
        self.bribe_frac_sum = np.zeros(num_players, dtype=np.float64)
        self.bribe_paid_rounds = np.zeros(num_players, dtype=np.int32)
        self.believe_public = np.zeros(num_players, dtype=np.int32)
        self.believe_private = np.zeros(num_players, dtype=np.int32)
        self.believe_random = np.zeros(num_players, dtype=np.int32)
        self.priv_truth_rounds = np.zeros(num_players, dtype=np.int32)

    def record_round(
        self,
        *,
        round_active: np.ndarray,
        bribe_fractions: np.ndarray,
        paid_bribes: np.ndarray,
        beliefs: np.ndarray,
        private_signals: np.ndarray,
        winning_door: int,
    ) -> None:
        """Aggregate per-player stats for one completed betting round."""
        effective = np.asarray(beliefs, dtype=np.int32).copy()
        effective[(paid_bribes <= 0) & (effective == int(PlayerBelief.BELIEVE_PRIVATE))] = int(
            PlayerBelief.RANDOM
        )
        for i in range(self.n):
            if paid_bribes[i] > 0:
                self.bribe_paid_rounds[i] += 1
                if int(private_signals[i]) == int(winning_door):
                    self.priv_truth_rounds[i] += 1
            if round_active[i]:
                self.alive_rounds[i] += 1
                self.bribe_frac_sum[i] += float(bribe_fractions[i])
                belief = int(effective[i])
                if belief == int(PlayerBelief.BELIEVE_PUBLIC):
                    self.believe_public[i] += 1
                elif belief == int(PlayerBelief.BELIEVE_PRIVATE):
                    self.believe_private[i] += 1
                elif belief == int(PlayerBelief.RANDOM):
                    self.believe_random[i] += 1


def render_player_summary(console: Console, stats: EvalPlayerStats, total_rounds: int) -> None:
    console.print(f"\n[bold yellow]=== Per-Player Summary ({total_rounds} rounds) ===[/bold yellow]")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Player", justify="right")
    table.add_column("Alive Rds", justify="right")
    table.add_column("Avg Bribe Frac", justify="right")
    table.add_column("Bribe>0 (cnt)", justify="right")
    table.add_column("pub", justify="right")
    table.add_column("priv", justify="right")
    table.add_column("rnd", justify="right")
    table.add_column("P(Priv Truth)", justify="right")

    for i in range(stats.n):
        alive = int(stats.alive_rounds[i])
        bribe_cnt = int(stats.bribe_paid_rounds[i])
        bribe_cnt_str = f"{bribe_cnt}/{total_rounds}"
        believe_pub = int(stats.believe_public[i])
        believe_priv = int(stats.believe_private[i])
        believe_rnd = int(stats.believe_random[i])
        if alive > 0:
            believe_pub_str = f"{believe_pub}/{alive}"
            believe_priv_str = f"{believe_priv}/{alive}"
            believe_rnd_str = f"{believe_rnd}/{alive}"
        else:
            believe_pub_str = believe_priv_str = believe_rnd_str = "N/A"

        if alive > 0:
            avg_frac = stats.bribe_frac_sum[i] / alive
            avg_frac_str = f"{avg_frac * 100:.2f}%"
        else:
            avg_frac_str = "-"

        if bribe_cnt > 0:
            p_priv = stats.priv_truth_rounds[i] / bribe_cnt
            priv_str = f"{int(stats.priv_truth_rounds[i])}/{bribe_cnt} ({p_priv * 100:.1f}%)"
        else:
            priv_str = "N/A"

        table.add_row(
            f"P{i}",
            str(alive),
            avg_frac_str,
            bribe_cnt_str,
            believe_pub_str,
            believe_priv_str,
            believe_rnd_str,
            priv_str,
        )

    console.print(table)
    console.print(
        "[grey62]Bribe>0 (cnt): paid bribe > 0 rounds / episode rounds. "
        "pub/priv/rnd: effective belief counts / alive rounds. "
        "P(Priv Truth): private honest rounds / bribe>0 rounds. "
        "Avg Bribe Frac: mean actor output while alive (before env min-$1).[/grey62]"
    )


if __name__ == "__main__":
    console = Console()
    
    PLAYER_CKPT = "checkpoints/player.pth"
    HOST_CKPT = "checkpoints/host.pth"

    config = OracleGambitConfig(
        num_players=10, 
        num_doors=4, 
        max_rounds=20, # Eval 時可以先看 20 局 
        initial_balance=1000.0,
    )
    env = OracleGambitEnv(config=config, seed=42)
    obs, info = env.reset()
    
    # 動態獲取 hist_dim 以確保網路輸入維度一致
    hist_dim = obs["host"]["history"].shape[1]
    
    # 初始化雙方 Agent (都從同一個 checkpoint 讀取)
    console.print("[bold yellow]載入 MARL Agents...[/bold yellow]")
    player_agent = TrainedPlayerAgent(PLAYER_CKPT, device="cuda")
    host_agent = TrainedHostAgent(HOST_CKPT, config, hist_dim=hist_dim, device="cuda")
    
    console.print(Panel(
        "OracleGambit - AI vs AI Evaluation\n"
        f"Players={config.num_players}  Doors={config.num_doors}  Rounds={config.max_rounds}", 
        style="bold yellow"
    ))
    
    round_count = 1
    player_stats = EvalPlayerStats(config.num_players)
    last_bribe_fractions = np.zeros(config.num_players, dtype=np.float32)

    while True:
        if env.phase == Phase.BRIBE:
            round_active = env.balances > 0
            # 1. 玩家推論：決定賄賂比例 (deterministic=True)
            bribe_fractions = player_agent.get_bribe_action(obs["players"], deterministic=True)
            last_bribe_fractions = np.asarray(bribe_fractions, dtype=np.float32).copy()
            action = {"player_bribe_fractions": bribe_fractions}
            obs, _, _, _, info = env.step(action)
            
        elif env.phase == Phase.SIGNAL:
            # 2. Host 推論：給出公頻與私頻訊號 (使用 RDQN)
            pub_sig, priv_sigs = host_agent.get_action(env)
            action = {
                "public_signal": pub_sig,
                "private_signals": priv_sigs
            }
            obs, _, _, _, info = env.step(action)
            
        elif env.phase == Phase.BET:
            # 3. 玩家推論：決定選哪個門與下注比例 (deterministic=True)
            beliefs, bet_fracs = player_agent.get_bet_action(obs["players"], deterministic=True)
            action = {
                "player_beliefs": beliefs,
                "bet_fractions": bet_fracs,
            }
            obs, rewards, terminated, truncated, info = env.step(action)
            
            # 4. 渲染結果
            render_round_log(console, env, rewards, info, round_count)

            player_stats.record_round(
                round_active=round_active,
                bribe_fractions=last_bribe_fractions,
                paid_bribes=env.hist_bribes[-1],
                beliefs=env.hist_beliefs[-1],
                private_signals=env.hist_private_signals[-1],
                winning_door=int(info["winning_door"]),
            )

            if terminated or truncated:
                console.print(f"\n[bold yellow]Game Over at Round {round_count}![/bold yellow]")
                render_player_summary(console, player_stats, round_count)
                break

            round_count += 1