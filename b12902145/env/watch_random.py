"""
OracleGambit (Simplified) — Random Agent 觀察器
================================================
用法：
    python env/watch_random.py              # 預設 20 回合
    python env/watch_random.py --rounds 50 --players 6 --seed 99
    python env/watch_random.py --obs        # 也印觀察向量摘要
"""

from __future__ import annotations

import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from env.oracle_gambit_env import OracleGambitEnv

# ── ANSI 色彩 ─────────────────────────────────────────────────────────────
R   = "\033[91m"; G  = "\033[92m"; Y  = "\033[93m"
B   = "\033[94m"; M  = "\033[95m"; C  = "\033[96m"
DIM = "\033[2m";  RST = "\033[0m"
DOOR_ICON = ["🚪A", "🚪B", "🚪C", "🚪D"]


def _bar(value: float, max_val: float, width: int = 20) -> str:
    filled = int(round(value / max_val * width)) if max_val > 0 else 0
    filled = max(0, min(filled, width))
    return "█" * filled + "░" * (width - filled)


def _color_reward(r: float) -> str:
    col = G if r > 0 else (R if r < 0 else DIM)
    return f"{col}{r:+.2f}{RST}"


def run(
    num_players: int = 6,
    num_doors:   int = 4,
    num_rounds:  int = 20,
    seed:        int = 42,
    show_obs:    bool = False,
) -> None:
    rng = np.random.default_rng(seed)
    env = OracleGambitEnv(
        num_players=num_players,
        num_doors=num_doors,
        payout_threshold=0.20,
        max_rounds=num_rounds,
        seed=seed,
    )
    env.reset(seed=seed)

    stat_host:     list[float] = []
    stat_win_ratio: list[float] = []
    stat_signal_honest: list[int] = []

    print(f"\n{Y}{'═'*68}{RST}")
    print(f"{Y}  OracleGambit (Simplified) — Random Agent Simulation{RST}")
    print(f"{Y}  Players={num_players}  Doors={num_doors}  Rounds={num_rounds}  "
          f"θ=0.20  Seed={seed}{RST}")
    print(f"{Y}{'═'*68}{RST}\n")

    for round_idx in range(num_rounds):
        # ── 隨機動作 ──────────────────────────────────────────────────────
        host_act     = rng.random()
        player_acts  = {f"player_{i}": rng.random() for i in range(num_players)}

        rewards = env.step_all(host_act, player_acts)
        info    = env.last_round_info

        correct   = info["correct_door"]
        pub_sig   = info["public_signal"]
        choices   = info["door_choices"]
        ratios    = info["door_ratios"]
        win_ratio = info["win_ratio"]
        host_rwd  = info["rewards"]["host"]
        honest    = int(correct == pub_sig)

        stat_host.append(host_rwd)
        stat_win_ratio.append(win_ratio)
        stat_signal_honest.append(honest)

        # ── 回合標頭 ──────────────────────────────────────────────────────
        host_col = G if host_rwd >= 0 else R
        print(f"{B}{'─'*68}{RST}")
        print(
            f"{B}Round {round_idx+1:>3}/{num_rounds}{RST}  "
            f"Correct: {M}{DOOR_ICON[correct]}{RST}  "
            f"Signal:  {C}{DOOR_ICON[pub_sig]}{RST}"
            f"{'✓' if honest else f'{DIM}✗{RST}'}  "
            f"Host: {host_col}{host_rwd:+.2f}{RST}"
        )

        # ── 各門選擇分布 ──────────────────────────────────────────────────
        print(f"  {DIM}Door choice distribution (N={num_players}){RST}")
        for d in range(num_doors):
            ratio = ratios[d]
            is_correct = d == correct
            bar_col = G if is_correct else DIM
            label = f"{Y}← CORRECT{RST}" if is_correct else ""
            print(
                f"    {DOOR_ICON[d]}  {bar_col}{_bar(ratio, 1.0)}{RST}"
                f"  {ratio*100:5.1f}%  ({int(round(ratio*num_players)):>2} players)  {label}"
            )

        # Win-ratio indicator
        ratio_col = G if win_ratio < 0.20 else R
        print(
            f"  Win-ratio x = {ratio_col}{win_ratio:.3f}{RST}  "
            f"[{ratio_col}{_bar(win_ratio, 1.0)}{RST}]  "
            f"θ=0.20  "
            + (f"{G}Host profits{RST}" if win_ratio < 0.20 else f"{R}Host loses{RST}")
        )

        # ── 玩家表格 ──────────────────────────────────────────────────────
        print(f"  {DIM}{'Player':<10} {'Signal':<9} {'Chose':<8} {'Reward':>8}{RST}")
        for pid in range(num_players):
            name = f"player_{pid}"
            door = choices.get(name, -1)
            rwd  = rewards.get(name, 0.0)
            won  = (door == correct)
            followed = (door == pub_sig)
            sig_str  = f"{DOOR_ICON[pub_sig]}"
            fol_mark = f"{'→' if followed else ' '}"
            door_str = f"{DOOR_ICON[door]}{'✓' if won else ' '}" if 0 <= door < num_doors else "??"
            print(
                f"  {name:<10} {sig_str:<9} {fol_mark}{door_str:<8}"
                f" {_color_reward(rwd):>8}"
            )

        if show_obs:
            obs = env.observe("player_0")
            valid = (obs != -1.0).sum()
            print(
                f"  {DIM}[obs] player_0 shape={obs.shape}  "
                f"valid={valid}/{obs.shape[0]}{RST}"
            )

    # ── 全場統計摘要 ──────────────────────────────────────────────────────
    print(f"\n{Y}{'═'*68}{RST}")
    print(f"{Y}  SUMMARY  ({num_rounds} rounds){RST}")
    print(f"{Y}{'═'*68}{RST}")

    cum_host  = float(np.sum(stat_host))
    avg_host  = float(np.mean(stat_host))
    avg_wr    = float(np.mean(stat_win_ratio))
    honest_pct = float(np.mean(stat_signal_honest)) * 100

    col_h = G if cum_host >= 0 else R
    print(f"\n  {M}【Host】{RST}")
    print(f"    Cumulative profit  : {col_h}{cum_host:+.2f}{RST}")
    print(f"    Avg profit / round : {col_h}{avg_host:+.2f}{RST}")
    print(f"    Signal honesty     : {honest_pct:.1f}%  "
          f"{DIM}(random host → ≈25% honest){RST}")

    print(f"\n  {B}【Market】{RST}")
    print(f"    Avg win-ratio      : {avg_wr:.3f}  "
          f"{'(Host profitable zone)' if avg_wr < 0.20 else f'{R}(Host loss zone){RST}'}")
    print(f"    Threshold θ        : 0.20  (break-even at 20% winners)")

    # Win-ratio sparkline
    print(f"\n  {DIM}Win-ratio sparkline (▇ = x<θ, ▂ = x≥θ):{RST}")
    line = ""
    for wr in stat_win_ratio:
        col = G if wr < 0.20 else R
        line += f"{col}{'▇' if wr < 0.20 else '▂'}{RST}"
    print(f"    {line}")

    print(f"\n{Y}{'═'*68}{RST}\n")


# ── CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OracleGambit Simplified — Watch")
    parser.add_argument("--rounds",  type=int, default=20, help="回合數")
    parser.add_argument("--players", type=int, default=6,  help="玩家數")
    parser.add_argument("--doors",   type=int, default=4,  help="門數")
    parser.add_argument("--seed",    type=int, default=42, help="隨機種子")
    parser.add_argument("--obs",     action="store_true",  help="印觀察向量摘要")
    args = parser.parse_args()
    run(
        num_players=args.players,
        num_doors=args.doors,
        num_rounds=args.rounds,
        seed=args.seed,
        show_obs=args.obs,
    )
