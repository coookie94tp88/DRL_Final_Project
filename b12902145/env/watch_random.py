"""
OracleGambit – Random Agent 模擬觀察器
======================================
讓全 random agent 互玩若干回合，印出豐富的回合日誌與統計摘要。

用法：
    python env/watch_random.py              # 預設 20 回合
    python env/watch_random.py --rounds 50  # 自訂回合數
    python env/watch_random.py --seed 99 --players 6 --rounds 30
"""

from __future__ import annotations

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from env.oracle_gambit_env import OracleGambitEnv

# ── ANSI 色彩 ──────────────────────────────────────────────────────────────
R  = "\033[91m"   # 紅（輸家 / host 賠）
G  = "\033[92m"   # 綠（贏家 / host 賺）
Y  = "\033[93m"   # 黃（標題 / 警告）
B  = "\033[94m"   # 藍（資訊）
M  = "\033[95m"   # 紫（Host）
C  = "\033[96m"   # 青（玩家餘額）
DIM = "\033[2m"   # 暗（次要資訊）
RST = "\033[0m"   # reset

DOOR_ICON = ["🚪A", "🚪B", "🚪C", "🚪D"]


def _bar(value: float, max_val: float, width: int = 20, char: str = "█") -> str:
    """畫一條比例長條。"""
    if max_val <= 0:
        filled = 0
    else:
        filled = int(round(value / max_val * width))
    filled = max(0, min(filled, width))
    return char * filled + "░" * (width - filled)


def _color_reward(r: float) -> str:
    col = G if r > 0 else (R if r < 0 else DIM)
    return f"{col}{r:+.1f}{RST}"


def run(
    num_players: int = 5,
    num_doors:   int = 4,
    num_rounds:  int = 20,
    seed:        int = 42,
    show_obs:    bool = False,
) -> None:
    rng = np.random.default_rng(seed)
    env = OracleGambitEnv(
        num_players=num_players,
        num_doors=num_doors,
        initial_balance=1000.0,
        payout_threshold=0.25,
        fee_rate=0.05,
        welfare_amount=10.0,
        max_rounds=num_rounds,
        seed=seed,
    )
    env.reset(seed=seed)

    # 累積統計
    stat_host_profit: list[float] = []
    stat_win_ratio:   list[float] = []
    stat_total_bribe: list[float] = []
    stat_total_bet:   list[float] = []
    stat_balances:    dict[str, list[float]] = {
        f"player_{i}": [] for i in range(num_players)
    }
    signal_accuracy_pub:  list[int] = []   # 1=public signal == correct door
    signal_accuracy_priv: list[int] = []   # fraction of correct private signals

    print(f"\n{Y}{'═'*68}{RST}")
    print(f"{Y}  OracleGambit — Random Agent Simulation{RST}")
    print(f"{Y}  Players={num_players}  Doors={num_doors}  Rounds={num_rounds}  Seed={seed}{RST}")
    print(f"{Y}{'═'*68}{RST}\n")

    for round_idx in range(num_rounds):
        correct_door = env._correct_door

        # ── Phase I: 隨機賄賂 + Host 隨機發訊號 ────────────────────────
        host_act = rng.random(1 + num_players).astype(np.float32)
        actions_p1: dict = {"host": host_act}
        for pid in range(num_players):
            actions_p1[f"player_{pid}"] = float(rng.uniform(0.0, 0.3))  # 最多押 30%

        # ── Phase II: 隨機選門 + 隨機下注 ───────────────────────────────
        actions_p2: dict = {}
        for pid in range(num_players):
            actions_p2[f"player_{pid}"] = (
                float(rng.random()),   # door_fraction
                float(rng.uniform(0.1, 0.5)),  # bet_fraction
            )

        # 先快照動作（用於顯示），再 step
        bribe_amounts = {
            f"player_{pid}": actions_p1[f"player_{pid}"] * env._balances[f"player_{pid}"]
            for pid in range(num_players)
        }

        rewards = env.step_all(actions_p1, actions_p2)

        # 從 last_round_info 讀取本回合快照（step_all 結束後仍有效）
        info       = env.last_round_info
        pub_sig    = info["public_signal"]
        priv_sigs  = info["private_signals"]
        door_choice = info["door_choices"]
        bets       = info["bets"]
        balances   = info["balances"]
        rewards    = info["rewards"]

        total_bet  = sum(bets.values()) if bets else 0.0
        total_bribe = sum(bribe_amounts.values())
        win_bets   = {n: b for n, b in bets.items()
                      if door_choice.get(n) == correct_door}
        total_win  = sum(win_bets.values())
        win_ratio  = total_win / total_bet if total_bet > 0 else 0.0
        host_rwd   = rewards["host"]

        # 訊號準確率統計
        signal_accuracy_pub.append(int(pub_sig == correct_door))
        if priv_sigs:
            correct_priv = sum(1 for s in priv_sigs.values() if s == correct_door)
            signal_accuracy_priv.append(correct_priv / len(priv_sigs))

        # 累積統計
        stat_host_profit.append(host_rwd)
        stat_win_ratio.append(win_ratio)
        stat_total_bribe.append(total_bribe)
        stat_total_bet.append(total_bet)
        for pid in range(num_players):
            name = f"player_{pid}"
            stat_balances[name].append(balances[name])

        # ── 回合標頭 ────────────────────────────────────────────────────
        host_col = G if host_rwd >= 0 else R
        print(f"{B}{'─'*68}{RST}")
        print(
            f"{B}Round {round_idx+1:>3}/{num_rounds}{RST}  "
            f"Correct Door: {M}{DOOR_ICON[correct_door]}{RST}  "
            f"Public Signal: {C}{DOOR_ICON[pub_sig]}{RST}"
            f"{'✓' if pub_sig == correct_door else f'{DIM}✗{RST}'}  "
            f"Host Reward: {host_col}{host_rwd:+.1f}{RST}"
        )

        # ── 各門下注分布 ─────────────────────────────────────────────────
        door_totals = [0.0] * num_doors
        for name, bet in bets.items():
            d = door_choice.get(name, -1)
            if 0 <= d < num_doors:
                door_totals[d] += bet

        print(f"  {DIM}Betting Distribution (total pool: {total_bet:.1f}){RST}")
        for d in range(num_doors):
            ratio = door_totals[d] / total_bet if total_bet > 0 else 0.0
            is_correct = "← ✓ CORRECT" if d == correct_door else ""
            bar_col = G if d == correct_door else DIM
            print(
                f"    {DOOR_ICON[d]}  {bar_col}{_bar(door_totals[d], total_bet)}{RST} "
                f"{ratio*100:5.1f}%  ({door_totals[d]:6.1f})  {Y}{is_correct}{RST}"
            )

        # winning ratio 指示
        threshold_marker = int(round(0.25 * 20))
        ratio_bar = _bar(win_ratio, 1.0, width=20)
        ratio_col = G if win_ratio < 0.25 else R
        print(
            f"  Win-ratio x = {ratio_col}{win_ratio:.3f}{RST}  "
            f"[{ratio_col}{ratio_bar}{RST}]  "
            f"threshold θ=0.25  "
            f"{'(Host profits)' if win_ratio < 0.25 else f'{R}(Host loses){RST}'}"
        )

        # ── 玩家詳細表格 ─────────────────────────────────────────────────
        print(f"  {DIM}{'Player':<10} {'PrivSig':<9} {'Door':<8} {'Bribe':>7} {'Bet':>7} {'Balance':>8} {'Reward':>8}{RST}")
        for pid in range(num_players):
            name = f"player_{pid}"
            psig  = priv_sigs.get(name, -1)
            door  = door_choice.get(name, -1)
            bribe = bribe_amounts[name]
            bet   = bets.get(name, 0.0)
            bal   = balances[name]
            rwd   = rewards.get(name, 0.0)
            won   = (door == correct_door and bet > 0)

            psig_str = f"{DOOR_ICON[psig]}{'✓' if psig==correct_door else ' '}" if 0<=psig<num_doors else " -- "
            door_str = f"{DOOR_ICON[door]}{'✓' if won else ' '}" if 0<=door<num_doors else " -- "
            bal_col  = Y if bal <= env.welfare_amount * 2 else C

            print(
                f"  {name:<10} {psig_str:<9} {door_str:<8} "
                f"{bribe:>7.1f} {bet:>7.1f} "
                f"{bal_col}{bal:>8.1f}{RST} "
                f"{_color_reward(rwd):>8}"
            )

        print(f"  {DIM}Bribes total: {total_bribe:.1f}   Fee (5% pool): {total_bet*0.05:.1f}{RST}")

        # ── 可選：觀察向量摘要 ──────────────────────────────────────────
        if show_obs:
            obs = env.observe("player_0")
            valid = (obs != -1.0).sum()
            print(f"  {DIM}[obs] player_0 shape={obs.shape}  valid dims={valid}/{obs.shape[0]}  "
                  f"min={obs[obs!=-1].min():.3f}  max={obs.max():.3f}{RST}")

    # ── 全場統計摘要 ─────────────────────────────────────────────────────────
    print(f"\n{Y}{'═'*68}{RST}")
    print(f"{Y}  SIMULATION SUMMARY  ({num_rounds} rounds){RST}")
    print(f"{Y}{'═'*68}{RST}")

    avg_host   = float(np.mean(stat_host_profit))
    cum_host   = float(np.sum(stat_host_profit))
    avg_wr     = float(np.mean(stat_win_ratio))
    avg_bribe  = float(np.mean(stat_total_bribe))
    avg_pool   = float(np.mean(stat_total_bet))
    pub_acc    = float(np.mean(signal_accuracy_pub)) * 100
    priv_acc   = float(np.mean(signal_accuracy_priv)) * 100 if signal_accuracy_priv else 0.0

    print(f"\n  {M}【Host】{RST}")
    print(f"    Cumulative profit : {G if cum_host>=0 else R}{cum_host:+.1f}{RST}")
    print(f"    Avg profit/round  : {G if avg_host>=0 else R}{avg_host:+.1f}{RST}")
    print(f"    Pub-signal acc.   : {pub_acc:.1f}%  "
          f"{DIM}(% of rounds where public signal == correct door){RST}")
    print(f"    Priv-signal acc.  : {priv_acc:.1f}%  "
          f"{DIM}(avg % of players who got correct private signal){RST}")

    print(f"\n  {B}【Market】{RST}")
    print(f"    Avg win-ratio     : {avg_wr:.3f}  "
          f"{'(Host profitable zone)' if avg_wr < 0.25 else f'{R}(Host loss zone){RST}'}")
    print(f"    Avg bribe / round : {avg_bribe:.1f}")
    print(f"    Avg pool / round  : {avg_pool:.1f}")

    print(f"\n  {C}【Players — Final Balance】{RST}")
    init = 1000.0
    for pid in range(num_players):
        name = f"player_{pid}"
        final = stat_balances[name][-1] if stat_balances[name] else init
        delta = final - init
        bar = _bar(max(final, 0), init * 2, width=24)
        col = G if delta >= 0 else R
        print(f"    {name:<10}  {col}{bar}{RST}  {col}{final:>8.1f}  ({delta:+.1f}){RST}")

    print(f"\n  {DIM}Win-ratio per round:{RST}")
    wr_line = ""
    for wr in stat_win_ratio:
        col = G if wr < 0.25 else R
        wr_line += f"{col}{'▇' if wr < 0.25 else '▂'}{RST}"
    print(f"    {wr_line}")
    print(f"    {DIM}▇ = x<θ (Host profits)   ▂ = x≥θ (Host loses){RST}")
    print(f"\n{Y}{'═'*68}{RST}\n")


# ── CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OracleGambit Random Agent Watch")
    parser.add_argument("--rounds",  type=int, default=20,  help="回合數")
    parser.add_argument("--players", type=int, default=5,   help="玩家數")
    parser.add_argument("--doors",   type=int, default=4,   help="門數")
    parser.add_argument("--seed",    type=int, default=42,  help="隨機種子")
    parser.add_argument("--obs",     action="store_true",   help="顯示 player_0 的觀察向量摘要")
    args = parser.parse_args()

    run(
        num_players=args.players,
        num_doors=args.doors,
        num_rounds=args.rounds,
        seed=args.seed,
        show_obs=args.obs,
    )
