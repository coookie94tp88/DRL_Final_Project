"""ANSI terminal rendering for DooRL episodes (watch / demo)."""

from __future__ import annotations

import sys
from typing import Any, Dict, Optional

import numpy as np

from doorl.env.types import EnvConfig, LastSettlement

DOOR_LABELS = ("A", "B", "C", "D")

# ANSI colors
R = "\033[0m"
B = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GRN = "\033[32m"
YLW = "\033[33m"
BLU = "\033[34m"
MAG = "\033[35m"
CYN = "\033[36m"
GRY = "\033[90m"


def _supports_color() -> bool:
    return sys.stdout.isatty()


def _c(code: str, text: str, enabled: bool) -> str:
    return f"{code}{text}{R}" if enabled else text


def _door(d: int) -> str:
    return DOOR_LABELS[int(d) % 4]


def _mark(ok: bool, color: bool) -> str:
    return _c(GRN, "✓", color) if ok else _c(RED, "✗", color)


def _bar(fraction: float, width: int = 28, color: bool = True) -> str:
    fraction = float(np.clip(fraction, 0.0, 1.0))
    filled = int(round(fraction * width))
    body = "█" * filled + "░" * (width - filled)
    return _c(CYN, body, color)


def print_episode_header(
    *,
    title: str,
    num_players: int,
    tau: float,
    max_rounds: int,
    seed: Optional[int],
    ckpt: Optional[str],
    color: bool = True,
) -> None:
    line = "═" * 72
    print(_c(YLW, line, color))
    print(_c(YLW + B, f"  {title}", color))
    print(
        _c(
            GRY,
            f"  players={num_players}  doors=A–D  τ={tau:.2f}  max_rounds={max_rounds}"
            + (f"  seed={seed}" if seed is not None else "")
            + (f"\n  ckpt={ckpt}" if ckpt else ""),
            color,
        )
    )
    print(_c(YLW, line, color))


def print_settlement_fancy(
    s: LastSettlement,
    cfg: EnvConfig,
    *,
    balances: np.ndarray,
    host_cumulative: float,
    color: bool = True,
    lang: str = "en",
) -> None:
    """OracleGambit-style round panel."""
    n = cfg.num_players
    pub_ok = s.public_signal == s.true_door
    misled = int(
        (s.public_signal == s.true_door) and (s.private_signals != s.true_door).sum()
    )

    if lang == "zh":
        hdr = f"第 {s.round_idx + 1} 局結算"
        true_l = "正確門"
        pub_l = "公開信號"
        host_l = "主辦方本局"
        cum_l = "主辦方累計"
        alive_l = "有下注玩家"
    else:
        hdr = f"Round {s.round_idx + 1}/{cfg.max_rounds}"
        true_l = "True door"
        pub_l = "Public"
        host_l = "Host round PnL"
        cum_l = "Host cumulative"
        alive_l = "Players with bets"

    print()
    print(_c(BLU + B, f"━━ {hdr} ━━", color))
    print(
        f"  {_c(YLW, true_l, color)}: {_c(GRN + B, _door(s.true_door), color)}   "
        f"{_c(YLW, pub_l, color)}: {_door(s.public_signal)} {_mark(pub_ok, color)}   "
        f"x={s.x:.3f}  mult={s.multiplier:.2f}  pool={s.pool_p:.0f}"
    )
    print(
        f"  {_c(YLW, host_l, color)}: {s.reward_host:+.2f}   "
        f"{_c(YLW, cum_l, color)}: {host_cumulative:+.2f}   "
        f"{_c(YLW, alive_l, color)}: {int((s.bets > 0).sum())}/{n}"
    )

    # door traffic
    print(_c(GRY, "  Bet traffic by door:", color))
    for d in range(4):
        share = float(s.door_share[d])
        tag = _c(GRN + B, " ◀ WIN", color) if d == s.true_door else ""
        print(
            f"    {_door(d)} {_bar(share, color=color)} {share * 100:5.1f}%{tag}"
        )

    # tau line
    tau = cfg.payout_threshold
    x_color = GRN if s.x <= tau else RED
    print(
        f"  win-ratio x: {_c(x_color, f'{s.x:.3f}', color)}  "
        f"{_c(GRY, f'(τ={tau:.2f}; host profits when x < τ)', color)}"
    )

    # player table
    if lang == "zh":
        cols = ("玩家", "私人", "下注門", "賄%", "下注", "本局R", "餘額")
    else:
        cols = ("Player", "Priv", "Door", "Bribe%", "Bet", "R", "Bal")

    print(_c(GRY, "  " + "  ".join(f"{c:>8}" for c in cols), color))

    for i in range(n):
        priv_ok = int(s.private_signals[i]) == s.true_door
        door_ok = int(s.chosen_doors[i]) == s.true_door
        rp = float(s.rewards_player[i])
        r_col = GRN if rp > 0 else (RED if rp < 0 else GRY)
        bal = float(balances[i])
        inactive = _c(GRY, " (out)", color) if bal < cfg.min_bet else ""
        row = (
            f"  {f'P{i}':>8}"
            f"  {_door(int(s.private_signals[i]))}{_mark(priv_ok, color):>3}"
            f"  {_door(int(s.chosen_doors[i]))}{_mark(door_ok, color):>3}"
            f"  {s.bribe_pcts[i] * 100:7.1f}%"
            f"  {s.bets[i]:8.0f}"
            f"  {_c(r_col, f'{rp:+8.1f}', color)}"
            f"  {bal:8.0f}{inactive}"
        )
        print(row)

    bribe_sum = float(s.bribes.sum())
    print(
        _c(
            GRY,
            f"  bribes total: {bribe_sum:.1f}  |  pool P: {s.pool_p:.0f}  |  "
            f"misled-by-host (pub✓ priv✗ count): {misled}",
            color,
        )
    )


def print_settlement_compact(
    s: LastSettlement,
    cfg: EnvConfig,
    *,
    balances: np.ndarray,
    host_cumulative: float,
    color: bool = True,
    lang: str = "en",
) -> str:
    """One-line round summary for in-training logs. Returns the line printed."""
    n = cfg.num_players
    alive = int(np.sum(balances >= cfg.min_bet))
    pub_ok = s.public_signal == s.true_door
    priv_ok = int((s.private_signals == s.true_door).sum())
    picks = "".join(_door(int(s.chosen_doors[i])) for i in range(n))
    if lang == "zh":
        line = (
            f"  第{s.round_idx + 1}局: 真={_door(s.true_door)} 公開={_door(s.public_signal)}"
            f"{_mark(pub_ok, color)} x={s.x:.2f} 主辦={s.reward_host:+.0f} "
            f"存活{alive}/{n} 私人對{priv_ok}/{n} 選門[{picks}]"
        )
    else:
        line = (
            f"  R{s.round_idx + 1}: true={_door(s.true_door)} pub={_door(s.public_signal)}"
            f"{_mark(pub_ok, color)} x={s.x:.2f} host={s.reward_host:+.0f} "
            f"alive={alive}/{n} priv_ok={priv_ok}/{n} picks=[{picks}]"
        )
    return line


def print_settlement_plain(
    s: LastSettlement,
    cfg: EnvConfig,
    *,
    balances: np.ndarray,
    host_cumulative: float,
    lang: str = "zh",
) -> None:
    """Simple block style (like the user's pasted log)."""
    alive = int(np.sum(balances >= cfg.min_bet))
    if lang == "zh":
        print("-" * 40)
        print(f"Round {s.round_idx + 1} 結算:")
        print(f"  👉 正確門牌: {_door(s.true_door)} ({s.true_door})")
        print(f"  👉 公開信號: {_door(s.public_signal)}")
        print(f"  👉 存活玩家數: {alive} / {cfg.num_players}")
        print(f"  👉 主辦方本局獲利: {s.reward_host:.2f}")
        print(f"  👉 玩家 0 的本局利潤: {s.rewards_player[0]:.2f}")
        print(f"  📊 主辦方累計淨利: {host_cumulative:.2f}")
        print(f"  💰 玩家餘額: {np.array2string(balances, precision=0)}")
    else:
        print("-" * 40)
        print(f"Round {s.round_idx + 1} settled:")
        print(f"  true door: {_door(s.true_door)}")
        print(f"  public: {_door(s.public_signal)}")
        print(f"  alive: {alive}/{cfg.num_players}")
        print(f"  host round: {s.reward_host:+.2f}  cumulative: {host_cumulative:+.2f}")
        print(f"  balances: {balances}")


def print_episode_footer(
    *,
    host_total: float,
    balances: np.ndarray,
    cfg: EnvConfig,
    lang: str = "en",
    color: bool = True,
) -> None:
    alive = int(np.sum(balances >= cfg.min_bet))
    print()
    print(_c(YLW, "═" * 72, color))
    if lang == "zh":
        print(_c(B + YLW, "=== 遊戲結束 ===", color))
        print(f"最終主辦方累計利潤: {host_total:.2f}")
        if alive == 0:
            print(_c(RED, "所有玩家皆已破產！主辦方大獲全勝。", color))
        else:
            print(f"仍有 {alive} 名玩家存活。")
    else:
        print(_c(B + YLW, "=== Episode over ===", color))
        print(f"Host total profit: {host_total:+.2f}  |  players alive: {alive}")
    print(_c(YLW, "═" * 72, color))
