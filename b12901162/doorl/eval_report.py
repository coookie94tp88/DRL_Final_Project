"""Human-readable eval summaries for quick judgment."""

from __future__ import annotations

from typing import Any, Dict, Optional

from doorl.metrics import MetricsSummary


def _verdict(
    summary: MetricsSummary,
    tau: float,
    initial_balance: float,
    lang: str = "zh",
) -> tuple[str, str]:
    """Return (emoji+label, one_line_reason)."""
    if lang == "zh":
        issues = []
        if summary.bankruptcy_rate >= 0.75:
            issues.append("玩家幾乎全破產")
        if summary.mean_player_balance < 0.25 * initial_balance:
            issues.append("餘額崩潰")
        if summary.mi_private_truth < 0.08 and summary.mi_public_truth < 0.08:
            issues.append("信號幾乎沒訊息")
        if summary.mean_R_host <= 0:
            issues.append("Host 不賺錢")
        good = []
        if summary.mean_R_host > 0:
            good.append("Host 有賺")
        if summary.mi_private_truth > 0.15 or summary.mi_public_truth > 0.15:
            good.append("信號有用")
        if summary.bankruptcy_rate < 0.4:
            good.append("玩家存活多")
        if 0.5 * tau <= summary.median_x <= 1.5 * tau:
            good.append("池子競爭正常")
        if len(issues) >= 2:
            return "🔴 偏差", "；".join(issues[:3])
        if issues and not good:
            return "🔴 偏差", issues[0]
        if issues and good:
            return "🟡 普通", f"{'、'.join(good[:2])}，但 {issues[0]}"
        if good:
            return "🟢 尚可", "、".join(good[:3])
        return "🟡 普通", "好壞參半"

    issues = []
    if summary.bankruptcy_rate >= 0.75:
        issues.append("players mostly bankrupt")
    if summary.mean_player_balance < 0.25 * initial_balance:
        issues.append("balances collapsed")
    if summary.mi_private_truth < 0.08 and summary.mi_public_truth < 0.08:
        issues.append("signals carry almost no info")
    if summary.mean_R_host <= 0:
        issues.append("host not profitable")
    good = []
    if summary.mean_R_host > 0:
        good.append("host earns")
    if summary.mi_private_truth > 0.15 or summary.mi_public_truth > 0.15:
        good.append("some signal value")
    if summary.bankruptcy_rate < 0.4:
        good.append("many players survive")
    if 0.5 * tau <= summary.median_x <= 1.5 * tau:
        good.append("pool competitive (x≈τ)")
    if len(issues) >= 2:
        return "🔴 Poor", "; ".join(issues[:3])
    if issues and not good:
        return "🔴 Poor", issues[0]
    if issues and good:
        return "🟡 Mixed", f"{', '.join(good[:2])}, but {issues[0]}"
    if good:
        return "🟢 OK", ", ".join(good[:3])
    return "🟡 Mixed", "mixed signals"


def print_eval_report(
    summary: MetricsSummary,
    accept: Dict[str, bool],
    *,
    episodes: int,
    tau: float,
    initial_balance: float,
    num_players: int,
    max_rounds: int,
    ckpt: Optional[str] = None,
    baseline: Optional[str] = None,
    lang: str = "zh",
    cross_play: bool = False,
) -> None:
    label, reason = _verdict(summary, tau, initial_balance, lang=lang)

    if lang == "zh":
        _print_zh(
            summary,
            accept,
            episodes=episodes,
            tau=tau,
            initial_balance=initial_balance,
            num_players=num_players,
            max_rounds=max_rounds,
            ckpt=ckpt,
            baseline=baseline,
            verdict_label=label,
            verdict_reason=reason,
            cross_play=cross_play,
        )
    else:
        _print_en(
            summary,
            accept,
            episodes=episodes,
            tau=tau,
            initial_balance=initial_balance,
            num_players=num_players,
            max_rounds=max_rounds,
            ckpt=ckpt,
            baseline=baseline,
            verdict_label=label,
            verdict_reason=reason,
            cross_play=cross_play,
        )


def _print_zh(
    s: MetricsSummary,
    accept: Dict[str, bool],
    **meta: Any,
) -> None:
    src = meta.get("baseline") or meta.get("ckpt") or "config"
    print("═" * 56)
    print(f"  DooRL 評估報告")
    print(
        f"  {meta['episodes']} 局 × {meta['num_players']} 玩家 | "
        f"每局最多 {meta['max_rounds']} 輪 | τ={meta['tau']:.2f}"
    )
    print(f"  來源: {src}")
    if meta.get("cross_play"):
        print("  （交叉對戰：Host / Player 來自不同 checkpoint）")
    print("═" * 56)
    print()
    print(f"  【整體】{meta['verdict_label']}  —  {meta['verdict_reason']}")
    print()
    print("  【快速判斷 — 看這 6 行就夠】")
    _line_zh("主辦方賺不賺", s.mean_R_host > 0, f"平均每局 Host 利潤 {s.mean_R_host:+.0f}")
    _line_zh(
        "玩家活下來嗎",
        s.bankruptcy_rate < 0.5,
        f"破產率 {s.bankruptcy_rate * 100:.0f}%（局末沒錢下注的人）",
    )
    _line_zh(
        "玩家還有錢嗎",
        s.mean_player_balance > 0.5 * meta["initial_balance"],
        f"平均餘額 {s.mean_player_balance:.0f} / 初始 {meta['initial_balance']:.0f}",
    )
    _line_zh(
        "有在賄賂嗎 (H1)",
        s.mean_bribe_pct > 0.03,
        f"平均賄賂比例 {s.mean_bribe_pct * 100:.1f}%",
    )
    _line_zh(
        "私人提示有用嗎",
        s.mi_private_truth > 0.08,
        f"MI(私人,真門)={s.mi_private_truth:.3f} 比特（>0.1 較好，~0 等於亂講）",
    )
    _line_zh(
        "公開提示有用嗎",
        s.mi_public_truth > 0.08,
        f"MI(公開,真門)={s.mi_public_truth:.3f} 比特",
    )
    print()
    print("  【進階】")
    print(f"    · 池子競爭度 median(x)={s.median_x:.2f}（目標接近 τ={meta['tau']:.2f}）")
    print(f"    · 跟私人門比例 {s.p_follow_private * 100:.0f}% | 跟公開門 {s.p_follow_public * 100:.0f}%")
    print(
        f"    · 公開對但私人錯的比例 {s.public_true_private_false_rate * 100:.0f}%"
        "（高=欺騙型 Host）"
    )
    print(f"    · 共統計 {s.rounds_collected} 個結算輪次（跨所有 eval 局）")
    print()
    print("  【課題達標？(spec 參考)】")
    for key, ok in accept.items():
        print(f"    {'✓' if ok else '✗'} {_accept_label_zh(key)}")
    print()
    print("  【怎麼解讀你這次的數字】")
    if s.bankruptcy_rate >= 0.9 and s.mean_R_host > 0:
        print("    → Host 把玩家「榨乾」型；常見於早期訓練或舊 bug 環境的 ckpt。")
    if s.mi_public_truth < 0.02:
        print("    → 公開信號和真門幾乎無關（或 Host 固定喊同一扇門）。")
    if "learn_4p/" in str(meta.get("ckpt", "")) and "v2" not in str(meta.get("ckpt", "")):
        print("    → 若這是 learn_4p（非 v2），建議改用 learn_4p_v2 的 ckpt 再 eval。")
    print()
    print("  原始 JSON：加 --json")
    print("═" * 56)


def _print_en(
    s: MetricsSummary,
    accept: Dict[str, bool],
    **meta: Any,
) -> None:
    src = meta.get("baseline") or meta.get("ckpt") or "config"
    print("=" * 56)
    print("  DooRL eval report")
    print(
        f"  {meta['episodes']} episodes × {meta['num_players']} players | "
        f"max {meta['max_rounds']} rounds | tau={meta['tau']:.2f}"
    )
    print(f"  source: {src}")
    print("=" * 56)
    print(f"\n  VERDICT: {meta['verdict_label']} — {meta['verdict_reason']}\n")
    _line_zh("Host profit", s.mean_R_host > 0, f"mean R_host {s.mean_R_host:+.0f}")
    _line_zh(
        "Players survive",
        s.bankruptcy_rate < 0.5,
        f"bankruptcy {s.bankruptcy_rate * 100:.0f}%",
    )
    print("\n  Use --json for raw output.\n")


def _line_zh(title: str, ok: bool, detail: str) -> None:
    mark = "✓" if ok else "✗"
    print(f"    {mark} {title} — {detail}")


def _accept_label_zh(key: str) -> str:
    labels = {
        "host_profit_positive": "Host 平均賺錢",
        "player_balance_half_initial": "玩家平均餘額 > 初始一半",
        "median_x_near_tau": "median(x) 接近 τ（池子有競爭）",
        "h2_private_beats_public_mi": "私人 MI > 公開 MI",
        "h4_pub_true_priv_false_low": "欺騙率 < 5%",
    }
    return labels.get(key, key)
