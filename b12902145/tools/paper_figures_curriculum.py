"""
三張論文圖 (OracleGambit pre-balance, MLP+REINFORCE):
  fig_overview        - 全程三 phase 概觀
  fig_honesty_follow  - Phase C: Signal Honesty vs Follow Rate
  fig_delta           - Phase C: Follow Rate - Honesty 震盪與收斂

用法: python tools/paper_figures_curriculum.py [curriculum_log.csv]
"""
import csv, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "legend.framealpha": 0.88,
    "lines.linewidth": 1.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.22, "grid.linewidth": 0.55,
})
SAVE_DPI = 300
RANDOM   = 0.25
C_HON = "#2166ac"; C_FOL = "#d6604d"; C_WR = "#5aae61"; C_RND = "#888888"
PHASE_COLOR = {"A": "#d1e5f0", "B": "#fddbc7", "C": "#e8f4e8"}
PHASE_LABEL = {
    "A": "Phase A\n(Honest Host,\nPlayers Only)",
    "B": "Phase B\n(Deceptive Host,\nHost Only)",
    "C": "Phase C\n(Joint Training)",
}

log_path = sys.argv[1] if len(sys.argv) > 1 else \
    "checkpoints/curriculum/curriculum_log.csv"
out_dir = os.path.dirname(log_path)
all_rows = list(csv.DictReader(open(log_path)))

def col(rs, k):
    return np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan
                     for r in rs])

def ma(x, w):
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(len(x)):
        lo, hi = max(0, i-w//2), min(len(x), i+w//2+1)
        v = x[lo:hi]; v = v[~np.isnan(v)]
        if len(v): out[i] = v.mean()
    return out

def save(fig, name):
    for ext in (".pdf", ".png"):
        p = os.path.join(out_dir, name + ext)
        fig.savefig(p, dpi=SAVE_DPI, bbox_inches="tight")
        print(f"Saved -> {p}")
    plt.close(fig)

def shade_phases(ax, bounds):
    for p, (lo, hi) in bounds.items():
        ax.axvspan(lo, hi, color=PHASE_COLOR[p], alpha=0.50, zorder=0)
    for p in ["B", "C"]:
        if p in bounds:
            ax.axvline(bounds[p][0], color="#aaaaaa", lw=0.9, ls="--", zorder=1)

# ── 計算 phase 邊界 ───────────────────────────────────────────
phase_bounds = {}
for p in "ABC":
    pr = [float(r["round"]) for r in all_rows if r["phase"] == p]
    if pr: phase_bounds[p] = (pr[0], pr[-1])

all_rounds = col(all_rows, "round")
all_hon    = col(all_rows, "signal_honesty")
all_fol    = col(all_rows, "follow_rate")
all_wr     = col(all_rows, "win_ratio")

# ══════════════════════════════════════════════════════════════
# 圖 1 ── 全程三 phase 概觀（2 subplots + phase strip）
# ══════════════════════════════════════════════════════════════
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(7, 4.4))
gs  = gridspec.GridSpec(2, 1, figure=fig,
                        height_ratios=[1, 1],
                        hspace=0.08)
ax_hon = fig.add_subplot(gs[0])
ax_fol = fig.add_subplot(gs[1], sharex=ax_hon)

x_min, x_max = all_rounds[0], all_rounds[-1]

# -- 說明框（Phase 色塊 + 曲線說明，放在 ax_hon）--------------
import matplotlib.patches as mpatches
legend_handles = [
    mpatches.Patch(color=PHASE_COLOR["A"], label="A: Honest Host, Players Only"),
    mpatches.Patch(color=PHASE_COLOR["B"], label="B: Deceptive Host, Host Only"),
    mpatches.Patch(color=PHASE_COLOR["C"], label="C: Joint Training"),
    plt.Line2D([0],[0], color=C_HON, lw=1.9, label="Signal Honesty"),
    plt.Line2D([0],[0], color=C_RND, ls="--", lw=1.1, label=f"Random ({RANDOM})"),
]

# -- 兩個主 subplot -------------------------------------------
for ax, arr, color, label, ylim in [
    (ax_hon, all_hon, C_HON, "Signal Honesty", (-0.05, 1.12)),
    (ax_fol, all_fol, C_FOL, "Follow Rate",    (-0.05, 0.87)),
]:
    shade_phases(ax, phase_bounds)
    for p in ["B", "C"]:
        if p in phase_bounds:
            ax.axvline(phase_bounds[p][0], color="#aaaaaa", lw=0.9, ls="--", zorder=1)
    ax.plot(all_rounds, arr,        color=color, lw=1.1, alpha=0.45)
    ax.plot(all_rounds, ma(arr, 5), color=color, lw=1.9)
    ax.axhline(RANDOM, color=C_RND, ls="--", lw=1.1)
    ax.set_ylabel(label); ax.set_ylim(ylim)

# 分別設 legend
ax_hon.legend(handles=legend_handles, loc="upper right", fontsize=8)
ax_fol.legend(handles=[
    plt.Line2D([0],[0], color=C_FOL, lw=1.9, label="Follow Rate"),
    plt.Line2D([0],[0], color=C_RND, ls="--", lw=1.1, label=f"Random ({RANDOM})"),
], loc="upper right", fontsize=8)

plt.setp(ax_hon.get_xticklabels(), visible=False)
ax_fol.xaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v, _: f"{v/1e5:.1f}"))
ax_fol.set_xlabel("Training Round (x10^5)")
fig.subplots_adjust(top=0.97, bottom=0.11, left=0.11, right=0.97, hspace=0.10)
save(fig, "fig_overview")

# ── Phase C 資料 ──────────────────────────────────────────────
c_rows  = [r for r in all_rows if r["phase"] == "C"]
c_round = col(c_rows, "round")
hon     = col(c_rows, "signal_honesty")
fol     = col(c_rows, "follow_rate")
W_FAST, W_SLOW = 15, 50
hon_s = ma(hon, W_FAST); fol_s = ma(fol, W_FAST)

# ══════════════════════════════════════════════════════════════
# 圖 2 ── Phase C: Signal Honesty vs Follow Rate
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.plot(c_round, hon,   color=C_HON, alpha=0.22, lw=0.9)
ax.plot(c_round, fol,   color=C_FOL, alpha=0.22, lw=0.9)
ax.plot(c_round, hon_s, color=C_HON, lw=1.9, label="Signal Honesty")
ax.plot(c_round, fol_s, color=C_FOL, lw=1.9, label="Follow Rate")
ax.axhline(RANDOM, color=C_RND, ls="--", lw=1.1,
           label=f"Random baseline ({RANDOM})")
ax.fill_between(c_round, hon_s, fol_s, where=(fol_s > hon_s),
                color=C_FOL, alpha=0.13, label="Over-trusting (follow > honesty)")
ax.fill_between(c_round, hon_s, fol_s, where=(fol_s < hon_s),
                color=C_HON, alpha=0.13, label="Under-trusting (follow < honesty)")
ax.set_xlim(c_round[0], c_round[-1]); ax.set_ylim(-0.02, 0.65)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e5:.1f}"))
ax.set_xlabel("Training Round (x10^5)"); ax.set_ylabel("Rate")
ax.legend(loc="upper right", ncol=2, fontsize=8.5)
fig.tight_layout()
save(fig, "fig_honesty_follow")

# ══════════════════════════════════════════════════════════════
# 圖 3 ── Phase C: Follow Rate - Honesty
# ══════════════════════════════════════════════════════════════
diff = fol - hon
diff_ma   = ma(diff, W_FAST)
diff_slow = ma(diff, W_SLOW)

CONV_THR = 0.03; conv_idx = len(diff_slow) - 1
for i in range(len(diff_slow)-1, 0, -1):
    if abs(diff_slow[i]) > CONV_THR: conv_idx = i; break
conv_round = c_round[conv_idx]

fig, ax = plt.subplots(figsize=(6.5, 3.0))
ax.axhline(0, color="#333333", lw=1.3, ls="--", zorder=3,
           label="Delta = 0 (balanced)")
ax.fill_between(c_round, diff_ma, 0, where=(diff_ma > 0),
                color=C_FOL, alpha=0.28, label="Over-trusting (Delta > 0)")
ax.fill_between(c_round, diff_ma, 0, where=(diff_ma < 0),
                color=C_HON, alpha=0.28, label="Under-trusting (Delta < 0)")
ax.plot(c_round, diff,      color="#888888", alpha=0.18, lw=0.8)
ax.plot(c_round, diff_ma,   color="#444444", lw=1.6,
        label=f"Delta (MA-{W_FAST})", zorder=4)
ax.plot(c_round, diff_slow, color="#222222", lw=1.2, ls=":",
        label=f"Trend (MA-{W_SLOW})", zorder=5)

early_end = c_round[len(c_round) // 4]
Y_BOT = -0.33
ax.axvspan(c_round[0], early_end, color="#f7c59f", alpha=0.22, zorder=0)
ax.text((c_round[0]+early_end)/2, Y_BOT, "Adversarial\nPhase",
        ha="center", va="bottom", fontsize=8, color="#b5451b", style="italic")
ax.axvspan(conv_round, c_round[-1], color="#cce5ff", alpha=0.25, zorder=0)
ax.text((conv_round+c_round[-1])/2, Y_BOT, "Nash\nEquilibrium",
        ha="center", va="bottom", fontsize=8, color="#1a4e8c", style="italic")

ax.set_xlim(c_round[0], c_round[-1])
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e5:.1f}"))
ax.set_xlabel("Training Round (x10^5)")
ax.set_ylabel("Follow Rate - Honesty (Delta)")
ax.legend(loc="upper right", ncol=2, fontsize=8.5)
fig.tight_layout()
save(fig, "fig_delta")
