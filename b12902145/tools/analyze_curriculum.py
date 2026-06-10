"""
詳細分析 curriculum_log.csv 並產生多面向圖表。
用法: python tools/analyze_curriculum.py checkpoints/curriculum/curriculum_log.csv
"""
import csv, sys, os, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── 讀資料 ────────────────────────────────────────────────
log_path = sys.argv[1] if len(sys.argv) > 1 else \
    "checkpoints/curriculum/curriculum_log.csv"
out_dir = os.path.dirname(log_path)

rows = list(csv.DictReader(open(log_path)))

def col(rows, key):
    return np.array([float(r[key]) for r in rows])

def moving_avg(x, w=20):
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")

all_rounds  = col(rows, "round")
all_phase   = [r["phase"] for r in rows]

phase_mask = {p: np.array([r["phase"] == p for r in rows]) for p in "ABC"}
c_rows  = [r for r in rows if r["phase"] == "C"]
c_round = col(c_rows, "round")

def c(key):
    return col(c_rows, key)

# ── 計算 cross-correlation: honesty vs follow_rate (lagged) ──
def xcorr_lag(x, y, max_lag=30):
    x = (x - x.mean()) / (x.std() + 1e-9)
    y = (y - y.mean()) / (y.std() + 1e-9)
    lags, corrs = [], []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            corrs.append(np.mean(x[:len(x)-lag] * y[lag:]))
        else:
            corrs.append(np.mean(x[-lag:] * y[:len(y)+lag]))
        lags.append(lag)
    return np.array(lags), np.array(corrs)

hon = c("signal_honesty")
fol = c("follow_rate")
wr  = c("win_ratio")
hr  = c("host_reward")
pr  = c("player_reward")
h_ent = c("host_entropy")
p_ent = c("player_entropy")

lags, xcorr = xcorr_lag(hon, fol, max_lag=40)

W = 25  # moving avg window
RANDOM_BASE = 1/4   # 4 doors

# ── 圖1: Phase C 詳細走勢 + moving average ───────────────
fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
fig.suptitle("Phase C Deep Dive — OracleGambit (pre-balance)", fontsize=14, fontweight="bold")

ax = axes[0]
ax.plot(c_round, hon, alpha=0.3, color="steelblue", lw=0.8)
ax.plot(c_round, moving_avg(hon, W), color="steelblue", lw=1.8, label="honesty (MA)")
ax.plot(c_round, fol, alpha=0.3, color="tomato", lw=0.8)
ax.plot(c_round, moving_avg(fol, W), color="tomato", lw=1.8, label="follow rate (MA)")
ax.axhline(RANDOM_BASE, color="gray", ls="--", lw=1, label=f"random={RANDOM_BASE:.2f}")
ax.set_ylabel("Rate")
ax.set_title("Signal Honesty vs Follow Rate")
ax.legend(loc="upper right", fontsize=8)
ax.set_ylim(-0.05, 0.75)

ax = axes[1]
ax.plot(c_round, h_ent, alpha=0.3, color="darkorange", lw=0.8)
ax.plot(c_round, moving_avg(h_ent, W), color="darkorange", lw=1.8, label="host entropy (MA)")
ax.plot(c_round, p_ent, alpha=0.3, color="mediumseagreen", lw=0.8)
ax.plot(c_round, moving_avg(p_ent, W), color="mediumseagreen", lw=1.8, label="player entropy (MA)")
ln4 = math.log(4)
ax.axhline(ln4, color="gray", ls="--", lw=1, label=f"ln(4)={ln4:.3f} (max entropy)")
ax.set_ylabel("Entropy (nats)")
ax.set_title("Policy Entropy — Host & Player")
ax.legend(loc="lower right", fontsize=8)

ax = axes[2]
ax.plot(c_round, wr, alpha=0.3, color="purple", lw=0.8)
ax.plot(c_round, moving_avg(wr, W), color="purple", lw=1.8, label="win ratio (MA)")
ax.axhline(RANDOM_BASE, color="gray", ls="--", lw=1)
ax.set_ylabel("Win Ratio")
ax.set_title("Player Win Ratio")
ax.legend(fontsize=8)
ax.set_ylim(0.1, 0.4)

ax = axes[3]
ax.plot(c_round, hr, alpha=0.3, color="firebrick", lw=0.8)
ax.plot(c_round, moving_avg(hr, W), color="firebrick", lw=1.8, label="host reward (MA)")
ax.plot(c_round, pr, alpha=0.3, color="dodgerblue", lw=0.8)
ax.plot(c_round, moving_avg(pr, W), color="dodgerblue", lw=1.8, label="player reward (MA)")
ax.axhline(0, color="gray", ls="--", lw=1)
ax.set_ylabel("Reward")
ax.set_title("Host & Player Reward")
ax.legend(fontsize=8)

ax = axes[4]
# honesty - follow_rate difference: negative = players follow more than host is honest
diff = fol - hon
ax.plot(c_round, diff, alpha=0.3, color="slategray", lw=0.8)
ax.plot(c_round, moving_avg(diff, W), color="slategray", lw=1.8, label="follow - honesty (MA)")
ax.axhline(0, color="gray", ls="--", lw=1, label="0 (balanced)")
ax.set_ylabel("Δ Rate")
ax.set_title("Follow Rate − Honesty  (+ = over-trusting, − = under-trusting)")
ax.legend(fontsize=8)
ax.set_xlabel("Round")

plt.tight_layout()
out1 = os.path.join(out_dir, "phaseC_detail.png")
plt.savefig(out1, dpi=130)
plt.close()
print(f"Saved → {out1}")

# ── 圖2: Cross-correlation + Entropy timeline ──────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("Host vs Player Dynamics Analysis", fontsize=13, fontweight="bold")

ax = axes[0]
ax.bar(lags, xcorr, color=["tomato" if x > 0 else "steelblue" for x in xcorr],
       width=0.8, alpha=0.8)
ax.axvline(0, color="black", lw=1)
ax.axhline(0, color="gray", lw=0.5)
best_lag = lags[np.argmax(xcorr)]
ax.axvline(best_lag, color="red", ls="--", lw=1.5, label=f"peak lag={best_lag}")
ax.set_xlabel("Lag (log steps, + = player lags host)")
ax.set_ylabel("Cross-correlation")
ax.set_title("Cross-correlation: Honesty → Follow Rate\n(positive lag = player reacts AFTER host changes)")
ax.legend(fontsize=8)

ax = axes[1]
# scatter: honesty vs follow_rate coloured by time
sc = ax.scatter(hon, fol, c=c_round, cmap="viridis", s=6, alpha=0.6)
plt.colorbar(sc, ax=ax, label="Round")
ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="follow = honesty")
ax.axvline(RANDOM_BASE, color="gray", ls=":", lw=1)
ax.axhline(RANDOM_BASE, color="gray", ls=":", lw=1)
ax.set_xlabel("Signal Honesty")
ax.set_ylabel("Follow Rate")
ax.set_title("Follow Rate vs Honesty\n(colour = training progress)")
ax.legend(fontsize=8)

plt.tight_layout()
out2 = os.path.join(out_dir, "dynamics_analysis.png")
plt.savefig(out2, dpi=130)
plt.close()
print(f"Saved → {out2}")

# ── 圖3: Entropy convergence 放大 ─────────────────────────
# 分段計算 moving avg 標準差 (local variance)
def rolling_std(x, w=20):
    result = np.full_like(x, np.nan)
    for i in range(w, len(x)):
        result[i] = np.std(x[i-w:i])
    return result

fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
fig.suptitle("Convergence Speed: Host vs Player", fontsize=13, fontweight="bold")

ax = axes[0]
ax.plot(c_round, moving_avg(h_ent, W), color="darkorange", lw=2, label="host entropy")
ax.plot(c_round, moving_avg(p_ent, W), color="seagreen", lw=2, label="player entropy")
ax.axhline(ln4, color="gray", ls="--", lw=1, label=f"max entropy ln(4)={ln4:.3f}")
ax.set_ylabel("Entropy")
ax.set_title("Entropy over Phase C (smoothed)")
ax.legend()

ax = axes[1]
ax.plot(c_round, rolling_std(h_ent, W), color="darkorange", lw=1.5, label="host entropy stdev")
ax.plot(c_round, rolling_std(fol, W), color="tomato", lw=1.5, label="follow rate stdev")
ax.plot(c_round, rolling_std(hon, W), color="steelblue", lw=1.5, label="honesty stdev")
ax.set_ylabel("Rolling Std (w=20)")
ax.set_xlabel("Round")
ax.set_title("Local Volatility — lower = more stable/converged")
ax.legend()

plt.tight_layout()
out3 = os.path.join(out_dir, "convergence_speed.png")
plt.savefig(out3, dpi=130)
plt.close()
print(f"Saved → {out3}")

# ── 終端統計摘要 ──────────────────────────────────────────
print()
print("=== Phase C Summary ===")
n = len(c_rows)
for key, arr, fmt in [
    ("honesty",     hon,   ".3f"),
    ("follow_rate", fol,   ".3f"),
    ("win_ratio",   wr,    ".3f"),
    ("host_entropy",h_ent, ".3f"),
    ("player_entropy",p_ent,".3f"),
]:
    early = arr[:n//5]
    late  = arr[4*n//5:]
    print(f"  {key:<18} early_mean={np.mean(early):{fmt}}  late_mean={np.mean(late):{fmt}}  overall_std={np.std(arr):{fmt}}")

print(f"\nCross-corr peak at lag={best_lag} (log-steps) → {'player lags host' if best_lag > 0 else 'host lags player'}")
print(f"Total Phase C log steps: {n}  ({c_round[0]:.0f} → {c_round[-1]:.0f})")
