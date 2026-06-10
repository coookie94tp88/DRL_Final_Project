"""
Curriculum Training Entry Point
================================
Phase A (Honest Host)  → Phase B (Adversarial)  → Phase C (Joint)

Story arc:
  A: Players randomly guess (25% wr) → learn to follow honest signal (wr↑)
  B: Host exploits trusting Players  → win rate drops, honesty collapses
  C: Joint arms-race                 → signal honesty oscillates

Usage examples
--------------
  # Default run (200k rounds total: A=50k + B=50k + C=100k)
  python experiments/run_curriculum.py

  # Quick smoke test
  python experiments/run_curriculum.py --rounds_a 1024 --rounds_b 1024 --rounds_c 1024 --log_interval 512 --spotlight_interval 512

  # Longer run to fully observe story arc
  python experiments/run_curriculum.py --rounds_a 100000 --rounds_b 100000 --rounds_c 200000

  # Adjust phase transition thresholds
  python experiments/run_curriculum.py --target_follow 0.70 --target_honesty 0.30
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from env.oracle_gambit_env import OracleGambitEnv
from agents.mlp_agent import MlpAgent
from training.curriculum_runner import CurriculumRunner


# ---------------------------------------------------------------------------
# Log tee — mirror stdout to a file (ANSI codes preserved in both)
# ---------------------------------------------------------------------------

class _LogTee:
    """Duplicate sys.stdout to a file so the terminal and a log file are in sync.

    ANSI escape codes are written as-is, so the file can be replayed with::

        cat terminal.log          # colours shown in any colour-capable terminal
        less -R terminal.log      # colours + scrolling
    """

    def __init__(self, path: str) -> None:
        self._path   = path
        self._file   = open(path, "w", encoding="utf-8")  # noqa: WPS515
        self._stdout = sys.stdout
        sys.stdout   = self

    def write(self, data: str) -> None:
        self._stdout.write(data)
        self._file.write(data)

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()

    def fileno(self) -> int:          # needed by some libraries
        return self._stdout.fileno()

    def close(self) -> None:
        sys.stdout = self._stdout
        self._file.close()
        self._stdout.write(f"\nTerminal log → {self._path}\n")
        self._stdout.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    _tee = _LogTee(os.path.join(args.save_dir, "terminal.log"))

    env = OracleGambitEnv(
        num_players=args.players,
        num_doors=4,
        payout_threshold=0.20,
        history_window=50,
        max_rounds=0,
        seed=args.seed,
    )
    env.reset(seed=args.seed)

    D              = env.num_doors
    host_obs_dim   = env.observation_space("host").shape[0]
    player_obs_dim = env.observation_space("player_0").shape[0]

    host_agent   = MlpAgent(host_obs_dim,   D, hidden_dims=(256, 128))
    player_agent = MlpAgent(player_obs_dim, D, hidden_dims=(256, 128))

    total = args.rounds_a + args.rounds_b + args.rounds_c
    print("=" * 64)
    print("OracleGambit — Curriculum Training  (3-Phase)")
    print("=" * 64)
    print(f"  Players          : {args.players}")
    print(f"  Doors            : {D}")
    print(f"  Host obs_dim     : {host_obs_dim}")
    print(f"  Player obs_dim   : {player_obs_dim}")
    print(f"  Phase A rounds   : {args.rounds_a:,}  "
          f"(early stop: fol >= {args.target_follow:.2f})")
    print(f"  Phase B rounds   : {args.rounds_b:,}  "
          f"(early stop: hon <= {args.target_honesty:.2f})")
    print(f"  Phase C rounds   : {args.rounds_c:,}  (joint, no early stop)")
    print(f"  Total max rounds : {total:,}")
    print(f"  Batch size       : {args.batch_size}")
    print(f"  LR host/player   : {args.lr_host} / {args.lr_player}")
    print(f"  Entropy coeff    : {args.entropy_coeff}")
    print("=" * 64)

    runner = CurriculumRunner(
        env,
        host_agent,
        player_agent,
        lr_host=args.lr_host,
        lr_player=args.lr_player,
        entropy_coeff=args.entropy_coeff,
    )

    log = runner.run_curriculum(
        rounds_a=args.rounds_a,
        rounds_b=args.rounds_b,
        rounds_c=args.rounds_c,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
        spotlight_interval=args.spotlight_interval,
        target_follow_rate_a=args.target_follow,
        target_honesty_b=args.target_honesty,
        target_entropy_b=args.target_entropy_b,
        min_rounds_b=args.min_rounds_b,
        save_dir=args.save_dir,
    )

    if log:
        _plot_curriculum(log, args.save_dir)
    _tee.close()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_curriculum(log: list[dict], save_dir: str) -> None:
    """
    Save a 3-panel figure showing win_ratio, signal_honesty, and follow_rate
    across all three phases with phase boundary markers.
    """
    phase_colors = {"A": "steelblue", "B": "crimson", "C": "darkorange"}

    rounds  = [e["round"]          for e in log]
    metrics = {
        "win_ratio":      [e["win_ratio"]      for e in log],
        "signal_honesty": [e["signal_honesty"] for e in log],
        "follow_rate":    [e["follow_rate"]    for e in log],
    }
    phases  = [e["phase"] for e in log]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(
        "OracleGambit — Curriculum Training (A→B→C)",
        fontsize=14, fontweight="bold"
    )

    panel_info = [
        ("win_ratio",      "Win Ratio",       "Player Win Rate  (↑ Players winning)"),
        ("signal_honesty", "Signal Honesty",  "Host Honesty     (↓ Host deceiving)"),
        ("follow_rate",    "Follow Rate",     "Follow Rate      (↑ Players trusting signal)"),
    ]

    for ax, (key, ylabel, title) in zip(axes, panel_info):
        vals = metrics[key]

        # Draw each phase segment in its own color
        seg_start  = 0
        prev_phase = phases[0]
        for i in range(1, len(rounds) + 1):
            cur_phase = phases[i] if i < len(phases) else None
            if cur_phase != prev_phase:
                xs = rounds[seg_start:i]
                ys = vals[seg_start:i]
                ax.plot(xs, ys, color=phase_colors[prev_phase], linewidth=1.5)
                seg_start  = i
                prev_phase = cur_phase

        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-0.05, 1.10)
        ax.axhline(0.25, color="gray", linestyle="--", alpha=0.4, linewidth=0.8,
                   label="random baseline (0.25)")
        ax.grid(axis="y", alpha=0.3)

    # Phase transition verticals
    for i in range(1, len(phases)):
        if phases[i] != phases[i - 1]:
            for ax in axes:
                ax.axvline(rounds[i], color="black", linestyle=":", alpha=0.6, linewidth=1.2)

    # Legend
    patches = [
        mpatches.Patch(color=phase_colors["A"], label="Phase A: Warm-Up"),
        mpatches.Patch(color=phase_colors["B"], label="Phase B: Adversarial"),
        mpatches.Patch(color=phase_colors["C"], label="Phase C: Joint"),
    ]
    axes[0].legend(handles=patches, loc="upper right", fontsize=8)

    axes[2].set_xlabel("Round", fontsize=10)
    plt.tight_layout()

    out = os.path.join(save_dir, "curriculum_plot.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot → {out}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OracleGambit Curriculum Training (Phase A → B → C)"
    )
    p.add_argument("--players",            type=int,   default=6,
                   help="Number of players (default: 6)")
    p.add_argument("--rounds_a",           type=int,   default=50_000,
                   help="Max rounds for Phase A warm-up (default: 50000)")
    p.add_argument("--rounds_b",           type=int,   default=50_000,
                   help="Max rounds for Phase B adversarial (default: 50000)")
    p.add_argument("--rounds_c",           type=int,   default=100_000,
                   help="Rounds for Phase C joint training (default: 100000)")
    p.add_argument("--batch_size",         type=int,   default=128,
                   help="Rounds per gradient update (default: 128)")
    p.add_argument("--log_interval",       type=int,   default=2_000,
                   help="Log every N rounds (default: 2000)")
    p.add_argument("--spotlight_interval", type=int,   default=10_000,
                   help="Print round-level spotlight every N rounds (default: 10000)")
    p.add_argument("--lr_host",            type=float, default=3e-4,
                   help="Host learning rate (default: 3e-4)")
    p.add_argument("--lr_player",          type=float, default=3e-4,
                   help="Player learning rate (default: 3e-4)")
    p.add_argument("--entropy_coeff",      type=float, default=0.01,
                   help="Entropy bonus coefficient (default: 0.01)")
    p.add_argument("--target_follow",      type=float, default=0.65,
                   help="Phase A early-stop: follow_rate threshold (default: 0.65)")
    p.add_argument("--target_honesty",     type=float, default=0.35,
                   help="Phase B early-stop: signal_honesty threshold (default: 0.35)")
    p.add_argument("--target_entropy_b",   type=float, default=0.9,
                   help="Phase B early-stop: host entropy threshold (default: 0.9; must be < ln4≈1.386)")
    p.add_argument("--min_rounds_b",       type=int,   default=20_000,
                   help="Phase B: min rounds before early-stop (default: 20000)")
    p.add_argument("--seed",               type=int,   default=42)
    p.add_argument("--save_dir",           type=str,   default="checkpoints/curriculum")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())
