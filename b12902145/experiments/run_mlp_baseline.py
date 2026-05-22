"""
Phase 1 Baseline: MLP + REINFORCE
===================================
Entry point for training and evaluation.

Usage examples
--------------
  # Quick smoke test (2000 rounds)
  python experiments/run_mlp_baseline.py --rounds 2000

  # Full training run
  python experiments/run_mlp_baseline.py --rounds 100000 --players 6

  # Custom hyperparameters
  python experiments/run_mlp_baseline.py --lr 1e-3 --entropy_coeff 0.05
"""
from __future__ import annotations

import argparse
import os
import sys

# Allow running from the b12902145/ root directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt

from env.oracle_gambit_env import OracleGambitEnv
from agents.mlp_agent import MlpAgent
from training.reinforce_runner import ReinforceRunner


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Environment ──────────────────────────────────────────────────────
    env = OracleGambitEnv(
        num_players=args.players,
        num_doors=4,
        payout_threshold=0.20,
        history_window=50,
        max_rounds=0,          # unlimited — round_frac feature = 0.0
        seed=args.seed,
    )
    env.reset(seed=args.seed)

    D = env.num_doors
    host_obs_dim   = env.observation_space("host").shape[0]
    player_obs_dim = env.observation_space("player_0").shape[0]

    # ── Agents ───────────────────────────────────────────────────────────
    host_agent   = MlpAgent(host_obs_dim,   D, hidden_dims=(256, 128))
    player_agent = MlpAgent(player_obs_dim, D, hidden_dims=(256, 128))

    print("=" * 60)
    print("OracleGambit  Phase 1 — MLP + REINFORCE")
    print("=" * 60)
    print(f"  Players        : {args.players}")
    print(f"  Doors          : {D}")
    print(f"  Host obs_dim   : {host_obs_dim}")
    print(f"  Player obs_dim : {player_obs_dim}")
    print(f"  Host params    : {sum(p.numel() for p in host_agent.parameters()):,}")
    print(f"  Player params  : {sum(p.numel() for p in player_agent.parameters()):,}")
    print(f"  LR             : {args.lr}")
    print(f"  Batch size     : {args.batch_size}")
    print(f"  Entropy coeff  : {args.entropy_coeff}")
    print(f"  Total rounds   : {args.rounds:,}")
    print("=" * 60)
    print()

    # ── Runner ───────────────────────────────────────────────────────────
    runner = ReinforceRunner(
        env,
        host_agent,
        player_agent,
        lr_host=args.lr,
        lr_player=args.lr,
        entropy_coeff=args.entropy_coeff,
    )

    # ── Train ────────────────────────────────────────────────────────────
    log = runner.train(
        total_rounds=args.rounds,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        save_dir=args.save_dir,
    )

    # ── Plot & save ───────────────────────────────────────────────────────
    if log:
        _plot(log, args.save_dir)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot(log: list[dict], save_dir: str) -> None:
    rounds         = [e["round"]          for e in log]
    host_rewards   = [e["host_reward"]    for e in log]
    player_rewards = [e["player_reward"]  for e in log]
    win_ratios     = [e["win_ratio"]      for e in log]
    honesties      = [e["signal_honesty"] for e in log]
    follow_rates   = [e["follow_rate"]    for e in log]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle("OracleGambit  Phase 1 — MLP + REINFORCE", fontsize=13, fontweight="bold")

    # Panel 1: rewards
    axes[0].plot(rounds, host_rewards,   label="Host",   color="purple",    linewidth=1.5)
    axes[0].plot(rounds, player_rewards, label="Player", color="steelblue", linewidth=1.5)
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Avg Reward / Round")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    # Panel 2: win ratio
    axes[1].plot(rounds, win_ratios, label="Win-ratio x", color="darkorange", linewidth=1.5)
    axes[1].axhline(0.25, color="navy",  linestyle=":",  linewidth=1.0, label="Random (0.25)")
    axes[1].axhline(0.20, color="red",   linestyle="--", linewidth=1.0, label="θ = 0.20")
    axes[1].set_ylabel("Win Ratio")
    axes[1].set_ylim(0, 1)
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    # Panel 3: honesty & follow rate
    axes[2].plot(rounds, honesties,   label="Signal Honesty", color="seagreen", linewidth=1.5)
    axes[2].plot(rounds, follow_rates, label="Follow Rate",   color="coral",    linewidth=1.5)
    axes[2].axhline(0.25, color="gray", linestyle=":", linewidth=0.8, label="Random (0.25)")
    axes[2].set_ylabel("Rate")
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("Round")
    axes[2].legend(loc="upper right")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "training_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {path}")
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 1: MLP + REINFORCE on OracleGambit"
    )
    p.add_argument("--players",        type=int,   default=6)
    p.add_argument("--rounds",         type=int,   default=100_000)
    p.add_argument("--batch_size",     type=int,   default=128)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--entropy_coeff",  type=float, default=0.01)
    p.add_argument("--log_interval",   type=int,   default=2_000)
    p.add_argument("--save_interval",  type=int,   default=20_000)
    p.add_argument("--save_dir",       type=str,   default="checkpoints/mlp_reinforce")
    p.add_argument("--seed",           type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(_parse())
