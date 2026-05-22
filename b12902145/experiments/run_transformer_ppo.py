"""
Phase 2: Transformer + PPO
===========================
Entry point for training and evaluation.

Usage examples
--------------
  # Quick smoke test (2000 rounds)
  python experiments/run_transformer_ppo.py --rounds 2000

  # Full training run (compare against Phase 1 MLP+REINFORCE)
  python experiments/run_transformer_ppo.py --rounds 100000

  # Custom architecture / PPO hyperparams
  python experiments/run_transformer_ppo.py \\
    --d_model 128 --nhead 4 --num_layers 3 \\
    --clip_eps 0.2 --gamma 0.99 --gae_lambda 0.95 \\
    --ppo_epochs 4 --value_coeff 0.5

Notes
-----
* Host    : hist_feat_size=4  (obs_dim=255)
* Player  : hist_feat_size=8  (obs_dim=455)
  These are hardcoded from the env constants and not exposed as CLI args.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt

from env.oracle_gambit_env import OracleGambitEnv
from agents.transformer_agent import TransformerAgent
from training.ppo_runner import PPORunner


# ---------------------------------------------------------------------------
# Log tee
# ---------------------------------------------------------------------------

class _LogTee:
    """Mirror stdout to a file, ANSI codes preserved.

    Replay with: cat terminal.log  or  less -R terminal.log
    """

    def __init__(self, path: str) -> None:
        self._path   = path
        self._file   = open(path, "w", encoding="utf-8")
        self._stdout = sys.stdout
        sys.stdout   = self

    def write(self, data: str) -> None:
        self._stdout.write(data)
        self._file.write(data)

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()

    def fileno(self) -> int:
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

    # ── Environment ──────────────────────────────────────────────────────
    env = OracleGambitEnv(
        num_players=args.players,
        num_doors=4,
        payout_threshold=0.20,
        history_window=50,
        max_rounds=0,
        seed=args.seed,
    )
    env.reset(seed=args.seed)

    D  = env.num_doors
    host_obs_dim   = env.observation_space("host").shape[0]
    player_obs_dim = env.observation_space("player_0").shape[0]

    # hist_feat_size: derived from env structure, not exposed as CLI args
    #   host   obs = history(50×4) + mask(50) + current(5) = 255
    #   player obs = history(50×8) + mask(50) + current(5) = 455
    HOST_HIST_FEAT   = 4
    PLAYER_HIST_FEAT = 8

    # ── Agents ───────────────────────────────────────────────────────────
    host_agent = TransformerAgent(
        obs_dim        = host_obs_dim,
        num_doors      = D,
        hist_feat_size = HOST_HIST_FEAT,
        history_window = 50,
        d_model        = args.d_model,
        nhead          = args.nhead,
        num_enc_layers = args.num_layers,
        ff_dim         = args.d_model * 4,
        hidden_dim     = 128,
    )

    player_agent = TransformerAgent(
        obs_dim        = player_obs_dim,
        num_doors      = D,
        hist_feat_size = PLAYER_HIST_FEAT,
        history_window = 50,
        d_model        = args.d_model,
        nhead          = args.nhead,
        num_enc_layers = args.num_layers,
        ff_dim         = args.d_model * 4,
        hidden_dim     = 128,
    )

    print("=" * 60)
    print("OracleGambit  Phase 2 — Transformer + PPO")
    print("=" * 60)
    print(f"  Players         : {args.players}")
    print(f"  Doors           : {D}")
    print(f"  Host obs_dim    : {host_obs_dim}  (hist_feat={HOST_HIST_FEAT})")
    print(f"  Player obs_dim  : {player_obs_dim}  (hist_feat={PLAYER_HIST_FEAT})")
    print(f"  Host params     : {sum(p.numel() for p in host_agent.parameters()):,}")
    print(f"  Player params   : {sum(p.numel() for p in player_agent.parameters()):,}")
    print(f"  Transformer     : d_model={args.d_model}  nhead={args.nhead}  layers={args.num_layers}")
    print(f"  PPO             : clip_ε={args.clip_eps}  γ={args.gamma}  λ={args.gae_lambda}")
    print(f"                    epochs={args.ppo_epochs}  v_coeff={args.value_coeff}  mb={args.minibatch_size}")
    print(f"  LR              : {args.lr}")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  Entropy coeff   : {args.entropy_coeff}")
    print(f"  Total rounds    : {args.rounds:,}")
    print("=" * 60)
    print()

    # ── Runner ───────────────────────────────────────────────────────────
    runner = PPORunner(
        env,
        host_agent,
        player_agent,
        lr_host       = args.lr,
        lr_player     = args.lr,
        clip_eps      = args.clip_eps,
        gamma         = args.gamma,
        gae_lambda    = args.gae_lambda,
        value_coeff   = args.value_coeff,
        entropy_coeff = args.entropy_coeff,
        ppo_epochs    = args.ppo_epochs,
        minibatch_size= args.minibatch_size,
    )

    # ── Train ────────────────────────────────────────────────────────────
    if args.curriculum:
        runner.run_curriculum(
            rounds_a            = args.rounds_a,
            rounds_b            = args.rounds_b,
            rounds_c            = args.rounds_c,
            batch_size          = args.batch_size,
            log_interval        = args.log_interval,
            spotlight_interval  = args.spotlight_interval,
            target_follow_rate_a= args.target_follow,
            target_honesty_b    = args.target_honesty,
            target_entropy_b    = args.target_entropy_b,
            min_rounds_b        = args.min_rounds_b,
            save_dir            = args.save_dir,
        )
        log = []   # curriculum runner logs to CSV; no in-memory return
    else:
        log = runner.train(
            total_rounds       = args.rounds,
            batch_size         = args.batch_size,
            log_interval       = args.log_interval,
            save_interval      = args.save_interval,
            spotlight_interval = args.spotlight_interval,
            save_dir           = args.save_dir,
        )

    # ── Plot & save ───────────────────────────────────────────────────────
    if log:
        _plot(log, args.save_dir)
    elif args.curriculum:
        _plot_curriculum(os.path.join(args.save_dir, "training_log.csv"), args.save_dir)
    _tee.close()


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
    fig.suptitle("OracleGambit  Phase 2 — Transformer + PPO",
                 fontsize=13, fontweight="bold")

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
    axes[2].plot(rounds, honesties,    label="Signal Honesty", color="seagreen", linewidth=1.5)
    axes[2].plot(rounds, follow_rates, label="Follow Rate",    color="coral",    linewidth=1.5)
    axes[2].axhline(0.25, color="gray", linestyle=":", linewidth=0.8, label="Random (0.25)")
    axes[2].set_ylabel("Rate")
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("Round")
    axes[2].legend(loc="upper right")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {path}")
    plt.show()


def _plot_curriculum(csv_path: str, save_dir: str) -> None:
    """Plot curriculum training curve with phase bands."""
    import csv
    if not os.path.exists(csv_path):
        return
    rows = []
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    rounds    = [int(r["round"])          for r in rows]
    h_rewards = [float(r["host_reward"])  for r in rows]
    p_rewards = [float(r["player_reward"]) for r in rows]
    win_r     = [float(r["win_ratio"])    for r in rows]
    honesties = [float(r["signal_honesty"]) for r in rows]
    fol_rates = [float(r["follow_rate"])  for r in rows]
    phases    = [r["phase"]               for r in rows]

    phase_colors = {"A": "#d0eaff", "B": "#ffd0d0", "C": "#d0ffd8"}
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("OracleGambit  Transformer + PPO  (Curriculum)",
                 fontsize=13, fontweight="bold")

    def _shade_phases(ax):
        prev_p, prev_r = phases[0], rounds[0]
        for i, (r, p) in enumerate(zip(rounds, phases)):
            if p != prev_p or i == len(phases) - 1:
                ax.axvspan(prev_r, r, alpha=0.18, color=phase_colors.get(prev_p, "#eeeeee"),
                           label=f"Phase {prev_p}")
                prev_p, prev_r = p, r

    for ax in axes:
        _shade_phases(ax)

    axes[0].plot(rounds, h_rewards, label="Host",   color="purple",    linewidth=1.5)
    axes[0].plot(rounds, p_rewards, label="Player", color="steelblue", linewidth=1.5)
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Avg Reward / Round")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.3)

    axes[1].plot(rounds, win_r, label="Win-ratio x", color="darkorange", linewidth=1.5)
    axes[1].axhline(0.25, color="navy", linestyle=":",  linewidth=1.0, label="Random (0.25)")
    axes[1].axhline(0.20, color="red",  linestyle="--", linewidth=1.0, label="θ=0.20")
    axes[1].set_ylabel("Win Ratio")
    axes[1].set_ylim(0, 1)
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.3)

    axes[2].plot(rounds, honesties, label="Signal Honesty", color="seagreen", linewidth=1.5)
    axes[2].plot(rounds, fol_rates, label="Follow Rate",    color="coral",    linewidth=1.5)
    axes[2].axhline(0.25, color="gray", linestyle=":", linewidth=0.8)
    axes[2].set_ylabel("Rate")
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("Round")
    axes[2].legend(loc="upper left")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "curriculum_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nCurriculum plot saved → {path}")
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2: Transformer + PPO on OracleGambit"
    )
    # ── Environment ──────────────────────────────────────────────────────
    p.add_argument("--players",    type=int,   default=6)
    p.add_argument("--rounds",     type=int,   default=100_000)
    p.add_argument("--batch_size", type=int,   default=128)
    p.add_argument("--seed",       type=int,   default=42)
    # ── Optimiser ────────────────────────────────────────────────────────
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--entropy_coeff", type=float, default=0.01)
    # ── Transformer architecture ─────────────────────────────────────────
    p.add_argument("--d_model",    type=int, default=64,
                   help="Transformer embedding dimension (default 64)")
    p.add_argument("--nhead",      type=int, default=4,
                   help="Number of attention heads (default 4)")
    p.add_argument("--num_layers", type=int, default=2,
                   help="Number of TransformerEncoder layers (default 2)")
    # ── PPO hyperparameters ───────────────────────────────────────────────
    p.add_argument("--clip_eps",    type=float, default=0.2,
                   help="PPO clipping epsilon (default 0.2)")
    p.add_argument("--gamma",       type=float, default=0.99,
                   help="Discount factor γ (default 0.99)")
    p.add_argument("--gae_lambda",  type=float, default=0.95,
                   help="GAE λ (default 0.95)")
    p.add_argument("--ppo_epochs",  type=int,   default=4,
                   help="PPO update epochs per batch (default 4)")
    p.add_argument("--value_coeff", type=float, default=0.5,
                   help="Value loss coefficient (default 0.5)")
    p.add_argument("--minibatch_size", type=int, default=256,
                   help="Mini-batch size inside each PPO epoch (default 256; 0=full-batch)")
    # ── Logging / saving ─────────────────────────────────────────────────
    p.add_argument("--log_interval",       type=int, default=2_000)
    p.add_argument("--save_interval",      type=int, default=20_000)
    p.add_argument("--spotlight_interval", type=int, default=10_000,
                   help="Print round-level spotlight every N rounds (0=disable)")
    p.add_argument("--save_dir", type=str, default="checkpoints/transformer_ppo")    # ── Curriculum mode ─────────────────────────────────────────────────
    p.add_argument("--curriculum", action="store_true",
                   help="Run 3-phase curriculum (A→B→C) instead of joint training")
    p.add_argument("--rounds_a",         type=int,   default=50_000)
    p.add_argument("--rounds_b",         type=int,   default=50_000)
    p.add_argument("--rounds_c",         type=int,   default=100_000)
    p.add_argument("--min_rounds_b",     type=int,   default=20_000)
    p.add_argument("--target_follow",    type=float, default=0.65)
    p.add_argument("--target_honesty",   type=float, default=0.35)
    p.add_argument("--target_entropy_b", type=float, default=0.9)
    return p.parse_args()


if __name__ == "__main__":
    main(_parse())
