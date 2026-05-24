#!/usr/bin/env python3
"""Compare player_model_800 on b139 native env vs b129 belief env (subprocess + in-process)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

from test_b139_player_on_b129 import B139PlayerAdapter, run_episode
from env import OracleGambitConfig, OracleGambitEnv


def run_b129_episodes(model: str, episodes: int, seed: int, max_rounds: int, device: str) -> list[dict]:
    cfg = OracleGambitConfig(num_players=10, num_doors=3, max_rounds=max_rounds, initial_balance=1000.0)
    env = OracleGambitEnv(config=cfg, seed=seed)
    player = B139PlayerAdapter(model, num_players=10, num_doors=3, device=device)
    rng = np.random.default_rng(seed)
    return [run_episode(env, player, rng) for _ in range(episodes)]


def run_b139_native_episodes(
    model: str, episodes: int, seed: int, max_rounds: int, device: str, b139_dir: str
) -> list[dict]:
    script = os.path.join(b139_dir, "run_native_player_test.py")
    cmd = [
        sys.executable,
        script,
        "--model",
        model,
        "--episodes",
        str(episodes),
        "--seed",
        str(seed),
        "--max-rounds",
        str(max_rounds),
        "--device",
        device,
    ]
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp2/b12902115/tmp/mpl")
    proc = subprocess.run(cmd, cwd=b139_dir, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        proc.check_returncode()
    return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]


def summarize(label: str, episodes: list[dict]) -> None:
    host_cums = [e["host_cumulative"] for e in episodes]
    bals = [np.array(e["final_balances"]) for e in episodes]
    alive = [int(np.sum(b > 0)) for b in bals]
    mean_bals = [float(np.mean(b)) for b in bals]
    print(f"\n{label}")
    print("-" * len(label))
    for i, ep in enumerate(episodes):
        print(
            f"  Ep {i + 1}: host_cum={ep['host_cumulative']:+10.1f}  "
            f"alive={alive[i]}/10  mean_bal={mean_bals[i]:8.1f}"
        )
    print(
        f"  Avg : host_cum={np.mean(host_cums):+10.1f}  "
        f"alive={np.mean(alive):.1f}/10  mean_bal={np.mean(mean_bals):8.1f}"
    )


def main() -> None:
    b139_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "b13902055")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.join(b139_dir, "player_model_800.zip"))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model_path = os.path.abspath(args.model)

    print(f"Model: {model_path}")
    print(f"Config: 10 players, 3 doors, {args.max_rounds} rounds, seed={args.seed}")
    print("Host: random (both envs)")

    eps_native = run_b139_native_episodes(
        model_path, args.episodes, args.seed, args.max_rounds, args.device, b139_dir
    )
    eps_b129 = run_b129_episodes(
        model_path, args.episodes, args.seed, args.max_rounds, args.device
    )

    summarize("b139 native env (door actions, native obs)", eps_native)
    summarize("b129 belief env (belief adapter + door→belief)", eps_b129)

    h_native = np.mean([e["host_cumulative"] for e in eps_native])
    h_b129 = np.mean([e["host_cumulative"] for e in eps_b129])
    b_native = np.mean([np.mean(e["final_balances"]) for e in eps_native])
    b_b129 = np.mean([np.mean(e["final_balances"]) for e in eps_b129])
    print("\nDelta (b129 − b139 native)")
    print(f"  avg host_cum: {h_b129 - h_native:+.1f}")
    print(f"  avg mean_bal: {b_b129 - b_native:+.1f}")


if __name__ == "__main__":
    main()
