#!/usr/bin/env python3
"""Evaluate SB3 PPO player + PG host on ProposalVoteEnv."""

from __future__ import annotations

import argparse
import glob
import os
import re
import zipfile

import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from proposal_vote_env import Phase, Proposal, ProposalVoteConfig, ProposalVoteEnv, Vote
from sb3_proposal_agent import SB3ProposalHostAgent, SB3ProposalPlayerAgent

CKPT_DIR = "checkpoints_sb3_proposal"


def _ckpt_step(path: str, prefix: str, suffix: str) -> int | None:
    m = re.match(rf"^{re.escape(prefix)}_(\d+)\.{re.escape(suffix)}$", os.path.basename(path))
    return int(m.group(1)) if m else None


def _is_valid_player_zip(path: str) -> bool:
    if not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None
    except (zipfile.BadZipFile, OSError):
        return False


def _is_valid_host_pt(path: str) -> bool:
    try:
        torch.load(path, map_location="cpu", weights_only=True)
        return True
    except Exception:
        return False


def _valid_ckpts(ckpt_dir: str, prefix: str, suffix: str, validator) -> dict[int, str]:
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.{re.escape(suffix)}$")
    out: dict[int, str] = {}
    for path in glob.glob(os.path.join(ckpt_dir, f"{prefix}_*.{suffix}")):
        m = pattern.match(os.path.basename(path))
        if m and validator(path):
            out[int(m.group(1))] = path
    return out


def _resolve_checkpoints(
    ckpt_dir: str, player: str | None, host: str | None
) -> tuple[str, str]:
    players = _valid_ckpts(ckpt_dir, "player_model", "zip", _is_valid_player_zip)
    hosts = _valid_ckpts(ckpt_dir, "host_model", "pt", _is_valid_host_pt)

    if player is None and host is None:
        common = set(players) & set(hosts)
        if not common:
            raise FileNotFoundError(f"No matching checkpoints in {ckpt_dir}")
        step = max(common)
        return players[step], hosts[step]

    if player is None:
        step = _ckpt_step(host, "host_model", "pt")
        if step is not None and step in players:
            return players[step], host
        return max(players.values()), host

    if host is None:
        step = _ckpt_step(player, "player_model", "zip")
        if step is not None and step in hosts:
            return player, hosts[step]
        return player, max(hosts.values())

    return player, host


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--ckpt-dir", default=CKPT_DIR)
    parser.add_argument("--num-players", type=int, default=10)
    parser.add_argument("--max-rounds", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    args.player, args.host = _resolve_checkpoints(args.ckpt_dir, args.player, args.host)

    console = Console()
    config = ProposalVoteConfig(num_players=args.num_players, max_rounds=args.max_rounds)
    env = ProposalVoteEnv(config=config, seed=args.seed)
    obs, _ = env.reset()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
    if args.device != "auto":
        device = args.device

    player = SB3ProposalPlayerAgent(args.player, device=device)
    host = SB3ProposalHostAgent(args.host, config=config, device=device)

    console.print(
        Panel(
            "ProposalVote — SB3 eval\n"
            f"player={args.player}\nhost={args.host}\n"
            f"N={config.num_players}  n={config.n}  max_rounds={config.max_rounds}",
            style="bold cyan",
        )
    )

    stats = {"A": 0, "B": 0, "accept": 0, "host_sum": 0.0, "player_sum": 0.0}
    round_num = 0

    while True:
        if env.phase != Phase.PROPOSE:
            raise RuntimeError(f"unexpected phase {env.phase.name}")

        proposal = host.get_proposal(env, deterministic=True)
        env.step_propose(proposal)
        votes = player.get_vote_action(obs["players"], config.num_players, deterministic=True)
        obs, rewards, terminated, truncated, info = env.step_vote(votes)
        round_num += 1

        prop_name = info["proposal_name"]
        stats[prop_name] += 1
        if info["accepted"]:
            stats["accept"] += 1
        stats["host_sum"] += rewards["host"]
        stats["player_sum"] += float(np.sum(rewards["players"]))

        accept_n = int(np.sum(votes == Vote.ACCEPT))
        console.print(
            f"[dim]R{round_num:03d}[/dim] prop={prop_name} "
            f"votes={accept_n}/{config.num_players} accept "
            f"→ {'PASS' if info['accepted'] else 'FAIL'}  "
            f"H={rewards['host']:+.0f} P_each={rewards['players'][0]:+.0f}"
        )

        if terminated or truncated:
            break

    table = Table(title="Episode summary")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Rounds", str(round_num))
    table.add_row("Proposal A / B", f"{stats['A']} / {stats['B']}")
    table.add_row("Accept rate", f"{100 * stats['accept'] / max(round_num, 1):.1f}%")
    table.add_row("Host cumulative", f"{env.host_cumulative_reward:+.1f}")
    table.add_row("Player cumulative (sum)", f"{stats['player_sum']:+.1f}")
    console.print(table)


if __name__ == "__main__":
    main()
