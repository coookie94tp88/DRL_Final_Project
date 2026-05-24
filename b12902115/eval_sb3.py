#!/usr/bin/env python3
"""Evaluate SB3 PPO player + MLP host from train_sb3.py checkpoints."""

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

from env import OracleGambitConfig, OracleGambitEnv, Phase, PlayerBelief
from eval import EvalPlayerStats, render_player_summary, render_round_log
from player_agent import TrainedPlayerAgent
from sb3_player_agent import SB3PlayerAgent
from train_sb3 import HostPolicy, host_obs_to_tensors

TrainedPlayerAgent.BELIEF_NAMES = {0: "pub", 1: "priv", 2: "rnd"}

CKPT_DIR = "checkpoints_sb3"


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


def _valid_sb3_ckpts(ckpt_dir: str, prefix: str, suffix: str, validator) -> dict[int, str]:
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.{re.escape(suffix)}$")
    out: dict[int, str] = {}
    for path in glob.glob(os.path.join(ckpt_dir, f"{prefix}_*.{suffix}")):
        m = pattern.match(os.path.basename(path))
        if m and validator(path):
            out[int(m.group(1))] = path
    return out


def _resolve_sb3_checkpoints(
    ckpt_dir: str, player: str | None, host: str | None
) -> tuple[str, str]:
    players = _valid_sb3_ckpts(ckpt_dir, "player_model", "zip", _is_valid_player_zip)
    hosts = _valid_sb3_ckpts(ckpt_dir, "host_model", "pt", _is_valid_host_pt)

    if player is None and host is None:
        common = set(players) & set(hosts)
        if not common:
            raise FileNotFoundError(f"No matching valid player/host checkpoints in {ckpt_dir}")
        step = max(common)
        return players[step], hosts[step]

    if player is None:
        step = _ckpt_step(host, "host_model", "pt")
        if step is not None and step in players:
            return players[step], host
        if not players:
            raise FileNotFoundError(f"No valid player_model_*.zip in {ckpt_dir}")
        step = max(players)
        print(f"Warning: no valid player for {host}; using {players[step]}")
        return players[step], host

    if host is None:
        step = _ckpt_step(player, "player_model", "zip")
        if step is not None and step in hosts:
            return player, hosts[step]
        if not hosts:
            raise FileNotFoundError(f"No valid host_model_*.pt in {ckpt_dir}")
        step = max(hosts)
        print(f"Warning: no valid host for {player}; using {hosts[step]}")
        return player, hosts[step]

    return player, host


class SB3HostAgent:
    def __init__(self, checkpoint_path: str, config: OracleGambitConfig, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        tmp = OracleGambitEnv(config=config)
        self.policy = HostPolicy(tmp.host_observation_space, tmp.host_action_space).to(self.device)
        self.policy.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        )
        self.policy.eval()
        print(f"Host loaded: {checkpoint_path} ({self.device})")

    @torch.no_grad()
    def get_action(self, env: OracleGambitEnv, deterministic: bool = True) -> tuple[int, np.ndarray]:
        host_obs = host_obs_to_tensors(env._get_observations()["host"])
        host_obs = {k: v.to(self.device) for k, v in host_obs.items()}
        pub_logits, priv_logits = self.policy(host_obs)
        if deterministic:
            pub = int(pub_logits.argmax().item())
            priv = priv_logits.argmax(dim=-1).cpu().numpy().astype(np.int32)
        else:
            pub = int(torch.distributions.Categorical(logits=pub_logits).sample().item())
            priv = torch.distributions.Categorical(logits=priv_logits).sample().cpu().numpy().astype(
                np.int32
            )
        return pub, priv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--player",
        default=None,
        help=f"SB3 player zip (default: latest valid paired step in {CKPT_DIR})",
    )
    parser.add_argument(
        "--host",
        default=None,
        help=f"host .pt (default: latest valid paired step in {CKPT_DIR})",
    )
    parser.add_argument("--num-doors", type=int, default=4)
    parser.add_argument("--max-rounds", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    args.player, args.host = _resolve_sb3_checkpoints(CKPT_DIR, args.player, args.host)

    console = Console()
    config = OracleGambitConfig(
        num_players=10,
        num_doors=args.num_doors,
        max_rounds=args.max_rounds,
        initial_balance=1000.0,
    )
    env = OracleGambitEnv(config=config, seed=args.seed)
    obs, _ = env.reset()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
    if args.device != "auto":
        device = args.device

    console.print("[bold yellow]Loading SB3 checkpoints...[/bold yellow]")
    player = SB3PlayerAgent(args.player, num_players=config.num_players, device=device)
    host = SB3HostAgent(args.host, config, device=device)

    console.print(
        Panel(
            "OracleGambit — SB3 eval (PPO player + PG host)\n"
            f"player={args.player}\nhost={args.host}\n"
            f"Players={config.num_players}  Doors={config.num_doors}  Rounds={config.max_rounds}",
            style="bold yellow",
        )
    )

    round_count = 1
    stats = EvalPlayerStats(config.num_players)
    last_bribe_fractions = np.zeros(config.num_players, dtype=np.float32)

    while True:
        if env.phase == Phase.BRIBE:
            round_active = env.balances > 0
            bribe_fractions = player.get_bribe_action(obs["players"], deterministic=True)
            last_bribe_fractions = np.asarray(bribe_fractions, dtype=np.float32).copy()
            obs, _, _, _, _ = env.step({"player_bribe_fractions": bribe_fractions})

        elif env.phase == Phase.SIGNAL:
            pub, priv = host.get_action(env, deterministic=True)
            obs, _, _, _, _ = env.step({"public_signal": pub, "private_signals": priv})

        elif env.phase == Phase.BET:
            beliefs, bet_fracs = player.get_bet_action(obs["players"], deterministic=True)
            obs, rewards, terminated, truncated, info = env.step(
                {"player_beliefs": beliefs, "bet_fractions": bet_fracs}
            )
            render_round_log(console, env, rewards, info, round_count)
            stats.record_round(
                round_active=round_active,
                bribe_fractions=last_bribe_fractions,
                paid_bribes=env.hist_bribes[-1],
                beliefs=env.hist_beliefs[-1],
                private_signals=env.hist_private_signals[-1],
                winning_door=int(info["winning_door"]),
            )

            if terminated or truncated:
                console.print(f"\n[bold yellow]Game Over at Round {round_count}![/bold yellow]")
                render_player_summary(console, stats, round_count)
                console.print(
                    f"[bold]Final host cumulative profit:[/bold] {env.host_cumulative_profit:+.2f}"
                )
                break
            round_count += 1


if __name__ == "__main__":
    main()
