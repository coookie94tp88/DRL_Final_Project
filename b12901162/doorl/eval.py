"""Evaluation entry point for DooRL.

Usage:
    python -m doorl.eval --ckpt runs/myrun/ckpt/latest.pt --episodes 500
    python -m doorl.eval --config config/default.yaml --baseline random_host
    # Cross-play: late host vs early players (and vice versa)
    python -m doorl.eval --host-ckpt runs/myrun/ckpt/late.pt \\
        --player-ckpt runs/myrun/ckpt/early.pt --episodes 200
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from doorl import parallel_env
from doorl.baselines import (
    GreedyPlayer,
    NoBribePlayer,
    NoisyTruthfulHost,
    RandomHost,
    TruthfulHost,
)
from doorl.checkpoint_utils import load_policies_from_checkpoints
from doorl.config_utils import load_config
from doorl.env.doorl_env import PHASE_BET, PHASE_BRIBE, PHASE_HOST
from doorl.eval_report import print_eval_report
from doorl.metrics import (
    MetricsAccumulator,
    MetricsSummary,
    check_acceptance_targets,
)
from doorl.models import HostPolicy, HostPolicyConfig, PlayerPolicy, PlayerPolicyConfig


# ---------------------------------------------------------------------------


def _build_from_cfg(cfg: Dict[str, Any]):
    env = parallel_env(**cfg["env"])
    p = PlayerPolicy(
        PlayerPolicyConfig(
            obs_dim=env.player_obs_dim,
            num_players=env.num_players,
            d_model=cfg["model"]["d_model"],
            n_layers=cfg["model"]["n_layers"],
            nhead=cfg["model"]["nhead"],
            dim_ff=cfg["model"]["dim_ff"],
            dropout=cfg["model"]["dropout"],
            parameter_sharing=cfg["model"]["parameter_sharing"],
        )
    )
    h = HostPolicy(
        HostPolicyConfig(
            obs_dim=env.host_obs_dim,
            num_players=env.num_players,
            d_model=cfg["model"]["d_model"],
            n_layers=cfg["model"]["n_layers"],
            nhead=cfg["model"]["nhead"],
            dim_ff=cfg["model"]["dim_ff"],
            dropout=cfg["model"]["dropout"],
        )
    )
    return env, p, h


def _policy_device(policy: torch.nn.Module) -> torch.device:
    return next(policy.parameters()).device


def _act_learned_player(policy: PlayerPolicy, env, obs, i: int):
    dev = _policy_device(policy)
    o = torch.tensor(obs[f"player_{i}"], dtype=torch.float32, device=dev).unsqueeze(
        0
    )
    idx = torch.tensor([i], dtype=torch.long, device=dev)
    act, _ = policy.act(o, idx, phase=env._phase, deterministic=True)
    return {
        "bribe_pct": np.array(act["bribe_pct"][0], dtype=np.float32).reshape(-1),
        "door": int(act["door"][0]),
        "bet_pct": np.array(act["bet_pct"][0], dtype=np.float32).reshape(-1),
    }


def _act_learned_host(policy: HostPolicy, env, obs):
    dev = _policy_device(policy)
    o = torch.tensor(obs["host"], dtype=torch.float32, device=dev).unsqueeze(0)
    act, _ = policy.act(o, deterministic=True)
    return {
        "public_door": int(act["public_door"][0]),
        "private_logits": act["private_logits"][0],
    }


# ---------------------------------------------------------------------------


def run_episodes(
    cfg: Dict[str, Any],
    player_policy: Optional[PlayerPolicy] = None,
    host_policy: Optional[HostPolicy] = None,
    host_baseline: Optional[Any] = None,
    player_baseline: Optional[Any] = None,
    episodes: int = 50,
    seed: int = 0,
) -> MetricsSummary:
    env = parallel_env(**cfg["env"])
    acc = MetricsAccumulator()
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            actions: Dict[str, Any] = {}
            # host action
            if host_baseline is not None:
                actions["host"] = host_baseline.act(env, obs)
            elif host_policy is not None:
                actions["host"] = _act_learned_host(host_policy, env, obs)
            else:
                actions["host"] = {
                    "public_door": 0,
                    "private_logits": np.zeros((env.num_players, 4), dtype=np.float32),
                }
            # player actions
            for i in range(env.num_players):
                if player_baseline is not None:
                    actions[f"player_{i}"] = player_baseline.act(
                        env, obs, i, env._phase
                    )
                elif player_policy is not None:
                    actions[f"player_{i}"] = _act_learned_player(
                        player_policy, env, obs, i
                    )
                else:
                    actions[f"player_{i}"] = {
                        "bribe_pct": np.array([0.0], dtype=np.float32),
                        "door": 0,
                        "bet_pct": np.array([0.1], dtype=np.float32),
                    }
            phase_before = env._phase
            obs, rewards, terms, _, _ = env.step(actions)
            done = all(terms.values())
            # log on settlement (after PHASE_BET)
            if phase_before == PHASE_BET:
                # last round just settled; pull from env.stats / history
                if env.stats.rounds_played > 0:
                    last_x = env.stats.x_values[-1] if env.stats.x_values else 0.0
                    # door_share from last history row
                    last_row = env._history[-1]
                    door_share = last_row[12:16].copy()
                    acc.add_round(
                        r_host=rewards["host"],
                        bribe_pcts=env._round_bribe_pcts.tolist(),
                        x=last_x,
                        door_share=door_share,
                        private_signals=env._round_private_signals.tolist(),
                        public_signal=int(env._round_public_signal),
                        true_door=int(env._round_true_door),
                        chosen_doors=env._last_chosen_doors.tolist(),
                        active_mask=env._active.tolist(),
                    )
        acc.add_end_of_episode(env._balances.tolist(), env._active.tolist())
    return acc.finalize()


# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None)
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--baseline", default=None,
                   help="Baseline name: random_host, truthful_host, noisy_truthful_host, greedy_players, no_bribe_players")
    p.add_argument(
        "--player-ckpt",
        default=None,
        help="Checkpoint for player policy (cross-play)",
    )
    p.add_argument(
        "--host-ckpt",
        default=None,
        help="Checkpoint for host policy (cross-play)",
    )
    p.add_argument("--override", action="append", default=[])
    p.add_argument(
        "--watch",
        action="store_true",
        help="Print one episode in the terminal (see doorl.watch) instead of JSON metrics",
    )
    p.add_argument(
        "--watch-style",
        choices=("fancy", "plain"),
        default="fancy",
    )
    p.add_argument("--watch-lang", choices=("en", "zh"), default="en")
    p.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON instead of human-readable report",
    )
    p.add_argument(
        "--lang",
        choices=("zh", "en"),
        default="zh",
        help="Language for human-readable report (default: zh)",
    )
    args = p.parse_args()

    cross_play = False
    has_learned = bool(args.ckpt or args.player_ckpt or args.host_ckpt)
    if has_learned:
        cfg, player_sd, host_sd = load_policies_from_checkpoints(
            args.player_ckpt,
            args.host_ckpt,
            default_ckpt=args.ckpt,
        )
    else:
        cfg = load_config(args.config, overrides=args.override)
        player_sd = host_sd = None

    if args.override:
        from doorl.config_utils import apply_overrides

        cfg = apply_overrides(cfg, args.override)

    env, player, host = _build_from_cfg(cfg)
    if has_learned:
        player.load_state_dict(player_sd)
        host.load_state_dict(host_sd)
        cross_play = bool(
            args.player_ckpt
            and args.host_ckpt
            and args.player_ckpt != args.host_ckpt
        ) or (
            args.player_ckpt is not None
            and args.host_ckpt is None
            and args.ckpt is not None
            and args.player_ckpt != args.ckpt
        )

    host_baseline = None
    player_baseline = None
    name = args.baseline
    if name == "random_host":
        host_baseline = RandomHost(env.num_players, seed=args.seed)
        player = player if args.ckpt else None
    elif name == "truthful_host":
        host_baseline = TruthfulHost(env.num_players, seed=args.seed)
    elif name == "noisy_truthful_host":
        host_baseline = NoisyTruthfulHost(env.num_players, seed=args.seed)
    elif name == "greedy_players":
        player_baseline = GreedyPlayer()
    elif name == "no_bribe_players":
        player_baseline = NoBribePlayer()
    elif name is not None:
        raise ValueError(f"Unknown baseline {name!r}")

    use_learned = has_learned and host_baseline is None and player_baseline is None

    if args.watch:
        from doorl.watch import watch_episode

        watch_episode(
            cfg,
            player_policy=player if use_learned else None,
            host_policy=host if use_learned else None,
            host_baseline=host_baseline,
            player_baseline=player_baseline,
            seed=args.seed,
            style=args.watch_style,
            lang=args.watch_lang,
            ckpt_label=args.ckpt or args.host_ckpt or args.player_ckpt,
        )
        return

    summary = run_episodes(
        cfg,
        player_policy=player if use_learned else None,
        host_policy=host if use_learned else None,
        host_baseline=host_baseline,
        player_baseline=player_baseline,
        episodes=args.episodes,
        seed=args.seed,
    )
    accept = check_acceptance_targets(
        summary,
        tau=float(cfg["env"]["payout_threshold"]),
        initial_balance=float(cfg["env"]["initial_balance"]),
    )

    if args.json:
        out = {
            "summary": asdict(summary),
            "acceptance": accept,
            "config": {
                "payout_threshold": cfg["env"]["payout_threshold"],
                "num_players": cfg["env"]["num_players"],
                "max_rounds": cfg["env"]["max_rounds"],
                "baseline": name,
                "cross_play": cross_play,
                "player_ckpt": args.player_ckpt or args.ckpt,
                "host_ckpt": args.host_ckpt or args.ckpt,
            },
        }
        print(json.dumps(out, indent=2))
    else:
        print_eval_report(
            summary,
            accept,
            episodes=args.episodes,
            tau=float(cfg["env"]["payout_threshold"]),
            initial_balance=float(cfg["env"]["initial_balance"]),
            num_players=int(cfg["env"]["num_players"]),
            max_rounds=int(cfg["env"]["max_rounds"]),
            ckpt=args.ckpt or args.host_ckpt or args.player_ckpt,
            baseline=name,
            lang=args.lang,
            cross_play=cross_play,
        )


if __name__ == "__main__":
    main()
