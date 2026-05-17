"""Watch DooRL episodes in the terminal (fancy or plain).

Usage:
    python -m doorl.watch --ckpt runs/learn_4p/ckpt/latest.pt --seed 0
    python -m doorl.watch --baseline truthful_host --override env.num_players=5
    python -m doorl.watch --ckpt runs/learn_4p/ckpt/latest.pt --style plain --lang zh
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Optional

import numpy as np

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
from doorl.console_render import (
    _supports_color,
    print_episode_footer,
    print_episode_header,
    print_settlement_fancy,
    print_settlement_plain,
)
from doorl.env.doorl_env import PHASE_BET
from doorl.eval import _act_learned_host, _act_learned_player, _build_from_cfg
from doorl.models import HostPolicy, PlayerPolicy


def _build_actions(
    env,
    obs: Dict[str, np.ndarray],
    *,
    player_policy: Optional[PlayerPolicy],
    host_policy: Optional[HostPolicy],
    host_baseline: Optional[Any],
    player_baseline: Optional[Any],
) -> Dict[str, Any]:
    actions: Dict[str, Any] = {}
    if host_baseline is not None:
        actions["host"] = host_baseline.act(env, obs)
    elif host_policy is not None:
        actions["host"] = _act_learned_host(host_policy, env, obs)
    else:
        actions["host"] = {
            "public_door": 0,
            "private_logits": np.zeros((env.num_players, 4), dtype=np.float32),
        }

    for i in range(env.num_players):
        name = f"player_{i}"
        if player_baseline is not None:
            actions[name] = player_baseline.act(env, obs, i, env._phase)
        elif player_policy is not None:
            actions[name] = _act_learned_player(player_policy, env, obs, i)
        else:
            actions[name] = {
                "bribe_pct": np.array([0.1], dtype=np.float32),
                "door": int(env._rng.integers(0, 4)),
                "bet_pct": np.array([0.1], dtype=np.float32),
            }
    return actions


def watch_episode(
    cfg: Dict[str, Any],
    *,
    player_policy: Optional[PlayerPolicy] = None,
    host_policy: Optional[HostPolicy] = None,
    host_baseline: Optional[Any] = None,
    player_baseline: Optional[Any] = None,
    seed: int = 0,
    max_rounds: Optional[int] = None,
    style: str = "fancy",
    lang: str = "en",
    color: Optional[bool] = None,
    ckpt_label: Optional[str] = None,
) -> None:
    env = parallel_env(**cfg["env"])
    if max_rounds is not None:
        env.cfg.max_rounds = int(max_rounds)

    use_color = _supports_color() if color is None else color

    print_episode_header(
        title="DooRL — Episode Watch",
        num_players=env.num_players,
        tau=float(env.cfg.payout_threshold),
        max_rounds=env.cfg.max_rounds,
        seed=seed,
        ckpt=ckpt_label,
        color=use_color,
    )

    obs, _ = env.reset(seed=seed)
    host_cumulative = 0.0
    done = False

    while not done:
        actions = _build_actions(
            env,
            obs,
            player_policy=player_policy,
            host_policy=host_policy,
            host_baseline=host_baseline,
            player_baseline=player_baseline,
        )
        phase_before = env._phase
        obs, rewards, terms, _, _ = env.step(actions)
        done = all(terms.values())

        if phase_before == PHASE_BET and env.last_settlement is not None:
            s = env.last_settlement
            host_cumulative += float(s.reward_host)
            balances = env._balances.copy()

            if style == "plain":
                print_settlement_plain(
                    s,
                    env.cfg,
                    balances=balances,
                    host_cumulative=host_cumulative,
                    lang=lang,
                )
            else:
                print_settlement_fancy(
                    s,
                    env.cfg,
                    balances=balances,
                    host_cumulative=host_cumulative,
                    color=use_color,
                    lang=lang,
                )

    print_episode_footer(
        host_total=host_cumulative,
        balances=env._balances.copy(),
        cfg=env.cfg,
        lang=lang,
        color=use_color,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Watch a DooRL episode in the terminal")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--baseline", default=None)
    p.add_argument("--player-ckpt", default=None)
    p.add_argument("--host-ckpt", default=None)
    p.add_argument("--override", action="append", default=[])
    p.add_argument(
        "--style",
        choices=("fancy", "plain"),
        default="fancy",
        help="fancy=OracleGambit-style bars; plain=emoji block (zh-friendly)",
    )
    p.add_argument("--lang", choices=("en", "zh"), default="en")
    p.add_argument("--no-color", action="store_true")
    p.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Cap rounds for a shorter demo (default: env config)",
    )
    args = p.parse_args()

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

    if args.override and has_learned:
        from doorl.config_utils import apply_overrides

        cfg = apply_overrides(cfg, args.override)
    elif args.override:
        cfg = load_config(args.config, overrides=args.override)

    env, player, host = _build_from_cfg(cfg)
    if has_learned:
        player.load_state_dict(player_sd)
        host.load_state_dict(host_sd)
        player.eval()
        host.eval()

    host_baseline = None
    player_baseline = None
    if args.baseline == "random_host":
        host_baseline = RandomHost(env.num_players, seed=args.seed)
    elif args.baseline == "truthful_host":
        host_baseline = TruthfulHost(env.num_players, seed=args.seed)
    elif args.baseline == "noisy_truthful_host":
        host_baseline = NoisyTruthfulHost(env.num_players, seed=args.seed)
    elif args.baseline == "greedy_players":
        player_baseline = GreedyPlayer()
    elif args.baseline == "no_bribe_players":
        player_baseline = NoBribePlayer()
    elif args.baseline is not None:
        raise ValueError(f"Unknown baseline {args.baseline!r}")

    ckpt_label = args.ckpt or args.host_ckpt or args.player_ckpt
    watch_episode(
        cfg,
        player_policy=player if has_learned and player_baseline is None else None,
        host_policy=host if has_learned and host_baseline is None else None,
        host_baseline=host_baseline,
        player_baseline=player_baseline,
        seed=args.seed,
        max_rounds=args.max_rounds,
        style=args.style,
        lang=args.lang,
        color=not args.no_color,
        ckpt_label=ckpt_label,
    )


if __name__ == "__main__":
    main()
