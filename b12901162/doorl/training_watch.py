"""In-training episode snapshots so you can see game behavior while training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

import numpy as np
import torch

from doorl import parallel_env
from doorl.console_render import (
    _c,
    _door,
    _mark,
    _supports_color,
    print_episode_footer,
    print_settlement_compact,
    print_settlement_fancy,
    print_settlement_plain,
)
from doorl.env.doorl_env import PHASE_BET
from doorl.eval import _act_learned_host, _act_learned_player
from doorl.models import HostPolicy, PlayerPolicy
from doorl.watch import _build_actions


def _tee_print(line: str, log_file: Optional[TextIO]) -> None:
    print(line, flush=True)
    if log_file is not None:
        log_file.write(line + "\n")
        log_file.flush()


def run_training_watch(
    cfg: Dict[str, Any],
    player_policy: PlayerPolicy,
    host_policy: HostPolicy,
    *,
    iteration: int,
    global_step: int,
    seed: int = 0,
    max_rounds: int = 8,
    style: str = "compact",
    lang: str = "en",
    log_path: Optional[Path] = None,
    color: Optional[bool] = None,
) -> Dict[str, Any]:
    """Play one short episode with current weights; print settlements to stdout."""
    was_training = player_policy.training
    player_policy.eval()
    host_policy.eval()

    env = parallel_env(**cfg["env"])
    env.cfg.max_rounds = int(max_rounds)

    use_color = _supports_color() if color is None else color
    log_file: Optional[TextIO] = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "a", encoding="utf-8")

    settlements: List[Dict[str, Any]] = []
    host_cumulative = 0.0
    try:
        with torch.no_grad():
            _tee_print("", log_file)
            _tee_print(
                _c(
                    "\033[35m",
                    f"════ TRAINING WATCH  iter={iteration}  step={global_step}  "
                    f"seed={seed}  rounds={max_rounds} ════",
                    use_color,
                ),
                log_file,
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
                    host_baseline=None,
                    player_baseline=None,
                )
                phase_before = env._phase
                obs, _, terms, _, _ = env.step(actions)
                done = all(terms.values())
                if phase_before == PHASE_BET and env.last_settlement is not None:
                    s = env.last_settlement
                    host_cumulative += float(s.reward_host)
                    balances = env._balances.copy()
                    settlements.append(
                        {
                            "round": int(s.round_idx),
                            "true": _door(s.true_door),
                            "public": _door(s.public_signal),
                            "pub_ok": bool(s.public_signal == s.true_door),
                            "x": float(s.x),
                            "host_r": float(s.reward_host),
                            "alive": int(np.sum(balances >= env.cfg.min_bet)),
                        }
                    )
                    if style == "fancy":
                        print_settlement_fancy(
                            s,
                            env.cfg,
                            balances=balances,
                            host_cumulative=host_cumulative,
                            color=use_color,
                            lang=lang,
                        )
                    elif style == "plain":
                        print_settlement_plain(
                            s,
                            env.cfg,
                            balances=balances,
                            host_cumulative=host_cumulative,
                            lang=lang,
                        )
                    else:
                        line = print_settlement_compact(
                            s,
                            env.cfg,
                            balances=balances,
                            host_cumulative=host_cumulative,
                            color=use_color,
                            lang=lang,
                        )
                        _tee_print(line, log_file)

            if style == "compact":
                alive = int(np.sum(env._balances >= env.cfg.min_bet))
                _tee_print(
                    f"  → episode end: host_total={host_cumulative:+.1f}  "
                    f"alive={alive}/{env.num_players}",
                    log_file,
                )
            elif style in ("fancy", "plain"):
                print_episode_footer(
                    host_total=host_cumulative,
                    balances=env._balances.copy(),
                    cfg=env.cfg,
                    lang=lang,
                    color=use_color,
                )
            _tee_print(
                _c("\033[35m", "════ END TRAINING WATCH ════\n", use_color),
                log_file,
            )
    finally:
        if log_file is not None:
            log_file.close()
        if was_training:
            player_policy.train()
            host_policy.train()

    return {"settlements": settlements, "host_total": host_cumulative}
