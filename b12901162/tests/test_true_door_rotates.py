"""Each game round must draw a new true door."""

from __future__ import annotations

import numpy as np

from doorl import parallel_env
from doorl.env.doorl_env import PHASE_BET, PHASE_BRIBE, PHASE_HOST


def _step_full_round(env, rng: np.random.Generator) -> None:
    n = env.num_players
    for phase_expected, builder in (
        (PHASE_BRIBE, lambda: _player_bribe_only(n, rng)),
        (PHASE_HOST, lambda: _host_only(n, rng)),
        (PHASE_BET, lambda: _all_bet(n, rng)),
    ):
        assert env._phase == phase_expected
        env.step(builder())


def _player_bribe_only(n: int, rng: np.random.Generator) -> dict:
    actions = {
        "host": {"public_door": 0, "private_logits": np.zeros((n, 4), np.float32)}
    }
    for i in range(n):
        actions[f"player_{i}"] = {
            "bribe_pct": np.array([0.05], np.float32),
            "door": 0,
            "bet_pct": np.array([0.1], np.float32),
        }
    return actions


def _host_only(n: int, rng: np.random.Generator) -> dict:
    actions = {
        "host": {
            "public_door": 0,
            "private_logits": rng.normal(0, 1, (n, 4)).astype(np.float32),
        }
    }
    for i in range(n):
        actions[f"player_{i}"] = {
            "bribe_pct": np.array([0.0], np.float32),
            "door": 0,
            "bet_pct": np.array([0.0], np.float32),
        }
    return actions


def _all_bet(n: int, rng: np.random.Generator) -> dict:
    actions = {
        "host": {"public_door": 0, "private_logits": np.zeros((n, 4), np.float32)}
    }
    for i in range(n):
        actions[f"player_{i}"] = {
            "bribe_pct": np.array([0.0], np.float32),
            "door": int(rng.integers(0, 4)),
            "bet_pct": np.array([0.1], np.float32),
        }
    return actions


def test_true_door_changes_each_round() -> None:
    env = parallel_env(num_players=2, max_rounds=5, seed=0)
    rng = np.random.default_rng(1)
    env.reset(seed=0)
    doors = []
    for _ in range(4):
        _step_full_round(env, rng)
        assert env.last_settlement is not None
        doors.append(env.last_settlement.true_door)
    assert len(set(doors)) > 1, f"true doors should vary, got {doors}"
