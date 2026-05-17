"""End-to-end random-policy smoke test for DooRLEnv."""

from __future__ import annotations

import numpy as np

from doorl import parallel_env


def _random_actions(rng: np.random.Generator, num_players: int) -> dict:
    actions = {
        "host": {
            "public_door": int(rng.integers(0, 4)),
            "private_logits": rng.normal(0.0, 1.0, size=(num_players, 4)).astype(
                np.float32
            ),
        }
    }
    for i in range(num_players):
        actions[f"player_{i}"] = {
            "bribe_pct": np.array([rng.uniform(0.0, 0.3)], dtype=np.float32),
            "door": int(rng.integers(0, 4)),
            "bet_pct": np.array([rng.uniform(0.05, 0.4)], dtype=np.float32),
        }
    return actions


def test_random_policy_smoke() -> None:
    num_players = 4
    env = parallel_env(num_players=num_players, max_rounds=10, seed=0)
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)
    total_host = 0.0
    total_player = np.zeros(num_players, dtype=np.float64)
    steps = 0
    done = False
    while not done and steps < 1000:
        actions = _random_actions(rng, num_players)
        obs, rewards, terms, _, _ = env.step(actions)
        total_host += rewards["host"]
        for i in range(num_players):
            total_player[i] += rewards[f"player_{i}"]
        steps += 1
        done = all(terms.values())
    assert env.stats.rounds_played > 0
    assert steps <= 3 * env.cfg.max_rounds + 1
    assert np.isfinite(total_host)
    assert np.all(np.isfinite(total_player))
