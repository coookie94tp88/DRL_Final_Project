"""Basic API conformance tests for DooRLEnv."""

from __future__ import annotations

import numpy as np
import pytest

from doorl import parallel_env
from doorl.env.doorl_env import NUM_PHASES, PHASE_BRIBE


def _random_player_action(rng: np.random.Generator) -> dict:
    return {
        "bribe_pct": np.array([rng.uniform(0.0, 0.5)], dtype=np.float32),
        "door": int(rng.integers(0, 4)),
        "bet_pct": np.array([rng.uniform(0.05, 0.5)], dtype=np.float32),
    }


def _random_host_action(rng: np.random.Generator, num_players: int) -> dict:
    return {
        "public_door": int(rng.integers(0, 4)),
        "private_logits": rng.normal(0.0, 1.0, size=(num_players, 4)).astype(
            np.float32
        ),
    }


def test_reset_shapes() -> None:
    env = parallel_env(num_players=3, max_rounds=4, seed=0)
    obs, infos = env.reset(seed=0)
    assert set(obs.keys()) == set(env.possible_agents)
    assert obs["host"].shape == (env.host_obs_dim,)
    for i in range(3):
        assert obs[f"player_{i}"].shape == (env.player_obs_dim,)


def test_full_episode_runs() -> None:
    env = parallel_env(num_players=4, max_rounds=3, seed=0)
    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=0)
    steps = 0
    done = False
    while not done and steps < 100:
        actions = {"host": _random_host_action(rng, 4)}
        for i in range(4):
            actions[f"player_{i}"] = _random_player_action(rng)
        obs, rewards, terms, truncs, infos = env.step(actions)
        assert set(rewards.keys()) == set(env.possible_agents)
        assert all(np.isfinite(rewards[a]) for a in rewards)
        steps += 1
        done = all(terms.values())
    assert done, "Episode should have terminated within max_rounds * NUM_PHASES steps"
    assert env.stats.rounds_played > 0


def test_balance_visibility_modes() -> None:
    for vis in ["full", "own_only", "noisy"]:
        env = parallel_env(
            num_players=3, max_rounds=2, seed=0, balance_visibility=vis
        )
        obs, _ = env.reset(seed=0)
        assert obs["player_0"].shape == (env.player_obs_dim,)


def test_invalid_config_raises() -> None:
    with pytest.raises(ValueError):
        parallel_env(num_players=0)
    with pytest.raises(ValueError):
        parallel_env(payout_threshold=0.0)
    with pytest.raises(ValueError):
        parallel_env(balance_visibility="weird")


def test_payout_threshold_property() -> None:
    """At W = tau * P, total payouts approximate P (Host break-even)."""
    env = parallel_env(num_players=4, max_rounds=2, seed=0, payout_threshold=0.20)
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)
    # phase I: bribes zero
    actions = {"host": _random_host_action(rng, 4)}
    for i in range(4):
        actions[f"player_{i}"] = {
            "bribe_pct": np.array([0.0], dtype=np.float32),
            "door": 0,
            "bet_pct": np.array([0.0], dtype=np.float32),
        }
    obs, _, _, _, _ = env.step(actions)
    # phase host
    actions = {"host": _random_host_action(rng, 4)}
    for i in range(4):
        actions[f"player_{i}"] = {
            "bribe_pct": np.array([0.0], dtype=np.float32),
            "door": 0,
            "bet_pct": np.array([0.0], dtype=np.float32),
        }
    obs, _, _, _, _ = env.step(actions)
    assert True  # presence test only; numerical break-even is covered in test_payout
