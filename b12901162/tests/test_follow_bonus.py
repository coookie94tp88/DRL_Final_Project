"""Reward shaping when players follow public/private signals."""

import numpy as np

from doorl import parallel_env
from doorl.baselines import GreedyPlayer, TruthfulHost
from doorl.env.doorl_env import PHASE_BET, PHASE_BRIBE, PHASE_HOST


def _run_one_settlement_round(*, follow_public_bonus: float) -> dict:
    env = parallel_env(
        num_players=2,
        max_rounds=5,
        initial_balance=100.0,
        follow_public_bonus=follow_public_bonus,
        follow_private_bonus=0.0,
        seed=0,
    )
    host = TruthfulHost(2, seed=0)
    greedy = GreedyPlayer(bribe_pct=0.0, bet_pct=0.5)
    obs, _ = env.reset(seed=0)

    actions = {
        "host": {"public_door": 0, "private_logits": np.zeros((2, 4))},
    }
    for i in range(2):
        actions[f"player_{i}"] = greedy.act(env, obs, i, PHASE_BRIBE)
    obs, _, _, _, _ = env.step(actions)

    actions = {"host": host.act(env, obs)}
    for i in range(2):
        actions[f"player_{i}"] = {
            "bribe_pct": np.array([0.0], dtype=np.float32),
            "door": 0,
            "bet_pct": np.array([0.0], dtype=np.float32),
        }
    obs, _, _, _, _ = env.step(actions)

    actions = {
        "host": {"public_door": 0, "private_logits": np.zeros((2, 4))},
    }
    for i in range(2):
        actions[f"player_{i}"] = greedy.act(env, obs, i, PHASE_BET)
    _, rewards, _, _, _ = env.step(actions)
    return rewards


def test_follow_public_bonus_adds_to_player_reward():
    r0 = _run_one_settlement_round(follow_public_bonus=0.0)
    r5 = _run_one_settlement_round(follow_public_bonus=5.0)
    for i in range(2):
        assert r5[f"player_{i}"] == r0[f"player_{i}"] + 5.0
