"""
Quick smoke-test for OracleGambitEnv.
Run: python env/test_env.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from env.oracle_gambit_env import OracleGambitEnv


def random_action(env: OracleGambitEnv, agent: str) -> np.ndarray:
    space = env.action_space(agent)
    return space.sample()


def test_basic(num_rounds: int = 10) -> None:
    env = OracleGambitEnv(num_players=4, max_rounds=num_rounds, seed=42)
    env.reset(seed=42)

    print("=== Observation space sizes ===")
    for agent in env.possible_agents:
        print(f"  {agent}: obs={env.observation_space(agent).shape}  "
              f"act={env.action_space(agent).shape}")

    print(f"\n=== Running {num_rounds} rounds via step_all() ===")
    for r in range(num_rounds):
        # Random host action (signal fractions)
        host_act = np.random.default_rng(r).random(1 + env.num_players).astype(np.float32)

        # Phase 1 actions: bribe fractions for players + host action
        actions_p1 = {"host": host_act}
        for pid in range(env.num_players):
            name = f"player_{pid}"
            actions_p1[name] = float(np.random.default_rng(r + pid).random())

        # Phase 2 actions: (door_frac, bet_frac) for players
        actions_p2 = {}
        for pid in range(env.num_players):
            name = f"player_{pid}"
            rng = np.random.default_rng(r * 100 + pid)
            actions_p2[name] = (float(rng.random()), float(rng.random()))

        rewards = env.step_all(actions_p1, actions_p2)
        env.render()
        print(f"  rewards: { {k: f'{v:.2f}' for k, v in rewards.items()} }")

    print("\n=== Observation sanity check ===")
    obs_host = env.observe("host")
    obs_p0 = env.observe("player_0")
    print(f"  host obs shape:     {obs_host.shape}  contains NaN: {np.isnan(obs_host).any()}")
    print(f"  player_0 obs shape: {obs_p0.shape}  contains NaN: {np.isnan(obs_p0).any()}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    test_basic()