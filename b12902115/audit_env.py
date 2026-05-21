#!/usr/bin/env python3
"""Runtime invariant checks for OracleGambitEnv."""

from __future__ import annotations

import sys

import numpy as np

from env import OracleGambitEnv, OracleGambitConfig, Phase, PlayerBelief

CURR_DIM, HIST_DIM, SEQ_LEN = 4, 12, 50
STATE_DIM = CURR_DIM + SEQ_LEN * HIST_DIM


def flatten_obs_np(obs_dict) -> np.ndarray:
    curr = obs_dict["current"]
    hist = obs_dict["history"]
    return np.concatenate([curr, hist.reshape(hist.shape[0], -1)], axis=1)


def run_audit(episodes: int = 200, seed: int = 42) -> bool:
    c = OracleGambitConfig(num_players=6, num_doors=4, max_rounds=30, history_window=SEQ_LEN)
    expected_flat = c.current_player_dim + c.history_window * c.hist_player_dim
    env = OracleGambitEnv(c, seed=seed)
    failures: list[str] = []

    obs, _ = env.reset(seed=seed)
    pc, ph = obs["players"]["current"].shape, obs["players"]["history"].shape
    if pc != (c.num_players, c.current_player_dim) or ph[2] != HIST_DIM:
        failures.append(f"player obs shape mismatch: current={pc}, history={ph}")
    flat = flatten_obs_np(obs["players"]).shape
    if flat != (c.num_players, expected_flat) or expected_flat != STATE_DIM:
        failures.append(f"flat dim {flat}, expected ({c.num_players}, {expected_flat})")

    rng = np.random.default_rng(seed + 1)
    for ep in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        done = False
        while not done:
            if env.phase == Phase.BRIBE:
                env.step_bribe(rng.random(c.num_players).astype(np.float32) * 0.3)
                obs = env._get_observations()
                for i in range(c.num_players):
                    if obs["players"]["current"][i, 0] > 0 and obs["players"]["current"][i, 3] != -1.0:
                        failures.append(f"ep{ep} BRIBE: signals_agree should be -1 before SIGNAL")

            elif env.phase == Phase.SIGNAL:
                pub = int(rng.integers(0, c.num_doors))
                priv = rng.integers(0, c.num_doors, size=c.num_players)
                env.step_signal(pub, priv)
                obs = env._get_observations()
                for i in range(c.num_players):
                    if obs["players"]["current"][i, 0] <= 0:
                        continue
                    bribe = obs["players"]["current"][i, 2]
                    ag = obs["players"]["current"][i, 3]
                    if bribe <= 0 and ag != -1.0:
                        failures.append(f"ep{ep} SIGNAL: agree={ag} with bribe=0")
                    if bribe > 0 and ag != float(pub == priv[i]):
                        failures.append(f"ep{ep} SIGNAL: agree mismatch")

            elif env.phase == Phase.BET:
                beliefs = rng.integers(0, 3, size=c.num_players)
                bet_frac = rng.random(c.num_players).astype(np.float32) * 0.5
                pre_pub = env.current_public_signal
                pre_priv = env.current_private_signals.copy()
                win = env.current_winning_door

                obs2, _, term, trunc, info = env.step_bet(beliefs, bet_frac)
                done = term or trunc
                chosen = info["chosen_doors"]

                m_pub = beliefs == PlayerBelief.BELIEVE_PUBLIC
                m_priv = beliefs == PlayerBelief.BELIEVE_PRIVATE
                if np.any(m_pub) and not np.all(chosen[m_pub] == pre_pub):
                    failures.append(f"ep{ep} BELIEVE_PUBLIC mapping failed")
                if np.any(m_priv) and not np.all(chosen[m_priv] == pre_priv[m_priv]):
                    failures.append(f"ep{ep} BELIEVE_PRIVATE mapping failed")

                bets = env.hist_bets[-1]
                n_bet = int(np.sum(bets > 0))
                if n_bet > 0:
                    alt = float(np.sum((bets > 0) & (chosen == win)) / n_bet)
                    if abs(info["frac_correct"] - alt) > 1e-5:
                        failures.append(f"ep{ep} frac_correct mismatch")
                    s = (
                        env.hist_frac_believe_public[-1]
                        + env.hist_frac_believe_private[-1]
                        + env.hist_frac_random[-1]
                    )
                    if abs(s - 1.0) > 1e-5:
                        failures.append(f"ep{ep} belief fractions sum to {s}")

                for i in range(c.num_players):
                    if obs2["players"]["current"][i, 0] <= 0:
                        continue
                    hit = obs2["players"]["history"][i, -1, 3]
                    b = env.hist_bribes[-1, i]
                    priv = env.hist_private_signals[-1, i]
                    if b <= 0 and hit != -1.0:
                        failures.append(f"ep{ep} p{i}: bribe_private_hit={hit} without bribe")
                    if b > 0 and hit != float(int(priv) == win):
                        failures.append(f"ep{ep} p{i}: bribe_private_hit wrong")

                obs = obs2

    if failures:
        print(f"AUDIT FAILED ({len(failures)} issues)")
        for f in failures[:10]:
            print(" ", f)
        return False
    print("AUDIT OK")
    return True


if __name__ == "__main__":
    sys.exit(0 if run_audit() else 1)
