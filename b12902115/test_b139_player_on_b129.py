#!/usr/bin/env python3
"""Test b13902055 SB3 player checkpoint on b12902115 belief-based env.

The b139 model outputs door indices; b129 env expects beliefs (pub/priv/rnd).
Observations are adapted from b129 structured obs to the b139 Dict layout.
Host uses random signals (no host checkpoint bundled).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# b129 env (this package)
from env import OracleGambitConfig, OracleGambitEnv, Phase, PlayerBelief

# b139 PPO feature extractor + loader
B139_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "b13902055")
if B139_DIR not in sys.path:
    sys.path.insert(0, B139_DIR)

from train_both import PlayerExtractor  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402


def compute_door_ratios(env: OracleGambitEnv) -> np.ndarray:
    """Per history step: bet share on each door (W, num_doors)."""
    c = env.cfg
    ratios = np.zeros((c.history_window, c.num_doors), dtype=np.float32)
    for t in range(c.history_window):
        bets = env.hist_bets[t]
        choices = env.hist_chosen_doors[t]
        total = float(bets.sum())
        if total <= c.epsilon:
            continue
        for door in range(c.num_doors):
            mask = (choices == door) & (bets > 0)
            ratios[t, door] = float(bets[mask].sum() / total)
    return ratios


def b129_to_b139_player_obs(env: OracleGambitEnv) -> dict[str, np.ndarray]:
    """Map b12902115 player obs dict to b13902055 layout for SB3 PPO."""
    c = env.cfg
    n, w = c.num_players, c.history_window
    hist_dim = 7 + c.num_doors

    current = np.zeros((n, 5), dtype=np.float32)
    for i in range(n):
        if env.balances[i] <= 0:
            continue
        bal = float(env.balances[i])
        if c.normalize_balance_in_obs:
            bal /= max(c.initial_balance, c.epsilon)
        current[i] = [
            1.0,
            bal,
            float(env.current_bribes[i]),
            float(env.current_public_signal),
            float(env.current_private_signals[i]),
        ]

    door_ratios = compute_door_ratios(env)
    history = np.zeros((n, w, hist_dim), dtype=np.float32)
    for i in range(n):
        if env.balances[i] <= 0:
            continue
        step_features = np.stack(
            [
                env.hist_chosen_doors[:, i],
                env.hist_public_signal,
                env.hist_private_signals[:, i],
                env.hist_bribes[:, i],
                env.hist_bets[:, i],
                env.hist_player_rewards[:, i],
                env.hist_host_profit,
            ],
            axis=1,
        )
        history[i] = np.concatenate([step_features, door_ratios], axis=1).astype(np.float32)

    return {"current": current, "history": history}


def doors_to_beliefs(
    doors: np.ndarray,
    public_signal: int,
    private_signals: np.ndarray,
    bribes: np.ndarray,
) -> np.ndarray:
    """Heuristic: predicted door -> closest belief action in b129 env."""
    n = doors.shape[0]
    beliefs = np.full(n, PlayerBelief.RANDOM, dtype=np.int32)
    for i in range(n):
        if doors[i] == public_signal:
            beliefs[i] = PlayerBelief.BELIEVE_PUBLIC
        elif bribes[i] > 0 and private_signals[i] >= 0 and doors[i] == private_signals[i]:
            beliefs[i] = PlayerBelief.BELIEVE_PRIVATE
        else:
            beliefs[i] = PlayerBelief.RANDOM
    return beliefs


class B139PlayerAdapter:
    def __init__(self, model_path: str, num_players: int, num_doors: int, device: str = "cpu"):
        self.model = PPO.load(
            model_path,
            device=device,
            custom_objects={
                "policy_kwargs": {
                    "features_extractor_class": PlayerExtractor,
                    "features_extractor_kwargs": {},
                }
            },
        )
        self.num_players = num_players
        self.num_doors = num_doors

    def _predict(self, env: OracleGambitEnv, deterministic: bool = True) -> np.ndarray:
        obs_b139 = b129_to_b139_player_obs(env)
        action, _ = self.model.predict(obs_b139, deterministic=deterministic)
        return np.asarray(action, dtype=np.float32)

    def get_bribe_fractions(self, env: OracleGambitEnv, deterministic: bool = True) -> np.ndarray:
        action = self._predict(env, deterministic)
        n = self.num_players
        return np.clip(action[:n], 0.0, 1.0)

    def get_bet_action(
        self, env: OracleGambitEnv, deterministic: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        action = self._predict(env, deterministic)
        n = self.num_players
        bet_frac = np.clip(action[n : 2 * n], 0.0, 1.0)
        door_vals = np.clip(action[2 * n :], 0.0, self.num_doors - 1)
        doors = np.rint(door_vals).astype(np.int32)
        beliefs = doors_to_beliefs(
            doors,
            int(env.current_public_signal),
            env.current_private_signals.astype(int),
            env.current_bribes,
        )
        return beliefs, bet_frac


def run_episode(env: OracleGambitEnv, player: B139PlayerAdapter, rng: np.random.Generator) -> dict:
    obs, _ = env.reset()
    round_num = 0
    host_profit_sum = 0.0
    player_reward_sum = np.zeros(env.cfg.num_players, dtype=np.float64)

    while True:
        if env.phase == Phase.BRIBE:
            bribes = player.get_bribe_fractions(env, deterministic=True)
            obs, _, _, _, _ = env.step({"player_bribe_fractions": bribes})

        elif env.phase == Phase.SIGNAL:
            pub = int(rng.integers(0, env.cfg.num_doors))
            priv = rng.integers(0, env.cfg.num_doors, size=env.cfg.num_players).astype(np.int32)
            obs, _, _, _, _ = env.step({"public_signal": pub, "private_signals": priv})

        elif env.phase == Phase.BET:
            beliefs, bet_fracs = player.get_bet_action(env, deterministic=True)
            obs, rewards, terminated, truncated, info = env.step(
                {"player_beliefs": beliefs, "bet_fractions": bet_fracs}
            )
            round_num += 1
            host_profit_sum += float(rewards["host"])
            player_reward_sum += rewards["players"].astype(np.float64)

            if terminated or truncated:
                return {
                    "rounds": round_num,
                    "host_profit_sum": host_profit_sum,
                    "host_cumulative": float(env.host_cumulative_profit),
                    "player_reward_sum": player_reward_sum,
                    "final_balances": env.balances.copy(),
                    "winning_door": int(info["winning_door"]),
                }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test b139 PPO player on b129 env")
    parser.add_argument(
        "--model",
        default=os.path.join(B139_DIR, "player_model_800.zip"),
        help="Path to b139 SB3 player zip",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--num-doors", type=int, default=3, help="Must match training (3)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = OracleGambitConfig(
        num_players=10,
        num_doors=args.num_doors,
        max_rounds=args.max_rounds,
        initial_balance=1000.0,
    )
    env = OracleGambitEnv(config=config, seed=args.seed)
    player = B139PlayerAdapter(
        args.model, num_players=config.num_players, num_doors=config.num_doors, device=args.device
    )
    rng = np.random.default_rng(args.seed)

    print(f"Model: {args.model}")
    print(f"b129 env: players={config.num_players} doors={config.num_doors} rounds={config.max_rounds}")
    print("Host: random signals | Player: b139 PPO -> belief adapter\n")

    for ep in range(args.episodes):
        stats = run_episode(env, player, rng)
        alive = int(np.sum(stats["final_balances"] > 0))
        mean_bal = float(np.mean(stats["final_balances"]))
        print(
            f"Ep {ep + 1}: rounds={stats['rounds']} "
            f"host_cum={stats['host_cumulative']:+.1f} "
            f"host_round_sum={stats['host_profit_sum']:+.1f} "
            f"alive={alive}/10 mean_bal={mean_bal:.1f}"
        )


if __name__ == "__main__":
    main()
