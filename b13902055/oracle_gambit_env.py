from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class OracleGambitConfig:
    """All tunable environment parameters are defined here before use."""

    num_players: int = 10
    num_doors: int = 4
    initial_balance: float = 1000.0
    minority_threshold: float = 0.20
    history_window: int = 50
    payout_threshold: float = 0.20
    min_winning_ratio_for_payout: float = 1e-3
    max_rounds: int = 500
    min_bet_fraction: float = 0.0
    max_bet_fraction: float = 1.0
    min_bribe: float = 0.0
    max_bribe: float = 1e6
    epsilon: float = 1e-8
    normalize_balance_in_obs: bool = True

    @property
    def surplus_coefficient(self) -> float:
        return 1.0 - self.payout_threshold


class OracleGambitEnv(gym.Env):
    """
    Multi-agent style environment for future DRL training.

    Step input is a joint action dictionary containing both player actions and host signals.
    Observations are fixed-size vectors to keep dimensions stable for sequence models.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: OracleGambitConfig | None = None, seed: int | None = None) -> None:
        super().__init__()
        self.cfg = config or OracleGambitConfig()
        self.rng = np.random.default_rng(seed)

        self._build_spaces()
        self._init_state()

    # -------------------------
    # Public API
    # -------------------------
    @property
    def player_observation_dim(self) -> int:
        """Fixed per-player observation vector length."""
        c = self.cfg
        # current features: alive, balance, last_bribe, current_public_signal, current_private_signal
        current = 5
        # history features per step: choice, public_signal, private_signal, bribe, bet, reward, host_profit + door_ratios(num_doors)
        hist = 7 + c.num_doors
        return current + c.history_window * hist

    @property
    def host_observation_dim(self) -> int:
        """Fixed host observation vector length."""
        c = self.cfg
        # current features: cumulative_profit, current_pool, current_bribes, winning_door_onehot(num_doors)
        current = 3 + c.num_doors
        # player snapshot: balances(num_players), active_mask(num_players)
        players = 2 * c.num_players
        # history per step: host_profit, public_signal, door_ratios(num_doors), private_signal_hist(num_doors)
        hist = 2 + c.num_doors + c.num_doors
        return current + players + c.history_window * hist

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._init_state()
        return self._get_observation(), self._get_info()

    def step(self, action: dict[str, np.ndarray | int | float]):
        c = self.cfg
        self.round_idx += 1

        player_bribes = np.asarray(action["player_bribes"], dtype=np.float32)
        player_doors = np.asarray(action["player_doors"], dtype=np.int32)
        player_bet_fractions = np.asarray(action["player_bet_fractions"], dtype=np.float32)
        host_public_signal = int(action["host_public_signal"])
        host_private_signals = np.asarray(action["host_private_signals"], dtype=np.int32)

        self._validate_action_shapes(player_bribes, player_doors, player_bet_fractions, host_private_signals)

        active = self.balances > 0

        clamped_bribes = np.where(
            active,
            np.minimum(np.maximum(player_bribes, c.min_bribe), np.minimum(self.balances, c.max_bribe)),
            0.0,
        )
        self.balances = self.balances - clamped_bribes

        clamped_bet_fractions = np.clip(player_bet_fractions, c.min_bet_fraction, c.max_bet_fraction)
        bets = np.where(active, self.balances * clamped_bet_fractions, 0.0)
        self.balances = self.balances - bets

        chosen_doors = np.clip(player_doors, 0, c.num_doors - 1)
        winning_door = int(self.rng.integers(0, c.num_doors))

        total_pool = float(np.sum(bets))
        winner_mask = active & (chosen_doors == winning_door) & (bets > 0)
        total_winning_vol = float(np.sum(bets[winner_mask]))

        payouts = np.zeros(c.num_players, dtype=np.float32)
        if total_pool > c.epsilon and total_winning_vol > c.epsilon:
            x = total_winning_vol / total_pool
            multiplier = 1.0 + (c.surplus_coefficient / max(x, c.min_winning_ratio_for_payout))
            payouts[winner_mask] = bets[winner_mask] * multiplier

        self.balances = self.balances + payouts

        total_payout = float(np.sum(payouts))
        total_bribes = float(np.sum(clamped_bribes))
        host_reward = total_pool - total_payout + total_bribes
        self.host_cumulative_profit += host_reward

        player_rewards = payouts - bets - clamped_bribes

        door_ratios = np.zeros(c.num_doors, dtype=np.float32)
        if total_pool > c.epsilon:
            for d in range(c.num_doors):
                door_ratios[d] = float(np.sum(bets[chosen_doors == d]) / total_pool)

        self._push_history(
            choices=chosen_doors,
            public_signal=host_public_signal,
            private_signals=np.clip(host_private_signals, 0, c.num_doors - 1),
            bribes=clamped_bribes,
            bets=bets,
            player_rewards=player_rewards,
            host_profit=host_reward,
            door_ratios=door_ratios,
            winning_door=winning_door,
        )

        self.current_public_signal = np.clip(host_public_signal, 0, c.num_doors - 1)
        self.current_private_signals = np.clip(host_private_signals, 0, c.num_doors - 1)
        self.current_total_pool = total_pool
        self.current_total_bribes = total_bribes
        self.current_winning_door = winning_door

        terminated = bool(self.round_idx >= c.max_rounds or np.all(self.balances <= 0))
        truncated = False

        rewards = {"players": player_rewards.astype(np.float32), "host": float(host_reward)}
        obs = self._get_observation()
        info = self._get_info()
        return obs, rewards, terminated, truncated, info

    # -------------------------
    # Internal methods
    # -------------------------
    def _build_spaces(self) -> None:
        c = self.cfg
        self.observation_space = spaces.Dict(
            {
                "players": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(c.num_players, self.player_observation_dim),
                    dtype=np.float32,
                ),
                "host": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.host_observation_dim,),
                    dtype=np.float32,
                ),
            }
        )

        self.action_space = spaces.Dict(
            {
                "player_bribes": spaces.Box(
                    low=c.min_bribe,
                    high=c.max_bribe,
                    shape=(c.num_players,),
                    dtype=np.float32,
                ),
                "player_doors": spaces.MultiDiscrete([c.num_doors] * c.num_players),
                "player_bet_fractions": spaces.Box(
                    low=c.min_bet_fraction,
                    high=c.max_bet_fraction,
                    shape=(c.num_players,),
                    dtype=np.float32,
                ),
                "host_public_signal": spaces.Discrete(c.num_doors),
                "host_private_signals": spaces.MultiDiscrete([c.num_doors] * c.num_players),
            }
        )

    def _init_state(self) -> None:
        c = self.cfg
        self.round_idx = 0
        self.balances = np.full(c.num_players, c.initial_balance, dtype=np.float32)
        self.host_cumulative_profit = 0.0

        self.current_public_signal = 0
        self.current_private_signals = np.zeros(c.num_players, dtype=np.int32)
        self.current_total_pool = 0.0
        self.current_total_bribes = 0.0
        self.current_winning_door = 0

        self.hist_choices = np.full((c.history_window, c.num_players), -1, dtype=np.int32)
        self.hist_public_signal = np.full(c.history_window, -1, dtype=np.int32)
        self.hist_private_signals = np.full((c.history_window, c.num_players), -1, dtype=np.int32)
        self.hist_bribes = np.zeros((c.history_window, c.num_players), dtype=np.float32)
        self.hist_bets = np.zeros((c.history_window, c.num_players), dtype=np.float32)
        self.hist_player_rewards = np.zeros((c.history_window, c.num_players), dtype=np.float32)
        self.hist_host_profit = np.zeros(c.history_window, dtype=np.float32)
        self.hist_door_ratios = np.zeros((c.history_window, c.num_doors), dtype=np.float32)
        self.hist_winning_door = np.full(c.history_window, -1, dtype=np.int32)

    def _push_history(
        self,
        *,
        choices: np.ndarray,
        public_signal: int,
        private_signals: np.ndarray,
        bribes: np.ndarray,
        bets: np.ndarray,
        player_rewards: np.ndarray,
        host_profit: float,
        door_ratios: np.ndarray,
        winning_door: int,
    ) -> None:
        self.hist_choices = np.roll(self.hist_choices, shift=-1, axis=0)
        self.hist_public_signal = np.roll(self.hist_public_signal, shift=-1, axis=0)
        self.hist_private_signals = np.roll(self.hist_private_signals, shift=-1, axis=0)
        self.hist_bribes = np.roll(self.hist_bribes, shift=-1, axis=0)
        self.hist_bets = np.roll(self.hist_bets, shift=-1, axis=0)
        self.hist_player_rewards = np.roll(self.hist_player_rewards, shift=-1, axis=0)
        self.hist_host_profit = np.roll(self.hist_host_profit, shift=-1, axis=0)
        self.hist_door_ratios = np.roll(self.hist_door_ratios, shift=-1, axis=0)
        self.hist_winning_door = np.roll(self.hist_winning_door, shift=-1, axis=0)

        self.hist_choices[-1] = choices
        self.hist_public_signal[-1] = public_signal
        self.hist_private_signals[-1] = private_signals
        self.hist_bribes[-1] = bribes
        self.hist_bets[-1] = bets
        self.hist_player_rewards[-1] = player_rewards
        self.hist_host_profit[-1] = host_profit
        self.hist_door_ratios[-1] = door_ratios
        self.hist_winning_door[-1] = winning_door

    def _player_obs_for(self, player_idx: int) -> np.ndarray:
        c = self.cfg
        if self.balances[player_idx] <= 0:
            # Requirement: if out of game, state is represented by 0.
            return np.zeros(self.player_observation_dim, dtype=np.float32)

        balance = self.balances[player_idx]
        if c.normalize_balance_in_obs:
            balance = balance / max(c.initial_balance, c.epsilon)

        current = np.array(
            [
                1.0,
                float(balance),
                float(self.hist_bribes[-1, player_idx]),
                float(self.current_public_signal),
                float(self.current_private_signals[player_idx]),
            ],
            dtype=np.float32,
        )

        hist_choices = self.hist_choices[:, player_idx].astype(np.float32)
        hist_pub = self.hist_public_signal.astype(np.float32)
        hist_priv = self.hist_private_signals[:, player_idx].astype(np.float32)
        hist_bribe = self.hist_bribes[:, player_idx]
        hist_bet = self.hist_bets[:, player_idx]
        hist_reward = self.hist_player_rewards[:, player_idx]
        hist_host_profit = self.hist_host_profit

        hist = np.concatenate(
            [
                hist_choices[:, None],
                hist_pub[:, None],
                hist_priv[:, None],
                hist_bribe[:, None],
                hist_bet[:, None],
                hist_reward[:, None],
                hist_host_profit[:, None],
                self.hist_door_ratios,
            ],
            axis=1,
        ).reshape(-1)

        return np.concatenate([current, hist.astype(np.float32)], axis=0)

    def _host_obs(self) -> np.ndarray:
        c = self.cfg

        winning_one_hot = np.zeros(c.num_doors, dtype=np.float32)
        if 0 <= self.current_winning_door < c.num_doors:
            winning_one_hot[self.current_winning_door] = 1.0

        balances = self.balances.copy()
        if c.normalize_balance_in_obs:
            balances = balances / max(c.initial_balance, c.epsilon)

        active_mask = (self.balances > 0).astype(np.float32)

        hist_private_hist = np.zeros((c.history_window, c.num_doors), dtype=np.float32)
        for t in range(c.history_window):
            valid = self.hist_private_signals[t] >= 0
            if np.any(valid):
                counts = np.bincount(self.hist_private_signals[t, valid], minlength=c.num_doors).astype(np.float32)
                hist_private_hist[t] = counts / max(np.sum(counts), 1.0)

        hist_block = np.concatenate(
            [
                self.hist_host_profit[:, None],
                self.hist_public_signal.astype(np.float32)[:, None],
                self.hist_door_ratios,
                hist_private_hist,
            ],
            axis=1,
        ).reshape(-1)

        current = np.concatenate(
            [
                np.array(
                    [
                        self.host_cumulative_profit,
                        self.current_total_pool,
                        self.current_total_bribes,
                    ],
                    dtype=np.float32,
                ),
                winning_one_hot,
            ]
        )

        return np.concatenate([current, balances.astype(np.float32), active_mask, hist_block.astype(np.float32)])

    def _get_observation(self) -> dict[str, np.ndarray]:
        players_obs = np.stack([self._player_obs_for(i) for i in range(self.cfg.num_players)], axis=0)
        host_obs = self._host_obs().astype(np.float32)
        return {"players": players_obs.astype(np.float32), "host": host_obs}

    def _get_info(self) -> dict[str, Any]:
        return {
            "round": self.round_idx,
            "active_players": int(np.sum(self.balances > 0)),
            "host_cumulative_profit": float(self.host_cumulative_profit),
            "player_obs_dim": self.player_observation_dim,
            "host_obs_dim": self.host_observation_dim,
        }

    def _validate_action_shapes(
        self,
        player_bribes: np.ndarray,
        player_doors: np.ndarray,
        player_bet_fractions: np.ndarray,
        host_private_signals: np.ndarray,
    ) -> None:
        n = self.cfg.num_players
        if player_bribes.shape != (n,):
            raise ValueError(f"player_bribes must have shape ({n},), got {player_bribes.shape}")
        if player_doors.shape != (n,):
            raise ValueError(f"player_doors must have shape ({n},), got {player_doors.shape}")
        if player_bet_fractions.shape != (n,):
            raise ValueError(
                f"player_bet_fractions must have shape ({n},), got {player_bet_fractions.shape}"
            )
        if host_private_signals.shape != (n,):
            raise ValueError(
                f"host_private_signals must have shape ({n},), got {host_private_signals.shape}"
            )
