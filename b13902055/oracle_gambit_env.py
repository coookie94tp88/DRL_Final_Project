from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# ─────────────────────────────────────────────────────────────
# Phase enum
# ─────────────────────────────────────────────────────────────

class Phase(IntEnum):
    """The three sub-phases that make up one game round."""
    BRIBE = 0   # players submit bribes → host sees them
    SIGNAL = 1  # host broadcasts public + private signals → players see them
    BET = 2     # players choose door + bet fraction → settlement


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OracleGambitConfig:
    """All tunable environment parameters — define here before using anywhere."""

    # ── Game rules ──────────────────────────────────────────
    num_players: int = 10
    num_doors: int = 4
    initial_balance: float = 1000.0
    minority_threshold: float = 0.20    # host profits if winning_ratio < this
    max_rounds: int = 500

    # ── History buffer ──────────────────────────────────────
    history_window: int = 50            # number of past rounds kept in obs

    # ── Payout formula ──────────────────────────────────────
    payout_threshold: float = 0.20              # break-even winning ratio
    min_winning_ratio_for_payout: float = 1e-3  # floor for division stability
    max_payout_multiplier: float | None = 100.0  # optional cap on multiplier

    # ── Action bounds ───────────────────────────────────────
    min_bet_fraction: float = 0.0
    max_bet_fraction: float = 1.0
    min_bribe_fraction: float = 0.0
    max_bribe_fraction: float = 1.0

    # ── Misc ────────────────────────────────────────────────
    epsilon: float = 1e-8
    normalize_balance_in_obs: bool = True

    # ── Derived ─────────────────────────────────────────────
    @property
    def surplus_coefficient(self) -> float:
        return 1.0 - self.payout_threshold

    @property
    def current_player_dim(self) -> int:
        """Per-player current-context vector length.
        Fields: [alive, balance, bribe_sent, public_signal, private_signal]
        Padding -1 is used for fields not yet available in the current phase.
        """
        return 5

    @property
    def hist_player_dim(self) -> int:
        """Feature count per time-step in a player's history buffer.
        Fields: choice, public_signal, private_signal, bribe, bet, reward,
                host_profit, door_ratio_0 … door_ratio_{num_doors-1}
        """
        return 7 + self.num_doors

    @property
    def current_host_dim(self) -> int:
        """Length of the host's current-context feature vector.
        Fields: cumulative_profit, current_total_bribes,
                winning_door_onehot (num_doors values)
        """
        return 2 + self.num_doors

    @property
    def host_player_state_dim(self) -> int:
        """Per-player state dimension visible to the host.
        Fields: [normalized_balance, active_flag, bribe_this_round]
        """
        return 3

    @property
    def hist_host_dim(self) -> int:
        """Feature count per time-step in the host's history buffer.
        Fields: host_profit, public_signal,
                door_ratio_0 … door_ratio_{num_doors-1},
                private_signal_distribution_0 … (num_doors values)
        """
        return 2 + 2 * self.num_doors


# ─────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────

class OracleGambitEnv(gym.Env):
    """OracleGambit: Strategic Information Manipulation in Multi-Agent RL.

    Round flow (three phases per round):
        1. step_bribe(player_bribes)
               Players simultaneously submit bribes to the host.
               Returns host observation (host now sees all bribe amounts).

        2. step_signal(public_signal, private_signals)
               Host broadcasts one public signal and one private signal per player.
               Returns player observations (players now have signal context).

        3. step_bet(player_doors, player_bet_fractions)
               Players choose a door and a bet fraction of their balance.
               Effective bribe/bet amounts are integer dollars via np.floor.
               Settlement is computed; rewards and new observations are returned.

    Separate action spaces (do NOT mix):
        player_bribe_action_space  — used in Phase.BRIBE
        player_bet_action_space    — used in Phase.BET
        host_action_space          — used in Phase.SIGNAL

    Observations are structured dicts (NOT flattened) to allow Transformer
    sequence models to process the history dimension directly:
        player_observation_space:
            current : (num_players, current_player_dim)   — phase-context scalars
            history : (num_players, history_window, hist_player_dim) — past rounds

        host_observation_space:
            current : (current_host_dim,)                 — global scalars
            players : (num_players, host_player_state_dim)— per-player snapshot
            history : (history_window, hist_host_dim)     — past rounds
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: OracleGambitConfig | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.cfg = config or OracleGambitConfig()
        self.rng = np.random.default_rng(seed)
        self._build_spaces()
        self._init_state()

    # ─────────────────────────────────────────────
    # Action / observation spaces
    # ─────────────────────────────────────────────

    def _build_spaces(self) -> None:
        c = self.cfg

        # ── Player action spaces ────────────────────────────────
        # Phase.BRIBE
        self.player_bribe_action_space = spaces.Box(
            low=c.min_bribe_fraction,
            high=c.max_bribe_fraction,
            shape=(c.num_players,),
            dtype=np.float32,
        )
        # Phase.BET
        self.player_bet_action_space = spaces.Dict({
            "doors": spaces.MultiDiscrete([c.num_doors] * c.num_players),
            "bet_fractions": spaces.Box(
                low=c.min_bet_fraction,
                high=c.max_bet_fraction,
                shape=(c.num_players,),
                dtype=np.float32,
            ),
        })

        # ── Host action space ───────────────────────────────────
        # Phase.SIGNAL
        self.host_action_space = spaces.Dict({
            "public_signal": spaces.Discrete(c.num_doors),
            "private_signals": spaces.MultiDiscrete([c.num_doors] * c.num_players),
        })

        # ── Player observation space (structured, NOT flattened) ─
        self.player_observation_space = spaces.Dict({
            # current context: 2-D so each row is one player
            "current": spaces.Box(
                low=-1.0,
                high=np.inf,
                shape=(c.num_players, c.current_player_dim),
                dtype=np.float32,
            ),
            # history: 3-D — (players, time-steps, features)
            "history": spaces.Box(
                low=-1.0,
                high=np.inf,
                shape=(c.num_players, c.history_window, c.hist_player_dim),
                dtype=np.float32,
            ),
        })

        # ── Host observation space (structured, NOT flattened) ───
        self.host_observation_space = spaces.Dict({
            "current": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(c.current_host_dim,),
                dtype=np.float32,
            ),
            # per-player snapshot: 2-D — (players, features)
            "players": spaces.Box(
                low=-1.0,
                high=np.inf,
                shape=(c.num_players, c.host_player_state_dim),
                dtype=np.float32,
            ),
            # history: 2-D — (time-steps, features)
            "history": spaces.Box(
                low=-1.0,
                high=np.inf,
                shape=(c.history_window, c.hist_host_dim),
                dtype=np.float32,
            ),
        })

        # gym.Env required attributes (combined for framework compatibility)
        self.observation_space = spaces.Dict({
            "players": self.player_observation_space,
            "host": self.host_observation_space,
        })
        # action_space is intentionally a placeholder; use the phase-specific
        # spaces (player_bribe_action_space, player_bet_action_space,
        # host_action_space) when building agents.
        self.action_space = spaces.Dict({
            "player_bribe_fractions": self.player_bribe_action_space,
            "player_bet": self.player_bet_action_space,
            "host": self.host_action_space,
        })

    # ─────────────────────────────────────────────
    # State initialisation
    # ─────────────────────────────────────────────

    def _init_state(self) -> None:
        c = self.cfg
        self.round_idx: int = 0
        self.phase: Phase = Phase.BRIBE
        self.balances: np.ndarray = np.full(c.num_players, c.initial_balance, dtype=np.float32)
        self.host_cumulative_profit: float = 0.0

        # Within-round transient state (reset at start of every round)
        self.current_bribes: np.ndarray = np.zeros(c.num_players, dtype=np.float32)
        self.current_total_bribes: float = 0.0
        self.current_public_signal: int = -1
        self.current_private_signals: np.ndarray = np.full(c.num_players, -1, dtype=np.int32)
        self.current_winning_door: int = -1

        # History buffers (oldest → newest, index -1 is most recent)
        # -1.0 padding indicates no data for that slot yet
        self.hist_choices: np.ndarray = np.full(
            (c.history_window, c.num_players), -1.0, dtype=np.float32)
        self.hist_public_signal: np.ndarray = np.full(
            (c.history_window,), -1.0, dtype=np.float32)
        self.hist_private_signals: np.ndarray = np.full(
            (c.history_window, c.num_players), -1.0, dtype=np.float32)
        self.hist_bribes: np.ndarray = np.zeros(
            (c.history_window, c.num_players), dtype=np.float32)
        self.hist_bets: np.ndarray = np.zeros(
            (c.history_window, c.num_players), dtype=np.float32)
        self.hist_player_rewards: np.ndarray = np.zeros(
            (c.history_window, c.num_players), dtype=np.float32)
        self.hist_host_profit: np.ndarray = np.zeros(
            (c.history_window,), dtype=np.float32)
        self.hist_door_ratios: np.ndarray = np.zeros(
            (c.history_window, c.num_doors), dtype=np.float32)

    # ─────────────────────────────────────────────
    # Public step API
    # ─────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._init_state()
        # Pre-draw the winning door for round 1 (host knows it from the start)
        self.current_winning_door = int(self.rng.integers(0, self.cfg.num_doors))
        return self._get_observations(), self._get_info()

    def step_bribe(
        self,
        player_bribe_fractions: np.ndarray,
    ) -> tuple[dict, dict, bool, bool, dict]:
        """Phase 1 — Players submit bribes.

        Args:
            player_bribe_fractions: shape (num_players,), each value is a fraction in [0, 1]
                of that player's current balance; paid bribe is floored to integer dollars.

        Returns (obs, {}, False, False, info).
        The returned obs carries updated *host* context (host now sees the bribes).
        """
        if self.phase != Phase.BRIBE:
            raise RuntimeError(
                f"step_bribe() must be called in Phase.BRIBE; current phase: {self.phase.name}"
            )
        c = self.cfg
        player_bribe_fractions = np.asarray(player_bribe_fractions, dtype=np.float32)
        if player_bribe_fractions.shape != (c.num_players,):
            raise ValueError(
                f"player_bribe_fractions must have shape ({c.num_players},), got {player_bribe_fractions.shape}"
            )

        active = self.balances > 0
        clamped_fractions = np.clip(
            player_bribe_fractions, c.min_bribe_fraction, c.max_bribe_fraction
        )
        clamped = np.where(active, np.floor(self.balances * clamped_fractions), 0.0).astype(np.float32)
        self.balances -= clamped
        self.current_bribes = clamped.astype(np.float32)
        self.current_total_bribes = float(np.sum(clamped))

        self.phase = Phase.SIGNAL
        return self._get_observations(), {}, False, False, self._get_info()

    def step_signal(
        self,
        public_signal: int,
        private_signals: np.ndarray,
    ) -> tuple[dict, dict, bool, bool, dict]:
        """Phase 2 — Host broadcasts signals.

        Args:
            public_signal: single door index (0 … num_doors-1) broadcast to all.
            private_signals: shape (num_players,), per-player door hints.

        Returns (obs, {}, False, False, info).
        The returned obs carries updated *player* context (players now see signals).
        """
        if self.phase != Phase.SIGNAL:
            raise RuntimeError(
                f"step_signal() must be called in Phase.SIGNAL; current phase: {self.phase.name}"
            )
        c = self.cfg
        private_signals = np.asarray(private_signals, dtype=np.int32)
        if private_signals.shape != (c.num_players,):
            raise ValueError(
                f"private_signals must have shape ({c.num_players},), got {private_signals.shape}"
            )

        self.current_public_signal = int(np.clip(public_signal, 0, c.num_doors - 1))
        self.current_private_signals = np.clip(private_signals, 0, c.num_doors - 1)

        self.phase = Phase.BET
        return self._get_observations(), {}, False, False, self._get_info()

    def step_bet(
        self,
        player_doors: np.ndarray,
        player_bet_fractions: np.ndarray,
    ) -> tuple[dict, dict[str, Any], bool, bool, dict]:
        """Phase 3 — Players place bets; settlement is computed.

        Args:
            player_doors: shape (num_players,), door index chosen by each player.
            player_bet_fractions: shape (num_players,), fraction of balance to bet [0, 1].
                Effective bet is floored to integer dollars, with a minimum bet of $1
                for alive players who can afford at least $1.

        Returns (obs, rewards, terminated, truncated, info).
            rewards = {"players": ndarray (num_players,), "host": float}
        """
        if self.phase != Phase.BET:
            raise RuntimeError(
                f"step_bet() must be called in Phase.BET; current phase: {self.phase.name}"
            )
        c = self.cfg
        player_doors = np.asarray(player_doors, dtype=np.int32)
        player_bet_fractions = np.asarray(player_bet_fractions, dtype=np.float32)
        if player_doors.shape != (c.num_players,):
            raise ValueError(f"player_doors must have shape ({c.num_players},), got {player_doors.shape}")
        if player_bet_fractions.shape != (c.num_players,):
            raise ValueError(
                f"player_bet_fractions must have shape ({c.num_players},), got {player_bet_fractions.shape}"
            )

        active = self.balances > 0
        chosen_doors = np.clip(player_doors, 0, c.num_doors - 1)
        clamped_fractions = np.clip(player_bet_fractions, c.min_bet_fraction, c.max_bet_fraction)
        max_affordable_dollars = np.floor(self.balances)
        can_afford_one = active & (max_affordable_dollars >= 1.0)
        raw_bets = np.floor(self.balances * clamped_fractions)
        bets = np.where(
            can_afford_one,
            np.minimum(np.maximum(raw_bets, 1.0), max_affordable_dollars),
            0.0,
        ).astype(np.float32)
        self.balances -= bets

        winning_door = self.current_winning_door
        total_pool = float(np.sum(bets))
        winner_mask = active & (chosen_doors == winning_door) & (bets > 0)
        total_winning_vol = float(np.sum(bets[winner_mask]))

        payouts = np.zeros(c.num_players, dtype=np.float32)
        if total_pool > c.epsilon and total_winning_vol > c.epsilon:
            x = total_winning_vol / total_pool
            multiplier = 1.0 + (c.surplus_coefficient / max(x, c.min_winning_ratio_for_payout))
            if c.max_payout_multiplier is not None:
                multiplier = min(multiplier, c.max_payout_multiplier)
            payouts[winner_mask] = bets[winner_mask] * multiplier

        self.balances += payouts

        total_payout = float(np.sum(payouts))
        host_reward = total_pool - total_payout + self.current_total_bribes
        self.host_cumulative_profit += host_reward

        # R_player = payout - bet_lost - bribe
        player_rewards = (payouts - bets - self.current_bribes).astype(np.float32)

        door_ratios = np.zeros(c.num_doors, dtype=np.float32)
        if total_pool > c.epsilon:
            for d in range(c.num_doors):
                door_ratios[d] = float(np.sum(bets[chosen_doors == d]) / total_pool)

        self._push_history(
            choices=chosen_doors.astype(np.float32),
            public_signal=float(self.current_public_signal),
            private_signals=self.current_private_signals.astype(np.float32),
            bribes=self.current_bribes,
            bets=bets,
            player_rewards=player_rewards,
            host_profit=host_reward,
            door_ratios=door_ratios,
        )

        self.round_idx += 1
        terminated = bool(self.round_idx >= c.max_rounds or np.all(self.balances <= 0))
        truncated = False

        # Prepare transient state for the next round
        self.current_bribes = np.zeros(c.num_players, dtype=np.float32)
        self.current_total_bribes = 0.0
        self.current_public_signal = -1
        self.current_private_signals = np.full(c.num_players, -1, dtype=np.int32)
        if not terminated:
            self.current_winning_door = int(self.rng.integers(0, c.num_doors))
            self.phase = Phase.BRIBE
        # (If terminated, phase stays at BET; caller should reset before reuse.)

        rewards: dict[str, Any] = {
            "players": player_rewards,
            "host": float(host_reward),
        }
        info = self._get_info()
        info["winning_door"] = winning_door
        return self._get_observations(), rewards, terminated, truncated, info

    def step(self, action: dict) -> tuple:
        """Dispatcher: routes to the correct phase method based on self.phase.

        Expected keys per phase:
            Phase.BRIBE  → action["player_bribe_fractions"] (or legacy action["player_bribes"])
            Phase.SIGNAL → action["public_signal"], action["private_signals"]
            Phase.BET    → action["player_doors"], action["bet_fractions"]
        """
        if self.phase == Phase.BRIBE:
            if "player_bribe_fractions" in action:
                return self.step_bribe(action["player_bribe_fractions"])
            if "player_bribes" in action:
                return self.step_bribe(action["player_bribes"])
            raise KeyError("BRIBE phase requires 'player_bribe_fractions' action key")
        if self.phase == Phase.SIGNAL:
            return self.step_signal(action["public_signal"], action["private_signals"])
        return self.step_bet(action["player_doors"], action["bet_fractions"])

    # ─────────────────────────────────────────────
    # Observation construction (structured, NOT flattened)
    # ─────────────────────────────────────────────

    def _get_player_obs(self) -> dict[str, np.ndarray]:
        """Build player observations.

        Returns:
            current : (num_players, current_player_dim)  — context scalars
                      Fields: [alive, balance, bribe_sent, public_signal, private_signal]
                      -1 padding for fields not yet available this phase.
                      All zeros for eliminated players (balance <= 0), using the
                      alive=0 sentinel to distinguish elimination from padding.
            history : (num_players, history_window, hist_player_dim)  — past rounds
                      All zeros for eliminated players.
        """
        c = self.cfg

        current = np.zeros((c.num_players, c.current_player_dim), dtype=np.float32)
        for i in range(c.num_players):
            if self.balances[i] <= 0:
                # Eliminated: full row stays zero
                continue
            bal = self.balances[i]
            if c.normalize_balance_in_obs:
                bal = bal / max(c.initial_balance, c.epsilon)
            bribe = float(self.current_bribes[i])
            pub = float(self.current_public_signal)    # -1 before SIGNAL phase
            priv = float(self.current_private_signals[i])  # -1 before SIGNAL phase
            current[i] = [1.0, bal, bribe, pub, priv]

        # history: (num_players, W, hist_player_dim)
        # Stack per-player slices then transpose to (players, W, features)
        # Fields: [choice, pub_sig, priv_sig, bribe, bet, reward, host_profit, *door_ratios]
        player_hist_list = []
        for i in range(c.num_players):
            if self.balances[i] <= 0:
                player_hist_list.append(
                    np.zeros((c.history_window, c.hist_player_dim), dtype=np.float32)
                )
                continue
            step_features = np.stack([
                self.hist_choices[:, i],           # (W,)
                self.hist_public_signal,            # (W,)
                self.hist_private_signals[:, i],    # (W,)
                self.hist_bribes[:, i],             # (W,)
                self.hist_bets[:, i],               # (W,)
                self.hist_player_rewards[:, i],     # (W,)
                self.hist_host_profit,              # (W,)
            ], axis=1)  # (W, 7)
            full = np.concatenate([step_features, self.hist_door_ratios], axis=1)  # (W, 7+D)
            player_hist_list.append(full.astype(np.float32))

        history = np.stack(player_hist_list, axis=0)  # (N, W, hist_player_dim)
        return {"current": current, "history": history}

    def _get_host_obs(self) -> dict[str, np.ndarray]:
        """Build host observation.

        Returns:
            current : (current_host_dim,)
                      Fields: [cumulative_profit, total_bribes_this_round,
                               winning_door_onehot × num_doors]
            players : (num_players, host_player_state_dim)
                      Fields per player: [normalized_balance, active_flag, bribe_this_round]
                      Zeros for eliminated players.
            history : (history_window, hist_host_dim)
                      Fields per step: [host_profit, public_signal,
                                        door_ratios × num_doors,
                                        private_signal_distribution × num_doors]
        """
        c = self.cfg

        # current
        winning_one_hot = np.zeros(c.num_doors, dtype=np.float32)
        if 0 <= self.current_winning_door < c.num_doors:
            winning_one_hot[self.current_winning_door] = 1.0
        current = np.concatenate([
            np.array([self.host_cumulative_profit, self.current_total_bribes], dtype=np.float32),
            winning_one_hot,
        ])  # (2 + num_doors,)

        # players snapshot — eliminated players are represented as all-zeros
        active_mask_bool = self.balances > 0
        active_mask = active_mask_bool.astype(np.float32)
        balances = self.balances.copy().astype(np.float32)
        if c.normalize_balance_in_obs:
            balances = balances / max(c.initial_balance, c.epsilon)
        balances = np.where(active_mask_bool, balances, 0.0)
        # Bribes for eliminated players are also zeroed for consistency
        visible_bribes = np.where(active_mask_bool, self.current_bribes, 0.0)
        players = np.stack([balances, active_mask, visible_bribes], axis=1)  # (N, 3)

        # history: private signal distribution per past step
        priv_dist = np.zeros((c.history_window, c.num_doors), dtype=np.float32)
        for t in range(c.history_window):
            valid_mask = self.hist_private_signals[t] >= 0
            if np.any(valid_mask):
                valid_sigs = self.hist_private_signals[t][valid_mask].astype(int)
                counts = np.bincount(valid_sigs, minlength=c.num_doors).astype(np.float32)
                priv_dist[t] = counts / max(float(counts.sum()), 1.0)

        history = np.concatenate([
            self.hist_host_profit[:, None],              # (W, 1)
            self.hist_public_signal[:, None],            # (W, 1)
            self.hist_door_ratios,                       # (W, D)
            priv_dist,                                   # (W, D)
        ], axis=1).astype(np.float32)  # (W, 2 + 2*D)

        return {
            "current": current,
            "players": players.astype(np.float32),
            "history": history,
        }

    def _get_observations(self) -> dict[str, dict]:
        return {
            "players": self._get_player_obs(),
            "host": self._get_host_obs(),
        }

    # ─────────────────────────────────────────────
    # Info dict
    # ─────────────────────────────────────────────

    def _get_info(self) -> dict[str, Any]:
        c = self.cfg
        return {
            "phase": int(self.phase),
            "phase_name": self.phase.name,
            "round": self.round_idx,
            "active_players": int(np.sum(self.balances > 0)),
            "host_cumulative_profit": float(self.host_cumulative_profit),
            # Observation shape hints (useful for model construction)
            "player_current_shape": (c.num_players, c.current_player_dim),
            "player_history_shape": (c.num_players, c.history_window, c.hist_player_dim),
            "host_current_shape": (c.current_host_dim,),
            "host_players_shape": (c.num_players, c.host_player_state_dim),
            "host_history_shape": (c.history_window, c.hist_host_dim),
        }

    # ─────────────────────────────────────────────
    # History buffer helpers
    # ─────────────────────────────────────────────

    def _push_history(
        self,
        *,
        choices: np.ndarray,
        public_signal: float,
        private_signals: np.ndarray,
        bribes: np.ndarray,
        bets: np.ndarray,
        player_rewards: np.ndarray,
        host_profit: float,
        door_ratios: np.ndarray,
    ) -> None:
        """Shift all history buffers by one step and write new values at [-1]."""
        self.hist_choices = np.roll(self.hist_choices, -1, axis=0)
        self.hist_public_signal = np.roll(self.hist_public_signal, -1, axis=0)
        self.hist_private_signals = np.roll(self.hist_private_signals, -1, axis=0)
        self.hist_bribes = np.roll(self.hist_bribes, -1, axis=0)
        self.hist_bets = np.roll(self.hist_bets, -1, axis=0)
        self.hist_player_rewards = np.roll(self.hist_player_rewards, -1, axis=0)
        self.hist_host_profit = np.roll(self.hist_host_profit, -1, axis=0)
        self.hist_door_ratios = np.roll(self.hist_door_ratios, -1, axis=0)

        self.hist_choices[-1] = choices
        self.hist_public_signal[-1] = public_signal
        self.hist_private_signals[-1] = private_signals
        self.hist_bribes[-1] = bribes
        self.hist_bets[-1] = bets
        self.hist_player_rewards[-1] = player_rewards
        self.hist_host_profit[-1] = host_profit
        self.hist_door_ratios[-1] = door_ratios
