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
    BET = 2     # players choose belief + bet fraction → settlement


class PlayerBelief(IntEnum):
    """Player bet-phase action: env maps to a concrete door internally."""
    BELIEVE_PUBLIC = 0   # follow public signal door
    BELIEVE_PRIVATE = 1  # follow private signal door (only if bribe > 0 this round)
    RANDOM = 2           # uniform random door (ignore signals)


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
    minority_threshold: float = 0.10    # host profits if winning_ratio < this
    max_rounds: int = 500

    # ── History buffer ──────────────────────────────────────
    history_window: int = 50            # number of past rounds kept in obs

    # ── Payout formula ──────────────────────────────────────
    payout_threshold: float = 0.10              # break-even winning ratio
    min_winning_ratio_for_payout: float = 1e-3  # floor for division stability
    max_payout_multiplier: float | None = 100.0  # optional cap on multiplier

    # ── Action bounds ───────────────────────────────────────
    min_bet_fraction: float = 0.0
    max_bet_fraction: float = 1.0
    min_bet_dollars: float = 1.0
    min_bribe_dollars: float = 1.0   # if fraction > 0 and balance allows, pay at least this
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
        Fields: [alive, balance, bribe_sent, signals_agree]
        signals_agree: 1/0 if bribe>0 and public==private/differ; -1 if no bribe or before SIGNAL.
        """
        return 4

    @property
    def hist_player_dim(self) -> int:
        """Feature count per time-step in a player's history buffer.
        Fields: my_belief, signals_agree, host_public_honest, bribe_private_hit,
                bribe, bet, reward, host_profit, frac_correct, frac_believe_public,
                frac_believe_private, frac_random
        signals_agree: 1/0 if bribe>0 and public==private/differ; -1 if no bribe.
        bribe_private_hit: 1 if bribe>0 and private==winner, 0 if bribe>0 and miss, -1 if no bribe.
        """
        return 12

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
        Fields: host_profit, public_signal, frac_correct, frac_believe_public,
                frac_believe_private, frac_random, host_public_honest,
                private_signal_distribution_0 … (num_doors values)
        """
        return 7 + self.num_doors


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
               Host broadcasts a public signal to all. Private indices are stored only for
               players with bribe > 0; others remain -1 (no insider channel).

        3. step_bet(player_beliefs, player_bet_fractions)
               Players choose BELIEVE_PUBLIC / BELIEVE_PRIVATE / RANDOM plus bet fraction.
               BELIEVE_PRIVATE without bribe is coerced to RANDOM. Doors are mapped internally.
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
        # Phase.BET — 0=believe public, 1=believe private, 2=random
        self.player_bet_action_space = spaces.Dict({
            "beliefs": spaces.MultiDiscrete([3] * c.num_players),
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
        self.hist_beliefs: np.ndarray = np.full(
            (c.history_window, c.num_players), -1.0, dtype=np.float32)
        self.hist_chosen_doors: np.ndarray = np.full(
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
        # Per-round market aggregates (not door-index ratios)
        self.hist_frac_correct: np.ndarray = np.full(
            (c.history_window,), -1.0, dtype=np.float32)
        self.hist_frac_believe_public: np.ndarray = np.full(
            (c.history_window,), -1.0, dtype=np.float32)
        self.hist_frac_believe_private: np.ndarray = np.full(
            (c.history_window,), -1.0, dtype=np.float32)
        self.hist_frac_random: np.ndarray = np.full(
            (c.history_window,), -1.0, dtype=np.float32)
        self.hist_host_public_honest: np.ndarray = np.full(
            (c.history_window,), -1.0, dtype=np.float32)
        self.hist_bribe_private_hit: np.ndarray = np.full(
            (c.history_window, c.num_players), -1.0, dtype=np.float32)

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
                of that player's current balance. Paid bribe is floored to integer dollars.
                If fraction > 0 and balance >= min_bribe_dollars, the player pays at least
                min_bribe_dollars (capped by balance). Fraction == 0 means no bribe ($0).

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
        raw_bribes = np.where(active, self.balances * clamped_fractions, 0.0)
        bribes = np.floor(raw_bribes).astype(np.float32)
        wants_bribe = active & (clamped_fractions > 0)
        can_afford_min = active & (self.balances >= c.min_bribe_dollars)
        apply_min = wants_bribe & can_afford_min
        bribes = np.where(apply_min, np.maximum(bribes, c.min_bribe_dollars), bribes)
        bribes = np.where(wants_bribe & ~can_afford_min, 0.0, bribes)
        bribes = np.minimum(bribes, self.balances)

        self.balances -= bribes
        self.current_bribes = bribes
        self.current_total_bribes = float(np.sum(bribes))

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
            private_signals: shape (num_players,), per-player door hints from the Host.
                Only players with bribe > 0 receive a stored private index; others get -1.

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
        bribed = self.current_bribes > 0
        self.current_private_signals = np.full(c.num_players, -1, dtype=np.int32)
        if np.any(bribed):
            self.current_private_signals[bribed] = np.clip(
                private_signals[bribed], 0, c.num_doors - 1
            )

        self.phase = Phase.BET
        return self._get_observations(), {}, False, False, self._get_info()

    @staticmethod
    def _map_beliefs_to_doors(
        beliefs: np.ndarray,
        public_signal: int,
        private_signals: np.ndarray,
        bribes: np.ndarray,
        num_doors: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Map BELIEVE_PUBLIC / BELIEVE_PRIVATE / RANDOM to concrete door indices."""
        beliefs = np.clip(beliefs.astype(np.int32), 0, 2)
        no_private = bribes <= 0
        beliefs = np.where(
            no_private & (beliefs == PlayerBelief.BELIEVE_PRIVATE),
            PlayerBelief.RANDOM,
            beliefs,
        )
        random_doors = rng.integers(0, num_doors, size=beliefs.shape[0], dtype=np.int32)
        has_private = private_signals >= 0
        private_doors = np.where(has_private, private_signals, random_doors)
        return np.where(
            beliefs == PlayerBelief.BELIEVE_PUBLIC,
            public_signal,
            np.where(beliefs == PlayerBelief.BELIEVE_PRIVATE, private_doors, random_doors),
        )

    def step_bet(
        self,
        player_beliefs: np.ndarray,
        player_bet_fractions: np.ndarray,
    ) -> tuple[dict, dict[str, Any], bool, bool, dict]:
        """Phase 3 — Players place bets; settlement is computed.

        Args:
            player_beliefs: shape (num_players,), each in {0, 1, 2}:
                0=BELIEVE_PUBLIC, 1=BELIEVE_PRIVATE (requires bribe > 0), 2=RANDOM.
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
        player_beliefs = np.asarray(player_beliefs, dtype=np.int32)
        player_bet_fractions = np.asarray(player_bet_fractions, dtype=np.float32)
        if player_beliefs.shape != (c.num_players,):
            raise ValueError(
                f"player_beliefs must have shape ({c.num_players},), got {player_beliefs.shape}"
            )
        if player_bet_fractions.shape != (c.num_players,):
            raise ValueError(
                f"player_bet_fractions must have shape ({c.num_players},), got {player_bet_fractions.shape}"
            )

        active = self.balances > 0
        chosen_doors = self._map_beliefs_to_doors(
            player_beliefs,
            self.current_public_signal,
            self.current_private_signals,
            self.current_bribes,
            c.num_doors,
            self.rng,
        )
        clamped_fractions = np.clip(player_bet_fractions, c.min_bet_fraction, c.max_bet_fraction)
        
        # 1. 【修改】計算下注金額並強制向下取整為整數
        raw_bets = np.where(active, self.balances * clamped_fractions, 0.0)
        bets = np.floor(raw_bets).astype(np.float32)
        
        # 2. 【新增】強制所有還活著的玩家，該局最低下注金額為 1 元
        bets = np.where(active, np.maximum(bets, 1.0), 0.0)
        
        # 3. 【防呆】確保強制設為 1 元時，不會超過玩家此時擁有的餘額（上限防禦）
        bets = np.minimum(bets, self.balances)
        
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
            
            # 4. 【修改】獲勝者分配到的獎金也強制向下取整為整數
            payouts[winner_mask] = np.floor(bets[winner_mask] * multiplier).astype(np.float32)

        self.balances += payouts

        total_payout = float(np.sum(payouts))
        host_reward = total_pool - total_payout + self.current_total_bribes
        self.host_cumulative_profit += host_reward

        # R_player = payout - bet_lost - bribe
        player_rewards = (payouts - bets - self.current_bribes).astype(np.float32)

        active_bettors = active & (bets > 0)
        n_active_bet = int(np.sum(active_bettors))
        if n_active_bet > 0:
            frac_correct = float(np.sum(active_bettors & (chosen_doors == winning_door)) / n_active_bet)
            frac_believe_public = float(
                np.sum(active_bettors & (player_beliefs == PlayerBelief.BELIEVE_PUBLIC)) / n_active_bet
            )
            frac_believe_private = float(
                np.sum(active_bettors & (player_beliefs == PlayerBelief.BELIEVE_PRIVATE)) / n_active_bet
            )
            frac_random = float(
                np.sum(active_bettors & (player_beliefs == PlayerBelief.RANDOM)) / n_active_bet
            )
        else:
            frac_correct = frac_believe_public = frac_believe_private = frac_random = 0.0

        host_public_honest = float(self.current_public_signal == winning_door)

        bribe_private_hit = np.full(c.num_players, -1.0, dtype=np.float32)
        bribed = self.current_bribes > 0
        bribe_private_hit[bribed] = (
            (self.current_private_signals[bribed] >= 0)
            & (self.current_private_signals[bribed] == winning_door)
        ).astype(np.float32)

        self._push_history(
            beliefs=player_beliefs.astype(np.float32),
            chosen_doors=chosen_doors.astype(np.float32),
            public_signal=float(self.current_public_signal),
            private_signals=self.current_private_signals.astype(np.float32),
            bribes=self.current_bribes,
            bets=bets,
            player_rewards=player_rewards,
            host_profit=host_reward,
            frac_correct=frac_correct,
            frac_believe_public=frac_believe_public,
            frac_believe_private=frac_believe_private,
            frac_random=frac_random,
            host_public_honest=host_public_honest,
            bribe_private_hit=bribe_private_hit,
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
        info["chosen_doors"] = chosen_doors
        info["player_beliefs"] = player_beliefs
        info["frac_correct"] = frac_correct
        info["frac_believe_public"] = frac_believe_public
        info["host_public_honest"] = host_public_honest
        return self._get_observations(), rewards, terminated, truncated, info

    def step(self, action: dict) -> tuple:
        """Dispatcher: routes to the correct phase method based on self.phase.

        Expected keys per phase:
            Phase.BRIBE  → action["player_bribe_fractions"] (or legacy action["player_bribes"])
            Phase.SIGNAL → action["public_signal"], action["private_signals"]
            Phase.BET    → action["player_beliefs"] (or legacy "beliefs"), action["bet_fractions"]
        """
        if self.phase == Phase.BRIBE:
            if "player_bribe_fractions" in action:
                return self.step_bribe(action["player_bribe_fractions"])
            if "player_bribes" in action:
                return self.step_bribe(action["player_bribes"])
            raise KeyError("BRIBE phase requires 'player_bribe_fractions' action key")
        if self.phase == Phase.SIGNAL:
            return self.step_signal(action["public_signal"], action["private_signals"])
        beliefs = action.get("player_beliefs", action.get("beliefs"))
        if beliefs is None:
            raise KeyError("BET phase requires 'player_beliefs' or 'beliefs' action key")
        return self.step_bet(beliefs, action["bet_fractions"])

    # ─────────────────────────────────────────────
    # Observation construction (structured, NOT flattened)
    # ─────────────────────────────────────────────

    def _get_player_obs(self) -> dict[str, np.ndarray]:
        """Build player observations.

        Returns:
            current : (num_players, current_player_dim)  — context scalars
                      Fields: [alive, balance, bribe_sent, signals_agree]
                      signals_agree: -1 before SIGNAL or if bribe==0; else 0/1 (public==private).
            history : (num_players, history_window, hist_player_dim)  — past rounds
                      All zeros for eliminated players.
        """
        c = self.cfg

        current = np.zeros((c.num_players, c.current_player_dim), dtype=np.float32)
        for i in range(c.num_players):
            if self.balances[i] <= 0:
                continue
            bal = self.balances[i]
            if c.normalize_balance_in_obs:
                bal = bal / max(c.initial_balance, c.epsilon)
            bribe = float(self.current_bribes[i])
            agree = -1.0
            if (
                bribe > 0
                and self.current_public_signal >= 0
                and self.current_private_signals[i] >= 0
            ):
                agree = float(self.current_public_signal == self.current_private_signals[i])
            current[i] = [1.0, bal, bribe, agree]

        market_cols = np.stack([
            self.hist_frac_correct,
            self.hist_frac_believe_public,
            self.hist_frac_believe_private,
            self.hist_frac_random,
        ], axis=1)  # (W, 4)

        player_hist_list = []
        for i in range(c.num_players):
            step_features = np.stack([
                self.hist_beliefs[:, i],
                np.full(c.history_window, -1.0, dtype=np.float32),  # signals_agree, filled below
                self.hist_host_public_honest,
                self.hist_bribe_private_hit[:, i],
                self.hist_bribes[:, i],
                self.hist_bets[:, i],
                self.hist_player_rewards[:, i],
                self.hist_host_profit,
            ], axis=1)
            # signals_agree only when that round's bribe > 0
            sig_agree_hist = np.full(c.history_window, -1.0, dtype=np.float32)
            for t in range(c.history_window):
                if self.hist_bribes[t, i] <= 0:
                    continue
                pub = self.hist_public_signal[t]
                priv = self.hist_private_signals[t, i]
                if pub >= 0 and priv >= 0:
                    sig_agree_hist[t] = float(int(pub) == int(priv))
            step_features[:, 1] = sig_agree_hist

            full = np.concatenate([step_features, market_cols], axis=1)
            player_hist_list.append(full.astype(np.float32))

        history = np.stack(player_hist_list, axis=0)
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
                      Fields per step: [host_profit, public_signal, frac_correct,
                                        frac_believe_public, frac_believe_private, frac_random,
                                        host_public_honest, private_signal_distribution × D]
        """
        c = self.cfg

        winning_one_hot = np.zeros(c.num_doors, dtype=np.float32)
        if 0 <= self.current_winning_door < c.num_doors:
            winning_one_hot[self.current_winning_door] = 1.0
        current = np.concatenate([
            np.array([self.host_cumulative_profit, self.current_total_bribes], dtype=np.float32),
            winning_one_hot,
        ])

        active_mask_bool = self.balances > 0
        active_mask = active_mask_bool.astype(np.float32)
        balances = self.balances.copy().astype(np.float32)
        if c.normalize_balance_in_obs:
            balances = balances / max(c.initial_balance, c.epsilon)
        balances = np.where(active_mask_bool, balances, 0.0)
        visible_bribes = np.where(active_mask_bool, self.current_bribes, 0.0)
        players = np.stack([balances, active_mask, visible_bribes], axis=1)

        priv_dist = np.zeros((c.history_window, c.num_doors), dtype=np.float32)
        for t in range(c.history_window):
            valid_mask = self.hist_private_signals[t] >= 0
            if np.any(valid_mask):
                valid_sigs = self.hist_private_signals[t][valid_mask].astype(int)
                counts = np.bincount(valid_sigs, minlength=c.num_doors).astype(np.float32)
                priv_dist[t] = counts / max(float(counts.sum()), 1.0)

        history = np.concatenate([
            self.hist_host_profit[:, None],
            self.hist_public_signal[:, None],
            self.hist_frac_correct[:, None],
            self.hist_frac_believe_public[:, None],
            self.hist_frac_believe_private[:, None],
            self.hist_frac_random[:, None],
            self.hist_host_public_honest[:, None],
            priv_dist,
        ], axis=1).astype(np.float32)

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
        beliefs: np.ndarray,
        chosen_doors: np.ndarray,
        public_signal: float,
        private_signals: np.ndarray,
        bribes: np.ndarray,
        bets: np.ndarray,
        player_rewards: np.ndarray,
        host_profit: float,
        frac_correct: float,
        frac_believe_public: float,
        frac_believe_private: float,
        frac_random: float,
        host_public_honest: float,
        bribe_private_hit: np.ndarray,
    ) -> None:
        """Shift all history buffers by one step and write new values at [-1]."""
        self.hist_beliefs = np.roll(self.hist_beliefs, -1, axis=0)
        self.hist_chosen_doors = np.roll(self.hist_chosen_doors, -1, axis=0)
        self.hist_public_signal = np.roll(self.hist_public_signal, -1, axis=0)
        self.hist_private_signals = np.roll(self.hist_private_signals, -1, axis=0)
        self.hist_bribes = np.roll(self.hist_bribes, -1, axis=0)
        self.hist_bets = np.roll(self.hist_bets, -1, axis=0)
        self.hist_player_rewards = np.roll(self.hist_player_rewards, -1, axis=0)
        self.hist_host_profit = np.roll(self.hist_host_profit, -1, axis=0)
        self.hist_frac_correct = np.roll(self.hist_frac_correct, -1, axis=0)
        self.hist_frac_believe_public = np.roll(self.hist_frac_believe_public, -1, axis=0)
        self.hist_frac_believe_private = np.roll(self.hist_frac_believe_private, -1, axis=0)
        self.hist_frac_random = np.roll(self.hist_frac_random, -1, axis=0)
        self.hist_host_public_honest = np.roll(self.hist_host_public_honest, -1, axis=0)
        self.hist_bribe_private_hit = np.roll(self.hist_bribe_private_hit, -1, axis=0)

        self.hist_beliefs[-1] = beliefs
        self.hist_chosen_doors[-1] = chosen_doors
        self.hist_public_signal[-1] = public_signal
        self.hist_private_signals[-1] = private_signals
        self.hist_bribes[-1] = bribes
        self.hist_bets[-1] = bets
        self.hist_player_rewards[-1] = player_rewards
        self.hist_host_profit[-1] = host_profit
        self.hist_frac_correct[-1] = frac_correct
        self.hist_frac_believe_public[-1] = frac_believe_public
        self.hist_frac_believe_private[-1] = frac_believe_private
        self.hist_frac_random[-1] = frac_random
        self.hist_host_public_honest[-1] = host_public_honest
        self.hist_bribe_private_hit[-1] = bribe_private_hit