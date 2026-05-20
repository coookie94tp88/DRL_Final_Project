"""Dataclasses describing per-round state and per-step settlement records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class EnvConfig:
    """Configuration for DooRLEnv."""

    num_players: int = 10
    num_doors: int = 4
    initial_balance: float = 1000.0
    payout_threshold: float = 0.20
    max_multiplier: Optional[float] = 50.0
    history_window: int = 50
    max_rounds: int = 200
    min_bet: float = 1.0
    balance_visibility: str = "full"  # {full, own_only, noisy}
    balance_noise_sigma: float = 0.1  # used only when balance_visibility == "noisy"
    seed: Optional[int] = None

    # Anti-babbling hooks read by the env (most are pure policy-side though).
    bribery_floor: float = 0.0  # forced minimum bribe_pct (0 disables)

    # Optional shaping (Stage-1 curriculum): bonus when a player bets on a signal door.
    follow_public_bonus: float = 0.0
    follow_private_bonus: float = 0.0

    def validate(self) -> None:
        if self.num_players < 1:
            raise ValueError(f"num_players must be >= 1, got {self.num_players}")
        if self.num_doors != 4:
            raise ValueError(
                f"DooRL is the 4-door game; num_doors must be 4, got {self.num_doors}"
            )
        if not (0.0 < self.payout_threshold < 1.0):
            raise ValueError(
                f"payout_threshold must be in (0, 1), got {self.payout_threshold}"
            )
        if self.balance_visibility not in {"full", "own_only", "noisy"}:
            raise ValueError(
                f"balance_visibility must be one of {{full, own_only, noisy}}, "
                f"got {self.balance_visibility!r}"
            )
        if self.history_window < 1:
            raise ValueError("history_window must be >= 1")
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        if self.max_multiplier is not None and self.max_multiplier <= 1.0:
            raise ValueError(
                "max_multiplier must be > 1 (or None to disable); "
                f"got {self.max_multiplier}"
            )


@dataclass
class RoundRecord:
    """Per-round settlement record used to feed the history buffer."""

    round_idx: int
    true_door: int
    public_signal: int
    private_signals: np.ndarray  # (N,) ints
    bribe_pcts: np.ndarray  # (N,) floats in [0, 1]
    bribes: np.ndarray  # (N,) absolute
    bet_pcts: np.ndarray  # (N,) floats in [0, 1]
    bets: np.ndarray  # (N,) absolute
    chosen_doors: np.ndarray  # (N,) ints
    door_share_vec: np.ndarray  # (4,) bet fraction per door
    payouts: np.ndarray  # (N,) absolute
    rewards_player: np.ndarray  # (N,) per-round R_player
    reward_host: float
    x: float  # winning ratio
    multiplier: float
    active_mask: np.ndarray  # (N,) 1.0 if active that round
    balances_after: np.ndarray  # (N,) absolute

    def history_row(self, num_players: int) -> np.ndarray:
        """Pack into a fixed-length feature vector for the history buffer.

        Layout (matches spec.md §3.B):
          own_door (4-onehot),              -> per-player call site uses own_door
          public_signal (4-onehot),
          private_signal_own (4-onehot),    -> per-player call site uses own private
          door_share_vec (4,),
          R_player_self (1,),
          R_player_all (N,),
          x (1,),
          R_host_delta (1,),
          active_mask (N,)
        """
        raise NotImplementedError("Use DooRLEnv._history_row_for_player instead.")


@dataclass
class LastSettlement:
    """Snapshot of the most recent round settlement (for watch / render)."""

    round_idx: int
    true_door: int
    public_signal: int
    private_signals: np.ndarray
    chosen_doors: np.ndarray
    bribe_pcts: np.ndarray
    bribes: np.ndarray
    bets: np.ndarray
    payouts: np.ndarray
    rewards_player: np.ndarray
    reward_host: float
    x: float
    multiplier: float
    door_share: np.ndarray
    pool_p: float


@dataclass
class EpisodeStats:
    """Aggregate statistics collected over one episode (for logging)."""

    rounds_played: int = 0
    bankruptcies: int = 0
    sum_R_host: float = 0.0
    sum_bribes: float = 0.0
    x_values: List[float] = field(default_factory=list)
    public_truth_rate_num: int = 0
    private_truth_rate_num: List[int] = field(default_factory=list)
    public_true_private_false_num: int = 0
    rounds_with_pool: int = 0
