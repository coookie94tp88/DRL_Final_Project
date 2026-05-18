"""Evaluation metrics for DooRL.

Includes:

* Mutual-information estimators between (private_signal_i, true_door) and
  (public_signal, true_door) using empirical joint-probability tables.
* Aggregate metrics: bankruptcy rate, x distribution, mean bribe_pct, door
  crowding entropy, public/private truth rates, follow-signal rates.

All metrics accept Python lists / numpy arrays — no torch dependency, no env
coupling. The trainer/eval scripts call these once per ``eval_interval``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


def _shannon_entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p[p > eps]
    return float(-np.sum(p * np.log2(p)))


def mutual_information(
    signal: Sequence[int],
    truth: Sequence[int],
    num_signal: int = 4,
    num_truth: int = 4,
) -> float:
    """Empirical I(signal; truth) in bits."""
    s = np.asarray(signal, dtype=np.int64)
    t = np.asarray(truth, dtype=np.int64)
    if s.size == 0:
        return 0.0
    joint = np.zeros((num_signal, num_truth), dtype=np.float64)
    for a, b in zip(s, t):
        joint[int(a), int(b)] += 1.0
    joint /= joint.sum()
    ps = joint.sum(axis=1)
    pt = joint.sum(axis=0)
    h_s = _shannon_entropy(ps)
    h_t = _shannon_entropy(pt)
    h_joint = _shannon_entropy(joint.flatten())
    return float(h_s + h_t - h_joint)


@dataclass
class MetricsSummary:
    """Aggregate metrics over a set of rounds collected during evaluation."""

    mean_R_host: float = 0.0
    std_R_host: float = 0.0
    mean_player_balance: float = 0.0
    bankruptcy_rate: float = 0.0
    mean_bribe_pct: float = 0.0
    mean_x: float = 0.0
    median_x: float = 0.0
    door_share_entropy: float = 0.0
    mi_private_truth: float = 0.0
    mi_public_truth: float = 0.0
    p_follow_private: float = 0.0
    p_follow_public: float = 0.0
    public_true_private_false_rate: float = 0.0
    h1_mean_bribe_pct: float = 0.0  # alias for analysis
    h4_pub_true_priv_false_rate: float = 0.0
    rounds_collected: int = 0


@dataclass
class MetricsAccumulator:
    """Streaming accumulator. Push one record per round."""

    R_host: List[float] = field(default_factory=list)
    end_balances: List[float] = field(default_factory=list)
    bankruptcies: int = 0
    total_players_seen: int = 0
    bribe_pcts: List[float] = field(default_factory=list)
    xs: List[float] = field(default_factory=list)
    door_shares: List[np.ndarray] = field(default_factory=list)
    private_signals: List[int] = field(default_factory=list)
    public_signals: List[int] = field(default_factory=list)
    true_doors: List[int] = field(default_factory=list)
    follow_private: List[int] = field(default_factory=list)
    follow_public: List[int] = field(default_factory=list)
    public_true_private_false: List[int] = field(default_factory=list)

    def add_round(
        self,
        r_host: float,
        bribe_pcts: Sequence[float],
        x: Optional[float],
        door_share: np.ndarray,
        private_signals: Sequence[int],
        public_signal: int,
        true_door: int,
        chosen_doors: Sequence[int],
        active_mask: Sequence[float],
    ) -> None:
        self.R_host.append(float(r_host))
        for p in bribe_pcts:
            self.bribe_pcts.append(float(p))
        if x is not None and x > 0.0:
            self.xs.append(float(x))
        self.door_shares.append(np.asarray(door_share, dtype=np.float64))
        for i, sig in enumerate(private_signals):
            self.private_signals.append(int(sig))
            self.true_doors.append(int(true_door))
            self.public_signals.append(int(public_signal))
            self.follow_private.append(int(chosen_doors[i] == sig))
            self.follow_public.append(int(chosen_doors[i] == public_signal))
        if public_signal == true_door:
            for sig in private_signals:
                self.public_true_private_false.append(int(sig != true_door))

    def add_end_of_episode(
        self, balances: Sequence[float], active_mask: Sequence[float]
    ) -> None:
        for b, a in zip(balances, active_mask):
            self.end_balances.append(float(b))
            self.total_players_seen += 1
            if a == 0.0:
                self.bankruptcies += 1

    def finalize(self) -> MetricsSummary:
        if not self.R_host:
            return MetricsSummary()
        s = MetricsSummary()
        s.mean_R_host = float(np.mean(self.R_host))
        s.std_R_host = float(np.std(self.R_host))
        s.mean_player_balance = (
            float(np.mean(self.end_balances)) if self.end_balances else 0.0
        )
        s.bankruptcy_rate = (
            self.bankruptcies / max(self.total_players_seen, 1)
            if self.total_players_seen
            else 0.0
        )
        s.mean_bribe_pct = (
            float(np.mean(self.bribe_pcts)) if self.bribe_pcts else 0.0
        )
        s.mean_x = float(np.mean(self.xs)) if self.xs else 0.0
        s.median_x = float(np.median(self.xs)) if self.xs else 0.0
        if self.door_shares:
            mean_share = np.mean(np.stack(self.door_shares, axis=0), axis=0)
            s.door_share_entropy = _shannon_entropy(mean_share)
        s.mi_private_truth = mutual_information(
            self.private_signals, self.true_doors
        )
        s.mi_public_truth = mutual_information(
            self.public_signals, self.true_doors
        )
        s.p_follow_private = (
            float(np.mean(self.follow_private)) if self.follow_private else 0.0
        )
        s.p_follow_public = (
            float(np.mean(self.follow_public)) if self.follow_public else 0.0
        )
        s.public_true_private_false_rate = (
            float(np.mean(self.public_true_private_false))
            if self.public_true_private_false
            else 0.0
        )
        s.h1_mean_bribe_pct = s.mean_bribe_pct
        s.h4_pub_true_priv_false_rate = s.public_true_private_false_rate
        s.rounds_collected = len(self.R_host)
        return s


def check_acceptance_targets(
    summary: MetricsSummary,
    tau: float,
    initial_balance: float,
) -> Dict[str, bool]:
    """Returns a dict of pass/fail flags for the decent-result targets."""
    return {
        "host_profit_positive": summary.mean_R_host > 0.0,
        "player_balance_half_initial": summary.mean_player_balance
        > 0.5 * initial_balance,
        "median_x_near_tau": 0.5 * tau <= summary.median_x <= 1.5 * tau,
        "h2_private_beats_public_mi": summary.mi_private_truth
        > summary.mi_public_truth,
        "h4_pub_true_priv_false_low": summary.h4_pub_true_priv_false_rate < 0.05,
    }
