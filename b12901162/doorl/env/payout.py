"""Payout / multiplier mathematics for DooRL.

The parimutuel-style dynamic-odds rule (see spec.md §7):

    alpha         = 1 - tau
    x             = W / P                    # winning volume / total pool
    multiplier(x) = min(1 + alpha / x, max_multiplier)
    payout_i      = bet_i * multiplier(x)    # winners only

Break-even property: at x = tau the total payout equals P.
"""

from __future__ import annotations

import math
from typing import Optional


def calculate_multiplier(
    x: float,
    tau: float,
    max_multiplier: Optional[float] = 50.0,
) -> float:
    """Return the per-winner multiplier given the winning ratio ``x``.

    Args:
        x: Winning pool ratio ``W / P``. ``W`` must be > 0 to use this function;
            callers must handle the ``W = 0`` edge case (E1) separately.
        tau: Break-even threshold in (0, 1).
        max_multiplier: Hard cap on the multiplier. ``None`` disables the cap.

    Returns:
        The clipped multiplier ``min(1 + alpha / x, max_multiplier)``.
    """
    if not (0.0 < tau < 1.0):
        raise ValueError(f"tau must be in (0, 1), got {tau}")
    if x <= 0.0:
        raise ValueError(
            "calculate_multiplier requires x > 0; handle W=0 case (E1) at the caller"
        )
    alpha = 1.0 - tau
    m = 1.0 + alpha / x
    if max_multiplier is not None and m > max_multiplier:
        return float(max_multiplier)
    return float(m)


def calculate_payout(
    individual_bet: float,
    total_winning_vol: float,
    total_pool: float,
    tau: float = 0.20,
    max_multiplier: Optional[float] = 50.0,
) -> float:
    """Return the per-winner payout.

    Returns 0.0 if either ``total_winning_vol`` or ``total_pool`` is zero
    (caller still needs to apply the E1 rule of "Host keeps pool").
    """
    if total_pool <= 0.0:
        return 0.0
    if total_winning_vol <= 0.0:
        return 0.0
    x = total_winning_vol / total_pool
    m = calculate_multiplier(x, tau=tau, max_multiplier=max_multiplier)
    return float(individual_bet) * m


def host_pool_pnl(
    total_pool: float,
    total_winning_vol: float,
    tau: float = 0.20,
    max_multiplier: Optional[float] = 50.0,
) -> float:
    """Host P&L from the betting pool alone (bribes added separately).

    Mathematically equivalent to ``P - sum(payouts) = P - W * m``.
    Used by the env for the Host per-round reward.
    """
    if total_pool <= 0.0:
        return 0.0
    if total_winning_vol <= 0.0:
        return float(total_pool)
    x = total_winning_vol / total_pool
    m = calculate_multiplier(x, tau=tau, max_multiplier=max_multiplier)
    return float(total_pool - total_winning_vol * m)


def break_even_check(tau: float, max_multiplier: Optional[float] = None) -> bool:
    """Sanity helper: confirm that at x = tau total payouts equal P."""
    m = calculate_multiplier(tau, tau=tau, max_multiplier=max_multiplier)
    # at x = tau, payouts = W * m = tau * P * m; want tau * m == 1
    return math.isclose(tau * m, 1.0, rel_tol=1e-9, abs_tol=1e-9)
