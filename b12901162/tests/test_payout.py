"""Unit tests for doorl.env.payout."""

from __future__ import annotations

import math

import pytest

from doorl.env.payout import (
    break_even_check,
    calculate_multiplier,
    calculate_payout,
    host_pool_pnl,
)


@pytest.mark.parametrize("tau", [0.10, 0.20, 0.25, 0.30])
def test_break_even_at_tau(tau: float) -> None:
    """At x = tau total payouts must equal P (Host break-even on the pool)."""
    P = 1000.0
    W = tau * P
    # one virtual winner with the entire winning volume
    payout = calculate_payout(W, W, P, tau=tau, max_multiplier=None)
    assert math.isclose(payout, P, rel_tol=1e-9, abs_tol=1e-7), (
        f"At tau={tau} expected payout=P={P}, got {payout}"
    )
    assert break_even_check(tau, max_multiplier=None)


@pytest.mark.parametrize("tau", [0.10, 0.20, 0.25, 0.30])
def test_host_profit_below_tau(tau: float) -> None:
    """For x < tau the Host nets > 0 from the pool."""
    P = 1000.0
    x = tau / 2.0
    W = x * P
    pnl = host_pool_pnl(P, W, tau=tau, max_multiplier=None)
    assert pnl > 0.0


@pytest.mark.parametrize("tau", [0.10, 0.20, 0.25, 0.30])
def test_host_loss_above_tau(tau: float) -> None:
    """For x > tau the Host nets < 0 from the pool."""
    P = 1000.0
    x = min(0.95, tau * 2.0)
    W = x * P
    pnl = host_pool_pnl(P, W, tau=tau, max_multiplier=None)
    assert pnl < 0.0


def test_cap_at_small_x() -> None:
    """At x = 1e-3 with cap=50, the multiplier must hit exactly 50."""
    m = calculate_multiplier(1e-3, tau=0.20, max_multiplier=50.0)
    assert math.isclose(m, 50.0)


def test_no_cap_at_small_x() -> None:
    """Without a cap, multiplier ~ 1 + alpha/x is huge but finite."""
    tau = 0.20
    x = 1e-3
    m = calculate_multiplier(x, tau=tau, max_multiplier=None)
    expected = 1.0 + (1.0 - tau) / x
    assert math.isclose(m, expected, rel_tol=1e-12)
    assert m > 500.0


def test_zero_pool_returns_zero() -> None:
    assert calculate_payout(0.0, 0.0, 0.0, tau=0.20) == 0.0
    assert host_pool_pnl(0.0, 0.0, tau=0.20) == 0.0


def test_zero_winning_volume_returns_zero_and_host_keeps_pool() -> None:
    """W = 0: payout undefined; host_pool_pnl returns full pool (E1 default)."""
    P = 500.0
    assert calculate_payout(0.0, 0.0, P, tau=0.20) == 0.0
    assert host_pool_pnl(P, 0.0, tau=0.20) == pytest.approx(P)


def test_invalid_tau_raises() -> None:
    with pytest.raises(ValueError):
        calculate_multiplier(0.5, tau=0.0)
    with pytest.raises(ValueError):
        calculate_multiplier(0.5, tau=1.0)


def test_invalid_x_raises() -> None:
    with pytest.raises(ValueError):
        calculate_multiplier(0.0, tau=0.20)
    with pytest.raises(ValueError):
        calculate_multiplier(-0.1, tau=0.20)
