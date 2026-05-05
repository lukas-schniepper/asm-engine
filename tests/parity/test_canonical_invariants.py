"""Invariants the canonical metric helper must satisfy on every regime.

These are correctness checks (not parity-vs-something), but they live in
tests/parity/ because they share the regime fixture panel and tolerance
classes. CI runs them on every commit that touches service/ or
AlphaMachine_core/.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from AlphaMachine_core.tracking._canonical_metrics import (
    calculate_calmar,
    calculate_cagr,
    calculate_max_drawdown,
    calculate_sharpe,
    calculate_sortino,
    calculate_ulcer_index,
    calculate_volatility,
    downside_semi_deviation,
)
from tests.parity.fixtures import REGIMES, all_regimes
from tests.parity.tolerance import PATH, SUMS, assert_class


# ---------------------------------------------------------------------------
# Per-regime sanity invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(REGIMES.keys()))
def test_metrics_finite_or_documented(name: str) -> None:
    """Every metric must produce a finite float for every regime, except
    where the helper documents NaN behavior (CAGR on cum<=0)."""
    r = REGIMES[name]()

    sortino = calculate_sortino(r)
    sharpe = calculate_sharpe(r)
    vol = calculate_volatility(r)
    mdd = calculate_max_drawdown(r)
    ulcer = calculate_ulcer_index(r)
    semi = downside_semi_deviation(r)

    for label, val in [
        ("sortino", sortino),
        ("sharpe", sharpe),
        ("vol", vol),
        ("mdd", mdd),
        ("ulcer", ulcer),
        ("semi", semi),
    ]:
        assert math.isfinite(val), f"{label} on {name} is non-finite: {val}"

    # CAGR can be NaN only when cumulative product is non-positive.
    cum = float(np.prod(1.0 + r.to_numpy()))
    cagr = calculate_cagr(r)
    if cum > 0:
        assert math.isfinite(cagr), f"CAGR on {name} non-finite despite cum={cum}"
    # Calmar mirrors CAGR's NaN behavior.


@pytest.mark.parametrize("name", list(REGIMES.keys()))
def test_max_drawdown_non_positive(name: str) -> None:
    """Max drawdown must always be <= 0 by definition."""
    mdd = calculate_max_drawdown(REGIMES[name]())
    assert mdd <= 0.0, f"MDD positive on {name}: {mdd}"


@pytest.mark.parametrize("name", list(REGIMES.keys()))
def test_volatility_non_negative(name: str) -> None:
    vol = calculate_volatility(REGIMES[name]())
    assert vol >= 0.0


@pytest.mark.parametrize("name", list(REGIMES.keys()))
def test_semi_deviation_le_volatility(name: str) -> None:
    """Semi-deviation must always be <= total volatility (semi is the
    sqrt of mean of squared *negative* deviations only; total vol is sqrt
    of mean of squared deviations including positives)."""
    r = REGIMES[name]()
    vol = calculate_volatility(r)
    # downside_semi_deviation returns annualized; calculate_volatility
    # returns annualized — same units.
    semi = downside_semi_deviation(r)

    # When MAR=0, semi <= vol when daily mean is approximately 0.
    # Strict inequality `semi <= vol` holds modulo tiny float noise.
    assert semi <= vol + 1e-9, f"Semi {semi} > Vol {vol} on {name}"


# ---------------------------------------------------------------------------
# Cross-metric mathematical guarantees
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(REGIMES.keys()))
def test_sortino_geq_sharpe_when_excess_positive(name: str) -> None:
    """Mathematical guarantee per the canonical helper docstring: when
    annualized excess return > 0, Sortino >= Sharpe (because
    semi-deviation <= total volatility by construction)."""
    r = REGIMES[name]()
    excess_ann = float(r.mean() * 252)
    if excess_ann <= 0:
        pytest.skip(f"{name} has non-positive excess return")
    sortino = calculate_sortino(r)
    sharpe = calculate_sharpe(r)
    # Allow PATH-class slack for floating-point noise.
    assert sortino + PATH[1] >= sharpe, f"Sortino {sortino} < Sharpe {sharpe} on {name}"


def test_all_cash_regime_yields_zero_metrics() -> None:
    """All-zero returns: every metric except no-downside Sortino must be 0."""
    r = REGIMES["all-cash"]()

    assert_class(calculate_volatility(r), 0.0, SUMS, "vol")
    assert_class(calculate_sharpe(r), 0.0, SUMS, "sharpe")
    assert_class(calculate_max_drawdown(r), 0.0, SUMS, "mdd")
    assert_class(downside_semi_deviation(r), 0.0, SUMS, "semi")
    # Sortino returns no_downside_value (default 10.0) when semi-dev == 0.
    assert calculate_sortino(r) == 10.0
    assert calculate_calmar(r) == 0.0
    assert_class(calculate_ulcer_index(r), 0.0, SUMS, "ulcer")


def test_negative_sharpe_regime_actually_negative() -> None:
    r = REGIMES["negative-sharpe"]()
    assert calculate_sharpe(r) < 0
    assert calculate_sortino(r) < 0


def test_full_drawdown_regime_mdd_below_minus_50pct() -> None:
    r = REGIMES["synthetic-full-drawdown"]()
    mdd = calculate_max_drawdown(r)
    assert mdd < -0.5, f"Synthetic full-drawdown should drop > 50%, got {mdd}"
