"""Property-based fuzz tests for canonical metrics.

Per the senior-quant review: 30 nightly fixed-fixture runs prove the same
fixture passes 30 times — that's one bit of information, not 30. Property
tests randomly sample inputs from the legal space, run the canonical
metrics, and check invariants. Catches edge cases the regime panel misses.

Default: 100 randomized runs per CI invocation. On release tags the CI
workflow bumps `--hypothesis-profile=release` for 1000.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st
from hypothesis.extra.numpy import arrays

from AlphaMachine_core.tracking._canonical_metrics import (
    calculate_max_drawdown,
    calculate_sharpe,
    calculate_sortino,
    calculate_volatility,
    downside_semi_deviation,
)


# Daily returns must be in [-0.5, 0.5] — anything outside that range would
# fail the Phase 1 DQ gate before reaching the engine, so we don't need to
# fuzz against it. We also exclude NaN/inf because _to_array filters NaN
# and (per F2 in the numerical-stability audit) inf must be DQ-rejected.
RETURN_ELEMENT = st.floats(
    min_value=-0.5,
    max_value=0.5,
    allow_nan=False,
    allow_infinity=False,
    width=64,
)


@settings(
    max_examples=100,
    deadline=None,  # canonical helpers are pure-numeric, hypothesis timeouts here are noise
    suppress_health_check=[HealthCheck.too_slow],
)
@given(arr=arrays(np.float64, st.integers(min_value=20, max_value=2000), elements=RETURN_ELEMENT))
def test_metrics_finite(arr) -> None:
    s = pd.Series(arr)

    for fn in (calculate_sharpe, calculate_sortino, calculate_volatility, downside_semi_deviation, calculate_max_drawdown):
        v = fn(s)
        assert math.isfinite(v), f"{fn.__name__} produced non-finite on length-{len(arr)} fuzz"


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(arr=arrays(np.float64, st.integers(min_value=20, max_value=2000), elements=RETURN_ELEMENT))
def test_max_drawdown_non_positive(arr) -> None:
    s = pd.Series(arr)
    mdd = calculate_max_drawdown(s)
    assert mdd <= 0.0, f"MDD positive: {mdd}"


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(arr=arrays(np.float64, st.integers(min_value=20, max_value=2000), elements=RETURN_ELEMENT))
def test_volatility_non_negative(arr) -> None:
    s = pd.Series(arr)
    assert calculate_volatility(s) >= 0.0


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(arr=arrays(np.float64, st.integers(min_value=20, max_value=2000), elements=RETURN_ELEMENT))
def test_semi_dev_bounded(arr) -> None:
    """Semi-deviation is non-negative and finite. The "semi <= vol" bound
    that the canonical docstring claims "by construction" only holds when
    E[r] ≈ MAR — see test_semi_deviation_bounded in
    test_canonical_invariants.py for the explanation. Property fuzzing
    routinely generates series with E[r] far from 0, where the inequality
    fails. The bound on Sortino vs Sharpe still holds; just not on the
    raw denominators."""
    s = pd.Series(arr)
    semi = downside_semi_deviation(s)
    assert math.isfinite(semi), f"semi non-finite: {semi}"
    assert semi >= 0.0, f"semi negative: {semi}"


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(arr=arrays(np.float64, st.integers(min_value=20, max_value=2000), elements=RETURN_ELEMENT))
def test_sortino_geq_sharpe_when_positive_excess(arr) -> None:
    """Mathematical guarantee from the canonical docstring."""
    s = pd.Series(arr)
    excess_ann = float(s.mean() * 252)
    if excess_ann <= 0:
        return  # property only holds when excess > 0
    sortino = calculate_sortino(s)
    sharpe = calculate_sharpe(s)
    # Skip cases where Sortino used no_downside_value (10.0) because there
    # was no downside in the sample — that doesn't speak to the comparison.
    if sortino == 10.0 and downside_semi_deviation(s) == 0.0:
        return
    # PATH-class slack for float-noise edge cases.
    assert sortino + 1e-7 >= sharpe, f"sortino {sortino} < sharpe {sharpe} ann_excess={excess_ann}"
