"""Tiered numerical tolerances for parity assertions.

Per the senior-quant review (2026-05-04): a single 1e-10 tolerance is the
wrong rule because float64 ULP at NAV magnitudes is ~1e-10, so we'd be
comparing at the noise floor and hit false alarms when pandas/numpy
versions, ARM-vs-x86 hosts, or accumulator orders shift things by 1 ULP.

Use the right tolerance for each metric class:

  EXACT — counts, dates, ids; bit-exact equality required
  SUMS  — annualized excess return, mean of returns; rtol=1e-9, atol=1e-12
  PATH  — rolling stats, cumulative products, drawdowns; rtol=1e-7, atol=1e-9
  OPT   — anything that touches scipy.optimize or rolling regressions;
          rtol=1e-5, atol=1e-6
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

EXACT: Tuple[float, float] = (0.0, 0.0)
SUMS: Tuple[float, float] = (1e-9, 1e-12)
PATH: Tuple[float, float] = (1e-7, 1e-9)
OPT: Tuple[float, float] = (1e-5, 1e-6)


def assert_class(actual, expected, tol_class: Tuple[float, float], msg: str = "") -> None:
    """Assert `actual` ≈ `expected` within the given tolerance class.

    Works on scalars, numpy arrays, and pandas Series.
    """
    rtol, atol = tol_class
    if rtol == 0.0 and atol == 0.0:
        if hasattr(actual, "equals"):
            assert actual.equals(expected), f"EXACT mismatch{(': ' + msg) if msg else ''}"
        else:
            assert actual == expected, f"EXACT mismatch{(': ' + msg) if msg else ''}: {actual!r} != {expected!r}"
        return
    np.testing.assert_allclose(
        actual, expected, rtol=rtol, atol=atol, err_msg=msg or None
    )


def metric_class(metric_name: str) -> Tuple[float, float]:
    """Map a metric/function name to its expected tolerance class.

    Add to this map as new metrics are wired through the parity tests.
    """
    name = metric_name.lower()
    if name in {
        "calculate_volatility", "calculate_sharpe", "calculate_sortino",
        "downside_semi_deviation",
    }:
        return SUMS
    if name in {
        "calculate_cagr", "calculate_max_drawdown", "calculate_calmar",
        "calculate_ulcer_index", "_nav_from_returns",
    }:
        return PATH
    if name in {"optimize_params", "run_backtest", "walk_forward"}:
        return OPT
    return PATH  # default — safer than SUMS for unknowns
