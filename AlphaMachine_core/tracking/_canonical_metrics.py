"""
GIPS-aligned quant performance metrics — VENDORED COPY.

This file is a **vendored copy** of `shared/data/src/utils/quant_metrics.py`
from the private `asm-data` repo. It exists in asm-engine because the
deploy target (Streamlit Cloud) cannot clone the private submodule —
Streamlit Cloud does not expose GH_PAT / deploy keys at the git-clone phase.

Source of truth: lukas-schniepper/asm-data, file `src/utils/quant_metrics.py`.

When you change `calculate_sortino` (or any other metric here) in asm-data,
sync the change into this file. The two should stay byte-equivalent — there
are unit tests in asm-data that pin the canonical formula.

Convention:
    - All metrics take a daily-return Series indexed by date (or a numpy
      array of daily returns). Inputs are decimals, e.g. 0.005 for +0.5%.
    - `mar` (Minimum Acceptable Return) is annualized; the function
      converts it to daily internally. Default 0% (no excess threshold).
    - `periods_per_year` defaults to 252 (US trading days). For weekly
      data, pass 52; for monthly, 12.
    - Annualized return uses the **compounded** form
      (1 + r)^periods_per_year - 1 where applicable, so results are
      GIPS-compliant (no arithmetic-mean shortcut).
    - All functions are pure: no I/O, no side effects, no global state.

Module is intentionally dependency-light: numpy + pandas only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "calculate_cagr",
    "calculate_volatility",
    "calculate_sharpe",
    "calculate_sortino",
    "calculate_max_drawdown",
    "calculate_calmar",
    "calculate_ulcer_index",
    "downside_semi_deviation",
]


TRADING_DAYS_PER_YEAR = 252


def _to_array(returns) -> np.ndarray:
    """Coerce input to a clean 1-D numpy array of daily returns (NaN-stripped)."""
    if isinstance(returns, pd.Series):
        return returns.dropna().to_numpy(dtype=float)
    arr = np.asarray(returns, dtype=float)
    return arr[~np.isnan(arr)]


def _daily_mar(mar: float, periods_per_year: int) -> float:
    """Convert annualized MAR (e.g. 0.04 = 4%) to per-period equivalent."""
    if mar == 0.0:
        return 0.0
    return float((1.0 + mar) ** (1.0 / periods_per_year) - 1.0)


# ---------------------------------------------------------------------------
# Core building blocks
# ---------------------------------------------------------------------------

def calculate_cagr(returns, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Compounded annual growth rate from a return series."""
    r = _to_array(returns)
    if len(r) < 2:
        return 0.0
    cum = float(np.prod(1.0 + r))
    if cum <= 0:
        return float("nan")
    n_years = len(r) / periods_per_year
    return float(cum ** (1.0 / n_years) - 1.0)


def calculate_volatility(returns, periods_per_year: int = TRADING_DAYS_PER_YEAR,
                          ddof: int = 1) -> float:
    """Annualized total volatility (standard deviation of returns)."""
    r = _to_array(returns)
    if len(r) < 2:
        return 0.0
    return float(np.std(r, ddof=ddof) * np.sqrt(periods_per_year))


def downside_semi_deviation(returns, mar: float = 0.0,
                              periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualized downside semi-deviation — denominator of standard Sortino.

        sqrt( mean( min(0, r_t - daily_mar)**2 ) ) * sqrt(periods_per_year)

    Critically, the mean is over the **full sample**, not just the negative-
    return subset.
    """
    r = _to_array(returns)
    if len(r) < 1:
        return 0.0
    daily_mar = _daily_mar(mar, periods_per_year)
    neg_dev = np.minimum(0.0, r - daily_mar)
    if (neg_dev != 0).sum() == 0:
        return 0.0
    return float(np.sqrt(np.mean(neg_dev ** 2)) * np.sqrt(periods_per_year))


# ---------------------------------------------------------------------------
# Risk-adjusted-return ratios
# ---------------------------------------------------------------------------

def calculate_sharpe(returns, mar: float = 0.0,
                      periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualized Sharpe ratio: (annualized excess return) / (annualized total vol)."""
    r = _to_array(returns)
    if len(r) < 2:
        return 0.0
    daily_mar = _daily_mar(mar, periods_per_year)
    excess = r - daily_mar
    vol = float(np.std(r, ddof=1) * np.sqrt(periods_per_year))
    if vol < 1e-12:
        return 0.0
    ann_excess = float(np.mean(excess) * periods_per_year)
    return ann_excess / vol


def calculate_sortino(returns, mar: float = 0.0,
                       periods_per_year: int = TRADING_DAYS_PER_YEAR,
                       no_downside_value: float = 10.0) -> float:
    """GIPS-aligned annualized Sortino ratio.

        Sortino = (annualized excess return) / (annualized downside semi-deviation)

    Downside semi-deviation uses the textbook formula
    `sqrt(mean(min(0, r - daily_mar)**2))` averaged over the FULL sample
    (NOT std of the negative-only subset, which is the common bug).

    Mathematical guarantee: when annualized excess return > 0, this Sortino
    is always >= the Sharpe ratio computed by `calculate_sharpe` on the same
    series, because semi-deviation <= total volatility by construction.
    """
    r = _to_array(returns)
    if len(r) < 2:
        return 0.0
    daily_mar = _daily_mar(mar, periods_per_year)
    excess = r - daily_mar
    semi_dev = downside_semi_deviation(returns, mar=mar, periods_per_year=periods_per_year)
    if semi_dev == 0:
        return float(no_downside_value)
    ann_excess = float(np.mean(excess) * periods_per_year)
    return ann_excess / semi_dev


# ---------------------------------------------------------------------------
# Drawdown family
# ---------------------------------------------------------------------------

def _nav_from_returns(returns) -> np.ndarray:
    """Build a cumulative-NAV series from daily returns, starting at 1.0."""
    r = _to_array(returns)
    if len(r) == 0:
        return np.array([1.0])
    return np.concatenate([[1.0], np.cumprod(1.0 + r)])


def calculate_max_drawdown(returns_or_nav, is_nav: bool = False) -> float:
    """Maximum drawdown as a negative decimal (e.g. -0.15 for -15%)."""
    if is_nav:
        nav = _to_array(returns_or_nav)
    else:
        nav = _nav_from_returns(returns_or_nav)
    if len(nav) < 2:
        return 0.0
    peak = np.maximum.accumulate(nav)
    dd = (nav / peak) - 1.0
    return float(np.min(dd))


def calculate_calmar(returns, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Calmar ratio: CAGR / |max drawdown|. Returns 0 when no drawdown."""
    cagr = calculate_cagr(returns, periods_per_year=periods_per_year)
    mdd = calculate_max_drawdown(returns)
    if mdd >= 0:
        return 0.0
    return cagr / abs(mdd)


def calculate_ulcer_index(returns) -> float:
    """Ulcer Index: RMS of percentage drawdowns."""
    nav = _nav_from_returns(returns)
    if len(nav) < 2:
        return 0.0
    peak = np.maximum.accumulate(nav)
    dd_pct = (nav / peak - 1.0) * 100.0
    return float(np.sqrt(np.mean(dd_pct ** 2)))
