"""Regime fixtures for parity testing.

Per the senior-quant review: a single NAV fixture is malpractice for a
GIPS-aligned strategy. We need adversarial scenarios so the harness
catches bugs that only surface in specific market regimes.

Each fixture returns a `pd.Series` of daily returns with a date index.
Synthesis is deterministic via a fixed seed so the fixtures are
reproducible across machines without committing parquet binaries.

All series share the same length (252 trading days = 1 year) so
property tests can mix and match without length surprises.
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import pandas as pd

LEN = 252  # 1 trading year — enough for annualized metrics


def _dates(start: str, n: int = LEN) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def _calm_drift(seed: int, mean: float = 0.0004, vol: float = 0.005) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=mean, scale=vol, size=LEN)


# ------------------------------------------------------------------
# Adversarial regimes (named to match the migration plan tracker)
# ------------------------------------------------------------------

def regime_2008_lehman() -> pd.Series:
    """High volatility + ~50% drawdown over the year."""
    rng = np.random.default_rng(2008)
    base = rng.normal(loc=-0.0015, scale=0.025, size=LEN)
    # Inject a Lehman-week shock
    base[180:185] = [-0.05, -0.07, -0.06, -0.04, -0.05]
    return pd.Series(base, index=_dates("2008-01-02"), name="r")


def regime_2020_q1_covid() -> pd.Series:
    """Rapid 35% drawdown concentrated in 25 trading days."""
    rng = np.random.default_rng(2020)
    r = _calm_drift(seed=20201)
    r[40:65] = rng.normal(loc=-0.018, scale=0.04, size=25)
    return pd.Series(r, index=_dates("2020-01-02"), name="r")


def regime_2022_grind() -> pd.Series:
    """Prolonged ~25% drawdown over ~9 months."""
    rng = np.random.default_rng(2022)
    r = rng.normal(loc=-0.0008, scale=0.012, size=LEN)
    return pd.Series(r, index=_dates("2022-01-03"), name="r")


def regime_2017_lowvol() -> pd.Series:
    """Calm bull-market low-vol year."""
    return pd.Series(
        _calm_drift(seed=2017, mean=0.00065, vol=0.004),
        index=_dates("2017-01-03"),
        name="r",
    )


def regime_all_cash() -> pd.Series:
    """All-zero return series. Tests no-downside / no-volatility paths."""
    return pd.Series(np.zeros(LEN), index=_dates("2024-01-02"), name="r")


def regime_single_position_high_vol() -> pd.Series:
    """Mimics holding a single concentrated name — fat-tailed daily moves."""
    rng = np.random.default_rng(7)
    r = rng.standard_t(df=4, size=LEN) * 0.025  # heavy-tail t(4)
    return pd.Series(r, index=_dates("2023-01-02"), name="r")


def regime_negative_sharpe() -> pd.Series:
    """Drift down series — Sharpe negative, Sortino very negative."""
    rng = np.random.default_rng(31)
    r = rng.normal(loc=-0.0012, scale=0.008, size=LEN)
    return pd.Series(r, index=_dates("2018-01-02"), name="r")


def regime_synthetic_full_drawdown() -> pd.Series:
    """Cascade to deep drawdown without going below -1.0 daily.

    Tests max-drawdown computation and the no_downside_value path edges.
    Starting NAV 1.0 → ending NAV ≈ 0.05 (95% drawdown).
    """
    n = LEN
    # Smooth -1.5% per day for the first 200 days, then -0.5% to settle
    r = np.empty(n)
    r[:200] = -0.015
    r[200:] = -0.005
    rng = np.random.default_rng(99)
    r = r + rng.normal(0, 0.002, size=n)
    return pd.Series(r, index=_dates("2010-01-04"), name="r")


REGIMES: Dict[str, Callable[[], pd.Series]] = {
    "2008-lehman": regime_2008_lehman,
    "2020-q1-covid": regime_2020_q1_covid,
    "2022-grind": regime_2022_grind,
    "2017-lowvol": regime_2017_lowvol,
    "all-cash": regime_all_cash,
    "single-position-high-vol": regime_single_position_high_vol,
    "negative-sharpe": regime_negative_sharpe,
    "synthetic-full-drawdown": regime_synthetic_full_drawdown,
}


def all_regimes() -> Dict[str, pd.Series]:
    return {name: fn() for name, fn in REGIMES.items()}
