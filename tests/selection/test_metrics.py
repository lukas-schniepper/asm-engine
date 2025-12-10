"""
Test suite for Portfolio Selection metrics calculations.

Tests verify that all risk metrics are calculated correctly according to
industry-standard formulas.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import date, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from AlphaMachine_core.selection.optimizer import (
    calculate_ulcer_index,
    calculate_upi,
    calculate_candidate_metrics,
)
from AlphaMachine_core.tracking.metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_cagr,
    calculate_max_drawdown,
    calculate_volatility,
    calculate_returns,
    TRADING_DAYS_PER_YEAR,
)


def create_synthetic_nav(
    annual_return: float = 0.10,
    annual_volatility: float = 0.15,
    days: int = 252,
    start_nav: float = 100.0,
    seed: int = 42,
) -> pd.Series:
    """
    Create synthetic NAV series with known characteristics.

    Args:
        annual_return: Target annual return (e.g., 0.10 for 10%)
        annual_volatility: Target annual volatility (e.g., 0.15 for 15%)
        days: Number of trading days
        start_nav: Starting NAV
        seed: Random seed for reproducibility

    Returns:
        NAV Series with DatetimeIndex
    """
    np.random.seed(seed)

    # Daily return parameters
    daily_return = annual_return / TRADING_DAYS_PER_YEAR
    daily_vol = annual_volatility / np.sqrt(TRADING_DAYS_PER_YEAR)

    # Generate daily returns
    returns = np.random.normal(daily_return, daily_vol, days)

    # Build NAV
    nav_values = [start_nav]
    for r in returns:
        nav_values.append(nav_values[-1] * (1 + r))

    # Create date index
    dates = pd.date_range(start=date.today() - timedelta(days=days), periods=days + 1, freq='B')

    return pd.Series(nav_values, index=dates[:len(nav_values)])


def test_sharpe_ratio_basic():
    """Test Sharpe ratio calculation with synthetic data."""
    print("\n=== Test: Sharpe Ratio Basic ===")

    # Create NAV with ~10% return, ~15% vol -> Sharpe ~0.67
    nav = create_synthetic_nav(annual_return=0.10, annual_volatility=0.15, days=252)
    returns = calculate_returns(nav)
    sharpe = calculate_sharpe_ratio(returns)

    print(f"  Expected Sharpe: ~0.67 (10%/15%)")
    print(f"  Calculated Sharpe: {sharpe:.3f}")

    # Allow some variance due to random sampling
    assert 0.3 < sharpe < 1.2, f"Sharpe {sharpe} outside expected range [0.3, 1.2]"
    print("  [PASS]")


def test_sharpe_ratio_zero_vol():
    """Test Sharpe ratio with zero volatility (constant returns)."""
    print("\n=== Test: Sharpe Ratio Zero Volatility ===")

    # Constant returns -> zero vol
    nav = pd.Series([100, 100.1, 100.2, 100.3, 100.4],
                    index=pd.date_range(start='2024-01-01', periods=5))
    returns = calculate_returns(nav)
    sharpe = calculate_sharpe_ratio(returns)

    print(f"  Calculated Sharpe: {sharpe}")
    # With very low variance, Sharpe should be very high or handled gracefully
    assert sharpe >= 0 or sharpe == 0, "Sharpe should be non-negative or zero for edge case"
    print("  [PASS]")


def test_sortino_ratio():
    """Test Sortino ratio calculation."""
    print("\n=== Test: Sortino Ratio ===")

    # Sortino should be >= Sharpe since it only penalizes downside
    nav = create_synthetic_nav(annual_return=0.10, annual_volatility=0.15, days=252)
    returns = calculate_returns(nav)

    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)

    print(f"  Sharpe: {sharpe:.3f}")
    print(f"  Sortino: {sortino:.3f}")

    # Sortino typically >= Sharpe (fewer downside observations than total vol)
    assert sortino != 0, "Sortino should not be zero for normal returns"
    print("  [PASS]")


def test_max_drawdown():
    """Test maximum drawdown calculation."""
    print("\n=== Test: Max Drawdown ===")

    # Known drawdown: 100 -> 120 -> 90 -> 100
    # Peak at 120, trough at 90, DD = (90-120)/120 = -25%
    nav = pd.Series([100, 110, 120, 100, 90, 95, 100],
                    index=pd.date_range(start='2024-01-01', periods=7))

    max_dd = calculate_max_drawdown(nav)

    print(f"  Expected Max DD: -0.25 (-25%)")
    print(f"  Calculated Max DD: {max_dd:.4f}")

    assert abs(max_dd - (-0.25)) < 0.001, f"Max DD {max_dd} != -0.25"
    print("  [PASS]")


def test_cagr():
    """Test CAGR calculation."""
    print("\n=== Test: CAGR ===")

    # Start: 100, End: 110 after 252 days (1 year) -> CAGR = 10%
    dates = pd.date_range(start='2024-01-01', periods=252, freq='B')
    nav = pd.Series([100] + [100 + i * 10 / 251 for i in range(1, 252)], index=dates)
    nav.iloc[-1] = 110  # Ensure exact end value

    cagr = calculate_cagr(nav)

    print(f"  Start NAV: {nav.iloc[0]}")
    print(f"  End NAV: {nav.iloc[-1]}")
    print(f"  Days: {len(nav)}")
    print(f"  Expected CAGR: ~0.10 (10%)")
    print(f"  Calculated CAGR: {cagr:.4f}")

    assert 0.08 < cagr < 0.12, f"CAGR {cagr} outside expected range [0.08, 0.12]"
    print("  [PASS]")


def test_calmar_ratio():
    """Test Calmar ratio calculation."""
    print("\n=== Test: Calmar Ratio ===")

    nav = create_synthetic_nav(annual_return=0.10, annual_volatility=0.15, days=252)

    cagr = calculate_cagr(nav)
    max_dd = calculate_max_drawdown(nav)
    calmar = calculate_calmar_ratio(nav)

    expected_calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    print(f"  CAGR: {cagr:.4f}")
    print(f"  Max DD: {max_dd:.4f}")
    print(f"  Expected Calmar: {expected_calmar:.4f}")
    print(f"  Calculated Calmar: {calmar:.4f}")

    assert abs(calmar - expected_calmar) < 0.01, f"Calmar mismatch"
    print("  [PASS]")


def test_ulcer_index():
    """Test Ulcer Index calculation."""
    print("\n=== Test: Ulcer Index ===")

    # Known drawdown pattern
    # 100 -> 120 -> 90 -> 100
    # Drawdowns: 0%, 0%, -25%, -16.67%
    nav = pd.Series([100, 110, 120, 100, 90, 95, 100],
                    index=pd.date_range(start='2024-01-01', periods=7))

    ulcer = calculate_ulcer_index(nav)

    print(f"  NAV series: {nav.values}")
    print(f"  Calculated Ulcer Index: {ulcer:.4f}")

    # Ulcer should be positive and reflect drawdown severity
    assert ulcer > 0, "Ulcer Index should be positive when drawdowns exist"
    assert ulcer < 30, "Ulcer Index seems too high"
    print("  [PASS]")


def test_ulcer_index_no_drawdown():
    """Test Ulcer Index with no drawdowns."""
    print("\n=== Test: Ulcer Index No Drawdown ===")

    # Monotonically increasing NAV
    nav = pd.Series([100, 101, 102, 103, 104, 105],
                    index=pd.date_range(start='2024-01-01', periods=6))

    ulcer = calculate_ulcer_index(nav)

    print(f"  NAV series: {nav.values}")
    print(f"  Calculated Ulcer Index: {ulcer:.4f}")

    assert ulcer == 0 or ulcer < 0.001, "Ulcer Index should be ~0 with no drawdowns"
    print("  [PASS]")


def test_upi():
    """Test Ulcer Performance Index calculation."""
    print("\n=== Test: Ulcer Performance Index ===")

    nav = create_synthetic_nav(annual_return=0.10, annual_volatility=0.15, days=252)

    cagr = calculate_cagr(nav)
    ulcer = calculate_ulcer_index(nav)
    upi = calculate_upi(nav)

    print(f"  CAGR: {cagr:.4f}")
    print(f"  Ulcer Index: {ulcer:.4f}")
    print(f"  Expected UPI: {cagr / (ulcer/100) if ulcer > 0 else 0:.4f}")
    print(f"  Calculated UPI: {upi:.4f}")

    assert upi != 0, "UPI should not be zero for normal returns with drawdowns"
    print("  [PASS]")


def test_candidate_metrics():
    """Test full candidate metrics calculation."""
    print("\n=== Test: Candidate Metrics ===")

    nav = create_synthetic_nav(annual_return=0.12, annual_volatility=0.18, days=252)
    returns = calculate_returns(nav)

    metrics = calculate_candidate_metrics(nav, returns)

    print(f"  Metrics calculated:")
    for key, value in metrics.items():
        print(f"    {key}: {value:.4f}")

    # Check all expected keys exist
    expected_keys = ["sharpe", "sortino", "calmar", "upi", "cagr", "max_dd", "volatility"]
    for key in expected_keys:
        assert key in metrics, f"Missing metric: {key}"

    # Check values are reasonable
    assert metrics["sharpe"] != 0, "Sharpe should not be zero"
    assert metrics["max_dd"] <= 0, "Max DD should be negative or zero"
    assert metrics["volatility"] > 0, "Volatility should be positive"

    print("  [PASS]")


def test_volatility():
    """Test volatility calculation."""
    print("\n=== Test: Volatility ===")

    # Create NAV with known volatility
    nav = create_synthetic_nav(annual_return=0.0, annual_volatility=0.20, days=252)
    returns = calculate_returns(nav)
    vol = calculate_volatility(returns)

    print(f"  Target annual vol: 20%")
    print(f"  Calculated annual vol: {vol*100:.1f}%")

    # Should be close to 20%
    assert 0.10 < vol < 0.35, f"Volatility {vol} outside expected range"
    print("  [PASS]")


def run_all_tests():
    """Run all metric tests."""
    print("=" * 60)
    print("PORTFOLIO SELECTION - METRIC TESTS")
    print("=" * 60)

    tests = [
        test_sharpe_ratio_basic,
        test_sharpe_ratio_zero_vol,
        test_sortino_ratio,
        test_max_drawdown,
        test_cagr,
        test_calmar_ratio,
        test_ulcer_index,
        test_ulcer_index_no_drawdown,
        test_upi,
        test_candidate_metrics,
        test_volatility,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  [FAILED]: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR]: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
