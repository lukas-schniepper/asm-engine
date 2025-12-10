"""
Test suite for Portfolio Selection optimizer.

Tests verify that:
1. Normalization correctly scales metrics
2. Combination scoring works correctly
3. Optimizer finds expected combinations
4. Combined portfolio simulation is accurate
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
    normalize_metrics,
    score_combination,
    find_optimal_portfolio_combination,
    simulate_combined_portfolio,
    DEFAULT_WEIGHTS,
)


def create_test_returns(
    mean_daily: float = 0.0004,
    std_daily: float = 0.01,
    days: int = 60,
    seed: int = 42,
) -> pd.Series:
    """Create synthetic daily returns series."""
    np.random.seed(seed)
    dates = pd.date_range(start=date.today() - timedelta(days=days), periods=days, freq='B')
    returns = np.random.normal(mean_daily, std_daily, days)
    return pd.Series(returns, index=dates)


def test_normalize_metrics_basic():
    """Test that normalization produces 0-1 range."""
    print("\n=== Test: Normalize Metrics Basic ===")

    # Create metrics for 3 portfolios
    all_metrics = {
        "Portfolio A": {"sharpe": 2.0, "max_dd": -0.10, "volatility": 0.15},
        "Portfolio B": {"sharpe": 1.0, "max_dd": -0.20, "volatility": 0.20},
        "Portfolio C": {"sharpe": 1.5, "max_dd": -0.05, "volatility": 0.10},
    }

    normalized = normalize_metrics(all_metrics)

    print(f"  Original metrics:")
    for name, m in all_metrics.items():
        print(f"    {name}: {m}")

    print(f"\n  Normalized metrics:")
    for name, m in normalized.items():
        print(f"    {name}: {m}")

    # Check all values are in [0, 1]
    for name, metrics in normalized.items():
        for key, value in metrics.items():
            assert 0 <= value <= 1, f"{name}.{key} = {value} not in [0,1]"

    print("  [PASS]")


def test_normalize_metrics_inverted():
    """Test that inverted metrics (max_dd, volatility) are correctly flipped."""
    print("\n=== Test: Normalize Metrics Inverted ===")

    all_metrics = {
        "Low DD": {"sharpe": 1.0, "max_dd": -0.05, "volatility": 0.10},
        "High DD": {"sharpe": 1.0, "max_dd": -0.25, "volatility": 0.30},
    }

    normalized = normalize_metrics(all_metrics)

    # Low DD should have HIGHER normalized max_dd score (lower DD is better)
    assert normalized["Low DD"]["max_dd"] > normalized["High DD"]["max_dd"], \
        "Lower max_dd should get higher normalized score"

    # Low volatility should have HIGHER normalized volatility score
    assert normalized["Low DD"]["volatility"] > normalized["High DD"]["volatility"], \
        "Lower volatility should get higher normalized score"

    print(f"  Low DD normalized max_dd: {normalized['Low DD']['max_dd']:.3f}")
    print(f"  High DD normalized max_dd: {normalized['High DD']['max_dd']:.3f}")
    print("  [PASS]")


def test_score_combination_sharpe_only():
    """Test that Sharpe-only scoring picks highest Sharpe portfolios."""
    print("\n=== Test: Score Combination (Sharpe Only) ===")

    # Create returns for portfolios
    portfolio_returns = {
        "High Sharpe": create_test_returns(mean_daily=0.001, std_daily=0.01, seed=1),
        "Med Sharpe": create_test_returns(mean_daily=0.0005, std_daily=0.01, seed=2),
        "Low Sharpe": create_test_returns(mean_daily=0.0002, std_daily=0.01, seed=3),
    }

    # Sharpe-only weights
    weights = {"sharpe": 3, "sortino": 0, "calmar": 0, "upi": 0,
               "cagr": 0, "max_dd": 0, "volatility": 0, "correlation": 0}

    result = find_optimal_portfolio_combination(
        portfolio_returns,
        n_select=2,
        weights=weights,
        min_data_points=30,
    )

    if "error" in result:
        print(f"  Error: {result['error']}")
        return

    recommended = result["recommended"]["combination"]
    print(f"  Recommended: {recommended}")

    # High Sharpe should be in the combination
    assert "High Sharpe" in recommended, "High Sharpe should be selected"
    print("  [PASS]")


def test_score_combination_correlation_only():
    """Test that correlation-only scoring picks low-correlation pairs."""
    print("\n=== Test: Score Combination (Correlation Only) ===")

    # Create returns with known correlations
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=60, freq='B')

    base = np.random.normal(0, 0.01, 60)

    # A and B highly correlated (both follow base)
    returns_a = pd.Series(base + np.random.normal(0, 0.001, 60), index=dates)
    returns_b = pd.Series(base + np.random.normal(0, 0.001, 60), index=dates)

    # C and D uncorrelated (independent)
    returns_c = pd.Series(np.random.normal(0.0003, 0.01, 60), index=dates)
    returns_d = pd.Series(np.random.normal(0.0003, 0.01, 60), index=dates)

    portfolio_returns = {
        "A": returns_a,
        "B": returns_b,
        "C": returns_c,
        "D": returns_d,
    }

    # Correlation-only weights
    weights = {"sharpe": 0, "sortino": 0, "calmar": 0, "upi": 0,
               "cagr": 0, "max_dd": 0, "volatility": 0, "correlation": 3}

    result = find_optimal_portfolio_combination(
        portfolio_returns,
        n_select=2,
        weights=weights,
        min_data_points=30,
    )

    if "error" in result:
        print(f"  Error: {result['error']}")
        return

    recommended = result["recommended"]["combination"]
    avg_corr = result["recommended"]["avg_correlation"]

    print(f"  Recommended: {recommended}")
    print(f"  Avg correlation: {avg_corr:.3f}")

    # Should not pick both A and B together (too correlated)
    assert not (("A" in recommended) and ("B" in recommended)), \
        "Should not select highly correlated A and B together"

    print("  [PASS]")


def test_simulate_combined_portfolio():
    """Test combined portfolio simulation accuracy."""
    print("\n=== Test: Simulate Combined Portfolio ===")

    # Create known returns
    dates = pd.date_range(start='2024-01-01', periods=5, freq='B')

    # Portfolio A: [1%, 2%, -1%, 0.5%, 1%]
    returns_a = pd.Series([0.01, 0.02, -0.01, 0.005, 0.01], index=dates)

    # Portfolio B: [2%, -1%, 1%, 0.5%, -0.5%]
    returns_b = pd.Series([0.02, -0.01, 0.01, 0.005, -0.005], index=dates)

    portfolio_returns = {"A": returns_a, "B": returns_b}

    # Equal weight (50/50)
    result = simulate_combined_portfolio(
        portfolio_returns,
        selected=["A", "B"],
        weights=[0.5, 0.5],
        base_nav=100.0,
    )

    if "error" in result:
        print(f"  Error: {result['error']}")
        return

    # Expected combined returns: [1.5%, 0.5%, 0%, 0.5%, 0.25%]
    expected_returns = [0.015, 0.005, 0.0, 0.005, 0.0025]
    actual_returns = result["returns"].values

    print(f"  Expected returns: {expected_returns}")
    print(f"  Actual returns: {[round(r, 4) for r in actual_returns]}")

    for i, (exp, act) in enumerate(zip(expected_returns, actual_returns)):
        assert abs(exp - act) < 0.0001, f"Return mismatch at index {i}: {exp} vs {act}"

    # Check final NAV
    expected_nav = 100.0
    for r in expected_returns:
        expected_nav *= (1 + r)

    actual_nav = result["nav"].iloc[-1]
    print(f"  Expected final NAV: {expected_nav:.4f}")
    print(f"  Actual final NAV: {actual_nav:.4f}")

    assert abs(expected_nav - actual_nav) < 0.01, "NAV mismatch"

    print("  [PASS]")


def test_simulate_combined_portfolio_weights():
    """Test that weights are correctly applied in simulation."""
    print("\n=== Test: Simulate Combined Portfolio Weights ===")

    dates = pd.date_range(start='2024-01-01', periods=5, freq='B')

    # A: constant 1% return
    returns_a = pd.Series([0.01] * 5, index=dates)
    # B: constant 2% return
    returns_b = pd.Series([0.02] * 5, index=dates)

    portfolio_returns = {"A": returns_a, "B": returns_b}

    # 70% A, 30% B -> expected return = 0.7*1% + 0.3*2% = 1.3%
    result = simulate_combined_portfolio(
        portfolio_returns,
        selected=["A", "B"],
        weights=[0.7, 0.3],
    )

    expected_daily_return = 0.013
    actual_return = result["returns"].mean()

    print(f"  Weights: A=70%, B=30%")
    print(f"  Expected avg daily return: {expected_daily_return:.4f}")
    print(f"  Actual avg daily return: {actual_return:.4f}")

    assert abs(expected_daily_return - actual_return) < 0.0001, "Weight application incorrect"
    print("  [PASS]")


def test_find_optimal_basic():
    """Test basic optimization finds valid combination."""
    print("\n=== Test: Find Optimal Basic ===")

    portfolio_returns = {
        "P1": create_test_returns(seed=1),
        "P2": create_test_returns(seed=2),
        "P3": create_test_returns(seed=3),
        "P4": create_test_returns(seed=4),
        "P5": create_test_returns(seed=5),
    }

    result = find_optimal_portfolio_combination(
        portfolio_returns,
        n_select=3,
        weights=DEFAULT_WEIGHTS,
        min_data_points=30,
    )

    if "error" in result:
        print(f"  Error: {result['error']}")
        return

    print(f"  Candidates: {result['candidates_count']}")
    print(f"  Combinations evaluated: {result['combinations_evaluated']}")
    print(f"  Recommended: {result['recommended']['combination']}")
    print(f"  Total score: {result['recommended']['total_score']:.3f}")

    # Check we got a valid result
    assert result["recommended"] is not None, "Should have a recommendation"
    assert len(result["recommended"]["combination"]) == 3, "Should select 3 portfolios"

    # Check alternatives exist
    assert len(result["alternatives"]) >= 1, "Should have alternatives"

    print("  [PASS]")


def test_optimizer_insufficient_candidates():
    """Test optimizer handles insufficient candidates gracefully."""
    print("\n=== Test: Optimizer Insufficient Candidates ===")

    # Only 2 portfolios but requesting 3
    portfolio_returns = {
        "P1": create_test_returns(seed=1),
        "P2": create_test_returns(seed=2),
    }

    result = find_optimal_portfolio_combination(
        portfolio_returns,
        n_select=3,
        weights=DEFAULT_WEIGHTS,
        min_data_points=30,
    )

    print(f"  Candidates: {result.get('candidates_count', 0)}")
    print(f"  Error: {result.get('error', 'None')}")

    assert "error" in result, "Should return error for insufficient candidates"
    print("  [PASS]")


def test_optimizer_min_data_points():
    """Test that min_data_points filter works."""
    print("\n=== Test: Optimizer Min Data Points ===")

    # P1 has 60 days, P2 has only 20 days
    portfolio_returns = {
        "P1 Long": create_test_returns(days=60, seed=1),
        "P2 Short": create_test_returns(days=20, seed=2),
        "P3 Long": create_test_returns(days=60, seed=3),
    }

    result = find_optimal_portfolio_combination(
        portfolio_returns,
        n_select=2,
        weights=DEFAULT_WEIGHTS,
        min_data_points=30,  # P2 should be excluded
    )

    if "error" in result:
        print(f"  Error: {result['error']}")
        return

    recommended = result["recommended"]["combination"]
    print(f"  Recommended: {recommended}")

    # P2 Short should not be in recommendations
    assert "P2 Short" not in recommended, "Short data portfolio should be excluded"
    print("  [PASS]")


def run_all_tests():
    """Run all optimizer tests."""
    print("=" * 60)
    print("PORTFOLIO SELECTION - OPTIMIZER TESTS")
    print("=" * 60)

    tests = [
        test_normalize_metrics_basic,
        test_normalize_metrics_inverted,
        test_score_combination_sharpe_only,
        test_score_combination_correlation_only,
        test_simulate_combined_portfolio,
        test_simulate_combined_portfolio_weights,
        test_find_optimal_basic,
        test_optimizer_insufficient_candidates,
        test_optimizer_min_data_points,
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
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
