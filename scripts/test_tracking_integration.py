#!/usr/bin/env python3
"""
Integration test for Portfolio Tracking System.

Tests the full flow without requiring database or S3 connections.
"""

import sys
from pathlib import Path
from datetime import date, timedelta
from decimal import Decimal

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np


def test_models():
    """Test SQLModel definitions."""
    print("\n=== Testing Models ===")

    # Import directly from models module to avoid tracker -> db -> config -> streamlit chain
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "models",
        project_root / "AlphaMachine_core" / "tracking" / "models.py"
    )
    models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(models)

    PortfolioDefinition = models.PortfolioDefinition
    PortfolioHolding = models.PortfolioHolding
    PortfolioDailyNAV = models.PortfolioDailyNAV
    OverlaySignal = models.OverlaySignal
    PortfolioMetric = models.PortfolioMetric
    Variants = models.Variants
    PeriodTypes = models.PeriodTypes

    # Test model instantiation
    portfolio = PortfolioDefinition(
        name="Test Portfolio",
        config={"num_stocks": 20},
        source="Test",
        start_date=date.today(),
    )
    print(f"  PortfolioDefinition: {portfolio.name}")

    holding = PortfolioHolding(
        portfolio_id=1,
        effective_date=date.today(),
        ticker="AAPL",
        weight=Decimal("0.05"),
    )
    print(f"  PortfolioHolding: {holding.ticker} @ {holding.weight}")

    nav = PortfolioDailyNAV(
        portfolio_id=1,
        trade_date=date.today(),
        variant="raw",
        nav=Decimal("100.00"),
    )
    print(f"  PortfolioDailyNAV: {nav.variant} = {nav.nav}")

    signal = OverlaySignal(
        trade_date=date.today(),
        model="conservative",
        target_allocation=Decimal("0.65"),
        signals={"rsi_14": 45.0},
    )
    print(f"  OverlaySignal: {signal.model} target={signal.target_allocation}")

    metric = PortfolioMetric(
        portfolio_id=1,
        variant="raw",
        period_type="month",
        period_start=date.today() - timedelta(days=30),
        period_end=date.today(),
        total_return=Decimal("0.0523"),
    )
    print(f"  PortfolioMetric: {metric.period_type} return={metric.total_return}")

    # Test constants
    print(f"  Variants.all(): {Variants.all()}")
    print(f"  PeriodTypes.all(): {PeriodTypes.all()}")

    print("  Models: OK")


def test_metrics():
    """Test metrics calculations."""
    print("\n=== Testing Metrics ===")

    # Import directly
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "metrics",
        project_root / "AlphaMachine_core" / "tracking" / "metrics.py"
    )
    metrics_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics_mod)

    calculate_all_metrics = metrics_mod.calculate_all_metrics
    calculate_returns = metrics_mod.calculate_returns
    calculate_sharpe_ratio = metrics_mod.calculate_sharpe_ratio
    calculate_max_drawdown = metrics_mod.calculate_max_drawdown
    calculate_drawdown_series = metrics_mod.calculate_drawdown_series
    get_period_boundaries = metrics_mod.get_period_boundaries

    # Create test NAV series
    dates = pd.date_range('2024-01-01', periods=252, freq='B')
    nav = pd.Series(100 * np.cumprod(1 + np.random.normal(0.0003, 0.01, 252)), index=dates)

    returns = calculate_returns(nav)
    print(f"  Returns: {len(returns)} data points")

    sharpe = calculate_sharpe_ratio(returns)
    print(f"  Sharpe: {sharpe:.2f}")

    max_dd = calculate_max_drawdown(nav)
    print(f"  Max Drawdown: {max_dd:.2%}")

    dd_series = calculate_drawdown_series(nav)
    print(f"  Drawdown Series: {len(dd_series)} data points")

    periods = get_period_boundaries(date.today())
    print(f"  Period Boundaries: {list(periods.keys())}")

    all_metrics = calculate_all_metrics(nav)
    print(f"  All Metrics: {list(all_metrics.keys())}")

    print("  Metrics: OK")


def test_overlay_calculations():
    """Test overlay allocation calculations."""
    print("\n=== Testing Overlay Calculations ===")

    # Read and modify overlay_adapter to remove relative import for testing
    with open(project_root / "AlphaMachine_core" / "tracking" / "overlay_adapter.py") as f:
        code = f.read()

    # Replace relative import with mock
    code = code.replace(
        'from .s3_adapter import S3DataLoader, get_s3_loader',
        '''
class S3DataLoader:
    def __init__(self, *args, **kwargs): pass
    def load_features_latest(self): return pd.DataFrame()
    def load_spy_prices(self): return pd.DataFrame()
    def load_model_config(self, model): return {}
def get_s3_loader(): return S3DataLoader()
'''
    )

    # Execute modified code
    exec(code, globals())

    # Now we have access to the functions in global namespace
    # Test RSI
    prices = pd.Series(np.random.randn(100).cumsum() + 100)
    rsi = calculate_rsi(prices)  # noqa: F821
    print(f"  RSI: range {rsi.min():.1f} - {rsi.max():.1f}")

    # Test registry
    print(f"  Overlay Registry: {list(OVERLAY_REGISTRY.keys())}")
    print(f"  Conservative display: {OVERLAY_REGISTRY['conservative'].display_name}")

    # Test default params
    print(f"  Default params: base_allocation={DEFAULT_PARAMS['conservative']['base_allocation']}")

    # Test allocation calculation
    dates = pd.date_range('2024-01-01', periods=252, freq='B')
    spy_prices = pd.Series(np.random.randn(252).cumsum() + 500, index=dates)

    # Create minimal enhanced features
    enhanced = pd.DataFrame(index=dates)
    enhanced["momentum_strength"] = np.random.randn(252).cumsum() * 0.01
    enhanced["momentum_consistent"] = (enhanced["momentum_strength"] > 0).astype(float)
    enhanced["low_vol_regime"] = 0.0
    enhanced["high_vol_regime"] = 0.0
    enhanced["low_stress"] = 0.0
    enhanced["extreme_stress"] = 0.0
    enhanced["high_correlation"] = 0.0
    enhanced["slope"] = 1.5
    enhanced["rsi_14"] = 50.0
    enhanced["rsi_21"] = 50.0
    enhanced["drawdown"] = -0.02
    enhanced["in_drawdown"] = 0.0
    enhanced["severe_drawdown"] = 0.0

    alloc, signals, impacts = calculate_allocation_conservative(
        trade_date=date(2024, 6, 15),
        params=DEFAULT_PARAMS["conservative"],
        enhanced_features=enhanced,
    )

    print(f"  Conservative allocation: {alloc:.2%}")
    print(f"  Signals: {list(signals.keys())[:5]}...")
    print(f"  Impacts: {list(impacts.keys())}")

    print("  Overlay Calculations: OK")


def test_ui_components():
    """Test UI components."""
    print("\n=== Testing UI Components ===")

    import importlib.util

    # Load styles
    spec = importlib.util.spec_from_file_location(
        "styles",
        project_root / "AlphaMachine_core" / "ui" / "styles.py"
    )
    styles = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(styles)

    get_dashboard_css = styles.get_dashboard_css
    COLORS = styles.COLORS
    format_percentage = styles.format_percentage
    format_number = styles.format_number
    get_variant_display_name = styles.get_variant_display_name

    css = get_dashboard_css()
    print(f"  Dashboard CSS: {len(css)} chars")

    print(f"  Colors: {len(COLORS)} defined")
    print(f"  format_percentage(0.1523): {format_percentage(0.1523)}")
    print(f"  format_number(12345.67): {format_number(12345.67)}")
    print(f"  get_variant_display_name('conservative'): {get_variant_display_name('conservative')}")

    # Load components by reading and modifying the code
    with open(project_root / "AlphaMachine_core" / "ui" / "components.py") as f:
        comp_code = f.read()

    # Replace relative import
    comp_code = comp_code.replace(
        '''from .styles import (
    COLORS,
    VARIANT_COLORS,
    format_percentage,
    format_number,
    format_ratio,
    get_value_class,
    get_variant_display_name,
    get_period_display_name,
)''',
        '# styles already imported into globals'
    )

    # We need to add the styles functions to the exec namespace
    exec_globals = globals().copy()
    exec_globals.update({
        'COLORS': COLORS,
        'VARIANT_COLORS': styles.VARIANT_COLORS,
        'format_percentage': format_percentage,
        'format_number': format_number,
        'format_ratio': styles.format_ratio,
        'get_value_class': styles.get_value_class,
        'get_variant_display_name': get_variant_display_name,
        'get_period_display_name': styles.get_period_display_name,
    })
    exec(comp_code, exec_globals)

    render_kpi_card = exec_globals['render_kpi_card']
    render_kpi_grid = exec_globals['render_kpi_grid']
    format_metrics_for_display = exec_globals['format_metrics_for_display']

    kpi_html = render_kpi_card("Return", "15.23%", "YTD", "positive")
    print(f"  KPI Card HTML: {len(kpi_html)} chars")

    metrics = {
        "total_return": 0.15,
        "sharpe_ratio": 1.2,
        "max_drawdown": -0.08,
    }
    kpis = format_metrics_for_display(metrics)
    print(f"  Formatted KPIs: {len(kpis)} items")

    # Load charts by reading and modifying the code
    with open(project_root / "AlphaMachine_core" / "ui" / "charts.py") as f:
        charts_code = f.read()

    # Replace relative import
    charts_code = charts_code.replace(
        'from .styles import COLORS, VARIANT_COLORS, get_variant_display_name',
        '# styles already imported into globals'
    )

    charts_globals = globals().copy()
    charts_globals.update({
        'COLORS': COLORS,
        'VARIANT_COLORS': styles.VARIANT_COLORS,
        'get_variant_display_name': get_variant_display_name,
    })
    exec(charts_code, charts_globals)

    create_nav_chart = charts_globals['create_nav_chart']
    create_drawdown_chart = charts_globals['create_drawdown_chart']
    create_allocation_chart = charts_globals['create_allocation_chart']

    dates = pd.date_range('2024-01-01', periods=100, freq='B')
    nav_data = {
        'raw': pd.Series(100 * np.cumprod(1 + np.random.normal(0.0003, 0.01, 100)), index=dates),
    }

    fig = create_nav_chart(nav_data, "Test Chart")
    print(f"  NAV Chart: {type(fig).__name__}")

    dd_data = {'raw': pd.Series(np.random.uniform(-0.1, 0, 100), index=dates)}
    fig2 = create_drawdown_chart(dd_data)
    print(f"  Drawdown Chart: {type(fig2).__name__}")

    alloc = pd.Series(np.random.uniform(0.5, 1.0, 100), index=dates)
    fig3 = create_allocation_chart(alloc)
    print(f"  Allocation Chart: {type(fig3).__name__}")

    print("  UI Components: OK")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Portfolio Tracking Integration Tests")
    print("=" * 60)

    try:
        test_models()
        test_metrics()
        test_overlay_calculations()
        test_ui_components()

        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
