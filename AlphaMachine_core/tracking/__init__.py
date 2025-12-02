"""
Portfolio Tracking Module

This module provides functionality for tracking portfolio performance
with and without risk overlay models.

Main components:
- PortfolioTracker: Core tracking engine for NAV calculation and storage
- OverlayAdapter: Applies risk overlay models (Conservative, TrendRegimeV2)
- S3DataLoader: Loads features and prices from asm-models S3 bucket
- Metrics: Performance metric calculations (Sharpe, Sortino, etc.)
"""

from .models import (
    PortfolioDefinition,
    PortfolioHolding,
    PortfolioDailyNAV,
    OverlaySignal,
    PortfolioMetric,
    Variants,
    PeriodTypes,
)
from .tracker import PortfolioTracker, get_tracker
from .overlay_adapter import OverlayAdapter, get_overlay_adapter, OVERLAY_REGISTRY
from .s3_adapter import S3DataLoader, get_s3_loader, load_features, load_spy
from .metrics import (
    calculate_all_metrics,
    calculate_returns,
    calculate_cumulative_returns,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_drawdown_series,
    calculate_cagr,
    calculate_calmar_ratio,
    calculate_volatility,
    calculate_win_rate,
    get_period_boundaries,
    compare_variants,
)
from .registration import (
    register_portfolio_from_backtest,
    get_suggested_portfolio_name,
    check_portfolio_name_exists,
)

__all__ = [
    # Models
    "PortfolioDefinition",
    "PortfolioHolding",
    "PortfolioDailyNAV",
    "OverlaySignal",
    "PortfolioMetric",
    "Variants",
    "PeriodTypes",
    # Tracker
    "PortfolioTracker",
    "get_tracker",
    # Overlay
    "OverlayAdapter",
    "get_overlay_adapter",
    "OVERLAY_REGISTRY",
    # S3
    "S3DataLoader",
    "get_s3_loader",
    "load_features",
    "load_spy",
    # Metrics
    "calculate_all_metrics",
    "calculate_returns",
    "calculate_cumulative_returns",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_max_drawdown",
    "calculate_drawdown_series",
    "calculate_cagr",
    "calculate_calmar_ratio",
    "calculate_volatility",
    "calculate_win_rate",
    "get_period_boundaries",
    "compare_variants",
    # Registration
    "register_portfolio_from_backtest",
    "get_suggested_portfolio_name",
    "check_portfolio_name_exists",
]
