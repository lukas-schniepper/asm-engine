"""
SQLModel definitions for Portfolio Tracking tables.

Tables:
- portfolio_definitions: Tracked portfolios with their configurations
- portfolio_holdings: Point-in-time position snapshots
- portfolio_daily_nav: Daily NAV for all variants (raw, conservative, trend_regime_v2, etc.)
- overlay_signals: Daily overlay recommendations and signal breakdowns
- portfolio_metrics: Pre-computed periodic performance metrics
"""

from sqlmodel import SQLModel, Field, Column
from sqlalchemy import BigInteger, JSON
from typing import Optional
from datetime import date, datetime
from decimal import Decimal


class PortfolioDefinition(SQLModel, table=True):
    """
    Tracked alpha portfolios with their backtest configurations.

    Each portfolio represents a specific alpha strategy with fixed parameters
    that we want to track over time.
    """
    __tablename__ = "portfolio_definitions"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(max_length=100, unique=True, index=True)
    description: Optional[str] = Field(default=None)

    # Full backtest configuration (num_stocks, window_days, optimizer, etc.)
    config: dict = Field(default={}, sa_column=Column(JSON))

    # Data source used (Topweights, TR20, etc.)
    source: Optional[str] = Field(default=None, max_length=50)

    # Tracking start date
    start_date: date

    # Whether this portfolio is actively being tracked
    is_active: bool = Field(default=True)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class PortfolioHolding(SQLModel, table=True):
    """
    Point-in-time position snapshots for each portfolio.

    Records the holdings at each rebalance date, allowing us to
    reconstruct the portfolio state at any point in time.
    """
    __tablename__ = "portfolio_holdings"

    id: Optional[int] = Field(default=None, primary_key=True)
    portfolio_id: int = Field(foreign_key="portfolio_definitions.id", index=True)

    # Date when these holdings became effective
    effective_date: date = Field(index=True)

    # Position details
    ticker: str = Field(max_length=20)
    shares: Optional[Decimal] = Field(default=None, max_digits=15, decimal_places=4)
    weight: Optional[Decimal] = Field(default=None, max_digits=8, decimal_places=6)
    entry_price: Optional[Decimal] = Field(default=None, max_digits=15, decimal_places=4)

    class Config:
        # Unique constraint on (portfolio_id, effective_date, ticker)
        pass


class PortfolioDailyNAV(SQLModel, table=True):
    """
    Daily NAV tracking for all portfolio variants.

    For each portfolio, we track:
    - 'raw': Pure alpha portfolio without any overlay
    - 'conservative': With Conservative Model (OCT16) overlay
    - 'trend_regime_v2': With TrendRegimeV2 overlay
    - (future overlays can be added without schema changes)
    """
    __tablename__ = "portfolio_daily_nav"

    id: Optional[int] = Field(default=None, primary_key=True)
    portfolio_id: int = Field(foreign_key="portfolio_definitions.id", index=True)
    trade_date: date = Field(index=True)

    # Variant: 'raw', 'conservative', 'trend_regime_v2', etc.
    variant: str = Field(max_length=30, index=True)

    # NAV value (starting from 100 or portfolio start value)
    nav: Decimal = Field(max_digits=15, decimal_places=4)

    # Daily return (as decimal, e.g., 0.0123 for +1.23%)
    daily_return: Optional[Decimal] = Field(default=None, max_digits=12, decimal_places=10)

    # Cumulative return since inception (as decimal)
    cumulative_return: Optional[Decimal] = Field(default=None, max_digits=12, decimal_places=10)

    # For overlay variants: current allocation percentages
    equity_allocation: Optional[Decimal] = Field(default=None, max_digits=8, decimal_places=6)
    cash_allocation: Optional[Decimal] = Field(default=None, max_digits=8, decimal_places=6)

    class Config:
        # Unique constraint on (portfolio_id, trade_date, variant)
        pass


class OverlaySignal(SQLModel, table=True):
    """
    Daily overlay signals and allocation recommendations.

    One row per model per trading day, containing:
    - Target vs actual allocation
    - Whether a trade was executed
    - Full signal breakdown (RSI, VIX, momentum, etc.)
    - Per-factor impact on allocation
    """
    __tablename__ = "overlay_signals"

    id: Optional[int] = Field(default=None, primary_key=True)
    trade_date: date = Field(index=True)

    # Model identifier: 'conservative', 'trend_regime_v2', etc.
    model: str = Field(max_length=30, index=True)

    # Allocation percentages (as decimals, e.g., 0.65 for 65%)
    target_allocation: Optional[Decimal] = Field(default=None, max_digits=8, decimal_places=6)
    actual_allocation: Optional[Decimal] = Field(default=None, max_digits=8, decimal_places=6)

    # Whether the allocation change triggered a trade
    trade_required: Optional[bool] = Field(default=None)

    # Full signal values (RSI, VIX, momentum, slope, etc.)
    signals: Optional[dict] = Field(default=None, sa_column=Column(JSON))

    # Per-factor allocation impacts
    impacts: Optional[dict] = Field(default=None, sa_column=Column(JSON))

    class Config:
        # Unique constraint on (trade_date, model)
        pass


class PortfolioMetric(SQLModel, table=True):
    """
    Pre-computed periodic performance metrics.

    Aggregated statistics for faster querying in the dashboard.
    Computed for various time periods: week, month, quarter, year, ytd, all.
    """
    __tablename__ = "portfolio_metrics"

    id: Optional[int] = Field(default=None, primary_key=True)
    portfolio_id: int = Field(foreign_key="portfolio_definitions.id", index=True)

    # Variant: 'raw', 'conservative', 'trend_regime_v2', etc.
    variant: str = Field(max_length=30)

    # Period type: 'week', 'month', 'quarter', 'year', 'ytd', 'all'
    period_type: str = Field(max_length=10)

    # Period boundaries
    period_start: date
    period_end: date

    # Performance metrics
    total_return: Optional[Decimal] = Field(default=None, max_digits=12, decimal_places=10)
    sharpe_ratio: Optional[Decimal] = Field(default=None, max_digits=10, decimal_places=6)
    sortino_ratio: Optional[Decimal] = Field(default=None, max_digits=10, decimal_places=6)
    cagr: Optional[Decimal] = Field(default=None, max_digits=12, decimal_places=10)
    max_drawdown: Optional[Decimal] = Field(default=None, max_digits=12, decimal_places=10)
    calmar_ratio: Optional[Decimal] = Field(default=None, max_digits=10, decimal_places=6)
    volatility: Optional[Decimal] = Field(default=None, max_digits=12, decimal_places=10)
    win_rate: Optional[Decimal] = Field(default=None, max_digits=8, decimal_places=6)

    class Config:
        # Unique constraint on (portfolio_id, variant, period_type, period_start)
        pass


# Constants for variant names
class Variants:
    """Standard variant names for consistency."""
    RAW = "raw"
    CONSERVATIVE = "conservative"
    TREND_REGIME_V2 = "trend_regime_v2"

    @classmethod
    def all(cls) -> list[str]:
        return [cls.RAW, cls.CONSERVATIVE, cls.TREND_REGIME_V2]


# Constants for period types
class PeriodTypes:
    """Standard period type names."""
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"
    YTD = "ytd"
    ALL = "all"

    @classmethod
    def all(cls) -> list[str]:
        return [cls.WEEK, cls.MONTH, cls.QUARTER, cls.YEAR, cls.YTD, cls.ALL]
