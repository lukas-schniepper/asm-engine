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
    CONSERVATIVE_V2 = "conservative_v2"
    TREND_REGIME_V2_ASYM = "trend_regime_v2_asym"
    HB1 = "hb1"
    # New blends added 2026-04-25 (asm-models PR #8 + #9). 2026-04-26: B_AVG and
    # A_MUMD switched from sub-model TARGETS to ACTUALS (asm-models v3.0) for
    # apples-to-apples turnover. Variant slot a_max_up_min_down now implements
    # C_DISAGREE_HOLD (consensus-or-follow) per sr-quant-dev review.
    RB1 = "rb1"                                    # RegimeBlend (CV1A+TV2A on VIX z-score)
    B_AVERAGE = "b_average"                        # (TV1_actual + TV2A_actual) / 2
    A_MAX_UP_MIN_DOWN = "a_max_up_min_down"        # MAX_OF_ACTUALS on TV1+TV2A (since 2026-05-02)
                                                   # Display name: "Max of Trend"
    # CDH comparison variants added 2026-05-03. Each tracks an alternative
    # blending rule alongside production a_max_up_min_down so the operator
    # can compare 4 rules' alpha-portfolio NAVs going forward.
    C_DH_DIRECTIONAL      = "c_dh_directional"      # Max-Up / Min-Down (directional, up wins on conflict)
    C_DH_CONSENSUS_FOLLOW = "c_dh_consensus_follow" # Old C_DISAGREE_HOLD rule restored for comparison
    C_DH_AGREE_15PP       = "c_dh_agree_15pp"       # take max when |TV1-TV2A| <= 15pp; else hold prev

    @classmethod
    def all(cls) -> list[str]:
        return [cls.RAW, cls.CONSERVATIVE, cls.TREND_REGIME_V2,
                cls.CONSERVATIVE_V2, cls.TREND_REGIME_V2_ASYM, cls.HB1,
                cls.RB1, cls.B_AVERAGE, cls.A_MAX_UP_MIN_DOWN,
                cls.C_DH_DIRECTIONAL, cls.C_DH_CONSENSUS_FOLLOW, cls.C_DH_AGREE_15PP]


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
