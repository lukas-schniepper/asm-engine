"""
Core Portfolio Tracking Engine.

Provides the main PortfolioTracker class that:
1. Registers portfolios for tracking
2. Calculates daily NAV for all variants (raw + overlays)
3. Records NAV and signals to database
4. Computes and stores periodic metrics

This is the main entry point for the tracking system.
"""

import logging
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Optional

import pandas as pd
from sqlmodel import Session, select

from ..db import engine
from .models import (
    PortfolioDefinition,
    PortfolioDailyNAV,
    PortfolioHolding,
    OverlaySignal,
    PortfolioMetric,
    Variants,
    PeriodTypes,
)
from .overlay_adapter import OverlayAdapter, OVERLAY_REGISTRY
from .metrics import (
    calculate_all_metrics,
    get_period_boundaries,
    calculate_returns,
)

logger = logging.getLogger(__name__)


def calculate_shares_from_weights(
    holdings_with_weights: list[dict],
    nav_value: Decimal,
    prices: dict[str, Decimal],
) -> list[dict]:
    """
    Convert weight-based holdings to shares-based holdings.

    At rebalance time, calculates how many shares each position should have
    based on its target weight and the current NAV/prices.

    Formula: shares = (weight × NAV) / price

    This is the industry-standard approach for buy-and-hold portfolios:
    - Set share counts at rebalance
    - Let position values drift with prices between rebalances

    Args:
        holdings_with_weights: List of dicts with 'ticker', 'weight'
            (weights should sum to 1.0)
        nav_value: Current portfolio NAV (e.g., 100.0 at inception)
        prices: Current prices for each ticker

    Returns:
        List of holding dicts with 'ticker', 'weight', 'shares', 'entry_price'

    Example:
        >>> holdings = [{"ticker": "AAPL", "weight": 0.5}, {"ticker": "MSFT", "weight": 0.5}]
        >>> prices = {"AAPL": Decimal("150.00"), "MSFT": Decimal("300.00")}
        >>> result = calculate_shares_from_weights(holdings, Decimal("100"), prices)
        >>> # AAPL: shares = (0.5 × 100) / 150 = 0.3333
        >>> # MSFT: shares = (0.5 × 100) / 300 = 0.1667
    """
    result = []

    for h in holdings_with_weights:
        ticker = h["ticker"]
        weight = Decimal(str(h.get("weight", 0)))
        price = prices.get(ticker)

        if price and price > 0 and weight > 0:
            # Position value = weight × NAV
            # Shares = position_value / price
            position_value = weight * nav_value
            shares = position_value / price

            result.append({
                "ticker": ticker,
                "weight": weight,
                "shares": shares,
                "entry_price": price,
            })
        else:
            # No valid price - keep weight only (backward compatible)
            logger.warning(
                f"Cannot calculate shares for {ticker}: "
                f"price={price}, weight={weight}"
            )
            result.append({
                "ticker": ticker,
                "weight": weight,
                "shares": None,
                "entry_price": price if price else None,
            })

    return result


def mark_to_market(
    holdings: list,
    prices: dict[str, float],
) -> float:
    """
    Calculate current market value of existing holdings.

    CRITICAL for NAV continuity during rebalances:
    - previous_raw_nav: Yesterday's NAV stored in database (STALE after market moves)
    - mark_to_market(): TODAY's value at TODAY's prices (CURRENT)

    When rebalancing, new shares must be sized using mark_to_market value,
    NOT the stale previous_raw_nav. Otherwise:
    - Day T-1: NAV=100 stored in DB
    - Day T: Market +2%, old holdings worth 102
    - BUG: Calculate new shares for NAV=100, not 102
    - RESULT: Phantom -2% NAV drop!

    Args:
        holdings: List with 'shares' and 'ticker' attributes/keys
                  (supports both PortfolioHolding objects and dicts)
        prices: Current prices {ticker: price}

    Returns:
        Total market value, or 0.0 if no valid holdings

    Example:
        >>> old_holdings = [{"ticker": "AAPL", "shares": 10}, {"ticker": "MSFT", "shares": 5}]
        >>> today_prices = {"AAPL": 153.0, "MSFT": 306.0}
        >>> mtm = mark_to_market(old_holdings, today_prices)
        >>> # mtm = 10*153 + 5*306 = 3060 (use this for new share sizing!)
    """
    if not holdings:
        return 0.0

    total_value = 0.0
    holdings_valued = 0

    for h in holdings:
        # Support both PortfolioHolding objects and dicts
        if hasattr(h, 'shares'):
            shares = h.shares
            ticker = h.ticker
        else:
            shares = h.get('shares')
            ticker = h.get('ticker')

        if shares is not None and ticker:
            price = prices.get(ticker)
            if price is not None and price > 0:
                total_value += float(shares) * float(price)
                holdings_valued += 1

    if holdings_valued > 0:
        logger.debug(
            f"mark_to_market: valued {holdings_valued} holdings, total={total_value:.2f}"
        )

    return total_value


class PortfolioTracker:
    """
    Main portfolio tracking engine.

    Example usage:
        tracker = PortfolioTracker()

        # Register a new portfolio
        portfolio = tracker.register_portfolio(
            name="TopWeights_20_MVO",
            config={"num_stocks": 20, "optimizer": "mvo"},
            source="Topweights",
            start_date=date(2024, 1, 1),
        )

        # Calculate and record daily NAV
        tracker.update_daily_nav(
            portfolio_id=portfolio.id,
            trade_date=date(2025, 11, 30),
            holdings_df=holdings_df,  # DataFrame with ticker, shares, weight
            prices_df=prices_df,  # DataFrame with ticker prices
        )

        # Get performance
        perf = tracker.get_portfolio_performance(
            portfolio_id=portfolio.id,
            variant="conservative",
            start_date=date(2024, 1, 1),
            end_date=date(2025, 11, 30),
        )
    """

    def __init__(self, overlay_adapter: Optional[OverlayAdapter] = None):
        """
        Initialize the tracker.

        Args:
            overlay_adapter: OverlayAdapter instance (creates default if not provided)
        """
        self.overlay_adapter = overlay_adapter or OverlayAdapter()

    # =========================================================================
    # Portfolio Registration
    # =========================================================================

    def register_portfolio(
        self,
        name: str,
        config: dict,
        source: str,
        start_date: date,
        description: Optional[str] = None,
    ) -> PortfolioDefinition:
        """
        Register a new portfolio for tracking.

        Args:
            name: Unique portfolio name
            config: Backtest configuration dict
            source: Data source (e.g., "Topweights", "TR20")
            start_date: Tracking start date
            description: Optional description

        Returns:
            Created PortfolioDefinition
        """
        with Session(engine) as session:
            # Check if portfolio already exists
            existing = session.exec(
                select(PortfolioDefinition).where(PortfolioDefinition.name == name)
            ).first()

            if existing:
                logger.info(f"Portfolio '{name}' already exists, returning existing")
                return existing

            portfolio = PortfolioDefinition(
                name=name,
                description=description,
                config=config,
                source=source,
                start_date=start_date,
                is_active=True,
            )
            session.add(portfolio)
            session.commit()
            session.refresh(portfolio)

            logger.info(f"Registered new portfolio: {name} (id={portfolio.id})")
            return portfolio

    def get_portfolio(self, portfolio_id: int) -> Optional[PortfolioDefinition]:
        """Get portfolio by ID."""
        with Session(engine) as session:
            return session.get(PortfolioDefinition, portfolio_id)

    def get_portfolio_by_name(self, name: str) -> Optional[PortfolioDefinition]:
        """Get portfolio by name."""
        with Session(engine) as session:
            return session.exec(
                select(PortfolioDefinition).where(PortfolioDefinition.name == name)
            ).first()

    def list_portfolios(self, active_only: bool = True) -> list[PortfolioDefinition]:
        """List all portfolios."""
        with Session(engine) as session:
            query = select(PortfolioDefinition)
            if active_only:
                query = query.where(PortfolioDefinition.is_active == True)
            return list(session.exec(query).all())

    # =========================================================================
    # Holdings Management
    # =========================================================================

    def record_holdings(
        self,
        portfolio_id: int,
        effective_date: date,
        holdings: list[dict],
    ) -> list[PortfolioHolding]:
        """
        Record portfolio holdings as of a specific date.

        Args:
            portfolio_id: Portfolio ID
            effective_date: Date holdings became effective
            holdings: List of dicts with keys: ticker, shares, weight, entry_price

        Returns:
            List of created PortfolioHolding records
        """
        with Session(engine) as session:
            # Delete existing holdings for this date (upsert)
            existing = session.exec(
                select(PortfolioHolding)
                .where(PortfolioHolding.portfolio_id == portfolio_id)
                .where(PortfolioHolding.effective_date == effective_date)
            ).all()
            for h in existing:
                session.delete(h)

            # Insert new holdings
            created = []
            for h in holdings:
                holding = PortfolioHolding(
                    portfolio_id=portfolio_id,
                    effective_date=effective_date,
                    ticker=h["ticker"],
                    shares=Decimal(str(h.get("shares", 0))) if h.get("shares") else None,
                    weight=Decimal(str(h.get("weight", 0))) if h.get("weight") else None,
                    entry_price=Decimal(str(h.get("entry_price", 0))) if h.get("entry_price") else None,
                )
                session.add(holding)
                created.append(holding)

            session.commit()
            logger.debug(f"Recorded {len(created)} holdings for portfolio {portfolio_id} on {effective_date}")
            return created

    def get_holdings(
        self,
        portfolio_id: int,
        as_of_date: date,
    ) -> list[PortfolioHolding]:
        """
        Get holdings as of a specific date.

        Returns the most recent holdings on or before the given date.
        """
        with Session(engine) as session:
            # Get most recent effective_date <= as_of_date
            subq = (
                select(PortfolioHolding.effective_date)
                .where(PortfolioHolding.portfolio_id == portfolio_id)
                .where(PortfolioHolding.effective_date <= as_of_date)
                .order_by(PortfolioHolding.effective_date.desc())
                .limit(1)
            )
            latest_date = session.exec(subq).first()

            if latest_date is None:
                return []

            return list(
                session.exec(
                    select(PortfolioHolding)
                    .where(PortfolioHolding.portfolio_id == portfolio_id)
                    .where(PortfolioHolding.effective_date == latest_date)
                ).all()
            )

    # =========================================================================
    # NAV Calculation and Recording
    # =========================================================================

    def calculate_raw_nav(
        self,
        holdings: list[PortfolioHolding],
        prices: dict[str, float],
        previous_nav: Optional[float] = None,
        previous_prices: Optional[dict[str, float]] = None,
    ) -> float:
        """
        Calculate raw NAV (100% equity) from holdings and prices.

        Args:
            holdings: List of PortfolioHolding
            prices: Dict mapping ticker to current price
            previous_nav: Previous day's NAV (used if holdings have no shares)
            previous_prices: Previous day's prices (for day-over-day return calculation)

        Returns:
            NAV value
        """
        if not holdings:
            return previous_nav or 100.0

        # If we have shares, calculate directly
        if holdings[0].shares is not None:
            nav = sum(
                float(h.shares) * prices.get(h.ticker, 0)
                for h in holdings
            )
            return nav if nav > 0 else (previous_nav or 100.0)

        # If we only have weights, use weight-based return calculation
        if previous_nav is None:
            return 100.0

        # Calculate weighted return
        total_return = 0.0
        holdings_with_weight = 0

        for h in holdings:
            if h.weight:
                ticker = h.ticker
                current_price = prices.get(ticker)

                # Get previous price: from previous_prices, entry_price, or current price
                if previous_prices and ticker in previous_prices:
                    prev_price = previous_prices[ticker]
                elif h.entry_price:
                    prev_price = float(h.entry_price)
                else:
                    prev_price = current_price  # No change if no baseline

                if current_price and prev_price and prev_price > 0:
                    position_return = (current_price / prev_price) - 1
                    total_return += float(h.weight) * position_return
                    holdings_with_weight += 1

        return previous_nav * (1 + total_return)

    def record_nav(
        self,
        portfolio_id: int,
        trade_date: date,
        variant: str,
        nav: float,
        daily_return: Optional[float] = None,
        cumulative_return: Optional[float] = None,
        equity_allocation: Optional[float] = None,
        cash_allocation: Optional[float] = None,
    ) -> PortfolioDailyNAV:
        """
        Record NAV for a specific date and variant.

        Uses upsert logic (updates if exists, inserts if new).
        """
        with Session(engine) as session:
            # Check for existing
            existing = session.exec(
                select(PortfolioDailyNAV)
                .where(PortfolioDailyNAV.portfolio_id == portfolio_id)
                .where(PortfolioDailyNAV.trade_date == trade_date)
                .where(PortfolioDailyNAV.variant == variant)
            ).first()

            if existing:
                existing.nav = Decimal(str(nav))
                existing.daily_return = Decimal(str(daily_return)) if daily_return is not None else None
                existing.cumulative_return = Decimal(str(cumulative_return)) if cumulative_return is not None else None
                existing.equity_allocation = Decimal(str(equity_allocation)) if equity_allocation is not None else None
                existing.cash_allocation = Decimal(str(cash_allocation)) if cash_allocation is not None else None
                session.add(existing)
                session.commit()
                session.refresh(existing)
                return existing

            nav_record = PortfolioDailyNAV(
                portfolio_id=portfolio_id,
                trade_date=trade_date,
                variant=variant,
                nav=Decimal(str(nav)),
                daily_return=Decimal(str(daily_return)) if daily_return is not None else None,
                cumulative_return=Decimal(str(cumulative_return)) if cumulative_return is not None else None,
                equity_allocation=Decimal(str(equity_allocation)) if equity_allocation is not None else None,
                cash_allocation=Decimal(str(cash_allocation)) if cash_allocation is not None else None,
            )
            session.add(nav_record)
            session.commit()
            session.refresh(nav_record)
            return nav_record

    def record_overlay_signal(
        self,
        trade_date: date,
        model: str,
        target_allocation: float,
        actual_allocation: float,
        trade_required: bool,
        signals: dict,
        impacts: dict,
    ) -> OverlaySignal:
        """Record overlay signal for a specific date and model."""
        with Session(engine) as session:
            # Check for existing
            existing = session.exec(
                select(OverlaySignal)
                .where(OverlaySignal.trade_date == trade_date)
                .where(OverlaySignal.model == model)
            ).first()

            if existing:
                existing.target_allocation = Decimal(str(target_allocation))
                existing.actual_allocation = Decimal(str(actual_allocation))
                existing.trade_required = trade_required
                existing.signals = signals
                existing.impacts = impacts
                session.add(existing)
                session.commit()
                return existing

            signal = OverlaySignal(
                trade_date=trade_date,
                model=model,
                target_allocation=Decimal(str(target_allocation)),
                actual_allocation=Decimal(str(actual_allocation)),
                trade_required=trade_required,
                signals=signals,
                impacts=impacts,
            )
            session.add(signal)
            session.commit()
            session.refresh(signal)
            return signal

    def update_daily_nav(
        self,
        portfolio_id: int,
        trade_date: date,
        raw_nav: float,
        previous_raw_nav: Optional[float] = None,
        initial_nav: float = 100.0,
        daily_return_override: Optional[float] = None,
    ) -> dict[str, dict]:
        """
        Update NAV for all variants (raw + overlays) for a specific date.

        Args:
            portfolio_id: Portfolio ID
            trade_date: Trading date
            raw_nav: Raw portfolio NAV (100% equity)
            previous_raw_nav: Previous day's raw NAV (for return calculation)
            initial_nav: Initial NAV at portfolio inception
            daily_return_override: Optional override for daily return (used on rebalance days
                to ensure GIPS-compliant return calculation using previous day's holdings)

        Returns:
            Dict mapping variant names to NAV data dicts with keys:
            - nav, daily_return, cumulative_return, equity_allocation, cash_allocation
        """
        results = {}

        # Calculate returns - use override if provided (for rebalance days per GIPS)
        if daily_return_override is not None:
            daily_return = daily_return_override
        elif previous_raw_nav and previous_raw_nav > 0:
            daily_return = (raw_nav / previous_raw_nav) - 1
        else:
            daily_return = 0.0

        cumulative_return = (raw_nav / initial_nav) - 1 if initial_nav else 0.0

        # Record raw NAV
        self.record_nav(
            portfolio_id=portfolio_id,
            trade_date=trade_date,
            variant=Variants.RAW,
            nav=raw_nav,
            daily_return=daily_return,
            cumulative_return=cumulative_return,
            equity_allocation=1.0,
            cash_allocation=0.0,
        )
        results[Variants.RAW] = {
            "nav": raw_nav,
            "daily_return": daily_return,
            "cumulative_return": cumulative_return,
            "equity_allocation": 1.0,
            "cash_allocation": 0.0,
        }

        # Calculate and record overlay variants
        for model_name in OVERLAY_REGISTRY.keys():
            try:
                # Get allocation for today from overlay model
                _, allocation, signals, impacts = self.overlay_adapter.apply_overlay(
                    model=model_name,
                    raw_nav=raw_nav,
                    trade_date=trade_date,
                )

                # Get previous overlay NAV from database
                prev_overlay_nav_df = self.get_nav_series(
                    portfolio_id, model_name,
                    end_date=trade_date - timedelta(days=1)  # Get NAV up to yesterday
                )

                if prev_overlay_nav_df.empty:
                    # First day - overlay NAV equals raw NAV (same starting point)
                    # This ensures all variants start at the same value for fair comparison
                    adjusted_nav = raw_nav
                    overlay_daily_return = 0.0
                else:
                    prev_overlay_nav = prev_overlay_nav_df["nav"].iloc[-1]
                    # Calculate overlay return: raw daily return scaled by allocation
                    overlay_daily_return = daily_return * allocation
                    # Calculate new overlay NAV: previous overlay NAV * (1 + overlay return)
                    adjusted_nav = prev_overlay_nav * (1 + overlay_daily_return)

                overlay_cumulative = (adjusted_nav / initial_nav) - 1 if initial_nav else 0.0
                self.record_nav(
                    portfolio_id=portfolio_id,
                    trade_date=trade_date,
                    variant=model_name,
                    nav=adjusted_nav,
                    daily_return=overlay_daily_return,
                    cumulative_return=overlay_cumulative,
                    equity_allocation=allocation,
                    cash_allocation=1 - allocation,
                )
                results[model_name] = {
                    "nav": adjusted_nav,
                    "daily_return": overlay_daily_return,
                    "cumulative_return": overlay_cumulative,
                    "equity_allocation": allocation,
                    "cash_allocation": 1 - allocation,
                }

                # Record overlay signal
                # Calculate if trade is required (allocation changed significantly)
                previous_signal = self._get_previous_signal(model_name, trade_date)
                if previous_signal:
                    prev_alloc = float(previous_signal.actual_allocation or 0)
                    trade_required = abs(allocation - prev_alloc) > 0.05  # 5% threshold
                else:
                    trade_required = True

                self.record_overlay_signal(
                    trade_date=trade_date,
                    model=model_name,
                    target_allocation=allocation,
                    actual_allocation=allocation,
                    trade_required=trade_required,
                    signals=signals,
                    impacts=impacts,
                )

            except Exception as e:
                logger.error(f"Error calculating overlay {model_name} for {trade_date}: {e}")
                continue

        return results

    def _get_previous_signal(self, model: str, trade_date: date) -> Optional[OverlaySignal]:
        """Get the previous day's signal for a model."""
        with Session(engine) as session:
            return session.exec(
                select(OverlaySignal)
                .where(OverlaySignal.model == model)
                .where(OverlaySignal.trade_date < trade_date)
                .order_by(OverlaySignal.trade_date.desc())
                .limit(1)
            ).first()

    # =========================================================================
    # Performance Retrieval
    # =========================================================================

    def get_nav_series(
        self,
        portfolio_id: int,
        variant: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Get NAV time series for a portfolio variant.

        Returns:
            DataFrame with columns: trade_date, nav, daily_return, cumulative_return,
            equity_allocation, cash_allocation
        """
        with Session(engine) as session:
            query = (
                select(PortfolioDailyNAV)
                .where(PortfolioDailyNAV.portfolio_id == portfolio_id)
                .where(PortfolioDailyNAV.variant == variant)
            )

            if start_date:
                query = query.where(PortfolioDailyNAV.trade_date >= start_date)
            if end_date:
                query = query.where(PortfolioDailyNAV.trade_date <= end_date)

            query = query.order_by(PortfolioDailyNAV.trade_date)
            records = session.exec(query).all()

            if not records:
                return pd.DataFrame(columns=[
                    "trade_date", "nav", "daily_return", "cumulative_return",
                    "equity_allocation", "cash_allocation"
                ])

            data = [
                {
                    "trade_date": r.trade_date,
                    "nav": float(r.nav),
                    "daily_return": float(r.daily_return) if r.daily_return else 0.0,
                    "cumulative_return": float(r.cumulative_return) if r.cumulative_return else 0.0,
                    "equity_allocation": float(r.equity_allocation) if r.equity_allocation else 1.0,
                    "cash_allocation": float(r.cash_allocation) if r.cash_allocation else 0.0,
                }
                for r in records
            ]

            df = pd.DataFrame(data)
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.set_index("trade_date")
            return df

    def get_portfolio_performance(
        self,
        portfolio_id: int,
        variant: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        risk_free_rate: float = 0.0,
    ) -> dict:
        """
        Get performance metrics for a portfolio variant.

        Returns:
            Dictionary with all performance metrics
        """
        nav_df = self.get_nav_series(portfolio_id, variant, start_date, end_date)

        if nav_df.empty:
            return {
                "portfolio_id": portfolio_id,
                "variant": variant,
                "start_date": start_date,
                "end_date": end_date,
                "total_return": 0.0,
                "cagr": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "calmar_ratio": 0.0,
                "volatility": 0.0,
                "win_rate": 0.0,
            }

        nav_series = nav_df["nav"]

        # Pass pre-calculated daily returns for GIPS-compliant total return
        # This ensures the first day's return is included in the calculation
        daily_returns = nav_df["daily_return"] if "daily_return" in nav_df.columns else None
        metrics = calculate_all_metrics(
            nav_series, risk_free_rate, start_date, end_date, daily_returns
        )

        return {
            "portfolio_id": portfolio_id,
            "variant": variant,
            "start_date": nav_df.index[0].date() if start_date is None else start_date,
            "end_date": nav_df.index[-1].date() if end_date is None else end_date,
            **metrics,
        }

    def compare_variants(
        self,
        portfolio_id: int,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        risk_free_rate: float = 0.0,
    ) -> pd.DataFrame:
        """
        Compare performance across all variants for a portfolio.

        Returns:
            DataFrame with variants as columns and metrics as rows
        """
        results = {}

        for variant in Variants.all():
            perf = self.get_portfolio_performance(
                portfolio_id, variant, start_date, end_date, risk_free_rate
            )
            results[variant] = {
                "total_return": perf["total_return"],
                "cagr": perf["cagr"],
                "sharpe_ratio": perf["sharpe_ratio"],
                "sortino_ratio": perf["sortino_ratio"],
                "max_drawdown": perf["max_drawdown"],
                "calmar_ratio": perf["calmar_ratio"],
                "volatility": perf["volatility"],
                "win_rate": perf["win_rate"],
            }

        return pd.DataFrame(results)

    # =========================================================================
    # Metrics Storage
    # =========================================================================

    def compute_and_store_metrics(
        self,
        portfolio_id: int,
        as_of_date: Optional[date] = None,
        risk_free_rate: float = 0.0,
    ) -> list[PortfolioMetric]:
        """
        Compute and store periodic metrics for a portfolio.

        Computes metrics for: week, month, quarter, year, ytd, all
        For all variants: raw, conservative, trend_regime_v2

        Returns:
            List of created/updated PortfolioMetric records
        """
        if as_of_date is None:
            as_of_date = date.today()

        period_boundaries = get_period_boundaries(as_of_date)
        results = []

        for variant in Variants.all():
            nav_df = self.get_nav_series(portfolio_id, variant)

            if nav_df.empty:
                continue

            # "all" period - full history
            full_start = nav_df.index[0].date()
            full_end = nav_df.index[-1].date()
            period_boundaries["all"] = (full_start, full_end)

            for period_type, (p_start, p_end) in period_boundaries.items():
                try:
                    perf = self.get_portfolio_performance(
                        portfolio_id, variant, p_start, p_end, risk_free_rate
                    )

                    metric = self._store_metric(
                        portfolio_id=portfolio_id,
                        variant=variant,
                        period_type=period_type,
                        period_start=p_start,
                        period_end=p_end,
                        metrics=perf,
                    )
                    results.append(metric)

                except Exception as e:
                    logger.error(
                        f"Error computing {period_type} metrics for "
                        f"portfolio {portfolio_id} variant {variant}: {e}"
                    )

        return results

    def _store_metric(
        self,
        portfolio_id: int,
        variant: str,
        period_type: str,
        period_start: date,
        period_end: date,
        metrics: dict,
    ) -> PortfolioMetric:
        """Store a single metric record (upsert)."""
        with Session(engine) as session:
            # Check for existing
            existing = session.exec(
                select(PortfolioMetric)
                .where(PortfolioMetric.portfolio_id == portfolio_id)
                .where(PortfolioMetric.variant == variant)
                .where(PortfolioMetric.period_type == period_type)
                .where(PortfolioMetric.period_start == period_start)
            ).first()

            def to_decimal(val, max_abs=9999.999999):
                """Convert to Decimal, capping at database column limit (10,6)."""
                if val is None:
                    return None
                # Cap to avoid numeric overflow (column is DECIMAL(10,6))
                val_float = float(val)
                if val_float > max_abs:
                    val_float = max_abs
                elif val_float < -max_abs:
                    val_float = -max_abs
                return Decimal(str(val_float))

            if existing:
                existing.period_end = period_end
                existing.total_return = to_decimal(metrics.get("total_return"))
                existing.sharpe_ratio = to_decimal(metrics.get("sharpe_ratio"))
                existing.sortino_ratio = to_decimal(metrics.get("sortino_ratio"))
                existing.cagr = to_decimal(metrics.get("cagr"))
                existing.max_drawdown = to_decimal(metrics.get("max_drawdown"))
                existing.calmar_ratio = to_decimal(metrics.get("calmar_ratio"))
                existing.volatility = to_decimal(metrics.get("volatility"))
                existing.win_rate = to_decimal(metrics.get("win_rate"))
                session.add(existing)
                session.commit()
                return existing

            metric = PortfolioMetric(
                portfolio_id=portfolio_id,
                variant=variant,
                period_type=period_type,
                period_start=period_start,
                period_end=period_end,
                total_return=to_decimal(metrics.get("total_return")),
                sharpe_ratio=to_decimal(metrics.get("sharpe_ratio")),
                sortino_ratio=to_decimal(metrics.get("sortino_ratio")),
                cagr=to_decimal(metrics.get("cagr")),
                max_drawdown=to_decimal(metrics.get("max_drawdown")),
                calmar_ratio=to_decimal(metrics.get("calmar_ratio")),
                volatility=to_decimal(metrics.get("volatility")),
                win_rate=to_decimal(metrics.get("win_rate")),
            )
            session.add(metric)
            session.commit()
            session.refresh(metric)
            return metric

    def get_stored_metrics(
        self,
        portfolio_id: int,
        variant: Optional[str] = None,
        period_type: Optional[str] = None,
    ) -> list[PortfolioMetric]:
        """Get pre-computed metrics from database."""
        with Session(engine) as session:
            query = select(PortfolioMetric).where(
                PortfolioMetric.portfolio_id == portfolio_id
            )

            if variant:
                query = query.where(PortfolioMetric.variant == variant)
            if period_type:
                query = query.where(PortfolioMetric.period_type == period_type)

            query = query.order_by(
                PortfolioMetric.variant,
                PortfolioMetric.period_type,
                PortfolioMetric.period_start.desc(),
            )

            return list(session.exec(query).all())

    # =========================================================================
    # Overlay Signals
    # =========================================================================

    def get_overlay_signals(
        self,
        model: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Get overlay signals for a model.

        Returns:
            DataFrame with signal history
        """
        with Session(engine) as session:
            query = select(OverlaySignal).where(OverlaySignal.model == model)

            if start_date:
                query = query.where(OverlaySignal.trade_date >= start_date)
            if end_date:
                query = query.where(OverlaySignal.trade_date <= end_date)

            query = query.order_by(OverlaySignal.trade_date)
            records = session.exec(query).all()

            if not records:
                return pd.DataFrame()

            data = []
            for r in records:
                row = {
                    "trade_date": r.trade_date,
                    "target_allocation": float(r.target_allocation) if r.target_allocation else None,
                    "actual_allocation": float(r.actual_allocation) if r.actual_allocation else None,
                    "trade_required": r.trade_required,
                }
                # Flatten signals and impacts
                if r.signals:
                    for k, v in r.signals.items():
                        row[f"signal_{k}"] = v
                if r.impacts:
                    for k, v in r.impacts.items():
                        row[f"impact_{k}"] = v
                data.append(row)

            df = pd.DataFrame(data)
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.set_index("trade_date")
            return df


# Convenience function
def get_tracker() -> PortfolioTracker:
    """Get a default PortfolioTracker instance."""
    return PortfolioTracker()
