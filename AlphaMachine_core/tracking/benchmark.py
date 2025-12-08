"""
Equal-Weight Benchmark Calculator.

Calculates equal-weight benchmark returns for a given source/universe
to compare against optimized portfolio performance.
"""

import logging
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import func, text
from sqlmodel import Session, select

from ..db import engine
from ..models import TickerPeriod, PriceData
from .metrics import calculate_all_metrics

logger = logging.getLogger(__name__)


def get_universe_tickers_for_date(
    source: str,
    trade_date: date,
    session: Session,
) -> list[str]:
    """
    Get all tickers in the universe for a specific source and date.

    Args:
        source: Data source (e.g., "Topweights")
        trade_date: Date to check
        session: Database session

    Returns:
        List of ticker symbols
    """
    # Find tickers where trade_date falls within their period
    query = (
        select(TickerPeriod.ticker)
        .where(TickerPeriod.source == source)
        .where(TickerPeriod.start_date <= trade_date)
        .where(TickerPeriod.end_date >= trade_date)
        .distinct()
    )
    return list(session.exec(query).all())


def calculate_ew_benchmark_returns(
    source: str,
    start_date: date,
    end_date: date,
    use_adjusted_close: bool = True,
) -> pd.DataFrame:
    """
    Calculate equal-weight benchmark returns for a given source.

    The benchmark is calculated as:
    - For each trading day, get all tickers in the universe
    - Calculate daily return for each ticker
    - Average the returns (equal-weight)

    Args:
        source: Data source (e.g., "Topweights", "TR20")
        start_date: Start date
        end_date: End date
        use_adjusted_close: Whether to use adjusted close prices (default True)

    Returns:
        DataFrame with columns: trade_date, daily_return, nav
    """
    with Session(engine) as session:
        # Get all ticker periods for this source
        ticker_periods = session.exec(
            select(TickerPeriod)
            .where(TickerPeriod.source == source)
            .where(TickerPeriod.start_date <= end_date)
            .where(TickerPeriod.end_date >= start_date)
        ).all()

        if not ticker_periods:
            logger.warning(f"No ticker periods found for source '{source}'")
            return pd.DataFrame()

        # Build a mapping of date -> active tickers
        all_tickers = set()
        for tp in ticker_periods:
            all_tickers.add(tp.ticker)

        logger.info(f"Found {len(all_tickers)} unique tickers for source '{source}'")

        # Fetch price data for all tickers
        price_col = "adjusted_close" if use_adjusted_close else "close"

        # Calculate a lookback date to get previous day's price for first day's return
        # We need to fetch data from before start_date to calculate pct_change correctly
        from datetime import timedelta
        lookback_date = start_date - timedelta(days=10)  # Go back 10 days to handle weekends/holidays

        # Use raw SQL for better performance
        query = text(f"""
            SELECT ticker, date as trade_date, close, adjusted_close
            FROM price_data
            WHERE ticker = ANY(:tickers)
            AND date >= :lookback_date
            AND date <= :end_date
            ORDER BY date, ticker
        """)

        result = session.execute(
            query,
            {
                "tickers": list(all_tickers),
                "lookback_date": lookback_date,
                "end_date": end_date,
            }
        )

        rows = result.fetchall()
        if not rows:
            logger.warning("No price data found for the specified period")
            return pd.DataFrame()

        # Convert to DataFrame
        price_df = pd.DataFrame(rows, columns=["ticker", "trade_date", "close", "adjusted_close"])
        price_df["trade_date"] = pd.to_datetime(price_df["trade_date"])

        # Use adjusted_close if available, otherwise fall back to close
        if use_adjusted_close and price_df["adjusted_close"].notna().any():
            price_df["price"] = price_df["adjusted_close"].fillna(price_df["close"])
        else:
            price_df["price"] = price_df["close"]

        # Pivot to get tickers as columns, dates as index
        price_pivot = price_df.pivot(index="trade_date", columns="ticker", values="price")

        # Build a mask of which tickers are active on each date
        # based on ticker_period start/end dates
        active_mask = pd.DataFrame(
            False,
            index=price_pivot.index,
            columns=price_pivot.columns
        )

        for tp in ticker_periods:
            if tp.ticker in active_mask.columns:
                mask = (active_mask.index >= pd.Timestamp(tp.start_date)) & \
                       (active_mask.index <= pd.Timestamp(tp.end_date))
                active_mask.loc[mask, tp.ticker] = True

        # Calculate daily returns
        returns = price_pivot.pct_change()

        # Apply active mask - only include returns for active tickers
        masked_returns = returns.where(active_mask)

        # Calculate equal-weight average return for each day
        # (mean of active tickers)
        ew_returns = masked_returns.mean(axis=1)

        # Filter to only include dates >= start_date (lookback was only for pct_change)
        ew_returns = ew_returns[ew_returns.index >= pd.Timestamp(start_date)]
        active_mask = active_mask[active_mask.index >= pd.Timestamp(start_date)]

        # Calculate NAV series starting at 100
        nav = (1 + ew_returns.fillna(0)).cumprod() * 100

        # Build result DataFrame
        result_df = pd.DataFrame({
            "trade_date": ew_returns.index,
            "daily_return": ew_returns.values,
            "nav": nav.values,
        })
        result_df = result_df.dropna(subset=["daily_return"])
        result_df = result_df.set_index("trade_date")

        # Count active tickers per day for reference
        active_counts = active_mask.sum(axis=1)
        result_df["active_tickers"] = active_counts

        logger.info(
            f"Calculated EW benchmark: {len(result_df)} trading days, "
            f"avg {result_df['active_tickers'].mean():.1f} active tickers/day"
        )

        return result_df


def get_benchmark_metrics(
    source: str,
    start_date: date,
    end_date: date,
    use_adjusted_close: bool = True,
) -> dict:
    """
    Calculate all metrics for the EW benchmark.

    Args:
        source: Data source
        start_date: Start date
        end_date: End date
        use_adjusted_close: Whether to use adjusted close prices

    Returns:
        Dictionary with all metrics
    """
    benchmark_df = calculate_ew_benchmark_returns(
        source, start_date, end_date, use_adjusted_close
    )

    if benchmark_df.empty:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "volatility": 0.0,
            "win_rate": 0.0,
        }

    nav_series = benchmark_df["nav"]
    return calculate_all_metrics(nav_series, risk_free_rate=0.0)


def get_benchmark_monthly_returns(
    source: str,
    start_date: date,
    end_date: date,
    use_adjusted_close: bool = True,
) -> dict[str, float]:
    """
    Calculate monthly returns for the EW benchmark.

    Args:
        source: Data source
        start_date: Start date
        end_date: End date
        use_adjusted_close: Whether to use adjusted close prices

    Returns:
        Dictionary mapping month (YYYY-MM) to return
    """
    benchmark_df = calculate_ew_benchmark_returns(
        source, start_date, end_date, use_adjusted_close
    )

    if benchmark_df.empty:
        return {}

    # Group by month and compound returns
    monthly_returns = {}
    for month, group in benchmark_df.groupby(benchmark_df.index.strftime("%Y-%m")):
        daily_rets = group["daily_return"].dropna()
        if len(daily_rets) > 0:
            monthly_total = (1 + daily_rets).prod() - 1
            monthly_returns[month] = float(monthly_total)

    return monthly_returns


def compare_portfolio_to_benchmark(
    portfolio_nav_df: pd.DataFrame,
    source: str,
    start_date: date,
    end_date: date,
    use_adjusted_close: bool = True,
) -> dict:
    """
    Compare portfolio performance to EW benchmark.

    Args:
        portfolio_nav_df: Portfolio NAV DataFrame (with 'nav' and 'daily_return' columns)
        source: Benchmark source
        start_date: Start date
        end_date: End date
        use_adjusted_close: Whether to use adjusted close prices

    Returns:
        Dictionary with comparison data:
        - portfolio_metrics: Dict of portfolio metrics
        - benchmark_metrics: Dict of benchmark metrics
        - monthly_comparison: List of monthly comparison records
        - portfolio_nav: Series of portfolio NAV
        - benchmark_nav: Series of benchmark NAV
    """
    # Get portfolio metrics
    if "nav" not in portfolio_nav_df.columns:
        raise ValueError("portfolio_nav_df must have 'nav' column")

    portfolio_nav = portfolio_nav_df["nav"]
    portfolio_metrics = calculate_all_metrics(portfolio_nav)

    # Get portfolio monthly returns
    portfolio_monthly = {}
    if "daily_return" in portfolio_nav_df.columns:
        for month, group in portfolio_nav_df.groupby(portfolio_nav_df.index.strftime("%Y-%m")):
            daily_rets = group["daily_return"].dropna()
            if len(daily_rets) > 0:
                portfolio_monthly[month] = float((1 + daily_rets).prod() - 1)

    # Get benchmark data
    benchmark_df = calculate_ew_benchmark_returns(
        source, start_date, end_date, use_adjusted_close
    )

    if benchmark_df.empty:
        return {
            "portfolio_metrics": portfolio_metrics,
            "benchmark_metrics": {},
            "monthly_comparison": [],
            "portfolio_nav": portfolio_nav,
            "benchmark_nav": pd.Series(dtype=float),
        }

    benchmark_nav = benchmark_df["nav"]
    benchmark_metrics = calculate_all_metrics(benchmark_nav)

    # Get benchmark monthly returns
    benchmark_monthly = {}
    for month, group in benchmark_df.groupby(benchmark_df.index.strftime("%Y-%m")):
        daily_rets = group["daily_return"].dropna()
        if len(daily_rets) > 0:
            benchmark_monthly[month] = float((1 + daily_rets).prod() - 1)

    # Build monthly comparison
    all_months = sorted(set(portfolio_monthly.keys()) | set(benchmark_monthly.keys()))
    monthly_comparison = []
    for month in all_months:
        port_ret = portfolio_monthly.get(month)
        bench_ret = benchmark_monthly.get(month)
        diff = None
        if port_ret is not None and bench_ret is not None:
            diff = port_ret - bench_ret

        monthly_comparison.append({
            "month": month,
            "portfolio_return": port_ret,
            "benchmark_return": bench_ret,
            "difference": diff,
        })

    return {
        "portfolio_metrics": portfolio_metrics,
        "benchmark_metrics": benchmark_metrics,
        "monthly_comparison": monthly_comparison,
        "portfolio_nav": portfolio_nav,
        "benchmark_nav": benchmark_nav,
    }


def calculate_stock_attribution(
    portfolio_id: int,
    source: str,
    month: str,
    tracker: "PortfolioTracker",
) -> dict:
    """
    Calculate stock-level attribution for a specific month.

    Shows weight, return, and contribution (weight × return) for each stock
    in both the portfolio and the benchmark universe.

    Uses simple buy-and-hold methodology:
    - Portfolio: weight at month start × (end_price / start_price - 1)
    - Benchmark: 1/N equal weight × (end_price / start_price - 1)

    Args:
        portfolio_id: Portfolio ID to analyze
        source: Data source for benchmark universe (e.g., "Topweights")
        month: Month in format "YYYY-MM"
        tracker: PortfolioTracker instance

    Returns:
        Dictionary with:
        - portfolio_holdings: List of {ticker, weight, return, contribution}
        - universe_holdings: List of {ticker, weight, return, contribution}
        - portfolio_total: Total portfolio return for the month
        - benchmark_total: Total benchmark return for the month
        - alpha: Portfolio return - benchmark return
    """
    from datetime import timedelta
    from calendar import monthrange

    # Parse month to get date range
    year, mon = int(month[:4]), int(month[5:7])
    month_start = date(year, mon, 1)
    _, last_day = monthrange(year, mon)
    month_end = date(year, mon, last_day)

    # Cap end date at today if month is current/future
    today = date.today()
    if month_end > today:
        month_end = today

    with Session(engine) as session:
        # Get portfolio holdings at start of month
        holdings = tracker.get_holdings(portfolio_id, month_start)

        # Get universe tickers
        universe_tickers = get_universe_tickers_for_date(source, month_start, session)

        if not universe_tickers:
            logger.warning(f"No universe tickers found for source '{source}' on {month_start}")
            return {
                "portfolio_holdings": [],
                "universe_holdings": [],
                "portfolio_total": 0.0,
                "benchmark_total": 0.0,
                "alpha": 0.0,
            }

        # Collect all tickers we need prices for
        portfolio_tickers = [h.ticker for h in holdings] if holdings else []
        all_tickers = list(set(portfolio_tickers + universe_tickers))

        # Get price data for start and end of month
        query = text("""
            WITH month_prices AS (
                SELECT
                    ticker,
                    date,
                    COALESCE(adjusted_close, close) as price,
                    ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date ASC) as rn_first,
                    ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) as rn_last
                FROM price_data
                WHERE ticker = ANY(:tickers)
                AND date >= :month_start
                AND date <= :month_end
            )
            SELECT
                ticker,
                MAX(CASE WHEN rn_first = 1 THEN price END) as start_price,
                MAX(CASE WHEN rn_last = 1 THEN price END) as end_price
            FROM month_prices
            GROUP BY ticker
        """)

        result = session.execute(
            query,
            {
                "tickers": all_tickers,
                "month_start": month_start,
                "month_end": month_end,
            }
        )

        # Build price lookup: ticker -> (start_price, end_price)
        price_data = {}
        for row in result.fetchall():
            ticker, start_price, end_price = row
            if start_price and end_price and start_price > 0:
                price_data[ticker] = {
                    "start_price": float(start_price),
                    "end_price": float(end_price),
                    "return": (float(end_price) / float(start_price)) - 1,
                }

        # Calculate portfolio holdings attribution
        portfolio_holdings = []
        portfolio_total = 0.0

        if holdings:
            # Normalize weights to sum to 1 (in case they don't)
            total_weight = sum(float(h.weight) for h in holdings if h.weight)

            for h in holdings:
                if not h.weight:
                    continue

                weight = float(h.weight) / total_weight if total_weight > 0 else 0
                ticker_data = price_data.get(h.ticker)

                if ticker_data:
                    ret = ticker_data["return"]
                    contribution = weight * ret
                    portfolio_total += contribution

                    portfolio_holdings.append({
                        "ticker": h.ticker,
                        "weight": weight,
                        "return": ret,
                        "contribution": contribution,
                    })
                else:
                    # No price data - still show holding with 0 return
                    portfolio_holdings.append({
                        "ticker": h.ticker,
                        "weight": weight,
                        "return": 0.0,
                        "contribution": 0.0,
                    })

        # Calculate universe (EW benchmark) attribution
        universe_holdings = []
        benchmark_total = 0.0

        # Count tickers with valid price data for equal weight calculation
        valid_universe_tickers = [t for t in universe_tickers if t in price_data]
        n_tickers = len(valid_universe_tickers)

        if n_tickers > 0:
            ew_weight = 1.0 / n_tickers

            for ticker in universe_tickers:
                ticker_data = price_data.get(ticker)

                if ticker_data:
                    ret = ticker_data["return"]
                    contribution = ew_weight * ret
                    benchmark_total += contribution

                    universe_holdings.append({
                        "ticker": ticker,
                        "weight": ew_weight,
                        "return": ret,
                        "contribution": contribution,
                    })
                else:
                    # No price data for this ticker
                    universe_holdings.append({
                        "ticker": ticker,
                        "weight": 0.0,  # Excluded from benchmark
                        "return": None,
                        "contribution": 0.0,
                    })

        return {
            "portfolio_holdings": portfolio_holdings,
            "universe_holdings": universe_holdings,
            "portfolio_total": portfolio_total,
            "benchmark_total": benchmark_total,
            "alpha": portfolio_total - benchmark_total,
        }


# Type hint for forward reference
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .tracker import PortfolioTracker