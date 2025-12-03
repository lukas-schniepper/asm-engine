#!/usr/bin/env python3
"""
Scheduled NAV Update Script for Portfolio Tracking.

This script updates daily NAV for all active portfolios, calculating:
- Raw NAV (100% equity)
- Conservative Model overlay NAV
- Trend Regime V2 overlay NAV

It also computes and stores periodic metrics (week, month, quarter, year, ytd, all).

Usage:
    python scripts/scheduled_nav_update.py

    # Specific portfolio
    python scripts/scheduled_nav_update.py --portfolio "TopWeights_20_MVO"

    # Specific date (for backfilling)
    python scripts/scheduled_nav_update.py --date 2025-11-30

    # Date range backfill
    python scripts/scheduled_nav_update.py --start-date 2025-01-01 --end-date 2025-11-30

Environment:
    DATABASE_URL must be set (or in .streamlit/secrets.toml)
    AWS credentials for S3 access (optional, uses cache if unavailable)

Scheduling:
    Run this script daily after market close (e.g., 6 PM ET via cron or GitHub Actions)

    Cron example:
        0 18 * * 1-5 cd /path/to/asm-engine && python scripts/scheduled_nav_update.py
"""

import os
import sys
import logging
import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load secrets from .streamlit/secrets.toml if not already in environment
def _load_secrets():
    """Load secrets from .streamlit/secrets.toml into environment variables."""
    secrets_path = project_root / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        with open(secrets_path, "rb") as f:
            secrets = tomllib.load(f)

        for key, value in secrets.items():
            if key not in os.environ and isinstance(value, str):
                os.environ[key] = value

_load_secrets()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def is_trading_day(d: date) -> bool:
    """Check if a date is a US trading day (weekday, not holiday)."""
    import pandas as pd
    from pandas.tseries.holiday import USFederalHolidayCalendar

    if d.weekday() >= 5:  # Weekend
        return False

    # Check for US federal holidays
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=str(d.year), end=str(d.year + 1))
    return d not in holidays.date


def get_trading_days(start_date: date, end_date: date) -> list[date]:
    """Get list of trading days between start and end dates."""
    import pandas as pd
    from pandas.tseries.holiday import USFederalHolidayCalendar
    from pandas.tseries.offsets import CustomBusinessDay

    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    trading_days = pd.date_range(start=start_date, end=end_date, freq=us_bd)
    return [d.date() for d in trading_days]


def update_portfolio_nav(
    tracker,
    portfolio,
    trade_date: date,
    price_data: dict[str, float],
    prev_price_data: Optional[dict[str, float]] = None,
) -> bool:
    """
    Update NAV for a single portfolio on a specific date.

    Args:
        tracker: PortfolioTracker instance
        portfolio: PortfolioDefinition object
        trade_date: Trading date
        price_data: Dict mapping ticker to price
        prev_price_data: Dict mapping ticker to previous day's price

    Returns:
        True if successful, False otherwise
    """
    try:
        # Get holdings as of trade date
        holdings = tracker.get_holdings(portfolio.id, trade_date)

        if not holdings:
            logger.warning(
                f"No holdings found for portfolio {portfolio.name} as of {trade_date}. "
                "Skipping NAV update."
            )
            return False

        # Get previous NAV for return calculation
        from AlphaMachine_core.tracking import Variants

        prev_nav_df = tracker.get_nav_series(
            portfolio.id,
            Variants.RAW,
            end_date=trade_date - timedelta(days=1),  # Get NAV up to yesterday
        )

        if prev_nav_df.empty:
            previous_raw_nav = 100.0  # Initial NAV
            initial_nav = 100.0
        else:
            previous_raw_nav = prev_nav_df["nav"].iloc[-1]
            initial_nav = prev_nav_df["nav"].iloc[0]

        # Calculate raw NAV using current and previous prices
        raw_nav = tracker.calculate_raw_nav(holdings, price_data, previous_raw_nav, prev_price_data)

        # Update NAV for all variants
        results = tracker.update_daily_nav(
            portfolio_id=portfolio.id,
            trade_date=trade_date,
            raw_nav=raw_nav,
            previous_raw_nav=previous_raw_nav,
            initial_nav=initial_nav,
        )

        logger.info(
            f"Updated NAV for {portfolio.name} on {trade_date}: "
            f"raw={raw_nav:.2f}, variants={list(results.keys())}"
        )
        return True

    except Exception as e:
        logger.error(f"Error updating NAV for {portfolio.name} on {trade_date}: {e}")
        return False


def run_daily_update(
    portfolio_name: Optional[str] = None,
    trade_date: Optional[date] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    compute_metrics: bool = True,
) -> dict:
    """
    Run daily NAV update for portfolios.

    Args:
        portfolio_name: Optional specific portfolio to update
        trade_date: Optional specific date (default: today)
        start_date: Optional start date for range backfill
        end_date: Optional end date for range backfill
        compute_metrics: Whether to compute periodic metrics after update

    Returns:
        Dict with update statistics
    """
    from AlphaMachine_core.tracking import PortfolioTracker, get_tracker
    from AlphaMachine_core.data_manager import StockDataManager

    logger.info("=" * 60)
    logger.info("Portfolio NAV Update - Starting")
    logger.info("=" * 60)

    # Initialize tracker and data manager
    tracker = get_tracker()
    dm = StockDataManager()

    # Determine dates to process
    if start_date and end_date:
        dates_to_process = get_trading_days(start_date, end_date)
        logger.info(f"Backfilling {len(dates_to_process)} trading days from {start_date} to {end_date}")
    elif trade_date:
        dates_to_process = [trade_date]
    else:
        # Use the most recent COMPLETED trading day
        # Since this runs after market close (typically overnight UTC),
        # we need to find the last trading day with available price data
        from datetime import datetime
        import pytz

        # Get current time in ET
        et_tz = pytz.timezone('America/New_York')
        now_et = datetime.now(et_tz)

        # Start with today in ET timezone
        trade_date = now_et.date()

        # If before market close (4 PM ET), use previous day
        # Market closes at 16:00 ET, add buffer for data availability
        if now_et.hour < 18:  # Before 6 PM ET - data may not be ready
            trade_date -= timedelta(days=1)
            logger.info(f"Before market data cutoff, starting from previous day: {trade_date}")

        # Find most recent trading day
        while not is_trading_day(trade_date):
            trade_date -= timedelta(days=1)

        if trade_date != now_et.date():
            logger.info(f"Using most recent completed trading day: {trade_date}")

        dates_to_process = [trade_date]

    # Get portfolios to update
    if portfolio_name:
        portfolio = tracker.get_portfolio_by_name(portfolio_name)
        if not portfolio:
            logger.error(f"Portfolio '{portfolio_name}' not found")
            return {"success": False, "error": "Portfolio not found"}
        portfolios = [portfolio]
    else:
        portfolios = tracker.list_portfolios(active_only=True)

    if not portfolios:
        logger.warning("No active portfolios to update")
        return {"success": True, "updated": 0, "message": "No portfolios to update"}

    logger.info(f"Updating {len(portfolios)} portfolio(s)")

    # Statistics
    stats = {
        "dates_processed": len(dates_to_process),
        "portfolios": len(portfolios),
        "successful_updates": 0,
        "failed_updates": 0,
        "errors": [],
    }

    # Collect all tickers needed
    all_tickers = set()
    for portfolio in portfolios:
        for d in dates_to_process:
            holdings = tracker.get_holdings(portfolio.id, d)
            all_tickers.update(h.ticker for h in holdings)

    if not all_tickers:
        logger.warning("No tickers found in portfolio holdings")
        return stats

    # Get price data for all dates
    logger.info(f"Loading prices for {len(all_tickers)} tickers...")

    min_date = min(dates_to_process)
    max_date = max(dates_to_process)

    price_dicts = dm.get_price_data(
        list(all_tickers),
        min_date.strftime("%Y-%m-%d"),
        max_date.strftime("%Y-%m-%d"),
    )

    # Build price lookup
    import pandas as pd

    price_df = pd.DataFrame(price_dicts)

    # If no price data, fail fast - the price update workflow should have run first
    if price_df.empty:
        logger.error(
            "No price data available in database. "
            "Run the 'Update Ticker Prices' workflow first to populate price cache."
        )
        return {"success": False, "error": "No price data - run price update first"}

    price_df["trade_date"] = pd.to_datetime(price_df["trade_date"]).dt.date

    # Also get previous day prices for return calculation
    prev_min_date = min_date - timedelta(days=7)  # Look back a week for previous prices
    prev_price_dicts = dm.get_price_data(
        list(all_tickers),
        prev_min_date.strftime("%Y-%m-%d"),
        max_date.strftime("%Y-%m-%d"),
    )
    prev_price_df = pd.DataFrame(prev_price_dicts)
    if not prev_price_df.empty:
        prev_price_df["trade_date"] = pd.to_datetime(prev_price_df["trade_date"]).dt.date

    # Process each date
    for process_date in dates_to_process:
        logger.info(f"\nProcessing date: {process_date}")

        # Get prices for this date
        date_prices = price_df[price_df["trade_date"] == process_date]
        price_data = dict(zip(date_prices["ticker"], date_prices["close"]))

        if not price_data:
            logger.warning(f"No price data for {process_date}")
            continue

        # Get previous day's prices
        prev_date = process_date - timedelta(days=1)
        # Find the most recent trading day with prices
        prev_prices_data = {}
        if not prev_price_df.empty:
            prev_dates = prev_price_df[prev_price_df["trade_date"] < process_date]
            if not prev_dates.empty:
                latest_prev_date = prev_dates["trade_date"].max()
                prev_day_prices = prev_price_df[prev_price_df["trade_date"] == latest_prev_date]
                prev_prices_data = dict(zip(prev_day_prices["ticker"], prev_day_prices["close"]))

        # Update each portfolio
        for portfolio in portfolios:
            success = update_portfolio_nav(tracker, portfolio, process_date, price_data, prev_prices_data)
            if success:
                stats["successful_updates"] += 1
            else:
                stats["failed_updates"] += 1

    # Compute metrics
    if compute_metrics and stats["successful_updates"] > 0:
        logger.info("\nComputing periodic metrics...")
        for portfolio in portfolios:
            try:
                metrics = tracker.compute_and_store_metrics(portfolio.id)
                logger.info(f"Computed {len(metrics)} metrics for {portfolio.name}")
            except Exception as e:
                logger.error(f"Error computing metrics for {portfolio.name}: {e}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Update Complete")
    logger.info("=" * 60)
    logger.info(f"Dates processed: {stats['dates_processed']}")
    logger.info(f"Portfolios: {stats['portfolios']}")
    logger.info(f"Successful updates: {stats['successful_updates']}")
    logger.info(f"Failed updates: {stats['failed_updates']}")

    return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Update portfolio NAV for tracking system"
    )
    parser.add_argument(
        "--portfolio",
        type=str,
        help="Specific portfolio name to update (default: all active)",
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Specific date to update (YYYY-MM-DD format)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for backfill range (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date for backfill range (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Skip computing periodic metrics",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without making changes",
    )

    args = parser.parse_args()

    # Parse dates
    trade_date = None
    start_date = None
    end_date = None

    if args.date:
        trade_date = datetime.strptime(args.date, "%Y-%m-%d").date()

    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()

    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()

    if (start_date and not end_date) or (end_date and not start_date):
        parser.error("Both --start-date and --end-date are required for range backfill")

    if args.dry_run:
        logger.info("DRY RUN - No changes will be made")
        logger.info(f"Would update portfolio: {args.portfolio or 'all active'}")
        if start_date and end_date:
            days = get_trading_days(start_date, end_date)
            logger.info(f"Would process {len(days)} trading days from {start_date} to {end_date}")
        elif trade_date:
            logger.info(f"Would process date: {trade_date}")
        else:
            logger.info("Would process: today (or most recent trading day)")
        return

    # Run update
    stats = run_daily_update(
        portfolio_name=args.portfolio,
        trade_date=trade_date,
        start_date=start_date,
        end_date=end_date,
        compute_metrics=not args.no_metrics,
    )

    # Exit code based on results
    if stats.get("failed_updates", 0) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
