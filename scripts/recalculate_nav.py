#!/usr/bin/env python3
"""
Recalculate NAV Script.

Recalculates NAV for portfolios using the shares-based buy-and-hold methodology.
This should be run AFTER backfill_holdings_shares.py to ensure all holdings have shares.

Usage:
    # Recalculate all portfolios from their start date
    python scripts/recalculate_nav.py

    # Specific portfolio
    python scripts/recalculate_nav.py --portfolio "SA Large Caps"

    # Specific date range
    python scripts/recalculate_nav.py --start-date 2024-01-01 --end-date 2024-12-31

    # Dry run (preview which dates would be recalculated)
    python scripts/recalculate_nav.py --dry-run

Methodology:
    NAV = sum(shares × current_price) for each holding

    This is the correct buy-and-hold approach where:
    - Share counts are fixed at rebalance
    - Position values drift with prices between rebalances
"""

import os
import sys
import logging
import argparse
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load secrets
def _load_secrets():
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
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def get_trading_days(start_date: date, end_date: date) -> list[date]:
    """Get list of trading days between start and end dates."""
    import pandas as pd
    from pandas.tseries.holiday import USFederalHolidayCalendar
    from pandas.tseries.offsets import CustomBusinessDay

    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    trading_days = pd.date_range(start=start_date, end=end_date, freq=us_bd)
    return [d.date() for d in trading_days]


def delete_nav_records(
    portfolio_id: int,
    start_date: date,
    end_date: date,
    dry_run: bool = False,
) -> int:
    """Delete NAV records for a portfolio in a date range."""
    from sqlmodel import Session, select, delete
    from AlphaMachine_core.db import engine
    from AlphaMachine_core.tracking.models import PortfolioDailyNAV

    with Session(engine) as session:
        # Count records to delete
        count_stmt = (
            select(PortfolioDailyNAV)
            .where(PortfolioDailyNAV.portfolio_id == portfolio_id)
            .where(PortfolioDailyNAV.trade_date >= start_date)
            .where(PortfolioDailyNAV.trade_date <= end_date)
        )
        records = session.exec(count_stmt).all()
        count = len(records)

        if count > 0 and not dry_run:
            # Delete records
            for record in records:
                session.delete(record)
            session.commit()

        return count


def recalculate_portfolio_nav(
    tracker,
    portfolio,
    start_date: date,
    end_date: date,
    dry_run: bool = False,
) -> dict:
    """
    Recalculate NAV for a portfolio over a date range.

    Returns:
        Dict with statistics
    """
    from AlphaMachine_core.tracking import Variants
    from AlphaMachine_core.data_manager import StockDataManager

    dm = StockDataManager()

    stats = {
        "portfolio": portfolio.name,
        "trading_days": 0,
        "nav_calculated": 0,
        "skipped_no_holdings": 0,
        "skipped_no_prices": 0,
        "errors": [],
    }

    logger.info(f"\nRecalculating NAV for: {portfolio.name}")
    logger.info(f"  Date range: {start_date} to {end_date}")

    # Get trading days
    trading_days = get_trading_days(start_date, end_date)
    stats["trading_days"] = len(trading_days)
    logger.info(f"  Trading days: {len(trading_days)}")

    if not trading_days:
        return stats

    # Delete existing NAV records
    if not dry_run:
        deleted = delete_nav_records(portfolio.id, start_date, end_date)
        logger.info(f"  Deleted {deleted} existing NAV records")

    # Get all tickers needed
    all_tickers = set()
    for d in trading_days:
        holdings = tracker.get_holdings(portfolio.id, d)
        all_tickers.update(h.ticker for h in holdings)

    if not all_tickers:
        logger.warning(f"  No holdings found for any date")
        return stats

    # Load all price data at once
    logger.info(f"  Loading prices for {len(all_tickers)} tickers...")
    import pandas as pd

    price_dicts = dm.get_price_data(
        list(all_tickers),
        (start_date - timedelta(days=7)).strftime("%Y-%m-%d"),  # Extra days for prev prices
        end_date.strftime("%Y-%m-%d"),
    )
    price_df = pd.DataFrame(price_dicts)

    if price_df.empty:
        logger.error("  No price data available")
        return stats

    price_df["trade_date"] = pd.to_datetime(price_df["trade_date"]).dt.date

    # Track NAV history for return calculations
    previous_nav = None
    initial_nav = None

    # Process each trading day
    for trade_date in trading_days:
        # Get holdings
        holdings = tracker.get_holdings(portfolio.id, trade_date)

        if not holdings:
            stats["skipped_no_holdings"] += 1
            continue

        # Check if holdings have shares
        if not all(h.shares for h in holdings):
            logger.warning(
                f"  {trade_date}: Holdings missing shares - run backfill_holdings_shares.py first"
            )
            stats["errors"].append(f"{trade_date}: Missing shares")
            continue

        # Get prices for this date
        date_prices = price_df[price_df["trade_date"] == trade_date]
        if date_prices.empty:
            stats["skipped_no_prices"] += 1
            continue

        # Build price dict (use adjusted_close if available)
        if "adjusted_close" in date_prices.columns:
            price_data = dict(zip(
                date_prices["ticker"],
                date_prices["adjusted_close"].fillna(date_prices["close"])
            ))
        else:
            price_data = dict(zip(date_prices["ticker"], date_prices["close"]))

        # Calculate NAV using shares: NAV = sum(shares × price)
        raw_nav = Decimal("0")
        for h in holdings:
            ticker_price = price_data.get(h.ticker)
            if ticker_price and h.shares:
                raw_nav += h.shares * Decimal(str(ticker_price))

        raw_nav = float(raw_nav)

        # Track initial NAV
        if initial_nav is None:
            initial_nav = raw_nav

        # =====================================================================
        # GIPS-COMPLIANT RETURN HANDLING
        # =====================================================================
        # 1. First day: daily_return = 0 (no prior NAV to compare)
        # 2. Rebalance days: daily_return = (prev_holdings × today_prices) / prev_NAV - 1
        #    This reflects ONLY price changes, not portfolio restructuring.

        daily_return_override = None

        # FIRST DAY: No previous NAV, so daily return must be 0
        if previous_nav is None:
            daily_return_override = 0.0
            logger.debug(f"  {trade_date}: FIRST DAY - setting daily_return=0%")

        # REBALANCE DETECTION (only if not first day)
        if previous_nav is not None:
            prev_holdings = tracker.get_holdings(portfolio.id, trade_date - timedelta(days=1))
            today_eff_date = holdings[0].effective_date if holdings else None
            prev_eff_date = prev_holdings[0].effective_date if prev_holdings else None

            if prev_holdings and today_eff_date != prev_eff_date:
                # Rebalance detected - calculate GIPS-compliant return
                # using previous holdings at today's prices
                if all(h.shares for h in prev_holdings):
                    prev_holdings_value_today = sum(
                        float(h.shares) * price_data.get(h.ticker, 0)
                        for h in prev_holdings
                    )

                    if prev_holdings_value_today > 0 and previous_nav > 0:
                        gips_daily_return = (prev_holdings_value_today / previous_nav) - 1
                        naive_return = (raw_nav / previous_nav) - 1

                        logger.debug(
                            f"  {trade_date}: REBALANCE - GIPS return={gips_daily_return*100:.2f}% "
                            f"(naive={naive_return*100:.2f}%)"
                        )

                        daily_return_override = gips_daily_return

        # Record NAV
        if not dry_run:
            # Get previous prices for overlay calculation
            prev_date = trade_date - timedelta(days=1)
            prev_prices_data = price_df[price_df["trade_date"] < trade_date]
            prev_price_data = {}
            if not prev_prices_data.empty:
                latest_prev = prev_prices_data["trade_date"].max()
                prev_day = prev_prices_data[prev_prices_data["trade_date"] == latest_prev]
                if "adjusted_close" in prev_day.columns:
                    prev_price_data = dict(zip(
                        prev_day["ticker"],
                        prev_day["adjusted_close"].fillna(prev_day["close"])
                    ))
                else:
                    prev_price_data = dict(zip(prev_day["ticker"], prev_day["close"]))

            # Use tracker's update method for all variants
            # Pass daily_return_override for GIPS compliance on rebalance days
            tracker.update_daily_nav(
                portfolio_id=portfolio.id,
                trade_date=trade_date,
                raw_nav=raw_nav,
                previous_raw_nav=previous_nav or 100.0,
                initial_nav=initial_nav,
                daily_return_override=daily_return_override,
            )

        stats["nav_calculated"] += 1
        previous_nav = raw_nav

        if stats["nav_calculated"] % 50 == 0:
            logger.info(f"  Processed {stats['nav_calculated']} days...")

    logger.info(
        f"  Completed: {stats['nav_calculated']} NAV records calculated"
    )

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Recalculate portfolio NAV using shares-based methodology"
    )
    parser.add_argument(
        "--portfolio",
        type=str,
        help="Specific portfolio name to recalculate",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD). Default: portfolio start date",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date (YYYY-MM-DD). Default: today",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying database",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Recalculate NAV (Shares-Based)")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be saved")

    from AlphaMachine_core.tracking import get_tracker

    tracker = get_tracker()

    # Get portfolios
    if args.portfolio:
        portfolio = tracker.get_portfolio_by_name(args.portfolio)
        if not portfolio:
            logger.error(f"Portfolio not found: {args.portfolio}")
            sys.exit(1)
        portfolios = [portfolio]
    else:
        portfolios = tracker.list_portfolios(active_only=False)

    logger.info(f"Processing {len(portfolios)} portfolio(s)")

    # Parse dates
    end_date = date.today()
    if args.end_date:
        end_date = date.fromisoformat(args.end_date)

    # Process each portfolio
    all_stats = []
    for portfolio in portfolios:
        try:
            # Determine start date
            if args.start_date:
                start_date = date.fromisoformat(args.start_date)
            else:
                start_date = portfolio.start_date

            stats = recalculate_portfolio_nav(
                tracker,
                portfolio,
                start_date,
                end_date,
                dry_run=args.dry_run,
            )
            all_stats.append(stats)
        except Exception as e:
            logger.error(f"Error processing {portfolio.name}: {e}")
            import traceback
            traceback.print_exc()
            all_stats.append({
                "portfolio": portfolio.name,
                "error": str(e),
            })

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Recalculation Summary")
    logger.info("=" * 60)

    total_calculated = 0
    total_skipped = 0

    for stats in all_stats:
        if "error" in stats and isinstance(stats.get("error"), str):
            logger.info(f"  {stats['portfolio']}: ERROR - {stats['error']}")
        else:
            calculated = stats.get("nav_calculated", 0)
            skipped = stats.get("skipped_no_holdings", 0) + stats.get("skipped_no_prices", 0)
            logger.info(
                f"  {stats['portfolio']}: "
                f"{calculated} calculated, {skipped} skipped"
            )
            total_calculated += calculated
            total_skipped += skipped

    logger.info("-" * 60)
    logger.info(f"Total: {total_calculated} NAV records calculated")

    if args.dry_run:
        logger.info("\nDRY RUN complete - run without --dry-run to apply changes")

    # Also recompute metrics after recalculation
    if not args.dry_run and total_calculated > 0:
        logger.info("\nRecomputing metrics for updated portfolios...")
        for stats in all_stats:
            if stats.get("nav_calculated", 0) > 0:
                portfolio_name = stats["portfolio"]
                portfolio = tracker.get_portfolio_by_name(portfolio_name)
                if portfolio:
                    try:
                        metrics = tracker.compute_and_store_metrics(portfolio.id)
                        logger.info(f"  {portfolio_name}: {len(metrics)} metrics computed")
                    except Exception as e:
                        logger.error(f"  {portfolio_name}: Error computing metrics - {e}")


if __name__ == "__main__":
    main()
