#!/usr/bin/env python3
"""
Fix NAV for EqualWeight portfolios.

This script:
1. Clears incorrectly populated shares from EqualWeight portfolio holdings
2. Deletes existing NAV records
3. Recalculates NAV using weight-based methodology

Usage:
    python scripts/fix_equalweight_nav.py --portfolio "TR10 Large Caps X_EqualWeight"
    python scripts/fix_equalweight_nav.py --portfolio "TR10 Large Caps X_EqualWeight" --dry-run
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

from utils.trading_calendar import get_trading_days


def clear_shares_from_holdings(portfolio_id: int, dry_run: bool = False) -> int:
    """Clear shares from all holdings for a portfolio."""
    from sqlmodel import Session, select, update
    from AlphaMachine_core.db import engine
    from AlphaMachine_core.tracking.models import PortfolioHolding

    with Session(engine) as session:
        # Find holdings with shares
        holdings = session.exec(
            select(PortfolioHolding)
            .where(PortfolioHolding.portfolio_id == portfolio_id)
            .where(PortfolioHolding.shares != None)
        ).all()

        count = len(holdings)
        if count > 0:
            logger.info(f"  Found {count} holdings with shares to clear")
            if not dry_run:
                for h in holdings:
                    h.shares = None
                    session.add(h)
                session.commit()
                logger.info(f"  Cleared shares from {count} holdings")
        else:
            logger.info("  No holdings with shares found")

        return count


def delete_nav_records(portfolio_id: int, start_date: date, end_date: date, dry_run: bool = False) -> int:
    """Delete NAV records for a portfolio in a date range."""
    from sqlmodel import Session, select
    from AlphaMachine_core.db import engine
    from AlphaMachine_core.tracking.models import PortfolioDailyNAV

    with Session(engine) as session:
        records = session.exec(
            select(PortfolioDailyNAV)
            .where(PortfolioDailyNAV.portfolio_id == portfolio_id)
            .where(PortfolioDailyNAV.trade_date >= start_date)
            .where(PortfolioDailyNAV.trade_date <= end_date)
        ).all()

        count = len(records)
        if count > 0 and not dry_run:
            for record in records:
                session.delete(record)
            session.commit()

        return count


def recalculate_nav_weight_based(
    tracker,
    portfolio,
    start_date: date,
    end_date: date,
    dry_run: bool = False,
    baseline_nav: Optional[float] = None,
) -> dict:
    """
    Recalculate NAV for a portfolio using weight-based methodology.

    NAV = prev_NAV * (1 + weighted_return)

    This simulates daily rebalancing where each position maintains its target weight.
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

    logger.info(f"\nRecalculating NAV (weight-based) for: {portfolio.name}")
    logger.info(f"  Date range: {start_date} to {end_date}")

    # Get trading days
    trading_days = get_trading_days(start_date, end_date)
    stats["trading_days"] = len(trading_days)
    logger.info(f"  Trading days: {len(trading_days)}")

    if not trading_days:
        return stats

    # Get all tickers needed
    all_tickers = set()
    for d in trading_days:
        holdings = tracker.get_holdings(portfolio.id, d)
        all_tickers.update(h.ticker for h in holdings)

    if not all_tickers:
        logger.warning("  No holdings found for any date")
        return stats

    # Load all price data at once
    logger.info(f"  Loading prices for {len(all_tickers)} tickers...")
    import pandas as pd

    price_dicts = dm.get_price_data(
        list(all_tickers),
        (start_date - timedelta(days=7)).strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
    )
    price_df = pd.DataFrame(price_dicts)

    if price_df.empty:
        logger.error("  No price data available")
        return stats

    price_df["trade_date"] = pd.to_datetime(price_df["trade_date"]).dt.date

    # Track NAV history - use baseline_nav if provided
    previous_nav = baseline_nav
    initial_nav = baseline_nav if baseline_nav else None
    previous_prices = None

    if baseline_nav:
        logger.info(f"  Using baseline NAV: {baseline_nav:.2f}")

    # CRITICAL FIX: Initialize previous_prices from the most recent trading day BEFORE start_date
    # This ensures we calculate correct returns on the first day of recalculation
    dates_before = price_df[price_df["trade_date"] < start_date]["trade_date"].unique()
    if len(dates_before) > 0:
        latest_prev_date = max(dates_before)
        prev_day_prices = price_df[price_df["trade_date"] == latest_prev_date]
        if "adjusted_close" in prev_day_prices.columns:
            previous_prices = dict(zip(
                prev_day_prices["ticker"],
                prev_day_prices["adjusted_close"].fillna(prev_day_prices["close"])
            ))
        else:
            previous_prices = dict(zip(prev_day_prices["ticker"], prev_day_prices["close"]))
        logger.info(f"  Initialized previous_prices from {latest_prev_date}: {len(previous_prices)} tickers")

    # Process each trading day
    for trade_date in trading_days:
        holdings = tracker.get_holdings(portfolio.id, trade_date)

        if not holdings:
            stats["skipped_no_holdings"] += 1
            continue

        # Get prices for this date
        date_prices = price_df[price_df["trade_date"] == trade_date]
        if date_prices.empty:
            stats["skipped_no_prices"] += 1
            continue

        # Build price dict
        if "adjusted_close" in date_prices.columns:
            price_data = dict(zip(
                date_prices["ticker"],
                date_prices["adjusted_close"].fillna(date_prices["close"])
            ))
        else:
            price_data = dict(zip(date_prices["ticker"], date_prices["close"]))

        # Calculate NAV using weight-based method with NORMALIZED weights
        # The tracker.calculate_raw_nav doesn't normalize, so we do it manually here

        # Get weights and normalize them to sum to 1.0
        weights = {}
        for h in holdings:
            if h.weight:
                weights[h.ticker] = float(h.weight)
            else:
                weights[h.ticker] = 1.0 / len(holdings)  # Default to equal weight

        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {t: w / total_weight for t, w in weights.items()}

        # Calculate weighted return
        total_return = 0.0
        for ticker, weight in weights.items():
            current_price = price_data.get(ticker)
            prev_price = previous_prices.get(ticker) if previous_prices else None

            if current_price and prev_price and prev_price > 0:
                position_return = (current_price / prev_price) - 1
                total_return += weight * position_return

        # On first day, initialize NAV to 100 then apply any return if we have previous prices
        if previous_nav is None:
            raw_nav = 100.0 * (1 + total_return) if previous_prices else 100.0
        else:
            raw_nav = previous_nav * (1 + total_return)

        # First day check
        if initial_nav is None:
            initial_nav = raw_nav
            logger.info(f"  Initial NAV: {raw_nav:.2f}")

        # Calculate daily return as decimal (not percentage)
        if previous_nav is not None:
            daily_return = (raw_nav / previous_nav) - 1
        else:
            daily_return = 0.0

        # Calculate cumulative return as decimal
        cumulative_return = (raw_nav / initial_nav) - 1

        if dry_run:
            logger.info(f"  {trade_date}: NAV={raw_nav:.2f}, daily={daily_return*100:+.2f}%, cumul={cumulative_return*100:+.2f}%")
        else:
            # Record NAV for raw variant (100% equity)
            tracker.record_nav(
                portfolio_id=portfolio.id,
                trade_date=trade_date,
                variant=Variants.RAW,
                nav=raw_nav,
                daily_return=daily_return,
                cumulative_return=cumulative_return,
            )

        stats["nav_calculated"] += 1
        previous_nav = raw_nav
        previous_prices = price_data

    if not dry_run and stats["nav_calculated"] > 0:
        logger.info(f"  Recorded {stats['nav_calculated']} NAV entries")
        final_nav = previous_nav
        total_return = (final_nav / initial_nav) - 1
        logger.info(f"  Final NAV: {final_nav:.2f} (total return: {total_return*100:+.2f}%)")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Fix NAV for EqualWeight portfolios")
    parser.add_argument("--portfolio", required=True, help="Portfolio name to fix")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    args = parser.parse_args()

    from AlphaMachine_core.tracking import PortfolioTracker

    tracker = PortfolioTracker()

    # Find portfolio
    portfolio = tracker.get_portfolio_by_name(args.portfolio)
    if not portfolio:
        logger.error(f"Portfolio not found: {args.portfolio}")
        return 1

    logger.info(f"Found portfolio: {portfolio.name} (ID: {portfolio.id})")

    # Verify it's an EqualWeight portfolio
    if "_EqualWeight" not in portfolio.name:
        logger.warning(f"This portfolio doesn't appear to be EqualWeight")
        confirm = input("Continue anyway? (y/N): ")
        if confirm.lower() != 'y':
            return 1

    # Get date range - find the first date with holdings
    from AlphaMachine_core.db import get_session
    from sqlalchemy import text
    from AlphaMachine_core.tracking import Variants

    with get_session() as session:
        result = session.execute(text('''
            SELECT MIN(effective_date) FROM portfolio_holdings WHERE portfolio_id = :pid
        '''), {'pid': portfolio.id})
        first_holdings_date = result.scalar()

    if not first_holdings_date:
        logger.error("No holdings found for this portfolio")
        return 1

    # Start from the first trading day on or after the holdings became effective
    start_date = first_holdings_date
    end_date = date.today()

    # Get the previous NAV (before first holdings date) to use as baseline
    existing_nav_df = tracker.get_nav_series(portfolio.id, Variants.RAW, end_date=first_holdings_date - timedelta(days=1))
    if not existing_nav_df.empty:
        baseline_nav = float(existing_nav_df["nav"].iloc[-1])
        baseline_date = existing_nav_df.index[-1].date() if hasattr(existing_nav_df.index[-1], 'date') else existing_nav_df.index[-1]
        logger.info(f"Found baseline NAV: {baseline_nav:.2f} from {baseline_date}")
    else:
        baseline_nav = None
        logger.info("No baseline NAV found - will start fresh at 100")

    logger.info(f"\n{'='*60}")
    logger.info(f"Fixing EqualWeight portfolio: {portfolio.name}")
    logger.info(f"{'='*60}")
    logger.info(f"First holdings date: {first_holdings_date}")
    logger.info(f"Recalculating from: {start_date} to {end_date}")

    # Step 1: Clear shares from holdings
    logger.info("\nStep 1: Clearing shares from holdings...")
    shares_cleared = clear_shares_from_holdings(portfolio.id, dry_run=args.dry_run)

    # Step 2: Delete existing NAV records only from first holdings date onwards
    logger.info("\nStep 2: Deleting existing NAV records...")
    nav_deleted = delete_nav_records(portfolio.id, start_date, end_date, dry_run=args.dry_run)
    logger.info(f"  {'Would delete' if args.dry_run else 'Deleted'} {nav_deleted} NAV records")

    # Step 3: Recalculate NAV using weight-based method
    logger.info("\nStep 3: Recalculating NAV (weight-based)...")
    stats = recalculate_nav_weight_based(tracker, portfolio, start_date, end_date, dry_run=args.dry_run, baseline_nav=baseline_nav)

    logger.info(f"\n{'='*60}")
    logger.info("Summary:")
    logger.info(f"  Shares cleared: {shares_cleared}")
    logger.info(f"  NAV records deleted: {nav_deleted}")
    logger.info(f"  NAV records calculated: {stats['nav_calculated']}")
    logger.info(f"  Trading days processed: {stats['trading_days']}")
    if args.dry_run:
        logger.info("\n  ** DRY RUN - no changes made **")
    logger.info(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
