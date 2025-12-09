#!/usr/bin/env python3
"""
Backfill Holdings Shares Script.

Populates the 'shares' field for existing portfolio holdings that only have weights.
This enables proper buy-and-hold NAV calculation.

Usage:
    # Backfill all portfolios
    python scripts/backfill_holdings_shares.py

    # Specific portfolio
    python scripts/backfill_holdings_shares.py --portfolio "SA Large Caps"

    # Dry run (preview changes without saving)
    python scripts/backfill_holdings_shares.py --dry-run

Formula:
    shares = (weight × NAV) / price

    Where NAV is the portfolio NAV as of the holding's effective_date.
    For the first holding date, NAV = 100 (initial value).
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


def get_prices_for_date(tickers: list[str], trade_date: date) -> dict[str, Decimal]:
    """Get prices for tickers as of a specific date."""
    from sqlmodel import Session, select
    from AlphaMachine_core.db import engine
    from AlphaMachine_core.models import PriceData

    prices = {}

    with Session(engine) as session:
        for ticker in tickers:
            stmt = (
                select(PriceData)
                .where(PriceData.ticker == ticker)
                .where(PriceData.trade_date <= trade_date)
                .order_by(PriceData.trade_date.desc())
                .limit(1)
            )
            result = session.exec(stmt).first()

            if result:
                price = result.adjusted_close if result.adjusted_close else result.close
                prices[ticker] = Decimal(str(price))

    return prices


def get_nav_for_date(portfolio_id: int, trade_date: date) -> Optional[Decimal]:
    """Get NAV for a portfolio as of a specific date."""
    from sqlmodel import Session, select
    from AlphaMachine_core.db import engine
    from AlphaMachine_core.tracking.models import PortfolioDailyNAV, Variants

    with Session(engine) as session:
        stmt = (
            select(PortfolioDailyNAV.nav)
            .where(PortfolioDailyNAV.portfolio_id == portfolio_id)
            .where(PortfolioDailyNAV.trade_date <= trade_date)
            .where(PortfolioDailyNAV.variant == Variants.RAW)
            .order_by(PortfolioDailyNAV.trade_date.desc())
            .limit(1)
        )
        result = session.exec(stmt).first()

        if result:
            return Decimal(str(result))

    return None


def backfill_portfolio_shares(
    portfolio_id: int,
    portfolio_name: str,
    dry_run: bool = False,
) -> dict:
    """
    Backfill shares for all holdings of a portfolio.

    Returns:
        Dict with statistics about the backfill
    """
    from sqlmodel import Session, select
    from AlphaMachine_core.db import engine
    from AlphaMachine_core.tracking.models import PortfolioHolding
    from AlphaMachine_core.tracking import calculate_shares_from_weights

    stats = {
        "portfolio": portfolio_name,
        "effective_dates": 0,
        "holdings_processed": 0,
        "holdings_updated": 0,
        "holdings_skipped": 0,
        "errors": [],
    }

    logger.info(f"\nProcessing portfolio: {portfolio_name} (id={portfolio_id})")

    with Session(engine) as session:
        # Get all distinct effective_dates for this portfolio
        stmt = (
            select(PortfolioHolding.effective_date)
            .where(PortfolioHolding.portfolio_id == portfolio_id)
            .distinct()
            .order_by(PortfolioHolding.effective_date)
        )
        effective_dates = session.exec(stmt).all()

        if not effective_dates:
            logger.warning(f"No holdings found for {portfolio_name}")
            return stats

        stats["effective_dates"] = len(effective_dates)
        logger.info(f"Found {len(effective_dates)} effective dates")

        # Process each effective date
        for eff_date in effective_dates:
            # Get holdings for this date
            holdings_stmt = (
                select(PortfolioHolding)
                .where(PortfolioHolding.portfolio_id == portfolio_id)
                .where(PortfolioHolding.effective_date == eff_date)
            )
            holdings = session.exec(holdings_stmt).all()

            # Check if any need shares calculated
            holdings_need_shares = [h for h in holdings if h.shares is None]

            if not holdings_need_shares:
                logger.debug(f"  {eff_date}: All {len(holdings)} holdings have shares")
                stats["holdings_skipped"] += len(holdings)
                continue

            stats["holdings_processed"] += len(holdings_need_shares)

            # Get NAV as of this date (or previous day for first calculation)
            nav = get_nav_for_date(portfolio_id, eff_date - timedelta(days=1))
            if nav is None:
                # First day - use initial NAV
                nav = Decimal("100")
                logger.debug(f"  {eff_date}: Using initial NAV=100")
            else:
                logger.debug(f"  {eff_date}: Using NAV={nav:.2f}")

            # Get prices for holdings
            tickers = [h.ticker for h in holdings_need_shares]
            prices = get_prices_for_date(tickers, eff_date)

            if not prices:
                error_msg = f"No prices found for {eff_date}"
                stats["errors"].append(error_msg)
                logger.warning(f"  {eff_date}: {error_msg}")
                continue

            # Build holdings list with weights
            holdings_with_weights = [
                {"ticker": h.ticker, "weight": float(h.weight) if h.weight else 0}
                for h in holdings_need_shares
            ]

            # Calculate shares
            holdings_with_shares = calculate_shares_from_weights(
                holdings_with_weights, nav, prices
            )

            # Update holdings
            updated_count = 0
            for h_orig, h_new in zip(holdings_need_shares, holdings_with_shares):
                if h_new.get("shares") is not None:
                    if not dry_run:
                        h_orig.shares = h_new["shares"]
                        h_orig.entry_price = h_new.get("entry_price")
                    updated_count += 1
                    logger.debug(
                        f"    {h_orig.ticker}: shares={h_new['shares']:.6f}"
                    )

            stats["holdings_updated"] += updated_count
            logger.info(
                f"  {eff_date}: Updated {updated_count}/{len(holdings_need_shares)} holdings"
            )

        # Commit if not dry run
        if not dry_run:
            session.commit()
            logger.info(f"Committed changes for {portfolio_name}")
        else:
            logger.info(f"DRY RUN - no changes saved for {portfolio_name}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Backfill shares for portfolio holdings"
    )
    parser.add_argument(
        "--portfolio",
        type=str,
        help="Specific portfolio name to backfill",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without saving to database",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Backfill Holdings Shares")
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

    # Process each portfolio
    all_stats = []
    for portfolio in portfolios:
        try:
            stats = backfill_portfolio_shares(
                portfolio.id, portfolio.name, dry_run=args.dry_run
            )
            all_stats.append(stats)
        except Exception as e:
            logger.error(f"Error processing {portfolio.name}: {e}")
            all_stats.append({
                "portfolio": portfolio.name,
                "error": str(e),
            })

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Backfill Summary")
    logger.info("=" * 60)

    total_updated = 0
    total_processed = 0
    total_skipped = 0

    for stats in all_stats:
        if "error" in stats:
            logger.info(f"  {stats['portfolio']}: ERROR - {stats['error']}")
        else:
            logger.info(
                f"  {stats['portfolio']}: "
                f"{stats['holdings_updated']} updated, "
                f"{stats['holdings_skipped']} already had shares"
            )
            total_updated += stats.get("holdings_updated", 0)
            total_processed += stats.get("holdings_processed", 0)
            total_skipped += stats.get("holdings_skipped", 0)

    logger.info("-" * 60)
    logger.info(f"Total: {total_updated} updated, {total_skipped} skipped")

    if args.dry_run:
        logger.info("\nDRY RUN complete - run without --dry-run to apply changes")


if __name__ == "__main__":
    main()
