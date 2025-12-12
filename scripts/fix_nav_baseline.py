#!/usr/bin/env python3
"""
Fix NAV Baseline Script.

This script fixes portfolios where shares were calculated incorrectly at inception.
It clears existing shares/entry_price values and recalculates them properly.

Problem:
- Some portfolios have wrong entry_price stored (e.g., price from wrong date)
- This causes wrong shares calculation: shares = (weight × NAV) / entry_price
- NAV = shares × price produces wrong absolute values
- Daily returns are correct (wrong shares cancel out in ratio)
- But MTD/YTD calculations are wrong due to wrong baseline

Solution:
1. Clear shares/entry_price for all holdings of affected portfolios
2. Recalculate shares using correct prices from each holding's effective_date
3. Recalculate NAV

Usage:
    # Check which portfolios have baseline issues (dry run)
    python scripts/fix_nav_baseline.py --check-only

    # Fix specific portfolio
    python scripts/fix_nav_baseline.py --portfolio "QQQ"

    # Fix all portfolios with baseline drift > 0.5%
    python scripts/fix_nav_baseline.py --fix-all --threshold 0.5

    # Dry run (show what would be changed)
    python scripts/fix_nav_baseline.py --portfolio "QQQ" --dry-run
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


def get_price_for_ticker_date(ticker: str, trade_date: date) -> Optional[Decimal]:
    """Get price for a ticker on a specific date."""
    from sqlmodel import Session, select
    from AlphaMachine_core.db import engine
    from AlphaMachine_core.models import PriceData

    with Session(engine) as session:
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
            return Decimal(str(price))

    return None


def check_portfolio_baseline(portfolio_id: int, portfolio_name: str) -> dict:
    """
    Check if a portfolio has baseline drift.

    Returns dict with:
    - has_drift: bool
    - drift_pct: float
    - expected_nav: float
    - stored_nav: float
    """
    from sqlmodel import Session, select
    from AlphaMachine_core.db import engine
    from AlphaMachine_core.tracking.models import (
        PortfolioHolding,
        PortfolioDailyNAV,
        Variants,
    )

    result = {
        "portfolio": portfolio_name,
        "portfolio_id": portfolio_id,
        "has_drift": False,
        "drift_pct": 0.0,
        "expected_nav": None,
        "stored_nav": None,
        "check_date": None,
    }

    with Session(engine) as session:
        # Get latest NAV date
        latest_nav_stmt = (
            select(PortfolioDailyNAV)
            .where(PortfolioDailyNAV.portfolio_id == portfolio_id)
            .where(PortfolioDailyNAV.variant == Variants.RAW)
            .order_by(PortfolioDailyNAV.trade_date.desc())
            .limit(1)
        )
        latest_nav = session.exec(latest_nav_stmt).first()

        if not latest_nav:
            return result

        check_date = latest_nav.trade_date
        result["check_date"] = check_date
        result["stored_nav"] = float(latest_nav.nav)

        # Get holdings for that date
        holdings_stmt = (
            select(PortfolioHolding)
            .where(PortfolioHolding.portfolio_id == portfolio_id)
            .where(PortfolioHolding.effective_date <= check_date)
            .order_by(PortfolioHolding.effective_date.desc())
        )
        all_holdings = session.exec(holdings_stmt).all()

        if not all_holdings:
            return result

        # Get latest effective date
        latest_eff_date = max(h.effective_date for h in all_holdings)
        holdings = [h for h in all_holdings if h.effective_date == latest_eff_date]

        # Calculate expected NAV from shares × price
        expected_nav = Decimal("0")
        for h in holdings:
            if h.shares:
                price = get_price_for_ticker_date(h.ticker, check_date)
                if price:
                    expected_nav += h.shares * price

        if expected_nav > 0:
            result["expected_nav"] = float(expected_nav)
            drift_pct = ((float(expected_nav) / result["stored_nav"]) - 1) * 100
            result["drift_pct"] = drift_pct
            result["has_drift"] = abs(drift_pct) > 0.1  # 0.1% threshold

    return result


def clear_portfolio_shares(portfolio_id: int, dry_run: bool = False) -> int:
    """Clear shares and entry_price for all holdings of a portfolio."""
    from sqlmodel import Session, select
    from AlphaMachine_core.db import engine
    from AlphaMachine_core.tracking.models import PortfolioHolding

    with Session(engine) as session:
        stmt = select(PortfolioHolding).where(
            PortfolioHolding.portfolio_id == portfolio_id
        )
        holdings = session.exec(stmt).all()

        count = 0
        for h in holdings:
            if h.shares is not None or h.entry_price is not None:
                if not dry_run:
                    h.shares = None
                    h.entry_price = None
                count += 1

        if not dry_run:
            session.commit()

        return count


def recalculate_portfolio_shares(
    portfolio_id: int, portfolio_name: str, dry_run: bool = False
) -> dict:
    """Recalculate shares for all holdings from scratch."""
    from sqlmodel import Session, select
    from AlphaMachine_core.db import engine
    from AlphaMachine_core.tracking.models import PortfolioHolding, PortfolioDailyNAV, Variants
    from AlphaMachine_core.tracking import calculate_shares_from_weights

    stats = {
        "portfolio": portfolio_name,
        "effective_dates": 0,
        "holdings_updated": 0,
        "errors": [],
    }

    with Session(engine) as session:
        # Get all distinct effective_dates
        stmt = (
            select(PortfolioHolding.effective_date)
            .where(PortfolioHolding.portfolio_id == portfolio_id)
            .distinct()
            .order_by(PortfolioHolding.effective_date)
        )
        effective_dates = session.exec(stmt).all()

        if not effective_dates:
            return stats

        stats["effective_dates"] = len(effective_dates)

        for eff_date in effective_dates:
            # Get holdings for this date
            holdings_stmt = (
                select(PortfolioHolding)
                .where(PortfolioHolding.portfolio_id == portfolio_id)
                .where(PortfolioHolding.effective_date == eff_date)
            )
            holdings = session.exec(holdings_stmt).all()

            # Get NAV from previous day (or 100 for first day)
            nav_stmt = (
                select(PortfolioDailyNAV.nav)
                .where(PortfolioDailyNAV.portfolio_id == portfolio_id)
                .where(PortfolioDailyNAV.trade_date < eff_date)
                .where(PortfolioDailyNAV.variant == Variants.RAW)
                .order_by(PortfolioDailyNAV.trade_date.desc())
                .limit(1)
            )
            nav_result = session.exec(nav_stmt).first()
            nav = Decimal(str(nav_result)) if nav_result else Decimal("100")

            # Get prices for holdings
            tickers = [h.ticker for h in holdings]
            prices = {}
            for ticker in tickers:
                price = get_price_for_ticker_date(ticker, eff_date)
                if price:
                    prices[ticker] = price

            if not prices:
                stats["errors"].append(f"No prices for {eff_date}")
                continue

            # Build holdings list with weights
            holdings_with_weights = [
                {"ticker": h.ticker, "weight": float(h.weight) if h.weight else 0}
                for h in holdings
            ]

            # Calculate shares
            holdings_with_shares = calculate_shares_from_weights(
                holdings_with_weights, nav, prices
            )

            # Update holdings
            for h_orig, h_new in zip(holdings, holdings_with_shares):
                if h_new.get("shares") is not None:
                    if not dry_run:
                        h_orig.shares = h_new["shares"]
                        h_orig.entry_price = h_new.get("entry_price")
                    stats["holdings_updated"] += 1
                    logger.debug(
                        f"  {eff_date} {h_orig.ticker}: "
                        f"shares={h_new['shares']:.6f}, entry_price={h_new.get('entry_price')}"
                    )

        if not dry_run:
            session.commit()

    return stats


def fix_portfolio(
    portfolio, tracker, dry_run: bool = False
) -> dict:
    """Fix a single portfolio's baseline."""
    from datetime import date

    logger.info(f"\n{'='*60}")
    logger.info(f"Fixing portfolio: {portfolio.name}")
    logger.info(f"{'='*60}")

    # Step 1: Check current state
    baseline_check = check_portfolio_baseline(portfolio.id, portfolio.name)
    logger.info(f"Current state:")
    logger.info(f"  Stored NAV: {baseline_check['stored_nav']:.2f}")
    logger.info(f"  Expected NAV: {baseline_check['expected_nav']:.2f}")
    logger.info(f"  Drift: {baseline_check['drift_pct']:.2f}%")

    if abs(baseline_check["drift_pct"]) < 0.1:
        logger.info("No significant drift detected - skipping")
        return {"portfolio": portfolio.name, "action": "skipped", "reason": "no drift"}

    # Step 2: Clear existing shares
    logger.info(f"\nStep 1: Clearing existing shares/entry_price...")
    cleared = clear_portfolio_shares(portfolio.id, dry_run)
    logger.info(f"  Cleared {cleared} holdings")

    # Step 3: Recalculate shares
    logger.info(f"\nStep 2: Recalculating shares with correct prices...")
    shares_stats = recalculate_portfolio_shares(
        portfolio.id, portfolio.name, dry_run
    )
    logger.info(f"  Updated {shares_stats['holdings_updated']} holdings")

    # Step 4: Recalculate NAV
    logger.info(f"\nStep 3: Recalculating NAV...")
    if not dry_run:
        # Delete existing NAV records
        from sqlmodel import Session, select
        from AlphaMachine_core.db import engine
        from AlphaMachine_core.tracking.models import PortfolioDailyNAV

        with Session(engine) as session:
            stmt = select(PortfolioDailyNAV).where(
                PortfolioDailyNAV.portfolio_id == portfolio.id
            )
            nav_records = session.exec(stmt).all()
            for record in nav_records:
                session.delete(record)
            session.commit()
            logger.info(f"  Deleted {len(nav_records)} NAV records")

        # Run NAV recalculation
        from scripts.recalculate_nav import recalculate_portfolio_nav

        nav_stats = recalculate_portfolio_nav(
            tracker,
            portfolio,
            portfolio.start_date,
            date.today(),
            dry_run=False,
        )
        logger.info(f"  Calculated {nav_stats['nav_calculated']} NAV records")

        # Recompute metrics
        logger.info(f"\nStep 4: Recomputing metrics...")
        metrics = tracker.compute_and_store_metrics(portfolio.id)
        logger.info(f"  Computed {len(metrics)} metrics")
    else:
        logger.info("  [DRY RUN] Would recalculate NAV from start date")

    # Step 5: Verify fix
    logger.info(f"\nVerifying fix...")
    new_baseline_check = check_portfolio_baseline(portfolio.id, portfolio.name)
    logger.info(f"  New stored NAV: {new_baseline_check['stored_nav']:.2f}")
    logger.info(f"  Expected NAV: {new_baseline_check['expected_nav']:.2f}")
    logger.info(f"  New drift: {new_baseline_check['drift_pct']:.2f}%")

    return {
        "portfolio": portfolio.name,
        "action": "fixed" if not dry_run else "dry_run",
        "before_drift": baseline_check["drift_pct"],
        "after_drift": new_baseline_check["drift_pct"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fix NAV baseline drift in portfolios"
    )
    parser.add_argument(
        "--portfolio",
        type=str,
        help="Specific portfolio name to fix",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check for drift, don't fix",
    )
    parser.add_argument(
        "--fix-all",
        action="store_true",
        help="Fix all portfolios with drift above threshold",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Drift threshold percentage for --fix-all (default: 0.5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying database",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Fix NAV Baseline")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be saved")

    from AlphaMachine_core.tracking import get_tracker

    tracker = get_tracker()

    # Get portfolios
    portfolios = tracker.list_portfolios(active_only=False)

    if args.check_only:
        logger.info("\nChecking all portfolios for baseline drift...")
        logger.info("-" * 60)

        issues = []
        for portfolio in portfolios:
            check = check_portfolio_baseline(portfolio.id, portfolio.name)
            if check["stored_nav"] and check["expected_nav"]:
                status = "DRIFT" if abs(check["drift_pct"]) > 0.1 else "OK"
                logger.info(
                    f"  {portfolio.name}: {status} "
                    f"(stored={check['stored_nav']:.2f}, "
                    f"expected={check['expected_nav']:.2f}, "
                    f"drift={check['drift_pct']:.2f}%)"
                )
                if abs(check["drift_pct"]) > 0.1:
                    issues.append(check)

        logger.info("-" * 60)
        logger.info(f"Found {len(issues)} portfolio(s) with baseline drift")
        return

    # Fix specific portfolio or all
    if args.portfolio:
        portfolio = tracker.get_portfolio_by_name(args.portfolio)
        if not portfolio:
            logger.error(f"Portfolio not found: {args.portfolio}")
            sys.exit(1)
        portfolios_to_fix = [portfolio]
    elif args.fix_all:
        # Find portfolios with drift above threshold
        portfolios_to_fix = []
        for portfolio in portfolios:
            check = check_portfolio_baseline(portfolio.id, portfolio.name)
            if abs(check.get("drift_pct", 0)) >= args.threshold:
                portfolios_to_fix.append(portfolio)
        logger.info(
            f"Found {len(portfolios_to_fix)} portfolio(s) with drift >= {args.threshold}%"
        )
    else:
        logger.error("Specify --portfolio, --fix-all, or --check-only")
        sys.exit(1)

    # Fix each portfolio
    results = []
    for portfolio in portfolios_to_fix:
        try:
            result = fix_portfolio(portfolio, tracker, dry_run=args.dry_run)
            results.append(result)
        except Exception as e:
            logger.error(f"Error fixing {portfolio.name}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "portfolio": portfolio.name,
                "action": "error",
                "error": str(e),
            })

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Fix Summary")
    logger.info("=" * 60)

    for result in results:
        if result.get("action") == "error":
            logger.info(f"  {result['portfolio']}: ERROR - {result['error']}")
        elif result.get("action") == "skipped":
            logger.info(f"  {result['portfolio']}: SKIPPED - {result['reason']}")
        else:
            logger.info(
                f"  {result['portfolio']}: {result['action']} "
                f"(drift: {result['before_drift']:.2f}% -> {result['after_drift']:.2f}%)"
            )


if __name__ == "__main__":
    main()
