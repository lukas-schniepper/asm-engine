#!/usr/bin/env python3
"""
Recalculate Overlay NAV Script.

Recalculates overlay variant NAVs (conservative, trend_regime_v2) based on raw NAV.
This fixes the bug where overlay NAVs started at different values than raw NAV.

The correct methodology:
- Day 1: overlay_nav = raw_nav (all variants start at same value)
- Day 2+: overlay_nav = prev_overlay_nav * (1 + raw_daily_return * allocation)

Usage:
    # Recalculate all portfolios
    python scripts/recalculate_overlay_nav.py

    # Specific portfolio
    python scripts/recalculate_overlay_nav.py --portfolio "SA Large Caps_EqualWeight"

    # Dry run
    python scripts/recalculate_overlay_nav.py --dry-run
"""

import os
import sys
import logging
import argparse
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

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


def recalculate_overlay_navs(
    portfolio_id: int,
    portfolio_name: str,
    dry_run: bool = False,
) -> dict:
    """
    Recalculate overlay NAVs for a portfolio using raw NAV as base.

    Methodology:
    - Day 1: overlay_nav = raw_nav
    - Day 2+: overlay_nav = prev_overlay_nav * (1 + raw_daily_return * allocation)

    Returns:
        Dict with statistics
    """
    from sqlmodel import Session, select
    from AlphaMachine_core.db import engine
    from AlphaMachine_core.tracking.models import PortfolioDailyNAV, Variants
    from AlphaMachine_core.tracking.overlay_adapter import OverlayAdapter, OVERLAY_REGISTRY

    stats = {
        "portfolio": portfolio_name,
        "raw_records": 0,
        "overlay_variants": {},
    }

    logger.info(f"\nRecalculating overlay NAVs for: {portfolio_name} (id={portfolio_id})")

    with Session(engine) as session:
        # Get all raw NAV records sorted by date
        raw_navs = session.exec(
            select(PortfolioDailyNAV)
            .where(PortfolioDailyNAV.portfolio_id == portfolio_id)
            .where(PortfolioDailyNAV.variant == Variants.RAW)
            .order_by(PortfolioDailyNAV.trade_date)
        ).all()

        if not raw_navs:
            logger.warning(f"No raw NAV records found for {portfolio_name}")
            return stats

        stats["raw_records"] = len(raw_navs)
        logger.info(f"  Found {len(raw_navs)} raw NAV records")
        logger.info(f"  Date range: {raw_navs[0].trade_date} to {raw_navs[-1].trade_date}")

        # Get initial NAV for cumulative return calculations
        initial_nav = float(raw_navs[0].nav)
        logger.info(f"  Initial NAV: {initial_nav:.4f}")

        # Create overlay adapter for getting allocations
        overlay_adapter = OverlayAdapter()

        # Process each overlay model
        for model_name in OVERLAY_REGISTRY.keys():
            logger.info(f"\n  Processing {model_name}...")

            # Delete existing overlay NAV records
            if not dry_run:
                existing = session.exec(
                    select(PortfolioDailyNAV)
                    .where(PortfolioDailyNAV.portfolio_id == portfolio_id)
                    .where(PortfolioDailyNAV.variant == model_name)
                ).all()
                deleted_count = len(existing)
                for record in existing:
                    session.delete(record)
                session.commit()
                logger.info(f"    Deleted {deleted_count} existing records")

            # Track overlay NAV
            prev_overlay_nav = None
            prev_raw_nav = None
            records_created = 0

            for raw_nav_record in raw_navs:
                trade_date = raw_nav_record.trade_date
                raw_nav = float(raw_nav_record.nav)

                # Get allocation from overlay model
                try:
                    _, allocation, signals, impacts = overlay_adapter.apply_overlay(
                        model=model_name,
                        raw_nav=raw_nav,
                        trade_date=trade_date,
                    )
                except Exception as e:
                    logger.warning(f"    {trade_date}: Error getting allocation - {e}")
                    allocation = 0.5  # Default to 50% if error

                if prev_overlay_nav is None:
                    # Day 1: overlay NAV equals raw NAV
                    overlay_nav = raw_nav
                    overlay_daily_return = 0.0
                else:
                    # Day 2+: apply allocation-scaled return
                    if prev_raw_nav and prev_raw_nav > 0:
                        raw_daily_return = (raw_nav / prev_raw_nav) - 1
                    else:
                        raw_daily_return = 0.0

                    overlay_daily_return = raw_daily_return * allocation
                    overlay_nav = prev_overlay_nav * (1 + overlay_daily_return)

                # Calculate cumulative return
                overlay_cumulative = (overlay_nav / initial_nav) - 1 if initial_nav > 0 else 0.0

                # Create NAV record
                if not dry_run:
                    nav_record = PortfolioDailyNAV(
                        portfolio_id=portfolio_id,
                        trade_date=trade_date,
                        variant=model_name,
                        nav=Decimal(str(overlay_nav)),
                        daily_return=Decimal(str(overlay_daily_return)),
                        cumulative_return=Decimal(str(overlay_cumulative)),
                        equity_allocation=Decimal(str(allocation)),
                        cash_allocation=Decimal(str(1 - allocation)),
                    )
                    session.add(nav_record)

                records_created += 1
                prev_overlay_nav = overlay_nav
                prev_raw_nav = raw_nav

            if not dry_run:
                session.commit()

            stats["overlay_variants"][model_name] = {
                "records_created": records_created,
                "final_nav": prev_overlay_nav,
            }

            logger.info(
                f"    Created {records_created} records, "
                f"final NAV: {prev_overlay_nav:.4f}"
            )

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Recalculate overlay NAVs based on raw NAV"
    )
    parser.add_argument(
        "--portfolio",
        type=str,
        help="Specific portfolio name to recalculate",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying database",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Recalculate Overlay NAVs")
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
        portfolios = tracker.list_portfolios(active_only=True)

    logger.info(f"Processing {len(portfolios)} portfolio(s)")

    # Process each portfolio
    all_stats = []
    for portfolio in portfolios:
        try:
            stats = recalculate_overlay_navs(
                portfolio.id, portfolio.name, dry_run=args.dry_run
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

    for stats in all_stats:
        if "error" in stats:
            logger.info(f"  {stats['portfolio']}: ERROR - {stats['error']}")
        else:
            logger.info(f"  {stats['portfolio']}:")
            logger.info(f"    Raw records: {stats.get('raw_records', 0)}")
            for variant, v_stats in stats.get("overlay_variants", {}).items():
                logger.info(
                    f"    {variant}: {v_stats['records_created']} records, "
                    f"final NAV={v_stats['final_nav']:.4f}"
                )

    if args.dry_run:
        logger.info("\nDRY RUN complete - run without --dry-run to apply changes")

    # Recompute metrics after recalculation
    if not args.dry_run:
        logger.info("\nRecomputing metrics for updated portfolios...")
        for stats in all_stats:
            if "error" not in stats and stats.get("raw_records", 0) > 0:
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
