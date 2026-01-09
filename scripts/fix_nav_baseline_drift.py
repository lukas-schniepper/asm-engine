#!/usr/bin/env python3
"""
Fix NAV Baseline Drift by Recalculating Shares.

Fixes NAV baseline drift by recalculating share counts so that:
    sum(shares × price) = current NAV

This is useful when:
- Shares were calculated at registration with different prices than NAV calculation
- There's drift between stored NAV and computed NAV from holdings

Usage:
    # Dry run for portfolio 26
    python scripts/fix_nav_baseline_drift.py --portfolio-id 26

    # Apply changes
    python scripts/fix_nav_baseline_drift.py --portfolio-id 26 --apply
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
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Import shared trading calendar utilities (now includes proper US holiday handling)
from utils.trading_calendar import get_last_trading_day


def get_current_nav(portfolio_id: int) -> tuple[Optional[Decimal], Optional[date]]:
    """Get the most recent NAV for a portfolio."""
    from sqlmodel import Session, select
    from AlphaMachine_core.db import engine
    from AlphaMachine_core.tracking.models import PortfolioDailyNAV, Variants

    with Session(engine) as session:
        stmt = (
            select(PortfolioDailyNAV)
            .where(PortfolioDailyNAV.portfolio_id == portfolio_id)
            .where(PortfolioDailyNAV.variant == Variants.RAW)
            .order_by(PortfolioDailyNAV.trade_date.desc())
            .limit(1)
        )
        result = session.exec(stmt).first()

        if result:
            return Decimal(str(result.nav)), result.trade_date

    return None, None


def get_current_holdings(portfolio_id: int):
    """Get the most recent holdings for a portfolio."""
    from sqlmodel import Session, select
    from AlphaMachine_core.db import engine
    from AlphaMachine_core.tracking.models import PortfolioHolding

    with Session(engine) as session:
        # Get most recent effective_date
        stmt = (
            select(PortfolioHolding.effective_date)
            .where(PortfolioHolding.portfolio_id == portfolio_id)
            .order_by(PortfolioHolding.effective_date.desc())
            .limit(1)
        )
        latest_date = session.exec(stmt).first()

        if not latest_date:
            return [], None

        # Get all holdings for that date
        stmt = (
            select(PortfolioHolding)
            .where(PortfolioHolding.portfolio_id == portfolio_id)
            .where(PortfolioHolding.effective_date == latest_date)
        )
        holdings = session.exec(stmt).all()

        return list(holdings), latest_date


def get_prices_for_tickers(tickers: list[str], trade_date: date) -> dict[str, Decimal]:
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


def fix_nav_baseline_drift(portfolio_id: int, apply: bool = False) -> dict:
    """
    Recalculate shares for a portfolio to match current NAV.

    Returns:
        Dict with before/after comparison
    """
    from sqlmodel import Session
    from AlphaMachine_core.db import engine
    from AlphaMachine_core.tracking import get_tracker

    logger.info("=" * 60)
    logger.info("Fix NAV Baseline Drift - Recalculate Shares")
    logger.info("=" * 60)

    tracker = get_tracker()

    # Get portfolio info
    portfolio = tracker.get_portfolio(portfolio_id)
    if not portfolio:
        logger.error(f"Portfolio {portfolio_id} not found")
        return {"error": "Portfolio not found"}

    logger.info(f"Portfolio: {portfolio.name} (id={portfolio_id})")

    # Get current NAV
    current_nav, nav_date = get_current_nav(portfolio_id)
    if current_nav is None:
        logger.error(f"No NAV found for portfolio {portfolio_id}")
        return {"error": "No NAV found"}

    logger.info(f"Current NAV: {current_nav:.4f} (as of {nav_date})")

    # Get current holdings
    holdings, holdings_date = get_current_holdings(portfolio_id)
    if not holdings:
        logger.error(f"No holdings found for portfolio {portfolio_id}")
        return {"error": "No holdings found"}

    logger.info(f"Holdings effective date: {holdings_date}")
    logger.info(f"Number of holdings: {len(holdings)}")

    # Get current prices
    tickers = [h.ticker for h in holdings]
    trade_date = get_last_trading_day()
    prices = get_prices_for_tickers(tickers, trade_date)

    logger.info(f"Using prices as of: {trade_date}")
    logger.info(f"Prices found: {len(prices)}/{len(tickers)}")

    if len(prices) < len(tickers):
        missing = set(tickers) - set(prices.keys())
        logger.warning(f"Missing prices for: {missing}")

    # Calculate current portfolio value from shares
    current_value = Decimal("0")
    for h in holdings:
        if h.shares and h.ticker in prices:
            current_value += h.shares * prices[h.ticker]

    logger.info("")
    logger.info("-" * 60)
    logger.info("BEFORE RECALCULATION:")
    logger.info(f"  Stored NAV:           {current_nav:.4f}")
    logger.info(f"  Computed from shares: {current_value:.4f}")
    drift = ((current_value / current_nav) - 1) * 100 if current_nav else 0
    logger.info(f"  Drift:                {drift:+.4f}%")
    logger.info("-" * 60)

    # Recalculate shares: shares = (weight × NAV) / price
    new_holdings_data = []
    total_weight = sum(float(h.weight or 0) for h in holdings)

    logger.info("")
    logger.info("Recalculating shares:")
    logger.info(f"{'Ticker':<8} {'Weight':>8} {'Old Shares':>14} {'New Shares':>14} {'Price':>10}")
    logger.info("-" * 60)

    for h in holdings:
        weight = Decimal(str(h.weight or 0))
        price = prices.get(h.ticker)

        if price and price > 0 and weight > 0:
            # Normalize weight if total != 1
            if total_weight > 0:
                normalized_weight = weight / Decimal(str(total_weight))
            else:
                normalized_weight = weight

            # shares = (weight × NAV) / price
            position_value = normalized_weight * current_nav
            new_shares = position_value / price

            old_shares = h.shares or Decimal("0")
            logger.info(
                f"{h.ticker:<8} {float(weight):>8.4f} {float(old_shares):>14.6f} "
                f"{float(new_shares):>14.6f} {float(price):>10.2f}"
            )

            new_holdings_data.append({
                "holding_id": h.id,
                "ticker": h.ticker,
                "new_shares": new_shares,
                "entry_price": price,
            })
        else:
            logger.warning(f"{h.ticker}: Cannot recalculate (weight={weight}, price={price})")

    # Calculate new portfolio value
    new_value = sum(
        d["new_shares"] * prices[d["ticker"]]
        for d in new_holdings_data
    )

    logger.info("")
    logger.info("-" * 60)
    logger.info("AFTER RECALCULATION:")
    logger.info(f"  Target NAV:           {current_nav:.4f}")
    logger.info(f"  Computed from shares: {new_value:.4f}")
    new_drift = ((new_value / current_nav) - 1) * 100 if current_nav else 0
    logger.info(f"  Drift:                {new_drift:+.6f}%")
    logger.info("-" * 60)

    if not apply:
        logger.info("")
        logger.info("DRY RUN - No changes applied.")
        logger.info("Run with --apply to save changes to database.")
        return {
            "portfolio_id": portfolio_id,
            "portfolio_name": portfolio.name,
            "nav": float(current_nav),
            "old_value": float(current_value),
            "new_value": float(new_value),
            "old_drift": float(drift),
            "new_drift": float(new_drift),
            "applied": False,
        }

    # Apply changes
    logger.info("")
    logger.info("Applying changes to database...")

    with Session(engine) as session:
        from AlphaMachine_core.tracking.models import PortfolioHolding

        for d in new_holdings_data:
            h = session.get(PortfolioHolding, d["holding_id"])
            if h:
                h.shares = d["new_shares"]
                h.entry_price = d["entry_price"]
                session.add(h)

        session.commit()

    logger.info(f"Updated {len(new_holdings_data)} holdings.")
    logger.info("")
    logger.info("=" * 60)
    logger.info("DONE - Shares recalculated to match NAV")
    logger.info("=" * 60)

    return {
        "portfolio_id": portfolio_id,
        "portfolio_name": portfolio.name,
        "nav": float(current_nav),
        "old_value": float(current_value),
        "new_value": float(new_value),
        "old_drift": float(drift),
        "new_drift": float(new_drift),
        "applied": True,
        "holdings_updated": len(new_holdings_data),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fix NAV baseline drift by recalculating portfolio shares"
    )
    parser.add_argument(
        "--portfolio-id",
        type=int,
        required=True,
        help="Portfolio ID to fix",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes (default is dry run)",
    )
    args = parser.parse_args()

    result = fix_nav_baseline_drift(args.portfolio_id, apply=args.apply)

    if "error" in result:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
