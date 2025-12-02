#!/usr/bin/env python3
"""
Import Existing Portfolios from ticker_period table and S3 allocation history.

This script:
1. Discovers portfolio sources from ticker_period table
2. Registers them as trackable portfolios
3. Imports monthly holdings from ticker_period
4. Backfills NAV history from S3 allocation_history.csv

Usage:
    python scripts/import_existing_portfolios.py [--dry-run]
"""

import os
import sys
from pathlib import Path
from datetime import date, timedelta
from decimal import Decimal
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load DATABASE_URL
import toml
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    secrets_path = project_root / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        secrets = toml.load(secrets_path)
        DATABASE_URL = secrets.get("DATABASE_URL")

if not DATABASE_URL:
    print("ERROR: DATABASE_URL not found")
    sys.exit(1)

os.environ["DATABASE_URL"] = DATABASE_URL

# Set AWS credentials
os.environ['AWS_SHARED_CREDENTIALS_FILE'] = str(Path.home() / 'codebase' / '.aws' / 'credentials')
os.environ['AWS_CONFIG_FILE'] = str(Path.home() / 'codebase' / '.aws' / 'config')

import pandas as pd
import numpy as np
from sqlmodel import Session, select, func
from sqlalchemy import create_engine, text

# Create engine
engine = create_engine(DATABASE_URL, echo=False)

# Import models
from AlphaMachine_core.models import TickerPeriod, PriceData
from AlphaMachine_core.tracking.models import (
    PortfolioDefinition,
    PortfolioHolding,
    PortfolioDailyNAV,
    Variants,
)


def get_portfolio_sources() -> list[str]:
    """Get all unique portfolio sources from ticker_period table."""
    with Session(engine) as session:
        results = session.exec(
            select(TickerPeriod.source).distinct()
        ).all()
        return [str(r) for r in results if r]


def get_source_date_range(source: str) -> tuple[date, date]:
    """Get the date range for a portfolio source."""
    with Session(engine) as session:
        min_date = session.exec(
            select(func.min(TickerPeriod.start_date))
            .where(TickerPeriod.source == source)
        ).first()
        max_date = session.exec(
            select(func.max(TickerPeriod.end_date))
            .where(TickerPeriod.source == source)
        ).first()
        return min_date, max_date


def get_monthly_holdings(source: str, month: str) -> list[dict]:
    """
    Get holdings for a specific month and source.

    Args:
        source: Portfolio source name
        month: Month in YYYY-MM format

    Returns:
        List of dicts with ticker info
    """
    with Session(engine) as session:
        results = session.exec(
            select(TickerPeriod)
            .where(TickerPeriod.source == source)
            .where(func.to_char(TickerPeriod.start_date, 'YYYY-MM') == month)
        ).all()

        holdings = []
        for r in results:
            holdings.append({
                "ticker": r.ticker,
                "start_date": r.start_date,
                "end_date": r.end_date,
            })
        return holdings


def get_all_months_for_source(source: str) -> list[str]:
    """Get all months that have data for a source."""
    with Session(engine) as session:
        results = session.exec(
            select(func.to_char(TickerPeriod.start_date, 'YYYY-MM').label('month'))
            .where(TickerPeriod.source == source)
            .distinct()
            .order_by(text("month"))
        ).all()
        return [str(r) for r in results if r]


def calculate_equal_weight_nav(
    tickers: list[str],
    start_date: date,
    end_date: date,
    initial_nav: float = 100.0,
) -> pd.Series:
    """
    Calculate NAV for an equal-weighted portfolio.

    Returns:
        Series with daily NAV values
    """
    # Load prices for tickers
    with Session(engine) as session:
        results = session.exec(
            select(PriceData)
            .where(PriceData.ticker.in_([t.upper() for t in tickers]))
            .where(PriceData.trade_date >= start_date)
            .where(PriceData.trade_date <= end_date)
            .order_by(PriceData.trade_date)
        ).all()

    if not results:
        return pd.Series(dtype=float)

    # Build price DataFrame
    price_data = []
    for r in results:
        price_data.append({
            "date": r.trade_date,
            "ticker": r.ticker,
            "close": r.close,
        })

    df = pd.DataFrame(price_data)
    if df.empty:
        return pd.Series(dtype=float)

    # Pivot to get prices per ticker
    price_df = df.pivot(index="date", columns="ticker", values="close")
    price_df = price_df.ffill()  # Forward fill missing

    # Calculate returns
    returns = price_df.pct_change()

    # Equal weight
    n_stocks = len(price_df.columns)
    weight = 1.0 / n_stocks if n_stocks > 0 else 0

    # Portfolio return = average of stock returns
    portfolio_returns = returns.mean(axis=1)

    # Calculate NAV
    nav = initial_nav * (1 + portfolio_returns).cumprod()
    nav.iloc[0] = initial_nav

    return nav


def register_portfolio(source: str, dry_run: bool = False) -> int:
    """
    Register a portfolio source as a trackable portfolio.

    Returns:
        Portfolio ID
    """
    min_date, max_date = get_source_date_range(source)

    portfolio_name = f"{source}_EqualWeight"

    with Session(engine) as session:
        # Check if already exists
        existing = session.exec(
            select(PortfolioDefinition)
            .where(PortfolioDefinition.name == portfolio_name)
        ).first()

        if existing:
            print(f"  Portfolio '{portfolio_name}' already exists (id={existing.id})")
            return existing.id

        if dry_run:
            print(f"  [DRY RUN] Would create portfolio '{portfolio_name}'")
            return -1

        # Create portfolio
        portfolio = PortfolioDefinition(
            name=portfolio_name,
            description=f"Equal-weighted portfolio from {source} source",
            config={
                "source": source,
                "weighting": "equal",
                "dynamic_monthly": True,
            },
            source=source,
            start_date=min_date,
            is_active=True,
        )
        session.add(portfolio)
        session.commit()
        session.refresh(portfolio)

        print(f"  Created portfolio '{portfolio_name}' (id={portfolio.id})")
        return portfolio.id


def import_monthly_holdings_for_portfolio(
    portfolio_id: int,
    source: str,
    dry_run: bool = False,
) -> int:
    """
    Import all monthly holdings for a portfolio.

    Returns:
        Number of holding records created
    """
    months = get_all_months_for_source(source)
    total_holdings = 0

    for month in months:
        holdings = get_monthly_holdings(source, month)
        if not holdings:
            continue

        # Use first day of month as effective date
        year, mon = month.split("-")
        effective_date = date(int(year), int(mon), 1)

        # Equal weight
        weight = Decimal("1") / Decimal(str(len(holdings))) if holdings else Decimal("0")

        if dry_run:
            print(f"    [DRY RUN] {month}: {len(holdings)} tickers")
            total_holdings += len(holdings)
            continue

        with Session(engine) as session:
            for h in holdings:
                # Check if exists
                existing = session.exec(
                    select(PortfolioHolding)
                    .where(PortfolioHolding.portfolio_id == portfolio_id)
                    .where(PortfolioHolding.effective_date == effective_date)
                    .where(PortfolioHolding.ticker == h["ticker"])
                ).first()

                if existing:
                    continue

                holding = PortfolioHolding(
                    portfolio_id=portfolio_id,
                    effective_date=effective_date,
                    ticker=h["ticker"],
                    weight=weight,
                )
                session.add(holding)
                total_holdings += 1

            session.commit()

    return total_holdings


def backfill_nav_from_prices(
    portfolio_id: int,
    source: str,
    dry_run: bool = False,
) -> int:
    """
    Backfill NAV by calculating equal-weighted returns from price data.

    Returns:
        Number of NAV records created
    """
    months = get_all_months_for_source(source)
    if not months:
        return 0

    all_nav_records = []
    current_nav = 100.0

    for i, month in enumerate(months):
        holdings = get_monthly_holdings(source, month)
        if not holdings:
            continue

        tickers = [h["ticker"] for h in holdings]

        # Get month date range
        year, mon = month.split("-")
        month_start = date(int(year), int(mon), 1)

        # Calculate month end
        if int(mon) == 12:
            month_end = date(int(year) + 1, 1, 1) - timedelta(days=1)
        else:
            month_end = date(int(year), int(mon) + 1, 1) - timedelta(days=1)

        # Calculate NAV for this month
        nav_series = calculate_equal_weight_nav(
            tickers, month_start, month_end, initial_nav=current_nav
        )

        if nav_series.empty:
            continue

        # Update current NAV for next month
        current_nav = float(nav_series.iloc[-1])

        # Store daily NAV records
        for trade_date, nav_value in nav_series.items():
            all_nav_records.append({
                "portfolio_id": portfolio_id,
                "trade_date": trade_date,
                "nav": nav_value,
            })

    if dry_run:
        print(f"    [DRY RUN] Would create {len(all_nav_records)} NAV records")
        return len(all_nav_records)

    # Calculate returns and insert
    nav_count = 0
    initial_nav = 100.0

    with Session(engine) as session:
        prev_nav = None

        for record in all_nav_records:
            nav_value = record["nav"]
            trade_date = record["trade_date"]

            # Calculate returns
            if prev_nav is not None and prev_nav > 0:
                daily_return = (nav_value / prev_nav) - 1
            else:
                daily_return = 0.0

            cumulative_return = (nav_value / initial_nav) - 1

            # Check if exists
            existing = session.exec(
                select(PortfolioDailyNAV)
                .where(PortfolioDailyNAV.portfolio_id == portfolio_id)
                .where(PortfolioDailyNAV.trade_date == trade_date)
                .where(PortfolioDailyNAV.variant == Variants.RAW)
            ).first()

            if existing:
                prev_nav = nav_value
                continue

            nav_record = PortfolioDailyNAV(
                portfolio_id=portfolio_id,
                trade_date=trade_date,
                variant=Variants.RAW,
                nav=Decimal(str(round(nav_value, 4))),
                daily_return=Decimal(str(round(daily_return, 6))),
                cumulative_return=Decimal(str(round(cumulative_return, 6))),
                equity_allocation=Decimal("1.0"),
                cash_allocation=Decimal("0.0"),
            )
            session.add(nav_record)
            nav_count += 1
            prev_nav = nav_value

        session.commit()

    return nav_count


def main():
    parser = argparse.ArgumentParser(description="Import existing portfolios")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--source", type=str, help="Only import specific source")
    args = parser.parse_args()

    print("=" * 60)
    print("Import Existing Portfolios")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN MODE - No changes will be made]\n")

    # Get all sources
    sources = get_portfolio_sources()
    print(f"\nFound {len(sources)} portfolio sources:")
    for s in sources:
        min_d, max_d = get_source_date_range(s)
        months = get_all_months_for_source(s)
        print(f"  - {s}: {min_d} to {max_d} ({len(months)} months)")

    # Filter if specific source requested
    if args.source:
        sources = [s for s in sources if s == args.source]
        if not sources:
            print(f"\nSource '{args.source}' not found!")
            return 1

    print("\n" + "-" * 60)

    for source in sources:
        print(f"\n[{source}]")

        # Register portfolio
        portfolio_id = register_portfolio(source, args.dry_run)
        if portfolio_id < 0 and args.dry_run:
            portfolio_id = 999  # Placeholder for dry run

        # Import holdings
        print(f"  Importing monthly holdings...")
        holdings_count = import_monthly_holdings_for_portfolio(
            portfolio_id, source, args.dry_run
        )
        print(f"    Created {holdings_count} holding records")

        # Backfill NAV
        print(f"  Backfilling NAV from price data...")
        nav_count = backfill_nav_from_prices(portfolio_id, source, args.dry_run)
        print(f"    Created {nav_count} NAV records")

    print("\n" + "=" * 60)
    print("Import complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
