#!/usr/bin/env python3
"""
Apply Overlay Allocations to All Portfolio NAV Data.

This script takes the existing RAW NAV for each portfolio and calculates
the overlay-adjusted NAV using the historical allocation signals from
the Conservative and TrendRegimeV2 models.

The overlay allocation determines what percentage of the portfolio is
invested in equities vs cash. For example:
- RAW NAV = 100% equity
- Conservative at 62.6% allocation = 62.6% equity, 37.4% cash

Usage:
    python scripts/apply_overlays_to_portfolios.py [--dry-run] [--portfolio-id ID]
"""

import os
import sys
from pathlib import Path
from datetime import date
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

import pandas as pd
import numpy as np
from sqlmodel import Session, select, func
from sqlalchemy import create_engine

# Create engine
engine = create_engine(DATABASE_URL, echo=False)

# Import models
from AlphaMachine_core.tracking.models import (
    PortfolioDefinition,
    PortfolioDailyNAV,
    OverlaySignal,
    Variants,
)


def get_overlay_allocations(model: str) -> pd.DataFrame:
    """
    Get historical overlay allocations from the overlay_signals table.

    Returns:
        DataFrame with trade_date index and allocation column
    """
    with Session(engine) as session:
        results = session.exec(
            select(OverlaySignal)
            .where(OverlaySignal.model == model)
            .order_by(OverlaySignal.trade_date)
        ).all()

        if not results:
            return pd.DataFrame()

        data = [
            {
                "trade_date": r.trade_date,
                "allocation": float(r.actual_allocation) if r.actual_allocation else 1.0,
            }
            for r in results
        ]

        df = pd.DataFrame(data)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.set_index("trade_date")
        return df


def get_raw_nav_series(portfolio_id: int) -> pd.DataFrame:
    """
    Get RAW NAV series for a portfolio.

    Returns:
        DataFrame with trade_date index and nav, daily_return columns
    """
    with Session(engine) as session:
        results = session.exec(
            select(PortfolioDailyNAV)
            .where(PortfolioDailyNAV.portfolio_id == portfolio_id)
            .where(PortfolioDailyNAV.variant == Variants.RAW)
            .order_by(PortfolioDailyNAV.trade_date)
        ).all()

        if not results:
            return pd.DataFrame()

        data = [
            {
                "trade_date": r.trade_date,
                "nav": float(r.nav),
                "daily_return": float(r.daily_return) if r.daily_return else None,
            }
            for r in results
        ]

        df = pd.DataFrame(data)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.set_index("trade_date")
        return df


def calculate_overlay_nav(
    raw_nav_df: pd.DataFrame,
    allocation_df: pd.DataFrame,
    initial_nav: float = 100.0,
) -> pd.DataFrame:
    """
    Calculate overlay-adjusted NAV based on allocation signals.

    The overlay allocation determines what % is in equity vs cash.
    When allocation is less than 100%, returns are scaled proportionally.

    Args:
        raw_nav_df: DataFrame with RAW NAV (nav, daily_return columns)
        allocation_df: DataFrame with allocation column (0-1)
        initial_nav: Starting NAV value

    Returns:
        DataFrame with overlay NAV, returns, and allocation
    """
    if raw_nav_df.empty or allocation_df.empty:
        return pd.DataFrame()

    # Merge on date, forward-fill allocation
    merged = raw_nav_df.join(allocation_df, how="left")
    merged["allocation"] = merged["allocation"].ffill().fillna(1.0)

    # Calculate daily returns from RAW NAV if not provided
    if "daily_return" not in merged.columns or merged["daily_return"].isna().all():
        merged["daily_return"] = merged["nav"].pct_change()

    # Calculate overlay return: allocation * raw_return
    # (When allocation < 1, excess goes to cash which we assume has 0% return)
    merged["overlay_return"] = merged["allocation"] * merged["daily_return"].fillna(0)

    # Calculate overlay NAV from returns
    merged["overlay_nav"] = initial_nav * (1 + merged["overlay_return"]).cumprod()
    merged.loc[merged.index[0], "overlay_nav"] = initial_nav

    # Calculate cumulative return
    merged["cumulative_return"] = merged["overlay_nav"] / initial_nav - 1

    return merged[["overlay_nav", "overlay_return", "cumulative_return", "allocation"]]


def apply_overlay_to_portfolio(
    portfolio_id: int,
    model: str,
    variant: str,
    dry_run: bool = False,
) -> int:
    """
    Apply overlay allocation to a portfolio's NAV.

    Returns:
        Number of NAV records created/updated
    """
    # Get RAW NAV
    raw_nav_df = get_raw_nav_series(portfolio_id)
    if raw_nav_df.empty:
        return 0

    # Get overlay allocations
    allocation_df = get_overlay_allocations(model)
    if allocation_df.empty:
        print(f"    No allocation data for {model}")
        return 0

    # Calculate overlay NAV
    overlay_df = calculate_overlay_nav(raw_nav_df, allocation_df)
    if overlay_df.empty:
        return 0

    if dry_run:
        print(f"    [DRY RUN] Would create {len(overlay_df)} {variant} NAV records")
        return len(overlay_df)

    nav_count = 0

    with Session(engine) as session:
        for trade_date, row in overlay_df.iterrows():
            trade_date_val = trade_date.date() if hasattr(trade_date, "date") else trade_date

            # Check if exists
            existing = session.exec(
                select(PortfolioDailyNAV)
                .where(PortfolioDailyNAV.portfolio_id == portfolio_id)
                .where(PortfolioDailyNAV.trade_date == trade_date_val)
                .where(PortfolioDailyNAV.variant == variant)
            ).first()

            if existing:
                # Update existing record
                existing.nav = Decimal(str(round(row["overlay_nav"], 4)))
                existing.daily_return = Decimal(str(round(row["overlay_return"], 6)))
                existing.cumulative_return = Decimal(str(round(row["cumulative_return"], 6)))
                existing.equity_allocation = Decimal(str(round(row["allocation"], 4)))
                existing.cash_allocation = Decimal(str(round(1.0 - row["allocation"], 4)))
                session.add(existing)
            else:
                # Create new record
                nav_record = PortfolioDailyNAV(
                    portfolio_id=portfolio_id,
                    trade_date=trade_date_val,
                    variant=variant,
                    nav=Decimal(str(round(row["overlay_nav"], 4))),
                    daily_return=Decimal(str(round(row["overlay_return"], 6))),
                    cumulative_return=Decimal(str(round(row["cumulative_return"], 6))),
                    equity_allocation=Decimal(str(round(row["allocation"], 4))),
                    cash_allocation=Decimal(str(round(1.0 - row["allocation"], 4))),
                )
                session.add(nav_record)

            nav_count += 1

            # Commit in batches
            if nav_count % 500 == 0:
                session.commit()

        session.commit()

    return nav_count


def main():
    parser = argparse.ArgumentParser(description="Apply overlays to portfolio NAV")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--portfolio-id", type=int, help="Only process specific portfolio")
    args = parser.parse_args()

    print("=" * 60)
    print("Apply Overlay Allocations to Portfolios")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN MODE - No changes will be made]\n")

    # Get all portfolios
    with Session(engine) as session:
        if args.portfolio_id:
            portfolios = [session.get(PortfolioDefinition, args.portfolio_id)]
            portfolios = [p for p in portfolios if p]
        else:
            portfolios = session.exec(
                select(PortfolioDefinition)
                .where(PortfolioDefinition.is_active == True)
            ).all()

    print(f"Found {len(portfolios)} active portfolios\n")

    # Skip SPY_Overlay_Backtest since it already has overlay data from S3
    portfolios = [p for p in portfolios if p.name != "SPY_Overlay_Backtest"]

    overlay_models = [
        ("conservative", Variants.CONSERVATIVE),
        ("trend_regime_v2", Variants.TREND_REGIME_V2),
    ]

    total_records = 0

    for portfolio in portfolios:
        print(f"\n[{portfolio.name}] (id={portfolio.id})")

        for model, variant in overlay_models:
            print(f"  Applying {model} overlay...")
            count = apply_overlay_to_portfolio(
                portfolio.id, model, variant, args.dry_run
            )
            print(f"    Created/updated {count} {variant} NAV records")
            total_records += count

    print("\n" + "=" * 60)
    print(f"Complete! Processed {total_records} total NAV records")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
