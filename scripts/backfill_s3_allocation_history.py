#!/usr/bin/env python3
"""
Backfill NAV History from S3 Allocation History.

This script imports the historical allocation_history.csv from S3
which contains daily strategy performance since 2019 for overlay models
(Conservative and TrendRegimeV2).

The S3 allocation_history.csv contains:
- date: Trading date
- allocation: Current equity allocation (0-1)
- strategy_return: Daily return of the overlay strategy
- spy_return: Daily SPY return
- target_allocation: Target allocation from signals
- trade_executed: Whether a trade occurred

Usage:
    python scripts/backfill_s3_allocation_history.py [--dry-run]
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

# Set AWS credentials
os.environ['AWS_SHARED_CREDENTIALS_FILE'] = str(Path.home() / 'codebase' / '.aws' / 'credentials')
os.environ['AWS_CONFIG_FILE'] = str(Path.home() / 'codebase' / '.aws' / 'config')

import pandas as pd
import numpy as np
from sqlmodel import Session, select
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
from AlphaMachine_core.tracking.s3_adapter import S3DataLoader


def get_or_create_spy_portfolio(session: Session, dry_run: bool = False) -> int:
    """Get or create the SPY tracking portfolio for overlay backtests."""
    portfolio_name = "SPY_Overlay_Backtest"

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

    portfolio = PortfolioDefinition(
        name=portfolio_name,
        description="SPY with overlay models - historical backtest from S3",
        config={
            "tickers": ["SPY"],
            "source": "s3_allocation_history",
            "overlay_models": ["conservative", "trend_regime_v2"],
        },
        source="s3_backtest",
        start_date=date(2019, 1, 2),
        is_active=True,
    )
    session.add(portfolio)
    session.commit()
    session.refresh(portfolio)

    print(f"  Created portfolio '{portfolio_name}' (id={portfolio.id})")
    return portfolio.id


def load_allocation_history(model: str) -> pd.DataFrame:
    """Load allocation history from S3 for a model."""
    loader = S3DataLoader()

    if not loader.is_connected:
        raise RuntimeError("S3 not connected - cannot load allocation history")

    df = loader.load_allocation_history(model)
    return df


def backfill_nav_from_allocation_history(
    portfolio_id: int,
    model: str,
    variant: Variants,
    dry_run: bool = False,
) -> int:
    """
    Backfill NAV from S3 allocation_history.csv.

    The allocation history contains strategy_return which is the daily
    return of the overlay strategy. We can reconstruct NAV from this.

    Returns:
        Number of NAV records created
    """
    print(f"  Loading {model} allocation history from S3...")
    df = load_allocation_history(model)

    if df.empty:
        print(f"    No data found for {model}")
        return 0

    print(f"    Loaded {len(df)} records from {df['date'].min()} to {df['date'].max()}")

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    # Calculate NAV from strategy returns
    initial_nav = 100.0
    df["nav"] = initial_nav * (1 + df["strategy_return"].fillna(0)).cumprod()

    # Calculate cumulative return
    df["cumulative_return"] = df["nav"] / initial_nav - 1

    if dry_run:
        print(f"    [DRY RUN] Would create {len(df)} NAV records")
        print(f"    Sample data:")
        print(f"      First date: {df.iloc[0]['date']}, NAV: {df.iloc[0]['nav']:.2f}")
        print(f"      Last date: {df.iloc[-1]['date']}, NAV: {df.iloc[-1]['nav']:.2f}")
        print(f"      Total return: {df.iloc[-1]['cumulative_return']:.2%}")
        return len(df)

    nav_count = 0

    with Session(engine) as session:
        for _, row in df.iterrows():
            trade_date = row["date"]
            if isinstance(trade_date, str):
                trade_date = pd.to_datetime(trade_date).date()
            elif hasattr(trade_date, "date"):
                trade_date = trade_date.date()

            # Check if exists
            existing = session.exec(
                select(PortfolioDailyNAV)
                .where(PortfolioDailyNAV.portfolio_id == portfolio_id)
                .where(PortfolioDailyNAV.trade_date == trade_date)
                .where(PortfolioDailyNAV.variant == variant)
            ).first()

            if existing:
                continue

            # Get allocation values
            allocation = float(row.get("allocation", 1.0))
            if pd.isna(allocation):
                allocation = 1.0

            nav_record = PortfolioDailyNAV(
                portfolio_id=portfolio_id,
                trade_date=trade_date,
                variant=variant,
                nav=Decimal(str(round(row["nav"], 4))),
                daily_return=Decimal(str(round(float(row.get("strategy_return", 0) or 0), 6))),
                cumulative_return=Decimal(str(round(row["cumulative_return"], 6))),
                equity_allocation=Decimal(str(round(allocation, 4))),
                cash_allocation=Decimal(str(round(1.0 - allocation, 4))),
            )
            session.add(nav_record)
            nav_count += 1

            # Commit in batches
            if nav_count % 500 == 0:
                session.commit()
                print(f"      Committed {nav_count} records...")

        session.commit()

    return nav_count


def backfill_overlay_signals(
    model: str,
    dry_run: bool = False,
) -> int:
    """
    Backfill overlay signals from S3 allocation_history.csv.

    The allocation history contains allocation and signal data that we
    can use to populate the overlay_signals table.

    Returns:
        Number of signal records created
    """
    print(f"  Loading {model} allocation history for signals...")
    df = load_allocation_history(model)

    if df.empty:
        print(f"    No data found for {model}")
        return 0

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    if dry_run:
        print(f"    [DRY RUN] Would create {len(df)} signal records for {model}")
        return len(df)

    signal_count = 0

    with Session(engine) as session:
        prev_allocation = None

        for _, row in df.iterrows():
            trade_date = row["date"]
            if isinstance(trade_date, str):
                trade_date = pd.to_datetime(trade_date).date()
            elif hasattr(trade_date, "date"):
                trade_date = trade_date.date()

            # Check if exists
            existing = session.exec(
                select(OverlaySignal)
                .where(OverlaySignal.trade_date == trade_date)
                .where(OverlaySignal.model == model)
            ).first()

            if existing:
                prev_allocation = float(row.get("allocation", 1.0) or 1.0)
                continue

            # Get allocation values
            allocation = float(row.get("allocation", 1.0) or 1.0)
            target_allocation = float(row.get("target_allocation", allocation) or allocation)

            # Determine if trade was required
            if prev_allocation is not None:
                trade_required = abs(allocation - prev_allocation) > 0.05
            else:
                trade_required = bool(row.get("trade_executed", False))

            # Build signals dict from available columns
            signals = {}
            impacts = {}

            # Common signal columns in S3 data
            signal_cols = ["rsi_avg", "volatility_regime", "stress_category",
                          "momentum_strength", "spy_pct", "cash_pct"]
            for col in signal_cols:
                if col in row and pd.notna(row[col]):
                    signals[col] = float(row[col]) if isinstance(row[col], (int, float)) else str(row[col])

            signal_record = OverlaySignal(
                trade_date=trade_date,
                model=model,
                target_allocation=Decimal(str(round(target_allocation, 4))),
                actual_allocation=Decimal(str(round(allocation, 4))),
                trade_required=trade_required,
                signals=signals if signals else None,
                impacts=impacts if impacts else None,
            )
            session.add(signal_record)
            signal_count += 1
            prev_allocation = allocation

            # Commit in batches
            if signal_count % 500 == 0:
                session.commit()
                print(f"      Committed {signal_count} signal records...")

        session.commit()

    return signal_count


def backfill_spy_raw_nav(
    portfolio_id: int,
    dry_run: bool = False,
) -> int:
    """
    Backfill RAW NAV (100% SPY) using spy_return from allocation history.

    Returns:
        Number of NAV records created
    """
    print("  Loading SPY returns from conservative allocation history...")
    df = load_allocation_history("conservative")

    if df.empty:
        print("    No data found")
        return 0

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    # Calculate SPY-only NAV from spy_return
    initial_nav = 100.0
    df["nav"] = initial_nav * (1 + df["spy_return"].fillna(0)).cumprod()
    df["cumulative_return"] = df["nav"] / initial_nav - 1

    if dry_run:
        print(f"    [DRY RUN] Would create {len(df)} RAW NAV records")
        print(f"    Sample data:")
        print(f"      First date: {df.iloc[0]['date']}, NAV: {df.iloc[0]['nav']:.2f}")
        print(f"      Last date: {df.iloc[-1]['date']}, NAV: {df.iloc[-1]['nav']:.2f}")
        print(f"      SPY Total return: {df.iloc[-1]['cumulative_return']:.2%}")
        return len(df)

    nav_count = 0

    with Session(engine) as session:
        for _, row in df.iterrows():
            trade_date = row["date"]
            if isinstance(trade_date, str):
                trade_date = pd.to_datetime(trade_date).date()
            elif hasattr(trade_date, "date"):
                trade_date = trade_date.date()

            # Check if exists
            existing = session.exec(
                select(PortfolioDailyNAV)
                .where(PortfolioDailyNAV.portfolio_id == portfolio_id)
                .where(PortfolioDailyNAV.trade_date == trade_date)
                .where(PortfolioDailyNAV.variant == Variants.RAW)
            ).first()

            if existing:
                continue

            nav_record = PortfolioDailyNAV(
                portfolio_id=portfolio_id,
                trade_date=trade_date,
                variant=Variants.RAW,
                nav=Decimal(str(round(row["nav"], 4))),
                daily_return=Decimal(str(round(float(row.get("spy_return", 0) or 0), 6))),
                cumulative_return=Decimal(str(round(row["cumulative_return"], 6))),
                equity_allocation=Decimal("1.0"),
                cash_allocation=Decimal("0.0"),
            )
            session.add(nav_record)
            nav_count += 1

            if nav_count % 500 == 0:
                session.commit()
                print(f"      Committed {nav_count} records...")

        session.commit()

    return nav_count


def main():
    parser = argparse.ArgumentParser(description="Backfill NAV from S3 allocation history")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    args = parser.parse_args()

    print("=" * 60)
    print("Backfill NAV from S3 Allocation History")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN MODE - No changes will be made]\n")

    with Session(engine) as session:
        # Get or create the SPY overlay portfolio
        print("\n[1] Setting up SPY Overlay Backtest portfolio...")
        portfolio_id = get_or_create_spy_portfolio(session, args.dry_run)

        if portfolio_id < 0 and args.dry_run:
            portfolio_id = 999  # Placeholder for dry run

    # Backfill RAW (SPY only)
    print("\n[2] Backfilling RAW NAV (100% SPY)...")
    raw_count = backfill_spy_raw_nav(portfolio_id, args.dry_run)
    print(f"    Created {raw_count} RAW NAV records")

    # Backfill Conservative overlay
    print("\n[3] Backfilling Conservative overlay NAV...")
    conservative_count = backfill_nav_from_allocation_history(
        portfolio_id, "conservative", Variants.CONSERVATIVE, args.dry_run
    )
    print(f"    Created {conservative_count} Conservative NAV records")

    # Backfill TrendRegimeV2 overlay
    print("\n[4] Backfilling TrendRegimeV2 overlay NAV...")
    try:
        trend_count = backfill_nav_from_allocation_history(
            portfolio_id, "trend_regime_v2", Variants.TREND_REGIME_V2, args.dry_run
        )
        print(f"    Created {trend_count} TrendRegimeV2 NAV records")
    except Exception as e:
        print(f"    Skipped TrendRegimeV2: {e}")
        trend_count = 0

    # Backfill overlay signals
    print("\n[5] Backfilling overlay signals...")
    conservative_signals = backfill_overlay_signals("conservative", args.dry_run)
    print(f"    Created {conservative_signals} Conservative signal records")

    trend_signals = backfill_overlay_signals("trend_regime_v2", args.dry_run)
    print(f"    Created {trend_signals} TrendRegimeV2 signal records")

    print("\n" + "=" * 60)
    print("Backfill complete!")
    print(f"  RAW: {raw_count} records")
    print(f"  Conservative: {conservative_count} records")
    print(f"  TrendRegimeV2: {trend_count} records")
    print(f"  Signals: {conservative_signals + trend_signals} records")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
