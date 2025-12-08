#!/usr/bin/env python3
"""
Backfill adjusted_close column for all existing price data.

Strategy: Delete existing data and re-fetch with adjusted_close.
This ensures data integrity and is faster than updating each row.
"""

import os
import sys
from pathlib import Path
from datetime import date, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load secrets
secrets_path = project_root / '.streamlit' / 'secrets.toml'
if secrets_path.exists():
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    with open(secrets_path, 'rb') as f:
        secrets = tomllib.load(f)
    for key, value in secrets.items():
        if key not in os.environ and isinstance(value, str):
            os.environ[key] = value

import pandas as pd
from sqlalchemy import text
from sqlmodel import select
from AlphaMachine_core.db import engine, get_session
from AlphaMachine_core.models import PriceData, TickerPeriod
from AlphaMachine_core.data_manager import StockDataManager

def get_all_tickers():
    """Get all unique tickers from ticker_period table."""
    with get_session() as session:
        results = session.exec(select(TickerPeriod.ticker).distinct()).all()
        return sorted([str(t) for t in results if t])

def clear_price_data():
    """Clear all existing price data to allow fresh re-fetch with adjusted_close."""
    print("Clearing existing price_data table...")
    with engine.connect() as conn:
        result = conn.execute(text("DELETE FROM price_data"))
        conn.commit()
        print(f"Deleted {result.rowcount} rows from price_data")

def backfill_with_adjusted_close(tickers=None, history_start='2019-01-01'):
    """
    Re-fetch all price data with adjusted_close.

    Args:
        tickers: List of tickers to backfill (None = all)
        history_start: Start date for historical data
    """
    print("\n" + "="*60)
    print("BACKFILL ADJUSTED CLOSE PRICES")
    print("="*60)

    # Initialize data manager
    dm = StockDataManager()

    # Get tickers
    if tickers is None:
        tickers = get_all_tickers()

    print(f"Tickers to backfill: {len(tickers)}")
    print(f"History start: {history_start}")

    # Clear existing data
    clear_price_data()

    # Re-fetch with adjusted_close
    print(f"\nFetching data for {len(tickers)} tickers...")
    result = dm.update_ticker_data(
        tickers=tickers,
        history_start=history_start,
        max_workers=10
    )

    print("\n" + "="*60)
    print("BACKFILL COMPLETE")
    print("="*60)
    print(f"Updated: {len(result['updated'])} tickers")
    print(f"Skipped: {len(result['skipped'])} tickers")

    # Verify adjusted_close populated
    print("\nVerifying adjusted_close data...")
    with engine.connect() as conn:
        # Count total rows
        total = conn.execute(text("SELECT COUNT(*) FROM price_data")).scalar()
        # Count rows with adjusted_close
        with_adj = conn.execute(text("SELECT COUNT(*) FROM price_data WHERE adjusted_close IS NOT NULL")).scalar()
        print(f"Total price records: {total}")
        print(f"Records with adjusted_close: {with_adj}")
        print(f"Coverage: {with_adj/total*100:.1f}%" if total > 0 else "N/A")

        # Sample data
        print("\nSample data (AAPL last 5 days):")
        sample = conn.execute(text("""
            SELECT ticker, date, close, adjusted_close,
                   ROUND((adjusted_close - close) / close * 100, 4) as diff_pct
            FROM price_data
            WHERE ticker = 'AAPL'
            ORDER BY date DESC
            LIMIT 5
        """))
        for row in sample:
            print(f"  {row[1]}: close={row[2]:.2f}, adj_close={row[3]:.2f if row[3] else 'NULL'}, diff={row[4]}%")

    return result

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backfill adjusted_close prices")
    parser.add_argument("--history-start", default="2019-01-01", help="Start date for history")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers (default: all)")
    args = parser.parse_args()

    backfill_with_adjusted_close(
        tickers=args.tickers,
        history_start=args.history_start
    )
