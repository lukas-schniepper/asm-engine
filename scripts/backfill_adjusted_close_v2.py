#!/usr/bin/env python3
"""
Backfill adjusted_close column for all existing price data.
Uses direct EODHD API calls to avoid encoding issues.
"""

import os
import sys
import time

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

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
from AlphaMachine_core.data_sources.eodhd_http_client import EODHDHttpClient

def get_all_tickers():
    """Get all unique tickers from ticker_period table."""
    with get_session() as session:
        results = session.exec(select(TickerPeriod.ticker).distinct()).all()
        return sorted([str(t) for t in results if t])

def fetch_and_store_ticker(client, ticker, history_start, today):
    """Fetch data for a single ticker and return PriceData objects."""
    try:
        raw = client.get_eod_data(
            ticker=ticker,
            start_date=history_start,
            end_date=(today + timedelta(days=1)).strftime('%Y-%m-%d')
        )

        if raw.empty:
            return ticker, [], "No data"

        # Prepare the data
        df = raw.reset_index()
        date_col = 'Date' if 'Date' in df.columns else 'index'

        price_objects = []
        for _, r in df.iterrows():
            try:
                if pd.isna(r['Close']) or pd.isna(r['Volume']):
                    continue

                price_objects.append(PriceData(
                    ticker=ticker,
                    trade_date=pd.to_datetime(r[date_col]).date(),
                    open=float(r['Open']),
                    high=float(r['High']),
                    low=float(r['Low']),
                    close=float(r['Close']),
                    adjusted_close=float(r['Adjusted_Close']) if 'Adjusted_Close' in r and pd.notna(r.get('Adjusted_Close')) else None,
                    volume=int(r['Volume'])
                ))
            except Exception:
                pass

        return ticker, price_objects, f"OK ({len(price_objects)} rows)"
    except Exception as e:
        return ticker, [], f"Error: {str(e)[:50]}"

def backfill_with_adjusted_close(tickers=None, history_start='2019-01-01', max_workers=10):
    """
    Re-fetch all price data with adjusted_close.
    """
    print("=" * 60)
    print("BACKFILL ADJUSTED CLOSE PRICES (V2)")
    print("=" * 60)

    # Initialize EODHD client
    api_key = os.getenv('EODHD_API_KEY')
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get('EODHD_API_KEY')
        except:
            pass

    if not api_key:
        print("ERROR: EODHD_API_KEY not found!")
        return

    client = EODHDHttpClient(api_key)
    print("EODHD client initialized")

    # Get tickers
    if tickers is None:
        tickers = get_all_tickers()

    today = date.today()
    print(f"Tickers to backfill: {len(tickers)}")
    print(f"History start: {history_start}")
    print(f"Max workers: {max_workers}")

    # Check current state
    with engine.connect() as conn:
        current_count = conn.execute(text("SELECT COUNT(*) FROM price_data")).scalar()
        print(f"Current price_data rows: {current_count}")

    # Fetch data in parallel
    print(f"\nFetching data for {len(tickers)} tickers...")
    start_time = time.time()

    all_price_objects = []
    results_summary = {'success': 0, 'failed': 0, 'no_data': 0}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_and_store_ticker, client, ticker, history_start, today): ticker
            for ticker in tickers
        }

        completed = 0
        for future in as_completed(futures):
            completed += 1
            ticker, price_objects, status = future.result()

            if price_objects:
                all_price_objects.extend(price_objects)
                results_summary['success'] += 1
            elif "No data" in status:
                results_summary['no_data'] += 1
            else:
                results_summary['failed'] += 1

            # Progress update every 50 tickers
            if completed % 50 == 0:
                elapsed = time.time() - start_time
                print(f"  Progress: {completed}/{len(tickers)} ({elapsed:.1f}s) - {len(all_price_objects)} rows collected")

    fetch_time = time.time() - start_time
    print(f"\nFetch completed in {fetch_time:.1f}s")
    print(f"  Success: {results_summary['success']}")
    print(f"  No data: {results_summary['no_data']}")
    print(f"  Failed: {results_summary['failed']}")
    print(f"  Total rows: {len(all_price_objects)}")

    # Save to database
    if all_price_objects:
        print(f"\nSaving {len(all_price_objects)} rows to database...")
        save_start = time.time()

        # Insert in batches of 10000
        batch_size = 10000
        with get_session() as session:
            for i in range(0, len(all_price_objects), batch_size):
                batch = all_price_objects[i:i+batch_size]
                session.add_all(batch)
                session.commit()
                print(f"  Saved batch {i//batch_size + 1}: {len(batch)} rows")

        save_time = time.time() - save_start
        print(f"Database save completed in {save_time:.1f}s")

    # Verify
    print("\nVerifying adjusted_close data...")
    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM price_data")).scalar()
        with_adj = conn.execute(text("SELECT COUNT(*) FROM price_data WHERE adjusted_close IS NOT NULL")).scalar()
        print(f"Total price records: {total}")
        print(f"Records with adjusted_close: {with_adj}")
        if total > 0:
            print(f"Coverage: {with_adj/total*100:.1f}%")

        # Sample data
        print("\nSample data (AAPL last 5 days):")
        sample = conn.execute(text("""
            SELECT ticker, date, close, adjusted_close
            FROM price_data
            WHERE ticker = 'AAPL'
            ORDER BY date DESC
            LIMIT 5
        """))
        for row in sample:
            adj = f"{row[3]:.2f}" if row[3] else "NULL"
            print(f"  {row[1]}: close={row[2]:.2f}, adj_close={adj}")

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.1f}s")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backfill adjusted_close prices (V2)")
    parser.add_argument("--history-start", default="2019-01-01", help="Start date for history")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers (default: all)")
    parser.add_argument("--max-workers", type=int, default=10, help="Parallel workers")
    args = parser.parse_args()

    backfill_with_adjusted_close(
        tickers=args.tickers,
        history_start=args.history_start,
        max_workers=args.max_workers
    )
