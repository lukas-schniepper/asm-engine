#!/usr/bin/env python3
"""
Scheduled Price Update Script for GitHub Actions
Updates all ticker prices in the database automatically
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path to import AlphaMachine modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from AlphaMachine_core.data_manager import DataManager
from AlphaMachine_core.models import TickerPeriod
from AlphaMachine_core.database import get_session
from sqlmodel import select


def main():
    """Run scheduled price update for all tickers in database"""

    print(f"\n{'='*80}")
    print(f"🕐 Scheduled Price Update - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    # Verify environment variables are set
    database_url = os.getenv('DATABASE_URL')
    eodhd_key = os.getenv('EODHD_API_KEY')

    if not database_url:
        print("❌ ERROR: DATABASE_URL environment variable not set")
        sys.exit(1)

    if not eodhd_key:
        print("❌ ERROR: EODHD_API_KEY environment variable not set")
        sys.exit(1)

    print("✅ Environment variables verified")
    print(f"   DATABASE_URL: {database_url[:30]}...{database_url[-20:]}")
    print(f"   EODHD_API_KEY: {eodhd_key[:10]}...{eodhd_key[-4:]}")

    try:
        # Initialize DataManager
        print("\n📊 Initializing DataManager...")
        dm = DataManager()
        print("✅ DataManager initialized successfully")

        # Get all tickers from database
        print("\n🔍 Fetching tickers from database...")
        with get_session() as session:
            results = session.exec(select(TickerPeriod.ticker).distinct()).all()
            tickers = sorted(list(set(str(t) for t in results if t)))

        if not tickers:
            print("⚠️  No tickers found in database")
            return

        print(f"✅ Found {len(tickers)} unique tickers")
        print(f"   Sample: {', '.join(tickers[:5])}...")

        # Update all tickers
        print(f"\n🔄 Starting update for {len(tickers)} tickers...")
        print(f"⏱️  Estimated duration: ~{len(tickers) * 0.5 / 60:.1f} minutes")
        print(f"{'='*80}\n")

        result = dm.update_ticker_data(tickers=None)  # None = update all in DB

        # Print results
        print(f"\n{'='*80}")
        print("📊 Update Summary")
        print(f"{'='*80}")
        print(f"✅ Updated: {len(result.get('updated', []))} tickers")
        print(f"⏭️  Skipped: {len(result.get('skipped', []))} tickers")
        print(f"📈 Total:   {result.get('total', 0)} tickers")

        # Show sample of updated tickers
        if result.get('updated'):
            sample_size = min(5, len(result['updated']))
            print(f"\nSample updated tickers:")
            for ticker in result['updated'][:sample_size]:
                detail = result.get('details', {}).get(ticker, {})
                if detail.get('saved'):
                    print(f"  ✅ {ticker}: {detail['saved']}")
                else:
                    print(f"  ✅ {ticker}")

        # Show sample of skipped tickers
        if result.get('skipped'):
            sample_size = min(5, len(result['skipped']))
            print(f"\nSample skipped tickers:")
            for ticker in result['skipped'][:sample_size]:
                detail = result.get('details', {}).get(ticker, {})
                reason = detail.get('reason', 'Unknown reason')
                print(f"  ⏭️  {ticker}: {reason}")

        print(f"\n{'='*80}")
        print(f"✅ Scheduled update completed successfully!")
        print(f"{'='*80}\n")

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"❌ ERROR during scheduled update:")
        print(f"{'='*80}")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
