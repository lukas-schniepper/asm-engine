#!/usr/bin/env python3
"""
Scheduled Price Update Script for GitHub Actions
Updates all ticker prices in the database automatically
"""

import os
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

# Add parent directory to path to import AlphaMachine modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from AlphaMachine_core.data_manager import StockDataManager
from AlphaMachine_core.models import TickerPeriod, PriceData
from AlphaMachine_core.db import get_session
from AlphaMachine_core.tracking.data_quality import (
    get_portfolio_critical_tickers,
    validate_portfolio_prices,
)
from sqlmodel import select, func


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
        # Initialize StockDataManager
        print("\n📊 Initializing StockDataManager...")
        dm = StockDataManager()
        print("✅ StockDataManager initialized successfully")

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

        # =====================================================================
        # PORTFOLIO-CRITICAL PRICE VALIDATION
        # =====================================================================
        # Validate that all tickers held in active portfolios have prices
        # for the most recent trading day. This prevents NAV calculation failures.
        print(f"\n{'='*80}")
        print("🔍 Validating Portfolio-Critical Prices")
        print(f"{'='*80}")

        # Determine the most recent trading day with price data
        # (This is the date we should validate, not "yesterday" which might be a weekend)
        with get_session() as session:
            latest_date = session.exec(
                select(func.max(PriceData.trade_date))
            ).first()

        if latest_date:
            print(f"📅 Latest price date in database: {latest_date}")

            # Get portfolio-critical tickers info
            critical_info = get_portfolio_critical_tickers()
            total_portfolios = len(critical_info)
            total_tickers = sum(len(info["tickers"]) for info in critical_info.values())

            print(f"📊 Active portfolios: {total_portfolios}")
            print(f"🎯 Portfolio-critical tickers: {total_tickers}")

            # Validate prices for the latest date
            validation = validate_portfolio_prices(latest_date)

            if validation["all_valid"]:
                print(f"✅ All {total_tickers} portfolio-critical tickers have prices for {latest_date}")
            else:
                print(f"\n{'='*80}")
                print("❌ CRITICAL: Missing prices for portfolio holdings!")
                print(f"{'='*80}")

                # Group missing by portfolio for cleaner output
                missing_by_portfolio = {}
                for m in validation["missing"]:
                    pname = m["portfolio_name"]
                    if pname not in missing_by_portfolio:
                        missing_by_portfolio[pname] = []
                    missing_by_portfolio[pname].append(m["ticker"])

                for pname, tickers in missing_by_portfolio.items():
                    print(f"\n  📁 {pname}:")
                    for ticker in tickers:
                        print(f"      ❌ {ticker}")

                print(f"\n{'='*80}")
                print(f"⚠️  {len(validation['missing'])} portfolio-critical tickers are missing prices!")
                print("These portfolios will fail NAV calculation until prices are available.")
                print(f"{'='*80}\n")

                # Exit with error to fail the workflow
                sys.exit(1)
        else:
            print("⚠️  No price data in database - skipping validation")

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
