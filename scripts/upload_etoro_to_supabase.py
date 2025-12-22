#!/usr/bin/env python3
"""
Upload eToro scraped data to Supabase

Saves each investor's stats to the etoro_stats table with:
- scraped_date: Last trading day (not the scrape date, to avoid weekend shifts)
- All stats including monthly_returns as JSON

This allows:
- Live data access without redeploying
- Historical tracking over time
- SQL queries for analysis
"""
import json
import sys
from datetime import date, timedelta
from pathlib import Path


def is_trading_day(check_date: date) -> bool:
    """
    Check if a given date is a US stock market trading day.

    Returns False for weekends (Saturday, Sunday).
    Note: Does not check for US market holidays.
    """
    if check_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    return True


def get_last_trading_day(reference_date: date = None) -> date:
    """
    Get the most recent trading day on or before the reference date.

    If reference_date is a weekend, returns the previous Friday.
    This ensures eToro data scraped on Saturday is labeled with Friday's date.
    """
    if reference_date is None:
        reference_date = date.today()

    check_date = reference_date
    max_lookback = 10  # Don't look back more than 10 days

    for _ in range(max_lookback):
        if is_trading_day(check_date):
            return check_date
        check_date -= timedelta(days=1)

    # Fallback - shouldn't happen
    return reference_date

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Local data file (created by update_etoro_data.py)
DATA_FILE = project_root / "data" / "etoro_scraped_data.json"


def upload_to_supabase():
    """Upload eToro data to Supabase database."""
    print("=" * 60)
    print("Uploading eToro data to Supabase")
    print("=" * 60)

    # Check if data file exists
    if not DATA_FILE.exists():
        print(f"ERROR: Data file not found: {DATA_FILE}")
        print("Run update_etoro_data.py first to scrape data.")
        return False

    # Load the scraped data
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded data from {DATA_FILE}")
    print(f"  Scraped at: {data.get('scraped_at', 'unknown')}")
    print(f"  Investors: {len(data.get('investors', []))}")

    # Import database components
    from AlphaMachine_core.db import get_session
    from AlphaMachine_core.models import EToroStats
    from sqlmodel import select

    # Use last trading day to avoid weekend date shifts
    # (Saturday scrape should be labeled as Friday's data)
    today = get_last_trading_day()
    print(f"  Using trading date: {today} ({today.strftime('%A')})")
    investors = data.get('investors', [])

    if not investors:
        print("ERROR: No investors in data file")
        return False

    with get_session() as session:
        for inv in investors:
            username = inv.get('username', '').lower()
            if not username:
                continue

            print(f"\nProcessing {username}...")

            # Check if record already exists for today
            existing = session.exec(
                select(EToroStats).where(
                    EToroStats.username == username,
                    EToroStats.scraped_date == today
                )
            ).first()

            if existing:
                # Update existing record
                existing.full_name = inv.get('full_name', username)
                existing.user_id = inv.get('user_id')
                existing.risk_score = inv.get('risk_score', 5)
                existing.copiers = inv.get('copiers', 0)
                existing.gain_1y = inv.get('gain_1y', 0.0)
                existing.gain_2y = inv.get('gain_2y', 0.0)
                existing.gain_ytd = inv.get('gain_ytd', 0.0)
                existing.win_ratio = inv.get('win_ratio', 50.0)
                existing.profitable_months_pct = inv.get('profitable_months_pct', 50.0)
                existing.monthly_returns = inv.get('monthly_returns', {})
                session.add(existing)
                print(f"  Updated existing record for {username}")
            else:
                # Create new record
                stats = EToroStats(
                    scraped_date=today,
                    username=username,
                    full_name=inv.get('full_name', username),
                    user_id=inv.get('user_id'),
                    risk_score=inv.get('risk_score', 5),
                    copiers=inv.get('copiers', 0),
                    gain_1y=inv.get('gain_1y', 0.0),
                    gain_2y=inv.get('gain_2y', 0.0),
                    gain_ytd=inv.get('gain_ytd', 0.0),
                    win_ratio=inv.get('win_ratio', 50.0),
                    profitable_months_pct=inv.get('profitable_months_pct', 50.0),
                    monthly_returns=inv.get('monthly_returns', {}),
                )
                session.add(stats)
                print(f"  Created new record for {username}")

    print("\n" + "=" * 60)
    print(f"Upload complete! Saved {len(investors)} investors for {today}")
    print("=" * 60)

    return True


if __name__ == '__main__':
    success = upload_to_supabase()
    exit(0 if success else 1)
