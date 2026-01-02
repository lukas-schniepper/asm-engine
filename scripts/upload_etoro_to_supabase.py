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

    Returns False for weekends and major US market holidays.
    """
    # Check weekends
    if check_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False

    # Check major US market holidays (fixed dates)
    # Note: Some holidays are observed on different days if they fall on weekend
    month, day = check_date.month, check_date.day
    year = check_date.year

    # New Year's Day (Jan 1, or observed on Jan 2 if Jan 1 is Sunday)
    if month == 1 and day == 1:
        return False
    if month == 1 and day == 2 and date(year, 1, 1).weekday() == 6:  # Sunday
        return False

    # Independence Day (July 4, or observed on July 3/5)
    if month == 7 and day == 4:
        return False
    if month == 7 and day == 3 and date(year, 7, 4).weekday() == 5:  # Saturday
        return False
    if month == 7 and day == 5 and date(year, 7, 4).weekday() == 6:  # Sunday
        return False

    # Christmas (Dec 25, or observed on Dec 24/26)
    if month == 12 and day == 25:
        return False
    if month == 12 and day == 24 and date(year, 12, 25).weekday() == 5:  # Saturday
        return False
    if month == 12 and day == 26 and date(year, 12, 25).weekday() == 6:  # Sunday
        return False

    # Thanksgiving (4th Thursday of November)
    if month == 11:
        # Find 4th Thursday
        first_day = date(year, 11, 1)
        first_thursday = (3 - first_day.weekday()) % 7 + 1
        thanksgiving = first_thursday + 21  # 4th Thursday
        if day == thanksgiving:
            return False

    return True


def get_last_trading_day(reference_date: date = None) -> date:
    """
    Get the most recent trading day on or before the reference date.

    If reference_date is a weekend, returns the previous Friday.
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


def get_previous_trading_day() -> date:
    """
    Get the most recent COMPLETED trading day.

    The scraper runs at 07:00 UTC, before US market open (14:30 UTC).
    So eToro shows the previous trading day's final data:
    - Friday 07:00 UTC → Thursday's close
    - Saturday 07:00 UTC → Friday's close
    - Monday 07:00 UTC → Friday's close (weekend)
    - Tuesday 07:00 UTC → Monday's close
    """
    yesterday = date.today() - timedelta(days=1)
    return get_last_trading_day(yesterday)

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
    from sqlalchemy.orm.attributes import flag_modified

    # Use the previous completed trading day
    # (Scraper runs at 07:00 UTC, before market open, so shows yesterday's close)
    trading_date = get_previous_trading_day()
    print(f"  Using trading date: {trading_date} ({trading_date.strftime('%A')})")
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

            # Check if record already exists for this trading date
            existing = session.exec(
                select(EToroStats).where(
                    EToroStats.username == username,
                    EToroStats.scraped_date == trading_date
                )
            ).first()

            # Get previous record to preserve December during year transition
            prev_record = session.exec(
                select(EToroStats)
                .where(EToroStats.username == username)
                .where(EToroStats.scraped_date < trading_date)
                .order_by(EToroStats.scraped_date.desc())
                .limit(1)
            ).first()

            # Handle monthly_returns - preserve previous year's values during year transition
            # (eToro resets ALL previous year's monthly values in January)
            new_monthly = inv.get('monthly_returns', {})
            today = date.today()

            # Year transition: today is January but trading_date is still December
            # eToro shows 2026 data (YTD reset, all 2025 months = 0%) even when querying Dec 31
            is_year_transition = (today.month == 1 and trading_date.month == 12)

            if is_year_transition and prev_record:
                old_monthly = prev_record.monthly_returns or {}
                prev_year = trading_date.year  # e.g., 2025

                # Preserve ALL months from the previous year (Jan-Dec)
                preserved_months = []
                for month in range(1, 13):
                    month_key = f"{prev_year}-{month:02d}"  # e.g., "2025-01", "2025-12"
                    if month_key in old_monthly and old_monthly[month_key] != 0.0:
                        scraped_val = new_monthly.get(month_key, 0.0)
                        stored_val = old_monthly[month_key]
                        # Preserve if scraped value is near zero OR significantly different
                        if abs(scraped_val) < 0.5 or abs(scraped_val - stored_val) > abs(stored_val) * 0.5:
                            new_monthly[month_key] = stored_val
                            preserved_months.append(f"{month_key}={stored_val}%")

                if preserved_months:
                    print(f"    Year transition fix: preserving {len(preserved_months)} months from {prev_year}")

                # Also preserve YTD if existing record has it (eToro resets YTD in January)
                if existing and existing.gain_ytd != 0.0:
                    print(f"    Year transition fix: preserving YTD = {existing.gain_ytd}% (scraped {inv.get('gain_ytd', 0.0)}%)")
                    inv['gain_ytd'] = existing.gain_ytd

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
                existing.monthly_returns = dict(new_monthly)  # Create new dict to trigger change detection
                flag_modified(existing, 'monthly_returns')  # Force SQLAlchemy to detect JSON change
                session.add(existing)
                print(f"  Updated existing record for {username}")
            else:
                # Create new record
                stats = EToroStats(
                    scraped_date=trading_date,
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
                    monthly_returns=new_monthly,
                )
                session.add(stats)
                print(f"  Created new record for {username}")

    print("\n" + "=" * 60)
    print(f"Upload complete! Saved {len(investors)} investors for {trading_date}")
    print("=" * 60)

    return True


if __name__ == '__main__':
    success = upload_to_supabase()
    exit(0 if success else 1)
