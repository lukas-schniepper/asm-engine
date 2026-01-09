#!/usr/bin/env python3
"""
Migration script: Fix eToro scraped_date to use actual trading date.

Problem: The scraper runs at 07:00 UTC (before market open), so it captures
the previous trading day's data but was labeling it with the scrape date.

This script:
1. Shifts all scraped_date values to the previous trading day
2. Removes duplicates (e.g., Sat/Sun/Mon scrapes all become Friday)

Run this ONCE after deploying the upload_etoro_to_supabase.py fix.
"""
import sys
from datetime import date, timedelta
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import shared trading calendar utilities (now includes proper US holiday handling)
from utils.trading_calendar import is_trading_day, get_previous_trading_day


def migrate_etoro_dates(dry_run: bool = True):
    """Migrate eToro dates to use previous trading day."""
    from AlphaMachine_core.db import get_session
    from AlphaMachine_core.models import EToroStats
    from sqlmodel import select

    print("=" * 60)
    print("eToro Date Migration")
    print("=" * 60)
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'LIVE (will modify database)'}")
    print()

    with get_session() as session:
        # Get all records
        all_stats = session.exec(select(EToroStats).order_by(EToroStats.scraped_date)).all()
        print(f"Found {len(all_stats)} total records")

        # Group by username to track changes
        changes_by_user = defaultdict(list)

        for stat in all_stats:
            old_date = stat.scraped_date
            new_date = get_previous_trading_day(old_date)

            if old_date != new_date:
                changes_by_user[stat.username].append({
                    'id': stat.id,
                    'old_date': old_date,
                    'new_date': new_date,
                    'stat': stat,
                })

        # Show sample changes
        print(f"\nRecords to change: {sum(len(v) for v in changes_by_user.values())}")
        print("\nSample changes (first 10):")
        print(f"{'Old Date':<12} {'Old Day':<10} {'New Date':<12} {'New Day':<10} {'User':<15}")
        print("-" * 65)

        sample_count = 0
        for username, changes in changes_by_user.items():
            for change in changes:
                if sample_count < 10:
                    print(f"{change['old_date']}   {change['old_date'].strftime('%A'):<10} "
                          f"{change['new_date']}   {change['new_date'].strftime('%A'):<10} {username}")
                    sample_count += 1

        # Find duplicates that will be created
        print("\n" + "=" * 60)
        print("Checking for duplicates after migration...")

        # Group by (username, new_date) to find duplicates
        new_date_groups = defaultdict(list)
        for username, changes in changes_by_user.items():
            for change in changes:
                key = (username, change['new_date'])
                new_date_groups[key].append(change)

        duplicates = {k: v for k, v in new_date_groups.items() if len(v) > 1}

        if duplicates:
            print(f"\nFound {len(duplicates)} (username, date) pairs with duplicates")
            print("Will keep the record with the LATEST original scraped_date\n")

            print("Example duplicates:")
            for (username, new_date), records in list(duplicates.items())[:5]:
                print(f"  {username} on {new_date}:")
                for r in records:
                    print(f"    - Original: {r['old_date']} ({r['old_date'].strftime('%A')})")
        else:
            print("No duplicates found.")

        if dry_run:
            print("\n" + "=" * 60)
            print("DRY RUN complete. No changes made.")
            print("Run with --live to apply changes.")
            return

        # Apply changes
        print("\n" + "=" * 60)
        print("Applying changes...")

        records_updated = 0
        records_deleted = 0

        for (username, new_date), records in new_date_groups.items():
            if len(records) == 1:
                # No duplicate, just update the date
                stat = records[0]['stat']
                stat.scraped_date = new_date
                session.add(stat)
                records_updated += 1
            else:
                # Multiple records for same date - keep the one with latest original date
                records.sort(key=lambda x: x['old_date'], reverse=True)

                # Keep the first (latest original date), update its date
                keeper = records[0]['stat']
                keeper.scraped_date = new_date
                session.add(keeper)
                records_updated += 1

                # Delete the rest
                for record in records[1:]:
                    session.delete(record['stat'])
                    records_deleted += 1

        # Commit is handled by get_session context manager
        print(f"\nRecords updated: {records_updated}")
        print(f"Duplicates deleted: {records_deleted}")
        print("\nMigration complete!")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Migrate eToro dates to previous trading day")
    parser.add_argument('--live', action='store_true',
                        help='Apply changes (default is dry run)')
    args = parser.parse_args()

    migrate_etoro_dates(dry_run=not args.live)


if __name__ == '__main__':
    main()
