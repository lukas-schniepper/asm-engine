#!/usr/bin/env python3
"""
Portfolio Data Integrity Audit Script.

Checks all portfolios for:
1. Duplicate holdings
2. Weight consistency (should sum to 100%)
3. Zero-return anomalies
4. NAV calculation accuracy
"""

import os
import sys
from pathlib import Path
from datetime import date, timedelta
from decimal import Decimal
from collections import Counter

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

from sqlmodel import Session, select, func
from sqlalchemy import create_engine

engine = create_engine(DATABASE_URL, echo=False)

from AlphaMachine_core.tracking.models import (
    PortfolioDefinition,
    PortfolioHolding,
    PortfolioDailyNAV,
    Variants,
)
from AlphaMachine_core.models import PriceData


def audit_holdings():
    """Audit all portfolio holdings for duplicates and weight issues."""
    print("=" * 80)
    print("PHASE 1: HOLDINGS AUDIT")
    print("=" * 80)

    portfolios_with_issues = []

    with Session(engine) as session:
        portfolios = session.exec(
            select(PortfolioDefinition).where(PortfolioDefinition.is_active == True)
        ).all()

        for p in sorted(portfolios, key=lambda x: x.name):
            holdings = session.exec(
                select(PortfolioHolding).where(PortfolioHolding.portfolio_id == p.id)
            ).all()

            if not holdings:
                print(f"[SKIP] {p.name}: No holdings")
                continue

            tickers = [h.ticker for h in holdings]
            unique_tickers = set(tickers)
            total_weight = sum(float(h.weight or 0) for h in holdings)

            has_duplicates = len(holdings) != len(unique_tickers)
            weight_issue = abs(total_weight - 1.0) > 0.01  # More than 1% off

            status = "OK"
            issues = []

            if has_duplicates:
                dup_count = len(holdings) - len(unique_tickers)
                issues.append(f"{dup_count} duplicates")
                status = "ERROR"

            if weight_issue:
                issues.append(f"weight={total_weight:.2%}")
                status = "ERROR"

            if status == "ERROR":
                portfolios_with_issues.append({
                    "portfolio": p,
                    "issues": issues,
                    "holdings": holdings,
                })
                print(f"[{status}] {p.name}: {len(holdings)} holdings, {len(unique_tickers)} unique - {' | '.join(issues)}")
            else:
                print(f"[OK]   {p.name}: {len(holdings)} holdings, weight={total_weight:.2%}")

    return portfolios_with_issues


def audit_nav_returns():
    """Audit NAV data for zero-return anomalies."""
    print()
    print("=" * 80)
    print("PHASE 2: NAV RETURNS AUDIT")
    print("=" * 80)

    anomalies = []

    with Session(engine) as session:
        portfolios = session.exec(
            select(PortfolioDefinition).where(PortfolioDefinition.is_active == True)
        ).all()

        for p in sorted(portfolios, key=lambda x: x.name):
            # Get RAW NAV data
            nav_records = session.exec(
                select(PortfolioDailyNAV)
                .where(PortfolioDailyNAV.portfolio_id == p.id)
                .where(PortfolioDailyNAV.variant == Variants.RAW)
                .order_by(PortfolioDailyNAV.trade_date)
            ).all()

            if len(nav_records) < 2:
                continue

            # Check for suspicious zero returns
            zero_return_count = 0
            suspicious_zeros = []

            for i, nav in enumerate(nav_records[1:], 1):  # Skip first day
                prev_nav = nav_records[i-1]

                # Check if daily return is exactly 0 but NAV should have changed
                if nav.daily_return is not None and float(nav.daily_return) == 0:
                    # Check if this is a weekday (not first day of portfolio)
                    if nav.trade_date > p.start_date:
                        zero_return_count += 1

                        # Is NAV identical to previous? That's suspicious
                        if float(nav.nav) == float(prev_nav.nav):
                            suspicious_zeros.append(nav.trade_date)

            if suspicious_zeros:
                anomalies.append({
                    "portfolio": p,
                    "suspicious_zeros": suspicious_zeros,
                })
                print(f"[WARN] {p.name}: {len(suspicious_zeros)} suspicious 0% returns")
                for d in suspicious_zeros[:5]:  # Show first 5
                    print(f"       - {d}")
                if len(suspicious_zeros) > 5:
                    print(f"       - ... and {len(suspicious_zeros) - 5} more")
            else:
                print(f"[OK]   {p.name}: {len(nav_records)} NAV records checked")

    return anomalies


def fix_holdings(portfolios_with_issues, dry_run=True):
    """Fix portfolios with duplicate holdings."""
    print()
    print("=" * 80)
    print(f"PHASE 3: FIX HOLDINGS {'(DRY RUN)' if dry_run else ''}")
    print("=" * 80)

    fixed_portfolios = []

    with Session(engine) as session:
        for item in portfolios_with_issues:
            p = item["portfolio"]
            holdings = item["holdings"]

            # Re-fetch in this session
            holdings = session.exec(
                select(PortfolioHolding).where(PortfolioHolding.portfolio_id == p.id)
            ).all()

            # Group by ticker, keep first occurrence
            seen_tickers = {}
            to_delete = []

            for h in holdings:
                if h.ticker in seen_tickers:
                    to_delete.append(h)
                else:
                    seen_tickers[h.ticker] = h

            if dry_run:
                print(f"[DRY] {p.name}: Would remove {len(to_delete)} duplicates, keep {len(seen_tickers)}")
            else:
                print(f"[FIX] {p.name}: Removing {len(to_delete)} duplicates...")

                for h in to_delete:
                    session.delete(h)

                # Recalculate weights
                unique_count = len(seen_tickers)
                new_weight = Decimal(str(round(1.0 / unique_count, 6)))

                for ticker, h in seen_tickers.items():
                    h.weight = new_weight
                    session.add(h)

                session.commit()

                # Delete NAV data (needs recalculation)
                navs = session.exec(
                    select(PortfolioDailyNAV).where(PortfolioDailyNAV.portfolio_id == p.id)
                ).all()

                for nav in navs:
                    session.delete(nav)
                session.commit()

                fixed_portfolios.append(p)
                print(f"       Set weight to {new_weight:.4%} for {unique_count} tickers")
                print(f"       Deleted {len(navs)} NAV records (need backfill)")

    return fixed_portfolios


def validate_nav_calculation(portfolio_id, sample_size=5):
    """Validate NAV calculations against actual price data."""
    print()
    print(f"Validating NAV calculation for portfolio {portfolio_id}...")

    with Session(engine) as session:
        # Get holdings
        holdings = session.exec(
            select(PortfolioHolding).where(PortfolioHolding.portfolio_id == portfolio_id)
        ).all()

        if not holdings:
            print("  No holdings found")
            return

        tickers = [h.ticker for h in holdings]
        weights = {h.ticker: float(h.weight) for h in holdings}

        # Get NAV data
        nav_records = session.exec(
            select(PortfolioDailyNAV)
            .where(PortfolioDailyNAV.portfolio_id == portfolio_id)
            .where(PortfolioDailyNAV.variant == Variants.RAW)
            .order_by(PortfolioDailyNAV.trade_date)
        ).all()

        if len(nav_records) < 2:
            print("  Not enough NAV data")
            return

        print(f"  Holdings: {len(holdings)}, NAV records: {len(nav_records)}")

        # Sample and validate
        errors = []
        for i in range(1, min(sample_size + 1, len(nav_records))):
            current = nav_records[i]
            previous = nav_records[i-1]

            # Get prices for both dates
            curr_prices = session.exec(
                select(PriceData)
                .where(PriceData.ticker.in_(tickers))
                .where(PriceData.trade_date == current.trade_date)
            ).all()

            prev_prices = session.exec(
                select(PriceData)
                .where(PriceData.ticker.in_(tickers))
                .where(PriceData.trade_date == previous.trade_date)
            ).all()

            curr_price_map = {p.ticker: p.close for p in curr_prices}
            prev_price_map = {p.ticker: p.close for p in prev_prices}

            # Calculate expected return
            expected_return = 0.0
            for ticker, weight in weights.items():
                curr = curr_price_map.get(ticker)
                prev = prev_price_map.get(ticker)
                if curr and prev and prev > 0:
                    ticker_return = (curr / prev) - 1
                    expected_return += weight * ticker_return

            actual_return = float(current.daily_return) if current.daily_return else 0

            diff = abs(expected_return - actual_return)
            if diff > 0.001:  # More than 0.1% difference
                errors.append({
                    "date": current.trade_date,
                    "expected": expected_return,
                    "actual": actual_return,
                    "diff": diff,
                })

            status = "MATCH" if diff <= 0.001 else "MISMATCH"
            print(f"  {current.trade_date}: expected={expected_return*100:+.2f}%, actual={actual_return*100:+.2f}% [{status}]")

        if errors:
            print(f"  WARNING: {len(errors)} mismatches found!")
        else:
            print(f"  All {sample_size} samples validated correctly")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Audit portfolio data integrity")
    parser.add_argument("--fix", action="store_true", help="Actually fix issues (default: dry run)")
    parser.add_argument("--validate", type=int, help="Validate NAV for specific portfolio ID")
    args = parser.parse_args()

    # Phase 1: Audit holdings
    holdings_issues = audit_holdings()

    # Phase 2: Audit NAV returns
    nav_anomalies = audit_nav_returns()

    # Phase 3: Fix holdings (if requested)
    if holdings_issues:
        fixed = fix_holdings(holdings_issues, dry_run=not args.fix)

        if not args.fix and holdings_issues:
            print()
            print("To fix these issues, run:")
            print("  python scripts/audit_portfolios.py --fix")

    # Phase 4: Validate specific portfolio
    if args.validate:
        validate_nav_calculation(args.validate)

    # Summary
    print()
    print("=" * 80)
    print("AUDIT SUMMARY")
    print("=" * 80)
    print(f"Holdings issues: {len(holdings_issues)} portfolios")
    print(f"NAV anomalies: {len(nav_anomalies)} portfolios")

    if holdings_issues or nav_anomalies:
        print()
        print("RECOMMENDED ACTIONS:")
        if holdings_issues:
            print("1. Run: python scripts/audit_portfolios.py --fix")
            print("2. Re-run NAV backfill for affected portfolios")
        if nav_anomalies:
            print("3. Investigate NAV anomalies and recalculate if needed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
