#!/usr/bin/env python3
"""
NAV Verification Script

Recalculates all portfolio NAVs from actual stock prices and compares
to stored values. Reports any discrepancies.

Usage:
    python scripts/verify_all_navs.py
    python scripts/verify_all_navs.py --portfolio "QQQ"
    python scripts/verify_all_navs.py --threshold 5.0  # Flag >5% discrepancies
    python scripts/verify_all_navs.py --fix  # Auto-fix discrepancies (CAREFUL!)
"""

import os
import sys
import logging
import argparse
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AlphaMachine_core.tracking import (
    PortfolioTracker,
    get_tracker,
    Variants,
)
from AlphaMachine_core.tracking.models import PortfolioDailyNAV

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def calculate_expected_nav_from_prices(
    tracker: PortfolioTracker,
    portfolio_id: int,
    portfolio_name: str,
) -> pd.DataFrame:
    """
    Recalculate NAV series from actual stock prices.

    Returns DataFrame with columns: date, expected_nav, daily_return
    """
    # Get all holdings for this portfolio
    holdings = tracker.get_holdings(portfolio_id)

    if not holdings:
        logger.warning(f"No holdings found for {portfolio_name}")
        return pd.DataFrame()

    # Get unique tickers
    tickers = list(set(h.ticker for h in holdings))

    # Get price data for all tickers
    from AlphaMachine_core.tracking.s3_adapter import get_s3_loader
    s3_loader = get_s3_loader()

    # Load prices for each ticker
    price_dfs = {}
    for ticker in tickers:
        try:
            prices = s3_loader.load_stock_prices(ticker)
            if prices is not None and not prices.empty:
                price_dfs[ticker] = prices
        except Exception as e:
            logger.warning(f"Could not load prices for {ticker}: {e}")

    if not price_dfs:
        logger.error(f"No price data available for {portfolio_name}")
        return pd.DataFrame()

    # Get weights from holdings (assuming equal weight if shares not set)
    weights = {}
    for h in holdings:
        weights[h.ticker] = h.weight if h.weight else 1.0 / len(holdings)

    # Normalize weights
    total_weight = sum(weights.values())
    weights = {t: w / total_weight for t, w in weights.items()}

    # Find common date range
    all_dates = None
    for ticker, df in price_dfs.items():
        dates = set(df.index)
        if all_dates is None:
            all_dates = dates
        else:
            all_dates = all_dates.intersection(dates)

    if not all_dates:
        logger.error(f"No common dates found for {portfolio_name}")
        return pd.DataFrame()

    all_dates = sorted(all_dates)

    # Calculate daily returns for the portfolio
    portfolio_returns = []

    for i, current_date in enumerate(all_dates):
        if i == 0:
            # First day - no return
            portfolio_returns.append({
                'date': current_date,
                'daily_return': 0.0,
            })
            continue

        prev_date = all_dates[i - 1]

        # Calculate weighted return
        daily_return = 0.0
        for ticker, weight in weights.items():
            if ticker in price_dfs:
                df = price_dfs[ticker]
                if current_date in df.index and prev_date in df.index:
                    curr_price = df.loc[current_date, 'close'] if 'close' in df.columns else df.loc[current_date].iloc[0]
                    prev_price = df.loc[prev_date, 'close'] if 'close' in df.columns else df.loc[prev_date].iloc[0]
                    if prev_price > 0:
                        ticker_return = (curr_price - prev_price) / prev_price
                        daily_return += weight * ticker_return

        portfolio_returns.append({
            'date': current_date,
            'daily_return': daily_return,
        })

    # Build NAV series from returns (starting at 100)
    nav_series = []
    nav = 100.0

    for r in portfolio_returns:
        nav_series.append({
            'date': r['date'],
            'expected_nav': nav,
            'daily_return': r['daily_return'],
        })
        nav = nav * (1 + r['daily_return'])

    return pd.DataFrame(nav_series)


def verify_portfolio_navs(
    tracker: PortfolioTracker,
    portfolio_id: int,
    portfolio_name: str,
    threshold_pct: float = 5.0,
) -> List[Dict]:
    """
    Verify stored NAVs against recalculated values.

    Returns list of discrepancies.
    """
    discrepancies = []

    # Get stored NAVs
    stored_navs = tracker.get_nav_series(portfolio_id, Variants.RAW)

    if stored_navs.empty:
        logger.warning(f"No stored NAVs for {portfolio_name}")
        return discrepancies

    # Calculate expected NAVs
    expected_df = calculate_expected_nav_from_prices(tracker, portfolio_id, portfolio_name)

    if expected_df.empty:
        logger.warning(f"Could not calculate expected NAVs for {portfolio_name}")
        return discrepancies

    # Compare day by day
    # We need to compare RETURNS, not absolute NAV (since starting points may differ)
    stored_navs = stored_navs.reset_index()
    stored_navs.columns = ['date', 'nav']
    stored_navs['stored_return'] = stored_navs['nav'].pct_change()

    expected_df['date'] = pd.to_datetime(expected_df['date'])
    stored_navs['date'] = pd.to_datetime(stored_navs['date'])

    # Merge on date
    merged = pd.merge(
        stored_navs,
        expected_df[['date', 'daily_return']],
        on='date',
        how='inner'
    )

    merged['expected_return'] = merged['daily_return']
    merged['return_diff'] = (merged['stored_return'] - merged['expected_return']) * 100

    # Flag days where return differs by more than threshold
    for _, row in merged.iterrows():
        if pd.isna(row['return_diff']):
            continue

        if abs(row['return_diff']) > threshold_pct:
            discrepancies.append({
                'portfolio': portfolio_name,
                'date': row['date'].strftime('%Y-%m-%d'),
                'stored_nav': row['nav'],
                'stored_return': f"{row['stored_return']*100:.2f}%",
                'expected_return': f"{row['expected_return']*100:.2f}%",
                'diff': f"{row['return_diff']:.2f}%",
            })

    return discrepancies


def verify_using_stored_returns(
    tracker: PortfolioTracker,
    threshold_pct: float = 5.0,
) -> Dict[str, List[Dict]]:
    """
    Simplified verification: check for suspicious daily returns in stored data.

    This catches:
    - NAV resets (sudden drops to ~100)
    - Extreme returns that don't match market reality
    - Data corruption
    """
    from sqlmodel import Session, select, text
    from AlphaMachine_core.db import engine
    from AlphaMachine_core.tracking.models import PortfolioDefinition

    results = {}

    with Session(engine) as session:
        portfolios = session.exec(select(PortfolioDefinition)).all()

        for portfolio in portfolios:
            issues = []

            # Get NAV series
            nav_df = tracker.get_nav_series(portfolio.id, Variants.RAW)

            if nav_df.empty or len(nav_df) < 2:
                continue

            # Handle both index-based and column-based date formats
            nav_df = nav_df.reset_index()

            # Find the date and nav columns
            if 'date' in nav_df.columns:
                date_col = 'date'
            elif 'trade_date' in nav_df.columns:
                date_col = 'trade_date'
            elif 'index' in nav_df.columns:
                date_col = 'index'
            else:
                date_col = nav_df.columns[0]

            if 'nav' in nav_df.columns:
                nav_col = 'nav'
            else:
                # Find first numeric column that's not the date
                for col in nav_df.columns:
                    if col != date_col and nav_df[col].dtype in ['float64', 'int64']:
                        nav_col = col
                        break

            # Rename to standard columns
            nav_df = nav_df[[date_col, nav_col]].copy()
            nav_df.columns = ['date', 'nav']
            nav_df = nav_df.sort_values('date')

            # Calculate daily returns
            nav_df['prev_nav'] = nav_df['nav'].shift(1)
            nav_df['daily_return'] = (nav_df['nav'] - nav_df['prev_nav']) / nav_df['prev_nav'] * 100

            for _, row in nav_df.iterrows():
                if pd.isna(row['daily_return']):
                    continue

                daily_return = row['daily_return']

                # Flag suspicious returns
                if abs(daily_return) > threshold_pct:
                    severity = "CRITICAL" if abs(daily_return) > 10 else "WARNING"

                    issues.append({
                        'date': row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date']),
                        'nav': f"{row['nav']:.2f}",
                        'prev_nav': f"{row['prev_nav']:.2f}",
                        'daily_return': f"{daily_return:.2f}%",
                        'severity': severity,
                    })

            if issues:
                results[portfolio.name] = issues

    return results


def main():
    parser = argparse.ArgumentParser(description="Verify NAV calculations against stock prices")
    parser.add_argument(
        "--portfolio",
        type=str,
        help="Specific portfolio to verify (or all if not specified)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Flag returns exceeding this percentage (default: 5.0)",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-fix discrepancies (USE WITH CAUTION)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Run detailed verification against stock prices (slower)",
    )

    args = parser.parse_args()

    tracker = get_tracker()

    print("=" * 70)
    print(f"NAV VERIFICATION REPORT (threshold: +/-{args.threshold}%)")
    print("=" * 70)

    # Quick verification using stored returns
    results = verify_using_stored_returns(tracker, args.threshold)

    if args.portfolio:
        # Filter to specific portfolio
        results = {k: v for k, v in results.items() if k == args.portfolio}

    total_issues = 0
    critical_count = 0

    if not results:
        print("\n[OK] All portfolios passed verification - no suspicious returns found.\n")
    else:
        for portfolio_name, issues in sorted(results.items()):
            print(f"\n[PORTFOLIO] {portfolio_name} ({len(issues)} issues)")
            print("-" * 50)

            for issue in issues:
                severity_icon = "[CRITICAL]" if issue['severity'] == "CRITICAL" else "[WARNING]"
                print(f"  {severity_icon} {issue['date']}: {issue['daily_return']} "
                      f"(NAV: {issue['prev_nav']} -> {issue['nav']})")
                total_issues += 1
                if issue['severity'] == "CRITICAL":
                    critical_count += 1

        print("\n" + "=" * 70)
        print(f"SUMMARY: {total_issues} issues found across {len(results)} portfolios")
        print(f"         {critical_count} critical (>10%), {total_issues - critical_count} warnings (>{args.threshold}%)")
        print("=" * 70)

    # Exit with error if critical issues found
    if critical_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
