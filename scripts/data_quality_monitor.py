#!/usr/bin/env python3
"""
Data Quality Monitor - Daily Health Check for Portfolio Tracking.

This script should be run daily (e.g., 7 AM) to detect data issues
BEFORE users see them in dashboards.

What a senior quant dev would want:
1. Overnight anomaly detection
2. Cross-portfolio consistency checks
3. Benchmark divergence alerts
4. Audit trail review
5. Email/Slack alerts for issues

Usage:
    python scripts/data_quality_monitor.py

    # Check specific portfolio
    python scripts/data_quality_monitor.py --portfolio "QQQ"

    # Check last N days
    python scripts/data_quality_monitor.py --days 14

    # Output to JSON for integration with monitoring systems
    python scripts/data_quality_monitor.py --json

    # Send email alerts on failure
    python scripts/data_quality_monitor.py --email your@gmail.com

Environment Variables for Email:
    SMTP_SERVER: SMTP server (default: smtp.gmail.com)
    SMTP_PORT: SMTP port (default: 587)
    SMTP_USER: Email sender address
    SMTP_PASSWORD: Email password or app-specific password
    ALERT_EMAIL: Default recipient email
"""

import os
import sys
import json
import logging
import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load secrets
def _load_secrets():
    secrets_path = project_root / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        with open(secrets_path, "rb") as f:
            secrets = tomllib.load(f)
        for key, value in secrets.items():
            if key not in os.environ and isinstance(value, str):
                os.environ[key] = value

_load_secrets()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def is_trading_day(check_date: date) -> bool:
    """
    Check if a given date is a US stock market trading day.

    Returns False for:
    - Weekends (Saturday, Sunday)
    - Major US holidays (approximate)
    """
    # Check weekend
    if check_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False

    # Major US market holidays (approximate - doesn't handle all edge cases)
    # These are fixed dates; actual holidays may shift if they fall on weekends
    year = check_date.year
    us_holidays = [
        date(year, 1, 1),   # New Year's Day
        date(year, 7, 4),   # Independence Day
        date(year, 12, 25), # Christmas
    ]

    # Variable holidays (approximate)
    # MLK Day: 3rd Monday of January
    # Presidents Day: 3rd Monday of February
    # Memorial Day: Last Monday of May
    # Labor Day: 1st Monday of September
    # Thanksgiving: 4th Thursday of November

    # For simplicity, just check weekends and fixed holidays
    # A more robust solution would use a library like `pandas_market_calendars`

    if check_date in us_holidays:
        return False

    return True


def get_last_trading_day(reference_date: date = None) -> date:
    """
    Get the most recent trading day on or before the reference date.

    If reference_date is a weekend or holiday, returns the previous trading day.
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


def check_nav_anomalies(
    portfolios: List,
    tracker,
    days_back: int = 7,
) -> List[Dict]:
    """
    Check for NAV anomalies across portfolios.

    Detects:
    - Extreme daily returns (>10%)
    - NAV near 100 (potential reset)
    - Missing data points
    - Stale data (same NAV for multiple days)
    """
    from AlphaMachine_core.tracking import Variants

    issues = []
    end_date = date.today()
    start_date = end_date - timedelta(days=days_back)

    for portfolio in portfolios:
        nav_df = tracker.get_nav_series(
            portfolio.id, Variants.RAW, start_date, end_date
        )

        if nav_df.empty:
            issues.append({
                "portfolio": portfolio.name,
                "type": "NO_DATA",
                "severity": "warning",
                "message": f"No NAV data for last {days_back} days",
            })
            continue

        # Check for extreme daily returns
        daily_returns = nav_df["daily_return"] * 100
        extreme_returns = daily_returns[abs(daily_returns) > 10]

        for idx, ret in extreme_returns.items():
            issues.append({
                "portfolio": portfolio.name,
                "date": idx.strftime("%Y-%m-%d"),
                "type": "EXTREME_RETURN",
                "severity": "error" if abs(ret) > 20 else "warning",
                "message": f"Daily return of {ret:+.2f}%",
                "value": float(ret),
            })

        # Check for NAV near 100 (potential reset) when it shouldn't be
        latest_nav = nav_df["nav"].iloc[-1]
        first_nav_date = nav_df.index[0].date() if hasattr(nav_df.index[0], 'date') else nav_df.index[0]
        nav_days = (end_date - first_nav_date).days
        if nav_days > 60:
            # Portfolio has >60 days of NAV data, NAV near 100 is suspicious
            if 95 < latest_nav < 105:
                issues.append({
                    "portfolio": portfolio.name,
                    "date": nav_df.index[-1].strftime("%Y-%m-%d"),
                    "type": "SUSPICIOUS_NAV",
                    "severity": "warning",
                    "message": f"NAV is {latest_nav:.2f} (near initial 100) after {nav_days} days of tracking",
                    "value": float(latest_nav),
                })

        # Check for stale data (same NAV for 3+ consecutive days)
        nav_values = nav_df["nav"].values
        if len(nav_values) >= 3:
            for i in range(2, len(nav_values)):
                if nav_values[i] == nav_values[i-1] == nav_values[i-2]:
                    issues.append({
                        "portfolio": portfolio.name,
                        "date": nav_df.index[i].strftime("%Y-%m-%d"),
                        "type": "STALE_DATA",
                        "severity": "warning",
                        "message": f"NAV unchanged for 3+ days at {nav_values[i]:.2f}",
                        "value": float(nav_values[i]),
                    })
                    break  # Only report once

        # Check for missing trading days
        expected_days = len(nav_df)  # Simplified check
        if expected_days < days_back * 0.6:  # Less than 60% of days have data
            issues.append({
                "portfolio": portfolio.name,
                "type": "SPARSE_DATA",
                "severity": "warning",
                "message": f"Only {expected_days} data points in {days_back} days",
            })

    return issues


def check_cross_portfolio_consistency(
    portfolios: List,
    tracker,
) -> List[Dict]:
    """
    Check for consistency issues across portfolios.

    Detects:
    - Portfolios with very different returns on same day
    - Missing updates for some portfolios
    """
    from AlphaMachine_core.tracking import Variants

    issues = []
    yesterday = date.today() - timedelta(days=1)

    # Collect yesterday's returns
    returns = {}
    for portfolio in portfolios:
        nav_df = tracker.get_nav_series(
            portfolio.id, Variants.RAW, yesterday, yesterday
        )
        if not nav_df.empty:
            returns[portfolio.name] = nav_df["daily_return"].iloc[0] * 100

    if len(returns) < 2:
        return issues

    # Check for outliers (>3 std from mean)
    import numpy as np
    values = list(returns.values())
    mean_ret = np.mean(values)
    std_ret = np.std(values)

    if std_ret > 0:
        for name, ret in returns.items():
            z_score = (ret - mean_ret) / std_ret
            if abs(z_score) > 3:
                issues.append({
                    "portfolio": name,
                    "date": yesterday.strftime("%Y-%m-%d"),
                    "type": "OUTLIER_RETURN",
                    "severity": "warning",
                    "message": f"Return {ret:+.2f}% is {z_score:.1f} std devs from mean ({mean_ret:.2f}%)",
                    "value": float(ret),
                })

    return issues


def check_audit_log_issues(days_back: int = 7) -> List[Dict]:
    """Check audit log for blocked updates or suspicious patterns."""
    from AlphaMachine_core.tracking.data_quality import get_audit_log

    issues = []
    audit_log = get_audit_log()

    # Get suspicious changes
    suspicious = audit_log.get_suspicious_changes(
        threshold_pct=10.0,
        days_back=days_back,
    )

    if not suspicious.empty:
        for _, row in suspicious.iterrows():
            issues.append({
                "portfolio": row.get("portfolio_name", "Unknown"),
                "date": str(row.get("trade_date")),
                "type": "AUDIT_SUSPICIOUS",
                "severity": "warning",
                "message": f"NAV changed {row.get('change_pct', 0):.1f}% (source: {row.get('source', 'unknown')})",
                "value": float(row.get("change_pct", 0)),
            })

    return issues


def reconcile_mtd_from_prices(
    portfolios: List,
    tracker,
    tolerance_pct: float = 0.5,
) -> List[Dict]:
    """
    Reconcile MTD (month-to-date) NAV against actual stock prices.

    Recalculates NAV from holdings and prices, compares to stored NAV.
    Flags any discrepancy > tolerance_pct.

    This catches:
    - NAV calculation bugs
    - Price data issues
    - Holdings weight mismatches
    """
    import numpy as np
    import pandas as pd
    from AlphaMachine_core.tracking import Variants
    from AlphaMachine_core.data_manager import StockDataManager

    issues = []
    today = date.today()

    # Get first day of current month
    mtd_start = today.replace(day=1)

    # Collect all tickers from all portfolios
    all_tickers = set()
    portfolio_holdings = {}

    for portfolio in portfolios:
        try:
            holdings = tracker.get_holdings(portfolio.id, today)
            if holdings:
                portfolio_holdings[portfolio.id] = holdings
                for h in holdings:
                    all_tickers.add(h.ticker)
        except Exception:
            pass

    if not all_tickers:
        return issues

    # Fetch price data for all tickers at once (efficient)
    try:
        dm = StockDataManager()
        # Fetch prices from a week before MTD start to ensure we have prev prices
        price_start = mtd_start - timedelta(days=7)
        price_dicts = dm.get_price_data(
            list(all_tickers),
            price_start.strftime("%Y-%m-%d"),
            today.strftime("%Y-%m-%d"),
        )

        if not price_dicts:
            return issues

        price_df = pd.DataFrame(price_dicts)
        if price_df.empty:
            return issues

        price_df["trade_date"] = pd.to_datetime(price_df["trade_date"]).dt.date

    except Exception as e:
        logger.warning(f"Could not load price data for reconciliation: {e}")
        return issues

    for portfolio in portfolios:
        try:
            # Get stored NAV for MTD
            stored_nav_df = tracker.get_nav_series(
                portfolio.id, Variants.RAW, mtd_start, today
            )

            if stored_nav_df.empty:
                continue

            holdings = portfolio_holdings.get(portfolio.id)
            if not holdings:
                continue

            # Get weights (default to equal weight), convert Decimal to float
            weights = {}
            for h in holdings:
                w = h.weight if h.weight else 1.0 / len(holdings)
                weights[h.ticker] = float(w)

            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {t: w / total_weight for t, w in weights.items()}

            # Calculate expected returns from prices for each day in MTD
            for nav_date in stored_nav_df.index:
                nav_date_dt = nav_date.date() if hasattr(nav_date, 'date') else nav_date

                stored_return = stored_nav_df.loc[nav_date, 'daily_return'] * 100

                # Skip first day (no return to compare)
                if np.isnan(stored_return):
                    continue

                # Skip first day of month (edge case with previous month boundary)
                if nav_date_dt.day == 1:
                    continue

                # Get prices for this date and previous trading day
                curr_prices = price_df[price_df["trade_date"] == nav_date_dt]
                if curr_prices.empty:
                    continue

                # Find previous trading day
                prev_dates = price_df[price_df["trade_date"] < nav_date_dt]["trade_date"].unique()
                if len(prev_dates) == 0:
                    continue
                prev_date = max(prev_dates)
                prev_prices = price_df[price_df["trade_date"] == prev_date]

                if prev_prices.empty:
                    continue

                # Calculate expected return from prices
                expected_return = 0.0
                tickers_used = 0

                for ticker, weight in weights.items():
                    curr_row = curr_prices[curr_prices["ticker"] == ticker]
                    prev_row = prev_prices[prev_prices["ticker"] == ticker]

                    if curr_row.empty or prev_row.empty:
                        continue

                    # Use adjusted_close if available
                    if "adjusted_close" in curr_row.columns and curr_row["adjusted_close"].notna().any():
                        curr_price = curr_row["adjusted_close"].fillna(curr_row["close"]).iloc[0]
                        prev_price = prev_row["adjusted_close"].fillna(prev_row["close"]).iloc[0]
                    else:
                        curr_price = curr_row["close"].iloc[0]
                        prev_price = prev_row["close"].iloc[0]

                    if prev_price > 0:
                        ticker_return = ((curr_price / prev_price) - 1) * 100
                        expected_return += weight * ticker_return
                        tickers_used += 1

                # Only compare if we got prices for at least some tickers
                if tickers_used < len(weights) * 0.5:
                    continue

                # Compare stored vs expected
                diff = abs(stored_return - expected_return)

                if diff > tolerance_pct:
                    severity = "error" if diff > 2.0 else "warning"
                    issues.append({
                        "portfolio": portfolio.name,
                        "date": str(nav_date_dt),
                        "type": "MTD_RECONCILIATION_MISMATCH",
                        "severity": severity,
                        "message": f"Stored return {stored_return:+.2f}% vs expected {expected_return:+.2f}% (diff: {diff:.2f}%)",
                        "stored_return": float(stored_return),
                        "expected_return": float(expected_return),
                        "difference": float(diff),
                    })

        except Exception as e:
            logger.warning(f"Error reconciling {portfolio.name}: {e}")

    return issues


def check_methodology_divergence(
    portfolios: List,
    tracker,
    days_back: int = 7,
    tolerance_pct: float = 2.0,
) -> List[Dict]:
    """
    Compare share-based NAV (stored) vs weight-based NAV for divergence.

    This catches methodology inconsistencies that can accumulate over time:
    - Share-based: NAV = sum(shares × price) - buy-and-hold
    - Weight-based: NAV = prev_NAV × (1 + weighted_return)

    For short periods, these should match within tolerance.
    Large divergence indicates data corruption or calculation bugs.

    Args:
        portfolios: List of portfolios to check
        tracker: Portfolio tracker instance
        days_back: Number of days to analyze
        tolerance_pct: Maximum allowed return difference (default 2%)

    Returns:
        List of issues found
    """
    import pandas as pd
    from decimal import Decimal
    from AlphaMachine_core.tracking import Variants
    from AlphaMachine_core.data_manager import StockDataManager

    issues = []
    today = date.today()
    start_date = today - timedelta(days=days_back)

    # Collect all tickers
    all_tickers = set()
    for portfolio in portfolios:
        holdings = tracker.get_holdings(portfolio.id, today)
        if holdings:
            for h in holdings:
                all_tickers.add(h.ticker)

    if not all_tickers:
        return issues

    # Load price data
    try:
        dm = StockDataManager()
        price_dicts = dm.get_price_data(
            list(all_tickers),
            (start_date - timedelta(days=7)).strftime("%Y-%m-%d"),
            today.strftime("%Y-%m-%d"),
        )
        if not price_dicts:
            return issues

        price_df = pd.DataFrame(price_dicts)
        price_df["trade_date"] = pd.to_datetime(price_df["trade_date"]).dt.date

    except Exception as e:
        logger.warning(f"Could not load price data for methodology check: {e}")
        return issues

    for portfolio in portfolios:
        try:
            # Get stored NAV
            nav_df = tracker.get_nav_series(
                portfolio.id, Variants.RAW, start_date, today
            )
            if nav_df.empty or len(nav_df) < 2:
                continue

            # Get holdings (we use latest holdings for weight reference)
            holdings = tracker.get_holdings(portfolio.id, today)
            if not holdings:
                continue

            # Build weights dict (normalize)
            weights = {}
            for h in holdings:
                w = float(h.weight) if h.weight else 1.0 / len(holdings)
                weights[h.ticker] = w
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {t: w / total_weight for t, w in weights.items()}

            # Compare stored daily return vs weight-based calculation
            divergence_days = []

            for i in range(1, len(nav_df)):
                nav_date = nav_df.index[i]
                nav_date_dt = nav_date.date() if hasattr(nav_date, 'date') else nav_date
                prev_date = nav_df.index[i - 1]
                prev_date_dt = prev_date.date() if hasattr(prev_date, 'date') else prev_date

                # Stored return (share-based)
                stored_return = nav_df.iloc[i]["daily_return"] * 100

                # Calculate weight-based return from prices
                curr_prices = price_df[price_df["trade_date"] == nav_date_dt]
                prev_prices = price_df[price_df["trade_date"] == prev_date_dt]

                if curr_prices.empty or prev_prices.empty:
                    continue

                weighted_return = 0.0
                tickers_used = 0

                for ticker, weight in weights.items():
                    curr_row = curr_prices[curr_prices["ticker"] == ticker]
                    prev_row = prev_prices[prev_prices["ticker"] == ticker]

                    if curr_row.empty or prev_row.empty:
                        continue

                    # Use adjusted close
                    if "adjusted_close" in curr_row.columns:
                        curr_price = curr_row["adjusted_close"].fillna(curr_row["close"]).iloc[0]
                        prev_price = prev_row["adjusted_close"].fillna(prev_row["close"]).iloc[0]
                    else:
                        curr_price = curr_row["close"].iloc[0]
                        prev_price = prev_row["close"].iloc[0]

                    if prev_price and prev_price > 0:
                        ticker_return = ((curr_price / prev_price) - 1) * 100
                        weighted_return += weight * ticker_return
                        tickers_used += 1

                if tickers_used < len(weights) * 0.5:
                    continue

                # Compare
                diff = abs(stored_return - weighted_return)
                if diff > tolerance_pct:
                    divergence_days.append({
                        "date": str(nav_date_dt),
                        "stored": stored_return,
                        "expected": weighted_return,
                        "diff": diff,
                    })

            # Report significant divergences
            if divergence_days:
                # Find worst divergence
                worst = max(divergence_days, key=lambda x: x["diff"])

                severity = "error" if worst["diff"] > 5.0 else "warning"
                issues.append({
                    "portfolio": portfolio.name,
                    "date": worst["date"],
                    "type": "METHODOLOGY_DIVERGENCE",
                    "severity": severity,
                    "message": (
                        f"Share-based return {worst['stored']:+.2f}% vs "
                        f"weight-based {worst['expected']:+.2f}% (diff: {worst['diff']:.2f}%)"
                    ),
                    "divergent_days": len(divergence_days),
                    "worst_diff": worst["diff"],
                })

        except Exception as e:
            logger.warning(f"Error checking methodology divergence for {portfolio.name}: {e}")

    return issues


def check_nav_baseline_accuracy(
    portfolios: List,
    tracker,
    tolerance_pct: float = 0.5,
) -> List[Dict]:
    """
    Check that stored NAV equals sum(shares × price) for each portfolio.

    This catches "baseline offset" issues where:
    - Daily returns are correct (wrong shares cancel out)
    - But absolute NAV values are wrong due to incorrect shares calculation
    - MTD/YTD calculations are wrong because baseline is off

    Args:
        portfolios: List of portfolios to check
        tracker: Portfolio tracker instance
        tolerance_pct: Acceptable drift percentage (default 0.5%)

    Returns:
        List of issues found
    """
    from decimal import Decimal
    from sqlmodel import Session, select
    from AlphaMachine_core.db import engine
    from AlphaMachine_core.models import PriceData
    from AlphaMachine_core.tracking.models import PortfolioDailyNAV, Variants

    issues = []
    today = date.today()

    for portfolio in portfolios:
        try:
            with Session(engine) as session:
                # Get latest NAV
                nav_stmt = (
                    select(PortfolioDailyNAV)
                    .where(PortfolioDailyNAV.portfolio_id == portfolio.id)
                    .where(PortfolioDailyNAV.variant == Variants.RAW)
                    .order_by(PortfolioDailyNAV.trade_date.desc())
                    .limit(1)
                )
                latest_nav = session.exec(nav_stmt).first()

                if not latest_nav:
                    continue

                check_date = latest_nav.trade_date
                stored_nav = float(latest_nav.nav)

                # Get holdings for that date
                holdings = tracker.get_holdings(portfolio.id, check_date)
                if not holdings:
                    continue

                # Calculate expected NAV = sum(shares × price)
                expected_nav = Decimal("0")
                missing_prices = []

                for h in holdings:
                    if not h.shares:
                        continue

                    # Get price for this ticker on check_date
                    price_stmt = (
                        select(PriceData)
                        .where(PriceData.ticker == h.ticker)
                        .where(PriceData.trade_date <= check_date)
                        .order_by(PriceData.trade_date.desc())
                        .limit(1)
                    )
                    price_record = session.exec(price_stmt).first()

                    if price_record:
                        price = price_record.adjusted_close or price_record.close
                        expected_nav += h.shares * Decimal(str(price))
                    else:
                        missing_prices.append(h.ticker)

                if missing_prices:
                    logger.warning(
                        f"  {portfolio.name}: Missing prices for {missing_prices}"
                    )
                    continue

                if expected_nav == 0:
                    continue

                # Calculate drift
                expected_nav_float = float(expected_nav)
                drift_pct = ((stored_nav / expected_nav_float) - 1) * 100

                if abs(drift_pct) > tolerance_pct:
                    issues.append({
                        "portfolio": portfolio.name,
                        "check": "nav_baseline_accuracy",
                        "severity": "error" if abs(drift_pct) > 2.0 else "warning",
                        "message": (
                            f"NAV baseline drift: stored={stored_nav:.2f}, "
                            f"expected={expected_nav_float:.2f}, drift={drift_pct:.2f}%"
                        ),
                        "details": {
                            "check_date": str(check_date),
                            "stored_nav": stored_nav,
                            "expected_nav": expected_nav_float,
                            "drift_pct": drift_pct,
                        },
                    })
                    logger.info(
                        f"  {portfolio.name}: DRIFT {drift_pct:.2f}% "
                        f"(stored={stored_nav:.2f}, expected={expected_nav_float:.2f})"
                    )
                else:
                    logger.debug(f"  {portfolio.name}: OK (drift={drift_pct:.3f}%)")

        except Exception as e:
            logger.warning(f"Error checking baseline accuracy for {portfolio.name}: {e}")

    return issues


def check_missing_ticker_data(
    portfolios: List,
    tracker,
    days_back: int = 7,
) -> List[Dict]:
    """
    Check for missing price data for portfolio holdings.

    Detects:
    - Tickers with no price data at all
    - Tickers missing recent trading days
    - Stale price data (last update > days_back ago)

    Note: Uses last trading day as reference (handles weekends/holidays).
    """
    import pandas as pd
    from AlphaMachine_core.data_manager import StockDataManager

    issues = []
    # Use last trading day instead of today (handles weekends/holidays)
    today = get_last_trading_day()
    cutoff_date = today - timedelta(days=days_back)

    logger.debug(f"Using reference date: {today} (last trading day)")

    # Collect all unique tickers across portfolios
    all_tickers = set()
    ticker_portfolios = {}  # Map ticker -> list of portfolios using it

    for portfolio in portfolios:
        holdings = tracker.get_holdings(portfolio.id, today)
        if not holdings:
            continue

        for holding in holdings:
            ticker = holding.ticker
            all_tickers.add(ticker)
            if ticker not in ticker_portfolios:
                ticker_portfolios[ticker] = []
            ticker_portfolios[ticker].append(portfolio.name)

    if not all_tickers:
        return issues

    # Fetch price data for all tickers at once
    try:
        dm = StockDataManager()
        price_dicts = dm.get_price_data(
            list(all_tickers),
            cutoff_date.strftime("%Y-%m-%d"),
            today.strftime("%Y-%m-%d"),
        )

        if not price_dicts:
            # No price data at all - report for all tickers
            for ticker, portfolios_list in ticker_portfolios.items():
                for portfolio_name in portfolios_list:
                    issues.append({
                        "portfolio": portfolio_name,
                        "type": "MISSING_TICKER_DATA",
                        "severity": "error",
                        "message": f"No price data available for {ticker}",
                        "ticker": ticker,
                    })
            return issues

        price_df = pd.DataFrame(price_dicts)
        price_df["trade_date"] = pd.to_datetime(price_df["trade_date"]).dt.date

    except Exception as e:
        logger.warning(f"Could not load price data for ticker check: {e}")
        return issues

    # Check each ticker
    checked_tickers = {}

    for ticker in all_tickers:
        ticker_prices = price_df[price_df["ticker"] == ticker]

        if ticker_prices.empty:
            checked_tickers[ticker] = {
                "has_issue": True,
                "type": "MISSING_TICKER_DATA",
                "severity": "error",
                "message": f"No price data available for {ticker}",
            }
            continue

        # Check last available date
        last_date = ticker_prices["trade_date"].max()

        # Only flag as stale if last_date is more than 3 trading days before reference
        # This handles weekends and short holidays gracefully
        if last_date < cutoff_date:
            days_stale = (today - last_date).days
            # Only report if significantly stale (more than a long weekend)
            if days_stale > 4:
                checked_tickers[ticker] = {
                    "has_issue": True,
                    "type": "STALE_TICKER_DATA",
                    "severity": "warning",
                    "message": f"{ticker} price data is {days_stale} days old (last: {last_date})",
                    "last_date": str(last_date),
                }
                continue

        # Count trading days
        trading_days = len(ticker_prices)

        # Expect at least 60% of days to have data (accounting for weekends/holidays)
        expected_min = int(days_back * 0.6)
        if trading_days < expected_min:
            checked_tickers[ticker] = {
                "has_issue": True,
                "type": "SPARSE_TICKER_DATA",
                "severity": "warning",
                "message": f"{ticker} has only {trading_days} data points in last {days_back} days",
                "trading_days": trading_days,
            }
            continue

        # No issues
        checked_tickers[ticker] = {"has_issue": False}

    # Report issues for each portfolio-ticker combination
    for ticker, status in checked_tickers.items():
        if status.get("has_issue"):
            for portfolio_name in ticker_portfolios.get(ticker, []):
                issue = {
                    "portfolio": portfolio_name,
                    "type": status["type"],
                    "severity": status["severity"],
                    "message": status["message"],
                    "ticker": ticker,
                }
                if "last_date" in status:
                    issue["last_date"] = status["last_date"]
                if "trading_days" in status:
                    issue["trading_days"] = status["trading_days"]
                issues.append(issue)

    return issues


def generate_report(issues: List[Dict]) -> str:
    """Generate human-readable report."""
    if not issues:
        return "[OK] All data quality checks passed. No issues detected."

    lines = [
        "=" * 60,
        "DATA QUALITY REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
    ]

    # Group by severity
    critical = [i for i in issues if i.get("severity") == "critical"]
    errors = [i for i in issues if i.get("severity") == "error"]
    warnings = [i for i in issues if i.get("severity") == "warning"]

    if critical:
        lines.append("[CRITICAL] CRITICAL ISSUES:")
        for issue in critical:
            lines.append(f"  - [{issue['portfolio']}] {issue['message']}")
        lines.append("")

    if errors:
        lines.append("[ERROR] ERRORS:")
        for issue in errors:
            date_str = f" ({issue['date']})" if 'date' in issue else ""
            lines.append(f"  - [{issue['portfolio']}]{date_str} {issue['message']}")
        lines.append("")

    if warnings:
        lines.append("[WARNING] WARNINGS:")
        for issue in warnings:
            date_str = f" ({issue['date']})" if 'date' in issue else ""
            lines.append(f"  - [{issue['portfolio']}]{date_str} {issue['message']}")
        lines.append("")

    lines.extend([
        "=" * 60,
        f"Summary: {len(critical)} critical, {len(errors)} errors, {len(warnings)} warnings",
        "=" * 60,
    ])

    return "\n".join(lines)


def send_email_alert(
    recipient: str,
    subject: str,
    body: str,
    html_body: Optional[str] = None,
) -> bool:
    """
    Send email alert using SMTP.

    Requires environment variables:
    - SMTP_SERVER (default: smtp.gmail.com)
    - SMTP_PORT (default: 587)
    - SMTP_USER (sender email)
    - SMTP_PASSWORD (app-specific password for Gmail)

    Returns:
        True if email sent successfully, False otherwise
    """
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")

    if not smtp_user or not smtp_password:
        logger.warning(
            "Email not configured. Set SMTP_USER and SMTP_PASSWORD environment variables."
        )
        return False

    try:
        # Create message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = recipient

        # Plain text version
        msg.attach(MIMEText(body, "plain"))

        # HTML version (if provided)
        if html_body:
            msg.attach(MIMEText(html_body, "html"))

        # Send
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, recipient, msg.as_string())

        logger.info(f"Alert email sent to {recipient}")
        return True

    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False


def generate_html_report(issues: List[Dict], result: Dict) -> str:
    """Generate HTML version of the report for email."""
    critical = [i for i in issues if i.get("severity") == "critical"]
    errors = [i for i in issues if i.get("severity") == "error"]
    warnings = [i for i in issues if i.get("severity") == "warning"]

    status_color = "#dc3545" if critical or errors else "#ffc107" if warnings else "#28a745"
    status_text = "CRITICAL" if critical else "ERRORS" if errors else "WARNINGS" if warnings else "PASS"

    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: {status_color}; color: white; padding: 15px; border-radius: 5px; }}
            .section {{ margin: 20px 0; }}
            .issue {{ padding: 10px; margin: 5px 0; border-left: 4px solid; }}
            .critical {{ border-color: #dc3545; background-color: #f8d7da; }}
            .error {{ border-color: #fd7e14; background-color: #fff3cd; }}
            .warning {{ border-color: #ffc107; background-color: #fffce8; }}
            .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4a5568; color: white; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>🔍 ASM Data Quality Report - {status_text}</h2>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
        </div>

        <div class="summary">
            <strong>Summary:</strong> {len(critical)} critical, {len(errors)} errors, {len(warnings)} warnings
            <br>
            <strong>Portfolios Checked:</strong> {result.get('portfolios_checked', 0)}
            <br>
            <strong>Days Analyzed:</strong> {result.get('days_checked', 7)}
        </div>
    """

    if critical:
        html += """
        <div class="section">
            <h3>🚨 Critical Issues</h3>
        """
        for issue in critical:
            html += f"""
            <div class="issue critical">
                <strong>{issue['portfolio']}</strong>
                {f" ({issue['date']})" if 'date' in issue else ""}
                <br>{issue['message']}
            </div>
            """
        html += "</div>"

    if errors:
        html += """
        <div class="section">
            <h3>❌ Errors</h3>
        """
        for issue in errors:
            html += f"""
            <div class="issue error">
                <strong>{issue['portfolio']}</strong>
                {f" ({issue['date']})" if 'date' in issue else ""}
                <br>{issue['message']}
            </div>
            """
        html += "</div>"

    if warnings:
        html += """
        <div class="section">
            <h3>⚠️ Warnings</h3>
        """
        for issue in warnings:
            html += f"""
            <div class="issue warning">
                <strong>{issue['portfolio']}</strong>
                {f" ({issue['date']})" if 'date' in issue else ""}
                <br>{issue['message']}
            </div>
            """
        html += "</div>"

    if not issues:
        html += """
        <div class="section">
            <h3>✅ All Checks Passed</h3>
            <p>No data quality issues detected.</p>
        </div>
        """

    html += """
        <hr>
        <p style="color: #666; font-size: 12px;">
            This is an automated alert from ASM Portfolio Tracking System.
            <br>
            To investigate, check the Performance Tracker → Scraper View.
        </p>
    </body>
    </html>
    """

    return html


def run_monitor(
    portfolio_name: Optional[str] = None,
    days_back: int = 7,
    output_json: bool = False,
    email_recipient: Optional[str] = None,
) -> Dict:
    """
    Run comprehensive data quality monitoring.

    Args:
        portfolio_name: Specific portfolio to check (or all if None)
        days_back: Number of days to check
        email_recipient: Email address to send alerts (on failure)
        output_json: If True, return JSON-serializable dict

    Returns:
        Dictionary with monitoring results
    """
    from AlphaMachine_core.tracking import get_tracker

    logger.info("Starting data quality monitoring...")

    # Check if today is a trading day
    today = date.today()
    last_trading = get_last_trading_day(today)

    if today != last_trading:
        logger.info(
            f"Note: Today ({today}, {today.strftime('%A')}) is not a trading day. "
            f"Using last trading day: {last_trading} ({last_trading.strftime('%A')})"
        )
    else:
        logger.info(f"Reference date: {today} (trading day)")

    tracker = get_tracker()

    # Get portfolios
    if portfolio_name:
        portfolio = tracker.get_portfolio_by_name(portfolio_name)
        portfolios = [portfolio] if portfolio else []
        if not portfolios:
            logger.error(f"Portfolio '{portfolio_name}' not found")
            return {"error": "Portfolio not found"}
    else:
        portfolios = tracker.list_portfolios(active_only=True)

    logger.info(f"Checking {len(portfolios)} portfolios over last {days_back} days")

    all_issues = []

    # Run checks
    logger.info("Checking NAV anomalies...")
    all_issues.extend(check_nav_anomalies(portfolios, tracker, days_back))

    logger.info("Checking cross-portfolio consistency...")
    all_issues.extend(check_cross_portfolio_consistency(portfolios, tracker))

    logger.info("Checking audit log...")
    all_issues.extend(check_audit_log_issues(days_back))

    logger.info("Reconciling MTD NAV against prices...")
    # Tolerance of 1% to account for weight drift in buy-and-hold portfolios
    all_issues.extend(reconcile_mtd_from_prices(portfolios, tracker, tolerance_pct=1.0))

    logger.info("Checking for missing ticker data...")
    all_issues.extend(check_missing_ticker_data(portfolios, tracker, days_back))

    logger.info("Checking for methodology divergence (share-based vs weight-based)...")
    all_issues.extend(check_methodology_divergence(portfolios, tracker, days_back, tolerance_pct=2.0))

    logger.info("Checking NAV baseline accuracy (shares x price = NAV)...")
    all_issues.extend(check_nav_baseline_accuracy(portfolios, tracker, tolerance_pct=0.5))

    # Generate report
    report = generate_report(all_issues)

    if not output_json:
        print("\n" + report)

    # Summary
    result = {
        "timestamp": datetime.now().isoformat(),
        "portfolios_checked": len(portfolios),
        "days_checked": days_back,
        "total_issues": len(all_issues),
        "critical_count": len([i for i in all_issues if i.get("severity") == "critical"]),
        "error_count": len([i for i in all_issues if i.get("severity") == "error"]),
        "warning_count": len([i for i in all_issues if i.get("severity") == "warning"]),
        "issues": all_issues,
        "status": "PASS" if len(all_issues) == 0 else "FAIL",
    }

    if output_json:
        print(json.dumps(result, indent=2))

    # Send email alert if issues found and email configured
    if email_recipient and all_issues:
        has_critical_or_error = (
            result["critical_count"] > 0 or result["error_count"] > 0
        )

        # Determine subject based on severity
        if result["critical_count"] > 0:
            subject = f"🚨 [CRITICAL] ASM Data Quality Alert - {result['critical_count']} critical issues"
        elif result["error_count"] > 0:
            subject = f"❌ [ERROR] ASM Data Quality Alert - {result['error_count']} errors detected"
        else:
            subject = f"⚠️ [WARNING] ASM Data Quality Alert - {result['warning_count']} warnings"

        # Generate email content
        html_body = generate_html_report(all_issues, result)

        # Send email
        email_sent = send_email_alert(
            recipient=email_recipient,
            subject=subject,
            body=report,
            html_body=html_body,
        )

        result["email_sent"] = email_sent
        result["email_recipient"] = email_recipient

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Data quality monitoring for portfolio tracking"
    )
    parser.add_argument(
        "--portfolio",
        type=str,
        help="Specific portfolio to check (default: all)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to check (default: 7)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--email",
        type=str,
        help="Email address to send alerts on failure",
    )
    parser.add_argument(
        "--always-email",
        action="store_true",
        help="Send email even if no issues found (daily digest)",
    )

    args = parser.parse_args()

    # Get email from args or environment
    email_recipient = args.email or os.getenv("ALERT_EMAIL")

    result = run_monitor(
        portfolio_name=args.portfolio,
        days_back=args.days,
        output_json=args.json,
        email_recipient=email_recipient,
    )

    # Send daily digest even if no issues (if requested)
    if args.always_email and email_recipient and not result.get("email_sent"):
        send_email_alert(
            recipient=email_recipient,
            subject="✅ ASM Data Quality - All Checks Passed",
            body=f"Daily data quality check completed.\n\n"
                 f"Portfolios checked: {result.get('portfolios_checked', 0)}\n"
                 f"Days analyzed: {result.get('days_checked', 7)}\n"
                 f"Status: PASS - No issues detected.\n\n"
                 f"Timestamp: {result.get('timestamp')}",
            html_body=generate_html_report([], result),
        )

    # Exit code based on results
    if result.get("error"):
        sys.exit(1)
    if result.get("critical_count", 0) > 0 or result.get("error_count", 0) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
