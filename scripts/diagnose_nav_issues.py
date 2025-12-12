#!/usr/bin/env python3
"""
NAV Issues Diagnostic Script.

Analyzes all portfolios to identify:
- NAV resets (mature portfolios with NAV near 100)
- Methodology divergence (share-based vs weight-based returns)
- Missing/corrupt data

Usage:
    python scripts/diagnose_nav_issues.py
    python scripts/diagnose_nav_issues.py --portfolio "TW30_EqualWeight"
    python scripts/diagnose_nav_issues.py --days 30
"""

import os
import sys
import logging
import argparse
from datetime import date, timedelta
from decimal import Decimal
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
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def check_nav_reset(
    tracker,
    portfolio,
    mature_threshold_days: int = 30,
    nav_reset_range: tuple = (95, 105),
) -> Optional[Dict]:
    """
    Check if a portfolio has NAV suspiciously near 100 (initial value).

    Returns issue dict if found, None otherwise.
    """
    from AlphaMachine_core.tracking import Variants

    # Check portfolio age
    if not portfolio.start_date:
        return None

    portfolio_age = (date.today() - portfolio.start_date).days
    if portfolio_age < mature_threshold_days:
        return None  # Portfolio is new, NAV near 100 is expected

    # Get latest NAV
    nav_df = tracker.get_nav_series(portfolio.id, Variants.RAW)
    if nav_df.empty:
        return {
            "type": "NO_NAV_DATA",
            "portfolio": portfolio.name,
            "message": "Portfolio has no NAV data",
            "severity": "error",
        }

    latest_nav = nav_df["nav"].iloc[-1]
    latest_date = nav_df.index[-1].date() if hasattr(nav_df.index[-1], 'date') else nav_df.index[-1]

    # Check if NAV is in reset range
    if nav_reset_range[0] < latest_nav < nav_reset_range[1]:
        # Find when the reset might have happened
        # Look for sudden drops to ~100
        reset_date = None
        for i in range(1, len(nav_df)):
            prev_nav = nav_df["nav"].iloc[i-1]
            curr_nav = nav_df["nav"].iloc[i]

            # Detect reset pattern: significant drop landing near 100
            if prev_nav > 110 and nav_reset_range[0] < curr_nav < nav_reset_range[1]:
                reset_date = nav_df.index[i]
                break

        return {
            "type": "NAV_RESET_DETECTED",
            "portfolio": portfolio.name,
            "current_nav": float(latest_nav),
            "nav_date": str(latest_date),
            "portfolio_age_days": portfolio_age,
            "reset_date": str(reset_date.date() if reset_date else "unknown"),
            "message": f"Mature portfolio ({portfolio_age} days) has NAV={latest_nav:.2f} (near initial 100)",
            "severity": "critical",
        }

    return None


def check_methodology_divergence(
    tracker,
    portfolio,
    days_back: int = 7,
    tolerance_pct: float = 2.0,
) -> List[Dict]:
    """
    Compare share-based NAV (stored) vs weight-based NAV (recalculated).

    Returns list of divergence issues found.
    """
    import pandas as pd
    from AlphaMachine_core.tracking import Variants
    from AlphaMachine_core.data_manager import StockDataManager

    issues = []
    today = date.today()
    start_date = today - timedelta(days=days_back)

    # Get stored NAV
    stored_nav_df = tracker.get_nav_series(portfolio.id, Variants.RAW, start_date, today)
    if stored_nav_df.empty or len(stored_nav_df) < 2:
        return issues

    # Get holdings
    holdings = tracker.get_holdings(portfolio.id, today)
    if not holdings:
        return issues

    # Get all tickers
    tickers = [h.ticker for h in holdings]

    # Get weights (normalize)
    weights = {}
    for h in holdings:
        w = float(h.weight) if h.weight else 1.0 / len(holdings)
        weights[h.ticker] = w
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {t: w / total_weight for t, w in weights.items()}

    # Load price data
    dm = StockDataManager()
    price_dicts = dm.get_price_data(
        tickers,
        (start_date - timedelta(days=7)).strftime("%Y-%m-%d"),
        today.strftime("%Y-%m-%d"),
    )

    if not price_dicts:
        return issues

    price_df = pd.DataFrame(price_dicts)
    price_df["trade_date"] = pd.to_datetime(price_df["trade_date"]).dt.date

    # Compare for each day
    for i in range(1, len(stored_nav_df)):
        nav_date = stored_nav_df.index[i]
        nav_date_dt = nav_date.date() if hasattr(nav_date, 'date') else nav_date

        # Get stored return
        stored_return = float(stored_nav_df["daily_return"].iloc[i]) * 100

        # Calculate weight-based return from prices
        curr_prices = price_df[price_df["trade_date"] == nav_date_dt]
        prev_dates = price_df[price_df["trade_date"] < nav_date_dt]["trade_date"].unique()

        if len(prev_dates) == 0 or curr_prices.empty:
            continue

        prev_date = max(prev_dates)
        prev_prices = price_df[price_df["trade_date"] == prev_date]

        if prev_prices.empty:
            continue

        # Calculate expected return
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

        if tickers_used < len(weights) * 0.5:
            continue

        # Compare
        diff = abs(stored_return - expected_return)

        if diff > tolerance_pct:
            issues.append({
                "type": "METHODOLOGY_DIVERGENCE",
                "portfolio": portfolio.name,
                "date": str(nav_date_dt),
                "stored_return": round(stored_return, 4),
                "expected_return": round(expected_return, 4),
                "difference": round(diff, 4),
                "message": f"Stored {stored_return:+.2f}% vs expected {expected_return:+.2f}% (diff: {diff:.2f}%)",
                "severity": "error" if diff > 5.0 else "warning",
            })

    return issues


def check_holdings_integrity(tracker, portfolio) -> List[Dict]:
    """
    Check for holdings data issues:
    - Missing shares
    - Invalid weights
    - Missing entry prices
    """
    issues = []
    today = date.today()

    holdings = tracker.get_holdings(portfolio.id, today)
    if not holdings:
        return [{
            "type": "NO_HOLDINGS",
            "portfolio": portfolio.name,
            "message": "Portfolio has no holdings",
            "severity": "warning",
        }]

    missing_shares = [h.ticker for h in holdings if h.shares is None]
    missing_weights = [h.ticker for h in holdings if h.weight is None]
    missing_entry_price = [h.ticker for h in holdings if h.entry_price is None]

    if missing_shares:
        issues.append({
            "type": "MISSING_SHARES",
            "portfolio": portfolio.name,
            "tickers": missing_shares,
            "count": len(missing_shares),
            "message": f"{len(missing_shares)} holdings missing shares: {', '.join(missing_shares[:5])}{'...' if len(missing_shares) > 5 else ''}",
            "severity": "error",
        })

    if missing_entry_price and not missing_shares:
        # Only warn if shares exist but entry_price missing
        issues.append({
            "type": "MISSING_ENTRY_PRICE",
            "portfolio": portfolio.name,
            "tickers": missing_entry_price,
            "count": len(missing_entry_price),
            "message": f"{len(missing_entry_price)} holdings missing entry_price",
            "severity": "info",
        })

    return issues


def get_nav_history_stats(tracker, portfolio) -> Dict:
    """Get NAV history statistics for a portfolio."""
    from AlphaMachine_core.tracking import Variants

    nav_df = tracker.get_nav_series(portfolio.id, Variants.RAW)

    if nav_df.empty:
        return {
            "total_days": 0,
            "first_date": None,
            "last_date": None,
            "first_nav": None,
            "last_nav": None,
            "min_nav": None,
            "max_nav": None,
            "total_return": None,
        }

    return {
        "total_days": len(nav_df),
        "first_date": str(nav_df.index[0].date() if hasattr(nav_df.index[0], 'date') else nav_df.index[0]),
        "last_date": str(nav_df.index[-1].date() if hasattr(nav_df.index[-1], 'date') else nav_df.index[-1]),
        "first_nav": round(float(nav_df["nav"].iloc[0]), 2),
        "last_nav": round(float(nav_df["nav"].iloc[-1]), 2),
        "min_nav": round(float(nav_df["nav"].min()), 2),
        "max_nav": round(float(nav_df["nav"].max()), 2),
        "total_return": round(float((nav_df["nav"].iloc[-1] / nav_df["nav"].iloc[0] - 1) * 100), 2) if nav_df["nav"].iloc[0] > 0 else None,
    }


def run_diagnostics(
    portfolio_name: Optional[str] = None,
    days_back: int = 7,
    divergence_tolerance: float = 2.0,
) -> Dict:
    """
    Run full diagnostics on portfolios.

    Returns:
        Dict with issues and summary
    """
    from AlphaMachine_core.tracking import get_tracker

    tracker = get_tracker()

    # Get portfolios
    if portfolio_name:
        portfolio = tracker.get_portfolio_by_name(portfolio_name)
        if not portfolio:
            logger.error(f"Portfolio not found: {portfolio_name}")
            return {"error": f"Portfolio not found: {portfolio_name}"}
        portfolios = [portfolio]
    else:
        portfolios = tracker.list_portfolios(active_only=False)

    logger.info(f"Analyzing {len(portfolios)} portfolio(s)...")

    results = {
        "portfolios_analyzed": len(portfolios),
        "issues": [],
        "portfolio_stats": [],
        "summary": {
            "nav_resets": 0,
            "methodology_divergence": 0,
            "missing_shares": 0,
            "no_holdings": 0,
            "healthy": 0,
        },
    }

    for portfolio in portfolios:
        logger.info(f"\n{'='*60}")
        logger.info(f"Portfolio: {portfolio.name}")
        logger.info(f"{'='*60}")

        # Get NAV stats
        stats = get_nav_history_stats(tracker, portfolio)
        stats["portfolio"] = portfolio.name
        stats["start_date"] = str(portfolio.start_date) if portfolio.start_date else None
        results["portfolio_stats"].append(stats)

        logger.info(f"  Start date: {portfolio.start_date}")
        logger.info(f"  NAV history: {stats['total_days']} days")
        if stats['first_nav']:
            logger.info(f"  NAV range: {stats['first_nav']} -> {stats['last_nav']} (total: {stats['total_return']}%)")

        portfolio_issues = []

        # Check 1: NAV Reset
        reset_issue = check_nav_reset(tracker, portfolio)
        if reset_issue:
            portfolio_issues.append(reset_issue)
            results["summary"]["nav_resets"] += 1
            logger.warning(f"  ⚠️  NAV RESET: {reset_issue['message']}")

        # Check 2: Holdings Integrity
        holdings_issues = check_holdings_integrity(tracker, portfolio)
        for issue in holdings_issues:
            portfolio_issues.append(issue)
            if issue["type"] == "MISSING_SHARES":
                results["summary"]["missing_shares"] += 1
                logger.warning(f"  ⚠️  MISSING SHARES: {issue['message']}")
            elif issue["type"] == "NO_HOLDINGS":
                results["summary"]["no_holdings"] += 1
                logger.warning(f"  ⚠️  NO HOLDINGS: {issue['message']}")

        # Check 3: Methodology Divergence (only if we have holdings with shares)
        if not any(i["type"] in ["MISSING_SHARES", "NO_HOLDINGS"] for i in holdings_issues):
            divergence_issues = check_methodology_divergence(
                tracker, portfolio, days_back, divergence_tolerance
            )
            for issue in divergence_issues:
                portfolio_issues.append(issue)
                results["summary"]["methodology_divergence"] += 1
                logger.warning(f"  ⚠️  DIVERGENCE on {issue['date']}: {issue['message']}")

        # Add all issues to results
        results["issues"].extend(portfolio_issues)

        if not portfolio_issues:
            results["summary"]["healthy"] += 1
            logger.info("  ✅ No issues detected")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose NAV data quality issues"
    )
    parser.add_argument(
        "--portfolio",
        type=str,
        help="Specific portfolio name to analyze",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to check for divergence (default: 7)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=2.0,
        help="Divergence tolerance percentage (default: 2.0)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("NAV Issues Diagnostic")
    logger.info("=" * 60)
    logger.info(f"Date: {date.today()}")
    logger.info(f"Checking last {args.days} days")
    logger.info(f"Divergence tolerance: {args.tolerance}%")

    results = run_diagnostics(
        portfolio_name=args.portfolio,
        days_back=args.days,
        divergence_tolerance=args.tolerance,
    )

    if args.json:
        import json
        print(json.dumps(results, indent=2, default=str))
        return

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("=" * 60)

    summary = results["summary"]
    logger.info(f"  Portfolios analyzed: {results['portfolios_analyzed']}")
    logger.info(f"  Healthy portfolios:  {summary['healthy']}")
    logger.info(f"  NAV resets detected: {summary['nav_resets']}")
    logger.info(f"  Methodology divergence issues: {summary['methodology_divergence']}")
    logger.info(f"  Missing shares:      {summary['missing_shares']}")
    logger.info(f"  No holdings:         {summary['no_holdings']}")

    # List critical issues
    critical_issues = [i for i in results["issues"] if i.get("severity") in ["critical", "error"]]

    if critical_issues:
        logger.info("\n" + "-" * 60)
        logger.info("CRITICAL/ERROR ISSUES REQUIRING ACTION:")
        logger.info("-" * 60)
        for issue in critical_issues:
            logger.info(f"  [{issue['severity'].upper()}] {issue['portfolio']}: {issue['message']}")

    # Recommendations
    logger.info("\n" + "-" * 60)
    logger.info("RECOMMENDATIONS:")
    logger.info("-" * 60)

    if summary["nav_resets"] > 0:
        logger.info("  1. Run recalculate_nav.py for portfolios with NAV reset")
        logger.info("     python scripts/recalculate_nav.py --dry-run")
        logger.info("     python scripts/recalculate_nav.py")

    if summary["missing_shares"] > 0:
        logger.info("  2. Run backfill_holdings_shares.py to populate shares")
        logger.info("     python scripts/backfill_holdings_shares.py")

    if summary["methodology_divergence"] > 0:
        logger.info("  3. Investigate divergence - may indicate data corruption")
        logger.info("     Review holdings and price data for affected dates")

    if summary["healthy"] == results["portfolios_analyzed"]:
        logger.info("  All portfolios are healthy! No action needed.")


if __name__ == "__main__":
    main()
