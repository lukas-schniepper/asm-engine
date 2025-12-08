#!/usr/bin/env python3
"""Capture baseline December returns for all portfolios before GIPS migration."""

import os
import sys
from pathlib import Path
from datetime import date

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
from AlphaMachine_core.tracking import get_tracker, Variants

def capture_baseline_returns():
    """Capture December 2025 returns for all portfolios."""
    tracker = get_tracker()

    # Get all active portfolios
    portfolios = tracker.list_portfolios(active_only=True)
    print(f"Found {len(portfolios)} active portfolios")

    results = []

    for portfolio in portfolios:
        print(f"\nProcessing: {portfolio.name}")

        # Get NAV series for December 2025
        for variant in Variants.all():
            try:
                nav_df = tracker.get_nav_series(
                    portfolio.id,
                    variant,
                    start_date=date(2025, 12, 1),
                    end_date=date(2025, 12, 31)
                )

                if nav_df.empty:
                    print(f"  {variant}: No data")
                    continue

                # Calculate December return (compounded)
                daily_returns = nav_df['daily_return'].dropna()
                if len(daily_returns) > 0:
                    dec_return = (1 + daily_returns).prod() - 1
                else:
                    dec_return = 0.0

                # Also get start and end NAV
                start_nav = nav_df['nav'].iloc[0] if not nav_df.empty else None
                end_nav = nav_df['nav'].iloc[-1] if not nav_df.empty else None

                results.append({
                    'portfolio': portfolio.name,
                    'variant': variant,
                    'dec_return': dec_return,
                    'dec_return_pct': f"{dec_return*100:.4f}%",
                    'start_nav': start_nav,
                    'end_nav': end_nav,
                    'trading_days': len(daily_returns)
                })

                print(f"  {variant}: {dec_return*100:.4f}% ({len(daily_returns)} days)")

            except Exception as e:
                print(f"  {variant}: Error - {e}")

    # Save to CSV
    df = pd.DataFrame(results)
    output_path = project_root / 'baseline_december_returns.csv'
    df.to_csv(output_path, index=False)
    print(f"\n\nBaseline saved to: {output_path}")
    print(f"Total records: {len(df)}")

    # Print summary table
    print("\n" + "="*80)
    print("BASELINE DECEMBER 2025 RETURNS (before GIPS migration)")
    print("="*80)

    # Pivot for easier reading
    if not df.empty:
        pivot = df.pivot(index='portfolio', columns='variant', values='dec_return_pct')
        print(pivot.to_string())

    return df

if __name__ == "__main__":
    capture_baseline_returns()
