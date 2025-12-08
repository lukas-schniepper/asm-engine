#!/usr/bin/env python3
"""Compare December returns before and after GIPS migration."""

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

def capture_new_returns():
    """Capture December 2025 returns after GIPS migration."""
    tracker = get_tracker()
    portfolios = tracker.list_portfolios(active_only=True)

    results = []
    for portfolio in portfolios:
        for variant in Variants.all():
            try:
                nav_df = tracker.get_nav_series(
                    portfolio.id,
                    variant,
                    start_date=date(2025, 12, 1),
                    end_date=date(2025, 12, 31)
                )

                if nav_df.empty:
                    continue

                daily_returns = nav_df['daily_return'].dropna()
                if len(daily_returns) > 0:
                    dec_return = (1 + daily_returns).prod() - 1
                else:
                    dec_return = 0.0

                results.append({
                    'portfolio': portfolio.name,
                    'variant': variant,
                    'new_dec_return': dec_return,
                    'new_dec_return_pct': f"{dec_return*100:.4f}%",
                })
            except Exception as e:
                print(f"Error for {portfolio.name} {variant}: {e}")

    return pd.DataFrame(results)

def compare_returns():
    """Compare baseline vs new returns."""
    print("="*80)
    print("GIPS MIGRATION COMPARISON REPORT")
    print("="*80)

    # Load baseline
    baseline_path = project_root / 'baseline_december_returns.csv'
    if not baseline_path.exists():
        print(f"ERROR: Baseline file not found at {baseline_path}")
        print("Run capture_baseline_returns.py first!")
        return

    baseline_df = pd.read_csv(baseline_path)
    print(f"Loaded baseline: {len(baseline_df)} records")

    # Capture new returns
    print("\nCapturing new returns...")
    new_df = capture_new_returns()
    print(f"Captured new returns: {len(new_df)} records")

    # Merge
    comparison = pd.merge(
        baseline_df[['portfolio', 'variant', 'dec_return', 'dec_return_pct']],
        new_df[['portfolio', 'variant', 'new_dec_return', 'new_dec_return_pct']],
        on=['portfolio', 'variant'],
        how='outer'
    )

    # Calculate difference
    comparison['difference'] = comparison['new_dec_return'] - comparison['dec_return']
    comparison['diff_pct'] = comparison['difference'].apply(lambda x: f"{x*100:.4f}%" if pd.notna(x) else "N/A")

    # Categorize impact
    def categorize_impact(row):
        if pd.isna(row['difference']):
            return "No data"
        diff = abs(row['difference']) * 100
        if diff < 0.01:
            return "Negligible (<0.01%)"
        elif diff < 0.1:
            return "Small (0.01-0.1%)"
        elif diff < 0.5:
            return "Medium (0.1-0.5%)"
        else:
            return "High (>0.5%)"

    comparison['impact'] = comparison.apply(categorize_impact, axis=1)

    # Save full report
    output_path = project_root / 'gips_migration_comparison.csv'
    comparison.to_csv(output_path, index=False)
    print(f"\nFull report saved to: {output_path}")

    # Print summary table for RAW variant
    print("\n" + "="*80)
    print("DECEMBER 2025 RETURNS COMPARISON (RAW variant)")
    print("="*80)

    raw_comparison = comparison[comparison['variant'] == 'raw'].copy()
    raw_comparison = raw_comparison.sort_values('difference', ascending=False)

    print(f"{'Portfolio':<45} {'Old':<12} {'New':<12} {'Diff':<12} {'Impact'}")
    print("-"*100)

    for _, row in raw_comparison.iterrows():
        old = row['dec_return_pct'] if pd.notna(row['dec_return']) else "N/A"
        new = row['new_dec_return_pct'] if pd.notna(row['new_dec_return']) else "N/A"
        diff = row['diff_pct']
        impact = row['impact']
        print(f"{row['portfolio']:<45} {old:<12} {new:<12} {diff:<12} {impact}")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    valid_diffs = comparison['difference'].dropna()
    if len(valid_diffs) > 0:
        print(f"Total comparisons: {len(valid_diffs)}")
        print(f"Average difference: {valid_diffs.mean()*100:.4f}%")
        print(f"Max positive diff: {valid_diffs.max()*100:.4f}%")
        print(f"Max negative diff: {valid_diffs.min()*100:.4f}%")
        print(f"Std deviation: {valid_diffs.std()*100:.4f}%")

    # Impact breakdown
    print("\nImpact breakdown:")
    print(comparison['impact'].value_counts().to_string())

    return comparison

if __name__ == "__main__":
    compare_returns()
