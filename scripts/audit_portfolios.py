#!/usr/bin/env python3
"""Detailed Portfolio Audit Script."""

import os
import sys
from pathlib import Path
from datetime import date

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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

from sqlmodel import Session, select
from AlphaMachine_core.db import engine
from AlphaMachine_core.tracking.models import (
    PortfolioDefinition, PortfolioDailyNAV, PortfolioHolding, PortfolioMetric
)

def check_metrics_sanity(m):
    """Check a PortfolioMetric object for sanity."""
    issues = []
    tr = float(m.total_return) if m.total_return is not None else None
    if tr is not None and abs(tr) > 10:
        issues.append(f"Total return {tr*100:.1f}% seems extreme")
    cagr = float(m.cagr) if m.cagr is not None else None
    if cagr is not None and abs(cagr) > 5:
        issues.append(f"CAGR {cagr*100:.1f}% seems extreme")
    sharpe = float(m.sharpe_ratio) if m.sharpe_ratio is not None else None
    if sharpe is not None and abs(sharpe) > 10:
        issues.append(f"Sharpe {sharpe:.2f} seems extreme")
    vol = float(m.volatility) if m.volatility is not None else None
    if vol is not None and vol > 2:
        issues.append(f"Volatility {vol*100:.1f}% seems extreme")
    mdd = float(m.max_drawdown) if m.max_drawdown is not None else None
    if mdd is not None and mdd < -1:
        issues.append(f"Max drawdown {mdd*100:.1f}% is impossible")
    return issues

def audit_portfolio(session, p):
    issues = []
    print(f"\n{'='*80}")
    print(f"PORTFOLIO: {p.name} (id={p.id})")
    print(f"{'='*80}")
    print(f"Start Date: {p.start_date} | Source: {p.source}")

    navs_raw = session.exec(
        select(PortfolioDailyNAV).where(PortfolioDailyNAV.portfolio_id == p.id)
        .where(PortfolioDailyNAV.variant == "raw")
        .order_by(PortfolioDailyNAV.trade_date)
    ).all()

    if not navs_raw:
        print("[!!] NO NAV DATA")
        return ["No NAV data"]

    print(f"\n--- NAV Data (raw) ---")
    print(f"Records: {len(navs_raw)} | Range: {navs_raw[0].trade_date} to {navs_raw[-1].trade_date}")
    print(f"First NAV: {navs_raw[0].nav:.4f} | Last NAV: {navs_raw[-1].nav:.4f}")

    first_nav = navs_raw[0].nav
    if first_nav < 10 or first_nav > 1000:
        msg = f"First NAV {first_nav:.2f} unusual (expected ~100)"
        print(f"[!!] {msg}")
        issues.append(msg)

    actual_return = (navs_raw[-1].nav / navs_raw[0].nav - 1)
    stored_cum_return = navs_raw[-1].cumulative_return
    print(f"Calculated Return: {actual_return*100:.2f}% | Stored: {stored_cum_return*100:.2f}%")

    if abs(actual_return - stored_cum_return) > 0.01:
        msg = f"Return mismatch: calc={actual_return*100:.2f}% vs stored={stored_cum_return*100:.2f}%"
        print(f"[!!] {msg}")
        issues.append(msg)

    holdings = session.exec(select(PortfolioHolding).where(PortfolioHolding.portfolio_id == p.id)).all()
    holding_dates = sorted(set(h.effective_date for h in holdings))
    nav_dates = [n.trade_date for n in navs_raw]

    print(f"\n--- Holdings ---")
    print(f"Records: {len(holdings)} | Dates: {holding_dates}")
    with_shares = len([h for h in holdings if h.shares])
    without_shares = len([h for h in holdings if not h.shares])
    print(f"With shares: {with_shares} | Without: {without_shares}")

    if without_shares:
        msg = f"{without_shares} holdings missing shares"
        print(f"[!!] {msg}")
        issues.append(msg)

    if holding_dates and nav_dates:
        first_holding = min(holding_dates)
        first_nav_date = min(nav_dates)
        if first_holding > first_nav_date:
            gap = (first_holding - first_nav_date).days
            msg = f"Holdings start {gap}d after NAV"
            print(f"[!!] {msg}")
            issues.append(msg)

    metrics = session.exec(select(PortfolioMetric).where(PortfolioMetric.portfolio_id == p.id)).all()
    print(f"\n--- Metrics ---")

    if metrics:
        # Group by variant and period_type
        by_variant = {}
        for m in metrics:
            key = f"{m.variant}_{m.period_type}"
            by_variant[key] = m

        for key, m in sorted(by_variant.items()):
            print(f"\n{m.variant.upper()} ({m.period_type}): {m.period_start} to {m.period_end}")

            def fmt(val, pct=False):
                if val is None:
                    return "N/A"
                v = float(val)
                return f"{v*100:.2f}%" if pct else f"{v:.2f}"

            print(f"  Total Return: {fmt(m.total_return, True)}")
            print(f"  CAGR: {fmt(m.cagr, True)}")
            print(f"  Volatility: {fmt(m.volatility, True)}")
            print(f"  Max Drawdown: {fmt(m.max_drawdown, True)}")
            print(f"  Sharpe: {fmt(m.sharpe_ratio)}")
            print(f"  Sortino: {fmt(m.sortino_ratio)}")
            print(f"  Calmar: {fmt(m.calmar_ratio)}")
            print(f"  Win Rate: {fmt(m.win_rate, True)}")

            for issue in check_metrics_sanity(m):
                print(f"  [!!] {issue}")
                issues.append(f"{m.variant}_{m.period_type}: {issue}")
    else:
        print("No metrics stored")

    print(f"\n--- Variants ---")
    variants = session.exec(
        select(PortfolioDailyNAV.variant).where(PortfolioDailyNAV.portfolio_id == p.id).distinct()
    ).all()
    for v in variants:
        vnavs = session.exec(
            select(PortfolioDailyNAV).where(PortfolioDailyNAV.portfolio_id == p.id)
            .where(PortfolioDailyNAV.variant == v).order_by(PortfolioDailyNAV.trade_date)
        ).all()
        if vnavs:
            ret = (vnavs[-1].nav / vnavs[0].nav - 1) * 100 if vnavs[0].nav > 0 else 0
            print(f"{v}: {len(vnavs)} recs, {vnavs[0].nav:.2f} -> {vnavs[-1].nav:.2f} ({ret:+.2f}%)")

    return issues

def main():
    print("=" * 80)
    print("DETAILED PORTFOLIO METRICS AUDIT")
    print("=" * 80)

    all_issues = {}
    with Session(engine) as session:
        portfolios = session.exec(
            select(PortfolioDefinition).where(PortfolioDefinition.is_active == True)
            .order_by(PortfolioDefinition.id)
        ).all()
        for p in portfolios:
            issues = audit_portfolio(session, p)
            if issues:
                all_issues[p.name] = issues

    print("\n" + "=" * 80)
    print("AUDIT SUMMARY")
    print("=" * 80)
    if all_issues:
        print("\nPortfolios with issues:")
        for name, issues in all_issues.items():
            print(f"\n{name}:")
            for issue in issues:
                print(f"  - {issue}")
    else:
        print("\nAll portfolios passed!")

if __name__ == "__main__":
    main()
