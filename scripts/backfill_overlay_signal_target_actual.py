#!/usr/bin/env python3
"""
Backfill OverlaySignal.target_allocation and .actual_allocation from S3 history.

The bug: tracker.update_daily_nav (line 672-673 historically) wrote the same
allocation value into BOTH target_allocation and actual_allocation columns. So
every existing OverlaySignal row has target == actual, hiding the real
divergence between the raw rule signal and the executed allocation (after
rebalance threshold or mirror logic).

This script reads S3 allocation_history.csv per model, extracts the real
target/actual via the canonical schema mapping (in
overlay_adapter._extract_target_and_actual), and updates each existing
OverlaySignal row in the DB.

Usage:
    python scripts/backfill_overlay_signal_target_actual.py
    python scripts/backfill_overlay_signal_target_actual.py --dry-run
    python scripts/backfill_overlay_signal_target_actual.py --model hb1 conservative
"""

import argparse
import logging
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd
from sqlmodel import Session, select

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from AlphaMachine_core.db import engine
from AlphaMachine_core.tracking.models import OverlaySignal
from AlphaMachine_core.tracking.overlay_adapter import (
    OVERLAY_REGISTRY, OverlayAdapter,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def backfill_model(model: str, dry_run: bool) -> dict:
    """Update all OverlaySignal rows for one model."""
    adapter = OverlayAdapter()
    history = adapter._load_allocation_history(model)
    if history is None or history.empty:
        logger.warning(f"  {model}: no S3 history available, skipping")
        return {"model": model, "checked": 0, "updated": 0, "no_real_target": 0}

    if "date" in history.columns:
        history = history.copy()
        history["date"] = pd.to_datetime(history["date"], format="mixed").dt.normalize()
        history = history.drop_duplicates(subset="date", keep="last").set_index("date").sort_index()

    checked = 0
    updated = 0
    no_real_target = 0  # rows where target ≡ actual by design (RB1/B-Avg/A-MUMD or missing column)

    with Session(engine) as session:
        rows = session.exec(
            select(OverlaySignal).where(OverlaySignal.model == model)
        ).all()
        logger.info(f"  {model}: {len(rows)} OverlaySignal rows in DB, {len(history)} in S3 history")

        for sig in rows:
            checked += 1
            ts = pd.to_datetime(sig.trade_date)
            if ts not in history.index:
                continue
            hist_row = history.loc[ts]
            if isinstance(hist_row, pd.DataFrame):
                hist_row = hist_row.iloc[-1]

            target_v, actual_v = OverlayAdapter._extract_target_and_actual(model, hist_row)
            if target_v is None or actual_v is None:
                continue

            # Track whether this model genuinely has a separate target column
            # (RB1, B_AVERAGE, A_MAX_UP_MIN_DOWN: target ≡ actual by design).
            if target_v == actual_v:
                no_real_target += 1

            old_target = float(sig.target_allocation) if sig.target_allocation is not None else None
            old_actual = float(sig.actual_allocation) if sig.actual_allocation is not None else None

            target_changed = (
                old_target is None or abs(old_target - target_v) > 1e-9
            )
            actual_changed = (
                old_actual is None or abs(old_actual - actual_v) > 1e-9
            )

            if not (target_changed or actual_changed):
                continue

            if not dry_run:
                sig.target_allocation = Decimal(str(target_v))
                sig.actual_allocation = Decimal(str(actual_v))
                session.add(sig)

            updated += 1

        if not dry_run:
            session.commit()

    logger.info(
        f"  {model}: checked={checked}, updated={updated}, "
        f"target≡actual_by_design={no_real_target}"
    )
    return {"model": model, "checked": checked, "updated": updated, "no_real_target": no_real_target}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true", help="Show what would be changed without writing")
    p.add_argument("--model", action="append", default=None,
                   help="Only backfill the listed models (can repeat). Default: all OVERLAY_REGISTRY entries")
    args = p.parse_args()

    models = args.model if args.model else list(OVERLAY_REGISTRY.keys())
    unknown = [m for m in models if m not in OVERLAY_REGISTRY]
    if unknown:
        logger.error(f"Unknown models: {unknown}. Available: {list(OVERLAY_REGISTRY.keys())}")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info(f"Backfilling OverlaySignal target/actual for: {models}")
    logger.info(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE WRITE'}")
    logger.info("=" * 80)

    summaries = []
    for m in models:
        s = backfill_model(m, dry_run=args.dry_run)
        summaries.append(s)

    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    total_checked = sum(s["checked"] for s in summaries)
    total_updated = sum(s["updated"] for s in summaries)
    for s in summaries:
        flag = " (target ≡ actual by design)" if s["no_real_target"] == s["checked"] and s["checked"] > 0 else ""
        logger.info(f"  {s['model']:<24}  checked={s['checked']:>5}  updated={s['updated']:>5}{flag}")
    logger.info(f"  TOTAL                     checked={total_checked:>5}  updated={total_updated:>5}")
    if args.dry_run:
        logger.info("\nDRY RUN — no changes written. Re-run without --dry-run to apply.")


if __name__ == "__main__":
    main()
