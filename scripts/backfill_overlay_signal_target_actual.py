#!/usr/bin/env python3
"""
Backfill OverlaySignal.target_allocation and .actual_allocation from S3 history.

Two operating modes (both run by default):

1. UPDATE: For OverlaySignal rows that already exist in the DB, recompute
   target/actual from the canonical schema mapping. Fixes the historical bug
   where tracker.update_daily_nav wrote the same allocation into both columns.

2. CREATE (--create-missing, on by default): For S3 history dates that have
   NO matching OverlaySignal row, insert a fresh row with target/actual and
   a signals dict containing the remaining columns from the S3 row. Used to
   bring late-onboarded variants (rb1, b_average, a_max_up_min_down) up to
   parity with the older variants that have full history.

The schema mapping lives in overlay_adapter._extract_target_and_actual:
- HB1: cv1a_target / active_alloc
- RB1, B_AVERAGE, A_MAX_UP_MIN_DOWN: active_alloc for both (target ≡ actual by design)
- Base models (CV1, CV1A, TV1, TV2A): target_allocation / allocation

Usage:
    python scripts/backfill_overlay_signal_target_actual.py
    python scripts/backfill_overlay_signal_target_actual.py --dry-run
    python scripts/backfill_overlay_signal_target_actual.py --model rb1 --model b_average
    python scripts/backfill_overlay_signal_target_actual.py --no-create-missing  # update-only
"""

import argparse
import logging
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

import numpy as np
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

# Columns whose values are already represented in target_allocation / actual_allocation
# columns and shouldn't be duplicated in the signals JSON. We DO retain
# 'target_allocation' as a key inside the signals dict because tracker.update_daily_nav
# reads it back via signals.get("target_allocation").
SKIP_COLS = {"date", "allocation", "active_alloc", "cv1a_target"}


def _build_signals_dict(target_v: float, hist_row: pd.Series) -> dict:
    """Mirror overlay_adapter._get_allocation_from_history's signals dict shape."""
    signals: dict = {"target_allocation": float(target_v)}
    for col in hist_row.index:
        if col in SKIP_COLS:
            continue
        val = hist_row[col]
        if pd.isna(val):
            continue
        if isinstance(val, (np.integer, np.floating)):
            signals[col] = float(val)
        elif isinstance(val, np.bool_):
            signals[col] = bool(val)
        elif isinstance(val, np.ndarray):
            signals[col] = val.tolist()
        elif isinstance(val, pd.Timestamp):
            signals[col] = val.isoformat()
        else:
            signals[col] = val
    return signals


def backfill_model(model: str, dry_run: bool, create_missing: bool) -> dict:
    """Update existing rows and (optionally) create missing rows for one model."""
    adapter = OverlayAdapter()
    history = adapter._load_allocation_history(model)
    if history is None or history.empty:
        logger.warning(f"  {model}: no S3 history available, skipping")
        return {"model": model, "checked": 0, "updated": 0, "created": 0, "no_real_target": 0}

    if "date" in history.columns:
        history = history.copy()
        history["date"] = pd.to_datetime(history["date"], format="mixed").dt.normalize()
        history = history.drop_duplicates(subset="date", keep="last").set_index("date").sort_index()

    checked = 0
    updated = 0
    created = 0
    no_real_target = 0  # rows where target ≡ actual by design (RB1/B-Avg/A-MUMD or missing column)

    with Session(engine) as session:
        rows = session.exec(
            select(OverlaySignal).where(OverlaySignal.model == model)
        ).all()
        existing_dates = {pd.to_datetime(r.trade_date).normalize() for r in rows}
        logger.info(
            f"  {model}: {len(rows)} OverlaySignal rows in DB, "
            f"{len(history)} in S3 history"
        )

        # --- Pass 1: UPDATE existing rows ---
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

        # --- Pass 2: CREATE missing rows from S3 history ---
        if create_missing:
            for ts, hist_row in history.iterrows():
                if ts in existing_dates:
                    continue
                if isinstance(hist_row, pd.DataFrame):
                    hist_row = hist_row.iloc[-1]

                target_v, actual_v = OverlayAdapter._extract_target_and_actual(model, hist_row)
                if target_v is None or actual_v is None:
                    continue

                if target_v == actual_v:
                    no_real_target += 1

                signals = _build_signals_dict(target_v, hist_row)

                if not dry_run:
                    new_sig = OverlaySignal(
                        trade_date=ts.date(),
                        model=model,
                        target_allocation=Decimal(str(target_v)),
                        actual_allocation=Decimal(str(actual_v)),
                        trade_required=None,  # not derivable from S3 history alone
                        signals=signals,
                        impacts={"source": "s3_backfill"},
                    )
                    session.add(new_sig)

                created += 1

        if not dry_run:
            session.commit()

    logger.info(
        f"  {model}: checked={checked}, updated={updated}, created={created}, "
        f"target≡actual_by_design={no_real_target}"
    )
    return {
        "model": model,
        "checked": checked,
        "updated": updated,
        "created": created,
        "no_real_target": no_real_target,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true", help="Show what would be changed without writing")
    p.add_argument("--model", action="append", default=None,
                   help="Only backfill the listed models (can repeat). Default: all OVERLAY_REGISTRY entries")
    p.add_argument("--no-create-missing", action="store_true",
                   help="Skip creating new rows for S3 dates without an existing OverlaySignal row")
    args = p.parse_args()

    create_missing = not args.no_create_missing
    models = args.model if args.model else list(OVERLAY_REGISTRY.keys())
    unknown = [m for m in models if m not in OVERLAY_REGISTRY]
    if unknown:
        logger.error(f"Unknown models: {unknown}. Available: {list(OVERLAY_REGISTRY.keys())}")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info(f"Backfilling OverlaySignal target/actual for: {models}")
    logger.info(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE WRITE'}, create_missing={create_missing}")
    logger.info("=" * 80)

    summaries = []
    for m in models:
        s = backfill_model(m, dry_run=args.dry_run, create_missing=create_missing)
        summaries.append(s)

    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    total_checked = sum(s["checked"] for s in summaries)
    total_updated = sum(s["updated"] for s in summaries)
    total_created = sum(s["created"] for s in summaries)
    for s in summaries:
        flag = " (target ≡ actual by design)" if s["no_real_target"] == (s["checked"] + s["created"]) and (s["checked"] + s["created"]) > 0 else ""
        logger.info(
            f"  {s['model']:<24}  checked={s['checked']:>5}  "
            f"updated={s['updated']:>5}  created={s['created']:>5}{flag}"
        )
    logger.info(
        f"  TOTAL                     checked={total_checked:>5}  "
        f"updated={total_updated:>5}  created={total_created:>5}"
    )
    if args.dry_run:
        logger.info("\nDRY RUN — no changes written. Re-run without --dry-run to apply.")


if __name__ == "__main__":
    main()
