"""KPI endpoint — computes inception/YTD/MTD performance metrics for a
portfolio + variant from `portfolio_daily_nav`.

Uses the `daily_return` column directly (not nav-derived) because nav can
contain resets that don't compound cleanly. Streamlit's
`_stitched_nav_series` does the same for the same reason.

Canonical metrics computed:
  total_return, cagr, sharpe, sortino, max_drawdown, calmar, volatility,
  ulcer_index, semi_deviation

Periods:
  inception — from portfolio start to latest NAV
  ytd       — from Jan 1 of latest NAV's year
  mtd       — from start of latest NAV's month

Cache key: sha256(portfolio_id|variant|as_of_date|policy_version)
TTL: 24h (per service/cache_contract.md)
"""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from ..auth import verify_hmac
from ..db import get_pool
from ..lineage import DEFAULT_POLICY_VERSION, build, hash_bytes
from ..models import JobRequest

router = APIRouter()


def _cache_key(portfolio_id: int, variant: str, as_of: date) -> str:
    raw = f"{portfolio_id}|{variant}|{as_of.isoformat()}|{DEFAULT_POLICY_VERSION}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _gips_cagr(returns: pd.Series, total_return: float) -> float:
    """GIPS-strict annualized return.

    GIPS 2020 II.5.A.4: annualize using calendar days / 365.25, NOT trading
    days / 252. Calendar-day basis matches how investors experience
    compounding (you wait wall-clock time, not just trading time).

    The canonical helper uses len(returns)/252 — fine for vol/Sharpe/Sortino
    annualization (annualizing variance over the active-trading window) but
    wrong for return-rate annualization. We override CAGR/Calmar at the
    service layer accordingly.
    """
    if len(returns) < 2:
        return 0.0
    if not isinstance(returns.index, pd.DatetimeIndex):
        # Fallback to canonical's trading-day basis if no date index.
        from AlphaMachine_core.tracking._canonical_metrics import calculate_cagr as _c
        return float(_c(returns))
    start = returns.index.min()
    end = returns.index.max()
    days = (end - start).days
    if days <= 0:
        return 0.0
    n_years = days / 365.25
    if 1.0 + total_return <= 0:
        return float("nan")
    return float((1.0 + total_return) ** (1.0 / n_years) - 1.0)


def _kpis(returns: pd.Series) -> Dict[str, float]:
    """Compute the standard KPI bundle for a daily-return series.

    CAGR and Calmar use GIPS-strict calendar-day annualization
    (calendar_days / 365.25). All other annualized stats (vol, Sharpe,
    Sortino) use the canonical sqrt(252) — that's annualizing variance,
    not time, and is correct on the trading-day basis.
    """
    from AlphaMachine_core.tracking._canonical_metrics import (
        TRADING_DAYS_PER_YEAR,
        calculate_max_drawdown,
        calculate_sharpe,
        calculate_sortino,
        calculate_ulcer_index,
        calculate_volatility,
        downside_semi_deviation,
    )

    if len(returns) < 2:
        return {
            "n_observations": int(len(returns)),
            "total_return": 0.0,
            "cagr": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "volatility": 0.0,
            "ulcer_index": 0.0,
            "semi_deviation": 0.0,
            "trading_days_per_year": int(TRADING_DAYS_PER_YEAR),
            "annualization_basis": "gips-365.25",
        }

    total_return = float(np.prod(1.0 + returns.to_numpy()) - 1.0)
    cagr = _gips_cagr(returns, total_return)
    mdd = float(calculate_max_drawdown(returns))
    calmar = (cagr / abs(mdd)) if (mdd < 0 and np.isfinite(cagr)) else 0.0

    return {
        "n_observations": int(len(returns)),
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": float(calculate_sharpe(returns)),
        "sortino": float(calculate_sortino(returns, no_downside_value=0.0)),
        "max_drawdown": mdd,
        "calmar": calmar,
        "volatility": float(calculate_volatility(returns)),
        "ulcer_index": float(calculate_ulcer_index(returns)),
        "semi_deviation": float(downside_semi_deviation(returns)),
        "trading_days_per_year": int(TRADING_DAYS_PER_YEAR),
        "annualization_basis": "gips-365.25",
    }


async def _fetch_nav(
    pool, portfolio_id: int, variant: str, start: Optional[date] = None, end: Optional[date] = None,
) -> List[Dict[str, Any]]:
    """Returns list of {trade_date, daily_return, nav}."""
    where = ["portfolio_id = $1", "variant = $2"]
    args: List[Any] = [portfolio_id, variant]
    if start:
        args.append(start)
        where.append(f"trade_date >= ${len(args)}")
    if end:
        args.append(end)
        where.append(f"trade_date <= ${len(args)}")
    sql = f"""
        SELECT trade_date, daily_return, nav
        FROM portfolio_daily_nav
        WHERE {' AND '.join(where)}
        ORDER BY trade_date
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *args)
    return [dict(r) for r in rows]


@router.post("/jobs/kpi-single", dependencies=[Depends(verify_hmac)])
async def kpi_single(req: JobRequest) -> Dict[str, Any]:
    """Params:
        portfolioId: int (required)
        variant: str (required, e.g. 'raw', 'conservative', 'hb1')
        asOfDate: ISO date string (optional; defaults to latest NAV date)
    """
    portfolio_id = req.params.get("portfolioId")
    variant = req.params.get("variant")
    as_of_str = req.params.get("asOfDate")

    if not isinstance(portfolio_id, int):
        raise HTTPException(status_code=400, detail="params.portfolioId must be an int")
    if not isinstance(variant, str) or not variant:
        raise HTTPException(status_code=400, detail="params.variant must be a non-empty string")

    pool = await get_pool()
    rows = await _fetch_nav(pool, portfolio_id, variant)
    if not rows:
        raise HTTPException(status_code=404, detail=f"no NAV data for portfolio {portfolio_id} variant {variant}")

    df = pd.DataFrame(rows)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["daily_return"] = df["daily_return"].astype(float)
    df = df.set_index("trade_date").sort_index()

    latest = df.index.max().date()
    as_of = date.fromisoformat(as_of_str) if as_of_str else latest

    df = df[df.index <= pd.Timestamp(as_of)]
    if df.empty:
        raise HTTPException(status_code=404, detail=f"no NAV data on or before {as_of}")

    inception_returns = df["daily_return"]
    ytd_returns = df[df.index >= pd.Timestamp(date(as_of.year, 1, 1))]["daily_return"]
    mtd_returns = df[df.index >= pd.Timestamp(date(as_of.year, as_of.month, 1))]["daily_return"]

    # Build display series: NAV indexed to 100 at first observation, plus
    # drawdown (peak-to-trough percentage from running max). Used by the
    # portal's equity-curve and underwater charts.
    nav_series_arr = (1.0 + inception_returns.to_numpy()).cumprod() * 100.0
    nav_indexed_first = 100.0  # implicit baseline before the first compounded day
    nav_full = np.concatenate([[nav_indexed_first], nav_series_arr])
    peaks = np.maximum.accumulate(nav_full)
    drawdown_pct_full = (nav_full / peaks) - 1.0  # negative or zero
    # Align to the trading-day index (drop the implicit baseline so length
    # matches the return series exactly).
    nav_indexed = nav_full[1:].tolist()
    drawdown_pct = drawdown_pct_full[1:].tolist()
    dates = [d.date().isoformat() for d in inception_returns.index]

    payload: Dict[str, Any] = {
        "portfolioId": portfolio_id,
        "variant": variant,
        "asOf": as_of.isoformat(),
        "firstObservation": df.index.min().date().isoformat(),
        "navAtAsOf": float(df["nav"].iloc[-1]) if not df["nav"].isna().all() else None,
        "series": {
            "dates": dates,
            "navIndexed": nav_indexed,
            "drawdownPct": drawdown_pct,
        },
        "inception": _kpis(inception_returns),
        "ytd": _kpis(ytd_returns),
        "mtd": _kpis(mtd_returns),
    }

    # Lineage: input_data_hash binds to the actual returns we computed on
    body_for_hash = json.dumps(
        {"portfolio_id": portfolio_id, "variant": variant, "as_of": as_of.isoformat(),
         "n_obs": len(inception_returns)},
        sort_keys=True,
    ).encode("utf-8")
    lineage = build(
        input_data_hash=hash_bytes(body_for_hash),
        nav_snapshot_id=f"db:portfolio_daily_nav@{latest.isoformat()}",
    )

    cache_key = _cache_key(portfolio_id, variant, as_of)
    cache_id = str(uuid.uuid4())
    expires_at = datetime.utcnow().replace(microsecond=0) + pd.Timedelta(hours=24)

    async with pool.acquire() as conn, conn.transaction():
        await conn.execute(
            """
            INSERT INTO portal_results_cache
                (id, kind, cache_key, payload,
                 engine_commit_sha, asm_data_sha, input_data_hash,
                 policy_version, nav_snapshot_id, expires_at)
            VALUES ($1, 'kpi-single', $2, $3::jsonb, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (kind, cache_key) DO UPDATE
              SET payload = EXCLUDED.payload,
                  computed_at = now(),
                  expires_at = EXCLUDED.expires_at,
                  engine_commit_sha = EXCLUDED.engine_commit_sha,
                  asm_data_sha = EXCLUDED.asm_data_sha,
                  input_data_hash = EXCLUDED.input_data_hash
            RETURNING id
            """,
            cache_id, cache_key, json.dumps(payload),
            lineage.engine_commit_sha, lineage.asm_data_sha,
            lineage.input_data_hash, lineage.policy_version,
            lineage.nav_snapshot_id, expires_at,
        )
        # If conflict path hit, the returned id may be the existing one — fetch
        existing = await conn.fetchval(
            "SELECT id FROM portal_results_cache WHERE kind='kpi-single' AND cache_key=$1", cache_key
        )
        cache_id = str(existing) if existing else cache_id

        await conn.execute(
            """
            UPDATE portal_compute_jobs
            SET status='complete', completed_at=now(), result_ref=$2
            WHERE id=$1
            """,
            req.jobId, cache_id,
        )

    return {"jobId": req.jobId, "resultRef": cache_id, "payload": payload}
