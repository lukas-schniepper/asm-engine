"""Sortino recompute endpoint — sync, sub-second. Wraps the canonical
`_canonical_metrics.calculate_sortino` helper.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from ..auth import verify_hmac
from ..db import get_pool
from ..lineage import build, hash_bytes
from ..models import JobRequest

router = APIRouter()


@router.post("/jobs/sortino-recompute", dependencies=[Depends(verify_hmac)])
async def sortino_recompute(req: JobRequest) -> Dict[str, Any]:
    """Params shape:
        { "returns": [<float>, ...], "mar": 0.0, "annualize": true }
    """
    # Lazy-import canonical helper. Keeps service startup fast and lets
    # the rest of the service load even if asm-data submodule is broken.
    from AlphaMachine_core.tracking._canonical_metrics import calculate_sortino

    raw_returns = req.params.get("returns")
    if not isinstance(raw_returns, list) or not raw_returns:
        raise HTTPException(status_code=400, detail="params.returns must be a non-empty list")

    try:
        returns: List[float] = [float(x) for x in raw_returns]
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="params.returns must be numbers")

    mar = float(req.params.get("mar", 0.0))
    no_downside_value = float(req.params.get("no_downside_value", 0.0))

    sortino = calculate_sortino(
        pd.Series(returns),
        mar=mar,
        no_downside_value=no_downside_value,
    )

    payload: Dict[str, Any] = {
        "sortino": float(sortino),
        "n_observations": len(returns),
        "mar": mar,
    }

    body_bytes = json.dumps(req.params, sort_keys=True, default=str).encode("utf-8")
    lineage = build(
        input_data_hash=hash_bytes(body_bytes),
        nav_snapshot_id=req.params.get("nav_snapshot_id", "ad-hoc"),
    )

    pool = await get_pool()
    cache_id = str(uuid.uuid4())
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute(
            """
            INSERT INTO portal_results_cache
                (id, kind, cache_key, payload,
                 engine_commit_sha, asm_data_sha, input_data_hash,
                 policy_version, nav_snapshot_id)
            VALUES ($1, 'sortino', $2, $3::jsonb, $4, $5, $6, $7, $8)
            ON CONFLICT (kind, cache_key) DO UPDATE
              SET payload = EXCLUDED.payload, computed_at = now()
            """,
            cache_id, req.jobId, json.dumps(payload),
            lineage.engine_commit_sha, lineage.asm_data_sha,
            lineage.input_data_hash, lineage.policy_version,
            lineage.nav_snapshot_id,
        )
        await conn.execute(
            """
            UPDATE portal_compute_jobs
            SET status = 'complete', completed_at = now(), result_ref = $2
            WHERE id = $1
            """,
            req.jobId, cache_id,
        )

    return {"jobId": req.jobId, "resultRef": cache_id, "payload": payload}
