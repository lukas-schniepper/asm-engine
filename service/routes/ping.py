"""Ping endpoint — the trivial round-trip that gates Phase 1.

Sync, sub-second. Used by /admin/engine-ping in the portal to verify the
full chain works: HMAC auth → DB write → result cache → Inngest callback.
"""
from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Depends

from ..auth import verify_hmac
from ..config import settings
from ..db import get_pool
from ..lineage import build, hash_bytes
from ..models import PingRequest, PingResponse

router = APIRouter()


@router.post("/jobs/ping", response_model=PingResponse, dependencies=[Depends(verify_hmac)])
async def ping(req: PingRequest) -> PingResponse:
    pool = await get_pool()

    payload = {"message": "pong", "received": req.params}
    body_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
    lineage = build(
        input_data_hash=hash_bytes(body_bytes),
        nav_snapshot_id="ping",  # ping has no NAV input
    )
    cache_id = str(uuid.uuid4())

    async with pool.acquire() as conn, conn.transaction():
        await conn.execute(
            """
            INSERT INTO portal_results_cache
                (id, kind, cache_key, payload,
                 engine_commit_sha, asm_data_sha, input_data_hash,
                 policy_version, nav_snapshot_id)
            VALUES ($1, 'ping', $2, $3::jsonb, $4, $5, $6, $7, $8)
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

    return PingResponse(
        jobId=req.jobId,
        message="pong",
        engine_commit_sha=settings.engine_commit_sha if settings else "unknown",
        asm_data_sha=settings.asm_data_sha if settings else "unknown",
        received_params=req.params,
    )
