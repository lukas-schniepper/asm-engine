"""Optimize endpoint — async. Returns 202; runs Optuna study in background.

Phase 0.2 ships this as a STUB. Phase 4 replaces with real `optimize_params.run_optimizer`
+ Optuna RDBStorage to persist trials to Postgres.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from ..auth import verify_hmac
from ..callback import emit_completed
from ..db import get_pool
from ..heartbeat import heartbeat
from ..lineage import build, hash_bytes
from ..models import JobAccepted, JobRequest

router = APIRouter()
logger = logging.getLogger(__name__)


async def _run_optimize_stub(job_id: str, params: dict) -> None:
    pool = await get_pool()
    try:
        async with heartbeat(job_id):
            await asyncio.sleep(3)
            payload = {
                "stub": True,
                "message": "Phase 0.2 stub — real Optuna study lands in Phase 4",
                "params_received": params,
            }
            body_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
            lineage = build(
                input_data_hash=hash_bytes(body_bytes),
                nav_snapshot_id=params.get("nav_snapshot_id", "stub"),
            )
            cache_id = str(uuid.uuid4())

            async with pool.acquire() as conn, conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO portal_results_cache
                        (id, kind, cache_key, payload,
                         engine_commit_sha, asm_data_sha, input_data_hash,
                         policy_version, nav_snapshot_id)
                    VALUES ($1, 'optimize', $2, $3::jsonb, $4, $5, $6, $7, $8)
                    ON CONFLICT (kind, cache_key) DO UPDATE
                      SET payload = EXCLUDED.payload, computed_at = now()
                    """,
                    cache_id, job_id, json.dumps(payload),
                    lineage.engine_commit_sha, lineage.asm_data_sha,
                    lineage.input_data_hash, lineage.policy_version,
                    lineage.nav_snapshot_id,
                )
                await conn.execute(
                    """
                    UPDATE portal_compute_jobs
                    SET status='complete', completed_at=now(), result_ref=$2
                    WHERE id=$1
                    """,
                    job_id, cache_id,
                )

            await emit_completed(job_id=job_id, result_ref=cache_id, status_str="complete")
    except Exception as exc:
        logger.exception("optimize job %s failed", job_id)
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE portal_compute_jobs SET status='failed', completed_at=now(), error=$2 WHERE id=$1",
                job_id, str(exc)[:500],
            )
        await emit_completed(job_id=job_id, result_ref=None, status_str="failed", error=str(exc)[:500])


@router.post("/jobs/optimize", response_model=JobAccepted, status_code=202, dependencies=[Depends(verify_hmac)])
async def start_optimize(req: JobRequest, background: BackgroundTasks) -> JobAccepted:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, status FROM portal_compute_jobs WHERE id=$1", req.jobId
        )
        if row is None:
            raise HTTPException(status_code=404, detail=f"unknown job {req.jobId}")
        if row["status"] not in ("queued", "running"):
            raise HTTPException(
                status_code=409,
                detail=f"job {req.jobId} already in terminal state ({row['status']})",
            )
        await conn.execute(
            "UPDATE portal_compute_jobs SET status='running', started_at=COALESCE(started_at, now()), heartbeat_at=now() WHERE id=$1",
            req.jobId,
        )

    background.add_task(_run_optimize_stub, req.jobId, req.params)
    return JobAccepted(jobId=req.jobId, status="running")
