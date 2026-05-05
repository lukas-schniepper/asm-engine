"""Job status polling — portal hits this to check on long-running jobs."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..auth import verify_hmac
from ..db import get_pool
from ..models import JobStatus

router = APIRouter()


@router.get("/jobs/{job_id}", response_model=JobStatus, dependencies=[Depends(verify_hmac)])
async def job_status(job_id: str) -> JobStatus:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, kind, status, result_ref::text AS result_ref, error,
                   started_at, completed_at, heartbeat_at
            FROM portal_compute_jobs WHERE id=$1
            """,
            job_id,
        )
    if row is None:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id}")

    return JobStatus(
        id=str(row["id"]),
        kind=row["kind"],
        status=row["status"],
        result_ref=row["result_ref"],
        error=row["error"],
        started_at=row["started_at"].isoformat() if row["started_at"] else None,
        completed_at=row["completed_at"].isoformat() if row["completed_at"] else None,
        heartbeat_at=row["heartbeat_at"].isoformat() if row["heartbeat_at"] else None,
    )
