"""Heartbeat — long-running jobs write their liveness to portal_compute_jobs
every 5s while computing. The Inngest dead-job sweeper (in asm-website)
marks any job as failed if its heartbeat is older than 30s, ensuring crashed
jobs don't sit in 'running' state forever.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from .db import get_pool

HEARTBEAT_INTERVAL_SECONDS = 5
logger = logging.getLogger(__name__)


async def _beat_once(job_id: str) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE portal_compute_jobs SET heartbeat_at = now() WHERE id = $1",
            job_id,
        )


async def _heartbeat_loop(job_id: str, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            await _beat_once(job_id)
        except Exception:
            logger.exception("heartbeat write failed for job %s", job_id)
        try:
            await asyncio.wait_for(stop.wait(), timeout=HEARTBEAT_INTERVAL_SECONDS)
        except asyncio.TimeoutError:
            pass


@asynccontextmanager
async def heartbeat(job_id: str) -> AsyncIterator[None]:
    """Use as: `async with heartbeat(job_id): await long_running_work()`.

    Runs a background task that updates heartbeat_at every 5s. Cancels
    cleanly when the with-block exits.
    """
    stop = asyncio.Event()
    task = asyncio.create_task(_heartbeat_loop(job_id, stop))
    try:
        yield
    finally:
        stop.set()
        try:
            await asyncio.wait_for(task, timeout=10)
        except asyncio.TimeoutError:
            task.cancel()
