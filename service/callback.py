"""Send compute/job.completed events back to Inngest when a long-running
job finishes. Inngest's step.waitForEvent in the portal picks them up.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import httpx

from .config import settings

logger = logging.getLogger(__name__)


async def emit_completed(
    job_id: str,
    result_ref: str | None,
    status_str: str = "complete",
    error: str | None = None,
) -> None:
    """POST a compute/job.completed event to Inngest.

    Silently logs (does not raise) on transport failure — the dead-job
    sweeper will eventually mark stale jobs failed if Inngest never gets
    the callback. We don't want to block job completion on Inngest health.
    """
    if settings is None:
        return

    payload: Dict[str, Any] = {
        "name": "compute/job.completed",
        "data": {
            "jobId": job_id,
            "status": status_str,
            "resultRef": result_ref,
            "error": error,
        },
    }
    headers = {"Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(settings.inngest_event_url, json=payload, headers=headers)
            r.raise_for_status()
    except Exception:
        logger.exception("Inngest callback failed for job %s", job_id)
