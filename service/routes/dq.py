"""Data quality gates — sync. Validates NAV inputs before metrics compute.

Stub for Phase 0.2; full DQ rules land alongside Phase 1.
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends

from ..auth import verify_hmac
from ..models import JobRequest

router = APIRouter()


@router.post("/jobs/dq", dependencies=[Depends(verify_hmac)])
async def run_dq(req: JobRequest) -> Dict[str, Any]:
    """Phase 0.2 stub. Phase 1 fills in:
        - stale-price detector (no update >5 trading days)
        - missing-day check (gaps in NAV series)
        - corporate-action sanity (>20% return on undocumented day)
    """
    return {
        "jobId": req.jobId,
        "status": "ok",
        "checks": {
            "stale_price": "not_implemented",
            "missing_days": "not_implemented",
            "corporate_action_sanity": "not_implemented",
        },
        "stub": True,
    }
