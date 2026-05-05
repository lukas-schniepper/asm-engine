"""Pydantic request/response shapes for the compute service."""
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class JobRequest(BaseModel):
    """All async job-trigger endpoints take this shape: a portal-issued jobId
    plus the kind-specific params block.
    """
    jobId: str = Field(..., description="UUID created by the portal in portal_compute_jobs")
    params: Dict[str, Any] = Field(default_factory=dict)


class JobAccepted(BaseModel):
    jobId: str
    status: str = "running"


class PingRequest(BaseModel):
    jobId: str
    params: Dict[str, Any] = Field(default_factory=dict)


class PingResponse(BaseModel):
    jobId: str
    message: str
    engine_commit_sha: str
    asm_data_sha: str
    received_params: Dict[str, Any]


class JobStatus(BaseModel):
    id: str
    kind: str
    status: str  # queued | running | complete | failed
    result_ref: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    heartbeat_at: Optional[str] = None
