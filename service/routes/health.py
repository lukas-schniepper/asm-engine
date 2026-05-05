"""Health endpoint — no auth, used by Railway/uptime checks."""
from __future__ import annotations

from fastapi import APIRouter

from ..config import settings

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "engine_commit_sha": settings.engine_commit_sha if settings else "unknown",
        "asm_data_sha": settings.asm_data_sha if settings else "unknown",
    }
