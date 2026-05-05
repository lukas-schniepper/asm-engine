"""FastAPI application entrypoint for the asm-engine compute service.

Run locally:
    ASM_SERVICE_RUNTIME=1 uvicorn service.main:app --reload --port 8001

Run on Railway (see Procfile):
    ASM_SERVICE_RUNTIME=1 uvicorn service.main:app --host 0.0.0.0 --port $PORT --workers 2
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .config import settings
from .db import close_pool, get_pool
from .routes import backtest, dq, health, jobs, optimize, ping, sortino


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.environ.get("ASM_SERVICE_RUNTIME") != "1":
        # Defensive: refuse to start without explicit runtime flag so accidental
        # imports from tests or scripts can't try to open prod connections.
        raise RuntimeError(
            "ASM_SERVICE_RUNTIME=1 is required to start the compute service"
        )
    logging.basicConfig(
        level=getattr(logging, (settings.log_level if settings else "INFO")),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("service.main")
    log.info(
        "asm-engine compute service starting (engine=%s asm-data=%s)",
        settings.engine_commit_sha if settings else "?",
        settings.asm_data_sha if settings else "?",
    )
    # Eagerly open DB pool to fail fast on bad credentials
    await get_pool()
    try:
        yield
    finally:
        await close_pool()
        log.info("asm-engine compute service stopped")


app = FastAPI(
    title="asm-engine compute service",
    version="0.1.0",
    description="Canonical Python compute exposed to the Veloris portal. "
                "All quant math runs here; portal renders cached results.",
    lifespan=lifespan,
)

# Routes
app.include_router(health.router)
app.include_router(ping.router)
app.include_router(sortino.router)
app.include_router(backtest.router)
app.include_router(optimize.router)
app.include_router(dq.router)
app.include_router(jobs.router)
