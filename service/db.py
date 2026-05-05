"""Async Postgres connection pool (asyncpg) for the compute service.

Uses the direct Supabase connection (port 5432), NOT the transaction pooler
(6543), because we need SAVEPOINT support and advisory locks for safe
job-state updates.
"""
from __future__ import annotations

import asyncpg
from typing import Optional

from .config import settings

_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        if settings is None:
            raise RuntimeError("Settings not initialised")
        _pool = await asyncpg.create_pool(
            dsn=settings.database_url,
            min_size=1,
            max_size=10,
            command_timeout=settings.request_timeout_seconds,
            # Disable JIT for short-lived analytical queries; saves planning overhead
            server_settings={"jit": "off"},
        )
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
