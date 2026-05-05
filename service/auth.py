"""HMAC + Cloudflare Access verification for portal→service auth.

Two layers:
1. HMAC signature over (timestamp, method, path, sha256(body)) — contract auth
2. CF Access JWT validation (optional but recommended once second tunnel exists)

Both must pass for a request to reach a route handler.
"""
from __future__ import annotations

import hashlib
import hmac
import time
from typing import Optional

from fastapi import Header, HTTPException, Request, status

from .config import settings

CLOCK_SKEW_SECONDS = 60


def _expected_signature(key: str, ts: str, method: str, path: str, body: bytes) -> str:
    body_digest = hashlib.sha256(body).hexdigest()
    msg = f"{ts}.{method.upper()}.{path}.{body_digest}".encode("utf-8")
    return hmac.new(key.encode("utf-8"), msg, hashlib.sha256).hexdigest()


def _verify_against_key(
    key: str,
    ts: str,
    method: str,
    path: str,
    body: bytes,
    signature: str,
) -> bool:
    if not key:
        return False
    expected = _expected_signature(key, ts, method, path, body)
    return hmac.compare_digest(expected, signature)


async def verify_hmac(
    request: Request,
    x_asm_timestamp: Optional[str] = Header(None),
    x_asm_signature: Optional[str] = Header(None),
) -> None:
    """FastAPI dependency: rejects request unless HMAC verifies.

    Accepts either primary or secondary key (rotation buffer).
    """
    if settings is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialised",
        )

    if not x_asm_timestamp or not x_asm_signature:
        raise HTTPException(status_code=401, detail="Missing X-Asm-Timestamp or X-Asm-Signature")

    try:
        ts_int = int(x_asm_timestamp)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid X-Asm-Timestamp")

    if abs(int(time.time()) - ts_int) > CLOCK_SKEW_SECONDS:
        raise HTTPException(status_code=401, detail="X-Asm-Timestamp out of window")

    body = await request.body()
    method = request.method
    # request.url.path includes any path; we sign on the raw path portion.
    path = request.url.path

    primary_ok = _verify_against_key(
        settings.signing_key_primary, x_asm_timestamp, method, path, body, x_asm_signature
    )
    secondary_ok = _verify_against_key(
        settings.signing_key_secondary, x_asm_timestamp, method, path, body, x_asm_signature
    )

    if not (primary_ok or secondary_ok):
        raise HTTPException(status_code=401, detail="Invalid HMAC signature")
