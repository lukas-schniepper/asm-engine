"""Build the compound lineage payload that gets written into
portal_results_cache for every computed result. Every metric on the portal
must trace back to these fields.

Five fields: engine_commit_sha, asm_data_sha, input_data_hash,
policy_version, nav_snapshot_id.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Dict

from .config import settings

# Default policy version — bump when the calculation policy changes.
# Format: "<methodology>+<config>+<cutoff>+<cagr-basis>"
# v2 (2026-05-05): CAGR uses GIPS-strict calendar days / 365.25 instead of
# the canonical helper's trading-days / 252 default. Required for strict
# GIPS 2020 II.5.A.4 compliance on annualized returns.
DEFAULT_POLICY_VERSION = "twr-v1+sortino-mar0+ny-1600+cagr-365.25"


@dataclass(frozen=True)
class Lineage:
    engine_commit_sha: str
    asm_data_sha: str
    input_data_hash: str
    policy_version: str
    nav_snapshot_id: str


def hash_bytes(data: bytes) -> str:
    """Content-addressed sha256 — used for input_data_hash and nav_snapshot_id."""
    return hashlib.sha256(data).hexdigest()


def build(
    *,
    input_data_hash: str,
    nav_snapshot_id: str,
    policy_version: str = DEFAULT_POLICY_VERSION,
) -> Lineage:
    if settings is None:
        raise RuntimeError("Settings not initialised")
    return Lineage(
        engine_commit_sha=settings.engine_commit_sha,
        asm_data_sha=settings.asm_data_sha,
        input_data_hash=input_data_hash,
        policy_version=policy_version,
        nav_snapshot_id=nav_snapshot_id,
    )


def to_dict(lineage: Lineage) -> Dict[str, Any]:
    return asdict(lineage)
