"""Service configuration — read from env, fail loud on missing required vars."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _required(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(f"Missing required env var: {key}")
    return val


def _optional(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


@dataclass(frozen=True)
class Settings:
    # Database (direct connection, NOT pooler — pooler 6543 breaks SAVEPOINTs)
    database_url: str

    # HMAC signing keys for portal→service auth (dual-key rotation)
    signing_key_primary: str
    signing_key_secondary: str  # may be empty during initial deploy

    # Inngest callback (service→Inngest event when async job completes)
    inngest_event_key: str
    inngest_event_url: str  # e.g. https://inn.gs/e/<key>

    # Lineage — baked at Railway build time so every result is traceable
    engine_commit_sha: str
    asm_data_sha: str

    # CF Access service token validation (optional — second layer; HMAC is the contract)
    cf_access_team: str  # e.g. "veloriscapital"
    cf_access_aud: str   # CF Access application AUD tag

    # Operational
    log_level: str
    request_timeout_seconds: int


def load() -> Settings:
    return Settings(
        database_url=_required("DATABASE_URL_DIRECT"),
        signing_key_primary=_required("ASM_ENGINE_SIGNING_KEY_PRIMARY"),
        signing_key_secondary=_optional("ASM_ENGINE_SIGNING_KEY_SECONDARY"),
        inngest_event_key=_required("INNGEST_EVENT_KEY"),
        inngest_event_url=_optional(
            "INNGEST_EVENT_URL",
            "https://inn.gs/e/" + _optional("INNGEST_EVENT_KEY", ""),
        ),
        engine_commit_sha=_optional("ENGINE_COMMIT_SHA", "dev"),
        asm_data_sha=_optional("ASM_DATA_SHA", "dev"),
        cf_access_team=_optional("CF_ACCESS_TEAM_DOMAIN", ""),
        cf_access_aud=_optional("CF_ACCESS_AUD_TAG", ""),
        log_level=_optional("LOG_LEVEL", "INFO"),
        request_timeout_seconds=int(_optional("REQUEST_TIMEOUT_SECONDS", "30")),
    )


# Singleton — load once at import. FastAPI reads at startup.
settings = load() if os.environ.get("ASM_SERVICE_RUNTIME") == "1" else None
