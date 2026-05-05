# asm-engine compute service

FastAPI app exposing canonical Python compute to the Veloris portal. The
Streamlit operator UI is being replaced by `asm-website` portal pages; this
service provides the compute backend.

See `asm-website/docs/MIGRATION_PLAN_STREAMLIT_TO_PORTAL.md` for the full plan.

## Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | none | Railway healthcheck + uptime monitor |
| POST | `/jobs/ping` | HMAC | Trivial round-trip; gates Phase 1 |
| POST | `/jobs/sortino-recompute` | HMAC | Sync Sortino calc via canonical helper |
| POST | `/jobs/backtest` | HMAC | Async — schedules SharpeBacktestEngine, 202 |
| POST | `/jobs/optimize` | HMAC | Async — schedules Optuna study, 202 |
| POST | `/jobs/dq` | HMAC | Data quality gates |
| GET | `/jobs/{id}` | HMAC | Poll job status |

## Auth contract

Every authenticated request must carry:

- `X-Asm-Timestamp: <unix>` — within 60s of server clock
- `X-Asm-Signature: hex(hmac_sha256(KEY, f"{ts}.{METHOD}.{PATH}.{sha256(body)}"))`

Server tries `ASM_ENGINE_SIGNING_KEY_PRIMARY` first, falls back to
`_SECONDARY` for rotation. See `auth.py`.

## Lineage

Every cached result carries 5 fields (see `lineage.py`):

- `engine_commit_sha` — service repo commit, baked at build time
- `asm_data_sha` — submodule commit, baked at build time
- `input_data_hash` — sha256 of the request body
- `policy_version` — calculation policy string (`twr-v1+sortino-mar0+ny-1600`)
- `nav_snapshot_id` — content-addressed S3 NAV parquet key (or `ad-hoc`/`ping` for non-NAV jobs)

## Local development

```bash
# From asm-engine repo root
pip install -r requirements.txt -r service/requirements.txt

# Required env vars (use a .env file)
export ASM_SERVICE_RUNTIME=1
export DATABASE_URL_DIRECT="postgresql://...:5432/postgres?sslmode=require"
export ASM_ENGINE_SIGNING_KEY_PRIMARY="$(openssl rand -hex 32)"
export INNGEST_EVENT_KEY="<from-inngest-dashboard>"
export ENGINE_COMMIT_SHA="$(git rev-parse HEAD)"
export ASM_DATA_SHA="$(git -C shared/data rev-parse HEAD)"

uvicorn service.main:app --reload --port 8001
```

Test the health endpoint:
```bash
curl http://localhost:8001/health
```

Test ping (with HMAC):
```python
import time, hmac, hashlib, json, requests

ts = str(int(time.time()))
body = json.dumps({"jobId": "00000000-0000-0000-0000-000000000001", "params": {"foo": "bar"}})
key = "<your_signing_key>"
msg = f"{ts}.POST./jobs/ping.{hashlib.sha256(body.encode()).hexdigest()}"
sig = hmac.new(key.encode(), msg.encode(), hashlib.sha256).hexdigest()

# (Insert a row into portal_compute_jobs first or the route returns 404)
print(requests.post(
    "http://localhost:8001/jobs/ping",
    data=body,
    headers={"Content-Type": "application/json", "X-Asm-Timestamp": ts, "X-Asm-Signature": sig},
).json())
```

## Production (Railway)

`Procfile` and `railway.json` configure Nixpacks build + uvicorn start.
Required env vars on Railway:

| Var | Source |
|---|---|
| `ASM_SERVICE_RUNTIME` | `1` (literal) |
| `DATABASE_URL_DIRECT` | Supabase direct connection (port 5432, NOT 6543 pooler) |
| `ASM_ENGINE_SIGNING_KEY_PRIMARY` | Shared with Vercel; rotate quarterly |
| `ASM_ENGINE_SIGNING_KEY_SECONDARY` | Rotation buffer; can start empty |
| `INNGEST_EVENT_KEY` | From Inngest dashboard |
| `ENGINE_COMMIT_SHA` | Set automatically via Railway build env (`$RAILWAY_GIT_COMMIT_SHA`) |
| `ASM_DATA_SHA` | Set via build hook reading `shared/data` submodule HEAD |
| `LOG_LEVEL` | optional, default INFO |

Deploy is path-filtered: only changes under `service/`, `AlphaMachine_core/`,
`shared/`, or root config files trigger a Railway rebuild. Pure Streamlit
edits don't redeploy the worker. See `.github/workflows/railway-deploy.yml`.
