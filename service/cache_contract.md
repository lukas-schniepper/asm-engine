# `portal_results_cache` contract

Per-`kind` rules for how the FastAPI compute service uses
`portal_results_cache` — what goes in `cache_key`, how long results live,
what triggers invalidation. Keeping this rigorous is what makes "the chip
on the metric panel matches the displayed number" a load-bearing claim.

All `cache_key` values are hex SHA-256 of a canonical input string. Use
`portal_results_cache.kind` + `cache_key` as the unique pair (DB index
already enforces this). Rows are upserted on collision: payload is
overwritten, `computed_at` is bumped to `now()`.

## Per-kind table

| `kind` | `cache_key` input | TTL | Invalidation triggers |
|---|---|---|---|
| `ping` | `jobId` | 7d | None — diagnostic only |
| `sortino` | `sha256(json(returns) \| mar \| no_downside_value \| policy_version)` | 30d | Policy bump |
| `kpi-single` | `sha256(portfolio_id \| variant \| as_of_date \| policy_version)` | 24h | New NAV ingest for `portfolio_id`; policy bump |
| `kpi-batch` | `sha256(YYYY-MM \| variant \| policy_version)` | 7d | NAV restatement that touches month |
| `backtest` | `sha256(canonicalized_params_json)` | 30d | None (deterministic by params); manual purge on engine code change |
| `optimize` | `sha256(study_name \| n_trials \| canonicalized_search_space \| seed)` | 30d | Manual purge on engine code change |
| `walk-forward` | `sha256(canonicalized_params \| fold_def)` | 30d | None |
| `correlation` | `sha256(sorted_portfolio_ids \| start_date \| end_date)` | 7d | NAV restatement that intersects window |
| `dq` | `sha256(portfolio_id \| as_of_date)` | 24h | New NAV ingest |

## Canonicalization rules

When a `cache_key` input includes JSON or a list:

1. **Object keys sorted lexicographically** before serialization — `JSON.stringify({a:1, b:2})` and `JSON.stringify({b:2, a:1})` must produce the same hash.
2. **Floating-point values rounded to 12 decimal places** — avoids hash drift from Python/JS round-trip artifacts.
3. **NaN / Infinity normalized to literal strings** `"NaN"` / `"Infinity"` / `"-Infinity"` (JSON does not natively support them).
4. **Lists of identifiers sorted** — `["a","b","c"]` and `["c","a","b"]` must produce the same hash for symmetric ops (e.g. correlation).
5. **Dates as ISO 8601 UTC strings** — `2026-04-30T20:00:00Z`, never local-tz strings.

Helper: `service/cache.py:canonicalize_for_hash()` (TODO Phase 1).

## Lineage fields (on every row)

| Field | Source | Required |
|---|---|---|
| `engine_commit_sha` | `$ENGINE_COMMIT_SHA` baked at Railway build | yes |
| `asm_data_sha` | `$ASM_DATA_SHA` baked at Railway build (vendored helper version) | yes |
| `input_data_hash` | `sha256` of the canonicalized request body | yes |
| `policy_version` | from `service/lineage.py::DEFAULT_POLICY_VERSION` (or per-job override) | yes |
| `nav_snapshot_id` | content-addressed S3 parquet key, `"ad-hoc"` for unbacked params, `"ping"` for ping | yes |
| `computed_at` | `now()` at write time | auto |
| `expires_at` | `computed_at + TTL` per table above | optional |

The lineage chip in the portal must surface `engine_commit_sha`,
`asm_data_sha`, and `computed_at` at minimum. The other three are visible
in the hover tooltip.

## Invalidation mechanics

### NAV ingest
Every new NAV bar / restatement should write a row to
`portal_nav_snapshots`. An Inngest cron (Phase 1) reads recent snapshots
and:
1. For new (non-restatement) rows: nothing — caches stay valid for as-of
   dates strictly before the new bar.
2. For restatement rows (`superseded_by IS NOT NULL`): purges
   `portal_results_cache` rows whose `nav_snapshot_id` matches the
   superseded id.

### Policy bump
When `service/lineage.py::DEFAULT_POLICY_VERSION` changes (e.g. switch
from `twr-v1` to `twr-v2`), the engine commit SHA also changes (since
the file is in the engine repo). The portal's recompute job for the
affected `kind` must run after the deploy that introduced the bump.

### Manual purge
Operator action via `/admin/cache` (Phase 1, not yet built). Until then:

```sql
DELETE FROM portal_results_cache WHERE kind = ? AND computed_at < ?;
```

## Cache hit path

1. Portal API route receives request with parameters.
2. Compute `cache_key` per the rules above.
3. `SELECT payload, lineage WHERE kind = ? AND cache_key = ? AND (expires_at IS NULL OR expires_at > now())`.
4. If hit: render with the cached lineage, no service call.
5. If miss: insert `portal_compute_jobs` row, send `compute/job.requested`, FastAPI service computes + writes cache, portal poll picks up.

## Cache miss without job
A read that finds no cache row but doesn't want to trigger compute (e.g.
sidebar quick-view) should render a "—" placeholder, not call the service
synchronously. Service calls go through Inngest, never inline.
