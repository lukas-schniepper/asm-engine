# Parity test harness

Phase 0.12 deliverable. Two layers of correctness checks for the canonical
metric helper:

1. **Regime invariants** (`test_canonical_invariants.py`) — fixed
   adversarial fixtures (2008, 2020-Q1, 2022, 2017 low-vol, all-cash,
   single-position, negative-Sharpe, synthetic full drawdown) plus
   parametrized invariant assertions per regime.

2. **Property-based fuzz** (`test_property_based.py`) — `hypothesis`
   draws random return arrays from the legal space (`[-0.5, 0.5]`,
   length 20–2000, no NaN/inf) and asserts mathematical invariants.

## Tolerance classes

`tolerance.py` exposes four classes — never use a flat 1e-10 tolerance.

| Class | rtol | atol | Use |
|---|---|---|---|
| `EXACT` | 0 | 0 | counts, dates, ids |
| `SUMS` | 1e-9 | 1e-12 | mean, annualized excess, Sharpe, Sortino, vol |
| `PATH` | 1e-7 | 1e-9 | rolling stats, cumprod, drawdowns, CAGR |
| `OPT` | 1e-5 | 1e-6 | scipy.optimize / Optuna outputs |

`metric_class("calculate_sortino")` returns the right class for a known
metric name; defaults to `PATH` for unknown names (safer than `SUMS`).

## Running locally

```bash
# From asm-engine repo root
pip install -r service/requirements.txt
pip install pytest hypothesis pyarrow

# Regime invariants only — fast (<2s)
pytest tests/parity/test_canonical_invariants.py -v

# Property fuzz (default 100 examples per test)
pytest tests/parity/test_property_based.py -v

# Release-grade (1000 examples) — used on release tags by CI
pytest tests/parity/test_property_based.py --hypothesis-profile=release
```

## CI integration

`.github/workflows/service-test.yml` already runs the parity tests on any
push that touches `service/`, `AlphaMachine_core/`, `shared/`, or these
test files.

The release profile (1000 examples) runs only on tag pushes — see
`pyproject.toml::[tool.hypothesis]` for the profile definition (added in
Phase 0.12).

## Adding a new regime

1. Add a function to `fixtures.py` that returns a `pd.Series` of length
   `LEN` (252) with a sensible date index.
2. Register it in the `REGIMES` dict.
3. The parametrized tests pick it up automatically — no other changes
   needed.

## What this harness does NOT do

- Cross-engine parity (Streamlit-path vs FastAPI-path on identical input)
  lives in `tests/parity/test_kpi.py` (Phase 1) and follows the same
  fixture/tolerance pattern.
- External validation (Excel reproduction of Sortino) lives at
  `tests/parity/external/sortino_independent_check.xlsx` (Phase 0.9).
- Realistic NAV-restatement scenarios are out of scope for Phase 0; the
  cache-invalidation Inngest function is exercised in Phase 1's tests.
