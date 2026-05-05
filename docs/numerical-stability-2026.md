# Numerical stability audit — `_canonical_metrics.py`

Date: 2026-05-05  
Reviewer: senior-quant agent (audit pass during Phase 0.10 of the
Streamlit→Portal migration)  
File audited: `AlphaMachine_core/tracking/_canonical_metrics.py` (vendored
copy of `asm-data:src/utils/quant_metrics.py`)

## Scope

Migration is a one-time chance to inspect the canonical helper that every
Veloris metric ultimately routes through. Goal: catch numerical edge cases
that produce silently wrong numbers — not refactor for taste.

## Findings

### F1 — Long-series CAGR precision (severity: LOW, fix: deferred)

```python
def calculate_cagr(returns, ...):
    cum = float(np.prod(1.0 + r))   # ← multiplies len(r) floats
    ...
    return float(cum ** (1.0 / n_years) - 1.0)
```

`np.prod(1.0 + r)` accumulates float64 multiplication errors over the
length of the series. For 5-year daily series (~1260 entries), the
accumulated relative error is typically below 1e-12 — well inside our
parity tolerance class `SUMS = (1e-9, 1e-12)`.

Better-conditioned alternative:
```python
cum = float(np.exp(np.sum(np.log1p(r))))
```

But: this changes the result by ~1e-15 in regime fixtures we tested, and
the change itself is below our tolerance. Switching introduces parity
drift between the asm-data canonical and asm-engine vendored copies
without a real correctness gain. **Decision: leave as-is. Document.**

If we ever extend to >50-year monthly series (extreme stress test), we
should revisit this.

### F2 — `±inf` not filtered (severity: MEDIUM, fix: deferred to Phase 1 DQ gates)

```python
def _to_array(returns) -> np.ndarray:
    if isinstance(returns, pd.Series):
        return returns.dropna().to_numpy(dtype=float)
    arr = np.asarray(returns, dtype=float)
    return arr[~np.isnan(arr)]
```

NaN is filtered, but `±inf` slips through. A bad upstream record could
inject `+inf` (e.g. from `(1+r) = 0` then divided), and downstream
`np.prod`, `cumprod`, `np.std` all propagate inf without warning, producing
`inf` or `nan` final metrics that get cached and displayed.

**Decision: do NOT modify the canonical helper.** The vendoring contract
requires byte-equivalence with asm-data. Instead, the fix lives in the
Phase 1 DQ gates (`service/routes/dq.py`) which run *before* metrics
compute. The DQ checklist must include:

- "no `±inf` in input returns"
- "no return < -1.0 (would imply NAV ≤ 0)"
- "no return > +5.0 unless flagged as known event"

DQ failure must short-circuit the job with `error='dq_gate: <code>'`.

### F3 — Sortino vs Sharpe zero-check inconsistency (severity: LOW, fix: deferred)

```python
# calculate_sharpe
if vol < 1e-12:
    return 0.0

# calculate_sortino
if semi_dev == 0:
    return float(no_downside_value)
```

Sharpe uses a tolerance, Sortino uses exact equality. In practice:
- Sharpe: `vol < 1e-12` is reached only on synthetic constant series.
- Sortino: `semi_dev == 0` is reached when there are no negative excess
  returns at all (not just very small) — which is a meaningful semantic
  signal, not a numerical edge case.

The inconsistency is intentional (different semantic meaning), and
changing Sortino to use `< 1e-12` would either:
- Lose the "no downside in sample" signal that triggers
  `no_downside_value` (default 10.0), or
- Silently return 0 for series that legitimately have no downside.

**Decision: leave as-is.** The asymmetry is correct.

### F4 — `calculate_max_drawdown` divide-by-near-zero peak (severity: LOW, fix: documented usage contract)

```python
peak = np.maximum.accumulate(nav)
dd = (nav / peak) - 1.0
```

If `nav` starts at 0 or goes to 0, `peak` could be 0 → division blows up.

In the default usage (`is_nav=False`), `nav` is built by
`_nav_from_returns` starting at 1.0 and compounded by `(1+r)`. As long as
no `r ≤ -1.0` arrives (which would imply 100% drawdown to zero), this is
safe. F2's DQ gate catches `r ≤ -1.0` before it reaches the engine.

In `is_nav=True` mode, the caller is responsible for passing a valid NAV
series. Document this in the docstring (TODO).

**Decision: do not modify the canonical helper.** Add the constraint to
the docstring vendored copy at the next sync, and depend on F2's DQ gate
for the returns-mode path.

### F5 — `calculate_calmar` returns 0 on `mdd >= 0` (severity: NONE, by design)

```python
mdd = calculate_max_drawdown(returns)
if mdd >= 0:
    return 0.0
```

A non-negative max drawdown means either (a) the series only went up
(`mdd == 0`), or (b) the series is empty/single-point (`mdd == 0` from
`len(nav) < 2` short-circuit). Returning 0 is the right behavior in both
cases — the alternative is dividing by zero.

**Decision: documented as-is.**

### F6 — `calculate_volatility` ddof=1 (severity: NONE, by convention)

Bessel correction uses `n-1` denominator. Standard for sample volatility.
**Decision: as-is.**

## Summary

- 6 items reviewed; 0 require code changes today.
- F1 is a stability-vs-parity tradeoff; the canonical formula stays.
- F2 is the only real risk — addressed by DQ gates, not by changing the helper.
- F3, F4, F5, F6 are intentional design choices, documented.

## Future work

When the asm-data canonical helper is next updated:

- Optional: switch CAGR / NAV builders to log1p-based form (F1).
- Required: add explicit docstring constraints for the `is_nav=True` path (F4).

The vendored copy in this repo (`AlphaMachine_core/tracking/_canonical_metrics.py`)
must be updated in lockstep with the asm-data source — see header comment
in that file for sync rules.
