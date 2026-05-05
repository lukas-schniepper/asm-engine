# External validation — Phase 0.9 spec

Per the senior-quant review: portal-vs-service parity proves consistency,
not correctness. Both could be wrong the same way. Without an external
reproduction, "GIPS-aligned" is unverifiable.

## What needs to be produced

A spreadsheet at `tests/parity/external/sortino_independent_check.xlsx`
that:

1. Computes Sortino for a fixture daily-return series.
2. Uses ONLY the textbook Sortino formula (no Python imports, no asm-data).
3. Is reviewed and signed off by someone other than the Python author.
4. Agrees with `_canonical_metrics.calculate_sortino` to 1e-8.

## Recommended fixture

Use the `2017-lowvol` regime from `tests/parity/fixtures.py`. Generate
the daily returns once and dump to CSV:

```bash
cd asm-engine
python -c "
from tests.parity.fixtures import regime_2017_lowvol
s = regime_2017_lowvol()
s.to_csv('tests/parity/external/fixture_2017_lowvol_returns.csv', header=['daily_return'])
print('Canonical Sortino:', __import__('AlphaMachine_core.tracking._canonical_metrics', fromlist=['calculate_sortino']).calculate_sortino(s))
"
```

This produces a 252-row CSV with `date, daily_return` and prints the
expected Sortino value.

## What the Excel workbook needs

Three sheets:

### Sheet 1: `returns`
- Column A: dates (copy from CSV)
- Column B: daily returns (copy from CSV)

### Sheet 2: `calc`
Implement Sortino step-by-step:

| Cell | Formula | Description |
|---|---|---|
| `B1` | `0` | MAR (annualized) |
| `B2` | `252` | Trading days per year |
| `B3` | `=(1+B1)^(1/B2)-1` | Daily MAR |
| `B4` | `=AVERAGE(returns!B:B)-B3` | Daily mean excess return |
| `B5` | `=B4*B2` | Annualized excess return |
| `D1:D252` | `=MIN(0, returns!B<row> - calc!$B$3)` | Per-day downside deviation |
| `D253` | `=AVERAGE(D1:D252^2)` | Mean squared downside (full sample!) |
| `D254` | `=SQRT(D253)` | Daily semi-deviation |
| `D255` | `=D254*SQRT(B2)` | Annualized semi-deviation |
| `B7` | `=B5/D255` | Sortino |

### Sheet 3: `verify`
- Cell A1: "Excel Sortino"
- Cell B1: `=calc!B7`
- Cell A2: "Canonical Sortino (paste from Python)"
- Cell B2: (paste the value printed by the Python script above)
- Cell A3: "abs delta"
- Cell B3: `=ABS(B1-B2)`
- Cell A4: "Pass/Fail"
- Cell B4: `=IF(B3<1E-8,"PASS","FAIL")`

## Sign-off

After verification, save the file at the path above and add a `signoff.md`
with:
- Name of reviewer
- Date
- Excel Sortino value
- Canonical Sortino value
- Delta
- Comments / observations

The reviewer must NOT be the person who wrote `_canonical_metrics.py`.
For Veloris that means: anyone other than the engine author can sign.

## Why this exists

If asm-data's `calculate_sortino` has a subtle bug, both the canonical
helper AND every parity test that uses it agree on the wrong number. The
only thing that catches such a bug is reproducing the math from scratch
in a different tool. Excel is intentionally chosen because:

- No dependency on the Python ecosystem
- Universally readable for a future regulator/copier audit
- The formula is short enough to fit on one screen

Re-run after any change to the canonical helper. The signoff.md file is
the audit-defensible artifact, not the spreadsheet itself — keep both.
