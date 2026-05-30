# B26 delta: cleanup bundle (D-B23.2 + D-B23.3 + D-B24.3)

## Requirements

R-B26-1 (closes D-B23.2): every non-`EnsembleLiftRollupRow`
RollupRow schema (`RollupRow`, `PairwiseRollupRow`,
`TrainingTimeRollupRow`, `HPOUpliftRollupRow`) gains a
cross-field `@model_validator(mode="after")` enforcing the
CI-sentinel consistency invariant:
- If `bootstrap_skipped_reason is None`, then
  `primary_metric_mean`, `primary_metric_ci_lo`, and
  `primary_metric_ci_hi` MUST all be non-None.
- If `bootstrap_skipped_reason is not None`, then
  `primary_metric_mean`, `primary_metric_ci_lo`, and
  `primary_metric_ci_hi` MUST all be None.

This mirrors the actual aggregator behavior documented in the
schema docstrings (`bootstrap_manifest.py:96-99, :272-275,
:464-468, :557-561`) and the
`EnsembleLiftRollupRow` precedent from B23 (D-B20.3).

R-B26-2 (closes D-B23.3 + D-B24.3): add the missing
`if not rollup: return ""` early-return guard to
`_render_partial_coverage_footnote` in
`benchmarks/report/ensemble_lift.py:504-513`. The sibling
`_render_oracle_partial_coverage_footnote` at `:516-545`
already has the analogous guard at `:528-529`. The B24 and
B25 shared helpers (`render_bca_health_footnote`,
`render_per_fold_cis_footnote`) also have the guard. The
asymmetry is cosmetic but contradicts the established helper
pattern. After R-B26-2, all 4 ensemble_lift / shared
footnote helpers have the same empty-input contract.

## Non-requirements

- v1 does NOT add the structural cell-count invariant
  `n_cells_paired + n_skipped_cells <= n_seeds * n_folds` to
  `HPOUpliftRollupRow`. The CI-sentinel invariant is
  schema-wide; the cell-count invariant is HPOUpliftRollupRow-
  specific and would warrant its own audit pass (deferred to
  D-B26.1).
- v1 does NOT change any aggregator code. The validators
  fire on existing parquet shards via `model_validate` and
  on factory construction; no aggregator emit shape change.
- v1 does NOT touch the `EnsembleLiftRollupRow` validator
  (already covered by B23's D-B20.3 + 3 invariants).

## B26.0 Background

### B26.0.1 What B23 / B24 left deferred

B23 added a cross-field `@model_validator(mode="after")` to
`EnsembleLiftRollupRow` enforcing three structural
invariants (oracle <= paired, oracle <= grid, paired <=
grid). B24 inherited the same validator. Both phases
deferred extending the pattern to the other 4 RollupRow
schemas as D-B23.2 (under B24's bigger D-B23.2 rewrite that
named candidate peer invariants).

### B26.0.2 What B23 / B24 left deferred (helper guard)

B23 added `_render_oracle_partial_coverage_footnote` with an
internal `if not affected: return ""` guard. Its sibling
`_render_partial_coverage_footnote` (pre-B23) lacks the same
guard. B23 reviewers flagged this asymmetry (qa-R1-I3 +
code-R1-N2 at B23's build-review R1). The deferral D-B23.3
captured the gap. B24 re-noted the asymmetry as D-B24.3
under the same "cross-helper sweep" reasoning.

### B26.0.3 CI-sentinel invariant precedent

The aggregator emits two row shapes per RollupRow type:
- **Sentinel** (skipped): `primary_metric_mean=None`,
  `primary_metric_ci_lo=None`, `primary_metric_ci_hi=None`,
  `bootstrap_skipped_reason="some_reason"`.
- **Non-sentinel** (happy path): all three metric fields
  non-None, `bootstrap_skipped_reason=None`.

The documented contract in each schema's docstring
(`bootstrap_manifest.py:96-99, :272-275, :464-468,
:557-561`) names this invariant in prose; B26 promotes it to
a structural enforcement so a future aggregator bug that
emits a half-populated row (e.g., `primary_metric_mean`
set but `ci_lo` None) raises at parquet read time rather
than silently corrupting a downstream rollup.

## B26.1 R-B26-1 design

### B26.1.1 Validator shape

Each of the 4 non-ensemble-lift RollupRow classes gets:

```python
@model_validator(mode="after")
def _validate_ci_sentinel_consistency(self) -> "<ClassName>":
    metric_fields = (
        self.primary_metric_mean,
        self.primary_metric_ci_lo,
        self.primary_metric_ci_hi,
    )
    all_none = all(f is None for f in metric_fields)
    all_set = all(f is not None for f in metric_fields)
    if not (all_none or all_set):
        raise ValueError(
            "primary_metric_mean, primary_metric_ci_lo, and "
            "primary_metric_ci_hi must be all-None or all-non-None; "
            f"got mean={self.primary_metric_mean!r}, "
            f"ci_lo={self.primary_metric_ci_lo!r}, "
            f"ci_hi={self.primary_metric_ci_hi!r}"
        )
    if all_none and self.bootstrap_skipped_reason is None:
        raise ValueError(
            "primary_metric_* are all None but "
            "bootstrap_skipped_reason is None; sentinel rows must "
            "populate bootstrap_skipped_reason"
        )
    if all_set and self.bootstrap_skipped_reason is not None:
        raise ValueError(
            "primary_metric_* are all populated but "
            "bootstrap_skipped_reason is set; non-sentinel rows "
            "must have bootstrap_skipped_reason=None"
        )
    return self
```

The validator runs on every construction including
`model_validate` inside `load_*_rollup`. The model is
`frozen=True`; `mode="after"` runs before the freeze.

### B26.1.2 Per-schema placement

| Schema | File:line | Insert after |
|---|---|---|
| `RollupRow` | `bootstrap_manifest.py:93-147` | last field declaration (`manifest_fingerprint: str`) |
| `PairwiseRollupRow` | `bootstrap_manifest.py:264-315` | last field declaration |
| `TrainingTimeRollupRow` | `bootstrap_manifest.py:318-366` | last field declaration |
| `HPOUpliftRollupRow` | `bootstrap_manifest.py:455-510` | last field declaration |

`EnsembleLiftRollupRow` already has its B23 validator;
adding the CI-sentinel invariant there is out of scope for
B26 (D-B26.2).

## B26.2 R-B26-2 design

Edit `benchmarks/report/ensemble_lift.py:504-513`:

```python
def _render_partial_coverage_footnote(rollup: list[EnsembleLiftRollupRow]) -> str:
    """Footnote listing per-dataset n_skipped_cells / n_cells_paired
    when any non-sentinel rollup row has a positive skipped count."""
    if not rollup:
        return ""
    lines = ["### Partial coverage", ""]
    ...
```

Single 2-line insertion. The caller in `_render_with_ci`
already gates `if partial:` so the new guard is unreachable
through the public API; the change brings the helper into
shape parity with its sibling and the B24/B25 shared
helpers.

## B26.3 Implementation outline

1. **Validators**: add `_validate_ci_sentinel_consistency`
   to each of the 4 non-ensemble-lift RollupRow classes in
   `benchmarks/bootstrap_manifest.py` per B26.1.1.
2. **Helper guard**: add the early-return to
   `_render_partial_coverage_footnote` per B26.2.
3. **Fixture audit**: scan existing test fixtures for any
   row that violates the new invariant (e.g., a fixture
   with `primary_metric_mean=0.5` + `bootstrap_skipped_reason="something"`
   that pre-B26 was silently accepted). Repair such
   fixtures.
4. **Tests**: add `tests/benchmarks/test_b26_cleanup_validators.py`
   per B26.4.
5. **Verify**: ruff + pyright + scoped pytest pass at
   1007 + N new tests.

## B26.4 Tests

Baseline (post-B25 main `c915015`): 1007 tests collected.

### B26.4.1 CI-sentinel validator per schema

For each of the 4 non-ensemble-lift schemas (RollupRow,
PairwiseRollupRow, TrainingTimeRollupRow,
HPOUpliftRollupRow), three tests:
- happy non-sentinel (all three metrics set, skipped=None) accepts
- happy sentinel (all three metrics None, skipped=str) accepts
- mixed (e.g., mean=0.5, ci_lo=None) rejects

4 schemas x 3 tests = 12 tests (#1-#12).

13. `test_rollup_row_rejects_metrics_set_with_skipped_reason`:
    construct a RollupRow with all metrics set AND
    `bootstrap_skipped_reason="some_reason"`; assert
    `ValidationError` matching the appropriate prose.
14. `test_rollup_row_rejects_metrics_none_with_skipped_none`:
    construct a RollupRow with all metrics None AND
    `bootstrap_skipped_reason=None`; assert `ValidationError`.
15. Same as #13 for PairwiseRollupRow.
16. Same as #14 for PairwiseRollupRow.
17. Same as #13 for TrainingTimeRollupRow.
18. Same as #14 for TrainingTimeRollupRow.
19. Same as #13 for HPOUpliftRollupRow.
20. Same as #14 for HPOUpliftRollupRow.

### B26.4.2 Helper guard (R-B26-2)

21. `test_render_partial_coverage_footnote_empty_input_returns_empty_string`:
    call `_render_partial_coverage_footnote([])` directly;
    assert the return is exactly `""`. Pins the new guard.

### B26.4.3 Existing-fixture compatibility

22. `test_existing_fixtures_satisfy_ci_sentinel_invariant`:
    informational test that constructs the canonical
    fixture rows from B17's helpers (already in the test
    suite); asserts each constructs without raising.
    Backstops the fixture-audit step in B26.3.

### B26.4.4 Expected test delta

Baseline: 1007.
- Existing tests: 1007 -> 1007 (assuming the fixture
  audit at B26.3.3 finds no violations or repairs them).
  If any existing test fixture violates the new invariant,
  it MUST be repaired in this same commit; the existing
  test count remains 1007.
- B26-new: 22 named tests.
- Total: 1007 + 22 = 1029.

## B26.5 Risks

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R-B26-Risk-1 | The new `_validate_ci_sentinel_consistency` rejects a pre-B26 parquet shard whose data violates the invariant. | Low | Bench-run shards are short-lived (B17 R-B17-3 precedent). The aggregator emits the documented shape; only a future bug could produce a malformed row, and that is what the validator catches. |
| R-B26-Risk-2 | Existing test fixtures may construct violating rows incidentally. | Medium | B26.3.3 mandates a fixture audit. The B17 fixtures' visible non-sentinel rows (all 4 byte-pin tests) pass non-None metrics + None skipped_reason; the existing sentinel construction sites in `test_bootstrap_manifest.py` set metrics None + populated skipped_reason. No expected violations, but the build phase verifies live. |
| R-B26-Risk-3 | The `_render_partial_coverage_footnote` empty-input guard is unreachable through the public API; adding it adds dead-code surface. | Low | The guard matches the established helper pattern across `_render_oracle_partial_coverage_footnote`, `render_bca_health_footnote`, `render_per_fold_cis_footnote`. The B25 reviewers (code-R1-build-I1) explicitly endorsed the defensive pattern as "documented contract for silent-render-on-missing-attr". |

## Deferred

- **D-B26.1**: add the structural cell-count invariant
  `n_cells_paired + n_skipped_cells <= n_seeds * n_folds`
  to `HPOUpliftRollupRow`. v1 of B26 keeps to the CI-
  sentinel invariant (uniform across 4 schemas);
  HPOUpliftRollupRow-specific invariants require an audit
  of the aggregator's emit shape to confirm the bound is
  tight.
- **D-B26.2**: add the CI-sentinel invariant to
  `EnsembleLiftRollupRow` (the 5th schema), composed with
  its existing B23 cell-count invariants. v1 of B26 scopes
  the new validator to the 4 schemas that lack any
  cross-field validator today; combining validators on
  `EnsembleLiftRollupRow` is a separate touch.
