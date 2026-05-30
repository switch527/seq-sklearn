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
3. **Fixture audit** (arch-R1-C1 + qa-R1-C2 closure): the
   R1 swarm identified two KNOWN sites that construct
   "schema-default zero-cell" rows leaving
   `primary_metric_*=None` (the schema default) AND
   `bootstrap_skipped_reason=None`. Under the new validator
   these raise (`all_none` + skipped=None branch). Repair
   sites:
   - `tests/benchmarks/test_b22_per_fold_cis.py:222-275`
     (`test_rollup_row_per_fold_cis_schema_default_is_none`):
     5 construction sites for the 5 schemas. Each row's
     intent is "test the per_fold_cis default on a
     minimal row", not "construct a valid non-sentinel".
     Repair: add `bootstrap_skipped_reason="test_fixture"`
     to the shared `common` dict at `:213-221` so the rows
     are valid sentinels (all-None metrics + populated
     reason).
   - `tests/benchmarks/test_b22_per_fold_cis.py:283`
     (`test_rollup_row_per_fold_cis_accepts_empty_list`):
     1 RollupRow construction with the same pattern.
     Repair: add `bootstrap_skipped_reason="test_fixture"`
     to the local kwargs.
   - `tests/benchmarks/test_b21_bca_ci.py:765-820`
     (`test_rollup_row_schema_default_ci_method_is_percentile`):
     5 construction sites. Same pattern, same repair:
     add `bootstrap_skipped_reason="test_fixture"` to the
     shared `common_base` dict.
   In addition, run a programmatic suite-wide audit during
   the build phase (test #22 below is the gate): if any
   other fixture constructs a violating row, the existing
   pytest count would regress and the build phase catches
   it.
4. **Tests**: add `tests/benchmarks/test_b26_cleanup_validators.py`
   per B26.4.
5. **Verify**: ruff + pyright + scoped pytest pass at
   1007 + N new tests.

## B26.4 Tests

Baseline (post-B25 main `c915015`): 1007 tests collected.

### B26.4.1 CI-sentinel validator per schema

For each of the 4 non-ensemble-lift schemas (RollupRow,
PairwiseRollupRow, TrainingTimeRollupRow,
HPOUpliftRollupRow):
- happy non-sentinel (all three metrics set, skipped=None) accepts
- happy sentinel (all three metrics None, skipped=str) accepts
- mixed-A (mean set, ci_lo None, ci_hi None) rejects
- mixed-B (mean None, ci_lo set, ci_hi None) rejects

Per qa-R1-I1 closure: 2 mixed variants kill a mutation that
only checks specific positions. 4 schemas x 4 tests = 16
tests (#1-#16).

Each rejection test (`mixed-A`, `mixed-B`) MUST use
`pytest.raises(ValidationError, match=r"must be all-None or
all-non-None")` to discriminate the first branch from the
two skipped-reason branches (qa-R1-C1 closure).

17. `test_rollup_row_rejects_metrics_set_with_skipped_reason`:
    construct a RollupRow with all three metrics set AND
    `bootstrap_skipped_reason="some_reason"`; assert
    `pytest.raises(ValidationError,
    match=r"non-sentinel rows must have
    bootstrap_skipped_reason=None")` (qa-R1-C1 closure: the
    `match=` pin discriminates this branch from the
    `all_none` + skipped=None branch; without it a mutation
    that swaps the two branch bodies would survive).
18. `test_rollup_row_rejects_metrics_none_with_skipped_none`:
    construct a RollupRow with all metrics None AND
    `bootstrap_skipped_reason=None`; assert
    `pytest.raises(ValidationError, match=r"sentinel rows
    must populate bootstrap_skipped_reason")`.
19. Same as #17 for PairwiseRollupRow.
20. Same as #18 for PairwiseRollupRow.
21. Same as #17 for TrainingTimeRollupRow.
22. Same as #18 for TrainingTimeRollupRow.
23. Same as #17 for HPOUpliftRollupRow.
24. Same as #18 for HPOUpliftRollupRow.

### B26.4.2 Helper guard (R-B26-2)

25. `test_render_partial_coverage_footnote_empty_input_returns_empty_string`:
    call `_render_partial_coverage_footnote([])` directly;
    assert the return is exactly `""`. Pins the new guard.

### B26.4.3 Suite-wide audit (qa-R1-C2 closure)

26. `test_existing_b17_byte_pin_fixtures_satisfy_ci_sentinel_invariant`:
    construct the canonical fixture rows from B17's
    helpers (`_make_pairwise_rollup`,
    `_make_training_time_rollup`, `_make_hpo_uplift_rollup`,
    `_make_ensemble_lift_rollup` in
    `tests/benchmarks/test_b17_byte_identity_pins.py`);
    assert each constructs without raising. Backstops the
    B17 (non-sentinel) fixture compatibility.

The b21 and b22 schema-default fixtures named in B26.3.3
are repaired in this same commit; their repaired versions
naturally re-pass when the suite runs as the build gate.
No dedicated b21/b22 backstop test is needed because the
existing tests themselves are the gate (post-repair they
must pass; pre-repair they would fail under the new
validator).

### B26.4.4 Expected test delta

Baseline: 1007.
- Existing tests: 1007 -> 1007 (after the b21 + b22
  schema-default fixture repairs land in the same commit;
  the repairs only add `bootstrap_skipped_reason="test_fixture"`
  without changing test semantics).
- B26-new: 26 named tests (4 schemas x 4 + 8 reject pairs + 1
  helper-guard + 1 B17 backstop = 26; was 22 in R1 draft,
  +4 from qa-R1-I1 closure adding mixed-B variants).
- Total: 1007 + 26 = 1033.

## B26.5 Risks

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R-B26-Risk-1 | The new `_validate_ci_sentinel_consistency` rejects a pre-B26 parquet shard whose data violates the invariant. | Low | Bench-run shards are short-lived (B17 R-B17-3 precedent). The aggregator emits the documented shape; only a future bug could produce a malformed row, and that is what the validator catches. |
| R-B26-Risk-2 | Existing test fixtures construct violating rows. | Medium-confirmed | The R1 design swarm identified KNOWN violating sites in `test_b21_bca_ci.py:765-820` (5 rows in `test_rollup_row_schema_default_ci_method_is_percentile`) and `test_b22_per_fold_cis.py:222-275` (5 rows in `test_rollup_row_per_fold_cis_schema_default_is_none`) and `test_b22_per_fold_cis.py:283-303` (1 row in `test_rollup_row_per_fold_cis_accepts_empty_list`). All 11 sites use "schema-default zero-cell" pattern that leaves metrics at the None default with `bootstrap_skipped_reason=None`. B26.3.3 enumerates the exact repair: add `bootstrap_skipped_reason="test_fixture"` to the shared `common` / `common_base` dict (or kwargs). The repair preserves test semantics since neither test asserts on metric values. The B17 byte-pin fixtures (4 helpers) pass non-None metrics + None skipped_reason: COMPATIBLE. Other 40+ row construction sites across the test suite were not enumerated; the build-phase suite run is the live gate. |
| R-B26-Risk-3 | The `_render_partial_coverage_footnote` empty-input guard is unreachable through the public API; adding it adds dead-code surface. | Low | The guard matches the established helper pattern across `_render_oracle_partial_coverage_footnote`, `render_bca_health_footnote`, `render_per_fold_cis_footnote`. The B25 reviewers (code-R1-build-I1) explicitly endorsed the defensive pattern as "documented contract for silent-render-on-missing-attr". |

## Addressed

R1 design swarm on commit `b10cc10`: architecture-reviewer
(1C / 3I / 1N REQUEST_CHANGES), qa-test-coverage (2C / 2I /
1N REQUEST_CHANGES), style-reviewer (0C / 0I / 0N APPROVE).
Deduplicated total: 3 CRITICAL, 5 IMPROVEMENT, 2 NITPICK.
Closures:

- **arch-R1-C1 + qa-R1-C2** (existing fixtures in
  test_b21 + test_b22 schema-default tests construct
  violating "all-None metrics + None skipped" rows): B26.3.3
  rewritten to enumerate the 11 KNOWN sites and prescribe
  the `bootstrap_skipped_reason="test_fixture"` repair.
  R-B26-Risk-2 elevated to "Medium-confirmed" with the
  exact sites named.
- **qa-R1-C1** (tests #13-#20 without `match=` cannot
  discriminate the two skipped-reason branches; swap
  mutation survives): every rejection test now uses
  `match=` with branch-specific message substrings
  ("non-sentinel rows must have
  bootstrap_skipped_reason=None" vs "sentinel rows must
  populate bootstrap_skipped_reason"). Mixed-rejection
  tests also pin `match=r"must be all-None or
  all-non-None"`.
- **qa-R1-I1** (mixed-reject tests cover only 1 of 6
  possible mixed variants per schema): added a second
  variant (mixed-B: `mean=None, ci_lo=set, ci_hi=None`)
  per schema. 4 schemas x 4 tests (happy non-sentinel,
  happy sentinel, mixed-A, mixed-B) = 16 tests instead of
  the R1-draft 12.
- **qa-R1-I2** (R-B26-Risk-2 + B26.3.3 didn't name b21/b22
  sites): covered by arch-R1-C1 closure above.
- **qa-R1-N1** (test #22 informational naming): renamed to
  `test_existing_b17_byte_pin_fixtures_satisfy_ci_sentinel_invariant`
  with explicit B17-helper enumeration in the test name.

Test count after R1 closures: 26 named (was 22; +4 from
qa-R1-I1's mixed-B variants per schema); 1033 collected
(was 1029).

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
