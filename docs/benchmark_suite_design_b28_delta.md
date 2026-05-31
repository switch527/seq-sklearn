# B28 delta: schema-validator cleanups (D-B27.2 + D-B26.1)

## Requirements

R-B28-1 (closes D-B27.2): extend `EnsembleLiftRollupRow`'s
existing `_validate_ci_sentinel_consistency` validator to
ALSO guard the `oracle_metric_*` triple:
- If `n_oracle_cells_paired == 0`, then `oracle_metric_mean`,
  `oracle_metric_ci_lo`, and `oracle_metric_ci_hi` MUST all
  be None.
- If `n_oracle_cells_paired > 0`, then `oracle_metric_mean`,
  `oracle_metric_ci_lo`, and `oracle_metric_ci_hi` MUST all
  be non-None.

The aggregator at `benchmarks/report/bootstrap_ensemble_lift.py:152-159`
(sentinel) and `:320-321 + :390-392` (happy path) emits both
shapes structurally. The new clause catches a future
aggregator bug that emits a half-populated oracle triple.

R-B28-2 (closes D-B26.1): add a new
`@model_validator(mode="after")` to `HPOUpliftRollupRow`
enforcing two structural cell-count invariants:
- `n_cells_paired <= n_seeds * n_folds`
- `n_skipped_cells <= n_cells_paired`

Per the schema docstring at `bootstrap_manifest.py:585-586`,
`n_cells_paired` counts cells where BOTH default and tuned
ran (bounded by `n_seeds * n_folds`); `n_skipped_cells` is
a SUBSET of paired cells dropped from the bootstrap due to
NaN loss (bounded above by `n_cells_paired`). The
aggregator at `benchmarks/report/bootstrap_hpo_uplift.py:252-264`
emits the `paired_but_no_valid_loss` sentinel with
`n_cells_paired=n_paired_total, n_skipped_cells=n_paired_total`
when all paired cells have NaN loss: both invariants
trivially satisfied (4<=4 and 4<=4).

## Non-requirements

- v1 does NOT add the cell-count invariant to other
  RollupRow schemas. RollupRow / PairwiseRollupRow /
  TrainingTimeRollupRow don't carry `n_folds`; their
  equivalent bound would require a different invariant
  shape (deferred under future audit if needed).
- v1 does NOT change aggregator code paths.
- v1 does NOT touch the oracle row-count invariants
  already enforced by B23's `_validate_row_count_invariants`
  on EnsembleLiftRollupRow.

## B28.0 Background

### B28.0.1 D-B27.2 closure scope

B27 composed the primary-metric CI-sentinel validator into
EnsembleLiftRollupRow but explicitly scoped out the oracle
triple. Oracle metrics on EnsembleLiftRollupRow have their
own semantic: they are gated by `n_oracle_cells_paired`
rather than `bootstrap_skipped_reason`. The aggregator at
`bootstrap_ensemble_lift.py:320-321` short-circuits to all-
None oracle metrics when `n_oracle_cells_paired == 0`,
regardless of the main bootstrap's skipped state.

### B28.0.2 D-B26.1 closure scope

B26 added CI-sentinel validators to the 4 non-EnsembleLift
schemas but explicitly scoped out HPOUpliftRollupRow-
specific structural invariants. The cell-count bound
(`paired + skipped <= n_seeds * n_folds`) is the natural
audit for the HPO-uplift inner-join pattern.

## B28.1 R-B28-1 design

In `benchmarks/bootstrap_manifest.py`, extend
`EnsembleLiftRollupRow._validate_ci_sentinel_consistency`
with the oracle clause. The composed validator body:

```python
@model_validator(mode="after")
def _validate_ci_sentinel_consistency(self) -> "EnsembleLiftRollupRow":
    # ... existing primary_metric_* checks ...

    # B28 / D-B27.2 closure: oracle_metric_* triple is gated
    # by n_oracle_cells_paired. All-None iff n_oracle_cells_paired
    # == 0; all-set iff n_oracle_cells_paired > 0.
    oracle_fields = (
        self.oracle_metric_mean,
        self.oracle_metric_ci_lo,
        self.oracle_metric_ci_hi,
    )
    oracle_all_none = all(f is None for f in oracle_fields)
    oracle_all_set = all(f is not None for f in oracle_fields)
    if not (oracle_all_none or oracle_all_set):
        raise ValueError(
            "oracle_metric_mean, oracle_metric_ci_lo, and "
            "oracle_metric_ci_hi must be all-None or all-non-None; "
            f"got mean={self.oracle_metric_mean!r}, "
            f"ci_lo={self.oracle_metric_ci_lo!r}, "
            f"ci_hi={self.oracle_metric_ci_hi!r}"
        )
    if oracle_all_none and self.n_oracle_cells_paired > 0:
        raise ValueError(
            "oracle_metric_* are all None but n_oracle_cells_paired > 0; "
            f"got n_oracle_cells_paired={self.n_oracle_cells_paired}"
        )
    if oracle_all_set and self.n_oracle_cells_paired == 0:
        raise ValueError(
            "oracle_metric_* are all populated but n_oracle_cells_paired == 0; "
            "rows with no oracle cells must leave oracle metrics None"
        )
    return self
```

Removes the IDENTICAL BODY comment marker since this
validator no longer mirrors the 4 other schemas exactly.

## B28.2 R-B28-2 design

In `benchmarks/bootstrap_manifest.py`, add to
`HPOUpliftRollupRow`:

```python
@model_validator(mode="after")
def _validate_cell_count_bounds(self) -> "HPOUpliftRollupRow":
    # B28 / D-B26.1 closure (R2 corrected): n_cells_paired is
    # bounded by the inner-join space (n_seeds * n_folds);
    # n_skipped_cells is a SUBSET of paired cells (paired-but-
    # NaN-loss) so bounded above by n_cells_paired.
    total_possible = self.n_seeds * self.n_folds
    if self.n_cells_paired > total_possible:
        raise ValueError(
            f"n_cells_paired ({self.n_cells_paired}) exceeds "
            f"n_seeds * n_folds ({self.n_seeds} * "
            f"{self.n_folds} = {total_possible})"
        )
    if self.n_skipped_cells > self.n_cells_paired:
        raise ValueError(
            f"n_skipped_cells ({self.n_skipped_cells}) exceeds "
            f"n_cells_paired ({self.n_cells_paired})"
        )
    return self
```

This composes with the existing
`_validate_ci_sentinel_consistency` from B26. Both run via
pydantic's mode="after" chain in declaration order.

## B28.3 Implementation outline

1. **R-B28-1**: extend the EnsembleLift CI-sentinel validator
   with the oracle clause; drop the IDENTICAL BODY marker.
2. **R-B28-2**: add the new cell-count validator to
   HPOUpliftRollupRow.
3. **Fixture audit + helper repair** (arch-R1-C1 + qa-R1-C1
   closure): the R1 design swarm identified 5 KNOWN
   violating sites across two helper factories. Both
   helpers currently default oracle metrics to a fixed
   triple (0.10/0.08/0.12) without conditioning on
   `n_oracle_cells_paired`. Under the new oracle CI-sentinel
   clause this produces invalid rows whenever a call passes
   `n_oracle_cells_paired=0`.

   **Helper repairs** (applied as part of the build):
   - `tests/benchmarks/test_b23_b20_nits_bundle.py:73-106`
     `_make_rollup_row`: change the oracle default block at
     `:98-100` from `bootstrap_skipped_reason is None` gating
     to `n_oracle_cells_paired > 0` gating. The new rule:
     oracle metrics default to 0.10/0.08/0.12 when
     `n_oracle_cells_paired > 0`, else None. This matches
     the new B28 invariant and removes the coupling to
     `bootstrap_skipped_reason` (which is the wrong axis
     for oracle gating).
   - `tests/benchmarks/test_ensemble_lift_report_b16.py:76-131`
     `_make_rollup_row`: same change at the oracle metric
     defaults (currently unconditional 0.10/0.08/0.12 at
     `:123-125`). Gate on `n_oracle_cells_paired > 0`.

   **Sites affected** (call sites that would fail
   pre-repair):
   - `tests/benchmarks/test_b23_b20_nits_bundle.py:170` (test
     #3 sentinel parametrize: `n_oracle_cells_paired=2`
     with `bootstrap_skipped_reason=<sentinel>`; current
     helper sets oracle=None, new validator rejects since
     n_oracle>0). Post-repair: helper sets oracle to
     0.10/0.08/0.12 since n_oracle=2.
   - `tests/benchmarks/test_b23_b20_nits_bundle.py:243-249`
     (test #5b: `n_oracle_cells_paired=0`, no sentinel
     reason; current helper sets oracle to 0.10/0.08/0.12,
     new validator rejects since n_oracle=0). Post-repair:
     helper sets oracle to None since n_oracle=0.
   - `tests/benchmarks/test_ensemble_lift_report_b16.py:303-310`
     (`no_gbm_predictions` sentinel with `n_cells_paired=0`,
     default `n_oracle_cells_paired=0`; current helper
     unconditional 0.10/0.08/0.12). Post-repair: oracle
     None.
   - `tests/benchmarks/test_ensemble_lift_report_b16.py:336-343`
     (`no_seq_predictions` same pattern). Same fix.
   - `tests/benchmarks/test_ensemble_lift_report_b16.py:361-368`
     (`all_cells_skipped_in_manifest` same pattern). Same
     fix.

   All 5 sites are repaired by the 2-line conditional in
   each of the 2 helper factories. No call-site changes
   needed.

   **Build-phase amendment** (qa-R1-build-N1 closure):
   live build also revealed a 6th site in a 3rd helper
   factory at `tests/benchmarks/test_b19_n_pair_grid.py:260-290`
   (`_make_rollup_row_for_renderer`). Same unconditional
   oracle-default pattern; same fix. The b19 helper was
   not enumerated in R1 because its constructor differs
   from b16/b23 (it lacks the
   `oracle_metric_mean: float | None = 0.10` parameters);
   the fix uses a local-variable assignment instead.
   Three helpers, six sites total.
4. **Tests**: add `tests/benchmarks/test_b28_schema_validators.py`
   per B28.4.
5. **Verify**: ruff + pyright + scoped pytest pass at
   1059 + N new tests.

## B28.4 Tests

Baseline (post-B27 main `37cbfc0`): 1059 tests collected.

### B28.4.1 Oracle CI-sentinel invariant (R-B28-1)

**Validator fire-order constraint** (arch-R1-I2 + qa-R1-C2
closure): the existing
`_validate_ci_sentinel_consistency` validator runs the
primary-metric clause FIRST (declaration order). Tests that
target the oracle clause MUST construct the fixture with
the primary axis in a valid state (all-set primary metrics
+ `bootstrap_skipped_reason=None`) so the primary clause
passes silently and the oracle clause is the one that
raises. Each test below must explicitly supply
`primary_metric_mean=0.20, primary_metric_ci_lo=0.15,
primary_metric_ci_hi=0.25, bootstrap_skipped_reason=None`
in its base kwargs.

1. `test_ensemble_lift_rollup_row_accepts_oracle_all_none_with_zero_oracle_cells`:
   non-sentinel row (primary all-set, skipped=None) with
   `n_oracle_cells_paired=0` and `oracle_metric_*=None`;
   assert no raise.
2. `test_ensemble_lift_rollup_row_accepts_oracle_all_set_with_positive_oracle_cells`:
   non-sentinel row with `n_oracle_cells_paired=4` and
   `oracle_metric_*` all populated; assert no raise.
3. `test_ensemble_lift_rollup_row_rejects_oracle_partially_set`:
   parametrize over 6 mixed-oracle variants (same 6-variant
   shape as B27 D-B26.3 closure). Each parametrize case uses
   `n_oracle_cells_paired=4` (so the partial-state branch
   fires, not the gating branches) and the valid-primary
   base. Assert
   `pytest.raises(ValidationError, match=r"oracle_metric_mean.*must be all-None or all-non-None")`.
4. `test_ensemble_lift_rollup_row_rejects_oracle_none_with_positive_oracle_cells`:
   non-sentinel row with `n_oracle_cells_paired=4` and
   `oracle_metric_*=None`; assert
   `pytest.raises(ValidationError, match=r"all None but n_oracle_cells_paired > 0")`.
5. `test_ensemble_lift_rollup_row_rejects_oracle_set_with_zero_oracle_cells`:
   non-sentinel row with `n_oracle_cells_paired=0` and
   `oracle_metric_*` all populated; assert
   `pytest.raises(ValidationError, match=r"all populated but n_oracle_cells_paired == 0")`.

Test #3 parametrizes 6 variants -> 6 collected. Tests #1,
#2, #4, #5 = 4 named. B28.4.1 totals 5 named, 10 collected.

### B28.4.2 HPOUplift cell-count invariant (R-B28-2)

**Validator fire-order constraint** (arch-R1-I1 closure):
HPOUpliftRollupRow has the B26 `_validate_ci_sentinel_consistency`
validator (declared first) AND the new B28
`_validate_cell_count_bounds` (second). To target only the
cell-count clause, each test must supply a valid CI-sentinel
state. The shared non-sentinel base ALWAYS supplies
`n_seeds=2, n_folds=2, primary_metric_mean=0.1,
primary_metric_ci_lo=0.05, primary_metric_ci_hi=0.15,
bootstrap_skipped_reason=None`; the sentinel base supplies
`n_seeds=2, n_folds=2, primary_metric_*=None,
bootstrap_skipped_reason="no_data"`. With `n_seeds * n_folds =
4` fixed, each test reasons about `n_cells_paired` (bounded
by 4) and `n_skipped_cells` (bounded by `n_cells_paired`)
independently (qa-R2-I1 closure: pin `n_seeds`/`n_folds`
in every base so test arithmetic is verifiable in isolation).

6. `test_hpo_uplift_rollup_row_accepts_n_cells_paired_equals_bound`:
   non-sentinel base, `n_cells_paired=4, n_skipped_cells=0`
   (paired = bound = 4); assert no raise.
7. `test_hpo_uplift_rollup_row_accepts_n_skipped_equals_n_paired`:
   non-sentinel base, `n_cells_paired=4, n_skipped_cells=4`
   (matches the `paired_but_no_valid_loss` sentinel emit
   shape at `bootstrap_hpo_uplift.py:252-264`); assert no
   raise.
8. `test_hpo_uplift_rollup_row_accepts_sentinel_with_zero_counts`:
   sentinel base, `n_cells_paired=0, n_skipped_cells=0`;
   assert no raise.
9. `test_hpo_uplift_rollup_row_rejects_n_cells_paired_exceeds_seeds_times_folds`:
   non-sentinel base, `n_cells_paired=5, n_skipped_cells=0`
   (paired > bound = 4); assert
   `pytest.raises(ValidationError, match=r"n_cells_paired.*exceeds n_seeds \* n_folds")`.
10. `test_hpo_uplift_rollup_row_rejects_n_skipped_cells_exceeds_n_cells_paired`
    (qa-R1-I1 closure: explicit `match=`): non-sentinel base,
    `n_cells_paired=3, n_skipped_cells=4` (skipped > paired);
    assert
    `pytest.raises(ValidationError, match=r"n_skipped_cells.*exceeds n_cells_paired")`.

5 named tests for R-B28-2.

### B28.4.3 Existing-fixture compatibility

11. `test_existing_b17_byte_pin_fixtures_satisfy_b28_invariants`:
    backstop test analogous to B26's B17 backstop. Construct
    the EnsembleLift + HPOUplift B17 helpers and assert
    they construct without raising.

Total B28-new: 11 named + 5 parametrize extras (test #3
runs 6 variants - 1 named = +5) = 16 collected.

### B28.4.4 Expected test delta

Baseline (post-B27): 1059.
- Existing tests: 1059 -> 1059 after the B28.3 step 3
  helper repairs land in the same commit (the repairs
  change only the oracle metric defaults in 2 helpers;
  test semantics unchanged).
- B28-new: 11 named + 5 parametrize extras = 16 collected.
- Total: 1059 + 16 = 1075.

## B28.5 Risks

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R-B28-Risk-1 | The new oracle CI-sentinel clause rejects existing EnsembleLiftRollupRow fixtures. | Medium-confirmed | R1 design swarm identified 5 KNOWN sites across two helper factories where oracle metric defaults are not gated on `n_oracle_cells_paired`. B28.3 step 3 enumerates each site and prescribes the helper-level fix (gate oracle defaults on `n_oracle_cells_paired > 0`). The aggregator itself (`benchmarks/report/bootstrap_ensemble_lift.py:152-159` sentinel + `:320-321` happy path) emits the two structural shapes correctly; the gap is helper-fixture-only. The b16:561 mutation pin uses `model_construct` post-B27 so it bypasses both validators. |
| R-B28-Risk-2 | The new HPOUplift cell-count validators reject an existing fixture. | Low | The aggregator emits sentinels with `n_cells_paired=0, n_skipped_cells=0` (trivially satisfies both bounds). Happy rows are inner-join-bounded; `paired_but_no_valid_loss` emits `n_cells_paired=n_paired_total, n_skipped_cells=n_paired_total` (e.g., 4/4 with bound 4: passes both). Audit of existing fixtures: B17 (`n_seeds=2, n_folds=1, n_cells_paired=1, n_skipped_cells=1` → 1<=2 and 1<=1, passes); B15 (`n_seeds=2, n_folds=2, n_cells_paired=4, n_skipped_cells=1` → 4<=4 and 1<=4, passes); B21/B22 (similar shapes). Live verification via the full suite during build. |
| R-B28-Risk-3 | The composed EnsembleLift validator becomes complex (primary clause + oracle clause + row-count clause). | Low | The two validator methods stay separate (`_validate_row_count_invariants` + `_validate_ci_sentinel_consistency`); only the latter grows. Pydantic v2's mode="after" chain handles ordering. |

## Addressed

R1 design swarm on commit `49f5f37`: architecture-reviewer
(1C / 2I / 1N REQUEST_CHANGES), qa-test-coverage (2C / 1I /
1N REQUEST_CHANGES), style-reviewer (0C / 0I / 0N APPROVE).
Deduplicated total: 3 CRITICAL, 3 IMPROVEMENT, 2 NITPICK.
Closures:

- **arch-R1-C1 + qa-R1-C1** (5 KNOWN fixture sites violate
  the new oracle clause via two helper factories that gate
  oracle defaults on the wrong axis): B28.3 step 3 rewritten
  with the per-site enumeration + prescribed helper-level
  fix (gate oracle defaults on `n_oracle_cells_paired > 0`,
  not on `bootstrap_skipped_reason`). R-B28-Risk-1 elevated
  to "Medium-confirmed".
- **arch-R1-I2 + qa-R1-C2** (tests #3-#5 don't pin the
  primary-axis state, so the wrong validator branch fires
  first): B28.4.1 prefixed with an explicit "Validator
  fire-order constraint" paragraph mandating all-set
  primary metrics + `bootstrap_skipped_reason=None` in
  every oracle-clause test fixture.
- **arch-R1-I1** (tests #6-#10 don't pin the CI-sentinel
  state, so the B26 validator may fire before the new B28
  cell-count validator): B28.4.2 prefixed with an analogous
  fire-order constraint; each test names the base state
  (non-sentinel for #6, #7, #9, #10; sentinel for #8).
- **qa-R1-I1** (test #10 missing `match=`): added
  `match=r"exceeds n_seeds \* n_folds"` to test #10
  matching test #9.
- **arch-R1-N1 + qa-R1-N1** (test count claim conditional
  on fixture audit): rewritten as "1059 -> 1059 after the
  B28.3 step 3 helper repairs land in the same commit".

Test count after R1 closures: 11 named / 16 collected
(unchanged from R1 draft; all closures sharpen pre-existing
test specs).

### R2 design swarm closure

R2 confirming swarm on commit `07c3ac5`: architecture-
reviewer (2C / 1I / 1N REQUEST_CHANGES), qa-test-coverage
(0C / 1I / 0N APPROVE), style-reviewer (0C / 0I / 0N
APPROVE). Deduplicated total: 2 CRITICAL, 2 IMPROVEMENT, 1
NITPICK. Closures:

- **arch-R2-C1** (HPOUplift invariant
  `n_cells_paired + n_skipped_cells <= n_seeds * n_folds`
  semantically wrong: per schema docstring at
  `bootstrap_manifest.py:585-586`, `n_skipped_cells` is a
  SUBSET of `n_cells_paired` not disjoint; the
  `paired_but_no_valid_loss` aggregator emit at
  `bootstrap_hpo_uplift.py:252-264` produces
  `n_cells_paired=4, n_skipped_cells=4` which the wrong
  invariant rejects): R-B28-2 reformulated as TWO
  invariants: `n_cells_paired <= n_seeds * n_folds` AND
  `n_skipped_cells <= n_cells_paired`. Validator body
  rewritten; pseudocode at B28.2 updated; risk table
  updated; tests #6-#10 rewritten to target the new
  branches with `paired_but_no_valid_loss`-shape test #7
  pinning the 4/4 emit.
- **arch-R2-C2** (b15 fixtures at
  `test_hpo_uplift_report_b15.py:213, :227` violate the
  WRONG invariant): resolves automatically under the
  corrected invariant since `4 <= 4` trivially passes.
  R-B28-Risk-2 narrative updated to cite the b15 shape
  explicitly.
- **arch-R2-I1** (site count contradiction: ":143 says 4,
  :190 says 5"): corrected ":143" to "5 KNOWN" matching
  the actual 5-site enumeration.
- **qa-R2-I1** (B28.4.2 base for tests #7/#9/#10 didn't pin
  n_seeds/n_folds): fire-order constraint paragraph rewritten
  to make the shared base ALWAYS supply `n_seeds=2,
  n_folds=2`; test docstrings reason about the fixed bound
  of 4 directly.
- **arch-R2-N1** (Risk-2 narrative imprecision on B17
  shape): risk table updated with the actual B17 fixture
  shape (`n_seeds=2, n_folds=1, n_cells_paired=1,
  n_skipped_cells=1`).

Test count after R2 closures: 11 named / 16 collected
(unchanged; closures were prose + invariant-shape only).

### R1 build-swarm closure

R1 build swarm on commit `f16a845`: code-reviewer (0C / 2I /
0N APPROVE), qa-test-coverage (0C / 0I / 1N APPROVE),
architecture-reviewer (APPROVE), style-reviewer (0C / 0I /
0N APPROVE). Deduplicated total: 0 CRITICAL, 2 IMPROVEMENT,
1 NITPICK. Closures:

- **code-R1-build-I1** ("IDENTICAL BODY TO RollupRow"
  comments on the 3 non-EnsembleLift schemas said
  "keep all 4 copies in sync", but B27 already added a 5th
  copy on EnsembleLiftRollupRow which B28 extends with the
  oracle clause): all 3 markers extended to note that
  EnsembleLiftRollupRow has an EXTENDED version (do not
  verbatim-import the body there).
- **code-R1-build-I2** (`dict[str, object]` base pattern
  adds pyright argument-type warnings): NOT changed.
  Pre-existing repo-wide convention since B26; deferring
  type narrowing would touch every benchmark test file.
  No new pyright errors introduced.
- **qa-R1-build-N1** (B28.3 enumerated 5 sites across 2
  helpers but the build repaired 6 sites across 3 helpers):
  B28.3 step 3 extended with build-phase amendment noting
  the 6th site at `test_b19_n_pair_grid.py:260-290` and
  the structural difference that excluded it from R1
  enumeration.

Test count after R1 build-swarm closures: 11 named / 16
collected (unchanged; all closures were comment + doc edits).

## Deferred

- **D-B28.1**: add the `n_cells_paired + n_skipped_cells
  <= n_seeds * n_folds` cell-count bound to
  `EnsembleLiftRollupRow` (which carries n_seeds + n_folds
  + n_cells_paired). v1 of B28 scopes the bound to
  HPOUpliftRollupRow per D-B26.1's literal text; a future
  audit could extend to EnsembleLift.
- **D-B28.2**: add structural cell-count bounds to
  RollupRow / PairwiseRollupRow / TrainingTimeRollupRow.
  These schemas don't carry n_folds so a bound of the
  form `paired + skipped <= seeds * folds` doesn't apply
  directly; would need n_evaluated bound instead.
