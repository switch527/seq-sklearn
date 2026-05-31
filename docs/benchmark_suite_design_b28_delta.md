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
enforcing the structural cell-count invariant:
- `n_cells_paired + n_skipped_cells <= n_seeds * n_folds`

The aggregator at `benchmarks/report/bootstrap_hpo_uplift.py:130-160`
emits sentinel rows with `n_cells_paired=0, n_skipped_cells=0`
(trivially satisfies the bound). Happy rows partition the
inner-join space: paired cells are those where both default
and tuned ran; skipped cells are paired-but-NaN-loss; the
sum is bounded by the total possible cells `n_seeds * n_folds`.

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
def _validate_cell_count_bound(self) -> "HPOUpliftRollupRow":
    # B28 / D-B26.1 closure: paired + skipped <= seeds * folds.
    # The aggregator's inner-join space is bounded by the total
    # possible (seed, fold) cells; paired (both default + tuned
    # ran) plus skipped (paired-but-NaN-loss) cannot exceed it.
    total_possible = self.n_seeds * self.n_folds
    if self.n_cells_paired + self.n_skipped_cells > total_possible:
        raise ValueError(
            f"n_cells_paired ({self.n_cells_paired}) + n_skipped_cells "
            f"({self.n_skipped_cells}) exceeds n_seeds * n_folds "
            f"({self.n_seeds} * {self.n_folds} = {total_possible})"
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
3. **Fixture audit**: scan EnsembleLift + HPOUplift
   construction sites; repair any that violate the new
   invariants.
4. **Tests**: add `tests/benchmarks/test_b28_schema_validators.py`
   per B28.4.
5. **Verify**: ruff + pyright + scoped pytest pass at
   1059 + N new tests.

## B28.4 Tests

Baseline (post-B27 main `37cbfc0`): 1059 tests collected.

### B28.4.1 Oracle CI-sentinel invariant (R-B28-1)

1. `test_ensemble_lift_rollup_row_accepts_oracle_all_none_with_zero_oracle_cells`:
   construct a non-sentinel row with `n_oracle_cells_paired=0`
   and `oracle_metric_*=None`; assert no raise.
2. `test_ensemble_lift_rollup_row_accepts_oracle_all_set_with_positive_oracle_cells`:
   construct with `n_oracle_cells_paired=4` and
   `oracle_metric_*` all populated; assert no raise.
3. `test_ensemble_lift_rollup_row_rejects_oracle_partially_set`:
   parametrize over 6 mixed-oracle variants (same shape as
   B27 D-B26.3 closure); assert
   `pytest.raises(ValidationError, match=r"oracle_metric_mean.*must be all-None or all-non-None")`.
4. `test_ensemble_lift_rollup_row_rejects_oracle_none_with_positive_oracle_cells`:
   construct with `n_oracle_cells_paired=4` and
   `oracle_metric_*=None`; assert
   `pytest.raises(ValidationError, match=r"all None but n_oracle_cells_paired > 0")`.
5. `test_ensemble_lift_rollup_row_rejects_oracle_set_with_zero_oracle_cells`:
   construct with `n_oracle_cells_paired=0` and
   `oracle_metric_*` all populated; assert
   `pytest.raises(ValidationError, match=r"all populated but n_oracle_cells_paired == 0")`.

Test #3 parametrizes 6 variants -> 6 collected. Tests #1,
#2, #4, #5 = 4 named. B28.4.1 totals 5 named, 10 collected.

### B28.4.2 HPOUplift cell-count invariant (R-B28-2)

6. `test_hpo_uplift_rollup_row_accepts_paired_plus_skipped_equals_bound`:
   construct with `n_seeds=2, n_folds=2,
   n_cells_paired=3, n_skipped_cells=1` (total = 4 = bound);
   assert no raise.
7. `test_hpo_uplift_rollup_row_accepts_paired_plus_skipped_below_bound`:
   construct with `n_cells_paired=2, n_skipped_cells=1`
   (total = 3 < 4); assert no raise.
8. `test_hpo_uplift_rollup_row_accepts_sentinel_with_zero_counts`:
   construct with `n_cells_paired=0, n_skipped_cells=0`
   (sentinel); assert no raise.
9. `test_hpo_uplift_rollup_row_rejects_paired_plus_skipped_exceeds_bound`:
   construct with `n_cells_paired=3, n_skipped_cells=2`
   (total = 5 > 4 = bound); assert
   `pytest.raises(ValidationError, match=r"exceeds n_seeds \* n_folds")`.
10. `test_hpo_uplift_rollup_row_rejects_paired_alone_exceeds_bound`:
    construct with `n_cells_paired=5, n_skipped_cells=0`
    (paired alone exceeds 4); assert `ValidationError`.

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
- Existing tests: 1059 -> 1059 (assuming fixture audit at
  B28.3.3 finds no violations).
- B28-new: 11 named + 5 parametrize extras = 16 collected.
- Total: 1059 + 16 = 1075.

## B28.5 Risks

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R-B28-Risk-1 | The new oracle CI-sentinel clause rejects an existing EnsembleLiftRollupRow fixture. | Low | The aggregator already emits the two structural shapes (sentinel: oracle all-None + n_oracle=0; happy: oracle all-set + n_oracle>0). Existing fixtures across B17/B19/B20/B23/B24/B25 all follow these shapes per the B27.2 per-site enumeration. The b16:561 mutation pin is already using `model_construct` post-B27 so its half-populated row bypasses the new clause too. |
| R-B28-Risk-2 | The new HPOUplift cell-count validator rejects an existing fixture. | Low | The aggregator emits sentinels with `n_cells_paired=0, n_skipped_cells=0` (trivially satisfies). Happy rows are inner-join-bounded. B17/B21/B22 fixtures supply `n_cells_paired=4` with appropriate `n_seeds=2, n_folds=2` (4 <= 4). Live verification via the suite. |
| R-B28-Risk-3 | The composed EnsembleLift validator becomes complex (primary clause + oracle clause + row-count clause). | Low | The two validator methods stay separate (`_validate_row_count_invariants` + `_validate_ci_sentinel_consistency`); only the latter grows. Pydantic v2's mode="after" chain handles ordering. |

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
