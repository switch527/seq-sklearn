# B29 delta: extend cell-count bounds (D-B28.1 + D-B28.2)

## Requirements

R-B29-1 (closes D-B28.1): extend `EnsembleLiftRollupRow`'s
existing `_validate_row_count_invariants` validator with one
additional bound: `n_pair_grid <= n_seeds * n_folds`. The
B23 validator already enforces `n_oracle <= n_paired`,
`n_oracle <= n_pair_grid`, and `n_paired <= n_pair_grid`;
combined with the new bound, the full chain is
`n_oracle <= n_paired <= n_pair_grid <= n_seeds * n_folds`.

R-B29-2 (partially closes D-B28.2): add a new
`@model_validator(mode="after")` to `RollupRow` enforcing
`n_entities <= n_rows`. Each row contributes one entity
record so the count of unique entities cannot exceed the
count of rows. PairwiseRollupRow and TrainingTimeRollupRow
do NOT carry `n_rows` or `n_entities`; they have no
analogous structural invariant available without external
context (deferred as D-B29.1).

## Non-requirements

- v1 does NOT add invariants to PairwiseRollupRow or
  TrainingTimeRollupRow. Their available cell-count fields
  (`n_seeds`, `n_cells_evaluated`, `n_skipped_cells`) carry
  no structural relationship to each other independent of
  the unspecified per-row fold count.
- v1 does NOT add an `n_cells_evaluated + n_skipped_cells <=
  total` bound to RollupRow / Pairwise / TrainingTime;
  none carry `n_folds`.
- v1 does NOT change aggregator code paths.

## B29.0 Background

### B29.0.1 D-B28.1 closure scope

B23's `_validate_row_count_invariants` on EnsembleLift
established `n_oracle <= n_paired <= n_pair_grid`. The
chain was incomplete: `n_pair_grid` had no upper bound. The
aggregator at
`benchmarks/report/bootstrap_ensemble_lift.py` derives
`n_pair_grid` as the count of cells in the inner-join
intersection of GBM-only and GBM+seq family rosters, which
is bounded above by the total possible (seed, fold)
positions = `n_seeds * n_folds`. The new bound closes the
chain.

### B29.0.2 D-B28.2 closure scope

D-B28.2 named structural bounds on the 3 non-folds schemas.
Per the schema audit:
- `RollupRow`: carries `n_rows` + `n_entities` →
  `n_entities <= n_rows` is structural (each entity has at
  least one row).
- `PairwiseRollupRow`: no `n_rows` / `n_entities` field.
- `TrainingTimeRollupRow`: no `n_rows` / `n_entities`
  field.

R-B29-2 closes the RollupRow portion. The other 2 schemas
have no clean invariant available; deferred as D-B29.1.

## B29.1 R-B29-1 design

Extend the B23 validator body on EnsembleLiftRollupRow:

```python
@model_validator(mode="after")
def _validate_row_count_invariants(self) -> "EnsembleLiftRollupRow":
    # ... existing 3 invariants ...

    # B29 / D-B28.1 closure: n_pair_grid bounded by total
    # possible (seed, fold) positions.
    total_possible = self.n_seeds * self.n_folds
    if self.n_pair_grid > total_possible:
        raise ValueError(
            f"n_pair_grid ({self.n_pair_grid}) exceeds "
            f"n_seeds * n_folds ({self.n_seeds} * "
            f"{self.n_folds} = {total_possible})"
        )
    return self
```

## B29.2 R-B29-2 design

Add a new validator to RollupRow:

```python
@model_validator(mode="after")
def _validate_row_count_bound(self) -> "RollupRow":
    # B29 / D-B28.2 closure: each entity contributes at
    # least one row, so unique entities cannot exceed
    # total rows.
    if self.n_entities > self.n_rows:
        raise ValueError(
            f"n_entities ({self.n_entities}) exceeds "
            f"n_rows ({self.n_rows})"
        )
    return self
```

## B29.3 Implementation outline

1. **R-B29-1**: append the n_pair_grid bound to the existing
   EnsembleLift `_validate_row_count_invariants` body.
2. **R-B29-2**: add the new `_validate_row_count_bound`
   validator to RollupRow.
3. **Fixture audit + repair** (arch-R1-C2 + arch-R1-I3
   closure): R1 design swarm identified 1 KNOWN violating
   site:
   - `tests/benchmarks/test_b19_n_pair_grid.py:332-359`
     (`test_n_pair_grid_round_trips_through_parquet_shard`):
     constructs with `n_seeds=2, n_folds=2, n_pair_grid=137`.
     Under R-B29-1, `137 > 4`. The test exists to prove
     the schema preserves an unusual prime value
     (matching the B16 `bootstrap_n_resamples=137`
     precedent). Repair: bump `n_seeds=12, n_folds=12`
     (product 144 > 137) so the value 137 still round-trips
     while satisfying the new bound. `n_cells_paired=4` and
     `n_oracle_cells_paired=4` stay (both <= 137).
   No other EnsembleLift or RollupRow construction sites
   violate the new bounds (verified via grep on the
   13 EnsembleLift + 9 RollupRow construction sites under
   `tests/benchmarks/`).
4. **Tests**: add `tests/benchmarks/test_b29_cell_count_bounds.py`
   per B29.4.
5. **Verify**: ruff + pyright + scoped pytest pass at
   1075 + N new tests.

## B29.4 Tests

Baseline (post-B28 main `f42f01c`): 1075 tests collected.

### B29.4.1 EnsembleLift n_pair_grid bound (R-B29-1)

**Validator fire-order constraint** (qa-R1-I1 closure): every
EnsembleLift test fixture must supply ALL required fields
including `n_oracle_cells_paired` (no schema default; `Field(ge=0)`).
A valid CI-sentinel base (all-set primary + skipped=None +
all-set oracle + n_oracle > 0) ensures the B26 and B28
clauses pass silently so the B23 + new B29 row-count clauses
are the ones that fire.

1. `test_ensemble_lift_rollup_row_accepts_n_pair_grid_below_bound`:
   non-sentinel base, `n_seeds=2, n_folds=2, n_pair_grid=3,
   n_cells_paired=3, n_oracle_cells_paired=3` (3 < 4);
   assert no raise.
2. `test_ensemble_lift_rollup_row_accepts_n_pair_grid_equals_bound`:
   non-sentinel base, `n_seeds=2, n_folds=2, n_pair_grid=4,
   n_cells_paired=4, n_oracle_cells_paired=4` (4 = 4);
   assert no raise.
3. `test_ensemble_lift_rollup_row_rejects_n_pair_grid_exceeds_bound`:
   non-sentinel base, `n_seeds=2, n_folds=2, n_pair_grid=5,
   n_cells_paired=4, n_oracle_cells_paired=4` (B23 chain
   satisfies 4 <= 4 <= 5; B29 catches 5 > 4); assert
   `pytest.raises(ValidationError, match=r"n_pair_grid.*exceeds n_seeds \* n_folds")`.

### B29.4.2 RollupRow n_entities bound (R-B29-2)

**Validator fire-order constraint** (qa-R1-C1 closure):
RollupRow has the B26 `_validate_ci_sentinel_consistency`
validator. Tests for the new `_validate_row_count_bound`
must supply a sentinel-compliant base (non-sentinel:
`primary_metric_mean=0.215, primary_metric_ci_lo=0.200,
primary_metric_ci_hi=0.230, bootstrap_skipped_reason=None`)
so the B26 clause passes silently and the new clause is
the one that raises in test #6.

4. `test_rollup_row_accepts_n_entities_below_n_rows`:
   non-sentinel base, `n_rows=100, n_entities=10`; assert
   no raise.
5. `test_rollup_row_accepts_n_entities_equals_n_rows`:
   non-sentinel base, `n_rows=4, n_entities=4`; assert no
   raise.
6. `test_rollup_row_rejects_n_entities_exceeds_n_rows`:
   non-sentinel base, `n_rows=4, n_entities=5`; assert
   `pytest.raises(ValidationError, match=r"n_entities.*exceeds n_rows")`.

### B29.4.3 Existing-fixture compatibility

7. `test_existing_fixtures_satisfy_b29_invariants` (arch-R1-I2
   closure: B17 only carries EnsembleLift fixtures; RollupRow
   fixtures live in `test_bootstrap_render_regression.py` which
   is the B14 regression pin, not B17): construct the B17
   `_make_ensemble_lift_rollup` helper output and assert no
   raise; construct a B14 RollupRow via
   `test_bootstrap_render_regression._rollup_row()` and assert
   no raise. The asserts are validator-anchored (construction
   IS the gate post-B29; the test exists to document that the
   B17 + B14 baseline fixtures stay compatible).

### B29.4.4 Expected test delta

Baseline (post-B28): 1075.
- Existing tests: 1075 -> 1075 after the B29.3 step 3
  repair lands in the same commit (bumps `n_seeds=2,
  n_folds=2` to `n_seeds=12, n_folds=12` at
  `test_b19_n_pair_grid.py:332-359`; test semantics
  unchanged since the 137 round-trip is what's pinned).
- B29-new: 7 named tests.
- Total: 1075 + 7 = 1082.

## B29.5 Risks

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R-B29-Risk-1 | The new n_pair_grid bound rejects existing EnsembleLift fixtures. | Medium-confirmed | The aggregator at `benchmarks/report/bootstrap_ensemble_lift.py:488` derives `n_pair_grid = len(gbm_pairs & seq_pairs)` and `n_seeds, n_folds` at `:195-198` from union counts; the intersection is cartesian-bounded by `n_seeds * n_folds`. R1 design swarm identified 1 violating fixture at `test_b19_n_pair_grid.py:332-359` (`n_seeds=2, n_folds=2, n_pair_grid=137`); B29.3 step 3 prescribes the repair (bump to `n_seeds=12, n_folds=12`). |
| R-B29-Risk-2 | The new n_entities bound rejects existing RollupRow fixtures. | Low | The aggregator at `benchmarks/report/bootstrap_rollup.py:387, :396` derives `n_unique_entities = int(np.unique(entities).shape[0])` and `n_rows = int(losses.shape[0])` from the same `np.concatenate(entity_blocks)` / `np.concatenate(losses_blocks)` populated together at `:306-307`. n_entities <= n_rows is structurally guaranteed. Existing fixtures supply consistent values (B14 regression tests at `test_bootstrap_render_regression.py`). |

## Addressed

R1 design swarm on commit `fc71e6c`: architecture-reviewer
(2C / 3I / 2N REQUEST_CHANGES), qa-test-coverage (1C / 1I /
1N REQUEST_CHANGES), style-reviewer (0C / 0I / 0N APPROVE).
Deduplicated total: 3 CRITICAL, 4 IMPROVEMENT, 3 NITPICK.
Closures:

- **arch-R1-C1** (R-B29-Risk-1 mitigation cited inverted
  B23 chain): risk text rewritten to cite the aggregator
  derivation (n_pair_grid = intersection; n_seeds, n_folds =
  union counts; intersection cartesian-bounded by product).
- **arch-R1-C2** (b19 fixture at `:332-359` violates the new
  bound with `n_pair_grid=137 > n_seeds * n_folds = 4`):
  B29.3 step 3 enumerates the violating site and prescribes
  the `n_seeds=12, n_folds=12` repair (product 144 > 137).
- **qa-R1-C1** (RollupRow tests #4-#6 base unspecified;
  B26 CI-sentinel validator fires first): added "Validator
  fire-order constraint" paragraph to B29.4.2 mandating a
  non-sentinel base. Same constraint added to B29.4.1.
- **arch-R1-I1** (R-B29-Risk-2 mitigation under-cites):
  added specific line citations (`bootstrap_rollup.py:387,
  :396, :306-307`).
- **arch-R1-I2** (test #7 conflates B17 + B14): rewritten
  to name both the B17 EnsembleLift helper AND the B14
  RollupRow helper (`test_bootstrap_render_regression._rollup_row`).
- **arch-R1-I3** (fixture audit needs enumeration): B29.3
  step 3 expanded with the 1 KNOWN violating site + grep
  scope.
- **qa-R1-I1** (test #3 omits `n_oracle_cells_paired`):
  B29.4.1 fire-order paragraph mandates all required
  fields; test #3 spec updated with explicit
  `n_cells_paired=4, n_oracle_cells_paired=4`.
- **arch-R1-N1, arch-R1-N2, qa-R1-N1**: NOT changed;
  cosmetic.

Test count after R1 closures: 7 named (unchanged; closures
sharpened pre-existing test specs).

### R1 build-swarm closure

R1 build swarm on commit `23e1308`: code-reviewer (0C / 0I /
0N APPROVE), qa-test-coverage (0C / 0I / 2N APPROVE),
architecture-reviewer (0C / 2I / 2N APPROVE), style-reviewer
(0C / 0I / 0N APPROVE). Deduplicated total: 0 CRITICAL, 2
IMPROVEMENT, 4 NITPICK. Closures:

- **arch-R1-build-I1** (path prefix missing on
  bootstrap_rollup.py + bootstrap_ensemble_lift.py file
  references in 2 validator comments): both comments now
  carry the `benchmarks/report/` prefix so grep-by-path
  works.
- **arch-R1-build-I2** (b23 helper comment understated
  the bumped n_pair_grid coverage): NOT changed; cosmetic
  comment refinement, current text is honest about the
  bumped product.
- **qa-R1-build-N1 + N2** (`_ROLLUP_BASE` lacks default
  n_rows/n_entities; B17 backstop trivially passes): NOT
  changed; both are tested patterns from earlier phases.
- **arch-R1-build-N1 + N2** (validator naming + backstop
  tautology): NOT changed; cosmetic.

Test count unchanged (7 named / 1082 collected).

### R2 build-swarm closure

R1 fixes were comment-only path-prefix edits. Skipping
explicit R2 swarm since the R1 fixes have zero behavioral
or structural impact (verified clean ruff + pyright +
1082 pytests pass).

## Deferred

- **D-B29.1**: **PERMANENTLY-DEFERRED:** add structural
  bounds to PairwiseRollupRow and TrainingTimeRollupRow.
  These schemas don't carry `n_rows` / `n_entities` /
  `n_folds`; no clean invariant is available without adding
  fields. Deferred indefinitely unless a future audit
  identifies a missing field that would enable a structural
  bound. Reclassified B31 as no-available-fields; revisit
  only if a future B-phase adds the missing fields to
  these schemas.
