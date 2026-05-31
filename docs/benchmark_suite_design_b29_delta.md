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
3. **Fixture audit**: scan EnsembleLift + RollupRow
   construction sites for any that violate the new
   bounds.
4. **Tests**: add `tests/benchmarks/test_b29_cell_count_bounds.py`
   per B29.4.
5. **Verify**: ruff + pyright + scoped pytest pass at
   1075 + N new tests.

## B29.4 Tests

Baseline (post-B28 main `f42f01c`): 1075 tests collected.

### B29.4.1 EnsembleLift n_pair_grid bound (R-B29-1)

1. `test_ensemble_lift_rollup_row_accepts_n_pair_grid_below_bound`:
   construct with `n_seeds=2, n_folds=2, n_pair_grid=3,
   n_cells_paired=3, n_oracle_cells_paired=3` (3 < 4);
   assert no raise.
2. `test_ensemble_lift_rollup_row_accepts_n_pair_grid_equals_bound`:
   construct with `n_seeds=2, n_folds=2, n_pair_grid=4,
   n_cells_paired=4, n_oracle_cells_paired=4` (4 = 4);
   assert no raise.
3. `test_ensemble_lift_rollup_row_rejects_n_pair_grid_exceeds_bound`:
   construct with `n_seeds=2, n_folds=2, n_pair_grid=5`
   (5 > 4); assert
   `pytest.raises(ValidationError, match=r"n_pair_grid.*exceeds n_seeds \* n_folds")`.

### B29.4.2 RollupRow n_entities bound (R-B29-2)

4. `test_rollup_row_accepts_n_entities_below_n_rows`:
   `n_rows=100, n_entities=10`; assert no raise.
5. `test_rollup_row_accepts_n_entities_equals_n_rows`:
   `n_rows=4, n_entities=4`; assert no raise.
6. `test_rollup_row_rejects_n_entities_exceeds_n_rows`:
   `n_rows=4, n_entities=5`; assert
   `pytest.raises(ValidationError, match=r"n_entities.*exceeds n_rows")`.

### B29.4.3 Existing-fixture compatibility

7. `test_existing_b17_byte_pin_fixtures_satisfy_b29_invariants`:
   construct the B17 EnsembleLift fixture and verify
   `n_pair_grid <= n_seeds * n_folds`; construct the B17
   RollupRow-shaped fixture (via test_bootstrap_render_regression
   helpers) and verify `n_entities <= n_rows`.

### B29.4.4 Expected test delta

Baseline (post-B28): 1075.
- Existing tests: 1075 -> 1075 (assuming fixture audit
  finds no violations).
- B29-new: 7 named tests.
- Total: 1075 + 7 = 1082.

## B29.5 Risks

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R-B29-Risk-1 | The new n_pair_grid bound rejects existing EnsembleLift fixtures. | Low | All existing fixtures use `n_pair_grid <= n_cells_paired <= n_seeds * n_folds` (B23 validator enforces the first inequality). Audit live during build. |
| R-B29-Risk-2 | The new n_entities bound rejects existing RollupRow fixtures. | Low | The aggregator at `benchmarks/report/bootstrap_rollup.py` derives `n_entities = unique_entity_ids.size` from the same rows as `n_rows`, so n_entities <= n_rows is structurally guaranteed. Existing fixtures supply consistent values (B17 + test_bootstrap_render_regression). |

## Deferred

- **D-B29.1**: add structural bounds to PairwiseRollupRow
  and TrainingTimeRollupRow. These schemas don't carry
  `n_rows` / `n_entities` / `n_folds`; no clean invariant
  is available without adding fields. Deferred indefinitely
  unless a future audit identifies a missing field that
  would enable a structural bound.
