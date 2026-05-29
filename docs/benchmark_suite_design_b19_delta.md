# B19 design delta: `n_pair_grid` on `EnsembleLiftRollupRow` for asymmetric-roster partial-coverage (D-B16.7)

**Scope**: D-B16.7 fixes a false-positive partial-coverage
asterisk on the B16 ensemble-lift CI cell when the two
families (B11 baseline + seq) ran on asymmetric (seed, fold)
rosters. The renderer currently computes
`expected = n_seeds * n_folds` where `n_seeds` and `n_folds`
are UNION counts across both families, then flags
`partial = n_cells_paired < expected`. For asymmetric rosters
(e.g., GBM on seeds `{0,1,2}` and seq on seeds `{0,1}`),
expected = 6 while the true intersection grid is only 4
cells; a fully-covered bootstrap on the intersection
produces `n_cells_paired = 4 < expected = 6` and the
asterisk fires incorrectly.

The fix: add `n_pair_grid: int` to `EnsembleLiftRollupRow`
carrying the size of the INTERSECTION of `(seed, fold)`
pairs across the two families. The aggregator computes this
from the inner join of `gbm_cells[["seed", "fold_index"]]`
and `seq_cells[["seed", "fold_index"]]`. The renderer uses
`n_pair_grid` as the `expected` value so the asterisk
fires only when `n_cells_paired < n_pair_grid` (true
under-coverage of the intersection grid).

The existing `n_seeds` and `n_folds` fields are PRESERVED
unchanged. They remain useful as audit data (which seeds
and folds were involved across either family) and on
sentinel rows. Only the renderer's partial-flag computation
changes.

## Requirements

The grading rubric for every reviewer finding traces back to
one of these.

- **R-B19-1** `EnsembleLiftRollupRow` gains an
  `n_pair_grid: int = Field(ge=0)` field. The four other
  RollupRow schemas (`RollupRow`, `PairwiseRollupRow`,
  `TrainingTimeRollupRow`, `HPOUpliftRollupRow`) are NOT
  touched (their partial-flag computation is correct as-is
  per their respective semantics).
- **R-B19-2** The B16 aggregator
  (`aggregate_bootstrap_ensemble_lift_rollup`) computes
  `n_pair_grid` as the cardinality of the intersection of
  `(seed, fold)` pairs across `gbm_cells` and `seq_cells`.
  Implementation: build a set of `(seed, fold)` tuples from
  each family's cells and take the intersection.
- **R-B19-3** The B16 renderer
  (`_render_complete_table_with_ci` in
  `benchmarks/report/ensemble_lift.py`) reads
  `rollup_row.n_pair_grid` instead of
  `rollup_row.n_seeds * rollup_row.n_folds` when computing
  `partial`. The other four families' renderers are NOT
  touched.
- **R-B19-4** Sentinel rows (when the aggregator emits
  `no_gbm_predictions`, `no_seq_predictions`, or
  `all_cells_skipped_in_manifest`) set `n_pair_grid = 0`.
  The renderer's `(no CI)` short-circuit at
  `if rollup_row.bootstrap_skipped_reason is not None`
  fires before the partial computation, so the sentinel
  value is documentary only.
- **R-B19-5** The Guard A invariant from B17 (no
  `primary_loss_*` field on any RollupRow other than
  `primary_loss_column`) continues to hold; the rename
  surface is untouched.
- **R-B19-6** The change ships in ONE commit. The risk is
  bounded by:
  - The existing 27 B11 driver tests pass byte-equivalent
    (none reference `n_pair_grid` so adding it is purely
    additive).
  - The existing 17 B16 aggregator tests pass; the schema
    addition is additive and the per-test
    `_make_ensemble_lift_row` factory updated to default
    `n_pair_grid` to the same value as `n_cells_paired`
    (the symmetric-roster happy-path default).
  - The existing 13 B16 wrapper tests pass byte-equivalent.
  - The existing 24 B16 renderer tests pass byte-equivalent
    EXCEPT the partial-asterisk tests; those are updated to
    set `n_pair_grid` explicitly on their fixtures.
  - The B17 byte-identity pin for the B16 renderer family
    has `n_pair_grid` added to its `EnsembleLiftRollupRow`
    fixture (arch-R1-C1 closure).

## B19.0 What the change actually adds

Old `EnsembleLiftRollupRow` field set (after B17 rename):

```python
class EnsembleLiftRollupRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dataset_name: str
    task_type: str
    primary_metric: str
    primary_loss_column: str
    n_seeds: int = Field(ge=0)
    n_folds: int = Field(ge=0)
    n_cells_paired: int = Field(ge=0)
    n_skipped_cells: int = Field(ge=0)
    primary_metric_mean: float | None = None
    primary_metric_ci_lo: float | None = None
    primary_metric_ci_hi: float | None = None
    bootstrap_seed: int
    bootstrap_n_resamples: int = Field(ge=0)
    bootstrap_confidence: float = 0.95
    bootstrap_rng_algorithm: str = "PCG64"
    bootstrap_numpy_version: str
    bootstrap_skipped_reason: str | None = None
    manifest_fingerprint: str
```

New: add ONE field between `n_cells_paired` and
`n_skipped_cells`:

```python
    n_pair_grid: int = Field(ge=0)
```

The field has no default (required keyword). Sentinel rows
pass `n_pair_grid=0`; happy-path rows pass the intersection
size.

**Placement note (arch-R1-N1)**: the new field is inserted
mid-schema (between the two `n_*` count fields it
semantically belongs with) rather than at the end. Pandas /
parquet writes columns in declaration order, so the new
column lands between `n_cells_paired` and
`n_skipped_cells` in the resulting parquet. The B16 schema
shipped in October 2025; no production shard reader depends
on column ordering (the round-trip test asserts on
`model_dump()` dict equality, not on column-order
preservation). Mid-schema placement keeps the field
semantically co-located with its companions; if a future
maintainer prefers strict end-of-schema, the only test
update needed is the column-order assertion in the B16
round-trip, which does not exist today.

## B19.1 The intersection computation

The intersection cardinality is computed ONCE PER DATASET at
the `aggregate_bootstrap_ensemble_lift_rollup` caller in
`benchmarks/report/bootstrap_ensemble_lift.py:323-335` (the
same site that builds `seed_fold_pairs` and slices
`gbm_ds` / `seq_ds`), then passed into `_build_dataset_rollup`
as an explicit kwarg alongside the existing parameters
(arch-R1-I2 closure: single source of truth for the
intersection; the helper does not recompute it):

```python
# At the caller, after gbm_ds / seq_ds slicing:
gbm_pairs = set(
    zip(
        gbm_ds["seed"].astype(int),
        gbm_ds["fold_index"].astype(int),
        strict=True,
    )
)
seq_pairs = set(
    zip(
        seq_ds["seed"].astype(int),
        seq_ds["fold_index"].astype(int),
        strict=True,
    )
)
n_pair_grid = len(gbm_pairs & seq_pairs)

# _build_dataset_rollup receives n_pair_grid as a new kwarg:
rows.append(
    _build_dataset_rollup(
        dataset_name=dataset_name,
        ...,
        n_pair_grid=n_pair_grid,
        ...,
    )
)
```

Edge case: if either family's cells DataFrame is empty,
the intersection is empty and `n_pair_grid = 0`. This is
benign: the empty-cells branch triggers a sentinel row anyway
(the aggregator's `computed.cells == ()` check on the
`compute_per_cell_lift_deltas` result), so `n_pair_grid = 0`
flows through to the sentinel-emit path. The Gate D check
at `:307-308` already returns `[]` from the top-level
aggregator when either `gbm_cells_all` or `seq_cells_all` is
empty, so the per-dataset n_pair_grid computation runs only
when both family rosters have at least one row globally.

The intersection-set construction has O(N+M) time complexity
where N and M are the row counts of the two family
DataFrames. This is cheaper than the existing
`pd.concat(...).nunique()` calls for `n_seeds` and `n_folds`,
which are O(N log N) due to the sort.

## B19.2 The renderer fix

In `_render_complete_table_with_ci` at
`benchmarks/report/ensemble_lift.py:116-172`:

Old:

```python
expected = rollup_row.n_seeds * rollup_row.n_folds
partial = rollup_row.n_cells_paired < expected and expected > 0
```

New:

```python
expected = rollup_row.n_pair_grid
partial = rollup_row.n_cells_paired < expected and expected > 0
```

The `and expected > 0` guard stays as a defensive double-
check. A non-sentinel row (i.e., `bootstrap_skipped_reason
is None`) always has `n_pair_grid >= 1` by construction
because the aggregator's Gate D at
`bootstrap_ensemble_lift.py:307-308` returns `[]` when
either family roster is empty, and the per-dataset
sentinel branch at `:195-223` emits a sentinel row
(setting `bootstrap_skipped_reason`) whenever
`computed.cells` is empty (which covers the case where the
intersection size is zero per-dataset). Sentinel rows
route through the `(no CI)` branch via the earlier
`bootstrap_skipped_reason is not None` check at `:148`,
never reaching the partial computation. The `expected > 0`
guard is therefore defensive, not load-bearing (arch-R1-I3
closure).

## B19.3 Test surface

### Existing tests touched

1. **`tests/benchmarks/test_bootstrap_manifest.py`**: the
   `_make_ensemble_lift_row` factory at
   `:534-559` (post-B17) needs a default value for
   `n_pair_grid`. Set it to the same value as `n_cells_paired`
   (the symmetric happy-path default). The round-trip,
   extra-forbid, empty-shard, absent-on-load, and
   ExperimentSpec tests don't assert on field count; they
   continue to pass.

2. **`tests/benchmarks/test_bootstrap_ensemble_lift.py`**:
   the `_per_cell_pairs` helper at `:153` (post-B17)
   constructs `ComputePerCellLiftDeltasResult` records and
   the aggregator test fixtures pass through to the
   aggregator. The aggregator under test now writes
   `n_pair_grid` on every row; the 17 aggregator tests need
   their assertions updated to either ignore `n_pair_grid`
   or assert specific values where the test's intent
   includes the partial-coverage flag.

3. **`tests/benchmarks/test_run_bootstrap_ensemble_lift_wrapper.py`**:
   the helper at `:114-152` constructs
   `ComputePerCellLiftDeltasResult` records via stubs; the
   13 wrapper tests don't assert on the rollup row's field
   shape (they assert on the rollup-file existence and the
   sentinel-file content). Pass byte-equivalent.

4. **`tests/benchmarks/test_ensemble_lift_report_b16.py`**:
   the `_make_rollup_row` factory at `:78-106` (post-B17)
   constructs `EnsembleLiftRollupRow` instances; needs a
   default for `n_pair_grid`. The 24 renderer tests
   currently exercise:
   - Test #11 (`_marks_partial_fold_cell_with_asterisk`):
     fixture has `n_seeds=2, n_folds=2, n_cells_paired=3`
     so expected=4, partial=True under the OLD logic. Under
     the NEW logic this test must explicitly set
     `n_pair_grid=4` to keep the asterisk firing (matches
     "all 4 intersection cells expected but only 3
     covered").
   - Test #9 (`_surfaces_skipped_cells_footnote_without_asterisk_when_no_pairing_asymmetry`):
     fixture has `n_seeds=2, n_folds=2, n_cells_paired=4`
     so expected=4, partial=False under the OLD logic.
     Under the NEW logic this test must explicitly set
     `n_pair_grid=4` to keep the asterisk absent.
   - All other tests have `bootstrap_skipped_reason=None`
     and the CI cell renders via `format_ci_cell`; the
     `n_pair_grid` value determines the asterisk on those.
     The factory default mirrors `n_cells_paired` so happy-
     path fixtures continue to match.

5. **`tests/benchmarks/test_b17_byte_identity_pins.py`**
   (arch-R1-C1 closure): the
   `_make_ensemble_lift_rollup` helper at `:314-334`
   directly constructs an `EnsembleLiftRollupRow` for the
   B16 family's byte-identity pin. Adding a REQUIRED
   `n_pair_grid` field makes the kwarg-construction at
   `:316` fail at module-import without an update. Fix:
   add `n_pair_grid=1` to the fixture (matches the
   fixture's `n_cells_paired=1`, preserving the existing
   pin behavior since `n_cells_paired == n_pair_grid`
   yields the same `partial=False` result as the pre-B19
   `n_cells_paired < n_seeds * n_folds = 1 * 1 = 1` did
   not. Actually the pre-B19 fixture has
   `n_seeds=2, n_folds=1, n_cells_paired=1` so expected=2
   and partial=True; the post-B19 fixture with
   `n_pair_grid=1, n_cells_paired=1` yields partial=False).
   The B16 byte-identity pin asserts the CI cell renders
   the expected `mean [lo, hi]*` shape with the MANDATORY
   trailing asterisk (per B17's mutation-sensitive regex
   tightening); to keep the asterisk firing post-B19, set
   `n_pair_grid=2` (matches the pre-B19 expected count)
   instead of `n_pair_grid=1`. This is the correct
   semantic: the pin fixture intentionally exercises the
   partial-coverage flag, and the post-B19 fixture must
   preserve that intent by setting `n_pair_grid` strictly
   greater than `n_cells_paired`.

### NEW tests (B19-specific)

`tests/benchmarks/test_b19_n_pair_grid.py` (NEW; 7 tests):

1. `test_aggregator_writes_n_pair_grid_as_intersection_cardinality`:
   fixture with GBM on seeds `{0,1,2}` x folds `{0,1}` (6
   cells) and seq on seeds `{0,1}` x folds `{0,1}` (4
   cells). Assert the rollup row's `n_pair_grid == 4`
   (the intersection size) AND `n_seeds == 3` (the union)
   AND `n_folds == 2` AND (qa-R1-I1 closure)
   `row.n_pair_grid != row.n_seeds * row.n_folds` so the
   union-vs-intersection discrimination is self-documenting
   and survives a fixture reshape: a future regression that
   reverts the aggregator to writing `n_seeds * n_folds`
   would fail this assertion specifically rather than
   silently passing on a 4 vs 6 numeric comparison.
2. `test_aggregator_writes_n_pair_grid_zero_when_intersection_empty`:
   fixture with GBM on seeds `{0,1}` and seq on seeds
   `{2,3}` (disjoint). The aggregator's existing
   `seen_no_gbm` / `seen_no_seq` flags both trip and the
   sentinel branch fires. Assert the sentinel row has
   `n_pair_grid == 0`.
3. `test_renderer_partial_flag_uses_n_pair_grid_not_n_seeds_times_n_folds`:
   construct a rollup row with `n_seeds=3, n_folds=2,
   n_cells_paired=4, n_pair_grid=4`. Under the OLD logic
   `expected = 6 > 4` would fire the asterisk; under the
   NEW logic `expected = 4 == 4` does NOT fire. Assert
   the rendered CI cell does NOT carry a trailing `*`.
4. `test_renderer_partial_flag_fires_when_n_cells_paired_less_than_n_pair_grid`:
   construct a row with `n_pair_grid=4, n_cells_paired=3`.
   Assert the rendered CI cell DOES carry a trailing `*`.
5. `test_n_pair_grid_round_trips_through_parquet_shard`:
   write an `EnsembleLiftRollupRow` with
   `n_pair_grid=137` via `write_ensemble_lift_rollup`,
   load it back, assert the loaded row's
   `n_pair_grid == 137` (qa-R1-I2 closure: 137 is an
   unusual prime far from any factory default, matching the
   B16 `bootstrap_n_resamples=137` precedent for
   detectable-value sentinels; a silent coercion to a
   small-integer default like `n_cells_paired` would not
   match 137).
6. `test_n_pair_grid_zero_passes_field_validator`:
   construct a row with `n_pair_grid=0`; assert no
   ValidationError (the `Field(ge=0)` constraint accepts
   the boundary value).
7. `test_n_pair_grid_negative_rejected_by_field_validator`
   (qa-R2-N1 closure): construct a row with
   `n_pair_grid=-1`; assert pydantic raises
   `ValidationError`. Pins the lower bound of the
   `Field(ge=0)` constraint from below; the boundary value
   0 is covered by test #6 from above.

Expected test delta after the build:
- Existing tests: 863 to 863 (no count change; fixtures
  updated in place).
- B19-new: 7 tests.
- Total: 863 + 7 = 870 expected post-refactor.

## B19.4 Risks

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R-B19-Risk-1 | An existing test asserts on the rollup row's exact field set (e.g., via `model_fields` length or `model_dump()` key count). Adding `n_pair_grid` breaks the assertion. | Medium | Guard A from B17 only asserts no `primary_loss_*` fields exist; it does NOT count fields. The B17 byte-pin in `test_bootstrap_render_regression.py` pins B5 leaderboard bytes, not B16. The B16 round-trip test asserts field-by-field equality on the values, not on the schema's field count. Audit by reading each existing assertion. |
| R-B19-Risk-2 | The intersection set construction uses `astype(int)` and `set(zip(...))` which would crash on non-int seed/fold values. | Low | The B5 manifest contract guarantees integer `seed` and `fold_index` columns. The existing aggregator at `bootstrap_ensemble_lift.py:323` already does `int(s)` / `int(f)` on the same columns. No new failure surface. |
| R-B19-Risk-3 | A future maintainer copy-pastes `n_pair_grid = 0` into a happy-path sentinel emission accidentally. | Low | The new test `test_aggregator_writes_n_pair_grid_as_intersection_cardinality` would fail (asserts 4, not 0). |
| R-B19-Risk-4 | The B17 Guard B source-tree grep test catches a new reference to `primary_loss_*` but does not catch a new reference to the old `n_seeds * n_folds` partial-flag computation in a future renderer. | Low | The OLD `n_seeds * n_folds` computation lives in only one renderer (B16); the other four are correct under their semantics. Out of scope for this delta. |
| R-B19-Risk-5 | A parquet shard written by a pre-B19 aggregator and loaded by a post-B19 renderer would fail validation on the new required field. | Low | Bench-run shards are not long-lived (B17 R-B17-3 already established this). A user with a stale shard re-runs the aggregator to refresh it. No backward-compat reader is added (matches the B17 precedent). |

## B19.5 Implementation outline

1. **Schema**: add `n_pair_grid: int = Field(ge=0)` to
   `EnsembleLiftRollupRow` in `benchmarks/bootstrap_manifest.py`
   between `n_cells_paired` and `n_skipped_cells`. Update
   the docstring.
2. **Aggregator**: in
   `benchmarks/report/bootstrap_ensemble_lift.py`, compute
   `gbm_pairs & seq_pairs` ONCE per dataset at the
   `aggregate_bootstrap_ensemble_lift_rollup` caller (the
   same site that builds `seed_fold_pairs` and slices
   `gbm_ds` / `seq_ds`), then pass it into
   `_build_dataset_rollup` as an explicit `n_pair_grid`
   kwarg. `_build_dataset_rollup` writes the caller-supplied
   `n_pair_grid` to both the sentinel and happy-path emit
   paths. Sentinel rows get `n_pair_grid=0` (the
   intersection is empty when either family is missing).
3. **Renderer**: in `benchmarks/report/ensemble_lift.py`'s
   `_render_complete_table_with_ci`, replace
   `rollup_row.n_seeds * rollup_row.n_folds` with
   `rollup_row.n_pair_grid`.
4. **Update existing fixtures**: the
   `_make_ensemble_lift_row` factory in
   `test_bootstrap_manifest.py`, the rollup-row construction
   sites in `test_bootstrap_ensemble_lift.py`, and the
   `_make_rollup_row` factory in
   `test_ensemble_lift_report_b16.py` each need a default
   value for `n_pair_grid`. Default to
   `n_cells_paired` so symmetric-roster fixtures match.
5. **NEW tests**: add
   `tests/benchmarks/test_b19_n_pair_grid.py` with the 6
   tests.
6. **Verify**: ruff + pyright clean; 870 tests pass.

## Addressed

R1 swarm: architecture-reviewer (2C / 3I / 2N
REQUEST_CHANGES), qa-test-coverage (0C / 2I / 1N APPROVE),
style-reviewer (0C / 0I / 0N APPROVE). Deduplicated total:
2 CRITICAL, 5 IMPROVEMENT, 3 NITPICK. Closures:

- **arch-R1-C1** (B19.3 omitted the `_make_ensemble_lift_rollup`
  fixture at `test_b17_byte_identity_pins.py:316` that
  constructs an `EnsembleLiftRollupRow` directly; adding a
  REQUIRED `n_pair_grid` field would `ValidationError` on
  import): added item 5 to the "Existing tests touched"
  enumeration prescribing `n_pair_grid=2` to preserve the
  byte-pin's partial-asterisk-firing intent.
- **arch-R1-C2 / arch-R1-I1** (test-count miscount: claimed
  16 B16 aggregator tests when there are 17; claimed
  21 B16 renderer tests in one place and 24 in another):
  reconciled. Verified via `grep -c "^def test_"`: 17
  aggregator tests, 24 renderer tests, 863 baseline. The
  863 to 869 math is correct.
- **arch-R1-I2** (intersection computed inside
  `_build_dataset_rollup` creates a second source of
  truth): rewrote B19.1 to compute `n_pair_grid` at the
  `aggregate_bootstrap_ensemble_lift_rollup` caller (the
  same site that builds `seed_fold_pairs` and slices
  `gbm_ds`/`seq_ds`) and pass it into `_build_dataset_rollup`
  as an explicit kwarg.
- **arch-R1-I3** (`expected > 0` guard semantic not
  documented): B19.2 now states the guard is defensive
  rather than load-bearing because the aggregator's Gate D
  + sentinel-emit path ensures non-sentinel rows always
  have `n_pair_grid >= 1` by construction.
- **arch-R1-N1** (field placement at end vs mid not
  defended): B19.0 now includes a placement note
  documenting the mid-schema decision and the
  no-column-order-asserting-test invariant.
- **arch-R1-N2** (triple-repeat of the "default
  `n_pair_grid` to `n_cells_paired`" rule): the rule is
  stated once in B19.3 item 1 (the `_make_ensemble_lift_row`
  factory paragraph); subsequent items reference it rather
  than restating.
- **qa-R1-I1** (test #1 should add explicit `!=` assertion
  for self-documentation): test #1 spec now includes
  `assert row.n_pair_grid != row.n_seeds * row.n_folds`.
- **qa-R1-I2** (test #5 uses `7` which is not unusual
  enough): changed to `137` matching the B16
  `bootstrap_n_resamples=137` precedent for detectable-
  value sentinels.
- **qa-R1-N1** (test #3 doesn't name the mutation it
  kills): the spec already says "Under the OLD logic
  `expected = 6 > 4` would fire the asterisk; under the
  NEW logic `expected = 4 == 4` does NOT fire", which
  IS the mutation-kill narrative. Accepted as-is.

### R2 swarm closure

R2 confirming swarm: architecture-reviewer (0C / 0I / 1N
APPROVE), qa-test-coverage (0C / 0I / 1N APPROVE),
style-reviewer (0C / 0I / 0N APPROVE). Total: 0 CRITICAL,
0 IMPROVEMENT, 2 NITPICK. Closures:

- **arch-R2-N1** (B19.5 step 2 still said "compute ... and
  write it to `n_pair_grid`" inside `_build_dataset_rollup`,
  contradicting B19.1's caller-side hoist): rewrote step 2
  to say the caller computes once per dataset and passes
  `n_pair_grid` as a kwarg; `_build_dataset_rollup` writes
  the caller-supplied value.
- **qa-R2-N1** (test #6 covers `ge=0` boundary from above
  but not from below): added test #7
  `test_n_pair_grid_negative_rejected_by_field_validator`
  asserting pydantic raises ValidationError on
  `n_pair_grid=-1`. Test count updated to 7 new tests;
  total 863 + 7 = 870 post-refactor.

## Deferred

None at R2.
