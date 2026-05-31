# B35 delta: oracle per-fold CIs (D-B22.2)

## Requirements

R-B35-1 (closes D-B22.2): add `per_fold_oracle_cis: list[FoldCI] | None`
field to `EnsembleLiftRollupRow`. Populated by a NEW third
bootstrap invocation in `aggregate_bootstrap_ensemble_lift_rollup`
that runs `compute_per_fold_cis` on the oracle delta path
when `ExperimentSpec.bootstrap_per_fold_cis_enabled=True`
AND `n_oracle_cells_paired > 0`. Uses an independent
`BOOTSTRAP_ORACLE_PER_FOLD_SEED_OFFSET` so the PCG64 stream
is independent from both the main per-fold and pooled
oracle bootstraps.

## Non-requirements

- v1 does NOT add a renderer surface for the new field.
  Parquet-audit-only at v1; renderer integration deferred
  as D-B35.1 (could extend the existing per-fold footnote
  with a "scope" column for main vs oracle, or add a
  sibling section).
- v1 does NOT change the main per-fold field
  (`per_fold_cis` continues to carry main delta only).
- v1 does NOT change the existing oracle bootstrap output;
  only adds a new field.

## B35.0 Background

B22 added the per-fold CI primitive
(`_bootstrap_aggregate.compute_per_fold_cis`) and surfaced
it via `per_fold_cis: list[FoldCI] | None` on all 5
RollupRow schemas. Each aggregator invokes the primitive
once on the MAIN delta when the per-fold flag is enabled.
B22's D-B22.2 explicitly deferred the parallel oracle
invocation for `EnsembleLiftRollupRow`.

## B35.1 BOOTSTRAP_ORACLE_PER_FOLD_SEED_OFFSET constant

In `benchmarks/report/_bootstrap_aggregate.py`:

```python
# B35 / D-B22.2: oracle per-fold seed offset. Independent
# from BOOTSTRAP_ORACLE_SEED_OFFSET (pooled oracle) and
# BOOTSTRAP_PER_FOLD_SEED_OFFSET (main per-fold) so the
# three oracle/per-fold/oracle-per-fold streams stay
# uncorrelated.
BOOTSTRAP_ORACLE_PER_FOLD_SEED_OFFSET: int = 0xB35_07A_FCD
```

Added to `__all__` alongside the existing offsets.

## B35.2 Schema field on EnsembleLiftRollupRow

In `benchmarks/bootstrap_manifest.py`:

```python
# B35 / D-B22.2: per-fold oracle CIs (opt-in via same
# bootstrap_per_fold_cis_enabled flag). Parallel to
# per_fold_cis but for the oracle delta path. None when
# the flag is off OR n_oracle_cells_paired == 0; a
# non-None list carries one FoldCI per fold with
# computable oracle data.
per_fold_oracle_cis: list[FoldCI] | None = None
```

Inserted near the existing `per_fold_cis` field on
`EnsembleLiftRollupRow`. The B23
`_validate_row_count_invariants` is not touched.

## B35.3 Aggregator population

In `benchmarks/report/bootstrap_ensemble_lift.py`,
inside the existing `n_oracle_cells_paired > 0` branch
after the pooled oracle bootstrap returns:

```python
per_fold_oracle_cis = None
if per_fold_enabled:
    # Map each oracle cell to its fold + seed for the
    # per-fold computation. Order matches oracle_deltas /
    # oracle_entity_ids.
    oracle_cells = [c for c in computed.cells if c.oracle_loss is not None]
    per_fold_oracle_cis = _bootstrap_aggregate.compute_per_fold_cis(
        losses=oracle_deltas,
        entities=oracle_entity_ids,
        fold_indices=np.array([c.fold_index for c in oracle_cells], dtype=np.int64),
        seeds=np.array([c.seed for c in oracle_cells], dtype=np.int64),
        n_resamples=n_resamples,
        confidence=BOOTSTRAP_CONFIDENCE,
        base_seed=BOOTSTRAP_DEFAULT_SEED,
        fold_seed_offset=_bootstrap_aggregate.BOOTSTRAP_ORACLE_PER_FOLD_SEED_OFFSET,
        ci_method=_bootstrap_aggregate.BOOTSTRAP_DEFAULT_CI_METHOD,
        metric_fn=lambda x: float(np.nanmean(x)),
        dataset_name=dataset_name,
        kind="ensemble_lift_oracle",
    )
```

Pass `per_fold_oracle_cis` into the `EnsembleLiftRollupRow(...)`
constructor call alongside the existing `per_fold_cis`.

When `n_oracle_cells_paired == 0` the field stays `None`
(parallel to oracle_metric_* fields).

## B35.4 Tests

Baseline (post-B34): 1090.

### B35.4.1 Schema tests

1. `test_ensemble_lift_rollup_row_per_fold_oracle_cis_default_is_none`:
   construct EnsembleLiftRollupRow without supplying
   `per_fold_oracle_cis`; assert the field reads back None.
2. `test_ensemble_lift_rollup_row_accepts_per_fold_oracle_cis_list`:
   construct with a populated list; assert round-trip.
3. `test_ensemble_lift_rollup_row_per_fold_oracle_cis_round_trips_through_parquet`:
   write + load the row; assert the list survives parquet
   serialization.

### B35.4.2 Aggregator tests

4. `test_aggregator_populates_per_fold_oracle_cis_when_enabled_and_oracle_cells_present`:
   spec with `bootstrap_per_fold_cis_enabled=True`, fixture
   producing oracle cells across multiple folds; assert the
   emitted row carries a non-None `per_fold_oracle_cis` with
   one FoldCI per fold.
5. `test_aggregator_per_fold_oracle_cis_is_none_when_flag_disabled`:
   spec with `bootstrap_per_fold_cis_enabled=False`; assert
   field is None.
6. `test_aggregator_per_fold_oracle_cis_is_none_when_n_oracle_cells_paired_is_zero`:
   spec with flag enabled but no cells carry oracle_loss;
   assert field is None.
7. `test_aggregator_per_fold_oracle_cis_uses_independent_seed_offset`:
   compare main per-fold CIs and oracle per-fold CIs on a
   fixture where the data shapes happen to align; assert
   the bootstrap results differ (proving independent PCG64
   streams).

### B35.4.3 Expected test delta

Baseline: 1090.
- 7 new B35 tests.
- Existing tests unchanged.
- Total: 1090 + 7 = 1097.

## B35.5 Risks

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R-B35-Risk-1 | Existing EnsembleLift fixture sites construct rows without specifying `per_fold_oracle_cis`. | Low | Schema default is `None`; existing fixtures pass without change. |
| R-B35-Risk-2 | The seed offset collides with an existing offset. | Low | `0xB35_07A_FCD` chosen to be distinct from `0xB20_07A_C7E` and `0xB22_F01D_C1`. |
| R-B35-Risk-3 | Parquet round-trip might silently drop the new nested list. | Low | The existing `per_fold_cis` field uses the same pattern; same write/load path handles both. |

## Deferred

- **D-B35.1**: surface per-fold oracle CIs in the renderer.
  v1 ships the field parquet-audit-only; renderer
  integration would extend the existing
  `render_per_fold_cis_footnote` with a "scope" column
  (main vs oracle) or add a sibling section. Defer until
  a consumer asks.
