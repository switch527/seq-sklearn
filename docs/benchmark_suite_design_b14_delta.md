# B14 design delta: pairwise and training-time CIs

**Scope**: D-B13.1 (B6 pairwise CI integration) + D-B13.2 (B7
training-time CI). Closes the two named B13 deferrals on the
report families that currently ship `mean ± std` only. v1 is
deliberately minimal: ONE CI column per report (the ranking
column), one new rollup shard per report, no metric expansion.

## Requirements

This design is graded against the following requirements
(load-bearing for the review swarm):

- **R1**: B6 (`pairwise.md`) gains a `complementarity_score
  [95% CI]` column for classification cells; the regression
  blocks remain mean-only (no `complementarity_score` is
  defined for regression cells, per B6).
- **R2**: B7 (`training_time.md`) gains a `wall_seconds_mean
  [95% CI]` column. RSS / CUDA-mem columns stay mean-only.
- **R3**: Both rollups REUSE the B13 entity-block bootstrap
  primitive (`benchmarks/metrics/bootstrap.py:entity_block_bootstrap_ci`)
  without modification. The cell-index plays the entity-id
  role; the contract degenerates to row bootstrap over cells.
- **R4**: Both rollups REUSE the B13 freshness machinery:
  `RunManifest.fingerprint()` lands on every rollup row; the
  renderer's stale-rollup fall-back fires on mismatch.
- **R5**: Both rollups REUSE the B13 `RawRollupError` typed
  failure surface and the CLI-wrapper-catches-and-writes-
  sentinel pattern. Each rollup gets its own sentinel file so
  the renderer dispatch stays per-report.
- **R6**: Both rollups opt out via a per-experiment
  `ExperimentSpec` field, matching B13's pattern
  (`bootstrap_rollup_enabled`).
- **R7**: NO new shard layout primitives. The two new
  `bootstrap_*_rollup.parquet` shards reuse the same atomic-
  rename write pattern as the B13 rollup.
- **R8**: NO change to the existing B6 / B7 manifests, drivers,
  or CLI dispatch order. The new aggregators run AFTER the
  existing experiment runners complete (the same place
  `_run_bootstrap_rollup` runs after `run_raw_loss`).

## Background

B13 shipped the B5.4 entity-block bootstrap primitive and wired
CIs onto the B5 raw-loss leaderboard. The same correctness gap
the B11 + B12 Gemini final-passes both flagged (panel-data
intra-entity correlation makes naive row bootstrap report CIs
sqrt(K) tighter than truth) applies to B6 / B7, but the gap
manifests differently:

- **B6 (pairwise)**: each (dataset, model_a, model_b) cell is
  one statistic per (seed, fold). The cells across seeds + folds
  are nominally independent; row bootstrap over cells is the
  right shape. The B13 entity-block primitive degenerates
  cleanly to row bootstrap when each cell is its own "entity".
- **B7 (training-time)**: each (dataset, model, hardware_tier)
  group reports `wall_seconds_mean` across (seed, fold) cells.
  Same shape as B6: row bootstrap over cells.

Neither report has WITHIN-CELL correlation to handle, so the
entity-block primitive's full block-resampling machinery
collapses to plain percentile bootstrap. The primitive's
`metric_fn` parameter is `np.nanmean` for both (the cells already
carry the per-cell metric; the across-cells aggregation is the
mean).

## B14.0 New typed surface declarations

Three new symbols mirror B13.0:

- `benchmarks/bootstrap_manifest.py:PairwiseRollupRow`
- `benchmarks/bootstrap_manifest.py:TrainingTimeRollupRow`
- `benchmarks/bootstrap_manifest.py:pairwise_rollup_path(root)
  -> Path` and `training_time_rollup_path(root) -> Path` helpers

Plus two new per-report sentinel-filename helpers mirroring
`aggregator_failed_sentinel_path`:

- `pairwise_aggregator_failed_sentinel_path(root) -> Path`
  (`{root}/bootstrap_pairwise_aggregator_failed.txt`)
- `training_time_aggregator_failed_sentinel_path(root) -> Path`
  (`{root}/bootstrap_training_time_aggregator_failed.txt`)

Three sentinels (one per report) keep the renderer dispatch
single-source-of-truth per report and let a B6 aggregator
failure NOT mask a B7 success on the same run.

The existing `RawRollupError(RuntimeError)` is REUSED as-is
(both new aggregators raise it). Reusing the type means the
CLI-wrapper exception-catch shape transfers without
modification.

### PairwiseRollupRow schema

```python
class PairwiseRollupRow(BaseModel):
    """One per-(dataset, model_a, model_b, task_type) pairwise CI entry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_name: str
    model_a: str
    model_b: str
    task_type: str  # "binary" | "multiclass" | "regression_point"
    primary_metric: str  # always "complementarity_score" at v1
    n_seeds: int = Field(ge=0)
    n_cells_evaluated: int = Field(ge=0)  # ok cells in the bootstrap
    n_skipped_cells: int = Field(ge=0)
    primary_loss_mean: float | None = None
    primary_loss_ci_lo: float | None = None
    primary_loss_ci_hi: float | None = None
    bootstrap_seed: int
    bootstrap_n_resamples: int = Field(ge=0)
    bootstrap_confidence: float = 0.95
    bootstrap_rng_algorithm: str = "PCG64"
    bootstrap_numpy_version: str
    bootstrap_skipped_reason: str | None = None
    manifest_fingerprint: str
```

Note on the `primary_loss_*` field-name reuse (arch I1
addressed): the field names are intentionally identical to
the B13 RollupRow even though B6 uses them for
`complementarity_score` and B7 uses them for `wall_seconds`.
The reuse pattern:

- `primary_metric: str` is the canonical machine-readable
  label (`"log_loss"` for B5, `"complementarity_score"` for
  B6, `"wall_seconds"` for B7). The renderer reads this label
  to pick the column header text.
- `primary_loss_mean / ci_lo / ci_hi` are the value triplet.
  The "loss" in the field name is a residual from B13; B14
  RollupRows use it for higher-is-better stats (B6
  complementarity) and time-stats (B7 wall_seconds) too. The
  alternative (renaming to `primary_metric_mean / ci_lo /
  ci_hi`) would touch the B13 RollupRow schema (a parquet-
  schema-level rename), which fails the freshness fingerprint
  on every existing B13 rollup. Out of scope for v1; deferred
  to D-B14.6 as a coordinated rename across all three
  schemas.

REGRESSION CELLS: `complementarity_score` is `None` on
regression rows (B6 defines it only for classification). The
aggregator emits a regression-cell rollup as a sentinel row
with `bootstrap_skipped_reason="regression_complementarity_undefined"`
and `primary_loss_mean/ci_lo/ci_hi = None`. The renderer
surfaces the existing "Bootstrap skipped" footnote (the same
machinery the B13 rollup already uses for loader-failed cells).

### TrainingTimeRollupRow schema

```python
class TrainingTimeRollupRow(BaseModel):
    """One per-(dataset, model, hardware_tier, task_type) training-time CI entry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_name: str
    model_name: str
    hardware_tier: str
    task_type: str
    primary_metric: str  # always "wall_seconds" at v1
    n_seeds: int = Field(ge=0)
    n_cells_evaluated: int = Field(ge=0)
    n_skipped_cells: int = Field(ge=0)
    primary_loss_mean: float | None = None
    primary_loss_ci_lo: float | None = None
    primary_loss_ci_hi: float | None = None
    bootstrap_seed: int
    bootstrap_n_resamples: int = Field(ge=0)
    bootstrap_confidence: float = 0.95
    bootstrap_rng_algorithm: str = "PCG64"
    bootstrap_numpy_version: str
    bootstrap_skipped_reason: str | None = None
    manifest_fingerprint: str
```

Same `primary_loss_*` reuse rationale as PairwiseRollupRow.

## B14.1 ExperimentSpec extensions

Two new fields on `benchmarks/config.py:ExperimentSpec`:

```python
bootstrap_pairwise_enabled: bool = True
bootstrap_training_time_enabled: bool = True
```

Semantics:

- `bootstrap_pairwise_enabled` is meaningful only when
  `kind="ensemble"`; ignored otherwise. Default True means the
  pairwise rollup runs after `run_ensemble` succeeds.
- `bootstrap_training_time_enabled` is meaningful only when
  `kind="training_time"`; ignored otherwise. Default True
  means the training-time rollup runs after `run_training_time`
  succeeds.
- The EXISTING `bootstrap_n_resamples: int | None` is SHARED
  per-ExperimentSpec but NOT across kinds (arch N2
  addressed): the field lives on `ExperimentSpec`; a config
  with three specs (kind="raw_loss", kind="ensemble",
  kind="training_time") can set a distinct override on each.
  When unset (the default) each kind falls through to the
  profile dispatch `_BOOTSTRAP_N_RESAMPLES_BY_PROFILE`. The
  shared-across-kinds reading is wrong: each ExperimentSpec
  is independent, and the field belongs to its own spec.

Field-validator preserves the existing B13 invariant that
`bootstrap_n_resamples > 0` when set; no new validator needed.

## B14.2 New aggregator: `benchmarks/report/bootstrap_pairwise.py`

Reads the B6 pairwise manifest (via the existing
`benchmarks.experiments.ensemble.load_pairwise`), groups by
`(dataset_name, model_a, model_b, task_type)`, filters to OK
cells, builds a per-cell `complementarity_score` vector, and
calls the entity-block bootstrap primitive with the cell index
as the entity id.

```python
RAW_PAIRWISE_FAILURE = RawRollupError  # alias for clarity

def aggregate_bootstrap_pairwise_rollup(
    config: BenchmarkConfig,
    *,
    output_root: Path,
    env: RunEnvironment,
    manifest: RunManifest,
) -> list[PairwiseRollupRow]:
    pairwise_df = load_pairwise(output_root)
    if pairwise_df.empty:
        return []
    # ... group by (dataset, A, B, task_type), filter OK cells,
    # extract per-cell complementarity_score for classification,
    # emit sentinel rows for regression, bootstrap with cell-index
    # as entity_id, emit one PairwiseRollupRow per group.
```

Failure modes (raise `RawRollupError`):

- Pairwise manifest absent or empty: the aggregator returns
  `[]` and the wrapper skips the rollup. NOT a failure.
- Malformed cell: a row with `skipped_reason is None` but
  `complementarity_score is None` is malformed and raises
  `RawRollupError`. (The B13 row-count drift check is NOT
  applicable here: the pairwise manifest is per-(seed, fold)
  not per-row, so there's no per-row index range to verify.)

Profile dispatch reuses `_BOOTSTRAP_N_RESAMPLES_BY_PROFILE`
from B13 (the constant is hoisted to a shared location, see
B14.5).

## B14.3 New aggregator: `benchmarks/report/bootstrap_training_time.py`

Reads the B5 manifest (NOT a separate training-time shard:
`run_training_time` is itself a report-only thin pass over
the B5 manifest per `benchmarks/experiments/training_time.py`,
so the data the CI rollup needs is the same `wall_seconds`
column `run_raw_loss` already wrote). Groups by
`(dataset_name, model_name, hardware_tier, task_type)`.

```python
def aggregate_bootstrap_training_time_rollup(
    config: BenchmarkConfig,
    *,
    output_root: Path,
    env: RunEnvironment,
    manifest: RunManifest,
) -> list[TrainingTimeRollupRow]:
    b5_df = load_run(output_root)
    if b5_df.empty:
        return []
    # ... group by (dataset, model, hardware_tier, task_type),
    # filter OK cells, extract per-cell wall_seconds,
    # bootstrap with cell-index as entity_id (each cell is one
    # measurement, mapped to a unique entity id so the primitive
    # resamples cells WITH replacement), emit one
    # TrainingTimeRollupRow per group.
```

Failure modes (raise `RawRollupError`):

- B5 manifest absent or empty: `[]` (matches B13's contract).
  NOT a failure; the wrapper returns silently.
- Classification or regression cell with `wall_seconds is None`
  but `skipped_reason is None`: malformed; raise.
- All cells in a (dataset, model, hardware_tier) group have
  `wall_seconds = 0.0`: not a failure (smoke-tier datasets can
  legitimately finish in <1ms). This is the n-entities-with-
  identical-zero-loss case (each cell is its own entity, n
  entities total, every entity's loss is 0.0). The primitive
  bootstraps all n entities, every resampled mean is 0.0, and
  `ci_lo = ci_hi = mean = 0.0`. This is NOT the single-entity
  degenerate path at `benchmarks/metrics/bootstrap.py:133-136`,
  which fires only when `n_entities == 1`. The two paths yield
  the same observable `(0, 0, 0)` triple but via different code.

## B14.4 Renderer integration

`benchmarks/report/ensemble.py` and `benchmarks/report/training_time.py`
ship the same dispatch shape as `benchmarks/report/raw_loss.py`:

- `render_pairwise_markdown` (existing) keeps its signature.
- NEW `render_pairwise_markdown_with_ci(manifest, rollup, *,
  expected_manifest_fingerprint, aggregator_error_class,
  manifest_unreadable)` adds the CI variant.
- NEW `render_training_time_markdown_with_ci(manifest, rollup,
  *, expected_manifest_fingerprint, aggregator_error_class,
  manifest_unreadable)` adds the CI variant.

Each renderer's CI variant follows the B13 4-source footnote
precedence verbatim:

1. `aggregator_error_class is not None` → std variant +
   "Bootstrap aggregator failed: <class>" footnote.
2. `not rollup` → std variant silently.
3. `expected_manifest_fingerprint set AND any row mismatches`
   → std variant + "rollup is stale" footnote.
4. CI variant body renders; if `manifest_unreadable=True`,
   append "freshness check skipped" footnote.

CI column format: `mean [ci_lo, ci_hi]` with 4 decimal places.
A `bootstrap_skipped_reason` row renders as `(no CI)` plus a
"Bootstrap skipped" footnote group (identical to B13.4's
machinery, the renderer helper is shared, see B14.5).

For the pairwise report, the CI column REPLACES the existing
mean-only `complementarity_score` column header
(`complementarity_score` → `complementarity_score [95% CI]`).
The 6 other pairwise statistics (yule_q, phi, disagreement_rate,
double_fault_rate, pearson_pred_corr, spearman_pred_corr,
pearson_error_corr) stay mean-only.

Partial-fold asterisk (arch I3 addressed): the B13 renderer
appends `*` to the CI cell when `n_cells_evaluated < n_seeds
* n_folds`. The pairwise grouping is by
`(dataset_name, model_a, model_b, task_type)`; the denominator
is `n_seeds * n_folds` derived from the OK rows for that
group (the same shape B13 uses, just with the 4-key group key
substituted for B13's 3-key). The training-time grouping is by
`(dataset_name, model_name, hardware_tier, task_type)`; the
denominator is `n_seeds * n_folds` derived the same way. Both
denominators are computed by the SHARED helper at
`benchmarks/report/_bootstrap_render.py:_folds_per_group` (the
extraction lifts the existing B13 helper).

For the training-time report, the CI column REPLACES the
existing `wall_seconds_mean` + `wall_seconds_std` pair with a
single `wall_seconds [95% CI]` column. The RSS / CUDA-mem
columns stay mean / max-only.

`render_from_dir(output_root)` in each module follows the B13
dispatch verbatim: check failure sentinel → check rollup
presence + freshness → dispatch to CI variant or std.

## B14.5 Shared bootstrap-renderer machinery

The B13 CI cell formatter, the partial-fold asterisk helper,
the skipped-cells footnote renderer, and the
`_BOOTSTRAP_N_RESAMPLES_BY_PROFILE` constant are HOISTED from
`benchmarks/report/raw_loss.py` + `benchmarks/report/bootstrap_rollup.py`
to two new shared modules under the existing
`benchmarks/report/` package (arch I4 addressed: the
constants belong in `report/` because both their producers and
consumers are renderers + aggregators, all under `report/`):

- `benchmarks/report/_bootstrap_render.py` (NEW): the cross-
  report helpers (`_format_ci_cell`, `_render_rollup_skipped_footnote`,
  `_folds_per_group`, the 4-source footnote-precedence dispatch
  as a parameterized helper).
- `benchmarks/report/_bootstrap_aggregate.py` (NEW): the
  `_BOOTSTRAP_N_RESAMPLES_BY_PROFILE` constant + the row-count
  ceiling constant + the manifest-fingerprint-attaching helper.

The `_` prefix marks both as package-internal (consumed by the
three rollup modules in `report/`; not part of the public CLI
surface).

The B13 raw_loss + bootstrap_rollup modules import from the
new shared modules; behavior is unchanged. A byte-string
regression test (qa I1, see B14.7) pins the renderer output
across the extraction.

This hoisting is mechanical (extract method + change import
sites) and is the only change to existing B13 code. The
extraction is justified by R3 (primitive reuse) and R4
(freshness machinery reuse): the alternative is duplicating ~80
lines of footnote/cell-format/profile-dispatch code three times.

## B14.6 CLI dispatch + sentinel writes

`benchmarks/run.py` gains two new wrappers paralleling
`_run_bootstrap_rollup`. Each wrapper attaches to its
DATA-SOURCE dispatch point, not the corresponding renderer's
dispatch point:

- `_run_bootstrap_pairwise_rollup(config, *, env, output_root)`:
  runs AFTER `run_ensemble` succeeds (the pairwise CI rollup's
  data source IS the B6 pairwise manifest that `run_ensemble`
  writes). Catches `RawRollupError`, deletes any partial
  `bootstrap_pairwise_rollup.parquet`, writes
  `bootstrap_pairwise_aggregator_failed.txt` with
  `type(exc).__name__`.
- `_run_bootstrap_training_time_rollup(config, *, env,
  output_root)`: the training-time CI rollup's data source is
  the B5 manifest (`run_training_time` is itself a thin
  report-only pass over the same B5 manifest; no new shard).
  Therefore this wrapper runs at the SAME hook point as
  `_run_bootstrap_rollup` (after `run_raw_loss` succeeds), gated
  by an `ExperimentSpec(kind="training_time",
  bootstrap_training_time_enabled=True)` presence check on the
  config. Decoupling from `run_training_time` means the CI
  rollup runs even when the user runs `--experiment=raw_loss`
  standalone, as long as the config declares the training-time
  spec. Catches `RawRollupError`, deletes any partial
  `bootstrap_training_time_rollup.parquet`, writes
  `bootstrap_training_time_aggregator_failed.txt`.

Each wrapper unlinks its OWN stale sentinel on successful
aggregate (the B13 R2-confirming arch-C1 pattern). The three
sentinels are independent: a B5 failure does not unlink a B6
sentinel; a B6 failure does not write or touch the B7 sentinel
(test pin in B14.7).

## B14.7 Test surface

`tests/benchmarks/test_bootstrap_pairwise.py` (NEW):

- `test_aggregate_bootstrap_pairwise_rollup_classification_cells_emit_ci`:
  build a B6 manifest with 2 seeds x 3 folds x 1 (dataset, A,
  B) pair, all OK, complementarity scores in `[0.5, 0.7]`;
  assert the rollup row has `primary_loss_mean ≈ mean`,
  `ci_lo < mean < ci_hi`, `n_cells_evaluated = 6`.
- `test_aggregate_bootstrap_pairwise_rollup_ci_width_nonzero_with_multiple_cells`
  (qa C1 oracle): build a manifest with at least 4 OK cells
  whose complementarity scores are NOT all identical; assert
  `ci_hi - ci_lo > 0` AND the width is within 3x of
  `np.std(scores) * 2 * 1.96 / np.sqrt(n_cells)`. This rules
  out the silent bug where the aggregator passes
  `entity_ids=np.zeros(n)` (one entity → degenerate ci_lo=ci_hi
  =mean) instead of `entity_ids=np.arange(n)` (n entities →
  proper bootstrap). The `ci_lo < mean < ci_hi` assertion in
  the first test passes under either pattern; this test
  distinguishes them.
- `test_aggregate_bootstrap_pairwise_rollup_regression_cells_emit_sentinel`:
  build a regression-only manifest; assert the rollup emits a
  sentinel row with the EXACT string
  `bootstrap_skipped_reason="regression_complementarity_undefined"`.
- `test_aggregate_bootstrap_pairwise_rollup_mixed_skip_runs_on_ok_subset`:
  mixed manifest with 1 OK cell + 2 skipped; assert the rollup
  bootstraps on the OK cell only (n=1 entity-degenerate case
  per the primitive's contract).
- `test_aggregate_bootstrap_pairwise_rollup_empty_manifest_returns_empty_list`:
  empty pairwise manifest → `[]` (no rollup file written).
- `test_aggregate_bootstrap_pairwise_rollup_malformed_cell_raises`:
  OK row (`skipped_reason is None`) with
  `complementarity_score=None` raises `RawRollupError`. (Was
  previously named "_drift_raises" but the pairwise manifest
  is not shard-joined per row, so there's no row-count drift
  to detect; the failure mode is a malformed cell, not a
  drift.)
- `test_aggregate_bootstrap_pairwise_rollup_records_manifest_fingerprint`:
  the emitted row's `manifest_fingerprint` equals
  `RunManifest.fingerprint()` of the live manifest.

`tests/benchmarks/test_bootstrap_training_time.py` (NEW): same
shape as the pairwise tests, replacing `complementarity_score`
with `wall_seconds`, plus:

- `test_aggregate_bootstrap_training_time_rollup_zero_wall_seconds_produces_zero_width_ci`
  (qa I3 reframe): all cells in a (dataset, model,
  hardware_tier) group have `wall_seconds=0.0`; assert the
  rollup row has `primary_loss_mean = ci_lo = ci_hi = 0.0`.
  Comment in the test docstring: "this is the n-entities
  zero-loss path, NOT the single-entity degenerate path. Each
  cell has its own entity id."
- `test_aggregate_bootstrap_training_time_rollup_single_entity_degenerate`:
  build a 1-cell group (one seed, one fold); assert the
  primitive's single-entity-degenerate path at
  `benchmarks/metrics/bootstrap.py:133-136` fires and the
  rollup row has `primary_loss_mean = ci_lo = ci_hi =
  wall_seconds`. Pinning this code path explicitly distinguishes
  it from the n-entities-zero-loss path above.
- `test_aggregate_bootstrap_training_time_rollup_ci_width_nonzero_with_multiple_cells`
  (qa C1 mirror for B7): same anti-degeneracy assertion as the
  pairwise version.

`tests/benchmarks/test_bootstrap_manifest.py` (extension): add
explicit name-list tests:

- `test_pairwise_rollup_path_format`: `pairwise_rollup_path(root)`
  returns `{root}/bootstrap_pairwise_rollup.parquet`.
- `test_training_time_rollup_path_format`: same for
  `training_time_rollup_path`.
- `test_pairwise_aggregator_failed_sentinel_path_format`: same for
  the pairwise sentinel helper.
- `test_training_time_aggregator_failed_sentinel_path_format`: same.
- `test_write_pairwise_rollup_then_load_round_trips_all_fields`.
- `test_write_training_time_rollup_then_load_round_trips_all_fields`.
- `test_pairwise_rollup_row_extra_forbid_rejects_unknown_field`.
- `test_training_time_rollup_row_extra_forbid_rejects_unknown_field`.
- `test_write_pairwise_rollup_empty_rows_writes_empty_shard`.
- `test_write_training_time_rollup_empty_rows_writes_empty_shard`.

`tests/benchmarks/test_ensemble_report.py` (NEW or extension
to the existing test_ensemble file):

- `test_render_pairwise_with_ci_renders_mean_and_interval`
- `test_render_pairwise_without_ci_falls_back_to_scalar`
- `test_render_pairwise_with_ci_drops_old_score_header_when_ci_present`
  (qa N1): asserts the bare `complementarity_score` header is
  ABSENT and `complementarity_score [95% CI]` is present when
  the CI variant is active.
- `test_render_pairwise_with_ci_falls_back_on_manifest_fingerprint_mismatch`
- `test_render_pairwise_with_ci_surfaces_aggregator_failed_footnote_on_wrapper_caught`
- `test_render_pairwise_with_ci_regression_sentinel_produces_bootstrap_skipped_footnote`
  (qa C2): construct a `PairwiseRollupRow` with
  `bootstrap_skipped_reason="regression_complementarity_undefined"`
  and pass to the renderer; assert the output contains BOTH
  "Bootstrap skipped" AND the exact reason string. Cross-pins
  the aggregator's emitted string against the renderer's
  consumption.
- `test_render_pairwise_with_ci_surfaces_rollup_skipped_in_separate_footnote`
- `test_render_from_dir_passes_freshness_fingerprint_at_cli_seam`
- `test_render_from_dir_falls_back_silently_when_rollup_file_exists_but_is_empty`
- `test_render_from_dir_renders_ci_without_freshness_footnote_when_manifest_absent`
- `test_render_from_dir_freshness_check_skipped_footnote_on_corrupt_manifest`
- `test_render_from_dir_surfaces_aggregator_failed_sentinel`

`tests/benchmarks/test_training_time_report.py` (extension):
mirror of the pairwise renderer tests for the training-time
renderer, replacing `complementarity_score` with
`wall_seconds`. Additionally:

- `test_render_training_time_with_ci_drops_old_mean_and_std_headers_when_ci_present`
  (qa N2): asserts BOTH `wall_seconds_mean` AND
  `wall_seconds_std` headers are ABSENT and
  `wall_seconds [95% CI]` is present when the CI variant is
  active. Pins the two-column-to-one replacement.

`tests/benchmarks/test_run_bootstrap_pairwise_wrapper.py` and
`tests/benchmarks/test_run_bootstrap_training_time_wrapper.py`
(NEW): named tests for each wrapper:

- `test_run_bootstrap_pairwise_rollup_writes_shard_on_happy_path`
- `test_run_bootstrap_pairwise_rollup_skips_via_opt_out`
- `test_run_bootstrap_pairwise_rollup_skips_when_pairwise_manifest_absent`
- `test_run_bootstrap_pairwise_rollup_skips_when_run_manifest_absent`
- `test_run_bootstrap_pairwise_rollup_skips_when_run_manifest_load_fails`
- `test_run_bootstrap_pairwise_rollup_catches_raw_rollup_error_and_continues`
- `test_run_bootstrap_pairwise_rollup_writes_failure_sentinel_on_raw_rollup_error`
- `test_run_bootstrap_pairwise_rollup_unlinks_stale_failure_sentinel_on_success`
- `test_run_bootstrap_pairwise_rollup_failure_does_not_touch_b5_or_b7_sentinels`
  (qa C3 isolation pin): monkeypatch the pairwise aggregator
  to raise `RawRollupError`; call the wrapper; assert
  `bootstrap_pairwise_aggregator_failed.txt` EXISTS AND
  `bootstrap_aggregator_failed.txt` (B5) AND
  `bootstrap_training_time_aggregator_failed.txt` (B7) both
  DO NOT exist.

For `tests/benchmarks/test_run_bootstrap_training_time_wrapper.py`,
named tests mirror the pairwise list with the qa-C3 isolation
test asserting symmetry in the other direction:

- `test_run_bootstrap_training_time_rollup_writes_shard_on_happy_path`
- `test_run_bootstrap_training_time_rollup_skips_via_opt_out`
- `test_run_bootstrap_training_time_rollup_skips_when_b5_manifest_absent`
- `test_run_bootstrap_training_time_rollup_skips_when_run_manifest_absent`
- `test_run_bootstrap_training_time_rollup_skips_when_run_manifest_load_fails`
- `test_run_bootstrap_training_time_rollup_catches_raw_rollup_error_and_continues`
- `test_run_bootstrap_training_time_rollup_writes_failure_sentinel_on_raw_rollup_error`
- `test_run_bootstrap_training_time_rollup_unlinks_stale_failure_sentinel_on_success`
- `test_run_bootstrap_training_time_rollup_failure_does_not_touch_b5_or_b6_sentinels`
  (qa-C3 B7 direction): monkeypatch the training-time
  aggregator to raise `RawRollupError`; call the wrapper;
  assert `bootstrap_training_time_aggregator_failed.txt`
  EXISTS AND both `bootstrap_aggregator_failed.txt` (B5) AND
  `bootstrap_pairwise_aggregator_failed.txt` (B6) DO NOT
  exist.

`tests/benchmarks/test_bootstrap_render_regression.py` (NEW,
qa I1 pin): asserts that after the `_bootstrap_render.py`
extraction, `render_leaderboard_markdown_with_ci` produces the
SAME output as before. Implementation: a fixture parquet rollup
+ a fixture manifest are passed to the renderer; the expected
Markdown output is stored as a multi-line string in the test
file (pre-extraction byte-for-byte). The test asserts equality.
If the extraction introduces ANY formatting drift, this test
fails immediately.

## B14.8 Migration / backward compatibility

- The two new shards are NEW filenames; absence is harmless.
- The two new sentinels are NEW filenames; absence is harmless.
- The two new ExperimentSpec fields default to True; existing
  configs get the CI variant automatically.
- The two new pydantic models are NEW classes; no schema
  changes to existing rows.
- The B13 RollupRow + raw_loss CI variant are unchanged.

## B14.9 Estimated effort

| Module | Estimated size |
|---|---|
| `benchmarks/bootstrap_manifest.py` (+2 schemas + 4 path helpers) | Small |
| `benchmarks/config.py` (+2 ExperimentSpec fields) | Trivial |
| `benchmarks/report/bootstrap_pairwise.py` (NEW) | ~300 lines |
| `benchmarks/report/bootstrap_training_time.py` (NEW) | ~250 lines |
| `benchmarks/report/_bootstrap_render.py` (NEW shared helpers) | ~150 lines |
| `benchmarks/report/_bootstrap_aggregate.py` (NEW shared constants) | ~50 lines |
| `benchmarks/report/ensemble.py` (+CI variant renderer + dispatch) | ~200 lines |
| `benchmarks/report/training_time.py` (+CI variant renderer + dispatch) | ~200 lines |
| `benchmarks/run.py` (+2 wrappers) | ~80 lines |
| `tests/benchmarks/test_bootstrap_pairwise.py` (NEW) | ~250 lines |
| `tests/benchmarks/test_bootstrap_training_time.py` (NEW) | ~200 lines |
| `tests/benchmarks/test_bootstrap_manifest.py` (extension) | ~80 lines |
| `tests/benchmarks/test_ensemble_report.py` (extension) | ~350 lines |
| `tests/benchmarks/test_training_time_report.py` (extension) | ~300 lines |
| `tests/benchmarks/test_run_bootstrap_pairwise_wrapper.py` (NEW) | ~200 lines |
| `tests/benchmarks/test_run_bootstrap_training_time_wrapper.py` (NEW) | ~200 lines |
| `docs/benchmark_suite_implementation_plan.md` (B14 actual-shape) | ~50 lines |

Total: ~2,860 lines. Smaller than B13 (which shipped ~3,500
lines with the primitive) because the primitive is reused
verbatim.

## Risk register (delta)

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R-B14-1 | Three sentinels triple the FS-contract surface; a future maintainer renames one and the other two drift. | Medium | All three live in `bootstrap_manifest.py` with the `_*_FAILED_SENTINEL_FILENAME` private constants exported via `_path()` helpers. Single-source-of-truth per sentinel; test pins each one. |
| R-B14-2 | The cell-as-entity contract collapses to row bootstrap; a reviewer flags "you didn't actually need the entity-block primitive". | Low | Doc note in B14.0 explains the reuse is for CODE consolidation, not statistical novelty. The primitive's single-entity-degenerate path covers the edge case. |
| R-B14-3 | The `_bootstrap_render.py` extraction touches B13 working code. A regression breaks the B5 CI variant. | High | The extraction is mechanical (extract method + change import sites). Existing B13 tests continue to pass unchanged. R1 swarm round will pin this with a regression check. |
| R-B14-4 | Three rollup aggregators run sequentially; a slow B6 aggregator adds latency to the run that didn't ask for B7 CIs. | Low | The three are gated by independent `bootstrap_*_enabled` flags + per-experiment `kind` checks. A smoke run can disable all three. |
| R-B14-5 | The B13 `_run_bootstrap_rollup` wrapper's exception-handling shape is duplicated three times. | Medium | The wrapper bodies are ~30 lines each with the only difference being the aggregator + sentinel-path symbols. A B14-followup may extract a wrapper-factory; v1 ships the three explicit wrappers because the symbol substitution would obscure the catch-and-log shape. |

## B14-followup deferrals (out of scope for v1)

- D-B14.1: CIs on the SECONDARY pairwise statistics (yule_q,
  phi, disagreement_rate, etc.). v1 ships CI on the
  ranking column only.
- D-B14.2: CIs on RSS / CUDA-mem in the training-time report.
  v1 ships CI on the ranking column only.
- D-B14.3: extract a shared CLI-wrapper factory for the three
  `_run_bootstrap_*_rollup` wrappers (R-B14-5).
- D-B14.4: BCa CI (inherited from D-B13.5).
- D-B14.5: per-fold CI (inherited from D-B13.6).
- D-B14.6: coordinated rename of `primary_loss_*` fields
  across all three RollupRow schemas (B13 + 2 new) to
  `primary_metric_*`. Out of scope at v1 because the rename
  is a parquet-schema-level change that fails the freshness
  fingerprint on every existing B13 rollup. Landed as a
  followup when an existing-rollup migration is also needed.

## Addressed

R1 swarm: architecture-reviewer (3C/4I/2N REQUEST_CHANGES),
qa-test-coverage (3C/3I/2N REQUEST_CHANGES), style-reviewer
(1C/0I/0N REQUEST_CHANGES). Total deduped: 7 CRITICAL + 7
IMPROVEMENT + 4 NITPICK. All CRITICALs addressed:

- **style-C1** (em dash in H1 title): title rewritten to
  `# B14 design delta: pairwise and training-time CIs`
  matching the B12/B13 convention.
- **arch-C1 + arch-C3** (training-time wrapper coupled to
  the wrong driver / D-B13.2 reversal): B14.3 + B14.6
  rewritten. `run_training_time` is itself a thin report-only
  pass over the B5 manifest; the CI rollup's data source is
  the B5 manifest, not a separate training-time shard. The
  wrapper now attaches AT THE SAME HOOK POINT as
  `_run_bootstrap_rollup` (after `run_raw_loss` succeeds),
  gated by an `ExperimentSpec(kind="training_time",
  bootstrap_training_time_enabled=True)` presence check on
  the config. The wrapper is decoupled from `run_training_time`
  so it runs even when the user invokes
  `--experiment=raw_loss` standalone with a training-time
  spec in the config.
- **arch-C2** (single-entity-degenerate misclaim against
  `benchmarks/metrics/bootstrap.py:133-136`): B14.3 rewritten
  to correctly describe the all-zero-wall_seconds case as the
  n-entities-zero-loss path (each cell is its own entity;
  every resampled mean is 0.0; the observable triple `(0, 0,
  0)` matches the single-entity-degenerate output but via a
  different code path). The test surface (B14.7) now names
  TWO distinct tests: `_zero_wall_seconds_produces_zero_width_ci`
  (n-entities path) and `_single_entity_degenerate` (the
  primitive's actual `n_entities==1` branch).
- **qa-C1** (cell-as-entity bootstrap-degeneracy oracle
  missing): B14.7 adds
  `test_aggregate_bootstrap_pairwise_rollup_ci_width_nonzero_with_multiple_cells`
  (and the B7 mirror). The assertion is `ci_hi - ci_lo > 0`
  PLUS the width is within 3x of `std * 2 * 1.96 / sqrt(n)`,
  killing the silent degeneracy bug where the aggregator
  passes `entity_ids=np.zeros(n)` instead of `np.arange(n)`.
- **qa-C2** (regression-cell sentinel string not cross-
  verified with renderer): B14.7 adds
  `test_render_pairwise_with_ci_regression_sentinel_produces_bootstrap_skipped_footnote`
  which constructs a `PairwiseRollupRow` with the exact
  sentinel string and asserts the renderer's output contains
  both "Bootstrap skipped" AND the exact reason text. Cross-
  pins the aggregator's emitted string against the renderer's
  consumption.
- **qa-C3** (cross-report sentinel isolation untested): B14.7
  adds
  `test_run_bootstrap_pairwise_rollup_failure_does_not_touch_b5_or_b7_sentinels`
  and the B7 mirror. Monkeypatches the named aggregator to
  raise `RawRollupError`, calls the wrapper, asserts the
  per-report sentinel EXISTS while the OTHER TWO sentinels
  do NOT.

All IMPROVEMENTs either addressed or deferred:

- **arch-I1** (primary_loss_* rename): rationale added inline
  to B14.0 PairwiseRollupRow schema declaration; the alternative
  (a parquet-schema rename) is named D-B14.6 in the
  deferral list with the freshness-fingerprint constraint.
- **arch-I2** (dead row-count-drift check): the test name
  changed from `_drift_raises` to `_malformed_cell_raises` and
  the design clarifies that pairwise data is not shard-joined
  per row, so the failure mode is a malformed cell, not row
  drift.
- **arch-I3** (undefined partial-fold denominator for the new
  4-key groups): B14.4 now spells out the `n_seeds * n_folds`
  denominator computation for both 4-key groupings and pins
  it to the shared `_folds_per_group` helper.
- **arch-I4** (constants module placement): B14.5 now states
  the two extracted modules live under `benchmarks/report/`
  with `_` prefix marking them package-internal.
- **qa-I1** (extraction regression pin): B14.7 adds
  `tests/benchmarks/test_bootstrap_render_regression.py` with
  byte-string fixture assertion across the extraction.
- **qa-I2** (manifest schema extension tests intent-only):
  B14.7 now names 10 explicit tests for the new schemas
  (path-format x4, round-trip x2, extra-forbid x2, empty-shard
  x2).
- **qa-I3** (zero-wall_seconds test structural confusion):
  test renamed to
  `_zero_wall_seconds_produces_zero_width_ci`; a SEPARATE
  `_single_entity_degenerate` test pins the actual
  single-entity primitive path.

All NITPICKs addressed:

- **arch-N1** (enumeration counts): the B14.7 test surface now
  enumerates each named test explicitly so the count is
  countable.
- **arch-N2** (bootstrap_n_resamples sharing semantics): B14.1
  clarifies the field is per-ExperimentSpec, not shared
  across kinds.
- **qa-N1** (pairwise renderer missing
  `drops_std_column_when_ci_present` test name): added as
  `_drops_old_score_header_when_ci_present`.
- **qa-N2** (training-time renderer two-column-to-one
  replacement unnamed): added as
  `_drops_old_mean_and_std_headers_when_ci_present`.

### R2 swarm closure

R2 confirming swarm: architecture-reviewer (0C/0I/1N APPROVE),
qa-test-coverage (0C/1I/1N APPROVE), style-reviewer
(0C/0I/0N APPROVE). Total: 0 CRITICAL, 1 IMPROVEMENT, 2
NITPICK. Closures:

- **qa-R2-IMP** (B7 wrapper test names): the training-time
  wrapper tests are now enumerated by exact name to match the
  pairwise side. The qa-C3 B7-direction test is named
  `_failure_does_not_touch_b5_or_b6_sentinels`.
- **arch-R2-NIT** (leftover "Row-count drift" prose): the B14.2
  failure-modes bullet rewritten to "Malformed cell" framing;
  the row-count-drift framing dropped entirely.
- **qa-R2-NIT** (regression-sentinel renderer test uses
  substring `in` check rather than footnote-structure
  equality): deferred. Rationale: matches the B13 precedent
  for footnote tests; a tighter assertion is a B14-followup.

## Deferred

- **qa-R2-NIT**: regression-sentinel renderer test uses
  substring `in` check rather than footnote-structure
  equality. Matches B13 precedent for footnote tests;
  tightening to a line-by-line footnote-structure equality
  check is a B14-followup that requires extracting a footnote-
  body extractor helper. Out of scope for v1.
