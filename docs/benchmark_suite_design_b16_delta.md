# B16 design delta: ensemble-lift per-dataset Δ paired CI (D-B13.4)

**Scope**: D-B13.4 closes the fourth and final B13 deferral.
Adds a paired CI on the per-(dataset) Δloss in the B11
ensemble-lift report. Reuses the B13/B14/B15 entity-block
bootstrap primitive via the cell-as-entity contract; per-cell
deltas are recomputed from the B5 manifest + predictions
shards (no new shard layout). The Wilcoxon block stays
unchanged.

## Requirements

This design is graded against the following requirements:

- **R1**: `ensemble_lift.md` gains a `Δloss [95% CI]` column
  on every per-dataset row. Sentinel rows
  (`no_gbm_predictions`, `no_seq_predictions`,
  `all_cells_skipped_in_manifest`) keep the existing
  "Incomplete groups" footnote treatment for the first two
  (B11 surface) and route the third via the shared helper;
  no CI for any of them.
- **R2**: The CI is PAIRED on the per-cell (GBM-only,
  GBM+seq) losses indexed by (seed, fold). The bootstrap
  resamples paired cells with replacement; each resample's
  mean is the Δloss for that resample.
- **R3**: REUSE the B13 entity-block bootstrap primitive
  (`benchmarks/metrics/bootstrap.py:entity_block_bootstrap_ci`)
  via the cell-as-entity contract.
- **R4**: REUSE the B13/B14/B15 freshness machinery
  (`RunManifest.fingerprint()` on every emitted row; stale-
  rollup fallback in the renderer).
- **R5**: REUSE the B13/B14/B15 `RawRollupError` typed
  failure + CLI-wrapper sentinel-write pattern. The B11
  wrapper gets its own sentinel independent from
  B5/B6/B7/B8 sentinels.
- **R6**: Opt out via
  `ExperimentSpec.bootstrap_ensemble_lift_enabled: bool = True`.
  Matches the B14/B15 patterns.
- **R7**: NO new shard layout primitives. The new
  `bootstrap_ensemble_lift_rollup.parquet` reuses the atomic-
  rename write pattern.
- **R8**: Minimal disruption to existing B11 driver. A small
  refactor extracts the per-cell delta computation from
  `_per_dataset_lift` into a shared helper that both the
  existing driver AND the new aggregator consume. The
  driver's external contract (the `EnsembleLiftExperimentResult`
  return shape) is unchanged.

## Background

B14 and B15 wired CIs onto B6 / B7 / B8 leveraging the
existing per-(seed, fold) cell granularity already in the B5
manifest (or B8's tuned-variant rows in the same manifest).
B11 ensemble-lift is structurally different: per-cell ensemble
losses do NOT live in the B5 manifest. They are computed
in-memory by `_per_dataset_lift` via `_join_predictions` +
`_per_row_ensemble_loss` and then collapsed to a `PerDatasetLift`
summary before persistence.

For B16 to bootstrap per-cell Δs, it must either:
- (a) Recompute per-cell Δs from scratch in the aggregator
  (full duplication of `_per_dataset_lift`'s body), OR
- (b) Persist per-cell Δs during `run_ensemble_lift` so the
  aggregator just reads them (new shard layout; violates R7), OR
- (c) Extract per-cell delta computation into a shared helper
  consumed by BOTH the existing driver and the new aggregator
  (small B11 refactor, no new shard), OR
- (d) Add a `return_per_cell: bool = False` kwarg to
  `_per_dataset_lift` that flips its return shape to
  `(PerDatasetLift, list[PerCellLiftDelta])`. No helper
  extraction; the aggregation arithmetic stays in one place;
  R-B16-1 byte-pin not needed.

Choice (c) preserves R7 and R8 with the smallest blast
radius for the AGGREGATOR side (the new code reads a clean
helper return rather than parsing a tuple) at the cost of a
small R-B16-1 byte-pin to lock the equivalence. Choice (d)
avoids the regression pin but introduces a boolean-flagged
dual return; the aggregator-side code is uglier. The trade
is "regression pin on the std view (c)" vs "boolean kwarg on
the existing function (d)". Choosing (c) here: the helper
extraction is cleaner at the call site and the byte-pin is
~80 lines of test code that documents the equivalence
explicitly.

## B16.0 New typed surface declarations

- `benchmarks/bootstrap_manifest.py:EnsembleLiftRollupRow`
  (new pydantic schema)
- `benchmarks/bootstrap_manifest.py:ensemble_lift_rollup_path(root)`
  helper for `{root}/bootstrap_ensemble_lift_rollup.parquet`
- `benchmarks/bootstrap_manifest.py:ensemble_lift_aggregator_failed_sentinel_path(root)`
  helper for
  `{root}/bootstrap_ensemble_lift_aggregator_failed.txt`
- `benchmarks/report/bootstrap_ensemble_lift.py:is_ensemble_lift_rollup_enabled(config)`
  predicate (mirrors the B15 predicate)
- `benchmarks/report/bootstrap_ensemble_lift.py:ensemble_lift_rollup_output_path()`
  stable filename token
- `write_ensemble_lift_rollup` + `load_ensemble_lift_rollup`
  I/O helpers using the existing shared `_write_rows_atomic`
  helper

The B13 `RawRollupError` is REUSED.

### Sentinel string enumeration

The aggregator emits THREE sentinel reason strings (closes
arch-C1: the prior draft included `no_paired_cells` as a
fourth string but B11 does NOT have that distinct semantic.
Per `benchmarks/experiments/ensemble_lift.py:478-489,556-564`
the driver iterates `seed_fold_pairs` (the UNION of seeds and
folds across the dataset's manifest) and at each pair sets
`seen_no_gbm=True` if GBM is missing OR `seen_no_seq=True` if
seq is missing, then `continue`s. When BOTH families have
rows but never co-occur on (seed, fold), every pair short-
circuits one of the two flags and the final result has the
relevant flag(s) set. The intersection-empty case is
indistinguishable from "neither family ever appeared on the
correct (seed, fold)" at the renderer surface, so B16
matches B11's behavior exactly):

1. `"no_gbm_predictions"`: the baseline (GBM) family
   contributed zero paired cells (matches B11
   `PerDatasetLift.no_gbm_predictions=True`).
2. `"no_seq_predictions"`: the seq family contributed zero
   paired cells (matches B11
   `PerDatasetLift.no_seq_predictions=True`).
3. `"all_cells_skipped_in_manifest"`: defense-in-depth
   fallback when every row in the dataset's manifest block
   carries a non-None `skipped_reason`. Inherited from the
   B14/B15 implementation precedent
   (`benchmarks/report/bootstrap_hpo_uplift.py:182`).

A "both families have rows globally but their (seed, fold)
intersection is empty" case routes through the existing pair
of B11 flags (both `no_gbm_predictions` AND `no_seq_predictions`
end up True, since neither family contributed a paired cell).
This is the same path B11's renderer at
`benchmarks/report/ensemble_lift.py:106-117` already
handles. The B16 rollup row mirrors the flag pair via the
sentinel string: it emits `"no_gbm_predictions"` (lexically
first of the two flags) and surfaces both flags in the
renderer's incomplete-block via the existing B11 path. No
new sentinel string is needed.

### EnsembleLiftRollupRow schema

```python
class EnsembleLiftRollupRow(BaseModel):
    """One per-dataset ensemble-lift Δloss CI entry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_name: str
    task_type: str
    primary_metric: str  # always "delta_loss" at v1
    primary_loss_column: str  # "log_loss" | "rmse"
    # seq_family + baseline_family are NOT on the schema (arch-I2
    # closure): they are constants per `run_ensemble_lift`
    # (`_DEFAULT_SEQ_FAMILY` / `_DEFAULT_BASELINE_FAMILY`) and
    # the renderer always has them via `EnsembleLiftExperimentResult.
    # seq_family` / `.baseline_family`. Storing them N times on
    # every row is redundant. A future multi-pair driver would add
    # them as a schema migration.
    n_seeds: int = Field(ge=0)
    n_folds: int = Field(ge=0)
    n_cells_paired: int = Field(ge=0)
    n_skipped_cells: int = Field(ge=0)
    primary_loss_mean: float | None = None  # Δloss = loss(GBM) - loss(GBM+seq)
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

The `primary_loss_column` field carries the underlying loss
column for audit (B11 / B15 pattern). `seq_family` and
`baseline_family` are NOT on the schema (closes arch-R2-C1
stale-prose contradiction): the renderer reads them from
`EnsembleLiftExperimentResult.seq_family` /
`.baseline_family` directly. A future multi-pair lift driver
would add them as a schema migration.

NOT INCLUDED in this rollup: `oracle_delta_loss_mean`. The
oracle Δ is a per-sample-best ceiling, not a bootstrap
statistic; v1 keeps it as the existing scalar with no CI
(D-B16.1 follow-up).

## B16.1 ExperimentSpec extension

One new field on `benchmarks/config.py:ExperimentSpec`:

```python
bootstrap_ensemble_lift_enabled: bool = True
```

Semantics:
- Meaningful only when `kind="ensemble_lift"`; ignored on
  other kinds.
- Default True so the rollup runs by default.

Note (arch-I5 closure): B11 currently consumes NO fields on
`ExperimentSpec(kind="ensemble_lift")`; the spec presence is
the only signal `_assert_ensemble_lift_configured` reads at
`benchmarks/experiments/ensemble_lift.py:167-181`. This is
the FIRST field on the B11 spec. The
`is_ensemble_lift_rollup_enabled(config)` predicate reads
"any spec with `kind="ensemble_lift"` AND
`bootstrap_ensemble_lift_enabled=True`", mirroring the B15
implementation at
`benchmarks/report/bootstrap_hpo_uplift.py:80-86`.

## B16.2 New aggregator: `benchmarks/report/bootstrap_ensemble_lift.py`

Reads the B5 manifest + predictions shards via the existing
B11 helpers (`_build_cells_table`, the inner-join machinery).
For each dataset:

1. Get the GBM and seq cells via the existing
   `benchmarks.experiments.ensemble_lift._build_cells_table`.
2. Sentinel-route per the three enumerated cases above.
3. For dataset groups that pass the sentinel checks, call the
   NEW shared helper
   `benchmarks.experiments.ensemble_lift.compute_per_cell_lift_deltas(
       dataset_name, task_type, seed_fold_pairs, gbm_cells,
       seq_cells, output_root
   ) -> list[PerCellLiftDelta]`
   (extracted from `_per_dataset_lift` in the B11 refactor;
   returns a list of `(seed, fold, delta_loss, oracle_delta_loss)`
   records).
4. Drop cells where `delta_loss is None` (these are the same
   cells that would have produced a sentinel in B11).
5. Bootstrap the per-cell `delta_loss` array with cell-index
   as entity-id (mirrors B15).
6. Emit one `EnsembleLiftRollupRow` per dataset.

```python
def aggregate_bootstrap_ensemble_lift_rollup(
    config: BenchmarkConfig,
    *,
    output_root: Path,
    env: RunEnvironment,
    manifest: RunManifest,
) -> list[EnsembleLiftRollupRow]:
    df = load_run(output_root)
    if df.empty:
        return []
    # ... per-dataset group iteration:
    # _build_cells_table -> gbm_cells, seq_cells per dataset
    # sentinel-route empty cases
    # compute_per_cell_lift_deltas -> per-cell deltas
    # bootstrap + emit row
```

Failure modes (raise `RawRollupError`):
- Malformed cell: a paired (seed, fold) cell where
  `delta_loss is None` despite the underlying B5 manifest
  showing no `skipped_reason`. Indicates predictions shard
  corruption.
- Duplicate (seed, fold): the existing B11 helpers already
  reject duplicate `panel_row_index` shards, so duplicates at
  the (seed, fold) level shouldn't reach the aggregator. The
  defensive raise is included for parity with B15.
- OOM ceiling: reuses `BOOTSTRAP_ROW_COUNT_CEILING`.

## B16.3 B11 refactor: extract `compute_per_cell_lift_deltas`

Minimal-disruption refactor of
`benchmarks/experiments/ensemble_lift.py`:

- Extract the body of `_per_dataset_lift` that loops over
  `seed_fold_pairs` and computes per-(seed, fold) delta_loss +
  oracle_delta_loss into a new public-ish helper
  `compute_per_cell_lift_deltas`. Returns a list of
  `PerCellLiftDelta` records.
- `_per_dataset_lift` then becomes a thin aggregation over
  these records (mean + std), producing the same
  `PerDatasetLift` it does today.
- The B16 aggregator imports and calls
  `compute_per_cell_lift_deltas` directly.

Existing B11 tests (`test_ensemble_lift_experiment.py`) MUST
continue to pass byte-identically. A byte-string regression
pin on the `PerDatasetLift` output of a synthetic fixture
locks the equivalence (R-B16-1 mitigation).

## B16.4 Renderer integration

`benchmarks/report/ensemble_lift.py` follows the B15
dispatch shape:

- `render_ensemble_lift_markdown` (existing) keeps its signature.
- NEW `render_ensemble_lift_markdown_with_ci(result,
  rollup, *, expected_manifest_fingerprint,
  aggregator_error_class, manifest_unreadable)` adds the CI
  variant.

The CI variant follows the B15 5-source footnote precedence
verbatim (the same 4 dispatch fallbacks + source 5
loss-dropout disclosure for `n_skipped_cells > 0` rows).

The per-dataset Δloss column header replaces:
- `delta_loss_mean` → `Δloss [95% CI]`
- `delta_loss_std` column → removed (the CI subsumes it)

Sentinel-row footnote sourcing (mirrors the B15 arch-I1
closure pattern: existing renderer-side incomplete-block keeps
the pre-existing sentinel surface; the shared helper takes
only the B16-specific new sentinel):
- The two pre-existing B11 incomplete-group flags
  (`no_gbm_predictions`, `no_seq_predictions`) flow through
  the EXISTING `_render_dataset_table`'s incomplete-block
  sub-table unchanged. The rollup row's
  `bootstrap_skipped_reason` (`"no_gbm_predictions"` or
  `"no_seq_predictions"`) is just the rollup-side mirror of
  the same state; the renderer reads the UpliftRow-side
  flags as before.
- The new `all_cells_skipped_in_manifest` defensive sentinel
  surfaces via the shared `render_rollup_skipped_footnote`
  (this is the ONLY rollup-specific footnote source for B16).

The Wilcoxon block is unchanged.

`render_from_dir(output_root)`: same dispatch as B15 (failure-
sentinel check → rollup-presence + freshness → CI variant or
std fallback).

## B16.5 CLI dispatch + sentinel writes

`benchmarks/run.py` gains one new wrapper:

- `_run_bootstrap_ensemble_lift_rollup(config, *, env, output_root)`:
  runs AFTER `run_ensemble_lift` succeeds. Gates:
  1. Gate A: `is_ensemble_lift_rollup_enabled(config)`.
  2. Gate B: `run_manifest_path` present.
  3. Gate C: `load_run_manifest` with the narrow except.
  4. Gate D (arch-I3 closure): aggregator returns `[]` when
     the manifest has zero OK rows mapping to either the
     seq family OR the baseline family. Specifically: the
     aggregator computes `_model_families(manifest)` then
     checks "is there at least one OK manifest row whose
     model_name maps to baseline_family AND at least one
     mapping to seq_family". When both checks pass it
     proceeds; when either fails it returns `[]`. The
     wrapper treats this as a no-op skip (no failure
     sentinel). The prior draft incorrectly described this
     as "i.e., `run_ensemble_lift` would have failed" but
     `run_ensemble_lift` itself does NOT raise on partial-
     family manifests; it produces a `PerDatasetLift` with
     `no_gbm_predictions` or `no_seq_predictions` flagged
     (`benchmarks/experiments/ensemble_lift.py:672-677`
     only raises on completely empty manifests). The Gate D
     check here is symmetric: BOTH families must have at
     least one OK row in the manifest for the aggregator to
     produce a non-empty rollup.
  5. Aggregator call; catches `RawRollupError`, deletes
     partial output, drops the B11-specific sentinel.

The sentinel is INDEPENDENT from B5/B6/B7/B8 sentinels.

## B16.6 Test surface

`tests/benchmarks/test_bootstrap_ensemble_lift.py` (NEW; 15
named tests):

- `test_aggregate_bootstrap_ensemble_lift_rollup_paired_cells_emit_ci`:
  fixture with 2 seeds × 2 folds × both families;
  `loss(GBM)=0.60, loss(GBM+seq)=0.40` per cell so the
  expected Δ is exactly `+0.20`; assert
  `primary_loss_mean == pytest.approx(0.20, abs=1e-9)`
  (closes qa-C1 sign-convention: a sign-flip implementation
  returning `-0.20` would fail this bare-equality assertion).
- `test_aggregate_bootstrap_ensemble_lift_rollup_sign_convention_baseline_minus_combined`
  (closes qa-C2 with explicit ASYMMETRIC fixture):
  `loss(GBM)=0.60, loss(GBM+seq)=0.40`; assert
  `primary_loss_mean == pytest.approx(0.20, abs=1e-9)` with
  BARE equality. A symmetric fixture (e.g.,
  GBM=0.5=GBM+seq) would yield 0.0 and a sign flip would
  also yield 0.0, so the test must use asymmetric values.
- `test_aggregate_bootstrap_ensemble_lift_rollup_ci_width_nonzero_with_multiple_cells`:
  anti-degeneracy oracle; strict `ci_hi - ci_lo > 0`.
- `test_aggregate_bootstrap_ensemble_lift_rollup_seed_disjoint_yields_no_gbm_predictions_sentinel`
  (closes qa-C1, tightened in qa-R3-I1): fixture places GBM
  at seeds {0,1} fold 0 and seq at seeds {2,3} fold 0
  (completely disjoint seed sets, not just fold-disjoint).
  Assert `n_cells_paired == 0` AND bare equality
  `bootstrap_skipped_reason == "no_gbm_predictions"` (both
  B11 flags trip; the aggregator emits the lexically-first
  sentinel deterministically per the rule at line 141, so
  the test pins the exact ordering rather than a set
  membership). Pins that the intersection join is keyed on
  `(seed, fold)` jointly, not on seed alone or fold alone;
  a self-join bug would yield non-zero `n_cells_paired`.
- `test_aggregate_bootstrap_ensemble_lift_rollup_single_paired_cell_degenerate`:
  monkeypatched 1-cell case; primitive's single-entity-
  degenerate path fires.
- `test_aggregate_bootstrap_ensemble_lift_rollup_no_gbm_predictions_emits_sentinel`:
  bare equality on `"no_gbm_predictions"`. Fixture: only seq
  rows in the dataset under test, plus a second dataset
  with both families to bypass Gate D.
- `test_aggregate_bootstrap_ensemble_lift_rollup_no_seq_predictions_emits_sentinel`:
  bare equality on `"no_seq_predictions"`. Symmetric fixture.
- `test_aggregate_bootstrap_ensemble_lift_rollup_all_cells_skipped_emits_sentinel`
  (closes qa-I4 Gate D bypass note): fixture with the
  target group ALL skipped + a second group with valid
  cells from both families (so Gate D passes). Bare
  equality on `"all_cells_skipped_in_manifest"`.
- `test_aggregate_bootstrap_ensemble_lift_rollup_empty_manifest_returns_empty_list`.
- `test_aggregate_bootstrap_ensemble_lift_rollup_records_manifest_fingerprint`.
- `test_aggregate_bootstrap_ensemble_lift_rollup_records_primary_loss_column_for_binary_and_regression`:
  parametrized over `task_type in {"binary",
  "regression_point"}`, asserts `primary_loss_column ==
  "log_loss"` and `"rmse"` respectively.
- `test_aggregate_bootstrap_ensemble_lift_rollup_malformed_cell_raises`:
  per-cell delta None despite no skipped_reason → raise.
- `test_aggregate_bootstrap_ensemble_lift_rollup_duplicate_seed_fold_pair_raises`
  (closes R2 qa-C1): feed the inner-join with a fixture
  where the same `(seed=0, fold=0)` pair appears twice in
  the paired-cell list; assert `RawRollupError` is raised
  rather than silently inflating `n_cells_paired` (parity
  with B15's `_duplicate_seed_fold_pair_raises`).
- `test_aggregate_bootstrap_ensemble_lift_rollup_oom_gate_raises`.
- `test_aggregate_bootstrap_ensemble_lift_rollup_respects_per_spec_n_resamples_override`:
  fixture sets `bootstrap_n_resamples=137` (an unusual,
  detectable value); monkeypatched primitive captures
  `n_resamples` and asserts equality on 137.

`tests/benchmarks/test_run_bootstrap_ensemble_lift_wrapper.py`
(NEW; 13 named tests; mirrors the B15 wrapper pattern with
an additional Gate D symmetry test):
- `test_run_bootstrap_ensemble_lift_rollup_writes_shard_on_happy_path`
- `test_run_bootstrap_ensemble_lift_rollup_skips_via_opt_out`
- `test_run_bootstrap_ensemble_lift_rollup_skips_when_b5_manifest_absent`
- `test_run_bootstrap_ensemble_lift_rollup_skips_when_seq_family_absent_from_manifest`
  (Gate D, seq direction)
- `test_run_bootstrap_ensemble_lift_rollup_skips_when_baseline_family_absent_from_manifest`
  (closes qa-C4 Gate D SYMMETRIC direction): manifest has
  only seq-family rows; assert wrapper returns without
  writing rollup or failure sentinel.
- `test_run_bootstrap_ensemble_lift_rollup_skips_when_run_manifest_absent`
- `test_run_bootstrap_ensemble_lift_rollup_skips_when_run_manifest_load_fails`
- `test_run_bootstrap_ensemble_lift_rollup_catches_raw_rollup_error_and_continues`
- `test_run_bootstrap_ensemble_lift_rollup_writes_failure_sentinel_on_raw_rollup_error`
- `test_run_bootstrap_ensemble_lift_rollup_unlinks_stale_failure_sentinel_on_success`
- `test_run_bootstrap_ensemble_lift_rollup_failure_does_not_touch_other_sentinels`
  (arch-N3 closure: name shortened so a future B17 doesn't
  need to rename to enumerate). The test body pre-plants ALL
  FOUR other sentinels (B5, B6, B7, B8) using their
  exported path helpers from `bootstrap_manifest`, writes
  distinct content strings ("PrevB5Failure" etc.), and
  asserts `read_text() == "PrevB[N]Failure"` on each after
  the B16 failure (B14 Stage-3 qa-I3 closure pattern: read
  content, not just exists).
- `test_run_bootstrap_ensemble_lift_rollup_writes_no_gbm_predictions_sentinel_row_in_shard`:
  end-to-end propagation test (analogous to B15's
  `no_paired_cells_sentinel_row_in_shard`): when the
  aggregator emits a `no_gbm_predictions` sentinel for a
  dataset, the wrapper writes the shard containing it AND
  does NOT write a failure sentinel.
- `test_run_bootstrap_ensemble_lift_rollup_writes_no_seq_predictions_sentinel_row_in_shard`
  (closes R2 qa-I2 symmetric mirror): same as above with
  the symmetric sentinel string, pinning both directions of
  the wrapper-level shard write.

`tests/benchmarks/test_bootstrap_manifest.py` (extension):
add the 8 schema/path/ExperimentSpec tests for the new
symbols (path-format × 2, round-trip × 1 with explicit
primary_loss_column + n_seeds + n_folds + n_cells_paired
pins, extra-forbid × 1, empty-shard × 1, absent-on-load × 1,
ExperimentSpec default-True + opt-out × 2).

`tests/benchmarks/test_ensemble_lift_report_b16.py` (NEW;
21 named tests; the count includes the R2 qa-I1 symmetric
mirror added below):

Footnote-precedence + behavior (15 tests):
- `test_render_ensemble_lift_with_ci_renders_mean_and_interval`
- `test_render_ensemble_lift_without_ci_falls_back_to_scalar`
- `test_render_ensemble_lift_with_ci_drops_delta_loss_mean_header_when_ci_present`
  (asserts the BARE `delta_loss_mean` column header is
  ABSENT from the per-dataset table header row AND
  `Δloss [95% CI]` IS present (both directions per qa-I2).
- `test_render_ensemble_lift_with_ci_drops_delta_loss_std_column_when_ci_present`
  (closes qa-I2 two-column-to-one pin: asserts
  `delta_loss_std` column header is ABSENT from the table
  header row AND `Δloss [95% CI]` IS present).
- `test_render_ensemble_lift_with_ci_falls_back_on_manifest_fingerprint_mismatch`
- `test_render_ensemble_lift_with_ci_surfaces_aggregator_failed_footnote_on_wrapper_caught`
- `test_render_ensemble_lift_with_ci_manifest_unreadable_appends_freshness_footnote`
- `test_render_ensemble_lift_with_ci_surfaces_skipped_cells_footnote_when_n_skipped_cells_nonzero`
  (source 5 footnote pin).
- `test_render_ensemble_lift_with_ci_surfaces_skipped_cells_footnote_without_asterisk_when_no_pairing_asymmetry`
  (source 5 vs partial-fold asterisk independence).
- `test_render_ensemble_lift_with_ci_surfaces_both_freshness_and_skipped_cells_footnotes_when_both_apply`
  (source 4 + source 5 co-occurrence).
- `test_render_ensemble_lift_with_ci_marks_partial_fold_cell_with_asterisk`
- `test_render_ensemble_lift_with_ci_renders_no_ci_when_rollup_missing_for_dataset`
  (manifest has a dataset, rollup has no row -> `(no CI)`
  sentinel cell).
- `test_render_ensemble_lift_with_ci_no_gbm_predictions_routes_to_incomplete_block`
  (closes qa-I1: pre-existing B11 incomplete-block path
  receives the sentinel rollup row, NOT the shared helper).
- `test_render_ensemble_lift_with_ci_no_seq_predictions_routes_to_incomplete_block`
  (closes R2 qa-I1 symmetric mirror): same as the
  `no_gbm_predictions` test with the symmetric sentinel
  string, pinning both directions of the routing.
- `test_render_ensemble_lift_with_ci_all_cells_skipped_routes_to_shared_render_rollup_skipped_footnote`
  (the only sentinel that uses the new shared-helper path).

render_from_dir dispatch (6 tests):
- `test_render_from_dir_falls_back_silently_when_rollup_file_absent`
- `test_render_from_dir_surfaces_aggregator_failed_sentinel`
- `test_render_from_dir_renders_ci_when_rollup_present_and_fingerprint_matches`
- `test_render_from_dir_falls_back_on_stale_rollup`
- `test_render_from_dir_freshness_check_skipped_when_manifest_corrupt`
- `test_render_from_dir_renders_ci_without_freshness_footnote_when_manifest_absent`

`tests/benchmarks/test_ensemble_lift_experiment.py`
(extension): one byte-string regression pin on
`_per_dataset_lift` output to lock the B11 refactor
equivalence (closes qa-C3 with explicit scope): the
assertion is `assert per_dataset_lift_row.model_dump() ==
EXPECTED_DICT` where `EXPECTED_DICT` is a hard-coded dict
covering EVERY field on `PerDatasetLift` (including
`delta_loss_std`, `n_cells_paired`, `oracle_loss_mean`,
`oracle_delta_loss_mean`, AND `primary_loss_column`). A pin
on a single field would let a refactor silently change the
others. The fixture is a synthetic 2-seed × 2-fold manifest
with controlled per-cell losses; the expected dict's values
are computed from the fixture inputs at test-write time and
committed in the test file.

## B16.7 Estimated effort

| Module | Size |
|---|---|
| `benchmarks/bootstrap_manifest.py` (+1 schema + 2 path helpers + 2 I/O helpers) | Small |
| `benchmarks/config.py` (+1 ExperimentSpec field) | Trivial |
| `benchmarks/report/bootstrap_ensemble_lift.py` (NEW) | ~320 lines |
| `benchmarks/experiments/ensemble_lift.py` (refactor: extract `compute_per_cell_lift_deltas`) | ~40 lines |
| `benchmarks/report/ensemble_lift.py` (+CI variant + dispatch) | ~280 lines |
| `benchmarks/run.py` (+1 wrapper) | ~80 lines |
| `tests/benchmarks/test_bootstrap_ensemble_lift.py` (NEW) | ~400 lines |
| `tests/benchmarks/test_run_bootstrap_ensemble_lift_wrapper.py` (NEW) | ~300 lines |
| `tests/benchmarks/test_bootstrap_manifest.py` (extension) | ~80 lines |
| `tests/benchmarks/test_ensemble_lift_report_b16.py` (NEW) | ~400 lines |
| `tests/benchmarks/test_ensemble_lift_experiment.py` (byte-string regression pin) | ~80 lines |

Total: ~1,980 lines. Slightly larger than B15 because the
B11 refactor adds a regression pin.

## Risk register (delta)

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R-B16-1 | Extracting `compute_per_cell_lift_deltas` from `_per_dataset_lift` could change the per-dataset summary output, breaking downstream B11 callers. | High | Byte-string regression pin on `_per_dataset_lift` output of a synthetic fixture; ALL existing B11 tests must continue to pass. |
| R-B16-2 | The aggregator recomputes per-cell deltas (it does not read a persisted shard); a future B11 driver change could make the aggregator's output diverge from the renderer's std view. | Medium | The shared helper is the single source of truth; both the std view AND the CI view consume it. R-B16-1 byte-pin catches drift. |
| R-B16-3 | The cross-report sentinel-isolation surface is now FOUR other sentinels (B5, B6, B7, B8). The test pre-plants and asserts all four. | Low | Test pin mirrors the B14 Stage-3 qa-I3 pattern. |
| R-B16-4 | The oracle Δ (per-sample best ceiling) has no CI at v1; reviewers may ask. | Low | Documented as D-B16.1 follow-up. |
| R-B16-5 | The `primary_loss_column` audit field is on 2 of the now-5 RollupRow schemas (B15 HPOUpliftRollupRow + B16 EnsembleLiftRollupRow); the 3 B13/B14 schemas still lack it. | Low | D-B16.5 covers a coordinated lift + rename across all five schemas. |

## B16-followup deferrals (out of scope for v1)

- D-B16.1: CI on the oracle Δ (per-sample best ceiling).
- D-B16.2: BCa CI (inherited from D-B13.5).
- D-B16.3: per-fold CIs (inherited from D-B13.6).
- D-B16.4: per-entity sufficient-statistics OOM optimization
  (inherited from D-B13.7).
- D-B16.5: coordinated `primary_loss_*` → `primary_metric_*`
  rename across all five RollupRow schemas.
- D-B16.6: shared CLI-wrapper factory across the now-FIVE
  `_run_bootstrap_*_rollup` wrappers (inherited from
  D-B14.3).

## Addressed

R1 swarm: architecture-reviewer (1C/5I/3N REQUEST_CHANGES),
qa-test-coverage (4C/4I/2N REQUEST_CHANGES), style-reviewer
(1C/0I/1N REQUEST_CHANGES). Total deduped: 6 CRITICAL,
9 IMPROVEMENT, 6 NITPICK. CRITICALs addressed:

- **arch-C1** (`no_paired_cells` sentinel mis-modeled because
  B11 doesn't have that semantic): the sentinel is REMOVED.
  B16 enumerates THREE sentinel strings (`no_gbm_predictions`,
  `no_seq_predictions`, `all_cells_skipped_in_manifest`)
  matching B11's actual flag pair. Intersection-empty rolls
  through the existing pair of B11 flags.
- **qa-C1** (pairing contract test missing): B16.6 adds
  `_seed_disjoint_yields_no_gbm_predictions_or_no_seq_predictions_flags`
  with completely disjoint seed sets so a self-join bug
  would yield non-zero `n_cells_paired` and fail the test.
- **qa-C2** (sign convention fixture unspecified): B16.6's
  `_paired_cells_emit_ci` AND `_sign_convention_baseline_minus_combined`
  now explicitly specify `loss(GBM)=0.60, loss(GBM+seq)=0.40`
  asymmetric fixture so a sign flip yields `-0.20` and fails
  the bare-equality assertion on `+0.20`.
- **qa-C3** (byte-string regression pin scope unspecified):
  the B11-extension test now mandates
  `assert per_dataset_lift_row.model_dump() == EXPECTED_DICT`
  with the EXPECTED_DICT covering EVERY field on
  PerDatasetLift.
- **qa-C4** (Gate D symmetric case unnamed): B16.6 adds
  `_skips_when_baseline_family_absent_from_manifest` plus the
  existing `_skips_when_seq_family_absent_from_manifest` so
  both Gate D directions are pinned.
- **style-C1** (em dash at the option (b) line): replaced
  with a semicolon.

IMPROVEMENTs addressed:

- **arch-I1** (option (d) boolean-kwarg alternative for the
  refactor): added to the Background evaluation with the
  explicit "(c) over (d)" choice rationale.
- **arch-I2** (`seq_family`, `baseline_family` redundant on
  every row): DROPPED from the schema. The renderer reads
  them from `EnsembleLiftExperimentResult` directly. A
  future multi-pair driver would add them as a schema
  migration.
- **arch-I3** (Gate D's "would have failed" claim is false):
  B16.5 Gate D rewritten to be precisely a content check on
  the manifest: at least one OK row mapping to baseline AND
  at least one mapping to seq. The "would have failed"
  parenthetical was removed.
- **arch-I4** (footnote-sourcing precedent citation):
  B16.4 now explicitly cites the B15 arch-I1 closure
  pattern.
- **arch-I5** (ExperimentSpec first-field-consumed note):
  B16.1 now states that this is the first field B11
  consumes on its spec.
- **qa-I1** (no_paired_cells vs all_cells_skipped routing
  test): B16.6 names the two distinct renderer tests
  explicitly.
- **qa-I2** (two-column-to-one assertion direction): both
  the absent-old-header AND present-new-header assertions
  are now mandated in the test descriptions.
- **qa-I3** (cross-report sentinel isolation enumerates four
  others): B16.6 explicitly mandates importing all four
  prior sentinel path helpers (B5/B6/B7/B8) by name; the
  test is renamed `_failure_does_not_touch_other_sentinels`
  per arch-N3.
- **qa-I4** (all_cells_skipped fixture Gate D bypass):
  B16.6 specifies the two-group fixture trick (target group
  all-skipped + second group with valid both-family rows).

NITPICKs addressed:

- **arch-N1** (all_cells_skipped is in B15 impl but not B15
  design): cited at the B15 impl path
  `benchmarks/report/bootstrap_hpo_uplift.py:182` directly,
  not at the B15 design doc.
- **arch-N2** (R-B16-5 schema-count phrasing): rephrased to
  "2 of the now-5 schemas have it; the 3 B13/B14 schemas
  still lack it."
- **arch-N3** (cross-sentinel test name unwieldy): renamed
  `_failure_does_not_touch_other_sentinels`.
- **qa-N1** (schema test field-combo coverage): the existing
  8-test extension matches the B13/B15 pattern; a
  frozen-mutation test is inherited from pydantic and not
  added.
- **qa-N2** (n_resamples override value unspecified): the
  B16.6 description now mandates `bootstrap_n_resamples=137`
  (an unusual, detectable value).
- **style-N1** (D-B16.4/5/6 wrap across lines): not
  reformatted; the wrapped form preserves clarity at the
  cost of line length.

### R2 swarm closure

R2 confirming swarm: architecture-reviewer (1C/0I/0N
REQUEST_CHANGES), qa-test-coverage (1C/2I/1N
REQUEST_CHANGES), style-reviewer (0C/0I/0N APPROVE). Total:
2 CRITICAL, 2 IMPROVEMENT, 1 NITPICK. Closures:

- **arch-R2-C1** (stale prose at B16.0 contradicting the
  arch-I2 schema closure): the offending paragraph that
  asserted `seq_family` + `baseline_family` are "captured
  on every row" has been rewritten to state explicitly that
  the renderer reads them from
  `EnsembleLiftExperimentResult` directly.
- **qa-R2-C1** (duplicate-(seed, fold) raise unnamed): B16.6
  adds
  `_duplicate_seed_fold_pair_raises` parity with B15.
- **qa-R2-I1** (symmetric `no_seq_predictions` renderer
  routing): B16.6 adds
  `_no_seq_predictions_routes_to_incomplete_block` mirror.
- **qa-R2-I2** (symmetric `no_seq_predictions` wrapper end-to-
  end): B16.6 adds the mirror wrapper test.
- **qa-R2-N1** (count mismatch in renderer block headers):
  bumped to "21 named tests" with "15 footnote-precedence +
  6 dispatch" breakdown matching the enumerated tests.

### R3 swarm closure

R3 confirming swarm: architecture-reviewer (0C/0I/1N
APPROVE), qa-test-coverage (0C/1I/1N APPROVE),
style-reviewer (0C/0I/0N APPROVE). Total: 0 CRITICAL,
1 IMPROVEMENT, 2 NITPICK. Consensus reached. Closures:

- **qa-R3-I1** (seed-disjoint test asserted set-membership
  but the doc specifies deterministic lexically-first
  ordering): tightened to bare equality on
  `"no_gbm_predictions"` and renamed
  `_seed_disjoint_yields_no_gbm_predictions_sentinel` to
  reflect the deterministic single-sentinel expectation.
- **arch-R3-N1** (stale "four enumerated cases" prose at
  B16.5 step 2): edited to "three enumerated cases" after
  `no_paired_cells` was dropped in R1.
- **qa-R3-N1** (carried NITPICK on a future per-fold CI
  test scaffold): deferred under D-B16.3.

## Deferred

R1 swarm deferrals carried forward (each documented above
with rationale):

- **arch-I1 option (d)**: documented as a viable alternative
  to (c). Choice (c) shipped.
- **arch-I2 alternative (b)**: keeping `seq_family` +
  `baseline_family` on the schema for forward-compatibility
  was the alternative; option (a) (drop) was chosen.
- **style-N1**: D-B16.4/5/6 line-wrap preserves clarity.

### Stage 1 R1 swarm closure (impl `5299eeb`, closure `8674fee`)

Stage 1 confirming swarm: code-reviewer (0C/1I/1N APPROVE),
architecture-reviewer (0C/0I/0N APPROVE), qa-test-coverage
(0C/2I/1N APPROVE), style-reviewer (0C/0I/0N APPROVE).
Total: 0 CRITICAL, 3 IMPROVEMENT, 2 NITPICK. Closures:

- **code-R1-I1** (round-trip rows[0] and rows[1] both used
  `dataset_name="fake_binary"`): second row now overrides
  `dataset_name="fake_regression"` so a hypothetical
  `dataset_name`-keyed dedup/reorder bug in the load path
  would surface.
- **code-R1-N1** (`_make_ensemble_lift_row` reused the B15
  factory's `"feedbeef" * 8` hex string): switched to
  `"beeff00d" * 8` so the B16 factory is orthogonal from the
  B15 factory.
- **qa-R1-I1** (parametrize list in
  `test_benchmark_config_rejects_duplicates_for_every_kind`
  was stale): added `"ensemble_lift"` so a future
  kind-specific carve-out in `_at_most_one_spec_per_kind`
  cannot silently exempt B11.
- **qa-R1-I2** (no dedicated atomic-replace overwrite test
  for `write_ensemble_lift_rollup`): DEFERRED. The
  `_write_rows_atomic` helper is shared with B14
  (`write_pairwise_rollup`, `write_training_time_rollup`)
  and B15 (`write_hpo_uplift_rollup`). B16's
  `write_ensemble_lift_rollup` invokes it with the same
  arguments shape. The helper is exercised transitively by
  the B14/B15 round-trip tests
  (`test_write_pairwise_rollup_then_load_round_trips_all_fields`,
  `test_write_training_time_rollup_then_load_round_trips_all_fields`,
  `test_write_hpo_uplift_rollup_then_load_round_trips_all_fields`)
  which prove the write/read contract end-to-end. The
  baseline `test_write_rollup_atomic_replace_on_overwrite`
  exercises `write_rollup`'s own inline implementation, NOT
  `_write_rows_atomic`; the helper's `tmp.replace(dest)` is
  a single OS call with no branching, so the risk of
  silent failure on overwrite-via-`_write_rows_atomic` is
  bounded. No silent-failure risk on the B16-specific path.
- **qa-R1-N1** (`manifest_fingerprint` accepts any string;
  no SHA-256 `pattern=` constraint): DEFERRED under D-B16.5
  (coordinated rename + audit across all five RollupRow
  schemas) so the constraint lands uniformly rather than
  drifting across phases.

### Stage 2 R1 swarm closure

Stage 2 confirming swarm (impl `79fb76e`): code-reviewer
(0C/1I/1N APPROVE), architecture-reviewer (0C/4I/3N APPROVE),
qa-test-coverage (0C/2I/1N APPROVE), style-reviewer
(0C/0I/0N APPROVE). Total: 0 CRITICAL, 7 IMPROVEMENT, 5
NITPICK. Closures:

- **code-R1-I1** (`arbitrary_types_allowed=True` on
  `ComputePerCellLiftDeltasResult` was unnecessary): removed.
  `tuple[PerCellLiftDelta, ...]` round-trips through pydantic
  v2 without the flag.
- **arch-R1-I1** (private `_ComputePerCellResult` as the
  return type of a public function): renamed to
  `ComputePerCellLiftDeltasResult` and added to B11's
  `__all__`. The cross-module field-access contract
  (`.cells`, `.seen_no_gbm`, `.seen_no_seq`, `.selector`) is
  now explicit.
- **arch-R1-I2** (`_PRIMARY_LOSS_COLUMN_BY_TASK.get(task_type,
  "log_loss")` default branch was dead given the
  `_PRIMARY_LOSS_COLUMN_BY_TASK` containment guard at the
  caller): replaced with direct subscription
  `_PRIMARY_LOSS_COLUMN_BY_TASK[task_type]`. A KeyError here
  now surfaces a caller-contract violation rather than
  silently mislabeling a future task type.
- **arch-R1-I3** (four B11 private symbols imported across
  the module boundary): promoted to public API in B11
  (`DEFAULT_BASELINE_FAMILY`, `DEFAULT_SEQ_FAMILY`,
  `model_families`, `build_cells_table`) and added to
  `__all__`. The B16 aggregator now imports public names; a
  future rename of any of those four must update both
  modules in lockstep, with the public symbol making the
  contract explicit.
- **arch-R1-I4** (predicate-priority vs lexical-ordering
  ambiguity at the sentinel-routing cascade): rewritten as
  an explicit `sorted(candidate_reasons)[0]` selection so
  the lexical-first rule is observably enforced. The
  comment now documents the rule as the source of truth.
- **qa-R1-I1** (Gate B/C wrapper tests missing failure-
  sentinel-absence assertion): both `_skips_when_run_
  manifest_absent` and `_skips_when_run_manifest_load_fails`
  now assert `not ensemble_lift_aggregator_failed_sentinel_
  path(output_root).exists()` after the skip path.
- **qa-R1-I2** (the dataset-loop `continue` branches were
  untested): added
  `_excludes_unsupported_and_inconsistent_task_type_datasets`
  with a 3-dataset fixture covering both branches
  (unsupported `regression_quantile`, mixed-task_type within
  a single dataset). Assert only the valid binary dataset
  reaches the rollup.

NITPICKs:

- **code-R1-N1** (`ensemble_lift_rollup_output_path` log-
  message helper naming): NOT changed. Matches the B15
  precedent at `bootstrap_hpo_uplift.py:89-91`; renaming
  would create a B15/B16 asymmetry without a consumer
  benefit.
- **arch-R1-N1** (design doc still referred to
  `_compute_per_cell_lift_deltas` with a leading underscore
  while the implementation is public): renamed throughout
  the doc.
- **arch-R1-N2** (collapses into arch-R1-I2): closed by
  the I2 fix.
- **arch-R1-N3** (B16 wrapper call after markdown render vs
  B5/B6/B7/B8 pattern placing the wrapper before the
  render): NOT changed. The B16 std renderer
  (`render_ensemble_lift_markdown`) does not consume the
  rollup; the future CI-aware `render_from_dir` dispatch
  ships in Stage 3 and will be placed before its own
  rollup-aware render at that point. The current ordering
  is harmless and preserves the existing call sequence.
- **qa-R1-N1** (byte-pin file location): NOT changed. The
  test exercises both `compute_per_cell_lift_deltas` AND
  `_per_dataset_lift` through a monkeypatched call; placing
  it next to the other B16 aggregator tests keeps the
  Stage-2 refactor pin co-located with the other B16
  coverage rather than fragmenting it across two files.
