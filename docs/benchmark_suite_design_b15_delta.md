# B15 design delta: HPO-uplift Δ-statistic paired CI (D-B13.3)

**Scope**: D-B13.3 closes the B13 deferral for the B8 HPO-uplift
report. The existing B8 renderer ships a per-(dataset, model)
scalar Δ = `default_mean - tuned_mean`; v1 adds a paired
percentile CI around each Δ via the existing entity-block
bootstrap primitive. The Friedman/Holm matrix block is
UNCHANGED.

## Requirements

This design is graded against the following requirements:

- **R1**: `hpo_uplift.md` gains a `Δ [95% CI]` column on every
  per-(dataset, model) Δ row. Sentinel rows
  (`default_only`, `tuned_only`, `paired_but_no_valid_loss`)
  keep the existing footnote treatment; no CI is computed for
  them.
- **R2**: The CI is PAIRED on the per-cell (default, tuned)
  losses indexed by (seed, fold). The bootstrap resamples
  paired cells with replacement; each resample's mean is the
  Δ statistic for that resample.
- **R3**: REUSE the B13 entity-block bootstrap primitive
  (`benchmarks/metrics/bootstrap.py:entity_block_bootstrap_ci`)
  via the cell-as-entity contract (each paired cell is its own
  entity).
- **R4**: REUSE the B13/B14 freshness machinery
  (`RunManifest.fingerprint()` on every emitted row; stale-
  rollup fallback in the renderer).
- **R5**: REUSE the B13/B14 `RawRollupError` + CLI-wrapper
  sentinel-write pattern. The B8 wrapper gets its own
  sentinel file independent from B5 / B6 / B7.
- **R6**: Opt out via a per-experiment ExperimentSpec field
  `bootstrap_hpo_uplift_enabled: bool = True`. Matches the
  B14 pattern for ensemble + training-time flags.
- **R7**: NO new shard layout primitives. The new
  `bootstrap_hpo_uplift_rollup.parquet` shard reuses the same
  atomic-rename write pattern.
- **R8**: NO change to the existing B8 manifest, driver, or
  CLI dispatch order. The CI rollup runs AFTER
  `run_hpo_uplift` succeeds.

## Background

B14 wired CIs onto B5 (raw_loss), B6 (ensemble pairwise), and
B7 (training-time). B13's deferral list named the remaining
three CI integrations: B6/B7 (closed in B14) and B8 HPO-uplift
(this delta). The fourth deferral, D-B13.4 (B11 ensemble-lift),
is a separate phase.

The B8 Δ-statistic has a NEW SHAPE compared to B5/B6/B7:
- B5: per-cell loss; mean over cells.
- B6: per-cell complementarity score; mean over cells.
- B7: per-cell wall_seconds; mean over cells.
- **B8: per-cell PAIRED difference `(default_loss_i - tuned_loss_i)`; mean over cells.**

The paired-difference shape is statistically standard for
"did tuning help on this (dataset, model)?" The bootstrap
resamples PAIRS (default + tuned at the same (seed, fold))
with replacement, computing the mean difference per resample.

## B15.0 New typed surface declarations

The new symbols mirror B14.0:

- `benchmarks/bootstrap_manifest.py:HPOUpliftRollupRow`
  (new pydantic schema)
- `benchmarks/bootstrap_manifest.py:hpo_uplift_rollup_path(root)`
  helper for `{root}/bootstrap_hpo_uplift_rollup.parquet`
- `benchmarks/bootstrap_manifest.py:hpo_uplift_aggregator_failed_sentinel_path(root)`
  helper for `{root}/bootstrap_hpo_uplift_aggregator_failed.txt`
- `benchmarks/report/bootstrap_hpo_uplift.py:is_hpo_uplift_rollup_enabled(config)`
  predicate (mirrors `is_pairwise_rollup_enabled` and
  `is_training_time_rollup_enabled`)
- `benchmarks/report/bootstrap_hpo_uplift.py:hpo_uplift_rollup_output_path()`
  stable filename token for log messages
- `write_hpo_uplift_rollup` + `load_hpo_uplift_rollup` I/O
  helpers using the existing shared `_write_rows_atomic` helper

The B13 `RawRollupError` is REUSED (no new exception type).

### Sentinel string enumeration

The aggregator emits FOUR distinct `bootstrap_skipped_reason`
strings on sentinel rollup rows:

1. `"default_only"`: the underlying B8 row carries
   `default_only=True` (no tuned arm cells in the manifest).
2. `"tuned_only"`: the underlying B8 row carries
   `tuned_only=True`.
3. `"paired_but_no_valid_loss"`: the underlying B8 row carries
   `paired_but_no_valid_loss=True` (both arms ran but every
   paired cell had a missing primary loss).
4. `"no_paired_cells"`: the inner-join on (seed, fold) between
   default + tuned arms yielded zero matches even though both
   arms produced cells (e.g., default at fold 0; tuned at
   fold 1; no overlap). Distinct from the three above because
   the existing B8 `UpliftRow` does NOT have a corresponding
   boolean flag for this case.

### HPOUpliftRollupRow schema

```python
class HPOUpliftRollupRow(BaseModel):
    """One per-(dataset, model, task_type) HPO-uplift Δ CI entry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_name: str
    model_name: str
    task_type: str
    primary_metric: str  # always "delta" at v1
    primary_loss_column: str  # the underlying loss column for audit (e.g. "log_loss")
    n_seeds: int = Field(ge=0)  # seeds in the OK B8 manifest for this group
    n_folds: int = Field(ge=0)  # folds in the OK B8 manifest for this group
    n_cells_paired: int = Field(ge=0)  # cells where BOTH default + tuned ran
    n_skipped_cells: int = Field(ge=0)  # paired cells where a NaN loss dropped the cell from the bootstrap
    primary_loss_mean: float | None = None  # Δ statistic
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

Field-name reuse rationale: `primary_loss_*` mirrors the B13
RollupRow + B14 PairwiseRollupRow + B14
TrainingTimeRollupRow so the shared `format_ci_cell` helper
works without a discriminator. The `primary_metric: str`
column carries the semantic label (`"delta"` at v1).

`primary_loss_column` is NEW relative to the B14 schemas: it
records the underlying loss column the Δ was computed against
(`"log_loss"` for binary/multiclass, `"rmse"` for
regression_point, `"pinball_mean"` for regression_quantile)
so a reader can audit the rollup against the manifest without
re-resolving the selector. Renderers ignore it; it's a
durability + audit field.

Sentinel rows: when the underlying B8 row is `default_only`,
`tuned_only`, `paired_but_no_valid_loss`, or when the aggregator's
own inner-join yields no paired cells (`no_paired_cells`), the
aggregator emits a sentinel rollup row with `n_cells_paired=0`,
`primary_loss_*=None`, and the corresponding string in
`bootstrap_skipped_reason` (one of the four enumerated above).

## B15.1 ExperimentSpec extension

One new field on `benchmarks/config.py:ExperimentSpec`:

```python
bootstrap_hpo_uplift_enabled: bool = True
```

Semantics:
- Meaningful only when `kind="hpo_uplift"`; ignored on other
  kinds.
- Default True so the rollup runs by default when an HPO-uplift
  spec is configured.
- The existing `bootstrap_n_resamples` is per-ExperimentSpec
  (per-kind override), reused as-is.

## B15.2 New aggregator: `benchmarks/report/bootstrap_hpo_uplift.py`

Reads the B5 manifest (the HPO-uplift driver writes default +
tuned variant rows into the same manifest under separate
`variant` values), pairs cells per (dataset, model, seed, fold),
computes per-cell Δ, and bootstraps.

```python
def aggregate_bootstrap_hpo_uplift_rollup(
    config: BenchmarkConfig,
    *,
    output_root: Path,
    env: RunEnvironment,
    manifest: RunManifest,
) -> list[HPOUpliftRollupRow]:
    df = load_run(output_root)
    if df.empty:
        return []
    # ... group by (dataset, model, task_type); for each group:
    #   - split into default_block + tuned_block by variant
    #   - inner-join on (seed, fold_index) so only PAIRED cells
    #     survive
    #   - compute per-cell delta = default_loss - tuned_loss
    #   - bootstrap with cell-index as entity-id
    #   - emit HPOUpliftRollupRow per group
    # Sentinel rows for default_only / tuned_only /
    # paired_but_no_valid_loss carry primary_loss_*=None.
```

Failure modes (raise `RawRollupError`):
- Malformed paired cell: a paired (seed, fold) cell where the
  primary loss is None on either side AND the underlying B8
  group row (keyed at the (dataset, model) grain, NOT the
  cell grain) carries `paired_but_no_valid_loss=False`. The
  grain mismatch matters: if the group is already flagged as
  `paired_but_no_valid_loss=True`, the aggregator emits a
  sentinel row (no raise). Only the "group says OK but
  individual cell is NaN" case is the malformed-cell raise.
- Duplicate (seed, fold) pair: within a single (dataset, model,
  variant) group, two rows sharing the same (seed, fold_index)
  indicates a corrupt manifest; raise.
- OOM ceiling: reuses the B13 `BOOTSTRAP_ROW_COUNT_CEILING`
  via the shared `_bootstrap_aggregate` constant.

Sentinel emission policy (no raise):
- The underlying B8 row has `default_only`, `tuned_only`, or
  `paired_but_no_valid_loss` set → emit the corresponding
  sentinel rollup row (`default_only`, `tuned_only`, or
  `paired_but_no_valid_loss`).
- The aggregator's inner-join on (seed, fold) yields zero
  paired cells → emit `no_paired_cells` sentinel rollup row.
- A paired cell has a NaN primary loss AND the underlying
  B8 group row is already flagged as
  `paired_but_no_valid_loss=True` → the cell is dropped
  silently from the bootstrap (no raise; the sentinel row
  is emitted at the group level).

Profile dispatch reuses `BOOTSTRAP_N_RESAMPLES_BY_PROFILE` via
the shared `resolve_n_resamples` helper (B14 Stage-3 hoist).

## B15.3 Renderer integration

`benchmarks/report/hpo_uplift.py` ships the same dispatch shape
as the B5/B6/B7 CI variants:

- `render_hpo_uplift_markdown` (existing) keeps its signature.
- NEW `render_hpo_uplift_markdown_with_ci(manifest, rollup, *,
  reference_model, expected_manifest_fingerprint,
  aggregator_error_class, manifest_unreadable)` adds the CI
  variant.

The CI variant follows the B14 4-source footnote precedence
plus one B15-specific source for loss-side dropouts:

1. `aggregator_error_class is not None` → std variant +
   "Bootstrap aggregator failed: <class>" footnote.
2. `not rollup` → std variant silently.
3. `expected_manifest_fingerprint` set AND any row mismatch
   → std variant + "rollup is stale" footnote.
4. CI variant body renders; if `manifest_unreadable=True`,
   append "freshness check skipped" footnote.
5. CI body iteration: if ANY rollup row has
   `n_skipped_cells > 0`, append a "Partial coverage"
   footnote that lists per-(dataset, model) the
   `n_skipped_cells / n_cells_paired` ratio. Independent
   from sources 1-4; this is loss-dropout disclosure, not a
   dispatch fallback.

The per-dataset Δ table's `Δ` column header is REPLACED with
`Δ [95% CI]`.

Partial-fold asterisk denominator (closes arch-I4): the
`format_ci_cell` partial flag fires when
`rollup_row.n_cells_paired < rollup_row.n_seeds * n_folds`
(the B14 convention). The B15 `HPOUpliftRollupRow` carries
`n_seeds` AND `n_folds` AND `n_cells_paired` so the renderer
can distinguish:
  - PAIRED-ASYMMETRY case: only some (seed, fold) cells were
    paired across the two arms (default has 2 seeds, tuned
    has 3); `n_cells_paired < n_seeds * n_folds`; `*` fires.
  - LOSS-DROPOUT case: paired cells exist but the bootstrap
    skipped some (e.g., NaN primary loss on one cell of the
    pair); this is reported via `n_skipped_cells` on the
    rollup row and surfaced as a secondary "partial coverage"
    footnote when `n_skipped_cells > 0`.

Sentinel-row footnote sourcing (closes arch-I1): the renderer
keeps the existing `_render_footnote` in `hpo_uplift.py:468`
for the three B8 sentinel cases (`default_only`, `tuned_only`,
`paired_but_no_valid_loss`); those footnotes already key off
the `UpliftRow` booleans and that surface is unchanged. The
NEW `no_paired_cells` sentinel string is surfaced via the
shared `render_rollup_skipped_footnote` helper (B14) with
`group_columns=("dataset_name", "model_name")` and
`header_labels=("Dataset", "Model")`. The two footnote sources
are independent (existing B8 footnote for the three pre-
existing cases; new bootstrap-skipped footnote for the new
fourth case only).

The Friedman/Holm block is unchanged.

`render_from_dir(output_root, *, reference_model)`: same
dispatch as B14 (failure-sentinel check → rollup-presence +
freshness → CI variant or std fallback).

## B15.4 CLI dispatch + sentinel writes

`benchmarks/run.py` gains one new wrapper paralleling the B14
wrappers:

- `_run_bootstrap_hpo_uplift_rollup(config, *, env, output_root)`:
  the wrapper body in dispatch order:
  1. Gate A: `if not is_hpo_uplift_rollup_enabled(config):
     return` (closes arch-C1; mirrors the
     `is_pairwise_rollup_enabled` / `is_training_time_rollup_enabled`
     pattern from B14).
  2. Gate B: `if not run_manifest_path(output_root).exists():
     return` (logged).
  3. Gate C: `load_run_manifest` with the standard narrow
     except tuple.
  4. Gate D (closes R-B15-5): the aggregator's first action is
     `if "variant" not in df.columns or not (df["variant"] ==
     "tuned").any(): return []`. This makes
     "manifest exists but contains zero tuned-variant rows"
     OBSERVABLY DISTINCT from "manifest absent" (the wrapper
     returns with no log warning, mirroring the empty-
     manifest path). The aggregator returns `[]` and the
     wrapper does NOT write a sentinel (this is normal
     "B8 was not run" state, not a failure).
  5. Aggregator call; catches `RawRollupError`, deletes any
     partial `bootstrap_hpo_uplift_rollup.parquet`, writes
     `bootstrap_hpo_uplift_aggregator_failed.txt` with
     `type(exc).__name__`. Unlinks stale sentinel on
     successful aggregate. INDEPENDENT from B5 / B6 / B7
     sentinels.

Dispatch order in `run.py:_dispatch_kinds`: the wrapper is
called from the `kind == "hpo_uplift"` branch AFTER
`run_hpo_uplift` returns successfully.

## B15.5 Test surface

`tests/benchmarks/test_bootstrap_hpo_uplift.py` (NEW; 17 named tests, with the R3 qa-I1 amendment to one existing test):

- `test_aggregate_bootstrap_hpo_uplift_rollup_paired_cells_emit_ci`:
  2 seeds × 2 folds × default+tuned for one (dataset, model);
  per-cell default loss 0.50, tuned loss 0.30; assert the
  rollup row has `primary_loss_mean == pytest.approx(0.20)`
  (closes qa-C1 sign convention: positive Δ means tuning
  helped, default - tuned > 0) and `ci_lo <= mean <= ci_hi`.
- `test_aggregate_bootstrap_hpo_uplift_rollup_sign_convention_default_minus_tuned`
  (closes qa-C1 redundant pin): construct a fixture where
  `default_loss=0.5, tuned_loss=0.3` for all cells; assert
  `primary_loss_mean == pytest.approx(0.2)` (positive) and
  fail loudly if an implementation flips the sign.
- `test_aggregate_bootstrap_hpo_uplift_rollup_ci_width_nonzero_with_multiple_cells`:
  anti-degeneracy oracle (B14 qa-C1); 4+ cells with
  non-identical Δ values; assert `ci_hi - ci_lo > 0` STRICT.
- `test_aggregate_bootstrap_hpo_uplift_rollup_single_paired_cell_degenerate`
  (closes qa-C3): monkeypatch `load_run` to return a 1-paired-
  cell manifest; assert the primitive's `n_entities==1`
  degenerate path at `benchmarks/metrics/bootstrap.py:133-136`
  fires (mean == ci_lo == ci_hi == per-cell Δ).
- `test_aggregate_bootstrap_hpo_uplift_rollup_default_only_emits_sentinel`:
  default arm present, tuned arm absent;
  `assert row.bootstrap_skipped_reason == "default_only"`
  (BARE equality; closes qa-C5).
- `test_aggregate_bootstrap_hpo_uplift_rollup_tuned_only_emits_sentinel`:
  mirror; `assert row.bootstrap_skipped_reason == "tuned_only"`.
- `test_aggregate_bootstrap_hpo_uplift_rollup_paired_but_no_valid_loss_emits_sentinel`:
  cells from both arms; every paired row has NaN primary loss;
  `assert row.bootstrap_skipped_reason == "paired_but_no_valid_loss"`.
- `test_aggregate_bootstrap_hpo_uplift_rollup_unpaired_cell_emits_no_paired_cells_sentinel`:
  default at (seed=0, fold=0); tuned at (seed=0, fold=1);
  inner-join → 0 paired cells;
  `assert row.bootstrap_skipped_reason == "no_paired_cells"`.
- `test_aggregate_bootstrap_hpo_uplift_rollup_empty_manifest_returns_empty_list`.
- `test_aggregate_bootstrap_hpo_uplift_rollup_no_tuned_variant_rows_returns_empty_list`
  (closes qa-C4 aggregator side): manifest with only
  `variant=default` rows → aggregator returns `[]` without
  raising; the wrapper's Gate D handles this case as a
  no-op skip.
- `test_aggregate_bootstrap_hpo_uplift_rollup_records_manifest_fingerprint`:
  every emitted row's `manifest_fingerprint` equals the live
  `RunManifest.fingerprint()`.
- `test_aggregate_bootstrap_hpo_uplift_rollup_records_primary_loss_column`:
  `row.primary_loss_column == "log_loss"` for binary cells;
  `"rmse"` for regression_point; the field round-trips
  through parquet (cross-pinned in `test_bootstrap_manifest.py`).
- `test_aggregate_bootstrap_hpo_uplift_rollup_malformed_paired_cell_raises`:
  paired (seed, fold) where one side has a non-None loss and
  the other has NaN with no `skipped_reason` → `RawRollupError`.
- `test_aggregate_bootstrap_hpo_uplift_rollup_nan_cell_with_skipped_reason_emits_sentinel_not_raises`
  (closes R2 qa-I1): the boundary case where a paired cell
  has NaN primary loss AND the underlying B8 group row IS
  flagged `paired_but_no_valid_loss=True`. Pin that the
  group emits a sentinel row (not a raise) AND that
  `row.bootstrap_skipped_reason == "paired_but_no_valid_loss"`
  (bare equality, closing R3 qa-I1) so a transposed
  implementation that emits a no-reason sentinel does not
  pass.
- `test_aggregate_bootstrap_hpo_uplift_rollup_duplicate_seed_fold_pair_raises`
  (closes qa-C2): two rows sharing (seed=0, fold=0) for the
  same (dataset, model, variant) → `RawRollupError`.
- `test_aggregate_bootstrap_hpo_uplift_rollup_oom_gate_raises`:
  monkeypatch `BOOTSTRAP_ROW_COUNT_CEILING` to 1; assert
  `RawRollupError`.
- `test_aggregate_bootstrap_hpo_uplift_rollup_respects_per_spec_n_resamples_override`:
  monkeypatched primitive captures `n_resamples`.

`tests/benchmarks/test_run_bootstrap_hpo_uplift_wrapper.py`
(NEW; 11 named tests, two more than the B14 pattern because
of qa-C4 Gate D coverage):

- `test_run_bootstrap_hpo_uplift_rollup_writes_shard_on_happy_path`
- `test_run_bootstrap_hpo_uplift_rollup_skips_via_opt_out`
  (closes arch-C1 / qa-C4 Gate A coverage: explicit
  `bootstrap_hpo_uplift_enabled=False` exit).
- `test_run_bootstrap_hpo_uplift_rollup_skips_when_b5_manifest_absent`
  (file-level absence).
- `test_run_bootstrap_hpo_uplift_rollup_skips_when_no_tuned_variant_rows_in_manifest`
  (closes qa-C4 Gate D coverage: manifest exists but
  `variant=tuned` is absent. Asserts NO rollup file written
  AND NO sentinel written; the wrapper returns cleanly).
- `test_run_bootstrap_hpo_uplift_rollup_skips_when_run_manifest_absent`
- `test_run_bootstrap_hpo_uplift_rollup_skips_when_run_manifest_load_fails`
- `test_run_bootstrap_hpo_uplift_rollup_catches_raw_rollup_error_and_continues`
- `test_run_bootstrap_hpo_uplift_rollup_writes_failure_sentinel_on_raw_rollup_error`
- `test_run_bootstrap_hpo_uplift_rollup_unlinks_stale_failure_sentinel_on_success`
- `test_run_bootstrap_hpo_uplift_rollup_failure_does_not_touch_b5_b6_b7_sentinels`:
  pre-plant all THREE other sentinels (B5, B6, B7) with
  distinct content strings; assert all three files exist with
  unchanged content after the B15 failure (B14 Stage-3 qa-I3
  closure pattern: read_text comparison, not just exists).
- `test_run_bootstrap_hpo_uplift_rollup_writes_no_paired_cells_sentinel_row_in_shard`
  (closes R2 qa-I2): manifest has both variants but
  default at (seed=0, fold=0); tuned at (seed=0, fold=1);
  inner-join → 0 paired cells. Assert the wrapper writes a
  shard containing the `no_paired_cells` sentinel row AND
  does NOT write a failure sentinel. Pins the end-to-end
  propagation from aggregator through the wrapper.

`tests/benchmarks/test_bootstrap_manifest.py` (extension):
add the standard schema/path tests for the new symbols
(path-format ×2, round-trip ×1, extra-forbid ×1, empty-shard ×1,
absent-on-load ×1) + the ExperimentSpec `bootstrap_hpo_uplift_enabled`
default-True and opt-out tests.

`tests/benchmarks/test_hpo_uplift_report.py` (extension):
renderer-side tests pinning the 4-source footnote precedence
and the render_from_dir dispatch (mirror B14):

- `test_render_hpo_uplift_with_ci_renders_mean_and_interval`
- `test_render_hpo_uplift_without_ci_falls_back_to_scalar`
- `test_render_hpo_uplift_with_ci_drops_old_delta_header_when_ci_present`
  (qa-N pin: assert per-dataset table header carries
  `Δ [95% CI]` and the bare `Δ` header is absent from the
  dataset-block table header row)
- `test_render_hpo_uplift_with_ci_falls_back_on_manifest_fingerprint_mismatch`
- `test_render_hpo_uplift_with_ci_surfaces_aggregator_failed_footnote_on_wrapper_caught`
- `test_render_hpo_uplift_with_ci_manifest_unreadable_appends_freshness_footnote`
- `test_render_hpo_uplift_with_ci_surfaces_no_paired_cells_sentinel_in_rollup_skipped_footnote`:
  pin the ONLY sentinel routed through the shared
  `render_rollup_skipped_footnote` helper. The three
  pre-existing sentinels (`default_only`, `tuned_only`,
  `paired_but_no_valid_loss`) flow through the existing
  `_render_footnote` in `hpo_uplift.py:468` unchanged; they
  are covered by the existing B8 footnote tests.
- `test_render_hpo_uplift_with_ci_marks_partial_fold_cell_with_asterisk`:
  fixture with `n_seeds=2, n_folds=3, n_cells_paired=4`;
  assert the rendered CI cell ends with `*`.
- `test_render_hpo_uplift_with_ci_surfaces_skipped_cells_footnote_when_n_skipped_cells_nonzero`
  (closes R2 qa-C1): fixture with `n_skipped_cells=2,
  n_cells_paired=4, n_seeds=2, n_folds=3`; assert the
  rendered Markdown contains the "Partial coverage" footnote
  with the `n_skipped_cells / n_cells_paired` ratio for
  that row. Pin source 5 of the footnote precedence list.
- `test_render_hpo_uplift_with_ci_surfaces_skipped_cells_footnote_without_asterisk_when_no_pairing_asymmetry`
  (closes R3 qa-I3): fixture with
  `n_seeds=2, n_folds=2, n_cells_paired=4, n_skipped_cells=1`;
  assert the CI cell does NOT end with `*` (no asymmetry)
  AND the "Partial coverage" footnote IS present (loss
  dropout). Pins source 5 independence from the asterisk.
- `test_render_hpo_uplift_with_ci_surfaces_both_freshness_and_skipped_cells_footnotes_when_both_apply`
  (closes R3 qa-I2): fixture with `manifest_unreadable=True`
  AND `n_skipped_cells=1`; assert BOTH the
  "Bootstrap freshness check skipped" footnote AND the
  "Partial coverage" footnote appear. Pins source 5 vs
  source 4 co-occurrence (the CI body still renders;
  source 4 appends freshness footnote; source 5 appends
  partial-coverage footnote).
- `test_render_hpo_uplift_with_ci_renders_no_ci_when_rollup_missing_for_row`:
  manifest has a (dataset, model) pair but the rollup has no
  matching row; CI cell renders as `(no CI)` and the rest of
  the table is unaffected.

render_from_dir × 6 dispatch tests:
- `test_render_from_dir_falls_back_silently_when_rollup_file_absent`
- `test_render_from_dir_surfaces_aggregator_failed_sentinel`
- `test_render_from_dir_renders_ci_when_rollup_present_and_fingerprint_matches`
- `test_render_from_dir_falls_back_on_stale_rollup`
- `test_render_from_dir_freshness_check_skipped_when_manifest_corrupt`
- `test_render_from_dir_renders_ci_without_freshness_footnote_when_manifest_absent`

## B15.6 Estimated effort

| Module | Size |
|---|---|
| `benchmarks/bootstrap_manifest.py` (+1 schema + 2 path helpers + 2 I/O helpers) | Small |
| `benchmarks/config.py` (+1 ExperimentSpec field) | Trivial |
| `benchmarks/report/bootstrap_hpo_uplift.py` (NEW) | ~330 lines |
| `benchmarks/report/hpo_uplift.py` (+CI variant + dispatch) | ~280 lines |
| `benchmarks/run.py` (+1 wrapper) | ~80 lines |
| `tests/benchmarks/test_bootstrap_hpo_uplift.py` (NEW) | ~400 lines |
| `tests/benchmarks/test_run_bootstrap_hpo_uplift_wrapper.py` (NEW) | ~280 lines |
| `tests/benchmarks/test_bootstrap_manifest.py` (extension) | ~80 lines |
| `tests/benchmarks/test_hpo_uplift_report.py` (extension) | ~350 lines |

Total: ~1,800 lines. Smaller than B14 because the shared
helpers and the primitive are now stable.

## Risk register (delta)

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R-B15-1 | The PAIRING constraint at inner-join time could silently drop unpaired (seed, fold) cells, understating the bootstrap sample size. | Medium | `n_cells_paired` is recorded on every emitted row; the rollup-skipped footnote surfaces the `no_paired_cells` sentinel. Test pin: `_unpaired_cell_dropped_silently`. |
| R-B15-2 | The per-cell Δ can be NaN if either side has a missing primary loss; the existing B8 aggregator already produces `paired_but_no_valid_loss` in that case but the CI aggregator must mirror the same logic. | Medium | Test pin: `_paired_but_no_valid_loss_emits_sentinel`. |
| R-B15-3 | The Friedman/Holm matrix block is UNCHANGED; a future reviewer may expect the CI to flow through to the Holm-adjusted pairwise too (i.e. CI on Wilcoxon's δ). | Low | Explicitly out of scope at v1 (D-B15.1 follow-up). The Wilcoxon paired CI is a distinct statistical procedure. |
| R-B15-4 | The new `primary_loss_column` audit field on the schema sets a precedent that B14 RollupRows don't have, inviting future drift. | Low | Documented as an HPO-specific audit need; the B14 rollups carry `primary_metric` only because B5/B6/B7 each have a fixed metric per task type. |
| R-B15-5 | The wrapper attaches to `run_hpo_uplift` which itself produces the tuned-variant manifest rows; gating on B5 manifest existence vs B8 driver-completion is a subtle ordering choice. | Medium | Gate explicitly on the existence of at least one `variant=tuned` row in the B5 manifest, NOT on `run_hpo_uplift` having succeeded structurally. A driver-completion check would conflate "B8 produced rows" with "B8 was configured and ran". |

## B15-followup deferrals (out of scope for v1)

- D-B15.1: CI on the Holm-adjusted Wilcoxon δ (pairwise model
  comparison across datasets). Distinct from the per-(dataset,
  model) Δ CI shipped here.
- D-B15.2: BCa CI (inherited from D-B13.5).
- D-B15.3: per-fold CI (inherited from D-B13.6).
- D-B15.4: per-entity sufficient-statistics OOM optimization
  (inherited from D-B13.7).
- D-B15.5: coordinated `primary_loss_*` → `primary_metric_*`
  rename across all four RollupRow schemas (inherited from
  D-B14.6).

## Addressed

R1 swarm: architecture-reviewer (1C/4I/2N REQUEST_CHANGES),
qa-test-coverage (5C/5I/3N REQUEST_CHANGES), style-reviewer
(0C/0I/0N APPROVE). Total deduped: 6 CRITICAL, 9
IMPROVEMENT, 5 NITPICK. CRITICALs addressed:

- **arch-C1** (B15.4 wrapper omitted opt-out gate predicate):
  B15.0 now declares `is_hpo_uplift_rollup_enabled(config)`
  alongside `hpo_uplift_rollup_output_path()`; B15.4 names
  the predicate as Gate A of the wrapper body.
- **qa-C1** (sign convention undetected): B15.5 adds
  `test_aggregate_bootstrap_hpo_uplift_rollup_sign_convention_default_minus_tuned`
  with `default_loss=0.5, tuned_loss=0.3` fixture asserting
  `primary_loss_mean == pytest.approx(0.2)`. The
  `_paired_cells_emit_ci` test was also tightened to assert
  the exact Δ value, so a sign-flip implementation now fails
  both tests.
- **qa-C2** (duplicate (seed, fold) pair raise unnamed):
  B15.5 adds
  `test_aggregate_bootstrap_hpo_uplift_rollup_duplicate_seed_fold_pair_raises`
  pinning the `RawRollupError` raise.
- **qa-C3** (single-entity-degenerate test missing): B15.5
  adds
  `test_aggregate_bootstrap_hpo_uplift_rollup_single_paired_cell_degenerate`
  monkeypatching `load_run` to return a 1-paired-cell
  manifest; pins the primitive's `n_entities==1` path.
- **qa-C4** (tuned-variant gate test missing): the wrapper's
  Gate D ("manifest exists but `variant=tuned` is absent"
  short-circuits to no-op skip) is now explicit in B15.4,
  AND B15.5 adds
  `test_run_bootstrap_hpo_uplift_rollup_skips_when_no_tuned_variant_rows_in_manifest`
  (wrapper level) PLUS
  `test_aggregate_bootstrap_hpo_uplift_rollup_no_tuned_variant_rows_returns_empty_list`
  (aggregator level) so the contract is pinned on both
  sides.
- **qa-C5** (sentinel string equality assertions
  underspecified): B15.5 sentinel tests now mandate
  `assert row.bootstrap_skipped_reason == "<exact string>"`
  BARE equality (not `in` or `startswith`) for all four
  sentinel strings.

IMPROVEMENTs addressed:

- **arch-I1** (renderer footnote source split): B15.3 now
  explicitly states the CI variant reuses the existing
  `_render_footnote` for the three pre-existing B8 sentinel
  cases (`default_only`, `tuned_only`,
  `paired_but_no_valid_loss`) and uses the shared
  `render_rollup_skipped_footnote` ONLY for the new
  `no_paired_cells` sentinel. The two footnote surfaces
  are independent. B15.5 names this dispatch in
  `test_render_hpo_uplift_with_ci_surfaces_no_paired_cells_sentinel_in_rollup_skipped_footnote`.
- **arch-I2** (`no_paired_cells` fourth sentinel string not
  enumerated): B15.0 now declares all four sentinel strings
  in a dedicated subsection with each case's trigger.
- **arch-I4** (partial-fold denominator masked asymmetric
  coverage): the schema now carries `n_seeds`, `n_folds`,
  AND `n_cells_paired`; B15.3 defines the asterisk as
  `n_cells_paired < n_seeds * n_folds` (the B14 convention)
  and surfaces `n_skipped_cells` as a secondary "partial
  coverage" footnote when loss-side dropouts occurred.
- **qa-I1** (`no_paired_cells` string anchor):
  `_unpaired_cell_emits_no_paired_cells_sentinel` now
  asserts the EXACT string equality.
- **qa-I3** (`primary_loss_column` round-trip): the
  `test_bootstrap_manifest.py` extension explicitly names
  `primary_loss_column` in the round-trip assertions.
- **qa-I4** (sentinel-renderer footnote coverage per
  string): the per-string renderer test is now named
  explicitly in the renderer test list above.

Deferred:

- **arch-I3** (`primary_loss_column` precedent asymmetry):
  rationale rewritten as "audit field surfaced in the
  per-dataset block header at `hpo_uplift.py:441` that the
  CI variant must echo, so the field is part of the
  rendering contract, not a generic audit add." A
  coordinated lift to all four B14 schemas is D-B15.5 in
  the followup list (already named with the
  `primary_loss_*` rename).
- **arch-N1** (Row-count drift framing): rewritten to
  "Duplicate (seed, fold) pair" in B15.2 failure-modes per
  the B14 R2 closure pattern.
- **arch-N2** (test enumeration count): B15.5 now opens with
  "17 named tests" (test_bootstrap_hpo_uplift.py) and
  "11 named tests" (test_run_bootstrap_hpo_uplift_wrapper.py)
  so the count is countable inline. Counts include the R2 +
  R3 closure additions.
- **qa-I2** (freshness fingerprint footnote string assertion):
  the renderer's `falls_back_on_manifest_fingerprint_mismatch`
  test description in B15.5 now reads as the B14-mirror
  shape, where the dispatch test asserts the "Bootstrap
  rollup is stale" footnote string verbatim.
- **qa-I5** (skip condition disambiguation): Gate D in B15.4
  is now explicit about content-vs-file-presence.
- **qa-N1** (partial-fold fixture concrete shape): the
  renderer test now names the exact fixture
  `n_seeds=2, n_folds=3, n_cells_paired=4` to make the
  partial=True path unambiguous.
- **qa-N2** (cross-report isolation reads content): the
  wrapper test now mirrors the B14 Stage-3 qa-I3 closure
  pattern (read_text assertion on each pre-planted sentinel).
- **qa-N3** (`regression_quantile` primary_loss_column): B8
  early-skips `regression_quantile` cells via the
  `regression_quantile_b5_followup` sentinel, so the rollup
  emits no row for that group. The
  `_records_primary_loss_column` test description above
  covers the binary + regression_point cases only;
  regression_quantile is handled by the upstream B8 skip
  and tested at the B8 layer.

### R2 swarm closure

R2 confirming swarm: architecture-reviewer (0C/2I/2N APPROVE),
qa-test-coverage (1C/2I/1N REQUEST_CHANGES), style-reviewer
(0C/0I/0N APPROVE). Total: 1 CRITICAL, 4 IMPROVEMENT, 3
NITPICK. Closures:

- **qa-R2-C1** (`n_skipped_cells > 0` secondary footnote
  untested): B15.3 footnote precedence list now enumerates
  source 5 (the loss-dropout disclosure footnote); B15.5 adds
  `test_render_hpo_uplift_with_ci_surfaces_skipped_cells_footnote_when_n_skipped_cells_nonzero`
  pinning the new source.
- **arch-R2-I1** (`n_skipped_cells` secondary footnote not
  in precedence list): folded into the precedence list as
  source 5.
- **arch-R2-I2** (malformed-paired-cell grain mismatch
  ambiguity): B15.2 failure-modes block rewritten to spell
  out the cell-grain vs group-grain distinction. The "group
  says OK but cell is NaN" case raises; the "group says
  paired_but_no_valid_loss=True and cell is NaN" case
  emits a sentinel.
- **qa-R2-I1** (malformed-paired-cell boundary at
  `skipped_reason` presence): B15.5 adds
  `test_aggregate_bootstrap_hpo_uplift_rollup_nan_cell_with_skipped_reason_emits_sentinel_not_raises`
  pinning the non-raise arm.
- **qa-R2-I2** (`no_paired_cells` wrapper-level propagation):
  B15.5 adds
  `test_run_bootstrap_hpo_uplift_rollup_writes_no_paired_cells_sentinel_row_in_shard`
  exercising the end-to-end aggregator → wrapper → shard
  path.
- **arch-R2-N1** (aggregator pseudocode omits no_paired_cells):
  the new "Sentinel emission policy (no raise)" block in
  B15.2 enumerates all four sentinel cases including
  `no_paired_cells`.
- **arch-R2-N2** (Gate D conflated wrapper + aggregator):
  the wrapper outline at B15.4 now reads Gate A-C as
  wrapper-body steps and Gate D as the aggregator's early-
  return contract; the wrapper observes `[]` and skips.
- **qa-R2-N1** (`render_from_dir` test for no_paired_cells):
  the lower-level renderer test
  (`_surfaces_no_paired_cells_sentinel_in_rollup_skipped_footnote`)
  covers the rendering path; the dispatch-level coverage is
  considered incremental and not added at this stage.

## Deferred

R1 swarm deferrals carried forward (each documented above
with rationale):

- **arch-I3**: `primary_loss_column` precedent asymmetry will
  be unified via D-B15.5 (coordinated `primary_loss_*` rename
  across all four schemas), itself a follow-up to D-B14.6.
- **qa-N3**: regression_quantile groups produce no rollup row
  by upstream B8 design; the primary_loss_column field is
  exercised on binary + regression_point cases only.
- **R2 qa-N1**: `render_from_dir` dispatch test for the
  `no_paired_cells` routing through
  `render_rollup_skipped_footnote`. The lower-level renderer
  test pins the rendering logic; the dispatch wrapper is
  inferentially covered by the six dispatch tests already
  named.

### R3 swarm closure

R3 confirming swarm: architecture-reviewer (0C/0I/1N APPROVE),
qa-test-coverage (0C/3I/1N APPROVE), style-reviewer
(0C/0I/0N APPROVE). Total: 0 CRITICAL, 3 IMPROVEMENT, 2
NITPICK. Closures:

- **qa-R3-I1** (sentinel reason assertion missing on the
  R2-added boundary test): the test description now
  mandates `assert row.bootstrap_skipped_reason ==
  "paired_but_no_valid_loss"` BARE equality.
- **qa-R3-I2** (source 5 vs source 4 footnote co-occurrence
  untested): B15.5 adds
  `_surfaces_both_freshness_and_skipped_cells_footnotes_when_both_apply`
  exercising `manifest_unreadable=True` AND
  `n_skipped_cells=1` simultaneously.
- **qa-R3-I3** (source 5 footnote independence from
  asterisk): B15.5 adds
  `_surfaces_skipped_cells_footnote_without_asterisk_when_no_pairing_asymmetry`
  exercising `n_cells_paired == n_seeds * n_folds` AND
  `n_skipped_cells > 0` (no asterisk; footnote fires).
- **arch-R3-N1** (stale count "16/10" in R1-arch-N2
  deferral entry): bumped to "17/11" to reflect R2 + R3
  additions.
- **qa-R3-N1** (R2 qa-N1 carried forward): unchanged.

**Consensus reached after R3.**
