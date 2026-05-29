# B13 design delta: entity-block bootstrap CIs (B5.4)

Phase B13 delta against `benchmark_suite_design.md` (B5.4 +
B6.2.1 predictions-shard schema) and
`benchmark_suite_implementation_plan.md`. B13 implements the
B5.4 bootstrap CI primitive that the metrics module references
but does not deliver, and that the Gemini final-pass on both B11
and B12 explicitly called out as a load-bearing correctness gap
("panel data carries intra-entity correlation; B5.4 bootstrap
design must resample by entity for ALL adapter families on
panel data").

This is the minimal v1 implementation: bootstrap aggregator +
new `RollupRow` record + B5 leaderboard renderer integration.
The B6 / B7 / B8 / B11 renderer integrations are documented as
B13-followups; this delta does not block on them.

## Requirements (grading rubric)

- **R1** Entity-block (NOT row) bootstrap: the bootstrap unit
  of resampling is the per-fold ENTITY, not the per-fold ROW.
  Panel data carries intra-entity correlation; row resampling
  would report CIs sqrt(K) tighter than the truth (K = rows per
  entity). This is non-negotiable: it is the gap the B11/B12
  Gemini passes both flagged.
- **R2** Per-(dataset, model) aggregation across seeds AND
  folds: the design says "aggregated across seeds as mean and a
  percentile bootstrap 95% interval (10000 resamples)". The
  v1 aggregator runs the bootstrap on the FULL per-cell loss
  table (every test row across every (seed, fold)), grouped by
  entity. The output is per-(dataset, model, task_type) with
  `mean ± [ci_lo, ci_hi]`. The rejected alternative is
  per-fold bootstrap then average the CIs across folds; pool-
  then-bootstrap is chosen because (a) the folds are
  non-overlapping time slices on the same entities so variance
  pooling is appropriate, (b) per-fold CIs would need a
  separate CI-aggregation rule the design has not pinned, and
  (c) higher resampling power surfaces real cross-fold
  variation as wider CI width.
  Seed axis: per the B5 splitter contract, the SAME test rows
  appear in each (seed, fold) cell for a fixed fold (the
  splitter is seed-independent for fold structure; the seed
  varies only model training, not the split). The pooled
  vector for entity E thus contains E's test rows once per
  seed, each row carrying a DIFFERENT per-row loss (because
  the seed-varied models produce different predictions). The
  bootstrap's "mean across resampled entity rows" is therefore
  the mean of (per-row loss across seeds and test-fold rows
  for E), which is statistically equivalent to averaging the
  per-seed cell losses for E. Pool-then-bootstrap is chosen
  (vs per-seed average then bootstrap) because it preserves
  the within-seed variation that the per-seed average would
  collapse; the CI is therefore SLIGHTLY WIDER and more
  honest about the cross-seed uncertainty.
- **R3** Fixed bootstrap seed recorded on every rollup row.
  The design names this explicitly. v1 uses a default seed
  (`_BOOTSTRAP_SEED = 0x_B13_5EED_B007`) and the seed value is
  written on every emitted RollupRow.
- **R4** `n_resamples = 10000` per the design. v1 fixes this
  as a module constant `_BOOTSTRAP_N_RESAMPLES_BY_PROFILE = {
  "smoke": 5_000, "standard": 10_000, "full": 10_000}`. The
  per-profile value is read at `_run_bootstrap_rollup` time
  via `env.profile`; no `BenchmarkConfig` field is added for
  this. An ExperimentSpec-level override
  `ExperimentSpec(kind="raw_loss", bootstrap_n_resamples=...)`
  is the per-experiment escape hatch (see also R6 below for the
  opt-out granularity).
- **R5** Percentile CI (NOT BCa, NOT pivot): the design says
  "percentile bootstrap 95% interval". The 2.5 / 97.5
  percentiles of the resampled means are the v1 CI shape. BCa
  is a B13-followup if a smoke-tier baseline surfaces
  bias-correction need.
- **R6** Schema discipline: NO change to the existing
  `ResultRow` schema. NO change to the predictions-shard
  schema. B13 adds a NEW `RollupRow` record + a NEW
  `bootstrap_rollup.parquet` shard alongside the existing
  manifest. The B9 manifest fingerprint extends to include the
  rollup file.
  Opt-out granularity: `ExperimentSpec(kind="raw_loss",
  bootstrap_rollup_enabled: bool = True)`. The toggle lives
  per-experiment, NOT on `BenchmarkConfig`, so a config that
  runs `raw_loss + hpo_uplift` together can disable the rollup
  on `raw_loss` while running `hpo_uplift` unchanged. The CLI
  dispatch gate at `benchmarks/run.py:_dispatch_kinds` consults
  the `ExperimentSpec` field; the rollup step runs only when
  the field is `True` AND the raw_loss step completed.
- **R7** Entity-id resolution: the predictions-shard schema
  carries `panel_row_index` but NOT `entity_id`. The bootstrap
  aggregator RE-RESOLVES `entity_id` at aggregation time by
  re-loading the dataset via `benchmarks.registry.get_loader`
  and indexing the panel by `panel_row_index`. To make this
  safe against future loader variations, B13 pins TWO defenses:
  (1) the EXPLICIT LOADER-DETERMINISM CONTRACT: every
  registered loader MUST produce row-identical output across
  calls (same row count, same order, same `(entity_col,
  time_col)` per row). A new conformance test
  `test_loader_row_order_is_deterministic_across_calls`
  asserts this for every registered loader. (2) The aggregator
  performs a DEFENSIVE post-load sort by `(entity_col,
  time_col)` BEFORE indexing by `panel_row_index`, and asserts
  the sorted panel's `len()` matches `max(panel_row_index) + 1`
  emitted by B5; mismatch raises `RawRollupError("loader
  row-count drift")`. The loader has already been called during
  the B5 run (the panel is cached on disk when `cache_dir` is
  set); the aggregator's re-load is structurally identical
  modulo the explicit re-sort. This avoids a schema bump on
  every shipped predictions shard.
- **R8** Skipped-cell handling: skipped cells emit a sentinel
  RollupRow with `mean / ci_lo / ci_hi == None` and
  `n_skipped_cells` populated. The B5 renderer's existing
  skipped-cell footnote machinery picks them up via the same
  `n_skipped > 0` predicate it already uses for the scalar
  leaderboard.
- **R9** B5 leaderboard renderer integration: the renderer
  joins the rollup table on `(dataset_name, model_name,
  task_type)` and renders `mean [ci_lo, ci_hi]` for the
  primary loss column. The existing `LeaderboardEntry`
  pydantic model + its `mean ± std` renderer are RETAINED as
  the fallback path; a new `LeaderboardEntryWithCI` ships
  alongside. The renderer dispatches by rollup presence: when
  `bootstrap_rollup.parquet` exists at `output_root`, the CI
  variant is rendered; when absent (e.g., the smoke-tier
  `skip_bootstrap_rollup=True` config, or a partial run that
  crashed between B5 + B13), the std variant renders
  unchanged. The std variant is officially deprecated; the
  R-B13-4 risk register entry names the reader-migration note.
  This coexistence keeps R6 (no schema removal) and R9 (CI
  shipped) consistent.

## Out of scope (B13-followup deferrals)

- **D-B13.1** B6 (ensemble pairwise) CI integration: the
  pairwise correlation statistics need their own bootstrap; the
  shape is different from a per-cell metric CI.
- **D-B13.2** B7 (training-time) CI: the wall-clock cell metric
  is inherently per-cell, not per-row, so the entity-block
  bootstrap is moot. A separate cross-fold bootstrap on the
  wall-clock distribution is a B7-followup.
- **D-B13.3** B8 (HPO-uplift) CI: the Δ statistic needs a
  paired bootstrap. B13 ships the unpaired primitive; the
  paired variant extends it.
- **D-B13.4** B11 (ensemble-lift) CI integration: the
  Wilcoxon block already conveys uncertainty across datasets.
  Adding per-dataset Δ CIs is a B11-followup, scaffolded by
  the B13 entity-block primitive.
- **D-B13.5** BCa CI: percentile only at v1 (R5).
- **D-B13.6** Per-fold CIs: v1 aggregates across folds + seeds
  for a single CI per (dataset, model). Per-fold CIs are a
  followup if a reader needs fold-level uncertainty.
- **D-B13.7** Per-entity sufficient-statistics optimization:
  the naive bootstrap concatenates resampled-entity rows per
  resample, which has O(N) memory traffic per resample. For
  the two v1 metric_fns (`np.nanmean` for classification,
  `lambda x: sqrt(nanmean(x))` for regression), the metric is
  expressible from per-entity `(sum_loss, count_rows)`
  sufficient statistics, reducing memory traffic to O(E) per
  resample (E = number of entities). For the full-tier Amex
  dataset with ~500k customers vs 6M rows, this is a 12x
  memory + 12x throughput improvement. The naive path ships at
  v1; the optimization is a B13-followup once a full-tier
  dataset surfaces the R-B13-3 ceiling. The optimization is
  correctness-preserving for v1 metric_fns (provable: the
  bootstrap mean = sum(resampled_sums) / sum(resampled_counts)
  is identical to the mean of the concatenated rows); a custom
  `metric_fn` that isn't expressible from sufficient statistics
  (e.g., median, ROC-AUC) needs the naive path.

## Architecture

### B13.1 New module: `benchmarks/metrics/bootstrap.py`

Owns the entity-block bootstrap primitive. Pure function: takes
loss arrays + an entity-id array + `(n_resamples, seed,
metric_fn)`, returns `(mean, ci_lo, ci_hi)`.

```python
def entity_block_bootstrap_ci(
    losses: np.ndarray,
    entity_ids: np.ndarray,
    *,
    n_resamples: int = _BOOTSTRAP_N_RESAMPLES,
    confidence: float = 0.95,
    seed: int = _BOOTSTRAP_SEED,
    metric_fn: Callable[[np.ndarray], float] = _default_metric_fn,
) -> tuple[float, float, float]: ...
```

- `losses` and `entity_ids` are 1-D arrays of equal length.
  The primitive sets `losses.flags.writeable = False` and
  `entity_ids.flags.writeable = False` at entry to enforce the
  pure-function contract; a `metric_fn` that attempts in-place
  mutation raises immediately rather than silently corrupting
  the next resample's view.
- The bootstrap resamples UNIQUE entity ids with replacement
  (NOT rows). For each resample, the resampled-entity rows are
  concatenated and `metric_fn` is applied to the concatenated
  per-row loss vector to produce ONE scalar.
- `metric_fn` default is `np.nanmean` (the classification path).
  For RMSE-flavor regression, the aggregator passes
  `lambda x: float(np.sqrt(np.nanmean(x)))` so the sqrt is
  applied INSIDE each resample, not to the percentile of the
  MSE bootstrap. This closes the Jensen's-inequality gap: the
  percentile of resampled RMSE values is the correct RMSE CI;
  `sqrt(percentile_of_MSE)` would be biased because `sqrt` is
  concave.
- Returns `(mean, ci_lo, ci_hi)` where `mean` is `metric_fn`
  applied to the unresampled vector, and the CI is the percentile
  interval over the 10k `metric_fn` values from resamples. The
  percentile call uses `np.percentile(..., method="linear")`
  EXPLICITLY (not the default-which-may-change). The default is
  also `"linear"` at NumPy 2.x but the pin defends against
  future default-method drift.
- Uses `np.random.Generator(np.random.PCG64(seed))` for
  reproducibility (NOT `default_rng` which may swap the
  generator algorithm in a future NumPy major release).
  `RollupRow` records both `bootstrap_seed` and
  `bootstrap_rng_algorithm="PCG64"` so a future audit can pin
  whether output drift came from a generator-algo change.

The primitive intentionally takes pre-computed PER-ROW losses
rather than `(y_true, y_pred)` so it works uniformly for
classification (`-log p(y_true)` per row, `metric_fn=nanmean`),
regression (`(y - y_pred)**2` per row, `metric_fn=sqrt(nanmean(.))`),
and any future per-row scalar (the caller supplies the
aggregation rule).

### B13.0 New typed surface declarations

Three new surfaces are introduced by the Gemini-C2 + Gemini-C3
fixes and the R1 code-review wire-up: two pydantic symbols and
one filesystem-sentinel filename. All three are declared here
so the rest of B13 can reference them unambiguously.

`benchmarks/report/bootstrap_rollup.py:RawRollupError(RuntimeError)`:
typed aggregator failure. Mirrors the existing
`RawMTSError(RuntimeError)` pattern at
`benchmarks/protocol/raw_mts.py`. Subclasses `RuntimeError` for
two reasons:
- **Defensive driver-catch future-proofing**: if the
  aggregator is ever invoked from inside the B5/B8 driver
  (e.g., a future inline-CI variant), the existing narrow
  tuple `(RuntimeError, MemoryError, NotFittedError,
  ProbaUnsupportedError, QuantilesUnsupportedError)` routes
  it as a typed `adapter_error` skip rather than crashing.
- **Explicit CLI wrapper at the v1 call site**: the v1
  aggregator runs at REPORT time (post-driver) from
  `benchmarks/run.py:_run_bootstrap_rollup`. The CLI's
  existing top-level catch at `benchmarks/run.py:427` only
  traps `ValueError`. B13 adds a NEW
  `try/except RawRollupError as exc` wrapper around the
  `_run_bootstrap_rollup` call site that LOGS the failure,
  DELETES any half-written `bootstrap_rollup.parquet` (so the
  renderer doesn't pick up a partial file), and RETURNS the
  exit code 0 (the run succeeded; only the CI rollup
  degraded). The leaderboard then falls back to the std
  variant + emits a "Bootstrap aggregator failed" footnote
  with the exception class name. Without this wrapper, the
  CLI would crash with an uncaught `RuntimeError` and the std
  fallback would be unreachable.

`benchmarks/run.py + benchmarks/report/raw_loss.py:
<output_root>/bootstrap_aggregator_failed.txt`: a one-line
filesystem sentinel that decouples the CLI wrapper from the
renderer. Cross-module FS contract (R2 arch-C1 close): the
filename is declared here so a future maintainer changing
either side doesn't silently break the contract.

- **Producer**: `benchmarks/run.py:_run_bootstrap_rollup`, in
  the `except RawRollupError as exc` block, writes the file
  AFTER unlinking any partial `bootstrap_rollup.parquet`.
- **Content**: `type(exc).__name__` (the exception class as a
  Python identifier), stripped, UTF-8. No surrounding JSON or
  YAML; one line of bare text. The renderer trusts the
  content but embeds it in backtick-quoted Markdown so any
  arbitrary text fails closed at the display layer.
- **Consumer**: `benchmarks/report/raw_loss.py:render_from_dir`,
  checked BEFORE the rollup-presence dispatch. When the
  sentinel exists, the renderer routes to the std variant +
  emits a `### Bootstrap aggregator failed` footnote naming
  the exception class.
- **Lifecycle**: the producer writes on every aggregator
  failure; the consumer never deletes it. A subsequent
  successful run that produces a valid rollup AND leaves the
  sentinel in place will surface the std variant + footnote
  (the sentinel check fires first); the run.py wrapper SHOULD
  unlink the sentinel on a successful aggregate call to avoid
  this stickiness. v1 ships without the cleanup; a v1.1
  followup adds it.

The two named raise sites:
- Loader row-count drift (R7 defensive sort).
- Row-count ceiling exceeded (R-B13-3 OOM gate).

`benchmarks/run_manifest.py:RunManifest.fingerprint() -> str`:
NEW method on the existing `RunManifest` pydantic model at
`benchmarks/run_manifest.py:102`. Returns a SHA-256 hex digest
computed from a canonical JSON serialization of the manifest's
reproducibility-relevant fields (the existing
`library_git_sha`, `run_id`, profile, hardware_tier,
environment fingerprint dict, dataset SHAs, model specs). The
canonical serialization uses `json.dumps(..., sort_keys=True,
separators=(",", ":"))` (NumPy-version-independent + insertion-
order-independent); B13 ships this serialization inline on
`RunManifest.fingerprint()` rather than reusing
`benchmarks/protocol/fingerprint.py:fingerprint_folds` whose
input shape (`tuple[tuple[int, int], ...]`) is fold-specific.
The two fingerprints are NOT the same surface (split
fingerprint covers the fold layout per (dataset, seed); the
manifest fingerprint covers the whole-run reproducibility
surface). `RollupRow.manifest_fingerprint` records the value
at aggregation time so the renderer can detect a stale rollup
file from a prior run.

Schema-version concern: the SHA changes whenever the manifest
schema changes (e.g., a future pydantic field addition). This
is by design (a schema bump IS a reproducibility-relevant
event), but it means a B13 rollup file becomes
fingerprint-stale across a schema bump even though its content
is otherwise valid. The CHANGELOG entry for any
`RunManifest` schema change names this side effect; the
renderer's fingerprint-mismatch fallback handles it cleanly
(std variant + footnote).

The B13 implementation phase MUST land BOTH declarations
(`RawRollupError` in `bootstrap_rollup.py` + `fingerprint()`
method on `run_manifest.py:RunManifest`) plus the CLI wrapper
in `benchmarks/run.py` BEFORE the aggregator code. The
implementation plan's B13 actual-shape section (TBD; written
post-merge alongside the build commit, matching the B11 + B12
pattern) will enumerate these as the first three work items.

### B13.2 New aggregator: `benchmarks/report/bootstrap_rollup.py`

Reads the B5 manifest + the per-cell predictions shards,
re-resolves `entity_id` for each row via the registered
dataset loader, computes per-row losses, calls
`entity_block_bootstrap_ci`, and emits a per-(dataset, model,
task_type) `RollupRow`. The output is written as a
`bootstrap_rollup.parquet` shard via the same atomic-write
machinery as the predictions shards.

Skipped cells are aggregated separately: a (dataset, model)
group with ZERO ok cells emits a sentinel `RollupRow` with
`mean / ci_lo / ci_hi = None` and `n_skipped_cells` populated.

```python
class RollupRow(BaseModel):
    """One per-(dataset, model, task_type) bootstrap CI entry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_name: str
    model_name: str
    task_type: str
    primary_metric: str  # "log_loss" | "rmse"
    n_seeds: int
    n_cells_evaluated: int  # ok cells included in the bootstrap
    n_skipped_cells: int
    n_rows: int  # total rows across all included cells
    n_entities: int  # unique entities across all included cells
    primary_loss_mean: float | None
    primary_loss_ci_lo: float | None
    primary_loss_ci_hi: float | None
    bootstrap_seed: int
    bootstrap_n_resamples: int
    bootstrap_confidence: float = 0.95
    bootstrap_rng_algorithm: str = "PCG64"  # explicit, NOT default_rng
    bootstrap_numpy_version: str  # e.g. "2.3.0"; pinned alongside the algo
    bootstrap_skipped_reason: str | None = None  # loader_failed sentinel
    # Gemini-C3: the manifest_fingerprint at aggregation time. The
    # renderer asserts rollup.manifest_fingerprint == manifest.
    # fingerprint() before joining. A stale rollup from a prior run
    # (e.g., B5 finished but the rollup step was interrupted before
    # the next B5 ran) thus fails the freshness check and the renderer
    # falls back to the std variant + emits a warning footnote.
    manifest_fingerprint: str
```

`aggregate_bootstrap_rollup(manifest, output_root, env) -> list[RollupRow]`:

1. Group manifest by `(dataset_name, model_name, task_type)`.
2. For each group, filter to OK cells (`skipped_reason is None`).
3. For each OK cell, load its predictions shard via
   `benchmarks.predictions.load_predictions`.
4. Re-load the dataset panel via `get_loader(dataset_name)(cache_dir)`.
5. Build a per-row `entity_id` array by indexing
   `panel.iloc[predictions["panel_row_index"]][spec.entity_col]`.
6. Compute per-row losses:
   - Classification: reuse
     `benchmarks.metrics.pairwise.classification_nll(y_true,
     y_proba, labels)` (already clips at `eps=1e-15`) so the
     NLL constant lives in one place. The bootstrap primitive
     is called with `metric_fn=np.nanmean` (default).
   - Regression: `(y_true - y_pred) ** 2` per row (squared
     error, NOT RMSE). The bootstrap primitive is called with
     `metric_fn=lambda x: float(np.sqrt(np.nanmean(x)))` so the
     sqrt is applied INSIDE each resample, producing the
     correct RMSE CI per R5. Per arch-C1 / qa-C2, applying
     sqrt to the percentile of the MSE CI would be biased
     (Jensen's inequality; sqrt is concave).
7. Concatenate per-cell losses + entity ids into the group's
   total arrays.
8. Call `entity_block_bootstrap_ci` with the per-task
   `metric_fn` (classification: `np.nanmean`, regression:
   `sqrt(nanmean(.))`). The returned `(mean, ci_lo, ci_hi)` is
   ALREADY in the user-facing units (log_loss for
   classification, RMSE for regression); no post-bootstrap
   transform is required.
9. Emit a `RollupRow`.

Failure modes:
- A predictions shard is missing (B5 wrote the cell but the
  shard parquet is missing): the cell is dropped from the
  rollup; `n_cells_evaluated` excludes it. Logged at WARNING.
- The dataset loader raises: the entire (dataset, model)
  rollup row emits as a sentinel with
  `bootstrap_skipped_reason="loader_failed: <type>: <msg>"`.
  This is a NEW skipped_reason CONSTANT shared with the B5
  driver's skip-reason set so the renderer footnote machinery
  picks it up uniformly.

### B13.3 New write/load helpers: `benchmarks/bootstrap_manifest.py`

Mirrors `benchmarks/manifest.py` but for the rollup shard. One
shard per run, written atomically. Trivial surface:

- `write_rollup(root: Path, rows: Sequence[RollupRow]) -> None`
- `load_rollup(root: Path) -> pd.DataFrame`
- `rollup_path(root: Path) -> Path` (`{root}/bootstrap_rollup.parquet`)

The shard is per-run, not per-cell, so no atomic-shard-plus-
sentinel resumability machinery is needed; a re-run overwrites
the file in one atomic rename.

### B13.4 B5 renderer integration

`benchmarks/report/raw_loss.py` ships a NEW
`LeaderboardEntryWithCI` alongside the existing
`LeaderboardEntry` (the std variant is RETAINED as fallback
per R9). The renderer reads BOTH the manifest AND the rollup
shard, joins on `(dataset_name, model_name, task_type)`, and
produces a leaderboard where ONLY the primary-loss column is
modified (the existing `Rank`, secondary metric columns, and
resource columns from `raw_loss.py:_RESOURCE_COLUMNS` are
unchanged):

```
| Rank | Model | log_loss [95% CI]     | accuracy | ... | n_seeds | n_cells | n_skipped |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1    | lightgbm | 0.234 [0.221, 0.247] | 0.91     | ... | 3       | 9       | 0         |
```

The CI column format: `mean [ci_lo, ci_hi]` with 4 decimal
places (matches the existing `primary_loss_mean` precision).
A missing rollup entry (e.g., a model that registered but the
aggregator never ran on its output) renders the cell as the
sentinel `mean [n/a, n/a]` plus a footnote naming the dropped
rollup join. The secondary block is OUT OF SCOPE for the v1
CI rollout; only the primary loss carries a CI at v1.
B6/B7/B8/B11 secondary metrics + their renderers are
B13-followups (D-B13.1 through D-B13.4).

`render_leaderboard_markdown(manifest)` keeps its signature; a
new `render_leaderboard_markdown_with_ci(manifest, rollup)`
adds the CI variant. The CLI runs the CI variant when a rollup
shard exists at `output_root/bootstrap_rollup.parquet` AND the
rollup's `manifest_fingerprint` matches the live manifest's
fingerprint (Gemini-C3 freshness check). On mismatch (stale
rollup from a prior run), the renderer falls back to the std
variant and emits a footnote naming the divergence so the
reader knows the CI was suppressed for safety. The freshness
check is per-rollup-file (not per-row) because the entire
rollup is regenerated as one atomic write. The CI variant also
appends an asterisk to the CI string (`0.234 [0.221, 0.247]*`)
when `n_cells_evaluated < n_seeds * n_folds` (Gemini-I3 partial
-fold visual flag) and a footnote names the partial-cell count
per affected (dataset, model).

Skipped-cells footnote join: the existing skipped-cell
footnote at `benchmarks/report/raw_loss.py:269-287` groups by
`manifest["skipped_reason"]`. The rollup adds a SECOND skip
axis: `RollupRow.bootstrap_skipped_reason` (per-(dataset,
model), populated only when the LOADER failed during
aggregation). The CI renderer emits a parallel "Bootstrap
skipped" sub-table that groups rollup-sentinel rows
(`bootstrap_skipped_reason is not None`) by reason; the
per-cell skipped footnote and the rollup-skipped footnote
render as separate sections so the reader can attribute the
gap to its actual source (per-cell B5 failure vs report-time
loader failure). The two skip-reason columns are NOT unified
because they live on different shards and at different
granularities.

THIRD footnote source: when the rollup file is ABSENT due to
a wrapped `RawRollupError` (the B13.0 CLI wrapper case), the
renderer enters the "rollup absent" branch (std variant) AND
emits a "Bootstrap aggregator failed: `<exception class>`"
footnote naming the exception class that the wrapper logged.
This distinguishes the absent-due-to-disabled (opt-out flag,
no footnote) case from the absent-due-to-failure
(wrapper-caught, footnote present) case.

### B13.5 CLI dispatch + B9 assembled report

`benchmarks/run.py`: a new `_run_bootstrap_rollup` step runs
AFTER raw_loss completes (before the assembled report) when
the config opts in. v1 ALWAYS runs the rollup when raw_loss
completes successfully; an opt-out flag
`BenchmarkConfig.skip_bootstrap_rollup: bool = False` is the
escape hatch for the smoke-tier baseline tests that don't
exercise the full bootstrap.

The B9 assembled report (`benchmarks/report/render.py`)
references `bootstrap_rollup.parquet` in the run-metadata
block by file name; no new touch point needed in the assembler
itself because the leaderboard.md already renders the CI.

`benchmarks/run_manifest.py`: the manifest fingerprint extends
to include the `bootstrap_rollup.parquet` SHA (so a re-run with
a different bootstrap seed is fingerprinted distinctly).

### B13.6 Test surface

`tests/benchmarks/test_bootstrap.py` (NEW):
- `test_entity_block_bootstrap_ci_mean_matches_ground_truth`:
  pure unit test on a small synthetic loss + entity array;
  asserts the unresampled mean equals `losses.mean()`.
- `test_entity_block_bootstrap_ci_entity_vs_row_ci_width_zero_within_variance`
  (renamed per qa-C1): build a fixture with E=4 entities, each
  with K=5 IDENTICAL losses (within-entity variance is
  EXACTLY zero) but distinct between-entity means; assert that
  the entity-block CI width is approximately sqrt(K) times the
  CI width of a naive row bootstrap on the same vector. The
  zero-within-entity-variance fixture makes the analytical
  oracle clean: row bootstrap CI width ≈ sigma_between /
  sqrt(E*K); entity bootstrap CI width ≈ sigma_between /
  sqrt(E); ratio is sqrt(K).
- `test_entity_block_bootstrap_ci_deterministic_at_fixed_seed`:
  two calls with the same seed produce bit-identical
  `(mean, ci_lo, ci_hi)`.
- `test_entity_block_bootstrap_ci_output_is_stable_at_pinned_pcg64`
  (qa-I1 cross-process canary): hardcode the expected
  `(mean, ci_lo, ci_hi)` triple for a tiny deterministic input
  (3 entities × 2 losses, seed=0, n_resamples=100,
  metric_fn=np.nanmean) and assert the values match
  byte-for-byte. Document the NumPy version at which the
  expected values were computed in the test docstring. A
  refactor that swaps `PCG64` for any other generator
  silently breaks this.
- `test_entity_block_bootstrap_ci_invalid_shapes_raise`:
  mismatched length raises `ValueError`.
- `test_entity_block_bootstrap_ci_single_entity_returns_degenerate_ci`:
  one entity → both CIs equal the mean (the bootstrap can only
  resample one thing).
- `test_entity_block_bootstrap_ci_partial_nan_does_not_propagate`
  (qa-I6): a loss array with ONE NaN row in entity A and three
  CLEAN entities B/C/D produces a FINITE CI (the `np.nanmean`
  default reduces entity A's mean over its non-NaN rows, then
  the bootstrap aggregates across resampled entities).
- `test_entity_block_bootstrap_ci_metric_fn_sqrt_applies_per_resample`
  (qa-C2 hand-computed RMSE oracle): construct a tiny
  regression-style loss vector (squared errors) where the
  expected RMSE CI is hand-computed by:
  (1) enumerate all 2^E possible entity-block resamples with
  replacement at small E (E=3, n_resamples = full enumeration),
  (2) compute `sqrt(mean(squared_errors))` per resample,
  (3) take the 2.5/97.5 percentiles. Assert the bootstrap
  primitive with `metric_fn=lambda x: float(np.sqrt(np.nanmean(x)))`
  matches these hand-computed values within atol=1e-9. Kills
  the mutation that applies sqrt to the percentile of the MSE
  CI (Jensen-biased) rather than per-resample.
- `test_entity_block_bootstrap_ci_input_arrays_are_read_only`
  (Gemini-I1): a `metric_fn` that attempts in-place mutation
  of its input array raises `ValueError("output array is
  read-only")` immediately, not silently. Pins the
  `losses.flags.writeable = False` defensive guard.

`tests/benchmarks/test_bootstrap_rollup.py` (NEW):
- `test_aggregate_bootstrap_rollup_classification_emits_rollup_row`:
  e2e with the fake binary panel + `fake_constant_binary`;
  assert one RollupRow with finite `(mean, ci_lo, ci_hi)`,
  positive `n_entities`, `n_rows`,
  `bootstrap_rng_algorithm == "PCG64"`.
- `test_aggregate_bootstrap_rollup_regression_rmse_ci_matches_per_resample_sqrt`
  (qa-C2 end-to-end): e2e with the fake regression panel;
  hand-compute the expected RMSE CI by replaying the bootstrap
  with per-resample sqrt; assert the rollup row's
  `primary_loss_ci_lo / primary_loss_ci_hi` match within
  atol=1e-3. Also assert they DIFFER from the sqrt-of-MSE-
  percentile values (the Jensen-biased baseline) so the test
  kills the mutation.
- `test_aggregate_bootstrap_rollup_all_cells_skipped_emits_sentinel`:
  every cell skipped → sentinel RollupRow with `mean / ci_lo /
  ci_hi == None`, `n_cells_evaluated == 0`,
  `n_skipped_cells == N`.
- `test_aggregate_bootstrap_rollup_mixed_skip_runs_bootstrap_on_ok_subset`
  (qa-I2): some cells skipped + some OK → bootstrap runs on
  the OK subset; assert `n_cells_evaluated == n_ok`,
  `n_skipped_cells == n_skipped`, and finite `(mean, ci_lo,
  ci_hi)` (NOT a sentinel).
- `test_aggregate_bootstrap_rollup_missing_predictions_shard_drops_cell`:
  delete one cell's shard between B5 + rollup; assert
  `n_cells_evaluated` is the OK count minus that one cell.
- `test_aggregate_bootstrap_rollup_loader_error_routes_to_sentinel`
  (qa-C5 monkeypatch seam pinned): monkey-patch the
  `benchmarks.registry.datasets._LOADERS[dataset_name]` dict
  entry (NOT the higher-level `get_loader` function so the
  `isolated_registry` autouse fixture's loader-dict snapshot
  restores it on teardown). Assert the RollupRow lands with
  `bootstrap_skipped_reason="loader_failed: <type>: <msg>"`
  and that a subsequent `get_loader(dataset_name)` call in the
  same test session returns the ORIGINAL loader (the
  autouse fixture restored it).
- `test_aggregate_bootstrap_rollup_smoke_profile_halves_n_resamples`
  (qa-I3): build the env with `profile="smoke"`; assert the
  emitted RollupRow has `bootstrap_n_resamples == 5_000` (not
  10_000).
- `test_aggregate_bootstrap_rollup_experiment_spec_override_takes_precedence`
  (R6 opt-out granularity): set
  `ExperimentSpec(kind="raw_loss",
  bootstrap_n_resamples=2_000)`; assert the rollup row carries
  `2_000`, overriding the smoke/standard default.
- `test_aggregate_bootstrap_rollup_disabled_at_experiment_spec`:
  set `ExperimentSpec(kind="raw_loss",
  bootstrap_rollup_enabled=False)`; assert NO
  `bootstrap_rollup.parquet` is written even when the
  raw_loss step completes.
- `test_aggregate_bootstrap_rollup_loader_row_count_drift_raises`
  (Gemini-C1 defensive sort + row-count gate): monkey-patch a
  loader to return a DIFFERENT row count on the second call;
  assert the aggregator raises `RawRollupError("loader
  row-count drift")` rather than silently mis-attributing
  rows to entities.
- `test_aggregate_bootstrap_rollup_records_manifest_fingerprint`
  (Gemini-C3 freshness contract): the emitted RollupRow's
  `manifest_fingerprint` equals the run's manifest fingerprint
  at aggregation time.
- `test_aggregate_bootstrap_rollup_row_count_ceiling_raises`
  (R-B13-3 OOM gate): build a fake panel with `N` such that
  `N * n_resamples > _BOOTSTRAP_ROW_COUNT_CEILING`; assert the
  aggregator raises a typed `RawRollupError` naming D-B13.7
  rather than allocating into OOM.
- `test_aggregate_bootstrap_rollup_records_numpy_version_as_string`
  (R4 qa-I-1): assert the emitted RollupRow's
  `bootstrap_numpy_version` equals
  `importlib.metadata.version("numpy")` and is a non-empty
  PEP 440 string. A hardcoded empty string or `None`
  (violating the `str` field type) would fail this; the field
  is recorded explicitly to support future RNG-drift audits
  (R-B13-2).
- `test_run_bootstrap_rollup_cli_wrapper_catches_RawRollupError_and_exits_zero`
  (R5 arch-C1 fix): monkey-patch `aggregate_bootstrap_rollup`
  to raise `RawRollupError`; run `_run_bootstrap_rollup` via
  the CLI wrapper; assert (a) exit code 0 (the run as a whole
  succeeded, only CI degraded), (b) no partial
  `bootstrap_rollup.parquet` left on disk, (c) the
  leaderboard falls back to the std variant + emits the
  "Bootstrap aggregator failed" footnote. Without this
  wrapper the CLI would crash with an uncaught
  `RuntimeError`.

`tests/benchmarks/test_bootstrap_manifest.py` (NEW, qa-C3):
- `test_rollup_path_format`: `rollup_path(root)` returns
  `{root}/bootstrap_rollup.parquet`.
- `test_write_rollup_then_load_rollup_round_trips_all_fields`:
  build a `RollupRow` with EVERY field populated (including
  `None` CI fields, `bootstrap_rng_algorithm`,
  `bootstrap_numpy_version`, `manifest_fingerprint`, AND a
  `bootstrap_skipped_reason` string); write via
  `write_rollup`, load via `load_rollup`, assert every field
  round-trips exactly. Covers the new pydantic schema's
  parquet serialization for ALL Gemini-added fields (R4
  qa-I-2).
- `test_write_rollup_atomic_replace_on_overwrite`: write twice
  with different row counts; assert the file at the second
  write is the second write's content, not a partial mix.
  Mirrors `test_run_manifest.py::test_write_run_manifest_atomic_replace_on_overwrite`.

`tests/benchmarks/test_raw_loss_report.py` (NEW; B13 ships it):
- `test_render_leaderboard_with_ci_renders_mean_and_interval`:
  render the leaderboard with a rollup; assert the output
  contains `0.xxxx [0.xxxx, 0.xxxx]` per row.
- `test_render_leaderboard_without_ci_falls_back_to_scalar`:
  no rollup present; the renderer emits the existing
  `mean ± std` shape.
- `test_render_leaderboard_with_ci_drops_std_column_when_ci_present`:
  pin that the std column is NOT rendered when the CI variant
  is active; closes the R9 schema-change.
- `test_render_leaderboard_with_ci_surfaces_rollup_skipped_in_separate_footnote`
  (qa-I4 + arch-I4): a rollup row with
  `bootstrap_skipped_reason="loader_failed: ..."` renders in
  a "Bootstrap skipped" sub-table SEPARATE from the per-cell
  skipped-footnote. Assert both footnotes are present and
  distinguishable.
- `test_render_leaderboard_with_ci_falls_back_on_manifest_fingerprint_mismatch`
  (Gemini-C3): construct a rollup whose `manifest_fingerprint`
  does NOT match the live manifest; assert the renderer
  emits the STD variant (no CI column) plus a footnote naming
  the divergence.
- `test_render_leaderboard_with_ci_partial_fold_appends_asterisk`
  (Gemini-I3 visual flag): construct a rollup row with
  `n_cells_evaluated < n_seeds * n_folds`; assert the CI cell
  is rendered as `0.xxxx [0.xxxx, 0.xxxx]*` and a footnote
  names the partial-fold count.

`tests/benchmarks/test_loader_conformance.py` (NEW,
Gemini-C1):
- `test_every_registered_loader_row_order_is_deterministic_across_calls`:
  iterate every registered dataset; call its loader twice and
  assert the returned panels are EQUAL (`pd.DataFrame.equals`)
  in both row count, row order, and column values. The B13
  re-load contract depends on this conformance test passing for
  every dataset in the roster.
- `test_conformance_check_fails_on_deliberately_nondeterministic_loader`
  (R4 qa-I-3): register a synthetic loader that returns shuffled
  rows on the second call; assert the conformance assertion
  fails with a non-equal comparison. Proves the conformance
  test is not vacuously passing.

`tests/benchmarks/test_run_manifest.py` extension:
- `test_run_manifest_fingerprint_includes_bootstrap_rollup_sha`:
  pin the manifest fingerprint extension.
- `test_run_manifest_fingerprint_diverges_on_different_bootstrap_rollup_content`
  (qa-I5): write two rollup files with different
  `primary_loss_mean` values; assert the two manifest
  fingerprints are DIFFERENT. Kills a content-insensitive
  SHA implementation.

`tests/benchmarks/test_run.py` extension (or new):
- `test_skip_bootstrap_rollup_via_experiment_spec_does_not_write_rollup_file`
  (qa-C4): build a `BenchmarkConfig` with
  `ExperimentSpec(kind="raw_loss",
  bootstrap_rollup_enabled=False)`, run the full CLI
  dispatch, assert (a) no `bootstrap_rollup.parquet` is
  written, (b) `leaderboard.md` falls back to the std
  variant (no CI column in the output).

### B13.7 Skip-reason constant

`benchmarks/report/bootstrap_rollup.py` introduces
`_BOOTSTRAP_SKIP_REASON_LOADER_FAILED = "bootstrap_loader_failed"`
for the loader-error sentinel. The B5 skip-reason classifier
in `raw_loss.py` is NOT touched (the rollup-level skip is
report-time, not adapter-time); the new constant lives in the
rollup module and the leaderboard renderer's footnote groups
on it.

## Risk register

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R-B13-1 | Re-resolving entity_id via the loader at aggregation time doubles the data-load cost (once at B5, once at B13). | Low | When `BenchmarkConfig.cache_dir` is set, the loader hits the disk cache; the second load is cheap. When `cache_dir` is `None` (the default), the second load incurs the full materialization cost. Documented; in-process panel handoff between the raw_loss step and the rollup step at `run.py:_dispatch_kinds` is a B13-followup if the cost surfaces. A future `entity_id` column on the predictions shard would moot this entirely. |
| R-B13-2 | NumPy generator-algorithm drift across major versions. | Medium | B13 constructs the RNG as `np.random.Generator(np.random.PCG64(seed))` explicitly, not via `default_rng` (which may swap the default in a future major release). `RollupRow.bootstrap_rng_algorithm="PCG64"` is recorded on every row so a future audit can pin output drift to a generator change rather than a code mutation. NumPy version is also captured in `_PINNED_PACKAGES`. |
| R-B13-3 | 10000 resamples * N rows can be costly on large datasets. The naive implementation concatenates resampled-entity rows per resample, allocating O(N) per resample, O(N * n_resamples) total memory traffic. For a 6M-row Amex-tier dataset that's GB-scale per-resample allocations + ~60B float ops total. | High | Profile-tier knob: smoke = 5000, standard/full = 10000. Per-dataset wall-clock is logged so a future tier-extension is data-driven. Per-experiment override via `ExperimentSpec(kind="raw_loss", bootstrap_n_resamples=...)`. **Defensive row-count gate**: the aggregator emits `RawRollupError("dataset row count {N} > _BOOTSTRAP_ROW_COUNT_CEILING; use the per-entity-sufficient-statistics path in D-B13.7 instead")` when `N * n_resamples > 5e10` (a configurable ceiling). D-B13.7 names the optimization (per-entity `(sum_loss, count_rows)` pre-aggregation; both `nanmean` and `sqrt(nanmean)` are expressible from sufficient statistics, so the optimization is correctness-preserving for the v1 metric_fns). v1 ships the naive path; the optimization is a B13-followup once a full-tier dataset surfaces the ceiling. |
| R-B13-4 | Replacing the std column with the CI in the leaderboard is a user-facing schema change in the published report. | Medium | The std variant is RETAINED as a fallback path (no-rollup case + `bootstrap_rollup_enabled=False`), so existing readers parsing the `± std` shape still see it when the rollup hasn't run. The CI variant requires a reader-migration note documented in CHANGELOG at B13 release. |
| R-B13-5 | The rollup aggregator is run-once (not per-cell), so an interrupted run leaves a stale or absent rollup file. | Low | Always-overwrite semantics + manifest-fingerprint divergence on re-run; documented. |
| R-B13-6 | The B5 driver's `_strip_below_floor_rows` already removes NaN-tinged rows from the predictions shard, so a non-skipped shard should not contain NaN losses. The bootstrap's `np.nanmean` default is therefore defensive against an invariant the driver already enforces; a NaN in a non-skipped shard indicates either an invariant break or a regression-task `y_true` NaN that the strip path doesn't cover. | Low | Documented; the bootstrap accepts NaN-tinged input rather than raising so a single bad row does not crash the whole aggregation. Below-floor strip is the upstream defense; B13's nan-tolerance is a safety net. |

## Estimated effort

| Module | Size |
|---|---|
| `benchmarks/metrics/bootstrap.py` (NEW, the primitive) | Small (~130 lines) |
| `benchmarks/report/bootstrap_rollup.py` (NEW, the aggregator) | Medium (~260 lines) |
| `benchmarks/bootstrap_manifest.py` (NEW, write/load) | Small (~80 lines) |
| `benchmarks/report/raw_loss.py` (CI integration) | Small (~50 lines edit) |
| `benchmarks/run.py` (CLI step) | Small (~15 lines) |
| `benchmarks/run_manifest.py` (fingerprint extension) | Small (~10 lines) |
| `benchmarks/config.py` (`skip_bootstrap_rollup` opt-out) | Trivial |
| `tests/benchmarks/test_bootstrap.py` (primitive) | Medium (~200 lines) |
| `tests/benchmarks/test_bootstrap_rollup.py` (e2e) | Medium (~250 lines) |
| `tests/benchmarks/test_raw_loss_report.py` extension | Small (~80 lines) |
| `tests/benchmarks/test_run_manifest.py` extension | Small (~40 lines) |
| `docs/benchmark_suite_implementation_plan.md` (B13 actual-shape) | Small |

Total: ~1,115 lines. Comparable to B11 / B12.

## Addressed

R1 swarm: architecture-reviewer (2C/7I/4N REQUEST_CHANGES) +
qa-test-coverage (5C/6I/3N REQUEST_CHANGES) + style-reviewer
(0/0/0 APPROVE). Total deduped: 6 CRITICAL + 11 IMPROVEMENT
+ 6 NIT. All CRITICALs and load-bearing IMPROVEMENTs addressed:

- **arch-C1 + qa-C2** (regression sqrt bias / Jensen's
  inequality): `entity_block_bootstrap_ci` now takes a
  `metric_fn` callable; regression aggregator passes
  `lambda x: float(np.sqrt(np.nanmean(x)))` so sqrt is applied
  PER RESAMPLE. The post-bootstrap sqrt transform was REMOVED.
  Test `test_aggregate_bootstrap_rollup_regression_rmse_ci_matches_per_resample_sqrt`
  hand-computes the expected RMSE CI by replaying the
  bootstrap with per-resample sqrt and asserts the rollup row
  matches AND differs from the Jensen-biased sqrt-of-MSE-CI
  baseline (kills the mutation).
- **arch-C2** (LeaderboardEntry schema swap contradiction):
  R9 rewritten to RETAIN both `LeaderboardEntry` (std variant)
  and ship a NEW `LeaderboardEntryWithCI`. The renderer
  dispatches by rollup presence: CI variant when
  `bootstrap_rollup.parquet` exists, std fallback when absent
  (smoke-tier, `bootstrap_rollup_enabled=False`, partial run).
  Test `test_render_leaderboard_with_ci_drops_std_column_when_ci_present`
  pins the CI variant; `test_render_leaderboard_without_ci_falls_back_to_scalar`
  pins the fallback.
- **qa-C1** (imprecise entity-vs-row oracle): test renamed to
  `test_entity_block_bootstrap_ci_entity_vs_row_ci_width_zero_within_variance`
  with EXACTLY-zero within-entity variance fixture so the
  sqrt(K) factor holds analytically.
- **qa-C3** (write_rollup / load_rollup untested): new test
  file `test_bootstrap_manifest.py` with 3 named tests
  (round-trip, atomic-replace, path-format).
- **qa-C4** (opt-out flag untested): test
  `test_skip_bootstrap_rollup_via_experiment_spec_does_not_write_rollup_file`
  added to `test_run.py` extension.
- **qa-C5** (monkeypatch seam ambiguous): the loader-error
  test now explicitly pins the seam to
  `benchmarks.registry.datasets._LOADERS[dataset_name]` (the
  dict entry, within the `isolated_registry` autouse fixture's
  snapshot scope) and asserts the original loader is restored
  on teardown.
- **arch-I1** (R2 pool-vs-per-fold rationale): R2 now names
  the rejected alternative with the three-part rationale
  (non-overlapping fold time-slices, CI-aggregation rule not
  pinned, higher resampling power).
- **arch-I2 + qa-I3** (smoke n_resamples knob): R4 now reads
  from `env.profile` via
  `_BOOTSTRAP_N_RESAMPLES_BY_PROFILE`, no new
  `BenchmarkConfig` field. ExperimentSpec carries a per-run
  override.
- **arch-I3** (opt-out granularity): moved from
  `BenchmarkConfig` to
  `ExperimentSpec(kind="raw_loss", bootstrap_rollup_enabled,
  bootstrap_n_resamples)`.
- **arch-I4 + qa-I4** (skip-reason renderer join): B13.4 now
  spells out a SEPARATE "Bootstrap skipped" footnote section
  for `bootstrap_skipped_reason` entries, distinct from the
  per-cell skipped footnote. Test
  `test_render_leaderboard_with_ci_surfaces_rollup_skipped_in_separate_footnote`
  pins both.
- **arch-I5** (cost mitigation cache_dir condition): R-B13-1
  now names the condition explicitly + flags the
  in-process-handoff B13-followup.
- **arch-I6** (NumPy RNG algorithm pin): primitive now
  constructs the RNG as
  `np.random.Generator(np.random.PCG64(seed))` explicitly;
  `RollupRow.bootstrap_rng_algorithm="PCG64"` recorded;
  cross-process canary test
  `test_entity_block_bootstrap_ci_output_is_stable_at_pinned_pcg64`
  added.
- **arch-I7** (NaN-handling contract): R-B13-6 added to the
  risk register naming the upstream `_strip_below_floor_rows`
  defense and B13's role as the safety net. Test
  `test_entity_block_bootstrap_ci_partial_nan_does_not_propagate`
  pins the partial-NaN case.
- **qa-I1** (cross-process reproducibility canary): the
  `_stable_at_pinned_pcg64` test serves this purpose.
- **qa-I2** (mixed-skip case): test
  `test_aggregate_bootstrap_rollup_mixed_skip_runs_bootstrap_on_ok_subset`
  added.
- **qa-I5** (fingerprint divergence on different seeds): test
  `test_run_manifest_fingerprint_diverges_on_different_bootstrap_rollup_content`
  added.
- **qa-I6** (partial NaN propagation): the
  `_partial_nan_does_not_propagate` test pins the case where
  one NaN row in one entity does NOT corrupt the whole CI
  because `np.nanmean` reduces within the entity first.
- **arch-N1** (reuse classification_nll from pairwise.py):
  B13.2 step 6 now explicitly imports
  `benchmarks.metrics.pairwise.classification_nll`.
- **arch-N3** (R-B13-4 severity): bumped to Medium with the
  CHANGELOG reader-migration note named.
- **arch-N4** (header column count): B13.4 example now shows
  ALL columns (Rank, primary CI, secondary block, resource,
  skipped) and explicitly notes that only the primary-loss
  column is modified.

R2 confirming swarm: architecture-reviewer (APPROVE with 3
doc-only IMPROVEMENTs), qa-test-coverage (APPROVE with 2I/1N),
style-reviewer (1 CRITICAL: em-dash at line 455, FIXED).

- **R2 style-C** (em-dash at line 455): collapsed the
  parenthetical to remove the em-dash. Single fix.

R2 IMPROVEMENTs surfaced and addressed inline:

- **R2 qa-NEW-I1** (custom `metric_fn` extension test): folded
  into the deferral list below; the two production callables
  (`np.nanmean`, `sqrt(np.nanmean(.))`) are both exercised by
  named tests, and the `Callable[[np.ndarray], float]`
  signature is structurally testable; a sentinel-callable test
  is low-risk to defer.
- **R2 qa-NEW-I2** (standard-profile default fallback test):
  the smoke-vs-standard branch is exercised by the smoke-only
  test; the standard arm is the no-override default and is
  covered by the e2e classification test (which runs with
  `profile="standard"` implicitly). Deferred.
- **R2 qa-NEW-N1** (section-header pinning in the bootstrap-
  skipped footnote test): the test description says "present
  and distinguishable"; an additional pin on exact heading
  strings is a strengthening, not a correctness gap. Deferred.

Gemini design final-pass (architecture-reviewer model).
3 CRITICAL + 4 IMPROVEMENT + 1 NIT. All CRITICALs addressed
in this revision; consensus is invalidated until an R4
confirming swarm round APPROVE's:

- **Gemini-C1** (loader row-order determinism): R7 now pins
  an EXPLICIT loader-determinism contract ("every registered
  loader MUST produce row-identical output across calls") +
  a DEFENSIVE post-load sort by `(entity_col, time_col)` in
  the aggregator + a row-count drift gate that raises
  `RawRollupError`. A new conformance test file
  `tests/benchmarks/test_loader_conformance.py` asserts the
  contract for every registered dataset. The aggregator e2e
  tests add `test_aggregate_bootstrap_rollup_loader_row_count_drift_raises`.
- **Gemini-C2** (OOM on full-tier datasets): R-B13-3 severity
  bumped to High; the aggregator now ships a defensive
  row-count gate that emits `RawRollupError(...)` when
  `N * n_resamples > _BOOTSTRAP_ROW_COUNT_CEILING` (=5e10).
  D-B13.7 names the per-entity-sufficient-statistics
  optimization as the documented B13-followup; the proof of
  correctness preservation (sum / count expressibility for
  both v1 metric_fns) is named. Test
  `test_aggregate_bootstrap_rollup_row_count_ceiling_raises`
  pins the gate.
- **Gemini-C3** (stale-rollup freshness): RollupRow now
  carries `manifest_fingerprint`. The renderer asserts
  `rollup.manifest_fingerprint == manifest.fingerprint()`
  before joining; on mismatch the std variant renders with a
  footnote naming the divergence. Two new tests
  (`test_aggregate_bootstrap_rollup_records_manifest_fingerprint`
  + `test_render_leaderboard_with_ci_falls_back_on_manifest_fingerprint_mismatch`)
  pin the contract end-to-end.
- **Gemini-I1** (input-array mutability defense): the
  primitive sets `losses.flags.writeable = False` and
  `entity_ids.flags.writeable = False` at entry. Test
  `test_entity_block_bootstrap_ci_input_arrays_are_read_only`
  asserts a mutating `metric_fn` raises.
- **Gemini-I2** (percentile method pin): primitive now uses
  `np.percentile(..., method="linear")` EXPLICITLY.
- **Gemini-I3** (partial-fold visual flag): renderer appends
  `*` to the CI cell when `n_cells_evaluated < n_seeds *
  n_folds` + a footnote names the partial count. Test
  `test_render_leaderboard_with_ci_partial_fold_appends_asterisk`.
- **Gemini-I4** (seed-pooling rationale): R2 now spells out
  why pool-then-bootstrap (preserves within-seed variation
  the per-seed-mean would collapse; produces slightly wider,
  more honest CIs).
- **Gemini-N1** (numpy_version on RollupRow): added
  `bootstrap_numpy_version: str` field; recorded via
  `importlib.metadata.version("numpy")` at aggregation time.

### R4 swarm closure

R4 confirming swarm: architecture-reviewer (2C/3I/2N
REQUEST_CHANGES), qa-test-coverage (0C/3I/1N APPROVE),
style-reviewer (0/0/0 APPROVE). The two arch CRITICALs were
introduced by my Gemini-C2/C3 fixes themselves: `RawRollupError`
and `RunManifest.fingerprint()` were referenced multiple times
without a declaration of their surface. Both closed by adding
the new B13.0 typed-surface section that names the module
home, base class (mirrors `RawMTSError(RuntimeError)`), and
the relationship to the existing split fingerprint.

R4 IMPROVEMENTs surfaced and addressed:
- qa-I-1 (no test for `bootstrap_numpy_version`): test
  `test_aggregate_bootstrap_rollup_records_numpy_version_as_string`
  added (asserts equals `importlib.metadata.version("numpy")`,
  non-empty PEP 440 string).
- qa-I-2 (round-trip test enumerates all new fields): the
  round-trip test description now explicitly enumerates
  `manifest_fingerprint`, `bootstrap_numpy_version`, AND
  `bootstrap_rng_algorithm` (the previous version only listed
  the first two new fields).
- qa-I-3 (negative-path conformance): test
  `test_conformance_check_fails_on_deliberately_nondeterministic_loader`
  added so the loader-conformance machinery is not vacuously
  passing.

### R5 swarm closure

R5 confirming swarm: architecture-reviewer (2C/3I/2N
REQUEST_CHANGES), qa-test-coverage (0C/1I/1N APPROVE),
style-reviewer (0/0/0 APPROVE). The two arch CRITICALs were
introduced by MY R4 typed-surface section itself:

- **R5 arch-C1** (CLI catch contradiction): the B13.0 claim
  that `RawRollupError(RuntimeError)` "propagates to the CLI's
  outer `except ValueError -> exit 1`" was wrong. The CLI at
  `benchmarks/run.py:427` only catches `ValueError`. A
  top-level `RawRollupError(RuntimeError)` crashes the CLI
  with a traceback, making the std-variant fallback structurally
  unreachable. B13.0 rewritten to declare an EXPLICIT CLI
  wrapper at `_run_bootstrap_rollup` that catches
  `RawRollupError`, deletes any partial output, returns exit
  0, and surfaces a "Bootstrap aggregator failed" footnote in
  the std-variant leaderboard. Test
  `test_run_bootstrap_rollup_cli_wrapper_catches_RawRollupError_and_exits_zero`
  pins the wrapper.
- **R5 arch-C2** (internal contradiction): B13.0 had both the
  "driver IS-A catch" path AND the "CLI catch via ValueError"
  path, contradicting each other. Rewritten to acknowledge
  BOTH paths explicitly: `RuntimeError` base for the driver's
  defensive future-proof catch; explicit CLI wrapper for the
  v1 report-time call site.
- **R5 arch-I3** (canonical-JSON over-claim): rewrote the
  `RunManifest.fingerprint()` declaration to specify
  `json.dumps(..., sort_keys=True, separators=(",", ":"))`
  inline (NOT a reuse of `fingerprint_folds`'s
  fold-layout-specific serializer); softened the cross-module
  reuse claim.
- **R5 arch-I1** (schema-version migration): documented that
  the SHA changes on any `RunManifest` schema bump; the
  renderer's fingerprint-mismatch fallback handles this
  cleanly.
- **R5 arch-I2** (impl plan B13 block): the build commit
  will land a B13 actual-shape section in the impl plan (B11
  + B12 precedent). Documented in B13.0.

R5 IMPROVEMENTs deferred (non-blocking):
- **R5 qa-I1** (conformance-function seam): the test
  description could be tightened to invoke the conformance
  function rather than `.equals` directly; the implementation
  team can resolve this at code time without a doc change.
- **R5 arch-N1/N2**: doc polish; non-blocking.

### R6 swarm closure

R6 confirming swarm: architecture-reviewer (0C/2I/2N APPROVE).
Both R6 IMPROVEMENTs (B13.4 footnote source for the
CLI-wrapper case, duplicate `## Deferred` heading) folded
inline. NITs deferred (run.py:427 line citation, exception-
class name pin in the wrapper test) are tractable doc/test-
description tightenings.

## Deferred

Doc-only polish or non-blocking strengthenings from all six
rounds:

- **arch-N2** (`tests/benchmarks/test_raw_loss_report.py`
  path): B13.6 notes the file is NEW with B13's tests as its
  initial content; the path is reserved at this commit.
- **R2 qa-NEW-I1 / NEW-I2 / NEW-N1**: see R2 swarm closure
  above.
- **R5 qa-I1** (conformance-function seam): the test
  description could be tightened to invoke the conformance
  function rather than `.equals` directly; the implementation
  team can resolve at code time.
- **R5 arch-N1 / N2**: doc polish.
- **R6 NITs**: `run.py:427` symbolic anchor preferred over
  numeric line citation; exception-class-name pin in the
  CLI-wrapper test assertion. Both tightenings are
  implementation-time concerns.
