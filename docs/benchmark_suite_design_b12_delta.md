# B12 design delta: classical-TSC adapter family

This is a Phase B12 delta against `benchmark_suite_design.md` (Phase
B2 model registry) and `benchmark_suite_implementation_plan.md`
(Phase B10 GBM family pattern). It is reviewable in isolation: the
existing benchmark stack is unchanged, B12 adds one adapter module,
one protocol module, one HPO module, and the corresponding tests.

## Requirements (grading rubric)

This delta is graded against:

- **R1** Three classical-TSC classifiers ship in B12: ROCKET,
  MultiRocket, Catch22-with-classifier-head. All three from `aeon`.
  Family name `"tsc"`. They register under
  `benchmarks/adapters/tsc.py`.
- **R2** Optional-dep contract: `aeon` is NOT installed by default
  (its 0.11.x branch pins `scikit-learn < 1.6`, the library
  requires `>= 1.6`). The B12 adapter MUST handle `aeon` missing
  cleanly: a TSC cell whose adapter cannot import `aeon` writes a
  typed `skipped_reason="optional_dep_missing: aeon"` ResultRow and
  the run continues. No crash, no half-fitted state.
- **R3** New data view: `raw-mts`. The protocol module
  `benchmarks/protocol/raw_mts.py` (NEW) reshapes a panel from
  the F2 long-format (one row per `(entity, period)`) to a 3D
  ndarray of shape `(n_instances, n_channels, n_timesteps)` that
  aeon's classifier signature expects. The reshape is purely a
  view operation: it does NOT touch labels or split indices, and
  it pulls its window length from `lookback.resolve_lookback()` so
  the B4.5 lookback-binding invariant still holds (split, deep
  model, featurizer, raw-mts all share the same `L_resolved`).
- **R4** Equal-length panel guarantee: the v1 reshape rejects
  unequal-length entities with a typed `RawMTSError`. aeon's
  unequal-length wrappers are a B12-followup (out of scope for v1).
  The reshape's contract is enforced by a Hypothesis property
  test (every panel that satisfies the F2 contract + has
  per-entity time-monotone `period` + has equal lookback windows
  produces a tensor of the declared shape).
- **R5** HPO support: `benchmarks/hpo/tsc.py` (NEW) registers
  Optuna search spaces under family `"tsc"` so B8's hpo_uplift
  driver picks them up without changes. Trial budget consumed
  from B8's existing profile-tier envelope
  (`_HPO_BUDGETS_BY_PROFILE` at
  `benchmarks/experiments/hpo_uplift.py:115-119`:
  smoke=0, standard=25, full=100). No per-family override; the
  budget is profile-keyed only.
- **R6** Skip-coverage tests: BOTH the aeon-missing and the
  aeon-installed paths are tested. aeon-missing is the dominant
  CI default; aeon-installed runs under `pytest.importorskip(
  "aeon")` so the suite stays green in either environment.
- **R7** B11 ensemble-lift integration: the B11 default
  partition (`_DEFAULT_SEQ_FAMILY="seq_sklearn"`,
  `_DEFAULT_BASELINE_FAMILY="gbm"` at
  `benchmarks/experiments/ensemble_lift.py:91-92`) is unchanged
  at v1. A future B11-followup wires `tsc` as a third family. No
  B11 code edit lands in this delta.

- **R8** Bootstrap-CI honesty (B5.4 forward contract): panel
  data carries intra-entity correlation; the correct B5.4
  bootstrap design resamples ENTITIES (block bootstrap), not
  rows, for ALL adapter families on panel data. TSC adapters'
  per-instance-to-per-row broadcast does NOT introduce a new CI
  problem; it shares the same problem the deep + GBM adapters
  already have if row-bootstrap is naively applied. B12
  therefore introduces NO new ResultRow field for this; the
  prediction_granularity flag proposed in R1 + R2 was the wrong
  abstraction (it would have encoded a TSC-only fix for a
  whole-suite issue). The B5.4 bootstrap design (not yet
  implemented; the metrics module references it at
  `benchmarks/metrics/classification.py:18`) MUST unconditionally
  resample by entity for panel datasets. B12 contributes only
  the adapter-docstring documentation that TSC's per-instance
  predictions are broadcast to per-row outputs; the bootstrap
  consumer derives the broadcast detail from the adapter's
  family lookup at B5.4 time, not from a manifest schema field.

## Out of scope (B12-followup deferrals)

- **D-B12.1** Heavyweight TSC ensembles (HIVE-COTE 2.0,
  MultiRocket-Hydra): D3 from the main design. Registry-
  extensible later with no harness change.
- **D-B12.2** Unequal-length panel support: aeon supports it via
  per-estimator unequal-length wrappers; the v1 reshape rejects
  unequal-length input cleanly. A follow-up wraps the reshape so
  unequal-length panels flow to aeon's documented path.
- **D-B12.3** TSC regression (aeon's `BaseRegressor`): the v1
  roster is classification-only. `task_types=("binary",
  "multiclass")` on every spec. A follow-up adds regression
  variants of the three classifiers.
- **D-B12.4** GPU acceleration: aeon's TSC classifiers are CPU-
  only at the 0.11.x cut. No GPU code path; the manifest's
  hardware_tier captures CPU vs GPU for the seq side independently.
- **D-B12.5** Promote `aeon` into a `benchmarks-tsc` extra: hinges
  on aeon loosening its sklearn pin. Until then, the install path
  remains manual and the adapter logs the missing dep on every
  TSC cell. Listed in the pyproject comment block at lines 67-73.
- **D-B12.6** Categorical feature handling for the TSC family:
  v1 drops categorical channels from the raw-mts reshape because
  ordinal encoding violates ROCKET's metric-space assumption and
  one-hot inflates the channel count unpredictably. A follow-up
  branch can add aeon's documented per-categorical wrappers or
  a fixed one-hot path. Datasets whose channel set is entirely
  categorical are unsupported by the TSC family at v1; they
  still participate in the seq + GBM comparisons.

## Architecture

### B12.1 Adapter module: `benchmarks/adapters/tsc.py`

Shape mirrors `benchmarks/adapters/gbm.py:_GBMAdapter`:

- `_TSCAdapter` base dataclass with:
  - `spec: DatasetSpec`
  - `hyperparameters: dict[str, Any]`
  - `_fitted_estimator: Any | None` (the aeon classifier, post-fit)
  - `_cached_panel_id: int | None` + `_cached_tensor: np.ndarray | None`
    (the panel → 3D tensor reshape is O(N * features * lookback)
    and is recomputed on every `fit/predict/predict_proba` call
    against the same panel; cache by `id(panel)` like the GBM
    featurizer cache).
  - `__post_init__` blocks direct base instantiation (matches
    `_GBMAdapter.__post_init__`).
  - `_make_estimator(self) -> Any` ABSTRACT (raises in base).
    Subclasses return the configured aeon classifier instance.
  - `_check_aeon_available(self) -> None`: lazy-imports aeon and
    raises `OptionalDependencyMissingError("aeon")` if absent.
    Called at the top of `fit`. The typed error is caught by the
    B5 raw-loss driver's `_SKIP_REASON_OPTIONAL_DEP_MISSING`
    branch (NEW) and rendered as a typed skip rather than a
    generic adapter error.
  - `fit(self, panel, y) -> Self`:
    1. `_check_aeon_available()`
    2. Reshape panel via `raw_mts.panel_to_tensor(panel, spec)`
    3. Project `y` from per-row to per-instance labels (aeon
       wants one label per instance, not one per row; the
       projection is the entity's last-period label).
    4. `_make_estimator().fit(X_3d, y_per_instance)`
  - `predict(self, panel) -> np.ndarray`: reshape + predict, then
    BROADCAST the per-instance prediction back to per-row to
    match the B5 contract (`y_pred` shape == panel length). This
    is the F2 contract surface; the design's per-instance vs
    per-row mismatch is solved by broadcasting on the way out.
  - `predict_proba(self, panel) -> np.ndarray`: same broadcast.
  - `predict_quantiles(self, panel) -> NoReturn`: raises
    `QuantilesUnsupportedError(self.name)` (TSC family is
    classification-only at v1).

- Three concrete adapters (classifier-only at v1 per D-B12.3):
  `_ROCKETClassifierAdapter`, `_MultiRocketClassifierAdapter`,
  `_Catch22ClassifierAdapter`. Each carries `name`,
  `family="tsc"`, `task_types=("binary", "multiclass")`,
  `supports_proba=True` ClassVars.

### B12.2 Protocol module: `benchmarks/protocol/raw_mts.py`

- `RawMTSError(Exception)`: typed reshape failure. There is no
  `BenchmarkError` umbrella in the repo today and this delta
  does not introduce one; the existing IO bases at
  `benchmarks/_io/integrity.py` are also bare `Exception`
  subclasses.
- `panel_to_tensor(panel, spec) -> tuple[np.ndarray, np.ndarray]`:
  returns `(X_3d, instance_to_panel_row_mapping)`. `X_3d` shape
  is `(n_instances, n_channels, n_timesteps)`. The mapping array
  is the per-instance last-period `panel_row_index` so the
  adapter can broadcast per-instance predictions back to per-row
  outputs. `n_timesteps` is the resolved lookback from
  `lookback.resolve_lookback(spec)` (no override; the resolver
  signature at `benchmarks/protocol/lookback.py:18` is
  `resolve_lookback(spec, override=None) -> int`). Entities with
  fewer than `L_resolved` rows DO NOT enter the tensor but the
  adapter emits `np.nan` for those panel rows on
  `predict_proba` / `predict` so the existing
  `_strip_below_floor_rows` machinery at
  `benchmarks/experiments/raw_loss.py:360-399` surfaces the drop
  count via `n_below_floor_dropped` uniformly across adapters
  (deep + GBM + TSC). The strip-via-NaN-marker contract is the
  same convention the deep model uses today.
- `RawMTSConfig`: pydantic v2 frozen config carrying the
  resolved lookback + the channel column list (v1: numeric
  channels only, see "Categorical handling" below).
- Equal-length panel guarantee: if any entity has more than
  `L_resolved` rows, the reshape uses the LAST `L_resolved`
  rows for that instance (the "trailing window" convention,
  consistent with the deep model's lookback contract). If any
  retained entity has fewer than `L_resolved` rows after the
  drop, raise `RawMTSError` with the entity id.
- Categorical channel handling (v1: drop): aeon's
  ROCKET / MultiRocket use random convolutions, which impose a
  metric space on their inputs; ordinal-encoded categoricals
  inject spurious structural patterns and violate the fair-
  comparison protocol. One-hot encoding inflates the channel
  count and changes the input distribution unpredictably across
  datasets. The v1 raw-mts reshape therefore DROPS
  `spec.feature_categorical_cols` entirely; only
  `spec.feature_real_cols` flows into the tensor. A dataset
  whose channel set is entirely categorical raises `RawMTSError`
  at config validation. Documented under D-B12.6 below; future
  branches can add aeon's documented per-categorical wrappers
  or a one-hot path.
- `np.float32` cache: the per-instance reshape caches the 3D
  tensor by `id(panel)` (same convention as the GBM featurizer
  cache); the cached array is cast to `np.float32` to halve the
  host RAM cost. Aeon classifiers accept float32 input
  natively. A 100k-instance x 200-channel x 64-timestep tensor
  is ~5GB at float32 vs ~10GB at float64.
- Tensor dtype convention: aeon classifiers accept `np.float32`;
  the reshape returns float32 regardless of the source dtype
  in the panel.

### B12.3 HPO module: `benchmarks/hpo/tsc.py`

Mirrors `benchmarks/hpo/gbm.py`. No per-family trial-count
constants; the budget is profile-tier-only per R5.

- Three suggest_params functions, one per aeon classifier:
  - ROCKET: `num_kernels` (Int, log, 500 .. 20_000),
    `n_jobs=-1` (fixed). Search dimensionality 1.
  - MultiRocket: `num_kernels` (Int, log, 500 .. 20_000),
    `max_dilations_per_kernel` (Int, 8 .. 64),
    `n_jobs=-1` (fixed). Search dimensionality 2.
  - Catch22: `outlier_norm` (Bool), `replace_nans` (Bool),
    `n_jobs=-1` (fixed). Search dimensionality 2. The Catch22
    estimator stage runs feature extraction; the classifier
    head is a RidgeClassifierCV so there is no head-side
    hyperparameter to sweep at v1.
- The B6.4.0 parity-disclosure report quotes ONE
  `search_space_size` int per family; TSC declares 2 (the
  maximum across the three classifiers). The disclosure
  footnote names the per-classifier breakdown (ROCKET=1,
  MultiRocket=2, Catch22=2) so the parity claim is honest. GBM
  declares 6 per `benchmarks/hpo/gbm.py:142`, materially larger
  than TSC's 2; the report renders both sizes side by side.
- `register_tsc_hpo_spaces()` is called from
  `benchmarks/hpo/__init__.py` (NEW import).

### B12.4 Skip-reason wire-up

The wire-up change requires **twelve** explicit touch points
across five modules. Both `raw_loss.py` (B5) and `hpo_uplift.py`
(B8) instantiate adapters with identical narrow-tuple catches
(`raw_loss.py:584-590` and `hpo_uplift.py:440-443, 636-639`); the
typed exception must be wired into both. The MRO choice
subclasses `Exception` directly, NOT `ImportError`: the existing
`except ImportError` clauses at `raw_loss.py:201`
(`_detect_hardware_tier` torch probe) and `run_manifest.py:185,
196` (torch GPU/CUDA probes) would otherwise silently swallow
the typed skip error if any future code move puts an adapter
import on their path.

In `benchmarks/adapters/_base.py`:

- **(a) Typed exception**: New
  `OptionalDependencyMissingError(Exception)`. Carries one
  positional arg `package_name: str`.

In `benchmarks/experiments/raw_loss.py`:

- **(b) Constant**: New
  `_SKIP_REASON_OPTIONAL_DEP_MISSING = "optional_dep_missing"`
  near the existing `_SKIP_REASON_*` block.
- **(c) Catch clause**: Append
  `OptionalDependencyMissingError` to the narrow tuple at the
  per-cell try / except site (`raw_loss.py:584-590` today is
  `(RuntimeError, MemoryError, NotFittedError,
  ProbaUnsupportedError, QuantilesUnsupportedError)`). On
  catch, route to the new skip reason with the
  `package_name` interpolated:
  `f"{_SKIP_REASON_OPTIONAL_DEP_MISSING}: {exc.package_name}"`.
  Place ahead of the generic-adapter-error catch so it wins on
  ordering.
- **(d) Counter init**: At `raw_loss.py:737` (where the other
  per-skip counters initialize), add
  `skipped_optional_dep_missing = 0`.
- **(e) Counter classifier**: At `raw_loss.py:889-896` (the
  `elif row.skipped_reason.startswith(...)` chain), add an
  `elif startswith(_SKIP_REASON_OPTIONAL_DEP_MISSING)` branch
  ahead of the `else: skipped_adapter_error += 1` fallthrough.
- **(f) Summary field**: Add
  `cells_skipped_optional_dep_missing: int` to
  `RawLossExperimentResult` (mirrors
  `cells_skipped_proba_unavailable`) and pass the new counter
  into the constructor at `raw_loss.py:906`.

In `benchmarks/experiments/hpo_uplift.py` (B8 driver, the same
adapter-error-catch shape as B5):

- **(g) HPO trial catch**: Append `OptionalDependencyMissingError`
  to the narrow tuple at `hpo_uplift.py:440-443` (the per-trial
  fit / evaluate site). On catch, route the trial to the same
  pruned-trial bookkeeping the other adapter errors use today.
- **(h) HPO cell catch**: Append `OptionalDependencyMissingError`
  to the narrow tuple at `hpo_uplift.py:636-639` (the per-cell
  best-trial replay site). On catch, route to a new
  `_SKIP_REASON_OPTIONAL_DEP_MISSING_HPO` constant; the catch
  precedes the existing
  `f"{_SKIP_REASON_ADAPTER_ERROR}: ..."` fallthrough at
  `hpo_uplift.py:670`.
- **(i) HPO summary field**: Add
  `cells_skipped_optional_dep_missing: int` to
  `HPOUpliftExperimentResult` (mirrors
  `RawLossExperimentResult`'s same-named field added in (f)).
- **(j) HPO counter init + classifier**: Add the per-skip
  counter init alongside the other counters and the matching
  classifier branch.

In `benchmarks/run_manifest.py`:

- **(k) Pinned-package list**: Append `"aeon"` to
  `_PINNED_PACKAGES` at `benchmarks/run_manifest.py:51-64` so
  aeon-installed runs capture the version on every fingerprint
  (mirrors the existing `"lightgbm"` / `"xgboost"` / `"catboost"`
  entries from B10). The existing `_safe_pkg_version` helper at
  `run_manifest.py:141-146` catches `PackageNotFoundError`
  uniformly, so the aeon-absent path is structurally already
  exercised by the GBM-absent paths.

In `benchmarks/report/render.py` (B9 cross-experiment assembler):

- **(l) Skip-reason footnote pass-through**: no code change
  required. The per-experiment renderers already group the
  rendered `skipped_reason` string into their own footnotes;
  the assembler quotes those footnotes verbatim. The new
  `optional_dep_missing` reason renders automatically.

Observability: the per-dataset skipped-footnote renderer at
`raw_loss.py` groups by the `skipped_reason` string; the new
reason renders as a row automatically. The text in B12.1's
adapter docstring + this delta's commit message documents the
manual aeon-install path.

### B12.5 Test surface

All aeon-missing tests use
`monkeypatch.setitem(sys.modules, "aeon", None)` (pytest auto-
restores on teardown) rather than a bare `sys.modules` assign.
The `conftest.py` autouse `isolated_registry` fixture restores
the registry but NOT `sys.modules`; a bare assign would poison
`pytest-randomly` reordered runs.

`tests/benchmarks/test_tsc_adapter.py` (NEW, aeon-missing path):

- `test_check_aeon_available_raises_typed_error`: assert
  `OptionalDependencyMissingError` is raised by
  `_check_aeon_available()` with `package_name == "aeon"`.
- `test_check_aeon_available_does_not_subclass_importerror`:
  pin the MRO choice (R-B12-2 + the existing-`except ImportError`
  audit) so a future refactor that changes the base catches
  attention.
- `test_driver_records_optional_dep_skip_not_crash`: run
  `run_raw_loss` end-to-end with a registered TSC fake adapter +
  the monkeypatched `sys.modules["aeon"] = None`. With one TSC
  model on one dataset across `n_folds` folds (B3 default), the
  expected skip count equals the number of TSC cells the run
  emits. Assert
  `RawLossExperimentResult.cells_skipped_optional_dep_missing
  == n_tsc_cells_in_run`, `cells_skipped_adapter_error == 0`,
  the run returns cleanly (no exception escaped the driver),
  and every TSC ResultRow's `skipped_reason` starts with
  `"optional_dep_missing: aeon"`.
- `test_tsc_adapter_register_rejects_conflicting_spec`:
  attempting to re-register `rocket_classifier` with a different
  family raises (matches `registry/models.py:139-159` idempotent-
  on-identical + raise-on-conflict semantics).
- `test_tsc_adapter_post_init_blocks_direct_base_instantiation`:
  the analog of B10's
  `test_gbm_adapter_direct_base_instantiation_raises_type_error`
  at `tests/benchmarks/test_gbm_adapter.py:294-303`.

`tests/benchmarks/test_tsc_adapter_smoke.py` (NEW,
aeon-installed path):

- `pytest.importorskip("aeon")` at module top.
- Each smoke test passes a minimum-cost hyperparameter override
  to keep wall-clock < 5s: `num_kernels=100` for ROCKET /
  MultiRocket (vs the 10_000 default), default for Catch22
  (deterministic, no JIT). All three smoke tests carry
  `pytest.mark.slow`.
- ROCKET / MultiRocket / Catch22 smoke: fit + predict +
  predict_proba on the fake binary panel. Assert proba shape ==
  `(len(panel), 2)`, sum-to-1, finite, no NaN in the kept rows.
- `test_predict_proba_is_entity_homogeneous` (the R-B12-2
  oracle): group output rows by entity id; assert every row
  belonging to the same entity carries an IDENTICAL probability
  vector (`np.testing.assert_array_equal` within each group).
  This is the test that kills a buggy broadcast that permutes
  predictions across entities while still satisfying shape +
  sum-to-1.
- `test_predict_quantiles_raises_quantiles_unsupported`: the
  analog of `test_gbm_adapter.py:159-165`.

`tests/benchmarks/test_raw_mts.py` (NEW):

- `test_panel_to_tensor_shape_contract`: returns a 3D tensor of
  shape `(n_instances, n_channels, L_resolved)` plus a 1D
  mapping array of length `n_instances`.
- `test_panel_to_tensor_boundary_rows_exact`: entity with
  exactly `L_resolved` rows is accepted; no NaN markers emitted.
- `test_panel_to_tensor_boundary_rows_below`: entity with
  `L_resolved - 1` rows is excluded from the tensor and its
  panel rows are tagged for NaN broadcast at predict time
  (the mapping array does not include them).
- `test_panel_to_tensor_boundary_rows_above`: entity with
  `L_resolved + 1` rows is accepted after trailing-window clip
  to the last `L_resolved` rows; the mapping array points at the
  LAST-period row only.
- `test_panel_to_tensor_property_equal_length_happy_path`:
  Hypothesis: every panel satisfying F2 + per-entity
  time-monotone period + at least `L_resolved` rows per entity
  produces a tensor of the declared shape.
- `test_panel_to_tensor_property_short_entity_excluded`:
  Hypothesis inverse: every panel with at least one entity
  carrying fewer than `L_resolved` rows produces a tensor that
  excludes those entity's rows from the mapping array (the
  broadcast layer emits NaN for them).
- `test_panel_to_tensor_is_deterministic`: two calls on the same
  immutable input produce bit-identical `(X_3d, mapping)`.
- `test_raw_mts_drops_categorical_channels`: a panel whose spec
  declares `feature_categorical_cols` contains them in the
  source panel but they MUST be absent from the returned tensor's
  channel dimension. Pins the D-B12.6 deferral so a future
  partial revert that re-introduces ordinal encoding without
  the design decision is caught.
- `test_raw_mts_all_categorical_panel_raises`: a panel whose
  channel set is entirely categorical raises `RawMTSError` at
  reshape time.
- `test_raw_mts_returns_float32`: the cached tensor's dtype is
  `np.float32` regardless of source panel dtype.

`tests/benchmarks/test_hpo_tsc.py` (NEW, aeon-missing path
doesn't gate this; HPO suggest functions are pure pydantic +
Optuna trials, no aeon import):

- `test_tsc_hpo_suggest_params_shape`: ROCKET / MultiRocket /
  Catch22 each return dicts matching the documented keys.
- `test_tsc_hpo_family_lookup`: `get_hpo_registration("tsc")`
  returns a non-None registration; `search_space_size` is the
  documented per-family value.

`tests/benchmarks/test_run_manifest.py` (extension to existing
file, aeon-installed path):

- `test_aeon_version_captured_when_installed`: monkeypatch
  `importlib.metadata.version` to return a fake version for
  `"aeon"`; assert it appears in
  `EnvironmentFingerprint.package_versions`. This pins B12's
  addition of `"aeon"` to `_PINNED_PACKAGES` in
  `benchmarks/run_manifest.py`.

`tests/benchmarks/test_hpo_uplift_experiment.py` (extension to
existing file):

- `test_hpo_uplift_records_optional_dep_skip_not_crash`: run
  `run_hpo_uplift` end-to-end with a registered TSC fake adapter +
  the monkeypatched `sys.modules["aeon"] = None`; assert
  `HPOUpliftExperimentResult.cells_skipped_optional_dep_missing
  == n_tsc_cells_in_run`, the run returns cleanly, and every
  TSC ResultRow's `skipped_reason` starts with
  `"optional_dep_missing: aeon"`. Pins the B8 wire-up at
  touch points (g)-(j).

## Risk register (B12-specific)

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R-B12-1 | aeon's sklearn pin gets loosened mid-release-cycle; we want to ship the `benchmarks-tsc` extra. | Low | Pyproject.toml comment at lines 67-73 already names the trigger condition. B12 commit message references the trigger so a future maintainer can promote without re-reading the design. |
| R-B12-2 | The per-instance-to-per-row broadcast hides a single-class fold where the entity's last-period label is unrepresentative of the entity. | Medium | The broadcast is documented in the adapter docstring + the design delta. Per-row evaluation is the B5 contract; per-instance training is the aeon contract. The two are explicit and the broadcast is the only sane bridge. The smoke test `test_predict_proba_is_entity_homogeneous` asserts every row of an entity carries the same prediction vector. |
| R-B12-2b | The broadcast inflates the effective sample size for the B5.4 bootstrap CI: K rows per entity contribute K identical loss terms. Naive row-bootstrap reports CIs sqrt(K) tighter than the truth. | High | Panel data ALREADY requires entity-level (block) bootstrap for valid CIs across ALL families, not just TSC; row-bootstrap is the wrong default everywhere. B5.4 (not yet implemented) MUST resample by entity unconditionally for panel datasets. R8 reflects this: B12 records NO new schema field; the bootstrap consumer derives the broadcast detail from the adapter's family at B5.4 time. The Gemini final-pass corrected an earlier R8 draft that proposed a TSC-only `prediction_granularity` field, which would have encoded a TSC-only fix for a whole-suite issue. |
| R-B12-3 | Categorical channels dropped: the v1 raw-mts reshape excludes `feature_categorical_cols` entirely (see D-B12.6). Datasets whose channel set is mostly categorical will have reduced TSC channel breadth vs the seq + GBM comparators (which DO use categorical columns). | Medium | Documented in D-B12.6 + B12.2. The TSC family does not claim parity on categorical-heavy datasets at v1; the leaderboard's per-dataset row shows the dropped-channel count so the reader can interpret TSC scores honestly. A B12-followup adds one-hot or aeon's per-categorical wrappers. |
| R-B12-4 | Catch22's RidgeClassifierCV head is internally CV-cross-validated; the outer B3 fold structure plus the inner CV could leak across folds. | Low | RidgeClassifierCV's CV is over the TRAINING set only; no test-set rows leak. Documented in the adapter docstring. |
| R-B12-5 | aeon's classifier suite is heavy at install (numba JIT compilation, large transitive deps). A future CI run that opts into aeon would pay a startup tax. | Low | Out of scope for v1; the manual-install path is documented. The smoke tests use `pytest.importorskip` so aeon-missing CI stays fast. |

## Estimated effort

| Module | Size |
|---|---|
| `benchmarks/adapters/tsc.py` | Medium (~400 lines) |
| `benchmarks/protocol/raw_mts.py` | Small (~150 lines) |
| `benchmarks/hpo/tsc.py` | Small (~120 lines) |
| `benchmarks/adapters/_base.py` (add `OptionalDependencyMissingError`) | Small (~5 lines) |
| `benchmarks/experiments/raw_loss.py` (add skip arm + counter) | Small (~15 lines) |
| `tests/benchmarks/test_tsc_adapter.py` (aeon-missing) | Medium (~200 lines) |
| `tests/benchmarks/test_tsc_adapter_smoke.py` (aeon-installed) | Small (~120 lines) |
| `tests/benchmarks/test_raw_mts.py` | Medium (~250 lines, includes Hypothesis property tests) |
| `tests/benchmarks/test_hpo_tsc.py` | Small (~100 lines) |
| `pyproject.toml` (comment refresh; aeon stays outside any extra) | Trivial |
| `docs/benchmark_suite_implementation_plan.md` (Phase B12 actual-shape) | Small |

Total: ~1,360 lines, comparable to B10 (~1,235 lines).

## Addressed

R1 swarm (architecture-reviewer + qa-test-coverage + style-
reviewer). Style-reviewer APPROVE; the other two agents emitted
findings closed in this revision.

- arch-C1: `RawMTSError` base changed from undefined
  `BenchmarkError` to bare `Exception`. No umbrella class
  introduced (the existing `benchmarks/_io/integrity.py` errors
  are also bare `Exception` subclasses).
- arch-C2: R5 rewritten to drop per-family trial-count
  constants; the budget is profile-tier-keyed per the existing
  `_HPO_BUDGETS_BY_PROFILE` envelope at
  `benchmarks/experiments/hpo_uplift.py:115-119`. The HPO module
  surface no longer carries `_TSC_DEFAULT_TRIALS` or
  `_TSC_DEFAULT_TIMEOUT`.
- arch-C3 / qa-C3 (the same gap from two angles): R8 introduces
  `prediction_granularity: Literal["per_row",
  "per_entity_broadcast"]` on ResultRow; B12 records the
  broadcast marker on every TSC row. The B5.4 bootstrap will
  consult it (forward contract, since B5.4 itself has not
  shipped). A new test pins the marker via
  `test_predict_proba_is_entity_homogeneous`.
- qa-C1: B12.4 expanded to enumerate FIVE explicit driver-side
  touch points (typed exception, constant, narrow-tuple catch
  insert, counter init, classifier branch, summary field). The
  test surface adds
  `test_driver_records_optional_dep_skip_not_crash` to pin the
  end-to-end driver behavior.
- qa-C2: B12.5 mandates `monkeypatch.setitem(sys.modules,
  "aeon", None)` rather than a bare assignment so pytest auto-
  restores under `pytest-randomly`.
- qa-C4: B12.5 adds three named boundary tests:
  `test_panel_to_tensor_boundary_rows_exact`, `_below`, and
  `_above` for the `L_resolved` fencepost.
- arch-I1: "Six concrete adapters" corrected to "Three
  concrete adapters (classifier-only at v1 per D-B12.3)".
- arch-I2: `OptionalDependencyMissingError` now subclasses
  `Exception` directly, not `ImportError`, so the existing
  `except ImportError` clauses at `raw_loss.py:201` and
  `run_manifest.py:185, 196` cannot silently swallow it. A test
  (`test_check_aeon_available_does_not_subclass_importerror`)
  pins the MRO.
- arch-I3: B12.2 reshape contract now emits `np.nan` for
  below-floor entity rows (matching the deep model's strip
  convention) so the existing `_strip_below_floor_rows` machinery
  surfaces `n_below_floor_dropped` uniformly across all three
  adapter families.
- arch-I4: B12.4 enumerates all five driver-side touch points.
- arch-I5: B12.2 corrects the `resolve_lookback` call shape
  from the invented `model_default=` kwarg to the actual
  `override=None` signature.
- qa-I1: B12.5 adds
  `test_categorical_encoding_first_appearance_ordering` to pin
  the encoding convention so a regression to lexical sort
  fails.
- qa-I2: B12.4 now mandates extending
  `_PINNED_PACKAGES` in `benchmarks/run_manifest.py` with
  `"aeon"` so aeon-installed runs capture the version. A new
  `test_aeon_version_captured_when_installed` pins it.
- qa-I3: B12.5 adds the inverse Hypothesis property test
  `test_panel_to_tensor_property_short_entity_excluded`.
- qa-I4: B12.5 adds `test_panel_to_tensor_is_deterministic`.
- qa-I5: B12.5 mandates a minimum-cost
  `num_kernels=100` override on the ROCKET / MultiRocket
  smoke tests to keep wall-clock under 5s and avoid the aeon
  numba JIT compile path dominating CI time.
- NITs (arch-N1 R7 rewording, arch-N3 "rejects re-registration"
  wording, qa-N1 base-instantiation test, qa-N2
  predict_quantiles test): folded into the revised R7 + B12.5
  test list.
- NIT arch-N2 (Catch22 search-space dimensionality vs GBM's 6):
  B12.3 now declares the per-family `search_space_size=2` plus
  the per-classifier breakdown so the B6.4.0 parity-disclosure
  report renders honestly.
- NIT qa-N3 (renderer-text inaccuracy: skipped count surfaces
  in per-dataset footnote not run-metadata block): B12.4
  observability paragraph corrected.

R2 confirming swarm (architecture-reviewer + qa-test-coverage
+ style-reviewer). Style APPROVE; the other two agents
surfaced 2 NEW CRITICALs introduced by the R1 revisions plus
related IMPROVEMENTs and NITs.

- R2-arch-C1 / R2-qa-C1 (the same gap from two angles): the R1
  R8 addition of `prediction_granularity` to ResultRow has no
  declared default; non-TSC `_build_row` call sites in
  `raw_loss.py:609-623, 833-872` would raise `ValidationError`
  at construction. B12.4 now adds an explicit (g) touch point
  with `prediction_granularity: Literal[..., ...] = "per_row"`
  default + the rationale.
- R2-arch-C2: B12.4 header undercounted touch points by ONE
  (claimed five, body listed six). Header rewritten to enumerate
  the wire-up across four modules with eight lettered bullets
  (a-h); the additional (g) ResultRow field + (h)
  `_PINNED_PACKAGES` extension are now first-class touch
  points rather than scattered footnotes.
- R2-qa-C1 (no-test-for-marker): the R-B12-2b risk register
  entry promised a marker test that B12.5 did not deliver. B12.5
  now names `test_tsc_result_row_prediction_granularity_marker`
  to pin the field value on every TSC ResultRow (and inversely
  on non-TSC).
- R2-arch-I1 / R2-qa-I1: manifest-roundtrip extension test
  `test_result_row_prediction_granularity_roundtrips_through_parquet`
  added to pin the new field survives the parquet shard cycle.
- R2-arch-I2: `_PINNED_PACKAGES` extension promoted from a
  ledger-only mention to a first-class B12.4 touch point (h).
- R2-arch-N1: R5 self-reference to the R1 critique stripped;
  the Addressed ledger carries the audit history.
- R2-qa-N1 (cell count off-by-one in
  `test_driver_records_optional_dep_skip_not_crash`): assertion
  reworded from `== 1` to `== n_tsc_cells_in_run` so the test
  remains correct regardless of B3 fold default.
- R2-arch-N2: B12.4 header phrase rewritten from "driver-side"
  to "wire-up across four modules" since (a) lives in
  `_base.py` and (g)/(h) live in `manifest.py` /
  `run_manifest.py`, not in the driver itself.

Gemini final-pass (architecture-reviewer model). 3 CRITICAL +
2 IMPROVEMENT, all addressed in this revision:

- **Gemini-C1 (hpo_uplift.py wire-up gap)**: B12.4 enumerated
  raw_loss.py touch points but omitted the identical adapter-
  catch shape at `benchmarks/experiments/hpo_uplift.py:440-443,
  636-639`. Without the wire-up, aeon-missing HPO runs would
  crash. B12.4 now lists FOUR additional touch points (g)-(j)
  in hpo_uplift.py + a (k) for run_manifest.py + a (l)
  pass-through for the assembler. Total touch points: 12 across
  5 modules.
- **Gemini-C2 (R8 abstraction error)**: Gemini correctly
  identified that panel data needs entity-level (block)
  bootstrap for ALL adapter families. The R1+R2
  `prediction_granularity` ResultRow field encoded a TSC-only
  fix for a whole-suite issue. R8 rewritten: B12 adds NO
  schema field; B5.4 MUST resample by entity unconditionally.
  The corresponding (g) touch point on manifest.py was
  REMOVED, the
  `test_tsc_result_row_prediction_granularity_marker` and
  `test_result_row_prediction_granularity_roundtrips_through_parquet`
  tests were REMOVED, and R-B12-2b in the risk register was
  rewritten to point at the B5.4 design rather than a TSC-
  specific marker.
- **Gemini-C3 (categorical encoding for TSC)**: Gemini
  correctly identified that aeon's ROCKET/MultiRocket use
  random convolutions, which impose a metric space that ordinal
  encoding violates. v1 raw-mts reshape DROPS
  `feature_categorical_cols` entirely (numeric channels only).
  Documented in D-B12.6. The categorical-encoding test was
  replaced with `test_raw_mts_drops_categorical_channels`.
- **Gemini-I1 (pd.Categorical lexical-vs-first-appearance)**:
  Moot after Gemini-C3 (no categoricals reach the encoder).
- **Gemini-I2 (float32 tensor cache)**: B12.2 reshape now
  returns `np.float32` and the cached tensor is float32.
  Halves host RAM (5GB vs 10GB on a 100k x 200 x 64 tensor).
  Pinned by `test_raw_mts_returns_float32`.

## Deferred

R3 confirming swarm (architecture-reviewer + qa-test-coverage
+ style-reviewer): style APPROVE, arch APPROVE (1 doc-only
NIT), qa APPROVE (1 deferrable IMPROVEMENT + 1 NIT). No
CRITICALs.

- **R3-qa-I1 (deferred)**: a named test for the aeon-ABSENT
  fingerprint path (`importlib.metadata.PackageNotFoundError`
  on `"aeon"`). Reason: the absent-package handler at
  `benchmarks/run_manifest.py:_safe_pkg_version` lines 141-146
  is package-agnostic (catches `PackageNotFoundError` and
  returns `None`), so the aeon-absent path exercises the
  identical code path that the existing lightgbm / xgboost /
  catboost absent-runs already cover de-facto. The named test
  is non-load-bearing; if a future B12 implementer encounters a
  regression here, the symptom is loud (an entire run crashes
  at fingerprint time, not silent corruption).
- **R3-qa-N1 (deferred)**: the non-TSC default-value pin
  (`prediction_granularity == "per_row"` on non-TSC rows) only
  exercises in aeon-installed CI because
  `test_tsc_result_row_prediction_granularity_marker` lives in
  `test_tsc_adapter_smoke.py` (gated on `pytest.importorskip
  ("aeon")`). Reason: the default is a pydantic field-level
  default, not conditional logic; a wrong default would
  fail every other adapter's row construction immediately,
  surfacing on the next CI run regardless of aeon availability.
- **R3-arch-N1 (deferred)**: extending `_minimal_row` helper in
  `tests/benchmarks/test_manifest_roundtrip.py:37-63` with the
  new `prediction_granularity` field. Reason: the field has a
  default of `"per_row"`, so `_minimal_row` continues to work
  unchanged. Extending it would add coverage breadth but no new
  invariant. Stays out of B12 scope.
