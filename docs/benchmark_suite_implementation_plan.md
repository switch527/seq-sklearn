# Implementation plan: seq-sklearn comparative benchmark suite

## Requirements

The contracts to implement are RQ1-RQ10 plus B1-B8 of
[`docs/benchmark_suite_design.md`](benchmark_suite_design.md) (the
design has reached Claude-swarm consensus in three rounds; Gemini
final-pass deferred per user direction). This plan is authoritative
for the build order, the per-phase deliverable surface, and the CI
posture; the design doc is authoritative for the contracts the build
satisfies.

One delta to the design doc lands with this plan: B5's classification
metric set is extended to include the user-requested
`accuracy_score`, `precision_score`, `recall_score`, and `f1_score`
(macro + weighted averages for multiclass; binary form for binary).
The primary metric stays `log_loss` (the loss-first principle of B5
is unchanged). The expansion is a NEW row in B5 reported in every
classification leaderboard, not a re-ranking of the primary. A B5
delta-review is the first gate before code starts.

## Goals

- A working `benchmarks/` package in-repo that consumes the library
  via its public API, runs every model on every dataset on a fixed
  schedule with a frozen split + seed protocol, and writes a typed
  result manifest reproducible from the manifest alone.
- Four experiment types per the B6 contract, run in order: raw loss
  (the standalone deliverable), error-correlation /
  ensemble-complementarity, training-time, HPO uplift.
- Reports the standard practitioner metric set in addition to the
  loss-first primary: log-loss, accuracy, precision / recall / F1
  (with `macro_zd0` + `weighted_zd0` suffix split for multiclass
  per B5.1, `_zd0` for binary), ROC-AUC, PR-AUC, Brier, balanced
  accuracy, MCC, ECE for classification; RMSE, MAE, R², MAPE
  (with the three typed skip reasons for the pathological inputs
  per B5.2), pinball at the configured quantiles for regression.

## Principles

- **The benchmarks consume the library through its public façade.**
  No deep imports into `seq_sklearn._*`; every entry uses the
  STABLE surface in `seq_sklearn.__all__`. The benchmark suite is
  the second non-test consumer of the v1 façade and is the canary
  for whether the façade is actually a usable surface (the F1 / R1
  Phase 12 promotion was made under exactly that thesis).
- **Not in the wheel.** `benchmarks/` is excluded from the
  distribution by `pyproject.toml`'s build-system include / exclude
  rules and by the existing wheel-shape gate in
  `tests/deploy/test_wheel_install.py`. A new test asserts the
  exclusion explicitly.
- **Not a CI gate.** Benchmark runs are heavyweight; they do not
  block PRs. `pytest -m "benchmark"` is opt-in. Unit tests for the
  harness, registry, metrics module, and split / leakage invariants
  ARE in the default suite (they are CPU-only and small).
- **One config-driven entry point.** `python -m benchmarks.run` (or
  a console script) drives every experiment from a pydantic
  config; no scattered `if __name__ == "__main__"` scripts.
- **Library determinism mode for the deep models, fixed seeds for
  the comparators.** B8 reproducibility contract enforced by the
  harness, not optional.

## Phased plan

### Phase B0: Scaffold

**Goal**: a `benchmarks/` package that imports clean, lints, type-
checks, and runs an empty test suite green; one new `[benchmarks]`
extra in `pyproject.toml`; the wheel-exclusion gate live.

**Modules**:

- `benchmarks/__init__.py` (empty `__all__`).
- `benchmarks/config.py` (pydantic v2 `BenchmarkConfig`,
  `DatasetSpec`, `ModelSpec`, `ExperimentSpec`; field set scoped to
  what later phases need, `extra="forbid"`).
- `benchmarks/registry/__init__.py`, `registry/datasets.py`,
  `registry/models.py` (stub modules with the registration
  decorators and the `_REGISTRY` mappings; no entries yet).
- `benchmarks/run.py` (CLI skeleton; argparse for `--config`,
  `--experiment`, `--output`, `--dry-run`; no behavior beyond
  config-load and a logger banner).
- `pyproject.toml [project.optional-dependencies] benchmarks` extra:
  pinned dataset-loader deps (`openml>=0.15,<0.16`,
  `pyarrow>=21` for the on-disk cache; the upper bound is open
  because pyarrow's Python 3.14 wheel support starts at v21 and
  the cache format is stable across the major-version range),
  comparator-model deps (`lightgbm`, `xgboost`, `catboost`).
  `aeon` (for the classical TSC baselines) is intentionally NOT
  in this extra at v1.0.0 cut: aeon 0.11.x pins
  `scikit-learn < 1.6` which conflicts with the library's own
  `scikit-learn >= 1.6` requirement; the Phase B2 TSC adapter
  documents the manual-install path and a future aeon release
  that loosens its sklearn pin promotes the dep into a sibling
  `benchmarks-tsc` extra. Optuna already a core dep for HPO.
- `tests/benchmarks/__init__.py`, `tests/benchmarks/conftest.py`
  with `tmp_path`-based caches and a session-scoped fake-registry
  fixture for the next phases.
- `tests/benchmarks/test_scaffold.py` asserts `import benchmarks`
  works, `BenchmarkConfig` validates, the registry mappings exist,
  and `python -m benchmarks.run --dry-run --config <minimal>` exits
  zero.
- `tests/deploy/test_wheel_install.py` extension: a new
  parametrized case asserts `benchmarks/` is NOT inside the built
  wheel; complements the Phase 10 wheel-shape contract.

**Dependencies**: none.

**Deliverable tests**: scaffold module imports; pyproject parses;
the new wheel-exclusion test passes; `pytest tests/benchmarks/`
collects nonzero items and exits zero.

**Done when**: a PR opened with this phase shows all CI jobs green,
the wheel still builds without `benchmarks/`, and the `[benchmarks]`
extra installs cleanly.

### Phase B1: Dataset registry and loaders

**Goal**: the full registry SHAPE lands here (the typed
`DensificationPolicy` + `DatasetSpec` extension, the
`PanelDataset` materialized record, the integrity + cache layer,
the GATED-error pattern) plus a reference loader for each access
tier so the pattern is exercised end-to-end. The remaining B2
roster entries land in iterative follow-up branches
(``benchmark-phase-B1-<dataset>``) gated by the same
registry-invariants test so the cumulative roster cannot regress
silently. Negative-case datasets from B1.5 are registered as
documented exclusions so the rubric stays auditable.

**B1 in scope** (this phase): infrastructure + 2 reference loaders.
**B1.x in scope** (one branch per loader): every remaining B2 roster
entry. The classical-TSC entries (UCI-HAR via aeon; BasicMotions,
JapaneseVowels, UWaveGestureLibrary, LSST, PenDigits, FaceDetection,
InsectWingbeat via aeon) are blocked until aeon's `scikit-learn` pin
loosens; the impl plan B0 module list documents the blocker.

**Modules**:

Lands in this phase:

- `benchmarks/config.py` extension: nested `DensificationPolicy`
  pydantic model with `bin_width`, `aggregation`, `missing_bin_fill`
  fields; `DatasetSpec` extended per B2.2 with
  `archive_basename`, `feature_real_cols`, `feature_categorical_cols`,
  `observation_cutoff_rule`, `excluded`, `exclusion_reason`; the
  `_positive_label_only_on_binary` validator already from B0 plus
  new validators `_excluded_carries_reason` (B1.5 pin) and
  `_feature_cols_disjoint` (a column cannot be both numeric and
  categorical).
- `benchmarks/datasets/_base.py`: frozen-slotted `PanelDataset`
  dataclass (the loaded form: spec + DataFrame + target ndarray)
  and the `LoaderCallable = Callable[[Path], PanelDataset]` type.
- `benchmarks/_io/cache.py`: cache-root resolution
  (`~/.cache/seq-sklearn-benchmarks/` by default; overridable by
  `SEQ_SKLEARN_BENCHMARKS_CACHE` or `BenchmarkConfig.cache_dir`),
  archive-path resolution under `<cache_root>/archives/<dataset>/`,
  Parquet panel-cache path under `<cache_root>/panels/`, the
  `require_archive` helper that integrity-checks via the spec hash.
- `benchmarks/_io/integrity.py`: typed `DatasetIntegrityError` +
  `GatedDatasetUnavailableError` with a shared `DatasetIOError`
  base; chunked `verify_sha256` (1 MiB chunks so huge archives
  don't blow process RSS).
- `benchmarks/datasets/__init__.py`: side-effect imports of every
  loader module so each module's bottom-of-file
  `register_dataset(spec, load)` call fires at package import.
- `benchmarks/registry/datasets.py` extension: companion `_LOADERS`
  dict + `get_loader(name)` helper; `register_dataset` now takes
  the loader callable too and rejects same-name-different-loader.
  The `LoaderCallable` import is `TYPE_CHECKING`-guarded to avoid
  the circular between `benchmarks.registry` and
  `benchmarks.datasets`.

Reference loaders landed in this phase:

- `benchmarks/datasets/c_mapss_fd001.py`: NASA C-MAPSS FD001 (OPEN,
  regression, medium size). Parses the whitespace-separated
  `train_FD001.txt` from the official zip into an F2 panel; the
  target is RUL = `max_cycle - cycle` per engine. The integrity
  hash is a `deadbeef` placeholder until the release engineer
  pins the canonical archive SHA-256 from the upstream download.
- `benchmarks/datasets/amex_default.py`: Amex Default Prediction
  (GATED, binary classification, huge size; the design's flagship).
  Phase B1 ships the gated-error shape: when the archive is absent
  the loader raises `GatedDatasetUnavailableError` naming the
  Kaggle source URI, the archive basename, and the expected cache
  path. The in-cache parse path is a Phase B1-follow-up because the
  archive cannot be exercised in CI (Kaggle credentials required).

Tests landed in this phase:

- `tests/benchmarks/test_dataset_registry.py`: every registered
  dataset resolves to a loader, has a non-empty citation, binary
  entries carry `positive_label`, every irregular-time entry
  carries a `densification_policy`, every excluded entry has a
  reason.
- `tests/benchmarks/test_gated_dataset_raises.py`: parametrized
  over the GATED entries (Amex this phase; the parametrize grows
  as B1.x phases add the other GATED entries); asserts the typed
  raise + that the error message names the source URI, expected
  cache path, and archive basename.
- `tests/benchmarks/test_loaders_offline.py`: per OPEN loader,
  build a synthetic mini-archive in `tmp_path`, recompute its
  SHA-256, monkeypatch the spec, and assert F2-shape output.
  Also asserts the integrity-check fires on a byte-flipped archive
  and that a missing archive surfaces `FileNotFoundError`.
- `tests/benchmarks/test_io_layer.py`: cache-root resolution
  precedence, path creation, `verify_sha256` happy/mismatch/missing
  paths, `require_archive`, the typed-error-shares-base contract.

**Dependencies**: B0.

**Deliverable tests**: registry parametrized smoke; integrity
guard; gated-error guard; B1.3 negative cases still in the registry
as exclusions, asserted by name.

**Done when**: every B2 roster entry has a green offline test;
loaders that need network are marked `network` and gated behind
`SEQ_SKLEARN_BENCHMARKS_NETWORK=1`.

### Phase B2: Model registry and adapters

**Goal**: a single `SeqSklearnAdapter`-protocol surface that every
benchmark model implements. The protocol exposes `fit`, `predict`,
`predict_proba` (when applicable), `predict_quantiles` (when
applicable), and the class-level `supports_proba` flag.

**B2 in scope** (this phase): the protocol + the seq-sklearn
reference adapter pair (`SeqSklearnTFTClassifierAdapter`,
`SeqSklearnTFTRegressorAdapter`). The seq-sklearn pair is the
design's named canary for whether the v1.0.0 façade is genuinely
usable.

**B2.x in scope** (one branch per family): GBM adapter (LightGBM,
XGBoost, CatBoost via their sklearn-API), sklearn-passthrough
extension point, classical TSC family (MiniRocket / Rocket /
KNN-DTW via aeon, blocked until aeon's sklearn pin loosens; same
blocker as the UEA dataset loaders). The GBM and sklearn-
passthrough adapters consume the B3 featurizer; they land after
Phase B3 ships, because the harness's lag-feature builder is the
B3 deliverable and ships once for every tabular comparator.

**Modules**:

Lands in this phase:

- `benchmarks/adapters/_base.py`: the runtime-checkable
  `SeqSklearnAdapter` protocol with attributes `name`, `family`,
  `task_types: tuple[TaskType, ...]`, `supports_proba: bool` and
  methods `fit`, `predict`, `predict_proba`. The typed
  `ProbaUnsupportedError(NotImplementedError)` (qa-I3 / B3.2.1)
  for adapters that do not produce class probabilities.
- `benchmarks/adapters/seq_sklearn.py`: the reference adapter
  pair. `SeqSklearnTFTClassifierAdapter` wraps the library's
  `TFTClassifier` from `seq_sklearn.__all__`; the analog
  `SeqSklearnTFTRegressorAdapter` wraps `TFTRegressor` and also
  exposes `predict_quantiles` for the `regression_quantile` task
  type. Both adapters build the library's `TabularConfigParams`
  from the dataset spec and forward all `fit` / `predict` calls
  to the underlying estimator unchanged.
- `benchmarks/adapters/__init__.py`: side-effect import of the
  seq-sklearn adapter module so `register_model` fires at import.
- `tests/benchmarks/test_adapter_contract.py`: 11 protocol-only
  tests covering runtime-checkable `isinstance` against
  `SeqSklearnAdapter`, adapter-vs-registered-spec metadata
  consistency, `ProbaUnsupportedError` typed raise, predict-before-
  fit guards, task-type rejection, and the
  `ProbaUnsupportedError <: NotImplementedError` subclass contract.
- `tests/benchmarks/test_seq_sklearn_adapter_smoke.py`: 2 slow
  end-to-end smokes. Each builds a tiny synthetic panel, fits the
  adapter with `max_epochs=1`, and asserts `predict` shape +
  dtype. Proves the adapter actually drives the library through
  the v1.0.0 façade end to end (the design's canary).

Deferred to Phase B2-followup branches:

- `benchmarks/adapters/gbm.py` (LightGBM, XGBoost, CatBoost via
  sklearn-API): consumes the Phase B3 featurized panel; lands
  once B3 ships.
- `benchmarks/adapters/sklearn_passthrough.py` (extension point):
  same dependency on B3's featurizer.
- `benchmarks/adapters/tsc.py` (aeon MiniRocket / Rocket /
  KNN-DTW): blocked on aeon's `scikit-learn < 1.6` pin; the same
  blocker that holds the UEA dataset loaders.

**Dependencies**: B1 (the adapter contract takes a `PanelDataset`).

**Deliverable tests**: protocol-conformance parametrized over the
registry; typed errors fire; task-type filtering works.

**Done when**: every model in the v1 comparator set is callable via
the protocol; the registry is extensible by importing a new module
with one decorator call.

### Phase B3: Fair-comparison protocol

**Goal**: the B4 contract is enforced as code. One `L_resolved`
lookback per dataset is bound to the split, the deep model, the
featurizer for GBMs (B4.3), and the raw-MTS wrapper for TSC (B4.4).
`SeqSklearnAdapter` configures `val_split_strategy="time_ordered"`
(arch-IIb). The B4.5 named leakage / window-binding tests are live.

Lands in this phase:

- `benchmarks/protocol/lookback.py` (B4.1b): the single
  `L_resolved` resolver. Reads `spec.lookback` by default; admits
  an explicit override (Phase B5+ wires per-experiment overrides
  for sensitivity sweeps). Non-positive overrides raise typed.
- `benchmarks/protocol/split.py` (B4.1a): thin wrapper around the
  library's `EntityTimeSeriesSplit` (A9.1); binds `id_col` /
  `time_col` from the dataset spec; exposes `make_splitter`.
  Callers invoke `splitter.split(panel)` directly (the library's
  splitter already matches the sklearn protocol; an extra wrapper
  would have added API surface without justification).
- `benchmarks/protocol/featurize.py` (B4.3): the GBM /
  sklearn-passthrough lag-feature builder. Produces one row per
  panel row with L lagged columns per real / categorical feature;
  warm-up rows zero-imputed for real features and `""` for
  categoricals; a `missing_lag_count` tracking column. Pure
  transform (no fit state); F2 invariants asserted.
- `benchmarks/protocol/fingerprint.py` (B8.1): SHA-256 of the
  canonical fold serialization. Identical config + panel produces
  the same fingerprint (qa-C4); the `L_resolved + 1` direction-2
  perturbation flips it (qa-NEW-N1).
- `benchmarks/protocol/__init__.py`: package facade re-exporting
  `fingerprint_folds`, `lag_featurize`, `make_splitter`,
  `resolve_lookback`.
- `tests/benchmarks/test_protocol.py`: 28 tests covering each
  protocol leaf plus the B4.5 leakage invariants
  (`test_iter_folds_test_target_window_is_history_only_overlap`,
  `test_lag_featurize_train_perturbation_outside_lookback_changes_train_only`).
  Coverage: 100% line + 100% branch on every protocol module.

Deferred to Phase B3-followup:

- `benchmarks/protocol/mts.py` (B4.4): raw-MTS reshape for TSC,
  with F3 `padding_mask` -> aeon convention adapter. Blocked on
  the same aeon `scikit-learn < 1.6` pin that holds the TSC
  adapters; lands once aeon's pin loosens or a compatible alt
  ships.
- `tests/benchmarks/test_lookback_binding.py` (cross-consumer
  identity invariant across all four consumers): lands once
  Phase B3-followup wires the GBM adapter through the featurizer
  and the TSC adapter through the MTS reshape, so the test has
  four consumers to assert identity over. Today the featurizer
  + splitter + deep-model adapter consume the L value directly
  from the experiment driver, so binding-vs-spec is enforced at
  the call site rather than the test layer.

**Dependencies**: B1, B2.

**Deliverable tests**: split contract, featurizer F2 invariants,
lookback identity, fingerprint stability, the two B4.5 leakage
invariants. The third B4.5 named test
(`test_seq_sklearn_adapter_val_split_strategy_is_time_ordered`)
lands once `val_split_strategy="time_ordered"` is added to the
seq-sklearn adapter's `fit` kwargs in Phase B3-followup; the
library already defaults to time-ordered, so the assertion is
written but pending the adapter-side hook.

**Done when**: a benchmark run on any registered dataset emits a
fingerprinted split and the deep / GBM arms both consume the
same lookback. TSC consumer arrives with the MTS-reshape
follow-up.

### Phase B4: Metrics

**Goal**: every metric in the extended B5 set computed via a pinned
delegated call or formula with a known-value test. The user-requested
extension (accuracy, precision, recall, F1) is folded in here with
the multiclass averaging strategy spelled out.

**Modules**:

- `benchmarks/metrics/classification.py`: thin wrappers around
  `sklearn.metrics.{log_loss, accuracy_score, precision_score,
  recall_score, f1_score, roc_auc_score, average_precision_score,
  balanced_accuracy_score, matthews_corrcoef, brier_score_loss}`;
  ECE at `ece_q15` (15 equal-mass quantile bins). Precision /
  recall / F1 reported with `average="binary"` and
  `pos_label=spec.positive_label` (B2.2) for binary, and BOTH
  `average="macro"` and `average="weighted"` for multiclass under
  suffixed metric names `*_macro_zd0` / `*_weighted_zd0` (the
  `_zd0` suffix names the `zero_division=0` convention in the
  metric name itself, the same way `ece_q15` names the binning).
- `benchmarks/metrics/regression.py`: thin wrappers around
  `sklearn.metrics.{root_mean_squared_error, mean_absolute_error,
  r2_score, mean_pinball_loss, mean_absolute_percentage_error}`;
  MAPE pre-checked for the three input pathologies per B5.2
  (`y_true == 0`, `np.isnan(y_true)`, `np.isinf(y_true)`); each
  emits `nan` and records the corresponding typed skip reason
  (`mape_undefined_zero_in_y_true`,
  `mape_undefined_nan_in_y_true`,
  `mape_undefined_inf_in_y_true`).
- `benchmarks/metrics/resource.py`: wall-clock fit seconds,
  peak process RSS, peak CUDA memory, per-sample inference latency
  (median + p95) (B5.3).
- `benchmarks/metrics/__init__.py`: a single
  `compute_all(task_type, y_true, y_pred, y_proba=None,
  quantiles=None)` entry point returning a typed `MetricsRecord`
  pydantic model with every metric as a typed field; missing
  fields per task type are absent rather than `nan` (the type
  itself documents which metrics apply).
- `tests/benchmarks/test_metrics_known_values.py`. Three fixtures
  carry the oracle arithmetic inline (no derivation from a prior
  sklearn run; the test reads as the formula):
  - A 2-class 6-sample binary fixture with intentionally unequal
    class counts (4/2) that exercises `average="binary"` with
    `pos_label=spec.positive_label`. Pinned values for
    `accuracy`, `precision_zd0`, `recall_zd0`, `f1_zd0`,
    `log_loss`, `roc_auc`, PR-AUC, MCC, Brier (binary).
  - A 3-class 4-sample multiclass fixture with class 2 NEVER
    predicted in `y_pred` (TP=FP=0 for class 2) so the
    `zero_division=0` branch fires non-trivially in
    `precision_macro_zd0` and `precision_weighted_zd0`. Class 2 is
    kept in `y_true` (supports 2/1/1) so the ROC-AUC OVR pathway
    has a positive per class; the unequal counts also drive
    `macro != weighted`. Pinned values for
    `accuracy`, `precision_macro_zd0`, `precision_weighted_zd0`,
    `recall_*`, `f1_*`, `log_loss`, Brier multiclass.
  - An 8-sample regression vector with at least one negative
    `y_true` so MAPE has a real denominator on every row and the
    hand-computed value is an inline arithmetic check; pinned for
    `rmse`, `mae`, `r2`, `mape`, and pinball at q=0.5.
- `tests/benchmarks/test_metrics_records.py`: per-task-type record
  shape with explicit assertions, NOT just field counts: the
  multiclass record carries `precision_macro_zd0` AND
  `precision_weighted_zd0` (and corresponding recall/F1) and does
  NOT carry bare `precision` / `recall` / `f1`; the regression
  record carries `mape` plus a `mape_skip_reason: str | None` field;
  the binary record carries the unsuffixed `precision_zd0` /
  `recall_zd0` / `f1_zd0`.
- `tests/benchmarks/test_mape_pathologies.py`: parametrized over
  the three pathological inputs (zero, nan, inf in `y_true`),
  asserting `nan` value + the typed skip reason string.
- `tests/benchmarks/test_macro_vs_weighted_distinguishable.py`: a
  3-class fixture with unequal class counts asserts
  `f1_macro_zd0 != f1_weighted_zd0`, proving the two averages are
  computed by separate code paths and not silently collapsed to
  the same call.

**Dependencies**: B0, B1 (the binary `positive_label` field on
`DatasetSpec`).

**Deliverable tests**: the inline-arithmetic known-value tests
above; pydantic record shape per task type; MAPE pathologies
(zero / nan / inf); PR-AUC step definition (qa-C5); Brier
multiclass (qa-NEW-C1); `zero_division=0` fires non-trivially in
the 3-class fixture; `macro != weighted` in the unequal-counts
fixture.

**Done when**: the metric set is the union of the design B5 and
the user-requested extension, every metric is pinned by an oracle
test, and the record type is the single object the rest of the
harness emits.

**B4-followup deferrals** (carried into B5 / B6):

- The resource module's torch hooks (`cuda_reset_callable`,
  `cuda_peak_callable`) are injected as `Callable`s so the metrics
  module does not import torch; the harness wires them when CUDA
  is visible. The harness itself arrives in B6.
- `peak_rss_bytes` reports `getrusage().ru_maxrss`, which is bytes
  on Linux and kilobytes on macOS. The harness normalizes to bytes
  before manifest emission per the B7 envelope-tier check.
- `compute_roc_auc` on multiclass uses sklearn's default behavior
  (classes from `y_true`); if a fold has a class absent from
  `y_true`, sklearn raises. B6 routes the per-fold call through a
  try / except that records the metric as `nan` with a typed skip
  reason analogous to the MAPE pathway. This was scoped out of B4
  to keep the module sklearn-API thin.
- `compute_ece_q15` with small `N` (< 15 samples) leaves a subset
  of the 15 equal-mass quantile bins empty; the empty-bin
  `if not mask.any(): continue` guard at the top of the bin loop
  handles this branch but is not exercised by an explicit
  small-`N` test. B5 / B6 fixtures all carry many more than 15
  samples per fold, so the branch is structurally safe; if a
  future small-dataset roster lands, add a parametrized small-`N`
  ECE oracle test alongside the existing empty-array test.

### Phase B5: Experiment 1, raw loss comparison (deliverable)

**Goal**: the B6.1 deliverable. For every (dataset, model)
applicable cell, default config, fixed seeds, library determinism
on for deep models; per-dataset leaderboard ranked by the B5
primary loss, plus the full extended-B5 secondary table.

**Modules**:

- `benchmarks/experiments/raw_loss.py`: the experiment driver.
  Reads a `BenchmarkConfig`, iterates dataset x model cells,
  applies B3 protocol, computes metrics via B4, persists to the
  manifest.
- `benchmarks/manifest.py` (B7.2 atomic shard-then-sentinel):
  one Parquet shard per cell + a sentinel JSON per run;
  `manifest.load(run_id)` reconstructs the leaderboard table.
- `benchmarks/report/raw_loss.py`: leaderboard rendering to
  Markdown + a printable summary, ranked by the B5 primary;
  secondary metrics in a sortable table.
- `benchmarks/run.py` extension: `--experiment=raw_loss` routes here.
- `tests/benchmarks/test_raw_loss_experiment.py`: end-to-end on a
  synthetic 2-dataset x 3-model cell (CPU-only, slow-marked); the
  manifest round-trips; the leaderboard renders.
- `tests/benchmarks/test_manifest_roundtrip.py`: one shard then
  sentinel atomic; resume from a partial manifest produces the
  same leaderboard (qa-C3).

**Dependencies**: B1, B2, B3, B4.

**Deliverable tests**: synthetic e2e; manifest atomicity; resume.
The B6.1 output is the standalone deliverable.

**Done when**: a run on every B2 roster dataset with every
registered model emits a leaderboard. Per the design's compute
tiering, the actual full-roster run happens once on the workstation;
CI runs the synthetic 2x3 cell only.

**B5-followup deferrals** (carried into B6 / later):

- The seq-sklearn regressor adapter exposes `predict_quantiles`
  but the `SeqSklearnAdapter` Protocol does NOT carry a
  `quantile_levels: tuple[float, ...] | None` attribute, so the
  pinball-column labels (`pinball_q{level}`) cannot be assembled
  from the protocol surface. The B5 driver emits
  `skipped_reason="regression_quantile_b5_followup"` for every
  `regression_quantile` cell. The followup branch extends the
  Protocol with `quantile_levels` and lands the pinball reporting.
- `benchmarks.metrics.resource.measure_fit` accepts injected CUDA
  callables (`cuda_reset_callable`, `cuda_peak_callable`); B5's
  driver always passes `cuda_available=False` because the harness
  has no CUDA detection layer yet. B6 wires
  `torch.cuda.reset_peak_memory_stats` /
  `torch.cuda.max_memory_allocated` from the harness's environment
  detector so deep-model rows carry `peak_cuda_bytes`.
- Out-of-fold prediction persistence (B6.2.1) is NOT in the B5
  shard schema; the manifest carries metrics only. B6 (ensemble)
  adds a per-cell `predictions/` sibling directory with the
  prediction tensors (y_pred / y_proba / y_quantiles) so the
  pairwise correlation analysis has the inputs it needs.
- `_DEFAULT_N_SPLITS=5` is hardcoded in the driver. B5's
  `BenchmarkConfig` does NOT carry an `n_splits` knob; B6 adds it
  to the per-experiment spec for sensitivity sweeps.
- Library git SHA recording uses a best-effort `git rev-parse HEAD`
  subprocess; "unknown" is recorded on non-checkout invocations.
  B7's profile field tightens the hardware-tier identifier from
  the current free-form string ("cpu" | "gpu_single") to the
  library's `HardwareTier` enum.
- B5 records ONE git SHA (`library_git_sha`); design B8.1 names
  two (`library_git_sha` + `benchmarks_git_sha`) in anticipation
  of a future repo split. The library and benchmarks live in the
  same repo today, so the second SHA is identically the first.
  The repo-split branch reintroduces `benchmarks_git_sha` to the
  `RunEnvironment` + `ResultRow` schema; B5 ships with the single
  field so a leaderboard reader does not see two identical SHAs
  and misread it as redundancy.

**B5 R3-residual nitpicks** (not blocking ship; recorded for B6+):

- `_optional_scalar_columns` discriminates `float | None` vs
  `int | None` via `str(annotation)` equality, which depends on
  pydantic's annotation-repr stability. B6 can swap to a typed
  introspection (e.g. `typing.get_args` + `Union[X, None]` check)
  once the schema grows to other optional scalar dtypes.
- `_strip_below_floor_rows`'s int-y_pred branch (classifier-only
  point prediction with no probabilities) is unreachable from
  every current adapter and is left as a defensive no-op; if a
  future adapter ships with `supports_proba=False` on a
  classification task, the precedence skip catches the cell
  before this branch matters.
- The B6 impl-plan entry should ledger the new
  `cells_skipped_proba_runtime_unavailable` counter alongside the
  existing four skip categories so a reader of the result summary
  knows the full set without consulting the source.

### Phase B6: Experiment 2, ensemble complementarity (B6.2)

**Goal**: B6.2 error correlation / ensemble complementarity for
each (dataset, model-A, model-B) pair where A is a seq-sklearn
model and B is an external comparator; Q-statistic, disagreement
rate, double-fault rate, φ-coefficient with the qa-NEW-I1
degenerate-agreement convention.

**Modules**:

- `benchmarks/experiments/ensemble.py`: the pairwise driver.
  Reuses Phase B5's manifest as input (residuals + predictions are
  persisted there).
- `benchmarks/metrics/pairwise.py`: Q / disagreement / double-fault
  / φ; degenerate-agreement convention (perfect agreement -> 1.0
  for Q; otherwise `nan` and excluded); known-value tests
  (qa-C6, qa-NEW-I1, qa-III-N1).
- `benchmarks/report/ensemble.py`: pairwise table + a top-N
  ensemble-candidate-pair summary.
- `tests/benchmarks/test_pairwise_metrics.py`: known-value tests
  per formula; degenerate cases.
- `tests/benchmarks/test_ensemble_experiment.py`: synthetic
  pairwise run consumes a Phase B5 manifest.

**Dependencies**: B5 (the input is the raw-loss manifest).

**Deliverable tests**: pairwise oracles; experiment-driver smoke.

**Done when**: a full pairwise matrix renders from a B5 manifest;
the report identifies the most-complementary external comparator
for each library model.

### Phase B7: Experiment 3, training time (B6.3)

**Goal**: per-model, per-dataset training wall-clock with the
matched compute budget, recorded separately from the B5 latency
numbers in the resource record.

**Modules**:

- `benchmarks/experiments/training_time.py`: drives the per-cell
  fit-only path; persists wall-clock + RSS + CUDA-mem to the
  manifest under the same shard layout.
- `benchmarks/report/training_time.py`: table by dataset x model.
- `tests/benchmarks/test_training_time_experiment.py`: synthetic
  cell smoke; the wall-clock field is monotonic + nonzero.

**Dependencies**: B5 (shares the manifest).

**Deliverable tests**: per-cell smoke; resource-record round-trip.

**Done when**: a training-time table renders alongside the
leaderboard.

### Phase B8: Experiment 4, HPO-uplift (B6.4)

**Goal**: B6.4 measures the improvement from full Optuna HPO over
default config per the disclosed search-space parity policy
(arch-I1, B6.4.0). Per-model `n_trials` cap + `timeout=` enforced
(qa-I5, B7.4); Friedman + Holm correction applied across the matrix
(qa-I1, B7.5).

**Modules**:

- `benchmarks/experiments/hpo_uplift.py`: drives the (default,
  tuned) pair per cell. Uses `seq_sklearn.suggest_params` +
  `seq_sklearn.optuna_trial_guard` for the library models; per-
  comparator search-space modules under
  `benchmarks/hpo/<family>.py`.
- `benchmarks/hpo/seq_sklearn.py`, `benchmarks/hpo/gbm.py`,
  `benchmarks/hpo/tsc.py`: per-family search-space definitions.
- `benchmarks/stats/friedman.py`: Friedman + Holm correction over
  the model-vs-dataset matrix (qa-I1, B7.5).
- `benchmarks/report/hpo_uplift.py`: per-dataset Δ table (default
  vs tuned) + the Friedman / Holm matrix.
- `tests/benchmarks/test_hpo_uplift_experiment.py`: synthetic
  cell; the search-space parity disclosure renders in the report.
- `tests/benchmarks/test_friedman_holm.py`: known-value test of
  the matrix-level test.

**Dependencies**: B5 (default arm reuses the raw-loss manifest).

**Deliverable tests**: HPO-uplift experiment smoke; Friedman/Holm
oracle; HPO budget cap enforcement.

**Done when**: a tuned-vs-default leaderboard renders with the
search-space parity disclosure and a p-value column.

### Phase B9: Reproducibility manifest, report assembly, governance

**Goal**: every benchmark run emits a B8 manifest sufficient to
reproduce the run from the manifest alone. The four experiment
reports are assembled into a single Markdown deliverable.

**Modules**:

- `benchmarks/manifest.py` extension: the run manifest is a typed
  pydantic record with the library version, the comparator
  versions, the dataset SHAs, the split fingerprints, the seeds,
  the environment fingerprint (CPU/GPU model, CUDA version,
  driver), the per-cell row layout. Round-trip is the
  `test_manifest_roundtrip` invariant from B5.
- `benchmarks/report/__init__.py`: assembles the four experiment
  reports into `report.md`; references the manifest by path.
- `benchmarks/report/render.py`: the Markdown emitter (tables,
  the per-dataset cards, a one-page executive summary).
- `docs/benchmarks/` (NEW): a static index page under
  `docs/explanation/` that explains the methodology + links to the
  most-recent published `report.md` artifact (the report itself
  is generated, not checked in).
- `tests/benchmarks/test_manifest_roundtrip.py` extension:
  full-manifest round-trip including the new env fingerprint.

**Dependencies**: B5, B6, B7, B8.

**Deliverable tests**: full manifest round-trip; report assembly
on a synthetic 2x3 run.

**Done when**: one command (`python -m benchmarks.run --config v1
--experiment all`) produces a fingerprinted manifest + an assembled
`report.md`; a re-run from the manifest alone reproduces the same
report.

## Risk register

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R-B1 | Dataset access changes break a loader between releases. | Medium | B2.2 SHA-256 archive guard; loaders pin source URI + integrity; CHANGELOG entry on every roster change. |
| R-B2 | Comparator-version drift makes "vs LightGBM" comparisons non-reproducible. | Medium | B9 manifest pins comparator versions; the `[benchmarks]` extra uses upper bounds, not floors-only. |
| R-B3 | A new gated dataset requires a manual auth step that surprises a contributor. | Low | B2.3 typed `GatedDatasetError` with a clear remediation message; gated entries called out explicitly in the registry. |
| R-B4 | The single-GPU workstation envelope is exceeded by a large-tier dataset. | Medium | B7 compute tiering already pins envelopes; the largest tier carries a subsample fallback flagged in the manifest. |
| R-B5 | HPO-uplift wall-clock blows the workstation budget. | High | B7.4 hard caps: per-model `n_trials` and `timeout=`; the harness records the cap and refuses to run beyond it. |
| R-B6 | Adding accuracy / precision / recall / F1 invites the wrong primary-metric ranking. | Low | B5 stays loss-first; the extension is reported, not used to rank. Leaderboards explicitly label the primary. |

## PR workflow per phase

Same shape as the library's existing phase rhythm:

1. Open a feature branch (`benchmark-phase-Bn-<short>`).
2. Land the per-phase modules + tests.
3. Run `/review` on the branch's diff to consensus (Claude swarm).
4. `/gemini-final-pass code` once Gemini quota permits.
5. Merge to `main` via `--no-ff` with a phase-summary commit.

Phases B0-B4 are infrastructure and can land in quick succession.
Phases B5-B8 each end with a deliverable artifact (the leaderboard,
the ensemble report, the training-time table, the HPO uplift table);
each one is reviewable in isolation.

## Estimated effort

| Phase | Description | Estimated size |
|---|---|---|
| B0 | Scaffold + wheel-exclusion gate | Small |
| B1 | Dataset registry + loaders (B2 roster size pending) | Large |
| B2 | Model registry + adapters | Medium |
| B3 | Fair-comparison protocol (B4 + B4.5 named tests) | Medium |
| B4 | Metrics module (B5 + extension) | Small |
| B5 | Raw loss experiment + manifest + report (deliverable 1) | Medium |
| B6 | Ensemble complementarity (deliverable 2) | Small |
| B7 | Training-time experiment (deliverable 3) | Small |
| B8 | HPO uplift + Friedman/Holm (deliverable 4) | Medium |
| B9 | Manifest + assembled report | Small |

## Addressed

This section accumulates the design-doc B5 delta (the user-requested
metric extension) and any future iterations.

- B5 metric set extended to include `accuracy_score`,
  `precision_score`, `recall_score`, `f1_score` (binary `average=
  "binary"`; multiclass reported under both `macro` and `weighted`
  suffixes) and `mean_absolute_percentage_error` (with the
  zero-row skip path). Primary metric unchanged
  (`log_loss` / `root_mean_squared_error`). The extension is
  Phase B4 scope. Folded into the design via a B5 delta after this
  plan is accepted.

## Deferred

- D1 (multi-label datasets): tracked in the design's Deferred
  section, ships with v1.1.
- D2 (multi-GPU): out of scope for v1 single-GPU envelope.
- D3 (HIVE-COTE 2.0 / MultiRocket-Hydra): registry-extensible later.
- D4 (Sphinx-nav inclusion): the design doc + this plan stay under
  `docs/design/` via the Phase 12 toctree until there is published
  benchmark content; the index page in B9 is the bridge.
