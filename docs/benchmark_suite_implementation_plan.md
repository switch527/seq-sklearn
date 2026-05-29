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

**B6-followup deferrals** (carried into B7 / later):

- B6.2.5 (complementarity-ensemble): the design names "a GBM-only
  ensemble and a GBM+seq-sklearn ensemble on the same folds; report
  ΔLogLoss / ΔRMSE of adding the deep model". B6 ships the
  PAIRWISE-statistics half (Q / phi / disagreement / double-fault
  / pearson_pred / spearman_pred / pearson_error_corr) plus a B6
  proxy "complementarity_score" of `(1 - error_corr) + disagreement`
  for the top-N ranking. The formal stacked-meta-learner Δloss
  ships in a B6-followup branch alongside the first external GBM
  adapter that lets the ensemble be built; the B6 report's
  ranking key swaps to ΔLogLoss / ΔRMSE at that point.
- Out-of-fold quantile predictions (regression_quantile cells):
  inherit the B5 deferral. The B5 driver already skips quantile
  cells with `regression_quantile_b5_followup`; B6 inherits the
  skip and the followup branch lands quantile pinball-correlation
  + pairwise stats alongside the protocol extension.
- Significance testing (B7.5): the design names Wilcoxon
  signed-rank on per-dataset `(GBM, GBM+seq)` Δloss with Holm
  correction. B6 does not run the test (no ensemble built yet);
  the report renders raw means and the formal test lands with the
  B6.2.5 ensemble in the B6-followup branch.
- Class-label sidecar: prediction shards encode classes as the
  integer column index `y_proba_{k}`. The B6 driver's
  `_resolve_classification_classes` returns `np.arange(n_proba_
  columns)` and `classification_nll` matches it row-by-row against
  `y_true`. For datasets whose true label set is non-integer
  (string labels, gapped integers like `[1, 2, 3]`), this
  mis-aligns the per-sample NLL silently (no exception; the
  `pearson_error_corr` value is just wrong). A B6-followup adds a
  `classes.json` sidecar per dataset so non-default label sets
  round-trip through the pairwise driver and `classification_nll`
  reads its own canonical class array.

**B6 R1-residual deferrals** (carried into B7 / later):

- Atomic-write helper (`_atomic_write_bytes` /
  `_atomic_write_parquet`) is now duplicated across three modules
  (`benchmarks/manifest.py`, `benchmarks/predictions.py`,
  `benchmarks/experiments/ensemble.py`). The three implementations
  are byte-identical thin wrappers around the
  write-to-temp-then-`Path.replace` POSIX-atomic pattern; a shared
  `benchmarks/_atomic_io.py` utility is the natural extraction.
  Deferred to a B6-followup branch so the R1 fix commit stays
  scoped to the swarm CRITICALs.
- `benchmarks/report/ensemble.py` imports `load_pairwise` from the
  experiment driver (`benchmarks.experiments.ensemble`); the B5
  symmetry is for the report to read via a sibling manifest module
  (`benchmarks/manifest.load_run`). A B6-followup splits the
  pairwise shard-IO surface into a `benchmarks/pairwise_manifest.py`
  mirror so the report module no longer reaches into the
  experiment driver.
- `PairwiseRow` records `n_samples` (the inner-join intersection
  size between two models' prediction shards) but loses the per-
  side strip counts (how many rows each model dropped). A B6-
  followup adds `n_a_only` and `n_b_only` columns to capture the
  partial-overlap shape; the current schema records only the
  intersection size.

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

**B7 actual shape** (what shipped):

The B5 driver already captures `wall_seconds`, `peak_rss_bytes`,
and `peak_cuda_bytes` per cell via `measure_fit`. B7 therefore
ships as a REPORT-ONLY phase over the existing B5 manifest:

- `benchmarks/report/training_time.py` aggregates the B5 manifest
  by `(dataset_name, model_name, hardware_tier, task_type)`,
  computes mean+std wall-clock + mean+max RSS / CUDA bytes, and
  renders a per-dataset Markdown table sorted by
  `wall_seconds_mean` ascending (fastest first). Fully-skipped
  groups land in a footnote.
- `benchmarks/experiments/training_time.py` is a thin driver:
  `run_training_time(config, *, output_root, env)` reads the B5
  manifest, calls the renderer, writes
  `output_root/training_time.md`, returns a
  `TrainingTimeExperimentResult` with `groups_evaluated +
  groups_fully_skipped + report_path` counters. No new shard
  layout; no per-cell fit-only re-execution.
- `--experiment=training_time` CLI dispatch wired in `run.py`.
- 17 tests across `tests/benchmarks/test_training_time_report.py`
  (aggregator + renderer edges) and
  `tests/benchmarks/test_training_time_experiment.py` (full
  B5 -> B7 e2e + error paths).

**B7-followup deferrals** (carried into B8 / later):

- Scaling-curve plot: design B6.3 names "reported against dataset
  size so the scaling curve is visible". The B5 manifest does not
  carry a per-row dataset-size column today; the B7-followup
  adds a `n_train_rows` / `n_test_rows` column to `ResultRow` and
  the report module renders the wall-clock-vs-size curve
  alongside the table.
- Repeated-fit timing: B5 fits exactly once per cell so the
  `wall_seconds_std` is the seed/fold dispersion, not within-cell
  jitter. A B7-followup with N-rep fit timing (for stable
  wall-clock under noisy systems) is available behind a profile
  flag.
- Profile-tier compute budget: B6.3 names the matched compute
  budget per profile (smoke / standard / full per B7.1). B5's
  driver doesn't enforce a per-cell wall-clock cap today; the
  followup wires the budget through the adapter's `fit` callable.

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

**B8 actual shape (as shipped)**:

- `benchmarks/stats/friedman.py`: `friedman_holm(matrix,
  reference_model)` wraps `scipy.stats.friedmanchisquare` for the
  omnibus and Wilcoxon signed-rank for pairwise vs the designated
  reference; `holm_correction(p_values)` is the reusable
  Holm-Bonferroni step-down. NaN-bearing dataset columns are
  dropped before testing, with `family_size` recorded on the
  result so the report can disclose it.
- `benchmarks/hpo/{__init__,_base,seq_sklearn}.py`: per-family
  HPO-space registry. Only the `seq_sklearn` family ships at B8
  (the only registered adapter family today); GBM and TSC follow
  with their B2-followup adapter branches. The registry seam is
  `register_hpo_space(HPOSpace, sampler) -> None` /
  `get_hpo_space(family) -> (HPOSpace, sampler)`.
- `benchmarks/experiments/hpo_uplift.py`: per-cell driver that
  runs an Optuna TPE study on an inner train/val split (B6.4.2:
  the test fold is untouched), refits the best config on the full
  train fold, and writes one `variant="tuned"` `ResultRow`. The
  default arm is the B5 manifest; the driver does NOT re-run it.
- `benchmarks/report/hpo_uplift.py`: joins default + tuned rows
  per `(dataset, model, seed, fold)`, computes Δ on the primary
  loss (B6.1 mapping), aggregates per `(dataset, model)`, and
  renders the table + Friedman/Holm matrix + skipped-groups
  footnote. The renderer keys `by_dataset` on
  `(dataset_name, task_type)` for the same reason B7 does
  (across-run config drift can produce heterogeneous task_types
  for one dataset name).
- `ResultRow` extension (`benchmarks/manifest.py`): seven new
  optional fields (`hpo_n_trials_completed`,
  `hpo_n_trials_pruned`, `hpo_search_space_size`,
  `hpo_timeout_seconds`, `hpo_best_trial_number`,
  `hpo_time_to_best_seconds`, `hpo_best_val_loss`). All default
  to `None`; B5/B6/B7 rows still validate cleanly.
- `ExperimentSpec` extension (`benchmarks/config.py`):
  `n_trials: int | None` and `timeout_seconds: float | None`
  HPO-budget overrides; profile-tier defaults (smoke=0,
  standard=25, full=100) apply when both are `None`.

**B8-followup deferrals** (carried into B9 / later):

- GBM and TSC search-space modules ship with their adapter
  families (the B2-followup branches). Cells whose model family
  has no HPO space registered today produce a typed
  `skipped_reason="hpo_family_not_registered"` row so the report's
  footnote lists them.
- Nemenyi critical-difference diagram (design B7.5): the
  Friedman + Holm matrix is rendered as a table at B8; the
  graphical CD diagram lands in B9's visualization pass.
- Predictions persistence for tuned cells: B6.2.1 writes a
  per-cell predictions shard for B5. B8 does NOT write a
  predictions shard for the tuned arm so the ensemble report
  cannot inadvertently mix default + tuned variants into one
  pairwise cell. B9 wires variant-aware predictions if needed.
- Cross-driver error-handling pass (carried over from Gemini's
  B7 final-pass deferrals): every driver currently raises
  `ValueError` on an empty/missing manifest. A unified
  CLI-side `try / except ValueError -> exit 1` lives in B9.

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

**B9 actual shape (as shipped)**:

- `benchmarks/run_manifest.py` (NEW): `RunManifest` +
  `EnvironmentFingerprint` pydantic records, atomic write/load
  via the same shard-then-rename pattern as B5's per-cell shards,
  best-effort env capture (platform, python, CPU/GPU model, CUDA
  runtime, CUDA driver via `nvidia-smi`, dependency versions via
  `importlib.metadata`).
- `benchmarks/report/render.py` (NEW): cross-experiment assembler
  that reads the four per-experiment Markdown files +
  `run_manifest.json` and emits `report.md` with one executive
  summary, a run-metadata block, four per-experiment sections in
  B5 -> B6 -> B7 -> B8 order, and a methodology footer.
- `benchmarks/run.py`: writes the manifest of intent BEFORE any
  experiment runs (so a crashed run leaves it on disk); rewrites
  it with `completed_at_utc` populated and assembles `report.md`
  after every kind succeeds. The CLI dispatch was refactored into
  `_dispatch_kinds(...)` so a single try/except wraps every
  driver's `ValueError` and exits 1 cleanly (absorbs the B7
  Gemini-deferral + B8 commit-msg note).
- `benchmarks/config.py`: new `_at_most_one_spec_per_kind`
  model_validator rejects duplicate `ExperimentSpec.kind` entries
  at config-load time (absorbs B8 arch-I2).
- `benchmarks/hpo/_base.py`: dual-dict `HPO_REGISTRY` +
  `_HPO_SAMPLERS` collapsed into a single `dict[ModelFamily,
  HPORegistration]` (absorbs B8 arch-I5).
- `docs/explanation/benchmarks/index.md` (NEW): methodology index
  the assembled report's footer links into.

**B8 deferrals absorbed in B9**: arch-I2 (duplicate-kinds
validator), arch-I5 (HPO registry collapse), cross-driver
ValueError -> exit 1 (B7 Gemini deferral).

**B9-followup deferrals** (out of scope):
- Comparator-version registry: B9 captures harness-process
  package versions via `importlib.metadata`. A future GBM / TSC
  adapter family registering pins extends `_PINNED_PACKAGES` in
  `run_manifest.py`; the registry is open by design.
- ResultRow run_id reconciliation: B9 ensures the CLI builds a
  single RunEnvironment per invocation so every cell shares the
  manifest's `run_id`. Older manifests (B5-B8 builds) carry one
  `run_id` per experiment kind; B9 onwards is uniform.

### Phase B11: Ensemble-lift experiment (B6.2.5 deliverable)

**Goal**: a separate experiment from B6's pairwise complementarity
that quantifies how much adding a `seq_sklearn` model to the
baseline GBM ensemble lowers loss across the dataset roster.
B6 answers "which two models complement each other"; B11 answers
"does adding the deep model to the GBM ensemble lift it, and is
the lift significant across datasets?"

**Modules**:

- `benchmarks/experiments/ensemble_lift.py` (NEW): report-only
  driver. Reads the B5 manifest + per-cell predictions shards,
  partitions OK cells by `ModelSpec.family` into a baseline
  family (default `"gbm"`) and the seq family (default
  `"seq_sklearn"`), builds two equal-weight averaged ensembles
  per (dataset, seed, fold) pair (`gbm` only vs `gbm + seq`),
  computes per-cell primary loss (`log_loss` for classification,
  `RMSE` for regression), and produces `PerDatasetLift` rows with
  seed-mean `delta_loss = loss(gbm) - loss(gbm + seq)`. A
  per-sample best oracle bound is computed alongside. The
  per-dataset means feed `scipy.stats.wilcoxon` with Holm
  correction (`family_size=1` at v1 since only one seq+baseline
  pair is shipped).
- `benchmarks/report/ensemble_lift.py` (NEW): renderer for the
  `EnsembleLiftExperimentResult` -> `ensemble_lift.md`. Sorted
  per-dataset table (`task_type ASC, delta_loss_mean DESC`),
  incomplete-datasets footnote (no GBM cells / no seq cells / no
  paired cells), Wilcoxon block with raw + Holm-adjusted p-values.
  Deliberately filesystem-free; the CLI calls the driver then
  pipes the structured result straight in. (No `render_from_dir`
  convenience because building the result requires the structured
  `BenchmarkConfig`.)
- `benchmarks/config.py`: `ExperimentKind` Literal extended with
  `"ensemble_lift"`.
- `benchmarks/run.py`: dispatch arm for `ensemble_lift`; writes
  `ensemble_lift.md` via the B7-style `atomic_write_bytes` shard.
- `benchmarks/experiments/__init__.py`: exports
  `EnsembleLiftExperimentResult` + `run_ensemble_lift`.
- `tests/benchmarks/test_ensemble_lift_experiment.py` (NEW): the
  e2e tests. Includes a `_ConstantSeqClassifierAdapter` test
  fixture (a fresh `family="seq_sklearn"` constant-classifier
  adapter; the existing `NaNProbaAdapter` fake returns NaN proba
  which would block `log_loss`). Covers: refusal when no
  `ensemble_lift` spec, refusal on empty manifest,
  `no_gbm_predictions` sentinel, `no_seq_predictions` sentinel,
  the happy path (paired cells produce Δloss + oracle row +
  Wilcoxon), Markdown rendering, and empty-result rendering.

**Dependencies**: B5 (per-cell predictions shards), B8 (Holm
correction reused), B10 (the GBM family providing the baseline
half of the pairing).

**Deliverable tests**: 7 tests; key assertion is
`oracle_loss_mean <= min(loss_gbm_only_mean, loss_gbm_plus_seq_mean)`
by construction.

**Done when**: `python -m benchmarks.run --config <cfg>
--experiment ensemble_lift` emits a per-dataset Δloss table +
Wilcoxon significance call against the B5 manifest, and the
result type round-trips through the renderer.

**B11-followup deferrals** (out of scope for v1):
- Stacked / meta-learner ensembles: v1 ships equal-weight
  averaging only. A logistic stacking head over the same per-cell
  predictions is a follow-up driver and does not change the B5
  shard schema.
- Cross-family-set lift: the v1 pairing is fixed to one baseline
  family + one seq family. A future driver could sweep
  `family_size > 1` by accepting a `tuple[Family, ...]` baseline
  pool; Holm correction is already shape-correct.
- Additional baseline families beyond GBM (e.g. TSC, MLP): the
  partition is keyed by `ModelSpec.family`; the driver accepts a
  `baseline_family` kwarg and is open by design. The v1 default
  is `"gbm"`.
- Per-fold confidence intervals: v1 ships seed-fold means + the
  Wilcoxon across datasets. Bootstrap CIs on Δloss within a
  dataset are a B11-followup; the per-cell loss vectors are
  already materialized for them.

### Phase B12: Classical-TSC adapter family (B2-followup)

**Goal**: round out the comparator roster with the classical
time-series-classification family from aeon (ROCKET, MultiRocket,
Catch22). Mirrors B10's GBM family pattern but with two unique
constraints: aeon is NOT installed by default (sklearn pin
conflict), and aeon's classifiers consume `raw-mts` (3D tensor)
rather than `tabular-window`.

**Modules** (12 wire-up touch points across 5 modules):

- `benchmarks/adapters/_base.py`: `OptionalDependencyMissingError(Exception)`
  (typed exception; subclasses Exception directly, NOT
  ImportError, so existing `except ImportError` clauses cannot
  swallow it).
- `benchmarks/protocol/raw_mts.py` (NEW): `panel_to_tensor`,
  `instance_labels`, `broadcast_per_instance_to_per_row`,
  `RawMTSError`. The reshape returns `np.float32` (Gemini-I2:
  halves host RAM at scale), drops categorical channels per
  D-B12.6 (Gemini-C3: ordinal encoding violates ROCKET's
  metric-space assumption), and emits NaN for below-floor rows
  so the existing `_strip_below_floor_rows` machinery surfaces
  the drop count uniformly across families.
- `benchmarks/adapters/tsc.py` (NEW): `_TSCAdapter` base +
  three concrete (`_RocketClassifierAdapter`,
  `_MultiRocketClassifierAdapter`, `_Catch22ClassifierAdapter`).
  `_check_aeon_available` lazy-imports aeon via
  `importlib.util.find_spec`; raises the typed exception when
  aeon is absent. fit/predict/predict_proba route through the
  raw-mts reshape; the per-instance prediction is broadcast back
  to per-row via the panel_row_to_instance mapping.
- `benchmarks/hpo/tsc.py` (NEW): three per-classifier samplers
  (ROCKET: 1 dim, MultiRocket: 2 dims, Catch22: 2 dims). No
  per-family trial-count constants; budget comes from B8's
  profile-tier envelope. `_TSC_SEARCH_SPACE_SIZE=2` with the
  per-classifier breakdown documented in the B6.4.0 parity
  disclosure footnote.
- `benchmarks/experiments/raw_loss.py`: B5 wire-up touch points
  (b)-(f). New `_SKIP_REASON_OPTIONAL_DEP_MISSING` constant,
  catch clause appended BEFORE the generic adapter-error tuple,
  counter init + classifier branch + summary field on
  `RawLossExperimentResult`.
- `benchmarks/experiments/hpo_uplift.py`: B8 wire-up touch
  points (g)-(j). The trial-level fit catch keeps
  `OptionalDependencyMissingError` OUT so it propagates to the
  cell-level catch (which routes to the typed skip without
  burning the budget on a missing-dep error that would never
  resolve). Counter + summary field added to
  `HPOUpliftExperimentResult`.
- `benchmarks/run_manifest.py`: touch point (k). `"aeon"`
  appended to `_PINNED_PACKAGES` so the fingerprint records the
  installed version (None when absent, structurally covered by
  the existing `_safe_pkg_version` PackageNotFoundError catch).
- `tests/benchmarks/test_raw_mts.py` (NEW, 18 tests): shape,
  boundary cases at L_resolved (exact/below/above), determinism,
  Hypothesis happy-path + inverse short-entity property,
  categorical-drop convention, all-categorical rejection,
  broadcast NaN-fill, entity-homogeneity oracle.
- `tests/benchmarks/test_tsc_adapter.py` (NEW, 7 tests):
  aeon-missing path via `monkeypatch.setitem(sys.modules,
  "aeon", None)` so pytest auto-restores under
  `pytest-randomly`. End-to-end e2e through `run_raw_loss` with
  the typed skip count matching n_tsc_cells_in_run.
- `tests/benchmarks/test_tsc_adapter_smoke.py` (NEW, 4 tests
  behind `pytest.importorskip("aeon")`): aeon-installed smoke
  with `num_kernels=100` override to keep wall-clock under 5s,
  ROCKET/MultiRocket/Catch22 fit+predict+proba, the
  entity-homogeneity oracle.
- `tests/benchmarks/test_hpo_tsc.py` (NEW, 7 tests):
  per-classifier suggest_params shape, family lookup,
  wrong-family rejection.
- `tests/benchmarks/test_hpo_uplift_experiment.py` extension:
  end-to-end B8 wire-up via a fake TSC adapter that raises the
  typed error at fit time (reuses the `rocket_classifier`
  registration name so the HPO sampler dispatches).

**Dependencies**: B2 model registry, B3 protocol layer (lookback
+ split), B4 metrics, B5 raw-loss driver (for the skip wire-up),
B8 HPO uplift driver (for the skip wire-up).

**Deliverable tests**: 538 total benchmark tests after B12 (was
505 before); 33 new tests pin the contract.

**Done when**: a CI run with aeon ABSENT emits typed
`optional_dep_missing: aeon` skip rows for every TSC cell with
the run completing cleanly; with aeon INSTALLED, the three
classifiers fit + predict + emit valid per-row probabilities
broadcast from per-instance predictions.

**B12-followup deferrals** (out of scope for v1):
- D-B12.1: heavyweight TSC ensembles (HIVE-COTE 2.0,
  MultiRocket-Hydra). Registry-extensible later.
- D-B12.2: unequal-length panel support via aeon's per-estimator
  wrappers.
- D-B12.3: TSC regression (aeon `BaseRegressor`); v1 is
  classification-only.
- D-B12.4: GPU acceleration. aeon 0.11.x is CPU-only.
- D-B12.5: promote `aeon` into a `benchmarks-tsc` extra. Hinges
  on aeon loosening its sklearn pin.
- D-B12.6: categorical channels. v1 drops them entirely from the
  raw-mts reshape; a follow-up wires aeon's per-categorical
  wrappers or a fixed one-hot path.

### Phase B13: Entity-block bootstrap CIs (B5.4)

**Goal**: implement the B5.4 entity-block bootstrap CI primitive
that the metrics module references but does not deliver, and
that the B11 + B12 Gemini final-passes both flagged as a
load-bearing correctness gap. Minimal v1 scope: bootstrap
primitive + RollupRow + B5 leaderboard renderer integration.

**Review trail**: 6 design-review rounds (R1: 6C/13I/7N
deduped -> R2: 1C/2I/3N -> R3 APPROVE) + Gemini design
final-pass (3 CRITICALs all addressed: loader determinism,
OOM ceiling, stale-rollup freshness) + R4 (2 NEW CRITICALs
from R4 fixes: undeclared `RawRollupError` + undeclared
`RunManifest.fingerprint()`) + R5 (2 NEW CRITICALs: CLI catch
contradiction + internal inconsistency in B13.0) + R6 APPROVE.

**Modules** (B13.0 foundations + the v1 surface):

- `benchmarks/adapters/_base.py`: unchanged.
- `benchmarks/run_manifest.py`: extend `RunManifest` with
  `fingerprint() -> str` (SHA-256 of canonical-JSON serialized
  reproducibility fields, omitting `completed_at_utc` which
  rewrites at run-end).
- `benchmarks/config.py`: extend `ExperimentSpec` with
  `bootstrap_rollup_enabled: bool = True` +
  `bootstrap_n_resamples: int | None = None`. Per-experiment
  opt-out + per-experiment override.
- `benchmarks/metrics/bootstrap.py` (NEW): the entity-block
  bootstrap primitive. Pure function:
  `entity_block_bootstrap_ci(losses, entity_ids, *, n_resamples,
  confidence, seed, metric_fn) -> (mean, ci_lo, ci_hi)`. Uses
  `np.random.Generator(np.random.PCG64(seed))` EXPLICITLY (not
  `default_rng`); `np.percentile(..., method="linear")`
  EXPLICITLY. Input arrays set `flags.writeable = False` at
  entry so a mutating `metric_fn` raises immediately.
- `benchmarks/bootstrap_manifest.py` (NEW): `RollupRow` pydantic
  model + `write_rollup` / `load_rollup` / `rollup_path`
  helpers. Atomic-rename parquet write.
- `benchmarks/report/bootstrap_rollup.py` (NEW): the aggregator.
  `RawRollupError(RuntimeError)` + `aggregate_bootstrap_rollup
  (config, *, output_root, env, manifest)`. Reads B5 manifest +
  per-cell predictions shards, re-resolves entity_id by
  re-loading the panel + defensive sort by `(entity_col,
  time_col)` + row-count drift check, builds per-row losses
  (classification via `metrics.pairwise.classification_nll`,
  regression as squared error), calls the primitive with
  per-task `metric_fn` (`np.nanmean` for classification,
  `sqrt(np.nanmean(.))` for regression so sqrt applies PER
  RESAMPLE — closes Gemini-C1 Jensen gap), emits `RollupRow`
  per (dataset, model, task_type). Row-count ceiling
  `5e10` gates OOM on huge datasets (D-B13.7 names the
  sufficient-statistics optimization for the followup).
- `benchmarks/report/raw_loss.py`: adds
  `render_leaderboard_markdown_with_ci(manifest, rollup, *,
  expected_manifest_fingerprint, aggregator_error_class)`
  alongside the existing `render_leaderboard_markdown`. The
  renderer dispatches by rollup presence; the CI variant
  joins on `(dataset, model, task_type)` and renders
  `mean [ci_lo, ci_hi]` with a `*` suffix when
  `n_cells_evaluated < n_seeds * n_folds` (partial-fold flag,
  Gemini-I3). Three footnote sources: per-cell skip (existing),
  rollup-level skip (`bootstrap_skipped_reason`), and
  aggregator-failed (CLI-wrapper case). Freshness check via
  `manifest_fingerprint` on every RollupRow.
- `benchmarks/run.py`: new `_run_bootstrap_rollup(config, *,
  env, output_root)` wrapper between raw_loss and the
  leaderboard render. Calls the aggregator; catches
  `RawRollupError`; deletes any partial output; returns
  silently so the run continues with exit code 0. Gated by
  `is_rollup_enabled(config)`.

**Dependencies**: B5 (manifest + predictions shards), B9 (run
manifest + fingerprint).

**Deliverable tests** (46 across 6 test files):

- `tests/benchmarks/test_bootstrap.py` (15 tests): primitive
  contracts (mean matches ground truth, entity-vs-row CI width
  ratio with zero within-entity variance, PCG64 determinism,
  cross-process canary, `method="linear"` pin via spy,
  writeable=False defense, partial-NaN does not propagate,
  custom metric_fn dispatch, per-resample sqrt oracle).
- `tests/benchmarks/test_bootstrap_manifest.py` (6 tests):
  RollupRow round-trip through parquet (all Gemini-added
  fields enumerated), atomic-replace on overwrite, empty
  shard, extra=forbid.
- `tests/benchmarks/test_bootstrap_rollup.py` (11 tests): e2e
  classification + regression paths, all-skipped sentinel,
  numpy_version capture, manifest_fingerprint capture,
  smoke-profile n_resamples (5000), per-experiment override,
  loader row-count drift raises, row-count ceiling raises,
  empty manifest raises, shard written to disk.
- `tests/benchmarks/test_raw_loss_report.py` (8 tests): CI
  variant rendering, fallback to std on no-rollup, std-column
  drop when CI present, freshness mismatch fallback,
  aggregator-failed footnote, partial-fold `*` flag,
  rollup-skipped separate footnote, std signature preserved.
- `tests/benchmarks/test_loader_conformance.py` (2 tests):
  every-registered-loader determinism check + a deliberately
  non-deterministic loader detected (negative-path proves the
  conformance check is not vacuously passing).
- `tests/benchmarks/test_run_bootstrap_rollup_wrapper.py` (4
  tests): happy-path writes shard, opt-out via
  `bootstrap_rollup_enabled=False`, RawRollupError caught +
  partial output deleted, missing run_manifest skip.

**Done when**: `python -m benchmarks.run --config <cfg>
--experiment=raw_loss` emits `bootstrap_rollup.parquet` next
to the manifest + the assembled `leaderboard.md` ships the CI
column. With `bootstrap_rollup_enabled=False`, the rollup is
skipped and the leaderboard ships the legacy `mean ± std`
shape.

**B13-followup deferrals** (out of scope for v1):

- D-B13.1: B6 (ensemble pairwise) CI integration.
- D-B13.2: B7 (training-time) CI.
- D-B13.3: B8 (HPO-uplift) Δ-statistic paired CI.
- D-B13.4: B11 (ensemble-lift) per-dataset Δ CI.
- D-B13.5: BCa CI (v1 ships percentile only).
- D-B13.6: per-fold CIs (v1 aggregates across folds + seeds
  for a single CI per (dataset, model)).
- D-B13.7: per-entity sufficient-statistics optimization for
  the OOM ceiling on full-tier datasets. Both v1 metric_fns
  are expressible from (sum, count) sufficient statistics.

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
