# Design: seq-sklearn comparative benchmark suite

## Requirements

The grading rubric for this design. Every finding must trace to one of
these, or to a fundamental correctness concern.

- **RQ1** Design a benchmark suite spanning many problem types: balanced,
  extremely imbalanced, all-numeric, mixed numeric+categorical, and the
  size tiers small / medium / large / huge.
- **RQ2** Choose well-known public datasets the library can actually be
  applied to. Datasets must genuinely suit a sequence model (panel /
  per-entity sequence with a supervised target), not forecasting and not
  single-snapshot tabular data.
- **RQ3** Build a harness that benchmarks every model the library ships
  and compares them against external ML models and methods.
- **RQ4** The external comparator set is GBMs plus classical
  multivariate-time-series-classification methods, and the registry must
  be extensible to any sklearn-API estimator (sklearn, cuML, and future
  classifiers/regressors) with no harness change.
- **RQ5** Report raw loss comparisons first, as a standalone deliverable.
- **RQ6** Then measure correlation of predictions and of errors to
  produce evidence of whether the library's models complement standard
  ML models in an ensemble.
- **RQ7** Benchmark training time.
- **RQ8** Benchmark the improvement obtained from full hyperparameter
  tuning versus default configuration.
- **RQ9** Dataset access policy: auto-download open datasets; allow
  gated datasets behind a documented one-time manual setup.
- **RQ10** The harness lives in-repo as a `benchmarks/` package, reusing
  library internals, and targets a single-GPU workstation envelope.

## Scope

This document specifies a comparative benchmark suite that measures
seq-sklearn's models against external classifiers and regressors on a
fixed roster of public datasets. It covers the dataset roster and the
inclusion rubric, the in-repo harness architecture, the fair-comparison
protocol, the metric set, the four experiment types (raw loss
comparison, error-correlation and ensemble complementarity, training
time, hyperparameter-tuning uplift), the single-GPU compute tiering,
the statistical methodology, and the reproducibility contract.

Numbered contracts (B1, B2, ...) are authoritative. Where this design
references library components (`TabularToSequence`,
`EntityTimeSeriesSplit`, `compute_three_way_split`, the Optuna
integration, the calibration modules), the behavior of those components
is fixed by `docs/requirements.md` and `docs/architecture.md`; this
document specifies how the benchmark consumes them, not how they work.

### Non-goals

- **B0.1** This suite is not the internal performance-regression gate.
  `tests/perf/` (architecture A13 + requirements N7) own
  step-time / memory / latency regression with checked-in per-cell
  baselines, run nightly, gated on absolute regressions. The benchmark
  suite here is comparative science: it answers "how good, versus what,
  and does it complement standard models", produces a report, and does
  not gate CI.
- **B0.2** Not a leaderboard submission target. No claim of
  state-of-the-art; the goal is honest relative measurement and
  ensemble-complementarity evidence.
- **B0.3** Not shipped in the wheel. The `benchmarks/` package is
  excluded from the distribution and from the library coverage-delta
  rule, but it still obeys repo rules: pydantic configs, type hints,
  `ruff format`, `pyright`, no `print`.

## B1: Dataset inclusion rubric

A dataset is admissible only if a sequence model can actually model it.
The rubric is a hard gate applied before any dataset enters the roster.

- **B1.1** The data must have a panel shape: a set of entities, each
  observed over an ordered index (time or step), with one or more
  features per observation. One sequence per entity.
- **B1.2** The supervised target must be a per-entity label or value
  that is independent of the trivial continuation of the input series.
  A target that is "the next value of an input channel" is forecasting
  and is excluded (the library's thesis is that forecasting is a
  different task).
- **B1.3** The sequence must carry signal. A dataset where the per-entity
  observation count is 1 (a single snapshot per entity) is excluded: a
  sequence model has nothing to model and the comparison is degenerate.
  Recorded as instructive negatives in B1.5.
- **B1.4** Label timing must be definable without leakage: there must be
  a well-defined observation cutoff at or before which features are read
  and after which the label is determined.
- **B1.5** Documented exclusions (kept in the roster file as negatives so
  the rubric is auditable):
  - Pure forecasting sets (electricity, traffic, ETT, Weather, M4):
    fail B1.2.
  - Telco Customer Churn (IBM), UCI Adult, ULB Credit-Card-Fraud:
    one row per entity, no per-entity sequence; fail B1.3.

## B2: Dataset roster

Every cell of the coverage matrix the suite must span is represented:
balance (balanced vs extreme imbalance), modality (all-numeric vs mixed
numeric+categorical), size (small / medium / large / huge), and task
(binary classification, multiclass classification, regression). The
driving payments-panel use case is directly represented by Amex.

Access tiers: **OPEN** auto-downloads with a content-hash pin; **GATED**
requires a documented one-time manual step (license click-through,
credentialed access, or Kaggle competition-rules acceptance) and then
caches identically.

| Dataset | Task | Balance | Modality | Size (entities) | Source / access | Panelization |
|---|---|---|---|---|---|---|
| BasicMotions | 4-class | balanced | numeric | ~80 (small) | UEA, OPEN (aeon) | native MTS |
| JapaneseVowels | 9-class | ~balanced | numeric | ~640 (small) | UEA, OPEN | native MTS, variable T |
| UWaveGestureLibrary | 8-class | balanced | numeric | ~4.5k (medium) | UEA, OPEN | native MTS |
| UCI-HAR | 6-class | ~balanced | numeric | ~10.3k windows (medium) | UCI, OPEN | windowed accelerometer |
| LSST | 14-class | imbalanced | numeric | ~4.9k (medium) | UEA, OPEN | native MTS (astronomical) |
| PhysioNet-2012 | binary (mortality) | imbalanced (~14%) | mixed | ~12k ICU stays (medium) | PhysioNet, GATED | irregular MTS, cutoff at 48h |
| C-MAPSS FD001 | regression (RUL) | n/a | numeric | 100 engines / ~20k cycles (medium) | NASA, OPEN | per-cycle sensor panel |
| PenDigits | 10-class | balanced | numeric | ~11k (large) | UEA, OPEN | native MTS, T=8 |
| FaceDetection | binary | balanced | numeric | ~9.4k (large) | UEA, OPEN | MEG, 144 channels |
| PTB-XL | binary (abnormal) | imbalanced | mixed | ~21.8k records (large) | PhysioNet, GATED | 12-lead ECG + metadata |
| C-MAPSS FD004 | regression (RUL) | n/a | numeric | ~25k cycles (large) | NASA, OPEN | multi-condition/fault |
| InsectWingbeat | 10-class | imbalanced | numeric | ~50k (huge) | UEA, OPEN | native MTS, 200 dims |
| Amex Default | binary (default) | imbalanced (~25%) | mixed | ~458k customers (huge) | Kaggle, GATED | ~13 monthly statements/customer |
| KKBox Churn | binary (churn) | imbalanced | mixed | huge | Kaggle (WSDM'18), GATED | transactions + daily user logs |
| IEEE-CIS Fraud | binary (fraud) | extreme (~3.5%) | mixed | huge | Kaggle, GATED | transaction sequence per card/uid |
| MIMIC-IV IHM | binary (mortality) | imbalanced | mixed | huge | PhysioNet credentialed, GATED | first-48h ICU stay |

- **B2.1** Roster is data, not code: `benchmarks/datasets/registry.py`
  holds one `DatasetSpec` per row plus the B1.5 negatives. A registry
  invariant test (`test_registry_invariants`, offline, no download)
  asserts every entry with `excluded=True` carries a non-empty
  `exclusion_reason`, every active entry resolves to a loader, and every
  active irregular-time entry (PhysioNet-2012, PTB-XL, MIMIC-IV) carries
  a non-null `densification_policy` with non-empty `bin_width`,
  `aggregation`, and `missing_bin_fill`. Adding a dataset is a registry
  entry plus a loader, no harness change.
- **B2.2** Each `DatasetSpec` declares: access tier, content hash,
  entity column, time/step column, feature columns with declared
  dtype (numeric vs categorical), target column, task type, the
  observation cutoff rule (B1.4), a default lookback, a
  densification policy, and (binary tasks only) a `positive_label`
  field naming the class that the imbalanced/event-of-interest
  framing treats as positive. `positive_label` is the value passed
  to sklearn's `pos_label=` for binary `precision_score` /
  `recall_score` / `f1_score` (B5.1); it is required on binary
  specs and absent on multiclass and regression specs (a pydantic
  cross-field validator pins this). The densification policy is required because the
  library (`docs/requirements.md` F2) treats consecutive panel rows as
  consecutive periods regardless of elapsed wall-clock time; for the
  irregular-time datasets in the roster (PhysioNet-2012, PTB-XL,
  MIMIC-IV) the loader, not the library, owns resampling to a fixed
  period grid, and the per-dataset rule (bin width, aggregation,
  missing-bin fill) is declared in the spec so the deep and baseline
  paths consume the identical densified panel. The content
  hash is the SHA-256 of the raw downloaded archive bytes. On a
  mismatch the cache layer raises a typed `DatasetIntegrityError`
  (never silently uses the file, never auto-retries), tested offline by
  writing a byte-flipped file and asserting the raise.
- **B2.3** GATED datasets ship a loader that, on missing local data,
  raises a typed `GatedDatasetUnavailableError` naming the exact manual
  step and the target cache path. No silent skips: a profile that
  includes a GATED dataset fails loudly if it is absent unless the
  dataset is explicitly deselected. This is verified fully offline by
  pointing the loader at a nonexistent `tmp_path` and asserting the
  typed raise and message contents; dataset downloads are never
  exercised in CI.
- **B2.4** Multi-label framings (PTB-XL full label set, MIMIC
  phenotyping) are reduced to binary for v1 (the library ships
  multi-label in v1.1 per the roadmap). The richer framings are
  recorded in the spec for later activation; see Deferred D1.

## B3: Harness architecture

In-repo, pydantic-typed, reuses library internals so the deep model is
evaluated exactly as a user would invoke it.

```
benchmarks/
  __init__.py
  config.py            BenchmarkConfig, ProfileSpec, RunSpec (pydantic v2)
  cli.py               python -m benchmarks run --profile ...
  datasets/
    registry.py        DATASET_REGISTRY, exclusion negatives
    base.py            DatasetSpec, PanelDataset, AccessTier
    cache.py           download, hash-pin, XDG cache, gated-access guard
    loaders/           uea.py, uci_har.py, cmapss.py, physionet.py, kaggle_panels.py
  models/
    registry.py        MODEL_REGISTRY
    adapters.py        ModelAdapter protocol + concrete adapters
    baselines.py       GBM / classical-MTSC / naive specs
  featurize.py         WindowedAggregateFeaturizer (fair tabular view)
  splits.py            wrappers over library splitters
  metrics.py           classification / regression / resource metrics
  ensemble.py          correlation, diversity, stacking, oracle, CD diagram
  hpo.py               Optuna runners (seq + baseline) with budget tiers
  runner.py            orchestrates dataset x model x seed x variant, resumable
  report.py            aggregate results -> tables + plots + markdown
  results/             schema'd parquet output (gitignored)
```

- **B3.1** `tests/benchmarks/` mirrors this tree (testing-conventions
  rule). Harness logic (featurizer leakage, split disjointness, metric
  correctness, registry integrity, adapter contract) is unit-tested on
  tiny synthetic panels from the library's `SyntheticPanelGenerator`.
  Dataset downloads are never exercised in CI.

### B3.2 Model registry and adapter contract

The user requirement is extensibility: any object following the sklearn
estimator API must be benchmarkable without harness changes, including
sklearn, cuML, aeon classifiers, and the library's own estimators.

- **B3.2.1** `ModelAdapter` is a Protocol with `fit(panel, y, *, seed)`,
  `predict(panel)`, `predict_proba(panel)`, `supports_proba: bool`,
  `name`, `family` (`seq-sklearn` | `gbm` | `classical-mts` | `linear` |
  `naive` | `external`), and `consumes` (`tensor-panel` |
  `tabular-window` | `raw-mts`). The adapter, not the harness, owns the
  input transformation. When the wrapped model has no `predict_proba`,
  `supports_proba` is `False` and `predict_proba` raises
  `ProbaUnsupportedError` naming the model; the harness reads
  `supports_proba` and excludes that model from log-loss / Brier / ECE
  and from probability-based ensembling rather than fabricating
  probabilities. Regressors set `supports_proba=False` by definition.
  Tested by asserting the raise on a `predict_proba`-less estimator.
- **B3.2.2** Concrete adapters:
  - `SeqSklearnAdapter`: wraps a library estimator; consumes
    `tensor-panel` via the library's own `TabularToSequence`. No
    benchmark-side preprocessing for the deep path.
  - `SklearnEstimatorAdapter`: any `fit`/`predict`[`_proba`] object
    (sklearn, cuML, XGBoost/LightGBM/CatBoost sklearn wrappers,
    RandomForest CPU or `cuml.ensemble.RandomForestClassifier`);
    consumes `tabular-window`.
  - `AeonClassifierAdapter`: ROCKET / MiniRocket / MultiRocket,
    Catch22+GBM, DTW-1NN; consumes `raw-mts`.
  - Naive: `DummyClassifier` / `DummyRegressor`, last-timestep linear.
- **B3.2.3** Registry entries are `ModelSpec(name, adapter_factory,
  default_params, hpo_space | None, optional_dependency | None)`.
  Missing optional dependency (e.g. cuML) skips that model with a
  non-empty `skipped_reason` written to the result row, never a crash;
  a registry invariant test asserts a skipped model always carries a
  non-empty reason (monkeypatch `import` to raise `ImportError` for the
  named dependency).

## B4: Fair-comparison protocol

The scientific core. Three model classes consume the same data three
ways; the comparison is fair only if the split, the label, the cutoff,
and the lookback are identical for all of them.

- **B4.1 (evaluation split).** The canonical cross-model split is the
  library's `EntityTimeSeriesSplit` (architecture A9.1), not
  `compute_three_way_split`. `EntityTimeSeriesSplit` is a per-entity
  time-expanding-window splitter: every entity appears in every fold,
  split along its own time axis, and for fold `i` the test segment is
  extended left by `lookback - 1` rows of train history per entity. The
  benchmark therefore does **not** claim entity-disjoint folds; it
  reuses exactly the per-entity time-expanding folds the library
  produces, so the deep model is evaluated as a user would invoke it.
  The same fold's `(train_idx, test_idx)` arrays (panel-row indices) are
  handed to every adapter for that `(dataset, seed, fold)` cell. The
  guaranteed invariant, restated for this layer, is the one A9.1
  provides: no test-segment **target** window draws its supervision
  label or its in-window features from a row that is also a train
  **target** row; the `lookback - 1` overlap is history-only context
  for the test windows.
- **B4.1a (three-way split is internal).** `compute_three_way_split`
  (architecture A5) is a deterministic, no-seed train/val/cal partition
  computed by the library Trainer *after* `TabularToSequence.transform`,
  on transformed windows, not raw panel rows. It is consumed only inside
  `SeqSklearnAdapter` via the library; it is never the cross-model
  partition and is not handed to baseline adapters. The benchmark `seed`
  governs only (i) huge-dataset subsampling (B7.3) and (ii) the
  bootstrap resampling of B5.4; the B4.1 time-ordered evaluation split
  is seed-invariant by construction, so the split-fingerprint (B8.1) is
  a function of `(dataset, lookback, fold)`, not of `seed`. To remove
  the F2 multi-entity random-split leakage confound
  (`docs/requirements.md` F2: random val splits leak future info on
  panels with more than one entity), `SeqSklearnAdapter` pins the
  library's internal `val_split_strategy="time_ordered"` for every
  benchmark panel; the deep path's internal val split is therefore also
  seed-invariant, not a random split.
- **B4.1b (single lookback source).** The lookback is resolved once per
  run from the `DatasetSpec` default (B2.2) into a single value
  `L_resolved`. `EntityTimeSeriesSplit(lookback=L_resolved)`, the deep
  model's `tabular_config__lookback`, the B4.3 featurizer window, and
  the B4.4 raw-mts window are all bound to this one value. A run asserts
  these four are equal before any fit (B4.5).
- **B4.2** `tensor-panel` consumers use the library's
  `TabularToSequence` with scalers/encoders fit on the train fold only
  (the library already enforces this; the benchmark must not refit on
  full data).
- **B4.3** `tabular-window` consumers receive a deterministic,
  leakage-safe feature vector per entity from
  `WindowedAggregateFeaturizer`, aggregating exactly the same per-entity
  window the deep model sees: the most recent `L_resolved` observations
  up to the B1.4 cutoff for that fold (the identical
  `EntityTimeSeriesSplit` left-extended history per entity, B4.1b), no
  more and no fewer. Per numeric channel {last, mean, std, min, max,
  OLS slope vs step index, first, last-minus-first, observed-count};
  per categorical channel {last, mode, n-unique}. Aggregate statistics
  and category encodings are fit on the train fold only. This gives
  GBMs a fair, conventional view of the same window rather than a
  crippled one.
- **B4.4** `raw-mts` consumers receive the same windowed
  `(entity, T=L_resolved, F)` panel the deep model sees. The library's
  F3 `padding_mask` convention is `True = padding (ignore)`; aeon
  estimators have no mask channel. The reconciliation is fixed:
  variable-length panels are converted to aeon's native unequal-length
  representation where the chosen aeon estimator supports it
  (MiniRocket, MultiRocket, DTW); for estimators that require equal
  length, sequences are pre-padded to `L_resolved` and the harness uses
  the aeon estimator's documented unequal-length wrapper rather than
  feeding pad values as real observations. The chosen mode per
  `ModelSpec` is recorded on the result row so a weakened classical
  baseline is visible, not silent (R3).
- **B4.5 (leakage and window-equality tests).** Four named, offline,
  falsifiable unit tests on `SyntheticPanelGenerator` panels:
  1. `test_featurizer_train_invariance`: fit
     `WindowedAggregateFeaturizer` on train fold A, transform a fixed
     test fold, record output O_A; re-fit on train fold B (same test-set
     entities, mutated *train-only* values in **both** a numeric and a
     categorical channel, including the train-fold category
     distribution), transform the same test fold, output O_B. Assert
     `O_A == O_B` exactly. The oracle is the transform output on the
     test fold, not the fitted internal state. The categorical-channel
     mutation specifically guards against a `fit_transform`-on-full-panel
     leak of test-fold category frequencies.
  2. `test_featurizer_not_vacuous`: assert transforming a train sample
     under the fold-A vs fold-B fits *does* differ, so test 1 cannot
     pass vacuously, and the symmetric check uses at least one test-fold
     entity absent from the train fold.
  3. `test_seq_sklearn_adapter_val_split_strategy_is_time_ordered`:
     monkeypatch the library Trainer constructor and assert
     `SeqSklearnAdapter` passes `val_split_strategy="time_ordered"` on a
     synthetic multi-entity panel (B4.1a), so the F2 confound cannot
     silently return.
  4. `test_window_binding`: assert
     `EntityTimeSeriesSplit.lookback == tabular_config__lookback ==
     featurizer.window == raw_mts.window == L_resolved` for every
     adapter in a run (B4.1b), and the B4.1 invariant stated at its own
     granularity: no test-window **target** row is also a train
     **target** row. The `lookback - 1` history/context rows that A9.1
     (`docs/architecture.md` A9.1) permits to overlap between folds are
     explicitly excluded from this assertion; they are context for test
     windows, not supervision, and a correct `EntityTimeSeriesSplit`
     produces that overlap by design.

## B5: Metrics

Loss-first, as required: the primary metric is a strictly proper
scoring rule, reported and ranked before anything else.

Every metric pins one formula or one delegated call so a known-value
unit test (`tests/benchmarks/test_metrics.py`) has a non-arbitrary
oracle. `metrics.py` is a thin wrapper over the pinned calls; it does
not reimplement them.

- **B5.1 (classification).** Primary: `sklearn.metrics.log_loss`
  (natural log, normalized mean, `labels=` passed explicitly so absent
  classes do not shift the value). Loss-first ranking: the leaderboard
  in B6.1 is ordered by this and only this; every metric below is
  reported in the secondary table and does NOT re-rank.
  Secondary metrics, grouped:
  - **Threshold-dependent.** `sklearn.metrics.accuracy_score`,
    `precision_score`, `recall_score`, `f1_score`. The decision is
    taken via the adapter's `predict()` (the library default: argmax
    over `predict_proba` for binary and multiclass; equivalent to a
    fixed 0.5 threshold for binary), NOT via a tuned threshold; the
    point of these metrics in B5 is the practitioner-familiar
    operating-point reading, not the optimal one. These four metrics
    are computed for every adapter regardless of `supports_proba`
    since they consume `predict()` only; the B3.2.1 exclusion is
    scoped to the probability-based metrics it names.
    For multiclass each of `precision`, `recall`, `f1` is reported
    under BOTH `macro` and `weighted` averages with the suffix in
    the metric name (`precision_macro_zd0`, `precision_weighted_zd0`,
    ...) so a reader cannot mistake one for the other; the convention
    is embedded in the metric name itself (no sidecar). Binary uses
    `average="binary"` and reports a single value per metric with
    only the `_zd0` suffix (`precision_zd0`, no averaging suffix);
    `pos_label` is pulled from the per-dataset
    `DatasetSpec.positive_label` field (B2.2) so the imbalanced
    binary datasets in the roster (mortality, default, churn, fraud,
    abnormal) score the minority event as positive instead of
    silently defaulting to `1`.
    The `_zd0` suffix encodes `zero_division=0` in the metric name
    on every form (binary and multiclass) for the same reason ECE
    uses `ece_q15`: a single read of the metric name names the
    convention.
  - **Rank-based.** `roc_auc_score` (`multi_class="ovr"`, macro);
    `average_precision_score` for PR-AUC (sklearn step-function
    definition, not trapezoid); balanced accuracy; MCC.
  - **Strictly-proper secondary.** Brier score, pinned as
    `sklearn.metrics.brier_score_loss` for binary and, for multiclass,
    the unweighted mean over classes of the one-vs-rest per-class
    `brier_score_loss`, i.e.
    `mean_k brier_score_loss(y == k, p[:, k])` (no extra division by K
    beyond that mean; normalized by N inside each call); a known-value
    test fixes a 3-class 4-sample matrix.
  - **Calibration.** Expected calibration error with the
    binning pinned to 15 equal-**mass** bins (quantile bins), the
    variant recorded in the metric name (`ece_q15`). ECE is reported
    for the deep model on two arms that hold the training fold fixed:
    the library is always configured to draw the calibration fold (so
    train size is identical), and the only toggle is whether the
    fitted calibrator is applied. This avoids the F2 fold-collapse
    confound (architecture A5: with `calibration_strategy='none'` and
    no threshold tuning the cal fold folds back into train, which
    would otherwise make the "uncalibrated" arm train on more data).
- **B5.2 (regression).** Primary: RMSE (`root_mean_squared_error`);
  the B6.1 leaderboard ranks by this and only this. Secondary:
  MAE; R² (`r2_score`); MAPE
  (`sklearn.metrics.mean_absolute_percentage_error`) with a typed
  pre-check in `benchmarks/metrics/regression.py` that emits `nan`
  and records a skip reason when MAPE is not well-defined on the
  vector. Three named skip reasons cover the input pathologies:
  `mape_undefined_zero_in_y_true` (any `y_true == 0`; sklearn
  divides by `max(|y_true|, epsilon)` on every row so a single zero
  silently distorts the GLOBAL number, not just that row, which is
  why an any-zero skip is the principled response rather than a
  threshold), `mape_undefined_nan_in_y_true` (any `np.isnan(y_true)`),
  and `mape_undefined_inf_in_y_true` (any `np.isinf(y_true)`).
  Then pinball loss at the library's configured quantile levels
  using the convention `mean(max(q·(y−ŷ), (q−1)·(y−ŷ)))` (lower is
  better), the same convention `sklearn.metrics.mean_pinball_loss`
  uses.
- **B5.3** Resource metrics on every run: wall-clock fit seconds,
  peak process RSS and peak CUDA memory, and per-sample inference
  latency (median and p95) measured separately from throughput-batched
  prediction.
- **B5.4** All metrics computed on held-out test folds only, aggregated
  across seeds as mean and a percentile bootstrap 95% interval
  (10 000 resamples, fixed bootstrap seed recorded on the row).

## B6: Experiment types

### B6.1 Raw loss comparison (run first)

For each dataset, every applicable model trained with default config,
fixed seeds, library determinism mode on for the deep models. Output:
a per-dataset leaderboard ranked by the B5 primary loss, plus the
secondary metric table. This is the standalone deliverable the rest
builds on; it ships before any ensemble analysis.

### B6.2 Error correlation and ensemble complementarity

The question the user wants answered: do the library's sequence models
complement standard ML models in an ensemble?

- **B6.2.1** Persist out-of-fold predictions per (dataset, model, seed):
  class probabilities for classification, point (and quantile)
  predictions for regression.
- **B6.2.2** Prediction agreement: pairwise Pearson and Spearman across
  models on the out-of-fold prediction vectors.
- **B6.2.3** Error correlation: pairwise Pearson correlation of
  per-sample errors. Classification error is the per-sample negative
  log-likelihood `-log p(y_true)`; the signed residual `(y - p_true)` is
  reported alongside. Regression error is the signed residual `(y - ŷ)`.
  This is the direct evidence of decorrelated errors.
- **B6.2.4** Diversity statistics for each (seq-sklearn model, baseline)
  pair, each pinned to one formula on hard 0/1 predictions (probabilities
  thresholded at the dataset's operating point) with `N11/N10/N01/N00`
  the joint correct/incorrect counts:
  - Yule's Q: `(N11·N00 − N01·N10) / (N11·N00 + N01·N10)`.
  - Disagreement: `(N01 + N10) / N`.
  - Double-fault: `N00 / N`.
  - Correlation: the φ coefficient
    `(N11·N00 − N01·N10) / sqrt((N11+N10)(N01+N00)(N11+N01)(N10+N00))`.
  Degenerate convention pinned: when the denominator is zero (perfect
  agreement, `N01 = N10 = 0`), Q and φ are defined as `1.0` (the limit
  of both as the two classifiers' predictions coincide); when all four
  marginals collapse otherwise the statistic is `nan` and excluded from
  aggregation. Each has a known-value test in
  `tests/benchmarks/test_ensemble.py` over all-agree (asserts Q == φ ==
  1.0), all-disagree, and half-agree tables, plus
  `test_phi_degenerate_non_perfect_agreement_is_nan` (a zero φ
  denominator from a collapsed marginal that is not perfect agreement
  returns `nan` and is dropped from aggregation).
- **B6.2.5** Complementarity test: build a GBM-only ensemble (average
  and a properly nested stacked meta-learner) and a GBM+seq-sklearn
  ensemble on the same folds. Report ΔlogLoss / ΔRMSE of adding the
  deep model, an oracle (per-sample best) upper bound, and whether the
  improvement is significant per B7. The headline result is the pairing
  of "seq model error is decorrelated from GBM error" with "adding it
  lowers ensemble loss".

### B6.3 Training-time benchmark

`fit` wall-clock per (dataset, model, profile) at fixed config, reported
against dataset size so the scaling curve is visible. Deep models report
single-GPU time; baselines report CPU (and cuML GPU where the optional
dependency is present, recorded separately).

### B6.4 Hyperparameter-tuning uplift

Quantifies the gain from full tuning, using the library's first-class
Optuna integration (with in-training pruning) for the deep models and an
Optuna search over the published hyperparameter space for each baseline.

- **B6.4.0 (parity policy).** Tuning uplift is only comparable when the
  search effort is declared. Each `ModelSpec.hpo_space` records the
  number of effective hyperparameters, the scale (log vs linear) per
  dimension, and the budget tier; these are written on every tuned
  result row. The deep search space is the library's STABLE-field set
  (advanced/extras are v1 no-ops per architecture A16); baseline spaces
  are the published spaces. The report presents Δ(default→tuned)
  per family with the declared space size next to it, so cross-family
  uplift is read with its search-breadth caveat rather than as a like
  comparison. v1 does not attempt to normalize search-space size across
  families; that asymmetry is stated in the report.
- **B6.4.1** Two variants per (model, dataset): `default` config and
  `tuned` under a fixed budget tier (B7.4). Report Δ(primary loss)
  default→tuned and time-to-best-trial.
- **B6.4.2** Tuning uses only the train/validation folds; the test fold
  is untouched until the final fit, so the uplift number is honest.

## B7: Compute tiering and methodology (single GPU)

- **B7.1** Three profiles selected by config:
  - `smoke`: small datasets only, 1 seed, no HPO. Minutes; can run in a
    nightly job alongside CI but is not a gate.
  - `standard`: small + medium + selected large, 3 seeds, light HPO
    (25 trials). Hours-to-overnight on one GPU.
  - `full`: adds huge datasets, 5 seeds, 100-trial HPO. Huge datasets
    are entity-subsampled to a configured cap under `full` (B7.3); a
    full-population pass on huge datasets is an explicit opt-in flag,
    not the default, given the single-GPU envelope. The Amex payments
    panel (the library's driving use case) is the one huge dataset that
    the standard report always includes at full population in at least
    one reported configuration, so the flagship result is never
    subsample-only.
- **B7.2 (resumability).** The completion ledger is a per-cell sentinel:
  before computing cell `(dataset, model, seed, variant)` the runner
  checks for a `results/_done/{cell_key}.json` marker; if present the
  cell is skipped. Each cell writes its result row to a per-cell
  parquet shard via write-to-temp-then-`os.replace` (atomic on POSIX),
  and only after the shard is durably renamed is the sentinel written.
  A crash between shard write and sentinel write re-runs that one cell
  (idempotent); a crash mid-shard-write leaves no shard and no sentinel.
  `report.py` concatenates shards. Tested by
  `test_runner_resume_skips_completed_cells` (inject an exception after
  cell 1's sentinel; rerun; assert the cell-1 adapter is invoked exactly
  once across both runs).
- **B7.3** Huge-dataset entity cap is a profile field with a documented
  default. Subsampling is seeded and stratified: by class label for
  binary/multiclass (imbalance ratio preserved within tolerance), by
  10-quantile bins of the target for regression. The stratification key
  per task is recorded on the row.
- **B7.4** HPO budget is a tier: a trial count and a wall-clock cap,
  whichever binds first. Enforcement is the Optuna `timeout=` study
  argument plus `n_trials=`; for the deep models the library's
  in-training pruning also applies. Both tier fields are written on
  every tuned row so uplift is compared only within a tier.
- **B7.5 (statistics).** Across datasets, the input matrix is one cell
  per (model, dataset) equal to the seed-mean of the B5 primary loss.
  `scipy.stats.friedmanchisquare` over that matrix tests the global
  rank-difference null; a Nemenyi critical-difference diagram visualizes
  it. The ensemble-lift claim (B6.2.5) uses a Wilcoxon signed-rank test
  paired over datasets on the per-dataset Δloss of (GBM+seq) vs
  (GBM-only), with Holm correction over the set of seq-model/baseline
  ensemble pairs tested (not over the dataset cross-product). All test
  statistics, p-values, and the correction family size are written to
  the report.

## B8: Reproducibility

- **B8.1** Every result row records: library git SHA, benchmark git SHA,
  dataset content hash, resolved dependency versions, seed, profile,
  HPO tier (B7.4), hardware tier (reusing the library's `HardwareTier`),
  and the split fingerprint. The split fingerprint is defined as the
  SHA-256 of the canonical-JSON list of `(fold_index,
  sorted(train_idx), sorted(test_idx))` produced by
  `EntityTimeSeriesSplit` for that `(dataset, L_resolved)`. Because the
  evaluation split is seed-invariant (B4.1a), the fingerprint is stable
  across seeds for a dataset and changes if and only if the fold indices
  change. `test_result_row_split_fingerprint_stability` asserts both
  directions: equality across two different seeds on the same
  `(dataset, L_resolved)`, and inequality when `L_resolved` is perturbed
  by `+1` (the named direction-2 perturbation, which changes the
  left-extension and therefore the fold index arrays).
- **B8.2** Deep-model runs use the library determinism mode; baselines
  are seeded explicitly. Non-deterministic ops that cannot be pinned
  are flagged on the row rather than hidden.
- **B8.3** A run manifest (one JSON) captures the full
  `BenchmarkConfig`, the resolved roster, and the environment, so a run
  is reconstructable from the manifest alone. The manifest must
  round-trip: `BenchmarkConfig.model_validate(json.loads(manifest))`
  reproduces the config object (`test_run_manifest_round_trips`).

## Risks

- **R1** Single-GPU envelope makes huge-dataset full-HPO impractical;
  mitigated by B7.1/B7.3 subsampling, with the un-subsampled pass an
  explicit opt-in.
- **R2** GATED datasets can change or disappear upstream; mitigated by
  B8.1 content-hash pinning and a typed missing-data error (B2.3).
- **R3** Fairness disputes on the baseline featurizer (B4.3): a too-weak
  tabular view would inflate the deep model. Mitigated by making the
  featurizer an explicit, reviewed, unit-tested contract and reporting
  the GBM-on-windowed-aggregates result transparently.
- **R4** Kaggle programmatic download requires credentials and rules
  acceptance; the loader documents the manual path and never embeds
  credentials.

## Deferred

- **D1** Multi-label dataset framings (PTB-XL full, MIMIC phenotyping):
  deferred to v1.1 when the library ships multi-label; specs carry the
  richer framing now, inactive.
- **D2** Distributed / multi-GPU sweep support: out of scope for the
  single-GPU envelope; revisit if the compute decision changes.
- **D3** Heavyweight TSC ensembles (HIVE-COTE 2.0, MultiRocket-Hydra):
  not in the initial baseline set; the registry admits them later with
  no harness change (B3.2.3).
- **D4** Wiring this doc into the Sphinx nav: it is an internal design
  doc (peer of `architecture.md` / `requirements.md`), left out of the
  user toctree until there is user-facing benchmark content.

## Tracking (review loop)

Addressed and deferred items are maintained here so successive swarm
runs see prior decisions and do not re-raise resolved points.

Round 1 (architecture-reviewer, qa-test-coverage, style-reviewer):

- Addressed (CRITICAL):
  - arch-C1 / arch-C2: B4.1 rewritten to the real `EntityTimeSeriesSplit`
    A9.1 contract (per-entity time-expanding folds, not entity-disjoint);
    B4.1a separates `compute_three_way_split` as library-internal and
    pins the seed's role; B8.1 redefines the split fingerprint
    accordingly.
  - arch-C3: B4.1b adds a single `L_resolved` lookback source bound to
    split, deep model, featurizer (B4.3) and raw-mts (B4.4), asserted in
    B4.5.
  - qa-C1: B4.5 specifies three named, offline, non-vacuous leakage /
    window-binding tests with explicit oracles.
  - qa-C2: B2.3 pins the offline `tmp_path` verification of the typed
    gated-data raise.
  - qa-C3: B7.2 pins the atomic shard-then-sentinel resumability
    mechanism and its test.
  - qa-C4: B8.1 defines the split fingerprint as SHA-256 of canonical
    fold indices with a stability test.
  - qa-C5: B5.1 / B5.2 pin every metric to a delegated call or formula
    (ECE = `ece_q15`, pinball convention, PR-AUC step definition).
  - qa-C6: B6.2.4 pins Q / disagreement / double-fault / φ formulas with
    known-value tests.
- Addressed (IMPROVEMENT):
  - arch-I1: B6.4.0 adds the HPO parity / search-breadth disclosure
    policy.
  - arch-I2: B5.1 holds the train fold fixed across the calibration
    on/off arms, citing the F2 fold-collapse rule.
  - arch-I3: B4.4 pins the F3 `padding_mask` vs aeon reconciliation.
  - qa-I1: B7.5 pins the Friedman input matrix and Holm correction
    scope.
  - qa-I2: B7.3 pins regression stratification (10-quantile bins).
  - qa-I3: B3.2.1 pins `supports_proba` + `ProbaUnsupportedError`.
  - qa-I4: B2.2 pins SHA-256-of-archive and `DatasetIntegrityError`.
  - qa-I5: B7.4 pins Optuna `timeout=`/`n_trials=` cap enforcement.
  - arch-N1: B7.1 guarantees the Amex flagship a full-population pass.
  - qa-N1 / qa-N2 / qa-N3: B2.1 / B3.2.3 / B8.3 add the registry-reason,
    skip-reason, and manifest-round-trip invariants.
Round 2 (style-reviewer APPROVE; architecture 0C/2I/1N; qa 1C/2I/1N):

- Addressed (CRITICAL):
  - qa-NEW-C1: B5.1 pins multiclass Brier to the unweighted mean over
    classes of one-vs-rest `brier_score_loss` with a known-value test.
- Addressed (IMPROVEMENT):
  - arch-IIa: B4.5 test 3 reworded to the target-vs-target invariant,
    explicitly excluding the A9.1 `lookback - 1` history/context overlap.
  - arch-IIb: B4.1a pins `SeqSklearnAdapter` to
    `val_split_strategy="time_ordered"` to remove the F2 multi-entity
    random-split leakage confound.
  - qa-NEW-I1: B6.2.4 pins the degenerate Q/φ convention (perfect
    agreement => 1.0; otherwise nan and excluded).
  - qa-NEW-I2: B4.5 test 1 mutates both a numeric and a categorical
    train channel to cover the categorical leakage path.
- Addressed (NITPICK):
  - arch-NIT: B2.2 adds a per-dataset densification policy field for the
    irregular-time roster datasets (F2 caller obligation).
  - qa-NEW-N1: B8.1 names the direction-2 perturbation (`L_resolved + 1`).

Round 3 (architecture APPROVE 0/0/0; style APPROVE 0/0/0; qa APPROVE,
2 IMPROVEMENT + 1 NITPICK, all named-test additions to already-approved
contracts, folded in):

- Addressed (IMPROVEMENT):
  - qa-III-1: B4.5 adds named test 3
    `test_seq_sklearn_adapter_val_split_strategy_is_time_ordered`.
  - qa-III-2: B2.1 registry invariant extended to assert irregular-time
    datasets carry a fully specified `densification_policy`.
- Addressed (NITPICK):
  - qa-III-N1: B6.2.4 adds
    `test_phi_degenerate_non_perfect_agreement_is_nan`.

Consensus: round 3. architecture-reviewer and style-reviewer APPROVE
with zero findings; qa-test-coverage zero CRITICAL with every
IMPROVEMENT now resolved in-doc. Gemini final pass not yet run (user
deferred Gemini for capacity).

- Deferred: D1, D2, D3, D4 below, each with a one-line reason. v1
  explicitly does not normalize HPO search-space size across model
  families (stated in B6.4.0); the asymmetry is reported, not removed.

Round 4 (B5 delta, user-requested metric extension; addressed
through a single delta-review round):

The user-facing metric set in B5.1 / B5.2 is expanded with the
practitioner-familiar threshold-dependent classification metrics
and a regression-side error-rate metric. The loss-first ranking
principle is unchanged: every added metric is reported in the
secondary table; the primary leaderboard order does NOT change.

- B5.1 adds `accuracy_score`, `precision_score`, `recall_score`,
  `f1_score`. Binary uses `average="binary"` with
  `pos_label=spec.positive_label` (new B2.2 field); multiclass
  reports BOTH `macro` and `weighted` under suffixed metric names
  with the convention embedded as `*_macro_zd0` / `*_weighted_zd0`
  (the `_zd0` suffix names the `zero_division=0` convention the
  same way `ece_q15` names ECE's binning). The decision is the
  adapter's `predict()` (argmax of `predict_proba`, the library
  default); no threshold tuning at this stage. These four metrics
  are computed for every adapter regardless of `supports_proba`
  since they consume `predict()` only. Loss-first ranking
  unchanged.
- B5.2 adds MAPE (`mean_absolute_percentage_error`) with three
  named typed skip reasons that cover every pathological input:
  `mape_undefined_zero_in_y_true` (any zero; sklearn's epsilon
  applies to every row globally so an any-zero skip is the
  principled response rather than a row-level mask),
  `mape_undefined_nan_in_y_true`, and
  `mape_undefined_inf_in_y_true`. Each emits `nan` for the metric
  value plus the reason string on the row.
- B2.2 adds `positive_label: int | str` on binary specs (absent on
  multiclass and regression; pydantic cross-field validator pins
  this) so the imbalanced binary datasets in the roster (mortality,
  default, churn, fraud, abnormal) score the minority event as
  positive instead of silently defaulting to sklearn's `1`.
- The Phase B4 owning test
  `tests/benchmarks/test_metrics_known_values.py` carries three
  inline-arithmetic fixtures (no derivation from a prior sklearn
  run): a 2-class 6-sample binary fixture with unequal class
  counts (4/2) that exercises `average="binary"` with the spec's
  `positive_label`; a 3-class 4-sample multiclass fixture with
  class 2 fully ABSENT from `y_true` (only classes 0 and 1 appear)
  so the `zero_division=0` branch fires non-trivially in
  `precision_macro_zd0` / `precision_weighted_zd0`, AND with
  unequal counts (2/1/1) so `macro` and `weighted` produce
  different values; an 8-sample regression vector with at least one
  negative `y_true` so MAPE's denominator is real on every row.
- Two new owning tests pin the record shape and the macro/weighted
  distinguishability:
  `tests/benchmarks/test_metrics_records.py` asserts the
  multiclass record carries `*_macro_zd0` + `*_weighted_zd0` and
  does NOT carry bare names; the regression record carries `mape`
  plus a `mape_skip_reason: str | None` field; the binary record
  carries `*_zd0` (no suffix).
  `tests/benchmarks/test_macro_vs_weighted_distinguishable.py`
  asserts `f1_macro_zd0 != f1_weighted_zd0` on the unequal-counts
  fixture, proving the two averages travel separate code paths.
- `tests/benchmarks/test_mape_pathologies.py` parametrizes over
  the three pathological inputs (zero / nan / inf in `y_true`)
  with the typed skip-reason string asserted on each.

Delta-review R1 resolutions (architecture 0C/4I/2N; qa 4C/3I/1N;
style APPROVE 0/0/0): all CRITICAL + IMPROVEMENT folded in above.
The four qa CRITICALs and the four arch IMPROVEMENTs overlapped on
the same root causes (binary-fixture absence, zero_division-branch
unexercised, MAPE pathologies undocumented, MetricsRecord field
shape unpinned, `pos_label` unspecified, `supports_proba` scope
ambiguous, MAPE skip rationale missing, `zero_division` convention
ambiguous in the manifest), all resolved by:
1. Adding `positive_label` to `DatasetSpec` (B2.2).
2. Specifying `predict()`-only computation regardless of
   `supports_proba` (B5.1).
3. Adding the binary 6-sample fixture + the class-absent
   3-class fixture + the unequal-count `macro != weighted`
   invariant + the inline regression arithmetic + the
   `*_zd0` suffix.
4. Naming the three MAPE skip reasons and the sklearn-epsilon
   rationale.
5. Replacing impl-plan B5-B8 phase heading em-dashes with commas
   (style-reviewer's out-of-scope CRITICAL observation; fixed
   pre-emptively).
Style APPROVE held on the B5 prose itself. Nitpicks (impl-plan
goals line listing single names without suffix, manifest
metric-name convention) folded in.

Consensus on the B5 delta: CRITICAL: 0, IMPROVEMENT: 0
outstanding, NITPICK: 0. The delta is ready for the Gemini
final-pass alongside the rest of the design doc (which was
already consensus through R3); Gemini was deferred per user
direction.
