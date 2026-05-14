# Requirements: seq-sklearn v1

## Context

The sklearn ecosystem covers tabular classification and regression broadly:
linear models, trees, ensembles, kernel methods, basic neural nets for
tabular data. The forecasting world has decent deep-learning coverage:
pytorch-forecasting, darts, neuralforecast, sktime-forecasting.

The gap sits at the intersection: **modern deep sequence models used for
standard supervised tasks (classification, regression), wrapped in the
sklearn estimator contract, on tabular panel data**. Today this means
one of three painful paths:

1. Hand-roll a custom PyTorch model with a sklearn-adapter wrapper. Every
   ML team that needs this writes their own version. Few do it well.
2. Use a forecasting library off-label. Impedance mismatch on task
   formulation (forecasting predicts the next N values of the input
   series; supervised learning predicts an independent target).
3. Use classical methods (DTW, KNN-DTW, ROCKET, Shapelets via pyts /
   tslearn / aeon). Good baselines, but they cannot capture the
   representations modern deep models learn.

`seq-sklearn` fills this gap. It provides sklearn-compatible classifiers
and regressors backed by modern deep sequence models, on tabular panel
input, sharing a common preprocessing pipeline, training infrastructure,
calibration story, and Optuna integration across every model family.

The driving real-world use case is customer-churn prediction in a
payments-processing ensemble. The library is built generally enough to
serve any tabular-time-series supervised workload where the panel-shape
input fits.

## Scope

The library covers three families of deep sequence models. Every model
implements the sklearn estimator contract for both classification and
regression on tabular panel data. The same `TabularToSequence`
preprocessing component feeds every model. The same training pipeline,
calibration strategies, and Optuna search infrastructure apply across
all of them.

### Recurrent family (planned for v3)

- **LSTM**: long short-term memory networks, optionally bidirectional
- **GRU**: gated recurrent unit, lighter alternative to LSTM
- **LSTM-FCN**: hybrid combining an LSTM branch (long-range
  dependencies) with a fully-convolutional 1D branch (local patterns),
  concatenated before the classification or regression head

### Transformer family (v1 first, v2 expansions)

- **TFT**: Temporal Fusion Transformer (Lim et al., 2021). **v1**, the
  first model the library ships. Adapted from native quantile regression
  to classification and standard regression. The detailed v1 spec lives
  in the "v1 concrete: TFT" section below.
- **PatchTST**: patch-based time-series transformer (Nie et al., 2023).
  Treats sub-sequences as patches; strong on long-horizon problems. v2.
- **iTransformer**: inverted attention transformer (Liu et al., 2024).
  Attends across variables instead of time; competitive on heterogeneous
  feature sets. v2.
- **TimesNet**: period-aware decomposition with 2D convolutions (Wu et
  al., 2023). v2.
- **TST**: vanilla time-series transformer (Zerveas et al., 2021). A
  baseline transformer with positional encoding and standard attention. v2.

### Foundation / pretrained family (v4)

- **Chronos**: T5-based pretrained foundation model (Ansari et al.,
  2024, AWS). v1 forecasting; library wraps it with adapter heads for
  classification and regression.
- **MOMENT**: pretrained time-series foundation model (Goswami et al.,
  2024).
- **TimesFM**: Google's time-series foundation model (Das et al., 2024).

All three foundation models share a load-pretrained-then-attach-head
pattern. The library handles the head architecture and fine-tuning
strategy.

## Non-goals

1. **Forecasting.** Multi-horizon prediction of future values of an
   input series. Existing libraries (pytorch-forecasting, darts,
   neuralforecast, sktime-forecasting) cover this well. `seq-sklearn`
   predicts independent supervised targets from sequence inputs, which
   is structurally different even when the target is "next month's
   value of something" (that's a forward-shifted regression target, not
   a multi-step forecast).
2. **Unsupervised methods.** No autoencoders, no contrastive learning,
   no clustering, no anomaly detection.
3. **Streaming or online updates.** Training and inference are
   batch-mode. Online updates from streaming data are out of scope.
4. **Model serving infrastructure.** The library is a Python package;
   deployment to a serving system is the caller's concern.
5. **Auto-feature engineering.** Feature construction (delta columns,
   rolling aggregates, seasonality encodings) is the caller's
   preprocessing. The library accepts whatever feature columns the
   caller declares.
6. **Classical baselines.** DTW, KNN-DTW, ROCKET, Shapelets, ESN, HIVE
   variants. `pyts`, `tslearn`, and `aeon` cover these. `seq-sklearn`
   focuses on modern deep models.
7. **Single-device training only.** v1 supports a single CPU or single
   GPU. DDP, FSDP, model parallelism, and DeepSpeed are out of scope.
   The model sizes targeted by v1 (typically 1-50M parameters across
   the v1-v4 roadmap) do not require distributed training; if a future
   model class outgrows single-device, that is a v5+ discussion.
8. **Multi-horizon forecasting in the regression head.** Each window
   predicts one target value at one configurable `prediction_step`,
   matching the classifier's contract. Multi-step output is a separate
   axis that fits the forecasting libraries above better than this one.

## Roadmap

| Version | Scope |
|---|---|
| **v1** | TFT classifier and regressor; full `TabularToSequence` pipeline; Optuna; all library-wide infrastructure detailed in this doc |
| v1.1 | Multi-output regression, multi-label classification (architectural constraints already in v1) |
| v2 | PatchTST, iTransformer, TimesNet, TST (transformer family completion) |
| v3 | LSTM, GRU, LSTM-FCN (recurrent family) |
| v4 | Chronos, MOMENT, TimesFM (foundation family with adapter heads) |

Versions are additive. v2 does not require waiting for v3; v3 does not
require waiting for v2.

The library-wide infrastructure (sklearn API, `TabularToSequence`,
training pipeline, Optuna integration, calibration, hardware/precision
policy, error contract, repo hygiene) is built ONCE in v1 and reused by
every subsequent model. Each subsequent model contributes only its
architecture, its specific hyperparameters, and any family-specific
shared patterns.

## Architectural philosophy

Three layers, separated cleanly:

1. **Library-wide infrastructure** (sections F1 through F11, N1 through
   N8 in this doc). Everything every model needs. Built once in v1.
   Stable thereafter.

2. **Family-level abstractions** (section "Per-family architectural
   patterns" below). Each model family carries shared patterns:
   recurrent models share BPTT and hidden-state semantics, transformer
   models share attention-mask handling, foundation models share
   pretrained-checkpoint loading and adapter-head conventions. v1
   defines the abstraction for the transformer family (since TFT lives
   there); v2 and v3 add the other family abstractions when their first
   model ships.

3. **Concrete models**. Each model is a thin shell that plugs into the
   family abstraction with its architecture and hyperparameters. v1's
   only concrete model is TFT (section "v1 concrete: TFT" below).

This is the architectural goal that lets the library ship TFT in v1 and
add nine more models over subsequent versions without rewriting the
infrastructure.

## Versioning and stability

The library uses **semantic versioning** strictly. `MAJOR.MINOR.PATCH`:

- **MAJOR** bump on any breaking change to the public API surface
  (renamed or removed public class, removed public method, narrowed
  argument type, new required argument).
- **MINOR** bump on additive changes (new model class, new method, new
  argument with a default, new acceptable value in a `Literal`).
  Behavior-changing default values bump MINOR with a CHANGELOG entry.
- **PATCH** bump on bug fixes that do not change behavior beyond
  fixing the bug.

The **public API** is exactly:

- What `seq_sklearn/__init__.py` re-exports.
- What is documented in the API reference under `docs/api/`.
- Module attributes reached without a leading underscore in the import
  path (e.g. `seq_sklearn.tuning.suggest_params` is public;
  `seq_sklearn._validate.check_y` is not).

Anything reachable only through an underscore-prefixed module or
attribute is **internal** and not covered by the stability guarantee.

### Per-module stability tiers (v1)

| Tier | Module / Symbol | Notes |
|---|---|---|
| STABLE | `TFTClassifier`, `TFTRegressor`: `fit`, `predict`, `predict_proba`, `predict_quantiles`, `score`, `get_params`, `set_params`, `save`, `load` | sklearn-contract methods; breaking change requires MAJOR |
| STABLE | `TabularToSequence`: fit/transform/inverse_transform | |
| STABLE | `seq_sklearn.hardware.detect`, `HardwareTier` | enum values may be added; existing values stable |
| STABLE | `seq_sklearn.model_selection.EntityTimeSeriesSplit` | |
| BETA | `TFTClassifier.export_onnx`, `TFTRegressor.export_onnx` | dependency on `[onnx]` extra; export shape may evolve |
| BETA | `predict_with_attention`, `AttentionOutput`, `RegressionAttentionOutput` | fields may be added in MINOR releases; consult attribute access, not tuple position |
| ALPHA | `seq_sklearn.tuning.suggest_params` default search space | search-space defaults may change without MINOR bump; pass an explicit search space for stable behavior |
| INTERNAL | `seq_sklearn._*` modules | not part of the public API |

### Deprecation policy

Removal of public functionality requires at least one MINOR release
emitting `DeprecationWarning` before removal. The warning message names
the replacement.

## Data shape

The library accepts a tabular panel: one row per entity per period.
Concrete shape at the calling reference team (payments-processing churn):

- Key: `(company_id, date)` where `date` is a monthly period
- Stationary descriptors: `industry`, `country`, `state`
- Time-varying real: `spend`, `logins`, `transactions`, `support_calls`,
  `support_emails`, plus pre-computed deltas
- Pre-computed deltas: each metric also appears as
  delta-vs-1-month-ago, delta-vs-2-months-ago, delta-vs-3-months-ago

This is the v1 reference panel. The same shape works for any
panel-structured tabular time-series problem: customers x months,
patients x visits, devices x days, sensors x hours. The model does not
know what an entity is; it sees a sequence per entity.

## Functional requirements (library-wide)

These apply to every model the library ships, in every release. New
models inherit the contract; new releases preserve it.

### F1: sklearn-compatible estimator contract

Every model exposes two estimator classes: `<Model>Classifier`
(`ClassifierMixin`) and `<Model>Regressor` (`RegressorMixin`). v1's
classes are `TFTClassifier` and `TFTRegressor`. v2 adds `PatchTSTClassifier`,
`PatchTSTRegressor`, etc.

All classifier and regressor classes expose the same methods:

- `fit(X, y)` where `X` is a pandas DataFrame in the panel shape above
  and `y` is array-like of targets aligned to the rows of `X`.
- `predict(X)` returning class predictions (classifier) or point
  predictions (regressor).
- `score(X, y)` returning accuracy by default for classifiers and R² for
  regressors, with a `scoring` argument for a callable scorer.
- `get_params(deep=True)` and `set_params(**params)` so every estimator
  composes into `sklearn.pipeline.Pipeline` and
  `sklearn.model_selection.cross_val_score`.
- `save(path)` and `load(path)` for PyTorch-native serialization
  (weights + pydantic config dict + a metadata block; see F4).
- `export_onnx(path)` for ONNX export. The dependency on `onnx` and
  `onnxruntime` lives in an optional extra (`pip install seq-sklearn[onnx]`).
- `predict_with_attention(X)` (or `predict_with_states(X)` for recurrent
  models) returning the prediction plus model-introspection outputs as a
  frozen dataclass that supports both attribute access and tuple
  unpacking via `__iter__`.

Classifier-only:
- `predict_proba(X)` returning class probabilities.

Regressor-only:
- `predict_quantiles(X, quantiles=[0.1, 0.5, 0.9])` returning quantile
  estimates per row. Requires the model to have been fit with a quantile
  loss (see F5). Available on every regressor that supports it; raises
  `NotImplementedError` on regressors fit in point mode.

Every classifier and regressor passes the named subset of
`sklearn.utils.estimator_checks.check_estimator` listed in F8 below.

**`y` shape contract (v1).** Every estimator accepts only 1D `y` in v1
(single-output classification or single-output regression). The
validator lives at `seq_sklearn._validate.check_y` and raises:

```
ValueError: seq-sklearn v1 supports single-output targets only.
Multi-output regression and multi-label classification are planned
for v1.1. Got y with shape (n_samples, n_outputs).
```

v1.1 flips this validator in one place; no other code touches `y` shape.

### F2: Input data contract

The caller declares column roles at construction time:

- `id_col`: identifies the entity across rows.
- `time_col`: a sortable column (datetime, period, or integer index).
- `static_categorical_cols`: entity-level categoricals, constant across
  the entity's rows.
- `static_real_cols` (optional): entity-level numerics.
- `time_varying_real_cols`: per-period numerics.
- `time_varying_categorical_cols` (optional): per-period categoricals.

At `fit` time the target column is passed via `y`. The library validates
column existence, that `(id_col, time_col)` is unique, and that the time
column is sortable.

The library does not impose a fixed schema; the caller picks which
columns map to which role.

**Time-axis semantics.**

- **Time zones.** Time columns are accepted as tz-naive or tz-aware
  datetimes, or as integer-indexed periods. Tz-aware datetimes are
  preserved; tz-naive datetimes are treated as the caller's intended
  reference frame (no implicit UTC conversion). Mixing tz-aware and
  tz-naive within one fit raises `DataContractError`.
- **Irregular sampling and gaps.** The library treats consecutive rows
  as consecutive periods regardless of elapsed time between them. A
  customer with a 3-month dormancy followed by 5 active months has
  8 rows; the model sees them as 8 consecutive timesteps. Callers who
  want gap-aware behavior densify their panel before passing (forward
  fill, zero fill, or explicit "dormant" categorical level).
  Documented prominently because it is surprising.
- **Duplicate timestamps.** The `(id_col, time_col)` uniqueness check
  raises `DataContractError` on duplicates. The library does not
  silently dedupe.
- **Validation-split policy.** The default training-time validation
  split is time-ordered per entity: the last `val_fraction` rows of
  each entity's sorted-by-time sequence form the validation set. This
  prevents future leakage. Random splits are available via
  `val_split_strategy="random"` but emit a `UserWarning` when more
  than one entity is present.

### F3: TabularToSequence preprocessing

A `TabularToSequence` transformer converts the validated panel into the
batched tensors every sequence model expects. Shared across all model
families. For each entity it:

1. Sorts the entity's rows by `time_col`.
2. Builds a sliding window of length `lookback` over the sorted rows.
   Default `lookback=12`. Configurable. If an entity has more history
   than `lookback`, only the most recent `lookback` periods are used.
3. Aligns the target to a configurable `prediction_step` relative to
   the end of the window. Default `prediction_step=1` (predict the
   period immediately after the lookback). Any non-negative integer is
   accepted, capped at the available horizon.
4. Emits per-window tensors: `static_categorical` (one set per window),
   `static_real`, `time_varying_real`, `time_varying_categorical`, the
   target, and a per-timestep boolean mask marking positions that are
   real data versus padding.

**Variable history length.** Entities with fewer rows than `lookback`
are NOT skipped by default. They are left-padded to `lookback` with the
mask flagging padded positions. The model respects this mask in its
sequence-handling layers (see F4 and the per-family sections), so the
model only sees valid timesteps. A customer with one month of tenure
and a customer with five years of tenure use the same model, the same
code path, the same lookback length, the same prediction-step setting.
The model just draws on less history when less is available.

Two configuration knobs control the strict edge:

- `min_periods` (default 1): entities with fewer than this many real
  rows are dropped at `fit` time.
- `min_periods_predict` (default 1): same gate at `predict` time;
  entities below the floor get a `NaN` prediction with one aggregated
  log warning per `predict()` call (not one per entity).

**Categorical encoding.** Categorical columns are encoded via a fitted
dictionary. Unseen categories at `predict` time map to a learned
`<unk>` slot. The `<unk>` slot is per-column (one `<unk>` index in each
column's vocabulary), not shared across columns.

**Embedding dimensions.** Default sizing uses the fastai heuristic
`min(50, round(1.6 * cardinality^0.56))`. The config accepts a per-column
override dict `categorical_embed_dims`; any column not in the dict
falls back to the heuristic.

**Categorical cardinality cap.** A configurable `max_categorical_cardinality`
(default 1000) caps the size of each column's vocabulary. Columns
exceeding the cap raise `ConfigError` by default with a message
suggesting the caller hash, bin, or hash-trick the column before
passing. A `hash_high_cardinality: bool = False` config knob applies
the hashing trick automatically when set.

**Continuous feature scaling.** Continuous features are scaled by
default. Strategies: `standard` (zero-mean unit-variance, default),
`robust` (median + IQR; useful for spend outliers), `quantile_uniform`,
`none`. Static-real columns and time-varying-real columns share the
policy by default but can be configured separately.

**Outlier clipping.** Optional clipping after scaling, controlled by
`clip_features: float | None = None` (clamp to ± N standard deviations
after scaling, off by default). Documented as an Optuna search point
when the caller's data has long tails.

### F4: Model abstraction

All concrete sequence models in the library inherit from two base
classes:

- `BaseSequenceClassifier(ClassifierMixin)`: every classifier
- `BaseSequenceRegressor(RegressorMixin)`: every regressor

Both inherit from a shared `BaseSequenceEstimator` carrying the
library-wide infrastructure (fit/predict shell, `TabularToSequence`
plumbing, save/load, validation hooks).

**Common contract every model implements:**

- `_build_module(self) -> nn.Module`: return the trainable backbone.
  Family bases provide partial implementations; concrete models fill in
  architecture-specific blocks.
- `_loss_function(self) -> nn.Module`: return the loss module
  appropriate to the task and config.
- `_head(self) -> nn.Module`: return the task-specific head module.
  Both classification and regression heads take an `n_outputs: int`
  constructor parameter so v1.1 multi-output / multi-label additions
  are small additive changes.

**Layer factory.** All `nn.Linear` and `nn.LayerNorm` instantiations in
`src/seq_sklearn/models/` route through a single factory in
`src/seq_sklearn/models/_layers.py`. v1 returns standard PyTorch
layers; v2 (the FP8-precision pass) can swap in Transformer Engine
equivalents in one place. See N5 for the broader precision story.

**`save` metadata block.** Every saved model includes:

- `seq_sklearn_version`
- `torch_version`
- `cuda_version` (or `None`)
- `python_version`
- `feature_schema_fingerprint` (sha256 of sorted column names + dtypes
  from the fit-time `X`)
- `created_at` (ISO 8601 timestamp)

`load(path)` warns on version mismatch but does not refuse to load.

### F5: Training pipeline

A `Trainer` wraps pytorch-lightning to handle:

- Optimizer and scheduler configuration via pydantic config.
- Early stopping on a validation metric.
- Mixed-precision training when CUDA is present (see N5).
- Deterministic mode driven by a single seed threaded through every
  randomness boundary.
- Checkpoint saving (last + best) and best-model selection.
- Checkpoint resume via `resume_path` constructor argument; restores
  model weights, optimizer state, scheduler state, RNG state.
- DataLoader defaults: `num_workers=min(4, os.cpu_count())`,
  `pin_memory=True` when CUDA, `persistent_workers=True` when
  `num_workers > 0`. All overridable.
- Gradient accumulation via `accumulate_grad_batches: int = 1`.
- Gradient clipping via `gradient_clip_val: float | None = None`.

**Learning-rate schedulers (supported menu).**

| Scheduler | Notes |
|---|---|
| `constant` | No schedule |
| `cosine_with_warmup` | Default; `warmup_steps` configurable |
| `one_cycle` | OneCycleLR-style |
| `reduce_on_plateau` | Step on validation metric stall |

**Class-imbalance strategies (classifier).**

| Strategy | What it does |
|---|---|
| `class_weighted_ce` | Cross-entropy with per-class weights from training-set frequencies. Default. |
| `focal_loss` | Focal loss with configurable `gamma` (default 2.0). |
| `oversample_minority` | Entity-window sampler. Configurable ratio. |
| `undersample_majority` | Entity-window sampler. Configurable ratio. |
| `none` | No reweighting. |

Threshold tuning (post-hoc adjustment of the decision boundary on a
held-out set) is independent of the loss strategy and combinable with
any of the above.

**Loss-function selection by task type.** A single dispatch in the loss
factory, keyed on `task_type`:

| `task_type` | Loss function |
|---|---|
| `binary` | `BCEWithLogitsLoss` (single-output sigmoid) |
| `multiclass` | `CrossEntropyLoss` (softmax) |
| `multilabel` (v1.1) | `BCEWithLogitsLoss` (multi-output sigmoid) |
| `regression_point` | Selectable (`mse`, `mae`, `huber`); default `mse` |
| `regression_quantile` | Pinball loss over the configured quantile list |
| `regression_multioutput` (v1.1) | Same as `regression_point` over `(N, n_outputs)` |

**Regression-loss strategies.**

| Strategy | What it does |
|---|---|
| `mse` | Mean squared error. Default for point regression. |
| `mae` | Mean absolute error. Reports the conditional median. |
| `huber` | Huber loss with configurable `delta`. |
| `pinball` | Pinball / quantile loss. Required for `predict_quantiles`. |

All regression loss functions accept output tensors of shape `(N, K)`
where `K = n_outputs * n_quantiles`. v1 is `(N, 1)` for point or
`(N, len(quantiles))` for quantile. v1.1 multi-output passes
`(N, n_outputs)` or `(N, n_outputs, len(quantiles))` with no loss
changes needed.

**Calibration strategies for classification** (post-hoc, fit on a
held-out validation fold during `fit()`):

| Strategy | What it does |
|---|---|
| `temperature` | Temperature scaling (Guo et al., 2017): one scalar dividing logits. Default. |
| `platt` | Platt scaling: logistic regression on logits. Binary only. |
| `isotonic` | Isotonic regression: non-parametric monotonic mapping. |
| `none` | Raw probabilities. |

**Calibration strategies for regression** (only meaningful for quantile
mode):

| Strategy | What it does |
|---|---|
| `conformal` | Split-conformal prediction adjusting quantiles to match empirical coverage. Default. |
| `isotonic_quantile` | Isotonic regression on the CDF. |
| `none` | Raw quantile predictions. |

The Optuna search space (F7) includes both the loss and the calibration
strategy for the relevant task.

### F6: Synthetic data generators

The library ships generators that produce panels of `(id, time,
features, target)` at configurable period grains: day, week, month,
quarter, year. Used by the test suite and as documentation examples.

Each generator exposes:

- `target_kind`: `"binary"`, `"multiclass"`, `"regression_point"`, or
  `"regression_quantile"`. Selects target distribution and target type.
- `num_entities`
- `periods_per_entity`: `int` (fixed length), `(min, max)` tuple
  (sampled per entity), or a callable. The ragged form is required for
  the test suite to exercise short-history handling.
- `num_static_categorical`, `num_static_real`,
  `num_time_varying_real`, `num_time_varying_categorical`
- Classification-only: `class_balance` (binary), `class_distribution`
  (multi-class)
- Regression-only: `target_scale`, `target_noise`
- `noise_level` (feature-side noise)
- `signal_strength` (how strongly the injected pattern drives the
  target; lets tests assert the model actually recovers it)
- `seed`

The test suite includes at least one scenario per period grain per
`target_kind` where `periods_per_entity` ranges from 1 to a moderate
maximum (e.g. `(1, 60)` for monthly), exercising the variable-history
path end-to-end.

### F7: Hyperparameter tuning compatibility (Optuna first-class)

All model and training hyperparameters live in a single pydantic config
class (`<Model>Config`, e.g. `TFTConfig` for v1) and are exposed as
constructor arguments. `get_params` / `set_params` cover the sklearn
search ecosystem (`GridSearchCV`, `RandomizedSearchCV`).

For Optuna specifically the library ships:

- `seq_sklearn.tuning.suggest_params(trial: optuna.Trial, model_class: type, base: ModelConfig | None = None) -> ModelConfig`
  with a per-model default search space covering architecture
  hyperparameters, training hyperparameters, class-imbalance strategy
  (F5), calibration strategy (F5), and optionally `lookback` and
  `prediction_step` (F3) when the caller wants them tuned.
- A runnable example at `docs/examples/optuna_search.py`.
- A pruning hook integrating with Optuna's `MedianPruner` /
  `HyperbandPruner` by reporting the validation metric after each
  epoch.

The default search space is ALPHA stability (may change without MINOR
bump); pass an explicit search space for stable behavior.

### F8: Error contract

The library defines a single exception hierarchy under `seq_sklearn.errors`:

```
SeqSklearnError                  base class
├── ConfigError                  pydantic validation failures, config inconsistencies
├── DataContractError            F2 column-existence, uniqueness, dtype, tz violations
├── TrainingError                NaN loss, divergence, calibration-set failures
└── PredictionError              fit-required, model-not-loaded, shape mismatches at predict
```

Every public method documents which exception classes it can raise.
`pydantic.ValidationError` is wrapped in `ConfigError` so callers do
not need to import pydantic to catch.

**`check_estimator` subset.** The library passes a named subset of
sklearn's estimator checks. The exact list is pinned in
`tests/conftest.py` as a tuple of check IDs. Checks that test array-only
inputs (the library accepts DataFrames) are explicitly excluded via
`_xfail_checks` with documented rationale. The sklearn version range
the contract is verified against is pinned in `pyproject.toml`.

### F9: Numerical contracts

**NaN-in-features.**

- At `fit` time: NaN in any required column raises `DataContractError`
  with the offending column name and row index of the first occurrence.
- At `predict` time: same. Callers handle imputation upstream.
- The mask path (F3) means "this period does not exist". NaN inside a
  feature at a valid timestep means "this value is missing" and is
  the caller's responsibility.

**NaN-in-loss during training.**

- Mid-training NaN loss is detected per step. Three consecutive NaN
  steps abort the run with `TrainingError` and a message suggesting
  `precision="32"` or a lower learning rate.
- Single NaN steps log a warning and skip the step (no gradient
  update), so a transient overflow does not kill the run.
- Optuna integration: a trial with three consecutive NaN steps returns
  `optuna.TrialPruned` rather than completing with a NaN objective.

**NaN-in-output at predict time.**

- `predict_proba` returns NaN-filled probabilities of the correct shape
  for entities below `min_periods_predict`. Never silently zero-filled.
- `predict_quantiles` returns NaN of the correct shape on the same
  condition.

**Mixed-precision overflow handling** (N5 cross-reference).

- `16-mixed` mode uses lightning's `GradScaler`; divergence is logged
  and training continues with skipped steps.
- Three consecutive skipped steps abort with `TrainingError`,
  recommending `bf16-mixed` or `32`.

**Conformal-calibration sanity.**

- The conformal-fold check raises `TrainingError` if the calibration
  set yields non-monotone quantiles (a real failure mode in an
  undertrained quantile regressor).

**Numerical tolerances for tests** (see N1):

- FP32 deterministic CPU path: exact equality via `torch.equal`.
- FP32 cross-device: `atol=1e-5, rtol=1e-5`.
- Mixed-precision paths: `atol=5e-3, rtol=5e-3`.
- ONNX parity: `atol=1e-4, rtol=1e-4`.

### F10: Cross-validation strategy

`seq_sklearn.model_selection.EntityTimeSeriesSplit(n_splits=5, gap=0, max_train_size=None)`
implements expanding-window splits per entity with a configurable gap
between train and validation to prevent target leakage at the seam.

The library does NOT prevent callers from using
`sklearn.model_selection.KFold` directly via `cross_val_score`. Random
splits on panel data leak future information; the library raises a
`UserWarning` at fit time when called from a non-time-aware splitter
and the data has > 1 entity-period.

### F11: Logging strategy

The library uses `logging.getLogger("seq_sklearn")` with child loggers
per submodule (`seq_sklearn.data`, `seq_sklearn.training`, etc.).

- No handlers are configured by default. The calling application
  configures handlers (library best practice).
- Default level is `WARNING`.
- Training progress is at `INFO`.
- Per-batch detail is at `DEBUG`.
- Lightning's progress bar is enabled by default and suppressed when
  `verbose=False` or stderr is not a tty.

Training observability emits structured log records:

- Gradient norm per step (`DEBUG`)
- Variable-selection-weight entropy per epoch (`INFO`, attention models)
- Attention-distribution entropy per epoch (`INFO`, attention models)
- Hidden-state norm per epoch (`INFO`, recurrent models)

Lightning loggers (`MLFlowLogger`, `WandbLogger`, `TensorBoardLogger`)
are pass-through; the library does not depend on any specific tracking
backend. An MLflow + Optuna example ships at
`docs/examples/mlflow_search.py`.

## Per-family architectural patterns

The library factors family-specific shared patterns into family base
classes. v1 defines the transformer-family base (since TFT lives there).
v2 and v3 add the other family bases when their first model ships.

### Transformer family (`TransformerSequenceEstimator`)

Base class for TFT (v1), PatchTST (v2), iTransformer (v2), TimesNet
(v2), TST (v2). Shared patterns:

- Attention-mask handling for variable-length sequences. The base
  implements the mask broadcast utilities; concrete models call them.
- Multi-head attention factory (uses `F.scaled_dot_product_attention`
  to inherit PyTorch's optimized kernels).
- Positional encoding policy: sinusoidal by default, learned optional,
  per-model overridable.
- Layer factory (Linear, LayerNorm) shared with the library-wide F4
  factory.
- Variable-selection networks: optional family component used by TFT;
  abstracted so PatchTST and iTransformer can opt out.

### Recurrent family (`RecurrentSequenceEstimator`, v3)

Base class for LSTM (v3), GRU (v3), LSTM-FCN (v3). Shared patterns:

- Bidirectional control via a single `bidirectional: bool` flag.
- Initial hidden-state strategy: zero (default), learned-parameter
  (alternative), per-entity (alternative for transfer settings).
- Sequence readout: last valid timestep (default), masked mean pool,
  attention readout.
- Recurrent dropout policy: variational (default), standard (Bernoulli).
- BPTT-truncation policy: full (default), truncated with
  configurable backprop length.

### Foundation family (`FoundationSequenceEstimator`, v4)

Base class for Chronos (v4), MOMENT (v4), TimesFM (v4). Shared
patterns:

- Pretrained-checkpoint loading from Hugging Face Hub or a local path.
- Frozen-backbone training (default) vs. full fine-tuning (alternative)
  vs. LoRA adapters (alternative).
- Adapter head architecture (the conversion from the pretrained model's
  forecasting output to the library's classification or regression
  task).
- Tokenizer / discretizer handling (Chronos quantizes inputs; the
  family base handles the round-trip).
- Model-card and license metadata propagation in `save(path)` metadata.

## v1 concrete: TFT

This section is the concrete v1 implementation against the library-wide
infrastructure above. TFT is the first model to ship.

### TFT architecture

`TFTBackbone` extends `TransformerSequenceEstimator` and implements the
TFT architecture from [Lim et al., 2021](https://arxiv.org/abs/1912.09363):

- Variable selection networks (static, past-temporal)
- Gated residual networks (GRN)
- Static covariate encoders
- Sequence-to-sequence attention
- Interpretable multi-head attention with mask broadcast (inherited
  from `TransformerSequenceEstimator`)

**Variable-length sequence handling.** Variable selection networks
zero out padded positions before computing their softmax weights.
Attention layers apply the mask to both keys and queries so attention
scores at padded positions are masked to `-inf` pre-softmax. A
one-month-tenure entity contributes exactly one valid attention key;
a sixty-month-tenure entity contributes sixty. The same code path
handles both.

**Prediction readout.** Two options, configurable:

- `last_valid` (default): representation at the last un-masked timestep
- `mean_pool`: masked mean across valid timesteps

### TFT classification head

`TFTClassifier` instantiates `TFTBackbone` plus a classification head:

- `Linear(d_model, n_outputs)` projection
- Sigmoid activation for `task_type=binary` (`n_outputs=1`)
- Softmax for `task_type=multiclass` (`n_outputs=num_classes`)
- v1.1 multi-label: same projection, sigmoid-per-output, `n_outputs=num_labels`

### TFT regression head

`TFTRegressor` instantiates `TFTBackbone` plus a regression head:

- `Linear(d_model, n_outputs * n_quantiles)` projection where
  `n_outputs=1` in v1 and `n_quantiles` is `1` for point regression or
  `len(quantiles)` for quantile regression.
- v1.1 multi-output: same projection, `n_outputs > 1`.

### TFT hyperparameters

In `TFTConfig` (pydantic):

- `hidden_size` (default 128)
- `attention_heads` (default 4; must divide `hidden_size`)
- `dropout` (default 0.1)
- `variable_selection_dropout` (default 0.1)
- `prediction_readout` (`"last_valid"` | `"mean_pool"`)

Plus training-side fields shared with all models: `learning_rate`,
`weight_decay`, `batch_size`, `max_epochs`, `warmup_steps`,
`gradient_clip_val`, `accumulate_grad_batches`, `precision`,
`task_type`, `loss_strategy`, `calibration_strategy`, etc.

`TFTConfig` extends `BaseModelConfig`. `model_config = ConfigDict(
extra="forbid", frozen=True)`. Cross-field validators enforce:
`prediction_step >= 0`, `lookback >= 1`, `0 <= dropout < 1`,
`attention_heads divides hidden_size`, `quantiles strictly increasing
in (0, 1)`. Unknown fields raise `ConfigError`. Mutability after
construction is disallowed.

## Non-functional requirements (library-wide)

### N1: Testing

Five test categories under `tests/`:

- `tests/unit/`: one component at a time. Every public function gets
  at least a happy path, one edge case, one error path.
- `tests/integration/`: paired components such as
  `TabularToSequence` → `TFTBackbone`, `TFTBackbone` → `Trainer`,
  `TFTClassifier.fit` → `TFTClassifier.save` → load-in-subprocess →
  `predict`.
- `tests/e2e/`: end-to-end against synthetic data, asserting recovery
  of injected signal.
- `tests/deploy/`: package builds, installs from the built wheel in a
  clean venv, imports cleanly, runs a minimal fit/predict round-trip.
- `tests/perf/`: nightly performance regression benchmarks (training
  step time, peak memory, inference latency) via `pytest-benchmark`,
  gated on absolute regressions per-cell baselines.

**Coverage gates.** `pytest --cov=seq_sklearn --cov-branch` at 85%
line coverage and 80% branch coverage on `src/seq_sklearn/`.
Coverage excludes: `if TYPE_CHECKING:` blocks, `@overload` stubs,
`__repr__` methods, abstract method bodies. Listed in
`[tool.coverage.report] exclude_also`.

**Concrete acceptance thresholds** (v1, on synthetic data with
`signal_strength=0.7`, three-seed median):

- Binary classifier: accuracy >= 0.75 on n=2000 windows.
- Multi-class classifier: macro-F1 >= 0.60 on 4-class n=2000 windows.
- Point regressor: R² >= 0.5 on n=2000 windows.
- Quantile regressor: empirical coverage on nominal 80% interval falls
  in [0.75, 0.85] after conformal calibration.
- Calibration ECE: temperature-scaled classifier ECE <= 0.05 on a
  held-out fold of n=2000.

**Test marks taxonomy.**

| Mark | Policy |
|---|---|
| `slow` | Runtime > 2s. Skipped in the dev inner loop and per-PR CI; run nightly. |
| `gpu` | Requires CUDA. Skipped on CPU CI. |
| `integration` | Pairs of components. Run in per-PR CI. |
| `e2e` | Full pipeline. Run in per-PR CI. |
| `deploy` | Wheel-install smoke. Run in per-PR CI. |
| `onnx` | Requires `[onnx]` extra and `onnxruntime`. Run in per-PR CI when extra is installed. |
| `perf` | Performance regression benchmarks. Run nightly. |
| `flaky` | Known non-deterministic. Retried once via `pytest-rerunfailures`. Each `flaky` mark obligates a tracked follow-up issue. |
| `determinism` | Verifies bit-identical output across runs. Run in per-PR CI on CPU. |

**Required tests (highest-impact correctness).**

- **Mask correctness.** For every variable-length-aware layer
  (variable-selection network, masked attention, mean pool), a test
  takes an entity, pads it, asserts the masked-and-padded prediction is
  bit-identical to the unpadded prediction.
- **Determinism.** Two `fit + predict` runs in the same process produce
  byte-identical output via `torch.equal`. A subprocess re-run produces
  the same hash.
- **save/load round-trip.** Fit, save, load in a fresh subprocess,
  predict on the same X, assert `torch.equal` on predictions and
  attribute equality on the public config.
- **ONNX parity.** Export, load in onnxruntime, predict on a fixed
  batch including a masked variable-length entity, assert agreement
  within `atol=1e-4` with the PyTorch path.
- **Unseen-category robustness.** Fit encoder on `{A, B}`, predict on
  a panel with `{A, C}`, assert `C` maps to the `<unk>` index without
  raising.
- **Short-entity warning.** Predict on a one-row entity with
  `min_periods_predict=3`. The row's prediction is NaN; `caplog`
  contains the expected warning.
- **`check_estimator` subset.** Parametrized over a named list of check
  IDs; explicitly skipped checks carry an `_xfail_checks` annotation.
- **Snapshot regression.** One snapshot test per task type. Pin
  synthetic-data seed and config. Store the expected output tensor in
  `tests/_snapshots/`. Assert byte-identical equality on CPU FP32
  deterministic path.
- **GPU/CPU parity.** Nightly. CPU FP32 deterministic vs. CUDA FP32
  agree within `atol=1e-5`.
- **NaN-loss guard.** Inject a NaN-producing forward; assert the
  trainer raises `TrainingError` after the documented three-strike
  threshold and logs the offending batch index.
- **Quickstart in CI.** `tests/e2e/test_quickstart.py` executes the
  README quickstart end-to-end on every PR.
- **Docstring examples.** `pytest --doctest-modules src/seq_sklearn/`
  in the CI workflow.

**Fixture policy.** Function-scoped fixtures for mutation-sensitive
cases; session-scoped fixtures for read-only large panels. A
`tests/conftest.py` declares the shared synthetic-data generators and
documents the policy. `pytest-randomly` is a dev dependency and runs
on CI to surface hidden state-sharing.

**Property-based testing.** Anything that takes a tensor-shape or
panel-shape input gets one hypothesis test asserting a shape, mask, or
dtype invariant. The hypothesis profile is fast in inner-loop tests,
deep in nightly tests.

**No-network policy.** Tests do not require network access. The
deployment-test job has one allowlist for installing from TestPyPI.

### N2: CI and review automation

**GitHub Actions matrix.** `{ubuntu-latest, macos-latest, windows-latest}`
x `{Python 3.11, 3.12, 3.13}`. Per-PR runs Linux on every Python
version; macOS and Windows nightly-only.

**Per-PR workflow (under 5-minute target).**

1. `ruff check .` and `ruff format --check .`
2. `pyright` strict mode
3. `pytest -m "not slow and not perf and not gpu"` with coverage gates
4. `pytest tests/deploy/` (wheel-install smoke)
5. Documentation build (mkdocs or sphinx, decided in architecture phase)

**Nightly workflow.** Full suite including `slow`, `perf`, and the
macOS/Windows matrix. GPU tests run when a self-hosted runner is
available.

**GitHub Copilot review.** Wired up on PRs for an automated first pass
before human review.

**Flaky-test retry.** `pytest-rerunfailures` retries `flaky`-marked
tests once. Each `flaky` mark obligates a tracked issue with a
follow-up.

### N3: Repository hygiene (open-source standard)

- **License.** Apache-2.0. Explicit patent grant. Compatible with most
  enterprise consumers (including payments processing). GPL-compatible.
- **`pyproject.toml`** with build metadata, dependencies, dev extras,
  optional extras (`[onnx]`, `[mlflow]`, `[wandb]`).
- **Dependency pinning.** Lower bounds only for `torch`,
  `pytorch-lightning`, `pydantic`, `pandas`, `numpy`. Upper bounds
  added reactively when an incompatibility is observed, never
  preemptively.
- **Python version policy.** The library supports the three most-recent
  Python releases at each release cut. When a fourth release ships,
  the oldest drops in the next MINOR version with one deprecation
  cycle. Initial v1 supports 3.11, 3.12, 3.13.
- **`CHANGELOG.md`** following Keep a Changelog conventions.
- **`CONTRIBUTING.md`** documenting the review workflow (see
  `/design-review`, `/review`, `/gemini-final-pass`).
- **`SECURITY.md`** with vulnerability disclosure address and supported
  versions for security patches.
- **`CODE_OF_CONDUCT.md`** based on Contributor Covenant 2.1.
- **`.github/ISSUE_TEMPLATE/`** with bug, feature, and question templates.
- **`.github/PULL_REQUEST_TEMPLATE.md`** with the review checklist.
- **Pre-commit config** wired to ruff and pyright.

### N4: Reproducibility

A single `seed` argument on every estimator's `__init__` threads to
every randomness boundary (data split, model init, dataloader shuffle,
cudnn backend, encoder vocabulary tie-breaks). Two runs with the same
seed and same input produce bit-identical predictions on CPU FP32.

CUDA non-determinism caveat: mixed-precision modes can produce minor
numerical drift run-to-run because TensorCore reduction order is not
guaranteed. The `seed` docstring states this and tells callers to set
`precision="32"` plus `torch.use_deterministic_algorithms(True)` when
bit-identical reproducibility on GPU is required.

Audit trail: `save(path)` writes the metadata block described in F4.

### N5: Hardware and precision

**CPU as first-class.** A 1-5M parameter model (every v1 model) fits
on modern CPUs. CPU is a fully supported runtime for development, CI
(GitHub Actions free tier gives no GPU), the synthetic-data unit
tests, and small-scale production. All tests except `gpu`-marked run
on CPU.

**GPU floor.** NVIDIA compute capability 6.0 (Pascal, 2016) and newer.
The library inherits PyTorch's minimum via the pinned `torch>=2.x`.
CUDA is auto-detected at runtime; fallback to CPU if absent.

**Precision: configurable, auto-detect default.**

```python
precision: Literal["bf16-mixed", "16-mixed", "32", "auto"] = "auto"
```

| Detected hardware | `auto` picks |
|---|---|
| Ampere / Ada / Hopper / Blackwell | `bf16-mixed` |
| Volta / Turing | `16-mixed` |
| Pascal | `32` |
| CPU | `32` |

**Single-device only (v1).** No DDP, no FSDP, no model parallelism. The
trainer rejects `devices > 1` with `ConfigError`. Multi-device support
is a future-version discussion when model sizes outgrow single-device.

**FP8 / FP4 out of scope for v1.** v1 architectural constraints keep
the door open cheaply:

1. Layer factory routes Linear and LayerNorm through one module (F4).
2. No hand-rolled CUDA kernels in v1.
3. `precision` Literal extends with `"fp8-mixed"` in v2 without API
   change.
4. `seq_sklearn.hardware.detect()` returns a `HardwareTier` enum with
   all six tiers populated in v1 (`CPU`, `PASCAL`, `VOLTA_TURING`,
   `AMPERE_ADA`, `HOPPER`, `BLACKWELL`); v1 branches only on the
   first four.

**Mixed-precision overflow.** Lightning's `GradScaler` is used for
`16-mixed`; loss-scale divergence is logged. Three consecutive
skipped steps abort the run with `TrainingError`.

### N6: Documentation

- `README.md` with a quickstart that fits in one screen and trains a
  binary classifier on synthetic monthly data.
- `docs/` folder: this requirements doc, the architecture doc, longer-form guides.
- `docs/examples/` with runnable Python scripts (preferred over
  notebooks for CI testability).
- API reference auto-generated from docstrings. Tool decided in the
  architecture phase (mkdocs + mkdocstrings, or sphinx). Documentation
  build is gated in CI.

### N7: Performance budgets

Defaults for v1 (TFT, 128 hidden_size, 4 heads, lookback 12):

- Memory: training on 100k entities x 24 months x 30 features fits in
  < 8 GB GPU memory at default config.
- Training time: < 30 minutes on a single A100, T4, or RTX 4090.
- Inference latency: batch of 1024 windows in < 100 ms on CPU,
  < 10 ms on GPU.
- The library is NOT recommended for > 10M entities or > 100 features
  per timestep without sharding the caller does themselves.

Each subsequent model carries its own budget block in the per-family
section.

## Acceptance criteria

### Library-wide criteria (every release)

1. `ruff check`, `ruff format --check`, `pyright` strict all pass.
2. `pytest -m "not slow and not perf"` passes with the 85% line / 80%
   branch coverage gates.
3. `tests/deploy/` smoke test passes against the built wheel.
4. The `/design-review` loop has reached consensus on changes since
   the previous release.
5. The `style-reviewer` agent reports zero CRITICAL findings.
6. `CHANGELOG.md` is updated.
7. A release-candidate wheel installs from TestPyPI and runs a minimal
   end-to-end script.

### v1-specific criteria (TFT release)

8. All F1-F11 requirements are implemented and tested.
9. All N1-N7 requirements are met.
10. Two quickstart examples exist and pass in CI:
    - A binary classifier on synthetic monthly data recovers
      accuracy >= 0.75 on the three-seed median (see N1).
    - A quantile regressor recovers empirical coverage on the nominal
      80% interval in [0.75, 0.85] after conformal calibration.
11. The `/gemini-final-pass design` against this requirements doc and
    the architecture doc surfaces no new CRITICAL findings.

## Open questions

Items resolved during requirements drafting are RESOLVED with the
decision. Items still open are OPEN and feed the design-review loop.

1. **Lookback default.** RESOLVED: 12, configurable. Short histories
   are padded and masked. See F3.
2. **Prediction step default.** RESOLVED: 1, configurable. See F3.
3. **Multi-output / multi-label.** RESOLVED: deferred to v1.1. v1
   architectural constraints in F1, F4, F5 keep the doors open.
4. **Saved-model format.** RESOLVED: PyTorch-native primary, ONNX
   optional. Metadata block in F4.
5. **Categorical encoding.** RESOLVED: fastai heuristic with per-column
   override + cardinality cap. See F3.
6. **Missing data / short tenure.** RESOLVED: mask path handles short
   tenure; NaN-in-features is caller's responsibility. See F3, F9.
7. **Attention output shape.** RESOLVED: frozen dataclass with
   `__iter__` for tuple unpacking. See F1.
8. **Hyperparameter search.** RESOLVED: F7 ships Optuna integration.
9. **PyPI publishable name.** RESOLVED: `seq-sklearn` (verified
   available on PyPI). Renames from earlier `tft-sklearn` and `tft-cls`
   noted in the project history.
10. **Class imbalance.** RESOLVED: multiple strategies in F5, all
    Optuna-tunable.
11. **Calibration.** RESOLVED: temperature default with platt /
    isotonic / none alternatives in F5.

12. **Documentation tool choice.** OPEN. mkdocs + mkdocstrings vs.
    sphinx. Decide in the architecture phase. Both are viable; the
    deciding factor will be how heavily the API reference relies on
    autodoc vs. authored prose.

13. **Recurrent family base implementation timing.** OPEN. The v1 doc
    sketches the recurrent base class; v3 builds it concretely. Open
    question: should the `RecurrentSequenceEstimator` skeleton ship in
    v1 (forward-compatible empty class) or appear fresh in v3?
    Implication: shipping the skeleton in v1 locks the interface
    earlier; deferring means v3 has more freedom to redesign.

14. **Foundation-model adapter strategy.** OPEN. v4 scope. The adapter
    head for Chronos / MOMENT / TimesFM needs to convert a pretrained
    forecasting backbone's output into the library's classification or
    regression task. Open question: pooled-token MLP, learned attention
    over hidden states, or trainable readout per task? Likely needs
    per-model exploration; decide at v4.

15. **Performance-benchmark baseline source.** OPEN. The performance
    regression tests in N1 / `tests/perf/` require checked-in baselines
    per `(hardware, torch-version)` cell. Open question: what hardware
    cells are tracked in v1? Suggested: `(cpu, torch-latest)` and
    `(t4, torch-latest)` as the public baselines, with optional
    self-hosted cells added by contributors.

## Addressed

(populated by the design-review loop)

## Deferred

(populated by the design-review loop)
