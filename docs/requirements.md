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

The library covers two families of deep sequence models. Every model
implements the sklearn estimator contract for both classification and
regression on tabular panel data. The same `TabularToSequence`
preprocessing component feeds every model. The same training pipeline,
calibration strategies, and Optuna search infrastructure apply across
all of them.

Model selection in scope is constrained by what is **validated for
classification and regression** outside the forecasting setting. The
v2 transformers (PatchTST, TimesNet, TST) have native classification
heads in their papers or reference implementations. The v3 recurrent
family is the historical home of time-series classification
(LSTM-FCN was explicitly proposed for it). TFT is the exception: the
original paper targets multi-horizon forecasting only, so the v1 TFT
classifier and regressor are a genuine architectural adaptation, not a
wrapper over an existing classifier. That adaptation is the project's
core contribution in v1.

### Recurrent family (planned for v3)

- **LSTM**: long short-term memory networks, optionally bidirectional.
  Encoder-decoder LSTM architectures have direct precedent in time
  series classification. v3 uses
  `nn.utils.rnn.pack_padded_sequence(enforce_sorted=False)` for
  variable-length handling and AWD-LSTM-style weight-drop dropout by
  default (PyTorch's `nn.LSTM(dropout=p)` is inter-layer only, not
  recurrent).
- **GRU**: gated recurrent unit. 2024-2025 comparative studies show
  GRU and LSTM are statistically indistinguishable on classification
  accuracy with GRU running 20-30% faster; v3's quickstart defaults
  to GRU.
- **LSTM-FCN**: hybrid combining an LSTM branch (long-range
  dependencies) with a fully-convolutional 1D branch (local patterns),
  concatenated before the classification head. Karim et al.
  (arXiv 1709.05206; ALSTM-FCN follow-up arXiv 1801.04503) proposed
  LSTM-FCN / ALSTM-FCN specifically for end-to-end time series
  classification on the UCR benchmark. **Regression caveat**: no
  published LSTM-FCN regression variant exists; v3 ships LSTM-FCN
  regression as an unvalidated extension of the family base, with a
  warning emitted at fit time. ConvTran and ROCKET-family methods
  outperform LSTM-FCN on multivariate benchmarks in 2024-2025
  (MONSTER 2025); LSTM-FCN ships as a small / fast / interpretable
  baseline, not as a leaderboard target.

### Transformer family (v1 first, v2 expansions)

- **TFT**: Temporal Fusion Transformer (Lim et al., 2021). **v1**, the
  first model the library ships. The original paper targets multi-horizon
  forecasting only; adapting it to classification and standard regression
  is genuine architectural work: swap the quantile-forecast head for a
  softmax / sigmoid / regression head, and replace the quantile loss
  with cross-entropy / MSE / pinball as appropriate. People have done
  this in ad hoc ways, but no native classification mode exists in the
  reference. The detailed v1 spec lives in the "v1 concrete: TFT"
  section below.
- **PatchTST**: patch-based time-series transformer (Nie et al.,
  ICLR 2023; arXiv 2211.14730). Treats sub-sequences as patches; the
  Hugging Face `PatchTSTForClassification` adds a CLS / mean / max
  pool head, and `PatchTSTForRegression` adds a regression head. The
  Time-Series-Library reference impl has a flatten-and-Linear
  classification head that bakes `seq_len` into the weight matrix
  (incompatible with seq-sklearn's variable-history mask path); v2
  ships the HF-style pool head, not the TSL flatten head. v2.
- **TimesNet**: period-aware decomposition with 2D convolutions (Wu et
  al., 2023; arXiv 2210.02186). The paper evaluates classification as
  one of five mainstream tasks alongside forecasting, imputation, and
  anomaly detection, but the reference classification head is
  flatten-and-Linear with `seq_len` baked in; v2 replaces it with a
  pool head. TimesNet is a CNN over a period-reshaped tensor, not a
  transformer, but lives in the transformer family for the abstraction
  taxonomy (attention-like masked operations and shared mask handling).
  The FFT period detection in the reference impl is batch-global and
  mask-unaware; v2 adds a mask-aware FFT variant. v2.
- **TST**: vanilla time-series transformer (Zerveas et al., 2021;
  arXiv 2010.02803). The paper explicitly proposes a transformer
  framework for unsupervised representation learning of multivariate
  time series with downstream regression, classification, and
  imputation. The reference (`gzerveas/mvts_transformer`) is the
  canonical implementation; the THUML Time-Series-Library does NOT
  ship TST. The original classification head is flatten-and-Linear
  (same `seq_len`-baked friction as PatchTST / TimesNet); v2 ships a
  pool head. Defaults to BatchNorm1d, which crashes on batch-of-1;
  v2 defaults to LayerNorm with BatchNorm selectable. ConvTran
  (Foumani 2023; ECML-PKDD; arXiv 2305.16642) is the noteworthy
  refinement, deferred to v2.x. v2.

### Experimental (future exploration)

- **iTransformer** (Liu et al., 2024): inverted attention; attends
  across variables instead of time. The paper positions it exclusively
  as a forecasting backbone, and no classification or regression
  evaluation appears in the literature. The "attention across
  variables, FFN across time" inversion is unlikely to produce useful
  representations for sequence classification where the discriminative
  signal is the temporal pattern within each variable. Kept on a watch
  list, not on the roadmap. Revisit if external classification or
  regression results surface.

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
   The model sizes targeted across the v1-v3 roadmap (typically 1-10M
   parameters) do not require distributed training; if a future model
   class outgrows single-device, that is a later-version discussion.
8. **Multi-horizon forecasting in the regression head.** Each window
   predicts one target value at one configurable `prediction_step`,
   matching the classifier's contract. Multi-step output is a separate
   axis that fits the forecasting libraries above better than this one.

## Roadmap

| Version | Scope |
|---|---|
| **v1** | TFT classifier and regressor; full `TabularToSequence` pipeline; Optuna; all library-wide infrastructure detailed in this doc |
| v1.1 | Multi-output regression, multi-label classification (architectural constraints already in v1) |
| v2 | PatchTST, TimesNet, TST (transformer family completion) |
| v3 | LSTM, GRU, LSTM-FCN (recurrent family) |

iTransformer is tracked under "Experimental (future exploration)"
above. Foundation models (Chronos, MOMENT, TimesFM) were considered
during scoping and dropped: of the three, only MOMENT has documented
classification support, and a one-model family does not justify the
adapter-head abstraction work. Revisit if a second
classifier-capable foundation model enters the field.

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
   N7 in this doc). Everything every model needs. Built once in v1.
   Stable thereafter.

2. **Family-level abstractions** (section "Per-family architectural
   patterns" below). Each model family carries shared patterns:
   recurrent models share BPTT and hidden-state semantics, transformer
   models share attention-mask handling. v1 defines the abstraction
   for the transformer family (since TFT lives there); v3 adds the
   recurrent abstraction when its first model ships.

3. **Concrete models**. Each model is a thin shell that plugs into the
   family abstraction with its architecture and hyperparameters. v1's
   only concrete model is TFT (section "v1 concrete: TFT" below).

This is the architectural goal that lets the library ship TFT in v1 and
add six more models over subsequent versions without rewriting the
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

- `fit(X, y, *, calibration_set=None, optuna_trial=None)` where `X`
  is a pandas DataFrame in the panel shape above, `y` is array-like
  of targets aligned to the rows of `X`, `calibration_set` is an
  optional `(X_cal, y_cal)` tuple matching the same shape contract,
  and `optuna_trial` is an optional `optuna.Trial` instance threaded
  into the Trainer for in-training pruning. See F2 for the
  three-way-split rules and the `calibration_set` / `cal_fraction`
  interaction. See F7 for the `optuna_trial` plumbing; the keyword
  lives on `fit` (not `__init__`) to preserve the pydantic config
  schema's `extra="forbid"` contract and to keep the trial out of
  `get_params` / `set_params` / `save` / `load`.
- `predict(X)` returning class predictions (classifier) or point
  predictions (regressor).
- `score(X, y)` returning accuracy by default for classifiers and R² for
  regressors, with a `scoring` argument for a callable scorer.
- `get_params(deep=True)` and `set_params(**params)` so every estimator
  composes into `sklearn.pipeline.Pipeline` and
  `sklearn.model_selection.cross_val_score`.
- `save(path)` and `load(path)` for safetensors + JSON serialization
  (the two-file directory format described in F4; no pickle in the
  public artifact).
- `export_onnx(path)` for ONNX export via `torch.onnx.export(dynamo=True)`
  at opset 20. The dependency on `onnx` and `onnxruntime` lives in an
  optional extra (`pip install seq-sklearn[onnx]`). Attention export
  uses the math backend of `F.scaled_dot_product_attention` selected
  via `torch.nn.attention.sdpa_kernel(SDPBackend.MATH)`; the fused
  flash and mem-efficient backends do not export to ONNX cleanly.
- `predict_with_attention(X)` (or `predict_with_states(X)` for recurrent
  models) (**BETA**, see stability tiers) returning the prediction plus
  model-introspection outputs as a frozen dataclass with attribute
  access only. Raises `NotFittedError` (F8) if `fit` has not been
  called. Tuple unpacking via `__iter__` is NOT supported on these
  outputs because their field set is BETA: new fields land in MINOR
  releases, and tuple-unpacking callers would break on the addition.
  Stable outputs (`predict`, `predict_proba`, `predict_quantiles`)
  return plain arrays and have no such concern.

Classifier-only:
- `predict_proba(X)` returning class probabilities.

Regressor-only:
- `predict_quantiles(X, quantiles=None)` returning quantile estimates
  per row. Requires the model to have been fit with `loss_strategy=
  "pinball"` (see F5). Raises `seq_sklearn.errors.NotFittedError`
  (subclass of `sklearn.exceptions.NotFittedError`) if `fit` has not
  run; raises `PredictionError` on point-mode regressors with a message
  naming `predict` as the supported alternative. The `quantiles`
  argument behavior:
  - `quantiles=None` (default): returns the full fit-time quantile
    vector exactly.
  - A subset of the fit-time quantile vector: returns those columns in
    the order requested.
  - Any value not present in the fit-time vector: raises `ValueError`
    naming the fit-time vector. v1 does NOT interpolate.

Every classifier and regressor passes the named subset of
`sklearn.utils.estimator_checks.check_estimator` listed in F1.1 and F8
below.

### F1.1: sklearn fit-state attributes and tags

The library opts into sklearn's estimator-check contract for the
subset listed below. These attributes and tags are part of the public
API; breaking changes require MAJOR.

**Required fit-state attributes** (set during `fit`, present afterward):

| Attribute | Classifier | Regressor | Shape / type |
|---|---|---|---|
| `classes_` | yes | no | 1D ndarray, sorted by `LabelEncoder` |
| `n_features_in_` | yes | yes | int, count of *declared columns* (`static_categorical_cols + static_real_cols + time_varying_real_cols + time_varying_categorical_cols`), not raw DataFrame columns |
| `feature_names_in_` | yes | yes | 1D ndarray of strings, the declared column names in the order above |
| `n_outputs_` | yes | yes | int, 1 in v1; v1.1 sets per task |
| `quantiles_` | no | yes (quantile mode) | 1D ndarray of fit-time quantiles |
| `decision_threshold_` | yes (binary, threshold-tuned) | no | float, used by `predict` |

`feature_schema_fingerprint` is the sha256 over the declared column
list and dtypes; matches `save`-time metadata (F4).

**sklearn tags.** Estimators declare tags via `__sklearn_tags__`, an
**instance method** introduced in sklearn 1.6 (NOT a classmethod;
implementing it as a classmethod silently breaks tag chaining, see
sklearn issue #30479). The method must chain `super().__sklearn_tags__()`
and mutate the returned `sklearn.utils.Tags` dataclass.

The v1 tag block reads roughly:

```python
def __sklearn_tags__(self) -> Tags:
    tags = super().__sklearn_tags__()
    tags.input_tags.dataframe = True       # accepts pandas DataFrame
    tags.input_tags.two_d_array = False    # NOT a numpy-array estimator
    tags.input_tags.allow_nan = False
    tags.target_tags.required = True
    tags.requires_fit = True
    tags.non_deterministic = False         # CPU FP32; flipped True on mixed precision (N5)
    return tags
```

The legacy `X_types = ["dataframe"]` list form (sklearn pre-1.6) and the
`_xfail_checks` dict on the estimator are both removed in 1.6; do not
use either. Skipped `check_estimator` checks are declared via the
`expected_failed_checks` argument to `parametrize_with_checks` /
`check_estimator`, not on the estimator class itself.

**Punted methods.** `partial_fit`, `fit_predict`, `fit_transform` (on
the estimators themselves; `TabularToSequence` does support
`fit_transform`) are NOT implemented in v1. Calling raises
`NotImplementedError("partial_fit is not supported in seq-sklearn v1;
see docs/roadmap.md")` with a stable message.

**`check_estimator` subset.** sklearn 1.6+ exposes
`parametrize_with_checks` decorator on the test side. The library
passes the full set of checks that DOES NOT exercise raw-array input,
multi-output `y` (v1.1), or `partial_fit`.

**Checks expected to FAIL** (passed as `expected_failed_checks` with
`xfail_strict=True` so a silent compatibility win breaks CI and
forces a doc update; the architecture phase pins the exact list,
the entries below are mandatory):

- `check_methods_sample_order_invariance`: attention is intentionally
  order-sensitive; reordering input rows changes the prediction.
- `check_fit_idempotent`: stochastic optimizer plus non-deterministic
  CUDA paths under mixed precision; only guaranteed on the CPU FP32
  deterministic path (see N4).
- `check_estimators_dtypes` / `check_dtype_object`: DataFrame input
  contract; numpy-array equivalents not supported.

**Checks expected to PASS** (NOT in `expected_failed_checks`; listed
here as the intentional baseline so a future regression that breaks
them is recognizable):

- `check_pandas_column_name_consistency`: passes; the library
  validates DataFrame column names against `feature_names_in_` at
  predict time.
- `check_n_features_in` / `check_n_features_in_after_fitting`: passes
  using the declared-column count, not the DataFrame's raw column
  count (F1.1 attribute table).
- `check_methods_subset_invariance`: passes; the library guarantees
  identical predictions on a subset of inputs.

The expected-failure list and the documented-passing list are two
separate constants in `tests/conftest.py` (`EXPECTED_FAILED_CHECKS`
and `EXPECTED_PASSING_CHECKS`); only the former is passed to
`parametrize_with_checks(..., expected_failed_checks=...)`. The
documented-passing list drives a meta-test that asserts each named
check appears in the test-collected check IDs (catches a sklearn
upgrade that renames or removes one of the passing checks).

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

At `fit` time the target column is passed via `y`. Validation checks
column existence, that `(id_col, time_col)` is unique, and that the
time column is sortable. No fixed schema is imposed; the caller picks
which columns map to which role.

**Time-axis semantics.**

- **Time-column dtype.** Accepted dtypes are `datetime64[ns]` and
  `datetime64[ns, <tz>]`, pandas `PeriodDtype`, and signed integer
  numpy dtypes (`int32`, `int64`). Object-dtype columns (including
  mixed string and Timestamp content) raise `DataContractError` even
  if elements happen to be sortable, since cross-row comparison is
  not well-defined.
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

**Three-way split policy (train / val / calibration).** Training
requires three disjoint folds: a training fold for optimization, a
validation fold for early stopping and model selection, and a
calibration fold for post-hoc calibration and threshold tuning.
Fitting calibration on the early-stopping fold produces optimistic
calibration estimates because the early-stopping criterion has already
been optimized against that data, so v1 keeps them separate.

- Default split: time-ordered per entity. The last `cal_fraction` rows
  form the calibration fold (default 0.1), the preceding `val_fraction`
  rows form the validation fold (default 0.1), the remainder is
  training. `val_fraction + cal_fraction < 1` enforced by config
  validator.
- When `calibration_strategy="none"` and `threshold_tuning=False`, the
  calibration fold is folded back into training and no separate fold is
  drawn.
- Random splits available via `val_split_strategy="random"` but emit a
  `UserWarning` when more than one entity is present (random splits
  leak future information on panel data).
- Callers who already have a calibration set pass it via the optional
  `calibration_set=(X_cal, y_cal)` keyword to `fit`. When
  `calibration_set` is provided AND `cal_fraction > 0`, the library
  raises `ConfigError` at fit time naming both fields; callers must
  set `cal_fraction=0.0` explicitly to opt into externally-supplied
  calibration. The validation fraction (`val_fraction`) is unchanged
  by this path: train + val are still drawn from `(X, y)` with the
  full `1 - val_fraction` / `val_fraction` split. `X_cal` and `y_cal`
  go through the same `TabularToSequence` pipeline as training data.

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
   target, and a per-timestep boolean `padding_mask` marking positions
   that are real data versus padding.

**Mask convention.** The internal canonical name is `padding_mask`
with the convention `True = padding (ignore)`. This matches
`nn.MultiheadAttention.key_padding_mask` and the `pack_padded_sequence`
length-based path. The library flips polarity once at the SDPA
boundary because `F.scaled_dot_product_attention`'s boolean `attn_mask`
uses the opposite convention (`True = participate`). The flip lives
in one helper in `seq_sklearn.models._attention` and is the only
place in the codebase where the polarity convention switches.

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

**`save` / `load` format.** `save(path)` writes a directory at `path/`
containing two files. The two-file split exists to enforce
"no pickle in the public artifact":

- `path/weights.safetensors`: tensor-only archive in the safetensors
  format. Holds the state dicts (backbone, head, calibrator if it
  exposes tensor state) plus tensorizable fit-state (`classes_`,
  `n_features_in_`, `n_outputs_`, `quantiles_` as a 1-D tensor,
  `decision_threshold_` as a 0-D tensor when present).
- `path/state.json`: human-readable JSON. Holds the pydantic config
  dump (`model_dump`), `feature_names_in_` (list of strings), the
  `tabular_to_sequence_state` (categorical-encoder vocabularies as
  arrays of strings; scaler statistics as floats), the calibrator's
  `serialize()` output (each calibrator returns a JSON-compatible
  dict), and the metadata block below.

**Why two files.** PyTorch 2.6+ flipped `torch.load`'s `weights_only`
default to `True` and refuses non-tensor Python objects unless
explicitly allowlisted; we never opt out, so the library cannot put
scalers / encoders / pydantic objects inside a `.pt` archive without
inviting a pickle-based code-execution path on `load`. Safetensors is
the 2025-2026 standard for tensor-only serialization (used by
Hugging Face transformers, diffusers, etc.).

**Metadata block** (inside `state.json`):

- `seq_sklearn_version`
- `torch_version`
- `cuda_version` (or `null`)
- `python_version`
- `feature_schema_fingerprint` (sha256 of sorted column names +
  dtypes from the fit-time `X`)
- `precision_resolved` (the concrete value `auto` resolved to at fit
  time per N5)
- `created_at` (ISO 8601 timestamp)
- `schema_version`: integer (v1 ships `1`). `load` rejects future
  major schema versions with `PredictionError("checkpoint schema X
  newer than library v1")`; older majors that the library still
  understands receive a single migration pass.

**`load(path)`** reconstructs the estimator. Loads `weights.safetensors`
via the safetensors library (no `torch.load` call). Emits a
`UserWarning` on any version-mismatch field. No `trust=True` /
`weights_only=False` escape hatch exists in v1; a future model that
requires pickled state would be a MINOR bump with an explicit security
note.

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
- DataLoader defaults: `num_workers=min(4, os.cpu_count() or 1)`
  (the `or 1` fallback handles indeterminable-CPU-count environments
  where `os.cpu_count()` returns `None`, which would otherwise raise
  `TypeError` in the `min()` call),
  `pin_memory=True` when CUDA, `persistent_workers=True` when
  `num_workers > 0`. All overridable.
- Gradient accumulation via `accumulate_grad_batches: int = 1`.
- Gradient clipping via `gradient_clip_val: float | None = None`.

**Optimizers (supported menu).** Single `optimizer` field on
`BaseModelConfig`.

| Optimizer | Notes |
|---|---|
| `adamw` | Default. `weight_decay` field applies. |
| `adam` | No decoupled weight decay. |
| `sgd` | `momentum=0.9` constant; for callers who want a non-adaptive baseline. |

**Learning-rate schedulers (supported menu).**

| Scheduler | Notes |
|---|---|
| `constant` | No schedule. `warmup_steps` ignored; setting it raises `ConfigError`. |
| `cosine_with_warmup` | Default. `warmup_steps` configurable (default 100). |
| `one_cycle` | OneCycleLR-style. `warmup_steps` re-interpreted as the fraction of total steps spent warming up. |
| `reduce_on_plateau` | Step on validation metric stall. `warmup_steps` ignored. |

**Loss strategies, imbalance strategies, calibration strategies.**
Three independent fields on the config:

- `loss_strategy`: the loss function class for classifiers, the loss
  family for regressors. Each value is gated against task type by the
  validity matrix below; the prose here lists task gating inline:
  `cross_entropy` (classifier tasks `binary` / `multiclass` /
  `multilabel`; default for `binary` / `multiclass`), `focal`
  (classifier tasks; uses focal loss with configurable `gamma`, default
  2.0), `mse` (regression point and multi-output; default for
  `regression_point`), `mae` (regression point and multi-output),
  `huber` (regression point and multi-output), `pinball` (only legal
  for `regression_quantile`). Selecting `focal` automatically disables
  class-weighting inside the loss; the imbalance handling moves to the
  sampler side. Selecting `pinball` requires `quantiles` to be set.
- `imbalance_strategy`: `none` (default, only legal value when
  `task_type` is any `regression_*`; non-`none` on a regression task
  raises `ConfigError`), `class_weighted` (applies frequency-based
  per-class weights to the loss; requires `loss_strategy=cross_entropy`),
  `oversample_minority` (sampler-side; ratio configurable),
  `undersample_majority` (sampler-side; ratio configurable). Threshold
  tuning is a separate post-hoc step (below) and combinable with any
  classification `imbalance_strategy`.
- `calibration_strategy`: `none`, `temperature` (default, classifier),
  `platt` (binary classifier only), `isotonic` (classifier),
  `conformal` (default, regression with `pinball` loss),
  `isotonic_quantile` (regression with `pinball` loss).

**Validity matrix (cross-field validator at `BaseModelConfig`).**
Enumerated mechanically per `(task_type, loss_strategy)` pair. Any
combination NOT listed below raises `ConfigError` at config
construction. The validator implementation and the Optuna search-space
sampler both consume this enumeration.

**Canonical field domains.** The illegal-combo test in N1 derives its
parametrization from the Cartesian product of the full domains listed
below minus the legal cells in the table. These domain enumerations
are the single source of truth and live as module-level constants in
`seq_sklearn.config._domains`:

- `TASK_TYPES`: `binary`, `multiclass`, `multilabel` (v1.1),
  `regression_point`, `regression_quantile`, `regression_multioutput`
  (v1.1).
- `LOSS_STRATEGIES`: `cross_entropy`, `focal`, `mse`, `mae`, `huber`,
  `pinball`.
- `IMBALANCE_STRATEGIES`: `none`, `class_weighted`, `oversample_minority`,
  `undersample_majority`.
- `CALIBRATION_STRATEGIES`: `none`, `temperature`, `platt`, `isotonic`,
  `conformal`, `isotonic_quantile`.

v1.1 task types are included in the v1 domain so the v1 validator
rejects them with a clear "scheduled for v1.1" message rather than
silently failing on shape mismatch later. The illegal-combo test
parametrizes over v1 task types only in v1; v1.1 rows are
xfail-marked.

| `task_type` | `loss_strategy` | Legal `imbalance_strategy` | Legal `calibration_strategy` |
|---|---|---|---|
| `binary` | `cross_entropy` | `none`, `class_weighted`, `oversample_minority`, `undersample_majority` | `none`, `temperature`, `platt`, `isotonic` |
| `binary` | `focal` | `none`, `oversample_minority`, `undersample_majority` | `none`, `temperature`, `platt`, `isotonic` |
| `multiclass` | `cross_entropy` | `none`, `class_weighted`, `oversample_minority`, `undersample_majority` | `none`, `temperature`, `isotonic` |
| `multiclass` | `focal` | `none`, `oversample_minority`, `undersample_majority` | `none`, `temperature`, `isotonic` |
| `multilabel` (v1.1) | `cross_entropy` | `none`, `oversample_minority`, `undersample_majority` | `none`, `temperature`, `isotonic` |
| `multilabel` (v1.1) | `focal` | `none`, `oversample_minority`, `undersample_majority` | `none`, `temperature`, `isotonic` |
| `regression_point` | `mse` | `none` | `none` |
| `regression_point` | `mae` | `none` | `none` |
| `regression_point` | `huber` | `none` | `none` |
| `regression_quantile` | `pinball` | `none` | `none`, `conformal`, `isotonic_quantile` |
| `regression_multioutput` (v1.1) | `mse` | `none` | `none` |
| `regression_multioutput` (v1.1) | `mae` | `none` | `none` |
| `regression_multioutput` (v1.1) | `huber` | `none` | `none` |

`ConfigError` messages name the offending combination and the legal
alternatives. The Optuna default search space (F7) samples only from
the legal cells.

**Threshold tuning (classifier only).** Post-hoc adjustment of the
decision boundary on the held-out calibration fold (F2). Independent of
`loss_strategy`, `imbalance_strategy`, and `calibration_strategy`; runs
after calibration. Controlled by `threshold_tuning: bool = False` and
`threshold_metric: Literal["f1", "balanced_accuracy", "youden_j"] =
"f1"`. Output: a stored decision threshold that `predict` consults; raw
probabilities are unaffected.

**Concrete loss-class dispatch.** The loss factory maps
`(task_type, loss_strategy)` to a torch module:

| `task_type` | `loss_strategy` | Class |
|---|---|---|
| `binary` | `cross_entropy` | `BCEWithLogitsLoss` |
| `binary` | `focal` | Binary focal loss with `gamma` |
| `multiclass` | `cross_entropy` | `CrossEntropyLoss` |
| `multiclass` | `focal` | Multi-class focal loss with `gamma` |
| `multilabel` (v1.1) | `cross_entropy` | `BCEWithLogitsLoss` per output |
| `multilabel` (v1.1) | `focal` | Binary focal per output |
| `regression_point` | `mse` / `mae` / `huber` | `MSELoss` / `L1Loss` / `HuberLoss(delta)` |
| `regression_quantile` | `pinball` | Pinball loss over configured quantiles |
| `regression_multioutput` (v1.1) | `mse` / `mae` / `huber` | Same, over `(N, n_outputs)` |

Regression output tensors have shape `(N, K)` where
`K = n_outputs * n_quantiles`. v1 is `(N, 1)` for point and
`(N, len(quantiles))` for quantile. v1.1 multi-output reshapes to
`(N, n_outputs)` / `(N, n_outputs, len(quantiles))` without changes to
the loss.

**Calibration mechanics.**

- Classification calibration runs on the calibration fold (F2's
  three-way split: train, val-for-early-stop, calibration). Strategies
  enumerated above are fit there and applied at `predict_proba`.
- Regression calibration runs on the same calibration fold for
  `conformal` and `isotonic_quantile`; `pinball` loss is required (per
  the validity matrix). Applied at `predict_quantiles`.
- `predict_quantiles` returns the **fit-time** quantile vector exactly.
  A predict-time `quantiles=` argument that requests values outside the
  fit-time set raises `ValueError`; requesting a strict subset is
  allowed and just selects columns. No interpolation is performed in
  v1. The fit-time vector is the stored ground truth for the calibrator
  and conformal adjustment.

The Optuna search space (F7) samples `loss_strategy`,
`imbalance_strategy`, and `calibration_strategy` only from the legal
cells in the validity matrix above.

### F6: Synthetic data generators

The library ships generators that produce panels of `(id, time,
features, target)` at configurable period grains: day, week, month,
quarter, year. Used by the test suite and as documentation examples.

**Data-generating process.** The DGP is fully specified so acceptance
thresholds (N1) are reproducible across re-implementations:

All sampling uses `numpy.random.Generator(PCG64(seed))`. A single
`Generator` instance is created at the start of `generate(seed=...)`
and threaded through every sampling step in the order listed below.
Splitting into substreams or constructing new `Generator` instances
mid-way is forbidden, since reordering breaks the byte-identical
contract.

1. Sample static categorical levels uniformly from `{0, ..., K_i-1}`
   per categorical column `i`, fixed across the entity's rows.
2. Sample static real values from `N(0, 1)` per static-real column,
   fixed across the entity's rows.
3. Generate time-varying real columns as an AR(1) process per entity:
   `x_t = 0.7 * x_{t-1} + N(0, 1)`, then add observation noise
   `N(0, noise_level)`. The initial state is drawn from the stationary
   distribution: `x_0 ~ N(0, 1 / sqrt(1 - 0.7^2))`. No burn-in.
4. Generate time-varying categorical columns by Markov-chaining over a
   fixed transition matrix sampled at generator construction time. The
   transition matrix is sampled per-column as
   `row_i ~ Dirichlet(alpha=ones(K_i))`. The initial state of each
   entity is drawn as `rng.integers(0, K_i)` (the single threaded
   `Generator`), giving a uniform distribution over `{0, ..., K_i-1}`.
   Subsequent states are drawn as `rng.choice(K_i, p=transition[prev_state])`.
5. Build a fixed projection matrix `W` once per generator
   (`W ~ N(0, 1)`, shape `(num_target_outputs, len(phi))`, frozen by
   seed at generator construction time).
6. Build the temporal weight vector `w_temporal` once per generator
   (frozen by seed at construction time): draw `w_raw ~ Dirichlet(alpha=
   ones(lookback))`, then enforce at least 3 non-trivial entries by
   the following deterministic procedure: zero out entries below `0.05`,
   renormalize; if fewer than 3 non-zero entries remain, redraw with
   the same Generator (advances the stream). Cap retries at 5. If the
   retry cap is reached, apply the fallback: identify the 3 entries of
   the most recent post-clip vector with the smallest non-zero
   values; if the post-clip vector has fewer than 3 non-zero entries,
   pick from the full pre-clip draw the 3 smallest values; in either
   case break ties by lowest index. Force those 3 entries to `0.1` and
   renormalize the full vector. This is deterministic for every seed
   and guarantees `w_temporal` always has at least 3 entries carrying
   load.
7. Compute `phi(window) = concat([static_real_features, static_cat_embeddings,
   sum_t w_temporal[t] * time_varying_real_features[t],
   last_time_varying_cat_embedding])`. `static_cat_embeddings` and the
   categorical embedding table are `N(0, 1)` matrices sampled once per
   generator and frozen.
8. Compute the target signal:
   `z = signal_strength * (W @ phi(window)) + (1 - signal_strength) * eps`
   where `eps ~ N(0, 1)` (per window, fresh draw from the same
   `Generator`).
9. Target sampling:
   - `binary`: `z` is a scalar logit; sample `y ~ Bernoulli(sigmoid(z))`.
     `class_balance` shifts `z` by the logit-offset needed to hit the
     requested positive rate in expectation.
   - `multiclass`: `z` is a `num_classes`-vector; sample
     `y ~ Categorical(softmax(z))`. `class_distribution` shifts the
     per-class biases to hit the requested marginal in expectation.
   - `regression_*`: `y = target_scale * z + N(0, target_noise)`.

The DGP version is stamped into `generator.dgp_version` and bumped on
any change. Acceptance thresholds in N1 are pinned to a specific
`dgp_version`.

**Canonical seeds.** Acceptance tests use the seed triple
`(42, 137, 9999)`. The "three-seed median" in N1 means the median of
results across these three seeds. Changing the canonical triple bumps
MINOR with a CHANGELOG entry.

**Each generator exposes:**

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
- `noise_level` (feature-side observation noise)
- `signal_strength` (in `[0, 1]`; how strongly the injected pattern
  drives the target relative to noise; lets tests assert the model
  actually recovers it)
- `seed`

**Coverage requirements.** The test suite includes at least one scenario
per period grain per `target_kind` where `periods_per_entity` ranges
from 1 to a moderate maximum (e.g. `(1, 60)` for monthly). At least one
e2e scenario per `target_kind` MUST produce a panel where at least one
entity has exactly 1 row and at least one entity has the full lookback
length, in the same `fit` call. This exercises the variable-history
path end-to-end and prevents the sampling variance from silently
excluding the 1-row case.

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
  `prediction_step` (F3) when the caller wants them tuned. The search
  space samples only from the legal cells of the F5 validity matrix.
- A runnable example at `docs/examples/optuna_search.py`.
- A pruning hook native to `_LightningModule` using the
  deferred-raise pattern: `on_validation_epoch_end` calls
  `trial.report(value, step=current_epoch)` and stashes the prune
  decision when `should_prune()` returns true;
  `on_train_epoch_end` raises `optuna.TrialPruned` at the END of
  the hook so the `train.epoch` and entropy events fire for the
  pruned epoch first. The library does NOT ship
  `optuna_integration.PyTorchLightningPruningCallback` because
  that callback raises from `on_validation_end`, before Lightning
  fires `on_train_epoch_end`, which would skip the structured-log
  events for the pruned epoch.
- The trial reaches `_LightningModule` via the `fit` keyword:
  `estimator.fit(X, y, optuna_trial=trial)` per F1. The trial is
  NOT a pydantic config field and NOT an `__init__` argument; this
  preserves `extra="forbid"` on the config and keeps the trial out
  of `get_params` / `save` / `load`.
- `MedianPruner` and `HyperbandPruner` are both supported. Tests
  asserting prune-at-epoch-0 behavior must construct the pruner with
  `MedianPruner(n_startup_trials=0, n_warmup_steps=0, n_min_trials=1)`;
  without `n_min_trials=1` the prune check defers and the test
  becomes flaky.

**Trial failure modes.** Optuna's default `study.optimize(catch=())`
means a non-`TrialPruned` exception terminates the study. The
trial-failure conversion is the user's `objective(trial)` wrapper's
job, not the library's Trainer. The library ships an
`optuna_trial_guard(trial)` context manager that callers wrap around
the entire objective body; `ConfigError` (illegal combo, cardinality
cap, etc.) and `TrainingError` (NaN loss, divergence) inside the
guard are re-raised as `optuna.TrialPruned` with the original message
attached. `DataContractError`, `KeyboardInterrupt`, and unexpected
exceptions propagate so the study fails fast on genuine bugs.

**Testing `suggest_params`.** Unit tests use a real
`optuna.create_study().ask()` loop to sample trials that actually
exercise the search space. A 1000-iteration sweep asserts every
sampled config passes the validity-matrix validator (`FixedTrial`
returns deterministic pre-loaded values and would not exercise the
sampling logic; `FixedTrial` remains the right tool for
`optuna_trial_guard` tests where the wrapped body needs
`report` / `should_prune` to be no-ops without standing up a
Study).

The default search space is ALPHA stability (may change without MINOR
bump); pass an explicit search space for stable behavior.

### F8: Error contract

The library defines a single exception hierarchy under `seq_sklearn.errors`:

```
SeqSklearnError                  base class (subclass of Exception)
├── ConfigError                  pydantic validation failures, config inconsistencies, illegal F5 combos
├── DataContractError            F2 column-existence, uniqueness, dtype, tz violations, NaN-in-features
├── TrainingError                NaN loss, divergence, calibration-set failures
├── PredictionError              shape mismatches at predict, model-not-loaded
└── NotFittedError               estimator used before fit; ALSO subclasses sklearn.exceptions.NotFittedError
```

`NotFittedError` is the sklearn-compatible exception raised by every
public method that requires fit (`predict`, `predict_proba`,
`predict_quantiles`, `predict_with_attention`, `score`,
`export_onnx`, `save`). It subclasses both `SeqSklearnError` and
`sklearn.exceptions.NotFittedError` so callers can catch on either
contract and `check_estimator`'s `check_estimators_unfitted` passes
without modification.

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
  `precision="32-true"` or a lower learning rate.
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

**Structured records.** All observability events use
`logger.info(message, extra={"event": "<name>", ...})` with a stable
event name and a documented payload schema in `docs/observability.md`.
Lightning callbacks emit the records; user code reads them via a
standard `logging.Handler` or via the lightning logger of choice.

Required events in v1:

| Event name | Level | When | Payload keys |
|---|---|---|---|
| `train.grad_norm` | DEBUG | every step | `step`, `grad_norm`, `lr` |
| `train.epoch` | INFO | end of epoch | `epoch`, `train_loss`, `val_loss`, `val_metric` |
| `train.var_selection_entropy` | INFO | end of epoch (attention models) | `epoch`, `static_entropy`, `temporal_entropy` |
| `train.attention_entropy` | INFO | end of epoch (attention models) | `epoch`, `entropy_per_head` |
| `train.hidden_norm` | INFO | end of epoch (recurrent models; v3 only, no v1 emission path) | `epoch`, `mean_hidden_norm` |
| `train.nan_step_skipped` | WARNING | single NaN-loss step | `step`, `consecutive_nan_count` |
| `train.mixed_precision_diverged` | ERROR | three consecutive GradScaler-skipped steps abort the run (N5/F9) | `step`, `precision`, `consecutive_skipped`, `reason` |
| `calibration.fit` | INFO | calibration fit completes | `strategy`, `cal_size`, `pre_ece`, `post_ece` (classifier) / `pre_coverage`, `post_coverage` (regressor) |
| `calibration.small_set` | WARNING | calibration fold has < 100 rows | `strategy`, `cal_size`, `min_recommended` |
| `optuna.trial_pruned` | INFO | trial pruned (any reason) | `trial_number`, `epoch`, `reason` |
| `data.duplicate_floor_breach_count` | WARNING | predict() entities below `min_periods_predict` | `count`, `min_periods_predict` (aggregated, one record per call) |
| `data.unseen_categories` | INFO | predict() saw unseen categorical values | `column`, `count` |
| `hardware.detect` | INFO | hardware detection runs at trainer setup | `tier`, `cuda_compute_capability`, `selected_precision` |

Lightning loggers (`MLFlowLogger`, `WandbLogger`, `TensorBoardLogger`)
are pass-through; the library does not depend on any specific tracking
backend. An MLflow + Optuna example ships at
`docs/examples/mlflow_search.py`.

## Per-family architectural patterns

The library factors family-specific shared patterns into family base
classes. v1 defines the transformer-family base (since TFT lives there).
v2 and v3 add the other family bases when their first model ships.

### Transformer family (`TransformerSequenceEstimator`)

Base class for TFT (v1), PatchTST (v2), TimesNet (v2), TST (v2).
Shared patterns:

- Attention-mask handling for variable-length sequences. The base
  implements the mask broadcast utilities; concrete models call them.
- Multi-head attention factory (uses `F.scaled_dot_product_attention`
  to inherit PyTorch's optimized kernels).
- Positional encoding policy: sinusoidal by default, learned optional,
  per-model overridable.
- Layer factory (Linear, LayerNorm) shared with the library-wide F4
  factory.
- Variable-selection networks: optional family component used by TFT;
  abstracted so other transformers in the family can opt out.

### Recurrent family (`RecurrentSequenceEstimator`, skeleton in v1, concrete in v3)

The skeleton (abstract base class with abstract methods, no concrete
LSTM/GRU code) ships in v1 to validate that `BaseSequenceEstimator`'s
contract supports the recurrent surface without a forward port in v3.
Concrete LSTM, GRU, and LSTM-FCN ship in v3.

**Surface area defined in v1 (abstract methods, no implementation):**

- `_init_hidden(self, batch_size: int, device) -> tuple[Tensor, ...]`
  returning the initial recurrent state. Default v3 strategies: zero,
  learned-parameter, per-entity.
- `_readout(self, hidden_seq: Tensor, mask: Tensor) -> Tensor` producing
  the per-window representation. Default v3 strategies: last valid
  timestep, masked mean pool, attention readout.
- `_bptt_window(self) -> int | None` returning the truncated-BPTT
  window length, or `None` for full BPTT.

**Shared config fields (in `RecurrentSequenceEstimatorConfig`, also
shipped as a skeleton in v1):**

- `bidirectional: bool` (default `False`)
- `recurrent_dropout: float` (default 0.1)
- `recurrent_dropout_kind: Literal["weight_drop", "variational", "bernoulli"]`
  (default `"weight_drop"`; AWD-LSTM style, keeps cuDNN. Variational
  loses cuDNN; Bernoulli applies per-step independently and is
  weaker. PyTorch's `nn.LSTM(dropout=p)` is inter-layer only, NOT
  recurrent; the kind selector dispatches to a library wrapper, not
  the upstream `dropout` argument.)
- `hidden_init_strategy: Literal["zero", "learned", "per_entity"]`
  (default `"zero"`)
- `readout: Literal["last_valid", "mean_pool", "attention"]` (default
  `"last_valid"`)
- `bptt_window: int | None` (default `None`)

The skeleton is INTERNAL-tier (not in the public API) in v1; promoting
to STABLE happens in v3 when the first concrete recurrent model ships.

## v1 concrete: TFT

This section is the concrete v1 implementation against the library-wide
infrastructure above. TFT is the first model to ship.

### TFT architecture

`TFTBackbone` extends `TransformerSequenceEstimator` and implements the
TFT architecture from [Lim et al., 2021](https://arxiv.org/abs/1912.09363),
adapted for classification and standard regression rather than
multi-horizon forecasting. The original paper has an encoder-decoder
shape consuming past inputs and known-future covariates and producing
quantile forecasts at each future horizon. v1 has no future window.
The block-by-block disposition below is the v1 contract.

**Blocks kept from the original TFT.**

- Static covariate encoders (one per static categorical / real input).
- Variable selection networks over the past window: one VSN for static
  inputs, one VSN for past time-varying inputs.
- Gated residual networks (GRN) and gating layers throughout the
  encoder.
- LSTM-based local processing of the past window (the locality-bias
  layer between the VSN and the self-attention).
- Interpretable multi-head self-attention over the past window, with
  mask broadcast inherited from `TransformerSequenceEstimator`.
- Skip connections, layer normalization, dropout policy.

**Blocks dropped from the original TFT.**

- Decoder-side temporal self-attention over future timesteps. v1 has
  no future window, so there is no decoder.
- VSN over known-future covariates. v1 ingests only past time-varying
  inputs.
- Quantile output projection per future horizon. Replaced by the
  task-specific head described below.
- The decoder-side cross-attention from future queries onto past keys.
  Folded into a single self-attention block over the past.

**Static-context vector consumption (verified against pytorch-forecasting
and Lim et al.'s reference impl).** The static GRN stack produces four
context vectors `(c_s, c_e, c_c, c_h)`. All four are encoder-path
consumers in v1:

- `c_s` conditions the past-VSN selection (gating input to the VSN).
- `c_c` initializes the LSTM **cell** state.
- `c_h` initializes the LSTM **hidden** state.
- `c_e` enriches the post-LSTM add-norm output via a GRN whose output
  is the INPUT to the interpretable self-attention block.

`nn.LSTM`'s signature is `(input, (h_0, c_0))`. The library calls
`lstm(input, (c_h, c_c))` (NOT `(c_c, c_h)`); writing the tuple in
positional reverse would feed the cell vector as hidden state and
vice versa.

The original paper feeds the static context vectors into both encoder
AND decoder pathways; v1 has no decoder, so the "decoder-bound"
consumers do not exist. There are no unused context vectors in v1.

**Net topology.** Encoder-only over the past window. One representation
per past timestep. The task head reads from a single readout vector
selected per the `prediction_readout` policy below.

**Interpretable multi-head attention is shared-V.** The original TFT
uses a SINGLE value projection shared across attention heads (with
per-head queries and keys), which produces the interpretable attention
weights. `nn.MultiheadAttention` does NOT support shared-V; the
library hand-rolls the attention block, broadcasting the shared
value tensor across heads and routing the per-head Q/K through
`F.scaled_dot_product_attention` (math backend for ONNX export per
N5).

**Variable-length sequence handling.** Three layers cooperate:

1. Variable selection networks zero out padded positions BEFORE
   computing their softmax selection weights.
2. The LSTM consumes the past window via
   `nn.utils.rnn.pack_padded_sequence(enforce_sorted=False)` /
   `pad_packed_sequence` so the cell never sees padded timesteps.
3. The interpretable self-attention block applies the
   `padding_mask` to both keys and queries; padded positions receive
   `-inf` pre-softmax scores via the `attn_mask` boolean (after the
   polarity flip described in F3).

A one-month-tenure entity contributes exactly one valid attention
key; a sixty-month-tenure entity contributes sixty. The same code
path handles both.

**Prediction readout.** Two options, configurable:

- `last_valid` (default): representation at the last un-masked timestep
- `mean_pool`: masked mean across valid timesteps

### TFT classification head

`TFTClassifier` instantiates `TFTBackbone` plus a classification head:

- `Linear(d_model, out_dim)` projection where
  `out_dim = 1` for `task_type=binary` and `out_dim = num_classes` for
  `task_type=multiclass`. v1.1 multi-label uses `out_dim = num_labels`.
- The head emits **raw logits**. `BCEWithLogitsLoss` and
  `CrossEntropyLoss` (F5) apply sigmoid / log-softmax internally
  during training. `predict_proba` applies sigmoid (binary) or softmax
  (multiclass) on the cached logits. `predict_with_attention` returns
  logits, not probabilities, in its prediction field; callers can
  apply the activation themselves if they want both logits and the
  attention surface.

The head parameter `out_dim` is distinct from sklearn's `n_outputs_`
attribute (F1.1): `n_outputs_` follows sklearn convention (1 for
binary and multiclass in v1; equals the label count for multi-label
in v1.1), while `out_dim` is the projection's tensor dimension which
matches `num_classes` for multiclass.

### TFT regression head

`TFTRegressor` instantiates `TFTBackbone` plus a regression head:

- `Linear(d_model, out_dim * n_quantiles)` projection where
  `out_dim = 1` in v1 and `n_quantiles` is `1` for point regression or
  `len(quantiles)` for quantile regression.
- v1.1 multi-output: same projection, `out_dim > 1`.
- The head emits raw scalars (no activation). `predict` returns
  `(N,)` for point regression and for `predict()` on a quantile-mode
  regressor (the median or `quantiles[len//2]`); `predict_quantiles`
  returns `(N, len(quantiles))`. This keeps `n_outputs_=1` aligned
  with the sklearn `check_estimator` regressor contract, with
  multi-output behavior gated on `predict_quantiles` as a separate
  entry point. `n_outputs_=1` holds for both point and quantile
  regression in v1; quantile dimensionality is exposed via the
  separate `quantiles_` attribute and the `predict_quantiles` entry
  point, NOT via `n_outputs_`.

### TFT hyperparameters

In `TFTConfig` (pydantic):

- `hidden_size` (default 128)
- `attention_heads` (default 4; must divide `hidden_size`)
- `dropout` (default 0.1)
- `variable_selection_dropout` (default 0.1)
- `prediction_readout` (`"last_valid"` | `"mean_pool"`)

Plus training-side fields shared with all models: `learning_rate`,
`weight_decay`, `batch_size`, `max_epochs`, `optimizer`, `scheduler`,
`warmup_steps`, `gradient_clip_val`, `accumulate_grad_batches`,
`precision`, `task_type`, `loss_strategy`, `imbalance_strategy`,
`calibration_strategy`, `verbose`. The optimizer default is `adamw`;
see F5 for the supported menu.

`TFTConfig` extends `BaseModelConfig`. `model_config = ConfigDict(
extra="forbid", frozen=True)`. Cross-field validators enforce:
`prediction_step >= 0`, `lookback >= 1`, `0 <= dropout < 1`,
`attention_heads divides hidden_size`, `quantiles strictly increasing
in (0, 1)`, and the loss / imbalance / calibration validity matrix
defined in F5. Unknown fields raise `ConfigError`. Mutability after
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

**Concrete acceptance thresholds** (v1, on the F6 DGP at
`dgp_version=1`, `signal_strength=0.7`, seed triple `(42, 137, 9999)`,
three-seed median, full lookback of 12):

- Binary classifier: accuracy >= 0.75 on n=2000 windows.
- Multi-class classifier: macro-F1 >= 0.60 on 4-class n=2000 windows.
- Point regressor: R² >= 0.5 on n=2000 windows.
- Quantile regressor: empirical coverage on nominal 80% interval falls
  in [0.75, 0.85] after conformal calibration.
- Temperature-scaled binary classifier ECE <= 0.05 on the calibration
  fold of n=2000.
- Platt-scaled binary classifier ECE <= 0.07 on the same fold.
- Isotonic-scaled binary classifier ECE <= 0.07 on the same fold.
- Conformal-calibrated regressor: empirical coverage in [0.75, 0.85].
- Isotonic-quantile regressor: empirical coverage in [0.72, 0.88]
  (wider band; non-parametric is more variance-prone on n=2000).

Acceptance thresholds are pinned to `dgp_version`. Bumping
`dgp_version` requires re-running these tests and updating the
thresholds in the same PR.

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

**Required tests (highest-impact correctness).** Each item below is
mandatory in v1; missing any blocks release.

- **Mask correctness.** For every variable-length-aware layer (VSN,
  masked attention, mean pool), the test puts the model into
  `model.eval()` with a fixed seed and `torch.use_deterministic_algorithms(True)`,
  forwards a single entity unpadded, then forwards the same entity
  padded to `lookback`, and asserts byte-equality of the model output
  at the valid time positions. The layer's tensor convention is
  `(batch, time, features)` per the transformer-family base; the
  valid-position slice is `out_padded[:, :valid_len, :]` and the
  comparison is `torch.equal(out_unpadded, out_padded[:, :valid_len, :])`.
  `torch.equal` is intentional, not `allclose`: BLAS accumulation-order
  drift between different input sizes is a real source of mask bugs
  and must NOT be tolerated. If the test breaks on a PyTorch upgrade
  because of this, the upgrade is what gets investigated, not the
  test.
- **Determinism.** Two `fit + predict` runs in the same process
  produce byte-identical output via `torch.equal`. A subprocess re-run
  produces the same hash. The fixture calls
  `torch.use_deterministic_algorithms(True, warn_only=False)`,
  `torch.manual_seed(seed)`, `os.environ["CUBLAS_WORKSPACE_CONFIG"] =
  ":4096:8"` at import time. The library itself sets these flags
  inside `seq_sklearn.training._determinism.enable_strict_mode()` and
  the Trainer calls it when `precision="32-true"` and the seed argument is
  set. This makes the bit-identical contract a library guarantee, not
  a caller responsibility.
- **save/load round-trip.** Fit, save, load in a fresh subprocess,
  predict on the same X, assert `torch.equal` on predictions and
  attribute equality on the public config.
- **save/load version-mismatch warning.** Save a model, mutate
  `seq_sklearn_version` in the checkpoint metadata to a fake older
  value, reload, assert a single `UserWarning` whose message contains
  "version mismatch" and both versions.
- **ONNX parity.** Export via `torch.onnx.export(dynamo=True)`, load
  in onnxruntime, predict on a fixed batch including a masked
  variable-length entity, assert agreement within `atol=1e-4` with
  the PyTorch path. The target opset is 20 (matches F1). Attention
  is exported via the math backend of
  `F.scaled_dot_product_attention` selected through
  `torch.nn.attention.sdpa_kernel(SDPBackend.MATH)`; the flash and
  mem-efficient backends do not export to ONNX cleanly. The
  architecture phase enumerates the restricted PyTorch op surface
  the backbone is allowed to use; ops outside the surface are caught
  by a static-analysis check in the deploy job.
- **Unseen-category robustness.** Fit encoder on `{A, B}`, predict on
  a panel with `{A, C}`, assert `C` maps to the `<unk>` index without
  raising.
- **Short-entity warning + shape.** Predict on a one-row entity with
  `min_periods_predict=3`. The row's prediction is NaN AND has the
  declared output shape (e.g. `(num_classes,)` for multiclass, not a
  scalar); `caplog` contains exactly one aggregated warning per
  `predict()` call regardless of the number of below-floor entities.
- **`check_estimator` subset.** Parametrized over the named list of
  check IDs in `tests/conftest.py`; explicitly skipped checks carry an
  `_xfail_checks` annotation with a one-sentence rationale.
- **Snapshot regression.** One snapshot test per task type. Pin DGP
  version, generator config, and seed. Store the expected output
  tensor and a snapshot manifest (DGP version, seed, model config
  hash) in `tests/_snapshots/`. Assert byte-identical equality on the
  CPU FP32 deterministic path. **Snapshot refresh procedure:**
  `pytest tests/snapshot/ --snapshot-update` regenerates the
  artifacts. The pre-commit hook rejects a commit that touches both
  source code and `tests/_snapshots/` files unless the commit message
  contains the literal string `SNAPSHOT_REVIEWED:` followed by a
  one-line human-written justification. CI refuses to merge a PR
  whose snapshot files were modified by a non-human author (e.g. a
  bot).
- **GPU/CPU parity.** Nightly. CPU FP32 deterministic vs. CUDA FP32
  agree within `atol=1e-5`.
- **NaN-loss guard.** Two variants. Variant A: monkey-patch the loss
  module to return `torch.tensor(float('nan'))` for three consecutive
  steps; assert `TrainingError` raised and the log record's `extra`
  field contains the offending `batch_idx`. Variant B: inject `Inf`
  into model weights at step 0; assert the same `TrainingError` path
  fires from the natural NaN-propagation route.
- **Optuna pruning hook.** Run an Optuna study with
  `MedianPruner(n_startup_trials=0, n_warmup_steps=0, n_min_trials=1)`,
  two trials, 3 epochs each. The `n_min_trials=1` is mandatory: without
  it the median is undefined at trial 2 epoch 0 and the prune check
  defers, producing a flaky test. Force the first trial to report
  `0.9` at every epoch and the second trial to report `0.1` at epoch
  0. Assert the second trial is pruned at epoch 0 (study state
  contains exactly one pruned-trial record with that trial number).
  A separate variant injects a `ConfigError` mid-trial and asserts
  it converts to `TrialPruned`. A third variant raises a
  `TrainingError` inside the trial body and asserts the same
  conversion path through `optuna_trial_guard`.
- **Calibration coverage per strategy.** A test per
  `calibration_strategy` value asserts the strategy hits its
  acceptance band (see thresholds above). Includes `none` (asserts no
  calibration is applied; raw probabilities pass through).
- **Imbalance-strategy smoke.** A test per `imbalance_strategy` value
  asserts the strategy actually changes the loss or sampler behavior
  (e.g. `class_weighted` produces a different first-epoch loss than
  `none` on an imbalanced panel; sampler variants change the per-batch
  class ratio).
- **Hardware-detect mocked.** Parametrized unit test for
  `seq_sklearn.hardware.detect()` covering each tier (CPU, Pascal,
  Volta/Turing, Ampere/Ada, Hopper, Blackwell). Uses
  `unittest.mock.patch` on `torch.cuda.is_available` and
  `torch.cuda.get_device_capability`. Each parametrized case asserts
  both the returned `HardwareTier` and the precision that `auto`
  maps to.
- **Three-way split correctness.** Fit on a panel where each entity
  has at least 20 rows. Assert: train fold, val fold, calibration
  fold are disjoint by `(id, time)`; calibration fold is the last
  `cal_fraction` rows per entity; val fold is the preceding
  `val_fraction` rows; setting `calibration_strategy="none"` and
  `threshold_tuning=False` collapses the calibration fold into
  training.
- **predict_quantiles error paths.** Three assertions: (1) calling on
  a point-mode regressor raises `PredictionError` naming `predict`;
  (2) calling before `fit` raises `NotFittedError` (sklearn-compatible);
  (3) passing a quantile not in the fit-time vector raises
  `ValueError` listing the fit-time vector.
- **Val-split warning.** Fit on a panel with > 1 entity with
  `val_split_strategy="random"`; assert a single `UserWarning` is
  emitted whose message contains "panel" and "random".
- **Conformal non-monotone guard.** Construct a calibration set where
  the trained quantile regressor's predictions are non-monotone
  across quantiles; assert `TrainingError` raised with a message
  naming "non-monotone quantiles".
- **Fit-state attribute contract.** After `fit`, assert each F1.1
  attribute is present with the documented shape and dtype.
  Classifier covers `classes_`, `n_features_in_`, `feature_names_in_`,
  `n_outputs_`. Regressor covers `n_features_in_`, `feature_names_in_`,
  `n_outputs_`, and (quantile mode) `quantiles_`. Binary classifier
  with `threshold_tuning=True` additionally asserts
  `decision_threshold_` is a float. Mutation of any attribute on a
  fitted estimator raises (frozen).
- **Validity-matrix illegal-combo rejection.** Parametrize over every
  illegal `(task_type, loss_strategy, imbalance_strategy,
  calibration_strategy)` cell enumerated in F5. Each parametrized
  case constructs the config and asserts `ConfigError` with a message
  that names the offending field combination and the legal
  alternatives.
- **Structured-log event emission.** Parametrize over each event in
  the F11 table. For each event, exercise the code path that should
  emit it and assert (via `caplog`) that exactly one record appears
  with `record.event == "<name>"` and `record.payload` keys matching
  the documented schema. Includes the WARNING-level events
  (`train.nan_step_skipped`, `train.mixed_precision_diverged`,
  `calibration.small_set`, `data.duplicate_floor_breach_count`) and
  the INFO/DEBUG-level events. Events tied to model families not
  represented in v1 (currently `train.hidden_norm`, recurrent-only)
  carry `pytest.mark.xfail(reason="no v1 emission path; recurrent
  family ships in v3", strict=True)` rather than being silently
  skipped, so the gap is explicit in CI output.
- **Calibration-set + cal_fraction conflict.** Construct a config with
  `calibration_set=(X_cal, y_cal)` AND `cal_fraction=0.1`; assert
  `ConfigError` at fit time naming both fields.
- **`enable_strict_mode` side-effect assertion.** Two scenarios.
  Scenario A: `CUBLAS_WORKSPACE_CONFIG` is unset in `os.environ` at
  call time. Call
  `seq_sklearn.training._determinism.enable_strict_mode()`; assert
  the four side effects fire
  (`torch.use_deterministic_algorithms(True, warn_only=False)`,
  `torch.backends.cudnn.deterministic=True`,
  `torch.backends.cudnn.benchmark=False`,
  `os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"`); call again
  and assert idempotency (no state change, no warning emitted).
  Scenario B: `CUBLAS_WORKSPACE_CONFIG` is pre-set to a non-default
  value (e.g. `":16:8"`). Call `enable_strict_mode()`; assert the
  three torch flags fire and the env var is left untouched at
  `":16:8"`.
- **DGP version bump regression.** Generate panel A with
  `seed=42, dgp_version=1`; without changing seed, generate panel B
  with the next `dgp_version`; assert panels differ. Prevents
  silent version bumps that do not change DGP behavior.
- **Optuna metric routing.** Variant of the pruning test that asserts
  `trial.intermediate_values[epoch]` equals the validation metric
  logged for that epoch (via `caplog`), confirming the metric is
  routed to the trial, not just any constant.
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
dtype invariant.

Hypothesis profiles registered in `tests/conftest.py`:

- `inner_loop` (default for local + per-PR CI):
  `settings(deadline=2000, max_examples=50, suppress_health_check=[HealthCheck.too_slow])`
- `nightly` (registered, activated via `HYPOTHESIS_PROFILE=nightly`):
  `settings(deadline=None, max_examples=500)`

CI fails if a hypothesis test runs without an explicit profile loaded.

**No-network policy.** Tests do not require network access. The
deployment-test job has one allowlist for installing from TestPyPI.

### N2: CI and review automation

**GitHub Actions matrix.** `{ubuntu-latest, macos-latest, windows-latest}`
x `{Python 3.12, 3.13, 3.14}`. Per-PR runs Linux on every Python
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
  optional extras (`[onnx]`, `[mlflow]`, `[wandb]`, `[docs]`).
- **Runtime dependency pins** (v1, verified against the 2026 stack):

  | Package | Pin | Rationale |
  |---|---|---|
  | `torch` | `>=2.6,<3` | weights_only default flip landed in 2.6 |
  | `lightning` | `>=2.6.1,<2.7` | 2.6.2 / 2.6.3 yanked after PyPI supply-chain compromise on 2026-04-30; the `lightning` package name (not legacy `pytorch-lightning`) is canonical |
  | `pydantic` | `>=2.12,<3` | skip 2.11.3 joblib regression (pydantic issue #11746); 2.12 fixes pickle round-trip for `frozen=True` models |
  | `scikit-learn` | `>=1.6,<2` | `Tags` dataclass + `__sklearn_tags__` API |
  | `optuna` | `>=4.4,<5` | 4.0 removed several legacy callbacks and the `MOTPESampler` |
  | `optuna-integration` | `>=4.4,<5` | installed for transitive dependency hygiene; `PyTorchLightningPruningCallback` is NOT used (the library ships a native pruning hook on `_LightningModule` to preserve Lightning 2.6 lifecycle events, see F7) |
  | `pandas` | `>=2.2` | dtype-extension stability |
  | `numpy` | `>=1.26` | numpy 2.x compatibility cutoff |
  | `safetensors` | `>=0.5` | save/load format (F4) |

  Optional extras pinned similarly: `onnx>=1.18`, `onnxruntime>=1.21`
  for `[onnx]`; `mkdocs>=1.6,<2`, `mkdocs-material>=9.7,<10`,
  `mkdocstrings[python]>=0.27`, `griffe-pydantic>=1.3` for `[docs]`.

- **Upper-bound policy.** Lower bounds by default; an upper bound is
  added preemptively ONLY when one of (a) a documented breaking
  rewrite is announced upstream (`mkdocs<2` after the hostile 2.0
  rewrite), (b) a security incident produces yanked versions
  (`lightning<2.7` after the 2.6.2 / 2.6.3 PyPI compromise), or (c) a
  known severity-1 regression exists in a specific minor (`pydantic`
  skip of 2.11.3). All preemptive upper bounds carry a one-line
  rationale comment in `pyproject.toml`.
- **Python version policy.** The library supports the three most-recent
  Python releases at each release cut. When a fourth release ships,
  the oldest drops in the next MINOR version with one deprecation
  cycle. Initial v1 supports 3.12, 3.13, 3.14 (Python 3.14 is the
  current release as of project start; 3.11 was the prior support
  floor in pre-v1 drafts and is dropped before v1 ships because the
  N3 "three most-recent releases" rule covers 3.12-3.14).
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

**The library, not the caller, configures determinism.** When
`precision="32-true"` and `seed` is set, the Trainer calls
`seq_sklearn.training._determinism.enable_strict_mode()`, which sets
`torch.use_deterministic_algorithms(True, warn_only=False)`,
`torch.backends.cudnn.deterministic = True`,
`torch.backends.cudnn.benchmark = False`, and exports
`CUBLAS_WORKSPACE_CONFIG=":4096:8"` if unset. The function is
idempotent. Setting `precision` to anything other than `"32-true"`
explicitly disables strict mode and emits an INFO log noting the
implication.

CUDA non-determinism caveat: mixed-precision modes can produce minor
numerical drift run-to-run because TensorCore reduction order is not
guaranteed. The `seed` docstring states this. Callers who want
bit-identical reproducibility on GPU keep `precision="32-true"`.

**RNG state on resume.** The F5 `resume_path` contract restores
"model weights, optimizer state, scheduler state, RNG state." Lightning's
`ModelCheckpoint(save_weights_only=False)` does NOT reliably round-trip
RNG state (known gap: Lightning issue #20204 on `load_from_checkpoint`).
The library ships a custom `RngStateCallback` that captures Python /
numpy / torch / `torch.cuda.get_rng_state_all()` into the checkpoint's
`extra` slot and restores them in `on_load_checkpoint`. The N1
determinism test asserts the restore is bit-exact.

**ATen ops without deterministic implementations** (2026 status). The
TFT v1 path uses none of the currently non-deterministic ops, but the
following remain non-deterministic under `use_deterministic_algorithms(True)`
and are forbidden in v1 backbones: `scatter_reduce(prod)`,
`EmbeddingBag(mode='max')` backward, several adaptive-pool /
interpolate backwards. v3 recurrent models that need them must declare
`tags.non_deterministic = True` in `__sklearn_tags__` (F1.1).

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
precision: Literal["bf16-mixed", "16-mixed", "32-true", "auto"] = "auto"
```

| Detected hardware | `auto` picks | Notes |
|---|---|---|
| Ampere / Ada / Hopper / Blackwell (CC>=8.0) | `bf16-mixed` | no `GradScaler` needed (bf16 has fp32's exponent range) |
| Volta / Turing (CC 7.x) | `32-true` | NOT `16-mixed`: TFT's quantile loss + softmax has documented fp16 NaN history; the 16-mixed gain is not worth the divergence risk on Turing-class hardware |
| Pascal (CC 6.x) | `32-true` | |
| CPU | `32-true` | |

The `"32-true"` literal is the Lightning 2.6+ API (the legacy
`"32"` form from Lightning 1.x was removed; passing it to Lightning
2.6's `Trainer(precision=...)` raises a validation error).

`16-mixed` remains a legal user-selectable value for callers who
explicitly opt in; `auto` never picks it.

**SDPA backend control.** Multi-head attention in v1 routes through
`F.scaled_dot_product_attention`. The backend is selected via
`torch.nn.attention.sdpa_kernel(SDPBackend.MATH)` (the modern 2026
API; the legacy `torch.backends.cuda.sdp_kernel` context manager is
deprecated). Selection is local to the ONNX-export path only; normal
training and inference let PyTorch pick the fused backend.

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

**Mixed-precision overflow.** `bf16-mixed` does not use `GradScaler`
on CC>=8.0. `16-mixed` uses Lightning's `GradScaler`; the library
ships a custom `GradScalerWatchdog` callback that watches
`trainer.precision_plugin.scaler.get_scale()` for consecutive
decreases (Lightning exposes no direct skip-count API). Three
consecutive scale decreases (the divergence signal) abort the run
with `TrainingError` and emit `train.mixed_precision_diverged` per
F11.

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
   attribute access only (no tuple unpacking; the field set is BETA
   and can grow in MINOR releases). See F1.
8. **Hyperparameter search.** RESOLVED: F7 ships Optuna integration.
9. **PyPI publishable name.** RESOLVED: `seq-sklearn` (verified
   available on PyPI). Earlier package names in this project's
   history were `tft-cls` and `tft-sklearn`; both were renamed before
   any code shipped, while scope was still being settled.
10. **Class imbalance.** RESOLVED: multiple strategies in F5, all
    Optuna-tunable.
11. **Calibration.** RESOLVED: temperature default with platt /
    isotonic / none alternatives in F5.

12. **Documentation tool choice.** OPEN. mkdocs + mkdocstrings vs.
    sphinx. Decide in the architecture phase. Both are viable; the
    deciding factor will be how heavily the API reference relies on
    autodoc vs. authored prose.

13. **Recurrent family base implementation timing.** RESOLVED:
    skeleton ships in v1 as an INTERNAL-tier abstract base class
    (`RecurrentSequenceEstimator`) with abstract methods only and no
    concrete LSTM/GRU code. Promoted to STABLE in v3 when the first
    concrete recurrent model ships. The skeleton validates that
    `BaseSequenceEstimator`'s contract supports the recurrent surface
    area without forcing a forward port in v3. See "Per-family
    architectural patterns" / Recurrent family.

14. **Foundation-model adapter strategy.** RESOLVED: foundation family
    removed from scope. Of Chronos, MOMENT, and TimesFM, only MOMENT
    has documented classification support; Chronos and TimesFM are
    forecasting-only. A one-model family does not justify the
    adapter-head abstraction work. Revisit if a second
    classifier-capable foundation model surfaces. See Scope and
    Roadmap.

15. **Performance-benchmark baseline source.** OPEN. The performance
    regression tests in N1 / `tests/perf/` require checked-in baselines
    per `(hardware, torch-version)` cell. Open question: what hardware
    cells are tracked in v1? Suggested: `(cpu, torch-latest)` and
    `(t4, torch-latest)` as the public baselines, with optional
    self-hosted cells added by contributors.

16. **Documentation toolchain.** RESOLVED: mkdocs + mkdocs-material +
    mkdocstrings (python handler) + `griffe-pydantic` for pydantic v2
    field-table rendering. Pin `mkdocs<2` (the 2.0 release was a
    breaking rewrite). Pin `mkdocs-material<10`. Source:
    `docs/research/mkdocstrings.md`. Was Q12.

17. **TFT block flow verification.** RESOLVED. The LSTM init order,
    static-context vector consumption, interpretable-attention
    shared-V structure, and mask handling are all sourced from
    pytorch-forecasting / Google reference / PlaytikaOSS impls. See
    `docs/research/tft.md` and the "v1 concrete: TFT" section.

18. **Stack pinning rationale.** RESOLVED. Concrete pins for torch,
    lightning, pydantic, sklearn, optuna, optuna-integration,
    safetensors, and the `[docs]` extra now live in N3 with one-line
    rationale each. The upper-bound policy is amended to allow
    preemptive caps when one of (a) a documented breaking rewrite is
    announced upstream, (b) a security incident produces yanked
    versions, or (c) a known severity-1 regression exists. Source:
    the 12 stack-specific docs at `docs/research/*.md`.

## Addressed

Round 1 (design-review swarm):

- **N7-section-count typo (arch C1).** "N1 through N8" -> "N1 through
  N7" in architectural philosophy. Line 168.
- **TFT decoder disposition (arch C2).** Added explicit
  blocks-kept / blocks-dropped / repurposed-blocks subsections to "v1
  concrete: TFT / TFT architecture". Encoder-only topology stated.
- **predict_quantiles fit-vs-predict contract (arch C3).** F1
  specifies fit-time vector is exact, predict-time subset selection is
  allowed, off-vector values raise ValueError, no interpolation in v1.
- **Loss x imbalance x calibration validity matrix (arch C4).** New
  F5 validity matrix lists the legal cells; illegal combos raise
  ConfigError at construction.
- **Optimizer field (arch C5).** `optimizer` added to BaseModelConfig
  with adamw / adam / sgd menu in F5. Listed in TFTConfig fields.
- **Calibration-on-validation-fold leak (arch C6).** F2 specifies a
  three-way split (train / val-for-early-stop / calibration) with
  default fractions and a callers-pass-their-own-set escape hatch.
- **sklearn fit-state attributes and tags (arch I1).** New F1.1
  subsection enumerates classes_, n_features_in_, feature_names_in_,
  n_outputs_, quantiles_, decision_threshold_; declares
  __sklearn_tags__ values; punts partial_fit / fit_predict with
  stable error messages.
- **Time-column dtype contract (arch I2).** F2 enumerates allowed
  dtypes (datetime64, PeriodDtype, signed int) and rejects object
  dtype.
- **Optuna ConfigError -> TrialPruned (arch I3).** F7 specifies trial
  failure-mode behavior (ConfigError, TrainingError convert to
  TrialPruned; DataContractError and unexpected exceptions
  propagate).
- **NotFittedError sklearn compatibility (arch I4).** F8 adds a
  NotFittedError class subclassing both SeqSklearnError and
  sklearn.exceptions.NotFittedError.
- **AttentionOutput tuple-unpacking (arch I5).** F1 drops tuple
  unpacking from predict_with_attention outputs; attribute access
  only. Q7 updated to match.
- **warmup_steps + constant interaction (arch I10).** F5 specifies
  warmup_steps is rejected with ConfigError when scheduler="constant"
  and re-interpreted as a fraction for one_cycle.
- **Structured-log mechanism (arch I11).** F11 specifies the
  extra={"event": ...} record convention and enumerates the v1
  event-name table with payload schemas.
- **Recurrent skeleton ships in v1 (arch I8/I9, Q13).** Skeleton is
  INTERNAL-tier abstract base in v1; promoted in v3. Surface area
  declared.
- **Synthetic-data DGP and canonical seeds (qa C1).** F6 specifies
  the data-generating process, the (42, 137, 9999) seed triple, and
  ties acceptance thresholds to a dgp_version stamp.
- **Mask correctness eval+seed precondition (qa C2).** N1 mask test
  requires model.eval() + fixed seed +
  torch.use_deterministic_algorithms(True). The use of torch.equal
  vs. allclose is explained.
- **Snapshot refresh procedure (qa C3).** N1 specifies
  --snapshot-update flag, pre-commit gate requiring
  SNAPSHOT_REVIEWED: line, CI rejection of bot-authored snapshot
  edits.
- **NaN-loss injection mechanism (qa C4).** N1 splits into Variant A
  (monkey-patch loss) and Variant B (Inf weights) so both detection
  and propagation paths are covered.
- **Optuna pruning hook test (qa C5).** N1 specifies a study with
  MedianPruner(n_startup_trials=0, n_warmup_steps=0, n_min_trials=1),
  two trials, asserting the lower-reporting trial prunes at epoch 0.
  The n_min_trials=1 argument was added in the architecture-review
  pass; without it the median at trial 2 epoch 0 is undefined and
  the prune check defers.
- **Hardware-detect mocked test (qa C6).** N1 specifies parametrized
  tests with mock.patch on torch.cuda.get_device_capability covering
  all six tiers.
- **Library-internal determinism flag (qa I3).** N4 specifies the
  library calls enable_strict_mode() inside the Trainer rather than
  asking callers to. Removes the test-fragility risk.
- **1-row entity required in e2e (qa I4).** F6 specifies that at
  least one e2e scenario per target_kind MUST include a 1-row entity
  alongside full-lookback entities in the same fit call.
- **predict_quantiles error-path tests (qa I5).** N1 adds three
  assertions for point-mode, not-fitted, and off-vector cases.
- **Random val-split warning test (qa I6).** N1 adds a test for the
  panel + random-split UserWarning.
- **Hypothesis profile values (qa I7).** N1 pins inner_loop and
  nightly profiles with concrete deadline / max_examples values.
- **Calibration coverage per strategy (qa I1).** N1 expands the
  acceptance thresholds to cover platt, isotonic, and
  isotonic_quantile beyond temperature-only.
- **Cross-version save/load warning test (qa I2).** N1 specifies a
  fixture-based test that mutates the saved version field and
  asserts UserWarning.
- **Conformal non-monotone guard test (qa N-3).** N1 adds the test
  for the non-monotone-quantiles TrainingError path.
- **Three-way split correctness test (arch C6 cross).** N1 adds the
  disjoint-folds assertion test.
- **Style: "landscape" metaphor (style C1).** Rewritten to "a second
  classifier-capable foundation model enters the field" in the
  roadmap footnote.
- **v1-v4 roadmap residue (scope reduction follow-up).** Updated
  Non-goals item 7 to "v1-v3 roadmap" with tightened parameter
  range.
- **Imbalance-strategy and "verbose" smoke (testing breadth).** N1
  adds tests for each imbalance_strategy value and the verbose
  config knob is now an enumerated field on the shared training
  config.

Round 2 (design-review swarm):

- **TFT head emits logits, not activated probabilities (arch r2-C1).**
  Classification head emits raw logits; `predict_proba` applies the
  activation. Aligns with `BCEWithLogitsLoss` / `CrossEntropyLoss`
  in F5. Same section disambiguates head `out_dim` from sklearn's
  `n_outputs_`.
- **Regression imbalance ignore-vs-raise (arch r2-C2).** F5 prose
  aligned with the validity matrix: non-`none` `imbalance_strategy`
  on a regression task raises `ConfigError`.
- **DGP binary/softmax bug (arch r2-C3).** F6 step 9 split per
  task_type: binary uses sigmoid + Bernoulli, multiclass uses
  softmax + Categorical.
- **DGP phi(window) coefficients (qa r2-C1).** F6 step 6 specifies
  `Dirichlet(alpha=ones(lookback))` draw, zero-clip + renormalize,
  retry-and-fallback rule that guarantees 3+ non-zero entries.
- **AR(1) initial state (qa r2-C2).** F6 step 3 pins `x_0` to the
  stationary-distribution draw `N(0, 1 / sqrt(1 - 0.49))`. The DGP
  also pins the entire generator to a single
  `numpy.random.Generator(PCG64(seed))` with strict step ordering.
- **calibration_set + cal_fraction conflict (qa r2-C3).** F2
  specifies `ConfigError` at fit time when both are set; callers
  must explicitly set `cal_fraction=0.0` to opt in.
- **Fit-state attribute test added (qa r2-C4).** N1 required test
  asserts each F1.1 attribute is set with correct shape and dtype
  post-fit.
- **Validity-matrix illegal-combo test added (qa r2-C5).** N1 required
  test parametrizes over every illegal cell and asserts ConfigError.
- **Structured-log event-emission test added (qa r2-C6).** N1
  required test parametrizes over every F11 event and asserts
  emission + payload schema.
- **Validity matrix mechanically enumerated (arch r2-I6).** F5 matrix
  rewritten to one row per `(task_type, loss_strategy)` pair, no
  conditional "any except" wording. Optuna search-space sampler
  consumes the enumeration directly.
- **`calibration_set` keyword in F1 fit signature (arch r2-I9).** F1
  `fit` signature updated to `fit(X, y, *, calibration_set=None)`.
- **`predict_with_attention` BETA marker at F1 (arch r2-I8).** F1
  bullet now annotates BETA at the call site so the stability tier is
  visible without scrolling.
- **`train.precision_fallback` event collision with N5 abort
  (arch r2-I10).** Renamed to `train.mixed_precision_diverged` at
  ERROR level, payload reflects the abort path.
- **`enable_strict_mode` side-effect test added (qa r2-I3).** N1
  required test asserts all four flags fire and idempotency.
- **DGP version bump regression test added (qa r2-I5).** N1 required
  test asserts version bump changes output.
- **Optuna metric-routing test added (qa r2-I6).** N1 variant asserts
  the metric value flows to `trial.intermediate_values[epoch]`, not
  just that pruning fires.
- **F2 back-to-back `The library` opener (style r2-I1).** Two
  consecutive paragraphs at lines 377/381 merged; openers varied.
- **Deferred-section "F1.1 opset 17" typo (arch r2-NITPICK 13).**
  Corrected to "N1's pinning to opset 17".

Round 3 (design-review swarm):

- **`train.hidden_norm` event in v1 test parametrization (qa r3-C1).**
  F11 row annotated "v3 only, no v1 emission path"; N1 event-emission
  test xfails recurrent-only events with `strict=True` so the gap
  shows up in CI output.
- **Canonical field-domain enumeration (qa r3-C2).** F5 now declares
  `TASK_TYPES` / `LOSS_STRATEGIES` / `IMBALANCE_STRATEGIES` /
  `CALIBRATION_STRATEGIES` as the single source of truth, sourced
  from `seq_sklearn.config._domains`. The illegal-combo test
  derives parametrization from the Cartesian product minus legal
  cells.
- **F5 prose `loss_strategy` bullet task-gating (arch r3-I1).** Each
  loss value now lists the task types it is legal for inline,
  mirroring the matrix.
- **Regression `n_outputs_=1` for both point and quantile (arch r3-I2).**
  Explicit sentence added at the regression head section: quantile
  dimensionality is exposed via `quantiles_` / `predict_quantiles`,
  not via `n_outputs_`.
- **`enable_strict_mode` env-var both-branch coverage (arch r3-I3).**
  N1 test split into Scenario A (CUBLAS_WORKSPACE_CONFIG unset) and
  Scenario B (pre-set to non-default).
- **`predict_with_attention` raises NotFittedError at F1 (arch r3-I4).**
  F1 bullet now says this locally instead of requiring a jump to F8.
- **DGP fallback tiebreak when no entries are zeroed (arch r3-I5 /
  qa r3-I2).** Step 6 specifies the fallback when the post-clip
  vector has fewer than 3 non-zero entries: fall back to the 3
  smallest pre-clip values with lowest-index tiebreak. Deterministic
  for every seed.
- **DGP step 4 RNG ambiguity (qa r3-I1).** Step 4 now names
  `rng.integers(0, K_i)` for the initial state and `rng.choice`
  for transitions; both use the single threaded `Generator`.
- **Mask test time-axis convention (arch r3-NITPICK 1).** N1 mask
  test names the `(batch, time, features)` convention explicitly and
  uses `out_padded[:, :valid_len, :]` slicing.

## Deferred

Round 1 (design-review swarm):

- **Architecture-reviewer NITPICK N2** (HardwareTier enum count
  cross-check at lines ~1009-1011): cosmetic, count is correct.
- **Architecture-reviewer NITPICK N4** (README quickstart sync
  contract): deferred to architecture phase, the doc build gate in
  N2 covers it indirectly.
- **QA NITPICK N-1** (ONNX op restriction enumeration): deferred to
  architecture phase. N1's pinning to opset 17 and the note about the
  math backend of scaled_dot_product_attention narrow the surface;
  the full op list is an implementation detail.
- **Q12** (mkdocs vs. sphinx): OPEN, decided in architecture phase.
- **Q15** (perf-benchmark baseline hardware cells): OPEN, decided
  when CI infra lands.
- **Architecture-reviewer I6** (quantiles validator location): minor
  doc-organization nit; left to architecture phase for placement.

Round 2 (design-review swarm):

- **`auto` precision serialization in F4 metadata (arch r2-NITPICK 14).**
  Save/load reproducibility across hardware tiers: the metadata block
  records the post-resolution precision. Mentioned in passing in N5;
  architecture phase to specify exact field name on the F4 metadata
  block.
- **Snapshot CI-enforcement implementation (qa r2-I2).** The
  `SNAPSHOT_REVIEWED:` commit-message gate and bot-author rejection
  are described as advisory; the concrete CI step (which action, what
  failure mode) is left to the architecture phase / CI workflow
  definition.
- **NaN-loss Variant B isolation (qa r2-I4).** Function-scoped
  fixture requirement is stated in N1; the exact pytest fixture
  pattern is an implementation detail.
- **Hardware-detect auto-precision dispatch (qa r2-I7).** The test
  contract is unambiguous; the question of whether `auto` resolution
  lives in `hardware.detect()` or `Trainer.__init__` is a design
  question for the architecture phase.
- **Hypothesis `suppress_health_check` extras (qa r2-I1).** The
  inner-loop profile pins `HealthCheck.too_slow`; the nightly profile
  is left untouched. If health checks fire spuriously in nightly,
  the profile is widened then (deferring rather than guessing).
- **K1 calibration-leak diagnostic probe (qa r2-K1).** A diagnostic
  test measuring ECE-with vs. ECE-without leak is interesting but
  costly; deferred to architecture phase to scope.
- **K2 `calibration_set` + `val_fraction` recompute (qa r2-K2).**
  Resolved implicitly by C3's text ("val_fraction is unchanged by
  this path"); leaving as Deferred because no required test
  enumerates this assertion explicitly.

Round 3 (design-review swarm):

- **`feature_schema_fingerprint` as fit-state vs. save-artifact (qa
  r3-I3).** F1.1 lists it as an attribute set via `fit`; F4 also
  lists it in the save metadata. The fit-state attribute test does
  NOT cover it. Architecture phase decides whether it lives at both
  layers and updates F1.1 + N1 accordingly.
- **Quickstart CI test acceptance threshold (qa r3-I4).** Whether
  `tests/e2e/test_quickstart.py` should re-assert the binary
  accuracy threshold or just smoke-test for runtime success is a
  test-design call for the architecture phase. The doc currently
  leaves it as a smoke test by default.
- **`decision_threshold_` absence on multiclass (qa r3-NITPICK 1).**
  Implementation detail; the architecture phase decides whether the
  attribute is absent or `None`.
- **Three-seed median aggregation correctness test (qa r3-NITPICK 2).**
  A meta-test asserting the e2e harness uses median (not mean / min)
  is useful but low priority; deferred.
- **Platt vs. temperature ECE strict-inequality test (qa r3-NITPICK 3).**
  Per-strategy thresholds are pinned; comparing strategies to each
  other (Platt strictly worse than temperature?) is a question for a
  later release.

Research pass (13 parallel agents, each producing a citation-cited
brief under `docs/research/`):

- **`__sklearn_tags__` rewritten (F1.1).** Confirmed as an instance
  method (NOT classmethod). Switched to the sklearn 1.6 `Tags`
  dataclass API: `tags.input_tags.dataframe = True`,
  `tags.target_tags.required = True`, etc. Dropped the legacy
  `X_types = ["dataframe"]` list and the `_xfail_checks` dict on the
  estimator (both removed in 1.6). Skip declarations moved to
  `expected_failed_checks` on `parametrize_with_checks`. Specific
  xfail list enumerated with rationale. Source: `docs/research/sklearn.md`.
- **Save / load format switched from .pt to safetensors + JSON (F4).**
  PyTorch 2.6+ flipped `torch.load`'s `weights_only` default to True;
  the .pt archive design would have invited a pickle-based code
  execution path on load. Two-file directory format
  (`weights.safetensors` + `state.json`) eliminates pickle from the
  public artifact. Source: `docs/research/pytorch.md`.
- **TFT block-flow corrections in v1 concrete section.** LSTM init
  tuple corrected from `(c_c, c_h)` to `(c_h, c_c)` (matches
  `nn.LSTM(input, (h_0, c_0))` signature). Static-enrichment ordering
  fixed: `c_e` feeds a GRN whose output is the INPUT to self-attention,
  not added before the post-LSTM add-norm. Interpretable multi-head
  attention declared as shared-V (hand-rolled, not
  `nn.MultiheadAttention`). `pack_padded_sequence(enforce_sorted=False)`
  named explicitly for the LSTM mask path. Source:
  `docs/research/tft.md`.
- **Mask convention split documented in F3.** Internal canonical name
  `padding_mask` with `True=padding (ignore)`; flip polarity once at
  the SDPA boundary because
  `F.scaled_dot_product_attention(attn_mask=...)` uses
  `True=participate`. Source: `docs/research/pytorch.md`.
- **N5 precision policy revised.** `auto` no longer picks `16-mixed`
  on Volta/Turing because of TFT's quantile-loss + softmax NaN
  history; falls back to `32`. `bf16-mixed` on CC>=8.0 no longer
  uses `GradScaler`. SDPA backend control uses
  `torch.nn.attention.sdpa_kernel(SDPBackend.MATH)` (not the
  deprecated `torch.backends.cuda.sdp_kernel`). Source:
  `docs/research/pytorch.md`.
- **N4 RNG-state callback documented.** Lightning's
  `ModelCheckpoint(save_weights_only=False)` does NOT round-trip RNG
  state reliably (issue #20204). Library ships a custom
  `RngStateCallback`. Source: `docs/research/lightning.md`.
- **N4 deterministic-ops note added.** TFT v1 uses none of the
  currently non-deterministic ATen ops; flagged the v3 forbidden list
  (`scatter_reduce(prod)`, `EmbeddingBag(mode='max')` backward, etc.).
  Source: `docs/research/pytorch.md`.
- **F1 ONNX path updated.** `torch.onnx.export(dynamo=True)` at opset
  20 is the supported 2026 path. Math backend of SDPA forced via
  `sdpa_kernel(SDPBackend.MATH)` for clean export. Source:
  `docs/research/pytorch.md`.
- **F5 / F9 GradScaler watchdog specified.** Lightning exposes no
  direct skip-count API; the library watches
  `trainer.precision_plugin.scaler.get_scale()` for consecutive
  decreases. Source: `docs/research/lightning.md`.
- **F7 Optuna integration concretized.** `optuna_integration.PyTorchLightningPruningCallback`
  (separate package, not `optuna.integration`). `n_min_trials=1`
  required for the N1 prune-at-epoch-0 test to be deterministic.
  `FixedTrial` is the canonical test entry point for `suggest_params`.
  Trial-failure conversion via `optuna_trial_guard(trial)` context
  manager wrapping the WHOLE objective (default `catch=()` means
  non-`TrialPruned` exceptions terminate the study). Source:
  `docs/research/optuna.md`.
- **Scope TST claim corrected.** Removed Time-Series-Library reference
  for TST (it is NOT in TSL). Canonical implementation is
  `gzerveas/mvts_transformer`. Flagged the BatchNorm-default trap and
  the flatten-and-Linear classification-head friction with seq-sklearn's
  variable-history path. Same friction documented for PatchTST (HF's
  pool head used instead) and TimesNet (mask-aware FFT variant
  needed). ConvTran flagged as v2.x candidate. Source:
  `docs/research/tst.md`, `docs/research/patchtst.md`,
  `docs/research/timesnet.md`.
- **Recurrent-skeleton config field updated.**
  `recurrent_dropout_kind` Literal extended from
  `["variational", "bernoulli"]` to
  `["weight_drop", "variational", "bernoulli"]` with `weight_drop` as
  the new default (AWD-LSTM-style, keeps cuDNN). Note added that
  `nn.LSTM(dropout=p)` is inter-layer only, not recurrent. Source:
  `docs/research/lstm_gru.md`.
- **LSTM-FCN regression caveat added.** No published regression
  variant exists; v3 ships LSTM-FCN regression as unvalidated extension
  of the family base, with a warning emitted at fit time. Source:
  `docs/research/lstm_fcn.md`.
- **N3 dependency pins concretized.** Replaced
  "lower-bounds-only by default" with concrete pins (`torch>=2.6,<3`,
  `lightning>=2.6.1,<2.7` after PyPI supply-chain compromise,
  `pydantic>=2.12,<3`, `sklearn>=1.6,<2`, `optuna>=4.4,<5`,
  `safetensors>=0.5`, `mkdocs<2` after hostile 2.0 rewrite). Upper-bound
  policy amended to allow preemptive caps in three named cases.
  Sources: every doc under `docs/research/`.
- **Q12 (docs toolchain) RESOLVED.** mkdocs + mkdocs-material +
  mkdocstrings + griffe-pydantic. Critical pin `mkdocs<2`. Source:
  `docs/research/mkdocstrings.md`.

Gemini final pass (cross-family review on requirements doc):

- **`expected_failed_checks` contradiction (gemini r1-C1).** F1.1
  previously listed `check_pandas_column_name_consistency`,
  `check_n_features_in`, and `check_methods_subset_invariance` in
  the `expected_failed_checks` list while annotating them as
  passing; with `xfail_strict=True`, a passing check marked XFAIL
  fails the test suite (XPASS). Restructured into two separate
  constants (`EXPECTED_FAILED_CHECKS` and `EXPECTED_PASSING_CHECKS`)
  with only the first passed to `parametrize_with_checks`.
- **ONNX opset inconsistency (gemini r1-C2).** N1 said opset 17
  while F1 and the research pass said opset 20. Aligned N1 to opset
  20; also added the dynamo=True flag and the SDPA math-backend
  selection.
- **Precision literal mismatch with Lightning 2.x (gemini r1-C3).**
  N5 used `"32"`; Lightning 2.6+ requires `"32-true"` (the legacy
  `"32"` was removed). Updated the Literal in the precision-config
  table and every `precision="32"` reference across F5 / F9 / N4 /
  N5 to `precision="32-true"`. Added a one-line note about the
  Lightning 1.x to 2.x rename.
- **`os.cpu_count()` None handling (gemini r1-I1).** F5
  `num_workers=min(4, os.cpu_count())` would raise TypeError on
  systems where `os.cpu_count()` returns None. Added the `or 1`
  fallback.

Gemini final pass (second; on the implementation plan, but with cross-doc impact on requirements):

- **F1 fit signature (gemini-impl r1-C3).** Added `optuna_trial:
  optuna.Trial | None = None` to `fit(X, y, *, calibration_set=None,
  ...)`. The architecture's Optuna integration (A16) now routes the
  trial via this keyword instead of polluting the pydantic config
  kwargs, preserving `extra="forbid"`.
- **F7 pruning hook design (gemini-impl r1-C4).** Replaced the
  `PyTorchLightningPruningCallback` paragraph with the native
  `_LightningModule` deferred-raise pattern. The upstream callback
  raises from `on_validation_end`, before
  `on_train_epoch_end` fires, which would skip the `train.epoch`
  and entropy events on the pruned epoch. The native hook stashes
  the decision in `on_validation_epoch_end` and raises at the END
  of `on_train_epoch_end` so logging fires first.
- **F7 `suggest_params` test guidance (gemini-impl r1-I1).**
  Switched from a 1000-iteration `FixedTrial` sweep to
  `optuna.create_study().ask()` so each sampled trial actually
  exercises the search space. `FixedTrial` is retained for
  `optuna_trial_guard` tests only.
- **N3 dependency-table rationale for `optuna-integration`
  (gemini-impl r1-C4).** Updated the rationale to "installed for
  transitive dependency hygiene; `PyTorchLightningPruningCallback`
  is NOT used".
