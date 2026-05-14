# Requirements: tft-sklearn v1

## Context

This library exists to fill one specific slot in a churn-prediction ensemble
at a payments processing company. The ensemble already includes tabular
classifiers (XGBoost, tabular transformer variants). Customer-churn signal
is notoriously noisy, and the ensemble's value depends on combining models
whose errors are *not* correlated with each other. Looking at the same
data through a sequence-modeling lens, rather than the flattened tabular
view existing classifiers use, is expected to produce predictions less
correlated with the rest of the ensemble. That decorrelation is the
primary justification for adding the model.

A Temporal Fusion Transformer (Lim et al., 2021) is a good fit for that
slot because it was designed for multi-horizon time-series modeling with:

- Static categorical covariates (entity-level features like industry, country)
- Time-varying real covariates (per-period numerics like spend, logins)
- Variable selection networks that learn which inputs matter at each step
- Interpretable temporal attention surfaces
- Native handling of ragged histories via masking

TFT was originally designed for regression (quantile forecasting). This
library exposes both task heads on top of the shared backbone:

- **`TFTClassifier`** for binary and multi-class classification. The
  primary driver of v1 (churn classification; churn-reason
  classification; segment classification).
- **`TFTRegressor`** for point and quantile regression on continuous
  targets. Reverts to TFT's native task. Useful for predicting expected
  spend, expected support volume, customer lifetime value as a number,
  or any temporal-panel regression workload where the same data shape
  applies.

Both estimators share the variable-selection networks, attention layers,
mask handling, and Optuna integration. Only the output head, loss
function, and the sklearn mixin differ. Same import path
(`tft_sklearn`), same `TabularToSequence` preprocessing component, same
synthetic-data generators (with a `target_kind` knob), same review
loop.

## Data shape (the operating reality)

The caller's data is a panel: one row per entity per period. The concrete
shape at the calling company:

- Key: `(company_id, date)` where `date` is monthly
- Stationary descriptors: `industry`, `country`, `state`, others
- Time-varying real: `spend`, `logins`, `transactions`, `support_calls`,
  `support_emails`, similar metrics
- Pre-computed deltas: each time-varying metric also appears as
  delta-vs-1-month-ago, delta-vs-2-months-ago, delta-vs-3-months-ago,
  encoding short-window temporal structure into the row

The library treats all of these as additional time-varying real features.
The caller's existing feature engineering is preserved, not stripped.

## Goals

1. Provide a TFT classifier (`TFTClassifier`) and a TFT regressor
   (`TFTRegressor`) with a sklearn-compatible API so both drop into
   existing ensemble code without re-architecting the pipeline.
2. Provide a `TabularToSequence` component that converts the caller's
   tabular panel into the tensor format the TFT expects. Shared between
   classifier and regressor.
3. Support binary classification, multi-class classification, point
   regression, and quantile regression. The classifier emits class
   probabilities; the regressor emits either a point estimate or a set
   of quantile estimates.
4. Ship with synthetic-data generators covering classification (binary,
   multi-class) AND regression (point, quantile) targets at day, week,
   month, quarter, and year period grains. These generators back the
   test suite and double as examples.
5. Ship with the test layers a legitimate open-source project carries:
   unit tests per component, integration tests across pairs of
   components, full-pipeline tests, deployment / install-smoke tests,
   plus CI wiring.

## Non-goals

1. Replacing existing tabular models. The library is one model in an
   ensemble, not a one-for-all replacement.
2. Streaming or online updates. Training and inference are batch-mode.
3. Hosting a model server. The library is consumed as a Python package;
   deployment to a serving system is the caller's concern.
4. Auto-feature engineering. If the caller's panel does not include the
   deltas they want as features, that is the caller's preprocessing,
   not the library's.
5. Multi-horizon forecasting in the original TFT sense (predicting
   multiple future steps in one forward pass). The regressor predicts
   one target per window at a configurable `prediction_step`, matching
   the classifier's contract. Future versions may extend this.

## Functional requirements

### F1: sklearn-compatible API

Two primary classes share the contract: `TFTClassifier`
(`ClassifierMixin`) and `TFTRegressor` (`RegressorMixin`). Both expose:

- `fit(X, y)` where `X` is a pandas DataFrame in the panel shape above
  and `y` is array-like of targets aligned to the rows of `X` (class
  labels for the classifier, floats for the regressor).
- `predict(X)` returning class predictions (classifier) or point
  predictions (regressor).
- `score(X, y)` returning accuracy by default for the classifier and R²
  for the regressor, with a `scoring` argument for a callable scorer.
- `get_params(deep=True)` and `set_params(**params)` so each estimator
  composes into `sklearn.pipeline.Pipeline` and
  `sklearn.model_selection.cross_val_score`.
- `save(path)` and `load(path)` for PyTorch-native serialization
  (weights + pydantic config dict).
- `export_onnx(path)` for ONNX export. The caller's own deployment path
  does not currently require ONNX, but it is becoming a common request
  for portable inference, so v1 supports it as an optional output. The
  ONNX dependency lives in an optional extra
  (`pip install tft-sklearn[onnx]`) so users without the need do not
  pay the install cost.
- `predict_with_attention(X)` returning the task-specific output plus
  the TFT's variable selection weights and temporal attention, for
  downstream interpretation. Returns a frozen dataclass
  `AttentionOutput` (classifier) or `RegressionAttentionOutput`
  (regressor) that supports both attribute access and tuple unpacking.

Classifier-only:

- `predict_proba(X)` returning class probabilities.

Regressor-only:

- `predict_quantiles(X, quantiles=[0.1, 0.5, 0.9])` returning quantile
  estimates per row. Requires the model to have been fit with a
  quantile loss (see F5).

Both classes satisfy enough of the sklearn estimator contract to pass
`sklearn.utils.estimator_checks.check_estimator` for the subset
applicable to non-array inputs.

**`y` shape contract (v1).** Both estimators accept only 1D `y` in v1
(single-output classification or single-output regression). The single
validator function lives in `tft_sklearn._validate.check_y` and raises
a clear error if a 2D `y` is passed:

```
ValueError: tft-sklearn v1 supports single-output targets only.
Multi-output regression and multi-label classification are planned
for v1.1. Got y with shape (n_samples, n_outputs).
```

v1.1 changes this validator in one place; no other code touches `y`
shape.

### F2: Input data contract

The caller must declare, at construction time, which columns are which:

- `id_col`: identifies the entity across rows (e.g. `company_id`)
- `time_col`: a sortable column (datetime, period, or integer index)
- `static_categorical_cols`: entity-level categoricals, constant across
  the entity's rows
- `static_real_cols` (optional): entity-level numerics
- `time_varying_real_cols`: per-period numerics
- `time_varying_categorical_cols` (optional): per-period categoricals

At `fit` time the target column is passed via `y`. The library validates
that the declared columns exist in `X`, that `(id_col, time_col)` is
unique, and that the time column is sortable.

The library does NOT impose a fixed schema; the caller picks which
columns map to which role.

### F3: Tabular-to-tensor restructuring

A `TabularToSequence` transformer converts the validated panel into the
batched tensors a TFT expects.

For each entity it:

1. Sorts the entity's rows by `time_col`.
2. Builds a sliding window of length `lookback` over the sorted rows.
   Default `lookback=12`. Configurable. If an entity has more history
   than `lookback`, only the most recent `lookback` periods are used.
3. Aligns the target to a configurable `prediction_step` relative to
   the end of the window. Default `prediction_step=1` (predict the
   period immediately after the lookback). Any non-negative integer is
   accepted, capped at the available horizon in the caller's data. The
   calling team uses `prediction_step=2` so their CX team has time to
   intervene; the library does not bake that choice in.
4. Emits per-window tensors: `static_categorical` (one set per window),
   `static_real`, `time_varying_real`, `time_varying_categorical`, the
   target, and a per-timestep boolean mask marking positions that are
   real data versus padding.

**Variable history length.** Entities with fewer rows than `lookback`
are NOT skipped by default. They are left-padded to `lookback` with the
mask flagging padded positions. The TFT respects this mask in its
variable-selection networks and attention layers (see F4), so the model
only sees valid timesteps. A customer with one month of tenure and a
customer with five years of tenure use the same model, the same code
path, the same lookback length, and the same prediction-step setting;
the model just draws on less history when less is available.

Two configuration knobs control the strict edge:

- `min_periods` (default 1): entities with fewer than this many real
  rows are dropped at `fit` time. Set higher for callers who want to
  exclude very-short-tenure entities.
- `min_periods_predict` (default 1): same gate at `predict` time;
  entities below the floor get a `NaN` prediction with a logged
  warning, not an error, so a batch with mixed tenures still produces
  output for the rest.

The component is deterministic given a seed. Categorical columns are
encoded via a fitted dictionary; unseen categories at `predict` time
map to a learned `<unk>` slot.

### F4: TFT model adapted for classification

A `TFTBackbone` module implements the TFT architecture from
[Lim et al. 2021](https://arxiv.org/abs/1912.09363): variable-selection
networks, gated residual networks, static covariate encoders,
sequence-to-sequence attention, interpretable multi-head attention.

**Variable-length sequences are native, not an afterthought.** The
backbone accepts the per-timestep mask emitted by F3 alongside the
input tensors. Variable-selection networks zero out padded positions
before computing their softmax weights, so the network does not learn
spurious "padded means churn" signals. Attention layers apply the mask
to both keys and queries so attention scores at padded positions are
masked to `-inf` pre-softmax. A one-month-tenure entity contributes
exactly one valid attention key; a sixty-month-tenure entity
contributes sixty. The same code path handles both.

**Prediction readout.** The classification head reads from either:
- the representation at the last valid (un-masked) timestep, or
- a masked mean-pool across all valid timesteps.

Configurable, default `last_valid`. Both options ignore padded
positions by construction.

**Layer factory.** All `nn.Linear` and `nn.LayerNorm` instantiations in
`src/tft_sklearn/models/` route through a single factory in
`src/tft_sklearn/models/_layers.py`. v1 returns standard PyTorch layers;
v2 can swap in Transformer Engine equivalents (`te.Linear`, `te.LayerNorm`)
in one place to enable FP8. See N5 for the rationale and the broader
precision story.

**Output heads.** The TFT's quantile output layer is replaced by a
task-specific head. The backbone is the same module in all cases; only
the head and loss change.

Both heads take an `n_outputs: int` constructor parameter so v1.1 can
add multi-output regression and multi-label classification as small
additive changes (see Open Question 3 resolution). In v1, `n_outputs`
is set to `1` for regression and `num_classes` for single-label
classification.

*Classification head* (used by `TFTClassifier`):
- `Linear(d_model, n_outputs)` projection
- sigmoid activation for binary (`n_outputs=1`)
- softmax activation over `n_outputs` for multi-class
- v1.1 multi-label support: same projection with `n_outputs=num_labels`
  and sigmoid-per-output activation

The classifier produces calibrated probabilities. Calibration strategy
is a configuration knob (default and alternatives discussed in F5
alongside class-imbalance strategies, since the two interact).

*Regression head* (used by `TFTRegressor`):
- `Linear(d_model, n_outputs * n_quantiles)` projection where
  `n_outputs=1` in v1 and `n_quantiles` is `1` for point regression
  or `len(quantiles)` for quantile regression
- v1.1 multi-output regression: same projection with `n_outputs>1`,
  no other changes

The regressor produces well-calibrated quantile coverage on validation
data when fit in quantile mode. Coverage calibration strategy is a
configuration knob; see F5.

### F5: Training pipeline

A `TFTTrainer` wraps pytorch-lightning to handle:

- Optimizer and scheduler configuration via pydantic config
- Early stopping on a validation metric
- Mixed-precision training when CUDA is present
- Deterministic mode driven by a single seed threaded through every
  randomness boundary (data split, model init, dataloader shuffle,
  cudnn backend)
- Checkpoint saving and best-model selection

**Loss strategies for classification.** Churn is structurally
imbalanced; this is the single biggest source of training-time pain.
The classifier exposes multiple strategies, all configurable and all
valid points in the Optuna search space (see F7):

| Strategy | What it does |
|---|---|
| `class_weighted_ce` | Cross-entropy with per-class weights computed from training-set frequencies. Default. |
| `focal_loss` | Focal loss with configurable `gamma` (default 2.0). Down-weights easy examples. |
| `oversample_minority` | Sampler at the entity-window level. Configurable ratio. |
| `undersample_majority` | Sampler at the entity-window level. Configurable ratio. |
| `none` | No reweighting. Use when the caller wants raw behavior. |

Threshold tuning (post-hoc adjustment of the decision boundary on a
held-out set) is independent of the loss strategy and can be combined
with any of the above.

**Loss strategies for regression.** Configurable via the pydantic
config and Optuna-tunable:

| Strategy | What it does |
|---|---|
| `mse` | Mean squared error. Default for point regression. |
| `mae` | Mean absolute error. Robust to outliers; reports the conditional median. |
| `huber` | Huber loss with configurable `delta`. Hybrid of MSE / MAE. |
| `pinball` | Pinball / quantile loss over a configurable quantile list (e.g. `[0.1, 0.5, 0.9]`). Required for `predict_quantiles`. Default when the regressor is constructed with quantile output. |

All regression loss functions accept output tensors of shape
`(N, K)` where `K = n_outputs * n_quantiles`. In v1 this is
`(N, 1)` for point regression or `(N, len(quantiles))` for quantile
regression. v1.1 multi-output regression passes `(N, n_outputs)` or
`(N, n_outputs, len(quantiles))` with no loss-function changes
needed (PyTorch's reductions handle the extra dimension). Tests assert
this on synthetic two-output data in v1 even though the public API
exposes single-output only.

**Loss strategies for classification.** The classification head's loss
function is selected by `task_type`:

| `task_type` | Loss function |
|---|---|
| `binary` | `BCEWithLogitsLoss` (single-output sigmoid) |
| `multiclass` | `CrossEntropyLoss` (softmax) |
| `multilabel` (v1.1) | `BCEWithLogitsLoss` (multi-output sigmoid) |

The selection point is a single dispatch in the loss factory. v1.1's
multi-label addition swaps `task_type` and `n_outputs`; the loss factory
returns the right loss without further changes.

**Calibration strategies.** The two tasks calibrate differently.

*Classification calibration* (post-hoc, fit on a held-out validation
fold during `fit()`):

| Strategy | What it does |
|---|---|
| `temperature` | Temperature scaling (Guo et al., 2017): one learned scalar dividing logits. Default. |
| `platt` | Platt scaling: logistic regression mapping logits to probabilities. Binary only. |
| `isotonic` | Isotonic regression: non-parametric monotonic mapping. |
| `none` | No post-hoc calibration; raw probabilities. |

*Regression coverage calibration* (only meaningful for quantile mode):

| Strategy | What it does |
|---|---|
| `conformal` | Split-conformal prediction (Vovk et al.): adjusts quantile predictions so that the empirical coverage on a held-out fold matches the nominal level. Default for quantile regression. |
| `isotonic_quantile` | Isotonic regression on the CDF. Non-parametric, more flexible. |
| `none` | No post-hoc adjustment; raw quantile predictions. |

The Optuna search space includes the calibration choice alongside the
loss choice for the relevant task.

### F6: Synthetic data generators

The library ships generators that produce panels of `(id, time,
features, target)` at configurable period grains: day, week, month,
quarter, year. Each generator exposes:

- `target_kind`: `"binary"`, `"multiclass"`, `"regression_point"`, or
  `"regression_quantile"`. Selects the target distribution and target
  type. The same backbone test scaffolding covers all four; only the
  metric to assert on differs.
- `num_entities`
- `periods_per_entity`: either an `int` (fixed length) or a
  `(min, max)` tuple sampled per entity, or a callable for custom
  distributions. The ragged form is required for the test suite to
  exercise short-history handling (see F3).
- `num_static_categorical`, `num_static_real`,
  `num_time_varying_real`, `num_time_varying_categorical`
- Classification-only: `class_balance` (binary) or
  `class_distribution` (multi-class).
- Regression-only: `target_scale` (output magnitude) and
  `target_noise` (irreducible noise added on top of the signal).
- `noise_level`: feature-side noise, independent of target noise.
- `signal_strength`: how strongly the injected pattern drives the
  target, so tests can assert the model actually recovers the signal
  rather than memorizing noise.
- `seed`

The test suite must include at least one scenario per period grain
per `target_kind` where `periods_per_entity` ranges from 1 to a
moderate maximum (e.g. `(1, 60)` for monthly), exercising the
variable-history path end-to-end.

The generators double as documentation examples.

### F7: Hyperparameter tuning compatibility (Optuna as a first-class concern)

The library is designed for Optuna search out of the box, not patched
in later. All model and training hyperparameters live in a single
pydantic config class (`TFTConfig`) and are exposed as constructor
arguments on `TFTClassifier`. `get_params` and `set_params` (F1)
already cover the sklearn search ecosystem (`GridSearchCV`,
`RandomizedSearchCV`).

For Optuna specifically the library ships:

- `tft_sklearn.tuning.suggest_params(trial: optuna.Trial, base: TFTConfig | None = None) -> TFTConfig`
  with a reasonable default search space covering model size
  (`hidden_size`, `attention_heads`, `dropout`,
  `variable_selection_dropout`), training (`learning_rate`,
  `weight_decay`, `batch_size`, `max_epochs`, `warmup_steps`,
  `gradient_clip_val`), the class-imbalance strategy (F5), the
  calibration strategy (F5), and optionally `lookback` and
  `prediction_step` (F3) when the caller wants to tune them rather
  than fix them.
- A runnable example at `docs/examples/optuna_search.py` showing the
  full search loop end-to-end on synthetic data.
- A pruning hook that integrates with Optuna's
  `MedianPruner` / `HyperbandPruner` by reporting the validation
  metric after each epoch.

Callers can use the default search space directly, extend it via the
`base` argument, or replace it with their own `suggest_params`
function.

## Non-functional requirements

### N1: Testing

Four test categories, each in its own folder under `tests/`:

- **Unit tests** (`tests/unit/`): one component at a time. Every public
  function gets at least a happy path, one edge case, one error path.
  Coverage gate: 85% line coverage on `src/tft_sklearn/`.
- **Integration tests** (`tests/integration/`): paired components such
  as `TabularToSequence` → `TFTBackbone` and `TFTBackbone` → `TFTTrainer`.
- **Full-pipeline tests** (`tests/e2e/`): end-to-end against synthetic
  data, asserting the model recovers above-chance accuracy on datasets
  with a known injected signal.
- **Deployment tests** (`tests/deploy/`): the package builds, installs
  from the built wheel in a clean venv, imports cleanly, runs a minimal
  fit/predict round-trip. Runs on every PR via CI.

Slow tests (>2s) are marked `pytest.mark.slow` and skipped in the dev
inner loop; GPU tests are marked `pytest.mark.gpu` and skipped on
CPU-only CI runners.

### N2: CI and review automation

GitHub Actions workflow on every PR:

1. `ruff check .` and `ruff format --check .`
2. `pyright`
3. `pytest -m "not slow"` with the 85% coverage gate
4. `pytest tests/deploy/` to validate the install smoke test
5. Nightly job runs the full suite including slow tests

GitHub Copilot review is wired up on PRs for an automated first pass
before human review.

### N3: Repository hygiene (open-source standard)

- `pyproject.toml` with build metadata, dependencies, dev extras,
  console entry points if any
- License file (`LICENSE`)
- `CHANGELOG.md` following Keep a Changelog conventions
- `CONTRIBUTING.md` documenting the review workflow (see
  `/design-review`, `/review`, `/gemini-final-pass`)
- Pre-commit config wired to ruff, pyright

### N4: Reproducibility

A single `seed` argument on `TFTClassifier.__init__` threads through to
every randomness boundary. Two runs with the same seed and same input
produce bit-identical predictions on CPU (CUDA non-determinism is a
caveat noted in the docstring).

### N5: Hardware and precision

**CPU as first-class.** A 1-5M parameter TFT fits comfortably on modern
CPUs. CPU is a fully supported runtime for development, CI (GitHub
Actions free tier gives no GPU), the synthetic-data unit tests, and
small-scale production where GPU infrastructure is not available. All
tests except those marked `pytest.mark.gpu` run on CPU.

**GPU floor.** NVIDIA compute capability 6.0 (Pascal, 2016) and newer.
The library inherits PyTorch's own minimum via the pinned `torch>=2.x`
version; we do not impose tighter constraints than PyTorch itself does.
CUDA is auto-detected at runtime, with fallback to CPU if absent.

**Precision: configurable, auto-detect default.**

```python
precision: Literal["bf16-mixed", "16-mixed", "32", "auto"] = "auto"
```

The `"auto"` mode detects hardware and picks the best supported option:

| Detected hardware | `auto` picks | Reason |
|---|---|---|
| Ampere / Ada / Hopper / Blackwell | `bf16-mixed` | Native BF16 TensorCores |
| Volta / Turing | `16-mixed` | FP16 TensorCores, no BF16 hardware |
| Pascal | `32` | No low-precision TensorCore acceleration |
| CPU | `32` | CPU mixed-precision adds complexity for marginal gain at this model size |

Callers override via the constructor or pydantic config.

**Out of scope for v1: FP8 and FP4.** Transformer Engine and the
FP8/FP4 hardware paths on Hopper and Blackwell are deliberately
deferred. Rationale:

- The model is bandwidth-bound at 1-5M parameters, not compute-bound.
  BF16 captures most of the achievable speedup; FP8 over BF16 is
  roughly 1.2x on this scale.
- FP8 requires per-tensor scale tracking and TE-aware layers, not a
  configuration flag.
- CI cost: FP8 paths require self-hosted Hopper or Blackwell runners.
- Engineering budget for v1 is better spent on the core library.

**v1 architecture must keep the FP8 door open cheaply.** These are
hard v1 constraints, not stretch goals:

1. **Layer factory** (see F4). All `nn.Linear` and `nn.LayerNorm`
   instantiations route through one module. v2's swap to `te.Linear`
   and `te.LayerNorm` is a one-place change.
2. **No hand-rolled CUDA kernels.** v1 sticks to standard PyTorch
   primitives (`nn.MultiheadAttention` or the F.scaled_dot_product
   variant). Custom fused kernels would have to be re-implemented for
   FP8 and are not justified at this model scale.
3. **Precision config is forward-extensible.** The current `Literal`
   type adds `"fp8-mixed"` in v2 with no API change; the factory
   branches on the new value.
4. **Hardware-tier helper in v1.** A single
   `tft_sklearn.hardware.detect() -> HardwareTier` enum function lives
   in v1, with values `CPU`, `PASCAL`, `VOLTA_TURING`, `AMPERE_ADA`,
   `HOPPER`, `BLACKWELL`. v1 branches only on the first four tiers; v2
   reads the last two without touching the function signature.

**Reproducibility caveat.** Deterministic mode (see N4) runs in FP32
only. Mixed-precision modes can produce minor numerical drift
run-to-run on CUDA because TensorCore reduction order is not
guaranteed. The `seed` docstring notes this and tells callers to set
`precision="32"` plus `torch.use_deterministic_algorithms(True)` when
bit-identical reproducibility is required.

### N6: Documentation

- `README.md` with a quickstart that fits in one screen and trains a
  binary classifier on synthetic monthly data
- `docs/` folder with this requirements doc, the architecture doc, and
  longer-form guides as they accrue
- `docs/examples/` with runnable Python scripts (preferred over
  notebooks for CI testability)
- API reference auto-generated from docstrings; mkdocs or sphinx,
  decided in the architecture phase

## Acceptance criteria

v1 is ready to release when:

1. Every F1 through F6 requirement is implemented and tested.
2. Every N1 through N6 requirement is met.
3. The architecture doc has passed `/design-review` consensus and
   `/gemini-final-pass design` with no new CRITICAL findings.
4. The `style-reviewer` agent reports zero CRITICAL findings on
   committed files.
5. Two quickstart examples are present and pass in CI:
   - A binary classifier trains on synthetic monthly data and recovers
     above-chance accuracy reproducibly.
   - A quantile regressor trains on synthetic monthly data and recovers
     calibrated coverage on a held-out fold (nominal 80% interval
     covers 75-85% of test points after conformal calibration).
6. The deployment smoke test (`tests/deploy/`) passes in CI from a
   freshly built wheel.
7. A release candidate is built and installable via
   `pip install tft-sklearn` against a TestPyPI index.

## Open questions

Items resolved during requirements drafting are noted as RESOLVED with
the decision. Items still open are flagged OPEN and are inputs to the
design-review loop.

1. **Lookback default.** RESOLVED: default `lookback=12`, configurable.
   Short-history entities (fewer rows than `lookback`) are padded and
   masked rather than dropped, so 1-month-tenure and 5-year-tenure
   entities share the model. See F3.
2. **Prediction step default.** RESOLVED: default `prediction_step=1`,
   configurable to any non-negative integer the caller's data
   supports. The calling team uses `prediction_step=2` for CX
   intervention lead time; the library does not bake that in. See F3.
3. **Multi-output / multi-label.** RESOLVED: both deferred to v1.1.
   v1 ships single-output classifier (binary or multi-class) and
   single-output regressor (point or quantile). v1 must be architected
   so v1.1 adds both extensions as small additive changes, not as
   architectural surgery. The v1 constraints that keep these doors
   open are listed below and enforced in F1, F4, and F5.

   **v1 constraints for cheap v1.1 multi-output / multi-label:**

   1. **Head modules take `n_outputs` as a constructor parameter**
      (see F4). The classification head accepts `n_outputs` defaulting
      to `1` for binary and `num_classes` for multi-class; v1 sets it
      to those values, v1.1 adds a `multilabel` task type that sets
      `n_outputs = num_labels` and switches the activation from
      softmax to sigmoid. The regression head accepts `n_outputs`
      defaulting to `1`; v1 hardcodes `1`, v1.1 lets the caller pass a
      higher value.

   2. **`y` shape validation is one function** (see F1). v1's
      validator accepts only 1D `y` and raises a clear error
      mentioning v1.1 if the caller passes a 2D array. v1.1 changes
      the validator to accept 2D `y` of shape `(n_samples, n_outputs)`;
      no other code touches `y` shape.

   3. **Loss functions accept `(N, K)` output shapes natively** (see
      F5). The regression losses already do this in PyTorch; the v1
      tests assert it on synthetic two-output data even though the
      public API exposes single-output only. The classification loss
      function selection is parameterized so swapping
      `CrossEntropyLoss` for `BCEWithLogitsLoss` in multi-label mode
      is a one-line change in v1.1.

   4. **`task_type` config enum is forward-extensible.** The pydantic
      config defines `task_type: Literal["binary", "multiclass",
      "regression_point", "regression_quantile"]` in v1. v1.1 extends
      the literal with `"multilabel"` and `"regression_multioutput"`
      without breaking the API.
4. **Saved-model format.** RESOLVED: PyTorch-native `save`/`load` is
   the primary path; ONNX export is supported via `export_onnx(path)`
   behind an optional `pip install tft-sklearn[onnx]` extra. See F1.
5. **Categorical encoding.** RESOLVED: learned embeddings, sized by
   default with the fastai heuristic
   `min(50, round(1.6 * cardinality^0.56))`, with a per-column
   override surface for callers who want explicit control. The config
   accepts a dict mapping column name to embedding dimension; any
   column not in the dict falls back to the heuristic. This matches
   the general configurability requirement: sensible defaults that
   actually work, with every knob reachable from the config for
   callers who need to tune.
6. **Missing data and short tenure.** RESOLVED: variable-length
   histories are handled by the mask path in F3 and F4. The library
   does not impose imputation; the mask is the contract. NaN values
   inside time-varying features still require caller-side imputation
   (or use of an explicit "missing" categorical level), since masking
   represents "this period does not exist", not "this period exists
   but a value is missing".
7. **Attention output shape.** RESOLVED: frozen dataclass
   `AttentionOutput` returned by `predict_with_attention`. Fields:
   `probas`, `variable_selection_weights` (a sub-dataclass or dict
   covering static and past-temporal selection weights),
   `temporal_attention`. Rationale: the output has heterogeneous
   field types, is likely to grow more fields over time (layer-wise
   attention, gate activations, calibration confidence), and benefits
   from helper methods (e.g. `.feature_importance() -> pd.DataFrame`).
   A frozen dataclass gives attribute access, immutability, clean
   `repr`, and additive evolution; a named tuple's positional
   unpacking is convenient but turns into a breaking change every
   time a field is added.

   The dataclass implements `__iter__` (returning
   `dataclasses.astuple(self)`) so callers who prefer the
   sklearn-flavored tuple unpacking style get it without an adapter:

   ```python
   out = model.predict_with_attention(X)
   probas = out.probas                       # attribute access
   probas, vsw, ta = out                      # tuple unpacking
   df = out.feature_importance()              # helper method
   ```

   Docstring notes that attribute access is the stable contract; the
   iter ordering is ergonomic sugar and may change if fields are added.
8. **Hyperparameter search.** RESOLVED and elevated to F7: Optuna
   support is a first-class concern, ships with a reference search
   space and example.
9. **PyPI publishable name.** RESOLVED: `tft-sklearn` is available on
   PyPI as of requirements drafting (HTTP 404 on
   `https://pypi.org/pypi/tft-sklearn/json`). PEP 503 name normalization
   means that `tft-sklearn`, `tft_sklearn`, and `tft.cls` collapse to the
   same registered name, so reserving `tft-sklearn` covers the underscore
   and dot variants too.

   Future name-availability check (one-line):
   `curl -s -o /dev/null -w "%{http_code}\n" https://pypi.org/pypi/<name>/json`
   A `200` means taken, a `404` means available. PyPI search is
   permanently disabled so the JSON-API check is the canonical method.
10. **Class-imbalance handling.** RESOLVED: multiple strategies
    available, all in the Optuna search space. Default
    `class_weighted_ce`. See F5.
11. **Calibration.** RESOLVED: default `temperature` scaling on a
    held-out fold; `platt`, `isotonic`, and `none` available as
    configuration options and Optuna search points. See F5 for the
    full list and the user-facing rationale.

## Addressed

(populated by the design-review loop)

## Deferred

(populated by the design-review loop)
