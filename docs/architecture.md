# Architecture: seq-sklearn v1

## Scope

This document maps the v1 requirements (`docs/requirements.md`) to
concrete implementation contracts: package layout, class hierarchy,
public-API signatures, configuration schemas, data flow, error
handling, hardware / precision dispatch, observability plumbing, test
fixtures, CI workflow, and documentation toolchain.

Where a requirements section is referenced (e.g. "F3", "N5"), the
contract there is authoritative. This document specifies HOW the
contract is satisfied, not WHAT the contract is. Conflicts between the
two docs are bugs; the requirements doc wins by default.

Every load-bearing decision is grounded in one of the briefs under
`docs/research/` (produced by 13 parallel research agents over the
2026 stack). Citations to those briefs are inline. The architecture
phase also resolves the items left **Deferred to the architecture
phase** in the requirements doc:

- Q15 (performance-baseline hardware cells): resolved in A13.
- `feature_schema_fingerprint` placement: resolved in A4.
- `auto`-precision dispatch location: resolved in A11.
- Snapshot CI workflow shape: resolved in A14.
- Quickstart CI acceptance threshold: resolved in A14.
- Hypothesis `suppress_health_check` extras: resolved in A14.
- NaN-loss Variant B fixture isolation: resolved in A14.
- `decision_threshold_` absence on multiclass: resolved in A2.

Q12 (mkdocs vs. sphinx) was resolved during the requirements doc
update; A12 here documents the implementation specifics.

## A1: Package layout

```
src/seq_sklearn/
  __init__.py                       re-exports public API
  errors.py                         SeqSklearnError hierarchy (F8)
  hardware.py                       detect(), HardwareTier (N5)
  _validate.py                      check_y, check_columns (internal)
  serialization.py                  safetensors + JSON I/O helpers (F4)
  logging.py                        Event enum, emit() helper (F11)

  config/
    __init__.py
    _domains.py                     TASK_TYPES, LOSS_STRATEGIES, etc. (F5)
    _params_adapter.py              BaseEstimator-side mutable mirror of pydantic configs
    base.py                         BaseTrainingConfig, BaseModelConfig
    tabular.py                      TabularToSequenceConfig
    tft.py                          TFTConfig
    recurrent.py                    RecurrentSequenceEstimatorConfig (v1 skeleton, INTERNAL)
    _validity.py                    F5 validity-matrix cross-field validator

  data/
    __init__.py
    tabular_to_sequence.py          TabularToSequence (sklearn transformer)
    splits.py                       compute_three_way_split (pure function, F2)
    encoders.py                     CategoricalEncoder, scaler factories
    synthetic/
      __init__.py
      generator.py                  SyntheticPanelGenerator (F6 DGP)
      _rng.py                       single-Generator threading helper

  models/
    __init__.py
    _layers.py                      Linear / LayerNorm / Embedding factory (F4)
    _backbone.py                    BackboneOutput dataclass base, BaseBackbone abstract (A15)
    _base.py                        BaseSequenceEstimator
    _classifier.py                  BaseSequenceClassifier
    _regressor.py                   BaseSequenceRegressor
    _heads.py                       ClassificationHead, RegressionHead
    _attention.py                   mask polarity flip helper (padding_mask -> attn_mask)
    transformer/
      __init__.py
      _backbone.py                  TransformerBackboneOutput dataclass (A15)
      _base.py                      TransformerSequenceEstimator
      _interpretable_attention.py   shared-V interpretable multi-head attention (TFT)
      _positional.py                positional encoding helpers
      tft/
        __init__.py
        backbone.py                 TFTBackbone (nn.Module)
        blocks.py                   VSN, GRN, GLU, AddNorm
        classifier.py               TFTClassifier
        regressor.py                TFTRegressor
    recurrent/
      __init__.py
      _base.py                      RecurrentSequenceEstimator (abstract, INTERNAL in v1)

  training/
    __init__.py
    trainer.py                      Trainer (Lightning wrapper)
    _lightning_module.py            _LightningModule
    _determinism.py                 enable_strict_mode() (N4)
    _precision.py                   resolve_precision(tier, requested) (N5)
    callbacks.py                    NaNLossGuard, GradScalerWatchdog, EventEmitter, RngStateCallback
    losses.py                       build_loss() dispatch (F5)
    optimizers.py                   build_optimizer() (F5)
    schedulers.py                   build_scheduler() (F5)
    sampling.py                     oversample_minority / undersample_majority (F5)

  calibration/
    __init__.py
    _protocol.py                    _Calibrator Protocol
    classification.py               TemperatureScaling, PlattScaling, IsotonicCalibrator
    regression.py                   ConformalCalibrator, IsotonicQuantileCalibrator
    threshold.py                    ThresholdTuner

  inference/
    __init__.py
    attention.py                    AttentionOutput, RegressionAttentionOutput (BETA)

  model_selection/
    __init__.py
    split.py                        EntityTimeSeriesSplit

  tuning/
    __init__.py
    suggest_params.py               suggest_params(trial, model_class)
    pruning.py                      optuna_trial_guard context manager
```

```
tests/
  conftest.py                       shared fixtures + hypothesis profiles + check_estimator subset
  _snapshots/                       pinned snapshot artifacts (N1)
  unit/
  integration/
  e2e/
  deploy/
  perf/
  snapshot/
```

```
docs/
  requirements.md                   v1 contract (authoritative)
  architecture.md                   this file
  observability.md                  F11 event-payload reference
  research/*.md                     13 grounded research briefs (read-only after consensus)
  api/                              auto-generated by mkdocstrings
  examples/                         runnable .py examples
```

## A2: Class hierarchy

```
sklearn.base.BaseEstimator
└── BaseSequenceEstimator (abstract)
    ├── BaseSequenceClassifier (sklearn.base.ClassifierMixin)
    │   └── TFTClassifier (composes TransformerSequenceEstimator.Classifier mixin)
    └── BaseSequenceRegressor (sklearn.base.RegressorMixin)
        └── TFTRegressor (composes TransformerSequenceEstimator.Regressor mixin)

BaseSequenceEstimator (abstract, INTERNAL surface for v3)
└── RecurrentSequenceEstimator (abstract, INTERNAL in v1, STABLE in v3)
    ├── RecurrentSequenceClassifier (abstract, INTERNAL in v1)
    └── RecurrentSequenceRegressor (abstract, INTERNAL in v1)
```

The transformer-family mixin classes live as nested classes on
`TransformerSequenceEstimator`:

```python
# src/seq_sklearn/models/transformer/_base.py
class TransformerSequenceEstimator:
    class Classifier:
        """Mixin: composes with BaseSequenceClassifier."""
        # attention-extraction hook, variable-selection-weight logging
    class Regressor:
        """Mixin: composes with BaseSequenceRegressor."""
```

Concrete model files inherit from the family mixin AND the
`BaseSequence*` class. Example:
`class TFTClassifier(TransformerSequenceEstimator.Classifier, BaseSequenceClassifier): ...`
The mixin overrides template methods the base class calls.

The `TFTBackbone` is a pure `nn.Module` owned by the estimator. The
estimator holds:

- a `TFTConfig` (frozen pydantic v2 BaseModel), built inside `fit`
  from mutable instance attributes per the sklearn convention.
  Sourced from `docs/research/pydantic_sklearn.md`: no surveyed
  library (sktime / skrub / feature-engine / category_encoders /
  imbalanced-learn) currently combines a frozen pydantic config with
  a sklearn estimator; the canonical pattern derived here is new.
- a fitted `TabularToSequence` after `fit`.
- a `_LightningModule` wrapper that wraps `TFTBackbone` + the head +
  the loss.
- a fitted optional calibrator (one of the calibration strategies in
  F5).
- the sklearn fit-state attributes (F1.1).

**`decision_threshold_` on multiclass / regression.** The attribute
is ABSENT (not set on the instance) for any non-binary classifier and
for any regressor; `hasattr(estimator, 'decision_threshold_')` returns
`False` on those instances. The N1 fit-state-attribute test
parametrizes both presence (binary + `threshold_tuning=True`) and
absence (multiclass, binary without tuning, regressor).

## A3: Public-API surface

```python
# src/seq_sklearn/__init__.py
from seq_sklearn.models.transformer.tft.classifier import TFTClassifier
from seq_sklearn.models.transformer.tft.regressor import TFTRegressor
from seq_sklearn.data.tabular_to_sequence import TabularToSequence
from seq_sklearn.model_selection.split import EntityTimeSeriesSplit
from seq_sklearn.config.tft import TFTConfig
from seq_sklearn.config.tabular import TabularToSequenceConfig
from seq_sklearn.hardware import HardwareTier, detect
from seq_sklearn.errors import (
    SeqSklearnError, ConfigError, DataContractError,
    TrainingError, PredictionError, NotFittedError,
)
from seq_sklearn.inference.attention import (
    AttentionOutput, RegressionAttentionOutput,
)
from seq_sklearn.tuning.suggest_params import suggest_params
from seq_sklearn.tuning.pruning import optuna_trial_guard

__all__ = [
    "TFTClassifier", "TFTRegressor",
    "TabularToSequence", "TabularToSequenceConfig",
    "TFTConfig",
    "EntityTimeSeriesSplit",
    "HardwareTier", "detect",
    "SeqSklearnError", "ConfigError", "DataContractError",
    "TrainingError", "PredictionError", "NotFittedError",
    "AttentionOutput", "RegressionAttentionOutput",
    "suggest_params", "optuna_trial_guard",
]
```

Everything else is INTERNAL per the requirements stability rules.
The `RecurrentSequenceEstimator` class and
`RecurrentSequenceEstimatorConfig` (A6.1) are INTERNAL-tier in v1 by
design and are absent from this re-export list; v3 promotes both to
STABLE.

## A4: Configuration schemas

The pydantic + sklearn integration pattern is novel; the research
brief at `docs/research/pydantic_sklearn.md` documents that no
surveyed library combines a frozen pydantic config with a sklearn
estimator. The canonical pattern derived for seq-sklearn:

1. **The estimator owns mutable scalar attributes**, one per pydantic
   field, exposed as keyword arguments to `__init__`. sklearn
   `set_params` mutates them in place. This is the only way to
   reconcile `frozen=True` with sklearn's mutation contract.
2. **The frozen pydantic config is constructed inside `fit`** and
   stored as `self.config_` (a trailing underscore marks the
   fit-state attribute). `__sklearn_tags__` reads raw instance
   attributes, never `self.config_`, because `check_estimator`
   invokes the tag method on unconfigured instances.
3. **Nested configs use the BaseEstimator-adapter pattern**, the
   approach recommended in `docs/research/sklearn.md` for combining
   pydantic v2 configs with sklearn's nested `get_params` /
   `set_params` protocol. Pydantic `BaseModel` does not implement
   `get_params` / `set_params` itself; rather than flatten the
   nested config into the outer estimator (which the research brief
   explicitly rejects because it duplicates the pydantic schema and
   loses validation grouping), the library wraps each pydantic
   config in a thin `BaseEstimator` adapter whose fields mirror the
   pydantic schema 1:1. The adapter's `to_pydantic()` method
   constructs the frozen pydantic instance inside the outer
   estimator's `fit`.

   ```python
   # src/seq_sklearn/config/_params_adapter.py
   class TabularConfigParams(BaseEstimator):
       def __init__(
           self,
           id_col: str = "id",
           time_col: str = "time",
           lookback: int = 12,
           # ... every TabularToSequenceConfig field mirrored
       ) -> None:
           self.id_col = id_col
           self.time_col = time_col
           self.lookback = lookback
           # ...

       def to_pydantic(self) -> TabularToSequenceConfig:
           return TabularToSequenceConfig(
               id_col=self.id_col, time_col=self.time_col,
               lookback=self.lookback, # ...
           )

   class TFTClassifier(ClassifierMixin, BaseEstimator):
       def __init__(
           self,
           tabular_config: TabularConfigParams | None = None,
           hidden_size: int = 128,
           # ... every TFTConfig field mirrored
       ) -> None:
           self.tabular_config = tabular_config or TabularConfigParams()
           self.hidden_size = hidden_size
           # ...
   ```

   sklearn's `get_params(deep=True)` recurses into `tabular_config`
   automatically because the adapter is a `BaseEstimator`, producing
   the canonical `tabular_config__lookback` flat keys.
   `set_params(tabular_config__lookback=6)` chains via standard
   sklearn double-underscore traversal:
   `self.tabular_config.set_params(lookback=6)`. The frozen pydantic
   instance is built inside `fit` as
   `self.config_ = self._build_config()`, where `_build_config`
   reads from `self.tabular_config.to_pydantic()` plus the outer
   estimator's mirrored fields. The pydantic validity-matrix
   validator runs inside `to_pydantic()`; failures wrap into
   `ConfigError` at the `_build_config` call site (step 4 below).

   **Clone safety**: `sklearn.base.clone(estimator)` calls
   `type(estimator)(**estimator.get_params(deep=False))`. The
   shallow-params dict contains the same `tabular_config` adapter
   instance by reference. The outer `__init__` defends against
   aliasing by deep-copying the adapter:
   `self.tabular_config = (
       sklearn.base.clone(tabular_config) if tabular_config is not None
       else TabularConfigParams()
   )`. `sklearn.base.clone` recursively constructs a fresh adapter
   instance from the original's params; this is the sklearn-idiomatic
   alternative to `copy.deepcopy` and works under both joblib
   `prefer='threads'` and `prefer='processes'` (the joblib-process
   path pickles each estimator independently, so adapter aliasing
   collapses by construction).
4. **Cross-field validators wrap `pydantic.ValidationError` into
   `ConfigError`** at the `_build_config()` call site inside `fit`,
   not inside the validator itself (the validator stays a pure
   `@model_validator(mode="after")` returning the model).

**`BaseTrainingConfig`** (shared across families):

```python
class BaseTrainingConfig(BaseModel):
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 50
    optimizer: Literal["adamw", "adam", "sgd"] = "adamw"
    scheduler: Literal["constant", "cosine_with_warmup", "one_cycle", "reduce_on_plateau"] = "cosine_with_warmup"
    warmup_steps: int = 100
    gradient_clip_val: float | None = None
    accumulate_grad_batches: int = 1
    precision: Literal["bf16-mixed", "16-mixed", "32-true", "auto"] = "auto"
    early_stopping_patience: int = 10
    val_check_interval: float = 1.0
    val_fraction: float = 0.1
    cal_fraction: float = 0.1
    val_split_strategy: Literal["time_ordered", "random"] = "time_ordered"
    num_workers: int | None = None         # None -> min(4, os.cpu_count() or 1)
    pin_memory: bool | None = None         # None -> True on CUDA, False on CPU
    seed: int = 42
    verbose: bool = True

    model_config = ConfigDict(extra="forbid", frozen=True)
```

**`BaseModelConfig`** (shared across all concrete models):

```python
class BaseModelConfig(BaseTrainingConfig):
    task_type: Literal[
        "binary", "multiclass", "multilabel",
        "regression_point", "regression_quantile", "regression_multioutput",
    ]
    loss_strategy: Literal[
        "cross_entropy", "focal", "mse", "mae", "huber", "pinball",
    ]
    imbalance_strategy: Literal[
        "none", "class_weighted", "oversample_minority", "undersample_majority",
    ] = "none"
    calibration_strategy: Literal[
        "none", "temperature", "platt", "isotonic",
        "conformal", "isotonic_quantile",
    ] = "none"
    threshold_tuning: bool = False
    threshold_metric: Literal["f1", "balanced_accuracy", "youden_j"] = "f1"
    focal_gamma: float = 2.0
    huber_delta: float = 1.0
    quantiles: tuple[float, ...] | None = None
    oversample_ratio: float = 1.0

    @model_validator(mode="after")
    def _check_validity_matrix(self) -> "BaseModelConfig":
        from seq_sklearn.config._validity import check_combo
        check_combo(self.task_type, self.loss_strategy,
                    self.imbalance_strategy, self.calibration_strategy)
        return self

    @model_validator(mode="after")
    def _check_quantiles_monotone(self) -> "BaseModelConfig":
        if self.quantiles is None:
            return self
        q = self.quantiles
        if any(not (0.0 < v < 1.0) for v in q):
            raise ValueError(f"quantiles must lie in (0, 1); got {q}")
        if any(q[i] >= q[i + 1] for i in range(len(q) - 1)):
            raise ValueError(f"quantiles must be strictly increasing; got {q}")
        return self
```

**`TabularToSequenceConfig`**:

```python
class TabularToSequenceConfig(BaseModel):
    id_col: str
    time_col: str
    static_categorical_cols: tuple[str, ...] = ()
    static_real_cols: tuple[str, ...] = ()
    time_varying_real_cols: tuple[str, ...] = ()
    time_varying_categorical_cols: tuple[str, ...] = ()
    lookback: int = 12
    prediction_step: int = 1
    min_periods: int = 1
    min_periods_predict: int = 1
    scaling_real: Literal["standard", "robust", "quantile_uniform", "none"] = "standard"
    scaling_static_real: Literal["standard", "robust", "quantile_uniform", "none", "inherit"] = "inherit"
    clip_features: float | None = None
    max_categorical_cardinality: int = 1000
    hash_high_cardinality: bool = False
    categorical_embed_dims: Mapping[str, int] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid", frozen=True)
```

`categorical_embed_dims` uses `Mapping[str, int]` (not `dict[str, int]`)
so the frozen model stays hashable per a pydantic v2 idiom.

**`TFTConfig`**:

```python
class TFTConfig(BaseModelConfig):
    hidden_size: int = 128
    attention_heads: int = 4
    dropout: float = 0.1
    variable_selection_dropout: float = 0.1
    prediction_readout: Literal["last_valid", "mean_pool"] = "last_valid"
    tabular_config: TabularToSequenceConfig
```

`TFTClassifier.__init__` accepts `tabular_config: TabularConfigParams`
plus every other `TFTConfig` field as a top-level keyword argument
(see step 3 above for the adapter pattern). `get_params(deep=True)`
recurses into the adapter automatically, yielding flat keys like
`tabular_config__lookback`. `set_params(tabular_config__lookback=6)`
chains via the standard sklearn double-underscore traversal into
`self.tabular_config.set_params(lookback=6)`. The
`clf__tabular_config__lookback` triple-underscore form chains
through `Pipeline` cleanly for the same reason.

`GridSearchCV(estimator=TFTClassifier(),
param_grid={"tabular_config__lookback": [6, 12, 24]})` works without
further plumbing.

**`feature_schema_fingerprint`** is computed in `TabularToSequence.fit`
as a sha256 over the sorted declared-column names plus their pandas
dtypes from the fit-time `X`. It is set on the estimator as
`feature_schema_fingerprint_` AND included in the F4 save metadata. The
N1 fit-state attribute test covers both layers; the save/load
round-trip test asserts the fingerprint matches across the round trip.

## A5: Data pipeline

```
TabularToSequence.fit(X: pd.DataFrame, y: ArrayLike) -> TabularToSequence
  1. Validate columns exist (DataContractError if not).
  2. Validate (id_col, time_col) uniqueness.
  3. Validate time_col dtype: datetime64[ns], datetime64[ns, <tz>],
     PeriodDtype, or signed int. Object dtype rejected.
  4. Validate y shape (1D in v1; delegates to _validate.check_y).
  5. Fit categorical encoders (column-wise; <unk> slot index 0 per
     column).
  6. Fit scalers (column-wise per scaling_real / scaling_static_real).
  7. Build feature_schema_fingerprint.
  8. Mark fitted.

TabularToSequence.transform(X: pd.DataFrame) -> dict[str, Tensor]
  1. Sort by (id_col, time_col).
  2. Per entity, slide a window of length lookback over sorted rows.
  3. Align target to prediction_step relative to the window end.
  4. Encode categoricals (unseen -> <unk> at index 0).
  5. Scale reals, clip if configured.
  6. Pad short entities to lookback length with mask.
  7. Emit dict:
     - static_categorical: LongTensor (B, sum(static_cat_card))
     - static_real: FloatTensor (B, len(static_real))
     - time_varying_real: FloatTensor (B, L, len(tv_real))
     - time_varying_categorical: LongTensor (B, L, len(tv_cat))
     - padding_mask: BoolTensor (B, L); True = padding (ignore)
     - target: appropriate dtype per task_type
     - entity_id: LongTensor (B,) for diagnostics

TabularToSequence.inverse_transform(X_seq: dict[str, Tensor]) -> pd.DataFrame
  Reverses scaling on numeric columns; decodes categorical indices
  back to original level strings. Columns that were hash-tricked raise
  NotImplementedError. <unk> indices map back to the literal "<unk>"
  string.
```

**Tensor convention**: `(batch, time, features)`. Masks are
`(batch, time)` with `True = padding (ignore)` per F3 / source
`docs/research/pytorch.md`.

**Split function** (`data/splits.py`, pure function so unit tests do
not need a Trainer):

```python
def compute_three_way_split(
    entity_ids: np.ndarray,           # shape (n_windows,), int
    window_time_index: np.ndarray,    # shape (n_windows,), int; ordinal per
                                      # window, ascending within each entity
    *,
    val_fraction: float,
    cal_fraction: float,
    val_split_strategy: Literal["time_ordered", "random"],
    calibration_set_provided: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx, cal_idx) integer index arrays.

    `window_time_index[i]` is the time ordinal of window i within its
    entity (0 = oldest valid window for that entity). The function
    uses this to identify the LAST cal_fraction rows per entity (F2),
    independent of the row order in `entity_ids`. Caller does not need
    to pre-sort.

    Time-ordered policy: for each entity, the tail `cal_fraction`
    windows by `window_time_index` form the calibration fold, the
    preceding `val_fraction` form the validation fold, the remainder
    is training.

    Random policy: ignores `window_time_index`; emits a UserWarning
    via the caller's logger if more than one unique `entity_id`
    appears.

    Returns:
    - (train_idx, val_idx, cal_idx) when calibration_set_provided=False
      AND cal_fraction > 0. cal_idx is non-empty.
    - (train_idx, val_idx, np.empty(0, dtype=int)) when
      calibration_set_provided=True AND cal_fraction == 0.0. cal_idx
      is an empty array; the calibration fold is supplied externally
      via the `calibration_set` keyword to `fit`.
    - (train_idx, val_idx, np.empty(0, dtype=int)) when
      calibration_strategy='none' AND threshold_tuning=False (the F2
      collapse rule); the calibration fold is folded back into training.

    Raises ConfigError when calibration_set_provided=True AND
    cal_fraction > 0 (F2 conflict rule)."""
```

The Trainer constructs `window_time_index` from
`TabularToSequence.transform`'s ordering: each entity's windows
appear contiguously in the output and are in ascending time order
(per step 1 of A5), so the index is simply
`np.concatenate([np.arange(n_i) for n_i in per_entity_counts])`.

## A6: TFTBackbone

Block flow verified against pytorch-forecasting, the Google Research
TF1 original, and PlaytikaOSS tft-torch per
`docs/research/tft.md`. Encoder-only adaptation; no decoder, no
future window.

```
Inputs (per window):
  static_categorical    (B, sum_static_cat)
  static_real           (B, len(static_real))
  time_varying_real     (B, L, len(tv_real))
  time_varying_cat      (B, L, sum_tv_cat)
  padding_mask          (B, L)  True = padding

Static encoders:
  Per-input embedding / passthrough -> (B, d_input_static)
  Static VSN (one VSN over the static inputs) -> static-selection
    weights + selected vector (B, hidden_size).
  Four context vectors via GRN stack, all consumed by the encoder.
  Listed below in the order they are USED by the encoder pipeline
  (the LSTM-init pair is in the (h_0, c_0) order that nn.LSTM
  expects, so the prose order matches the tuple-construction order
  in code):
    c_s -> conditions past-VSN selection (gating)
    c_h -> initializes LSTM hidden state    (h_0 in nn.LSTM signature)
    c_c -> initializes LSTM cell state      (c_0 in nn.LSTM signature)
    c_e -> enriches the post-LSTM add-norm output via GRN before
           attention

Past variable selection:
  Per-timestep VSN over time-varying inputs (B, L, hidden_size) +
  variable-selection weights. Conditioned on c_s. Padded positions
  zeroed BEFORE softmax over variables.

LSTM encoder:
  Initial state from (c_h, c_c) (positional ORDER per nn.LSTM(input,
  (h_0, c_0))):
    lstm(input, (c_h, c_c))
  Variable-length handling via pack_padded_sequence(enforce_sorted=False)
  + pad_packed_sequence. Hidden size = hidden_size. Output (B, L, hidden_size).

Post-LSTM gating:
  GLU + AddNorm against pre-LSTM VSN output (skip connection).
  c_e enrichment: GRN(post_lstm, context=c_e) -> attn_in (B, L, hidden_size).

Interpretable multi-head self-attention (custom, shared-V):
  Per-head Q, K projections; SINGLE V projection shared across heads.

  Two forward paths exist on the same module:

  (a) Fast path (training, inference): uses SDPA, no per-head score
      tensor is materialized.
      q = Q_proj(attn_in)            -> view (B, H, L, d)
      k = K_proj(attn_in)            -> view (B, H, L, d)
      v = V_proj(attn_in)            -> (B, L, d)
      v_broadcast = v.unsqueeze(1).expand(B, H, L, d)
      attn_mask_bool = ~padding_mask                # True = participate
      out_per_head = F.scaled_dot_product_attention(
          q, k, v_broadcast, attn_mask=...)         # (B, H, L, d)
      out = out_per_head.mean(dim=1)                # average across heads
      out = out_proj(out)                           # (B, L, hidden_size)

  (b) Interpretable path (predict_with_attention only): replaces SDPA
      with a manual softmax to capture the per-head score tensor.
      scores = (q @ k.transpose(-2, -1)) / sqrt(d)  # (B, H, L, L)
      scores = scores.masked_fill(~attn_mask_bool[:, None, None, :], -inf)
      attn_weights = scores.softmax(dim=-1)         # (B, H, L, L); post-softmax, pre-V
      # NaN safety: softmax over an all-(-inf) key row produces NaN.
      # The architecture's mask.any(dim=1).all() check guarantees at
      # least one valid key per batch element, so this should never
      # fire. The torch.nan_to_num is defensive belt-and-braces against
      # an upstream mask bug; it costs one O(BHLL) pass and zeroes any
      # NaN before it propagates into v_broadcast.
      attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
      out_per_head = attn_weights @ v_broadcast     # (B, H, L, d)
      out = out_per_head.mean(dim=1)
      out = out_proj(out)
      attn_weights is returned alongside the output for TransformerBackboneOutput.

  The fast path skips capture entirely; the interpretable path
  materializes attn_weights once, on demand. predict_with_attention
  routes through (b); fit / predict / predict_proba route through (a).
  The shared-V correctness test asserts equality of outputs between
  (a) and (b) on a fixed input within float tolerance.

  Mathematically equivalent to a per-head V where all per-head V
  projections are tied to one another. The N1 V-projection weight
  test asserts the V-projection weight count equals 1, not H, by
  counting named parameters with the expected key.

  Math backend forced when ONNX-exporting via
  torch.nn.attention.sdpa_kernel(SDPBackend.MATH).
  Output (B, L, hidden_size).

Post-attention gating:
  GLU + AddNorm against attn_in.
  Position-wise feed-forward (GRN) + GLU + AddNorm.

Readout (per prediction_readout):
  last_valid: index of last True in (~padding_mask) per batch element;
              gather hidden[b, last_idx, :] -> (B, hidden_size).
  mean_pool:  mean over (~padding_mask) positions per batch element.

Asserts mask.any(dim=1).all() before readout; raises
PredictionError("window had zero valid timesteps after preprocessing")
otherwise. L=1 windows are supported (last_valid index = 0).
```

**Layer factory** (`models/_layers.py`): every `nn.Linear`,
`nn.LayerNorm`, `nn.Embedding` instantiation routes through one
module. v1 returns standard PyTorch layers; v2 (the FP8 pass per
N5) can swap in Transformer Engine equivalents here without changing
calling code.

**Heads** (`models/_heads.py`):

```python
class ClassificationHead(nn.Module):
    def __init__(self, d_model: int, out_dim: int): ...
    def forward(self, h: Tensor) -> Tensor:
        return self.proj(h)  # emits raw logits, no activation

class RegressionHead(nn.Module):
    def __init__(self, d_model: int, out_dim: int, n_quantiles: int): ...
    def forward(self, h: Tensor) -> Tensor:
        # (B, d_model) -> (B, out_dim * n_quantiles)
        return self.proj(h)
```

The classification head emits raw logits; `BCEWithLogitsLoss` /
`CrossEntropyLoss` apply sigmoid / log-softmax internally during
training. `predict_proba` applies the activation post-hoc on cached
logits. `predict_with_attention` returns logits, not probabilities, in
its prediction field. The head parameter `out_dim` is distinct from
sklearn's `n_outputs_` attribute (F1.1): `n_outputs_` follows the
sklearn convention (1 for binary and multiclass in v1; equals label
count for multi-label in v1.1) while `out_dim` is the projection's
tensor dimension which equals `num_classes` for multiclass. For
quantile regressors, `n_outputs_=1` in v1; quantile dimensionality is
exposed via the `quantiles_` attribute and the `predict_quantiles`
entry point, NOT via `n_outputs_`.

## A6.1: Recurrent skeleton (INTERNAL, v1)

The skeleton ships in v1 as an abstract base class plus its config
schema; no concrete LSTM / GRU exists in v1. Its purpose is to
validate that `BaseSequenceEstimator`'s contract supports the
recurrent surface area without requiring a v3 forward port. Sourced
from `docs/research/lstm_gru.md`.

```python
# src/seq_sklearn/models/recurrent/_base.py
class RecurrentSequenceEstimator(BaseSequenceEstimator, ABC):
    @abstractmethod
    def _init_hidden(
        self, batch_size: int, device: torch.device
    ) -> tuple[Tensor, ...]: ...

    @abstractmethod
    def _readout(self, hidden_seq: Tensor, mask: Tensor) -> Tensor: ...

    @abstractmethod
    def _bptt_window(self) -> int | None: ...
```

```python
# src/seq_sklearn/config/recurrent.py
class RecurrentSequenceEstimatorConfig(BaseModelConfig):
    bidirectional: bool = False
    recurrent_dropout: float = 0.1
    recurrent_dropout_kind: Literal["weight_drop", "variational", "bernoulli"] = "weight_drop"
    hidden_init_strategy: Literal["zero", "learned", "per_entity"] = "zero"
    readout: Literal["last_valid", "mean_pool", "attention"] = "last_valid"
    bptt_window: int | None = None
```

The v1 test instantiates a no-op concrete subclass (defined inline in
the test module) that fills the three abstract methods with trivial
bodies, then asserts the subclass composes with
`BaseSequenceEstimator`'s `fit` / `predict` shell. The recurrent
config is NOT in the public API surface in v1; v3 promotes both to
STABLE.

## A7: Training pipeline

`training.trainer.Trainer` is the library-side wrapper. It:

1. Accepts a fitted `TabularToSequence`, a `BaseModelConfig`, and a
   model factory (callable returning the `nn.Module` backbone + head).
2. Calls `data.splits.compute_three_way_split(...)` for the train /
   val / cal split (F2). All split logic lives in `data/splits.py`;
   the Trainer does not implement split semantics inline.
3. Builds DataLoaders with the F5 defaults
   (`num_workers=min(4, os.cpu_count() or 1)`, `pin_memory=True` on CUDA,
   `persistent_workers=True` when `num_workers > 0`).
4. Builds a `pl.Trainer` with:
   - precision via `_precision.resolve_precision()`.
   - `deterministic=True` when `precision == "32-true"` and `seed` is set;
     the Trainer separately calls `enable_strict_mode()` so the four
     N4 flags are guaranteed (Lightning's `deterministic=True` calls
     `torch.use_deterministic_algorithms(True)` but does NOT set
     `cudnn.benchmark=False` or `CUBLAS_WORKSPACE_CONFIG`).
   - callbacks: `EarlyStopping`, `ModelCheckpoint(save_last=True,
     save_top_k=1)`, `NaNLossGuard`, `GradScalerWatchdog` (mixed
     precision only), `EventEmitter`, `RngStateCallback`. Source:
     `docs/research/lightning.md`.
   - logger: pass-through. The library does NOT register a default
     logger; Lightning auto-attaches `TensorBoardLogger` unless
     explicitly suppressed. The Trainer passes `logger=False` by
     default; callers attach `MLFlowLogger` / `WandbLogger` / etc.
     manually.
5. Wraps the model in `_LightningModule`.
6. Calls `pl_trainer.fit(lightning_module, train_loader, val_loader)`.
7. After fit, builds the calibrator on the calibration fold and runs
   the threshold tuner if `threshold_tuning=True`.

**`_LightningModule`** with an explicit constructor for unit testing:

```python
class _LightningModule(pl.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        loss: nn.Module,
        optimizer_factory: Callable[[Iterable[nn.Parameter]], optim.Optimizer],
        scheduler_factory: Callable[[optim.Optimizer], dict[str, object]] | None,
        val_metric_name: str = "val_loss",
        bptt_window: int | None = None,
        optuna_trial: optuna.Trial | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.loss = loss
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.val_metric_name = val_metric_name
        self.bptt_window = bptt_window
        self._optuna_trial = optuna_trial
        self._consecutive_nan = 0
        self._pending_prune: tuple[int, float] | None = None
        self.automatic_optimization = True   # v1 stays on automatic

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        # ...forward + loss; MUST also store self._last_train_output = backbone_out
        ...

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        # MUST call self.log(self.val_metric_name, loss, on_step=False, on_epoch=True)
        # so Lightning populates self.trainer.callback_metrics[self.val_metric_name],
        # which on_validation_epoch_end (and the Optuna pruning hook) reads.
        ...

    def configure_optimizers(self) -> dict[str, object]:
        opt = self.optimizer_factory(self.parameters())
        if self.scheduler_factory is None:
            return {"optimizer": opt}
        sched_dict = self.scheduler_factory(opt)
        # sched_dict shape per Lightning 2.6:
        #   {"scheduler": ..., "monitor": self.val_metric_name,
        #    "interval": "epoch" | "step", "frequency": 1, "strict": True}
        # "monitor" is mandatory for ReduceLROnPlateau.
        return {"optimizer": opt, "lr_scheduler": sched_dict}

    def on_validation_epoch_end(self) -> None:
        # Lifecycle in Lightning 2.6: on_validation_epoch_end fires
        # BEFORE on_train_epoch_end (per Lightning discussion 14318).
        # We MUST NOT raise optuna.TrialPruned here directly because
        # doing so would skip on_train_epoch_end and the train.epoch /
        # entropy structured-log events would never fire for the
        # pruned epoch. Instead, stash the prune decision on self and
        # raise from the end of on_train_epoch_end so logging always
        # runs first.
        val_metric = self.trainer.callback_metrics.get(self.val_metric_name)
        if self._optuna_trial is not None and val_metric is not None:
            self._optuna_trial.report(val_metric.item(), step=self.current_epoch)
            if self._optuna_trial.should_prune():
                self._pending_prune = (self.current_epoch, val_metric.item())

    def on_train_epoch_end(self) -> None:
        # Emits train.epoch, train.var_selection_entropy,
        # train.attention_entropy via the structured-log helper.
        ...
        # Raise the deferred prune AFTER logging has fired.
        if self._pending_prune is not None:
            epoch, metric = self._pending_prune
            self._pending_prune = None
            raise optuna.TrialPruned(f"epoch={epoch} metric={metric}")

    def _bptt_step(self, batch: dict[str, Tensor]) -> Tensor:
        # v1 no-op (encoder-style TFT). v3 implements truncated BPTT
        # against bptt_window for recurrent models.
        ...
```

**Construction order** (the Estimator owns it):

1. Estimator `fit` builds `self.config_` (frozen pydantic) from its
   flat attribute dict.
2. Estimator passes `self.config_` to the `Trainer` wrapper.
3. The `Trainer` wrapper builds the curried factories:
   `optimizer_factory = partial(build_optimizer, config=self.config_)` and
   `scheduler_factory = partial(build_scheduler, config=self.config_)`.
4. The `Trainer` constructs `_LightningModule(backbone, head, loss,
   optimizer_factory, scheduler_factory, ...)`. The factories close
   over the post-fit `config_`; the LightningModule never reads the
   estimator directly.
5. The `Trainer` constructs `pl.Trainer(...)` and calls
   `pl_trainer.fit(lightning_module, train_loader, val_loader)`.

This decouples the LightningModule from the Estimator and from
pre-fit state, so unit tests can construct the module with plain
callables (e.g. `lambda params: torch.optim.AdamW(params, lr=1e-3)`)
without standing up either layer.

**Unit-test fixture pattern.** Tests stub `backbone`, `head`, `loss`
independently. For tests that touch hooks reading `self.trainer.*`
(notably `on_validation_epoch_end`), the fixture sets
`module._trainer = MagicMock(...)` directly because
`pl.LightningModule.trainer` is a property backed by `_trainer`:

```python
def make_test_module(loss=None, optuna_trial=None) -> _LightningModule:
    mod = _LightningModule(
        backbone=_DummyBackbone(),
        head=_DummyHead(),
        loss=loss or _LossReturningScalar(),
        optimizer_factory=lambda params: torch.optim.AdamW(params, lr=1e-3),
        scheduler_factory=None,
        optuna_trial=optuna_trial,
    )
    mock_trainer = MagicMock()
    mock_trainer.callback_metrics = {"val_loss": torch.tensor(0.5)}
    mock_trainer.current_epoch = 0
    mod._trainer = mock_trainer
    return mod
```

The NaN-loss Variant A test mocks `loss` to a module returning
`torch.tensor(float('nan'))` on the third call; no real Estimator or
real Trainer is needed.

**Callbacks**:

- **`NaNLossGuard`** implements `on_train_batch_end(trainer, pl_module,
  outputs, batch, batch_idx)`. Lightning passes `outputs` as the
  scalar tensor returned by `training_step` when
  `automatic_optimization=True`; the callback checks
  `torch.isnan(outputs)`. On a NaN, increments an internal counter and
  emits `train.nan_step_skipped`. On the third consecutive NaN,
  raises `TrainingError("3 consecutive NaN training steps; aborting
  per F9")` with the offending `batch_idx` in the log payload. A
  non-NaN step resets the counter to zero.
- **`GradScalerWatchdog`** (mixed precision only) implements
  `on_train_batch_end`. Defensively checks
  `hasattr(trainer.precision_plugin, "scaler")` and is a no-op when
  the attribute is absent (CPU path, `bf16-mixed` on CC>=8.0, FP32).
  When the scaler is present, tracks `scaler.get_scale()` across
  batches; a decrease signals an overflow-driven skipped step.
  Three consecutive decreases raise `TrainingError` and emit
  `train.mixed_precision_diverged` at ERROR. The no-op-on-CPU
  guarantee makes the callback safe in the unit-test fixture that
  attaches it to a CPU-only `Trainer`; a separate
  `test_grad_scaler_watchdog_mock_scaler_decrease` injects a fake
  precision plugin to exercise the decrement path.
- **`EventEmitter`** exposes `self.emit(event, **payload)` from
  inside Lightning hooks. Uses
  `logging.getLogger("seq_sklearn.training").info(message,
  extra={"event": event.value, "payload": payload})`. The `extra`
  keys land as attributes on the `LogRecord`, so tests access them
  via `record.event` and `record.payload` (NOT
  `record.extra["event"]`); `caplog` captures correctly because
  `propagate=True` per the `propagate_seq_sklearn_logger` autouse
  fixture (A14).
- **`RngStateCallback`** snapshots Python / numpy / torch /
  `torch.cuda.get_rng_state_all()` into `checkpoint["seq_sklearn_rng"]`
  on `on_save_checkpoint` and restores in `on_load_checkpoint`.
  Lightning's own RNG capture has a known gap on
  `load_from_checkpoint` (issue #20204).

## A8: Loss factory

```python
def build_loss(
    task_type: str,
    loss_strategy: str,
    *,
    class_weights: Tensor | None,
    focal_gamma: float,
    huber_delta: float,
    quantiles: tuple[float, ...] | None,
) -> nn.Module:
    """F5 loss-class dispatch. Raises ConfigError for any
    (task_type, loss_strategy) pair not in the validity matrix."""
```

`class_weights` is non-None only when
`imbalance_strategy == "class_weighted"`. Binary class-weighting uses
`BCEWithLogitsLoss(pos_weight=neg_count / pos_count)` derived from
the train fold; multiclass uses `CrossEntropyLoss(weight=per_class_weights)`.

## A9: Calibration pipeline

```python
class _Calibrator(Protocol):
    def fit(self, logits: Tensor, y_true: Tensor) -> None: ...
    def transform(self, logits: Tensor) -> Tensor: ...
    def serialize(self) -> dict[str, object]: ...
    @classmethod
    def deserialize(cls, blob: dict[str, object]) -> "_Calibrator": ...
```

Concrete classes:

- `TemperatureScaling` (single scalar T; LBFGS-optimized on cal-set NLL).
- `PlattScaling` (binary only; logistic regression on logits).
- `IsotonicCalibrator` (`sklearn.isotonic.IsotonicRegression` wrapper;
  multiclass fits one regressor per class).
- `ConformalCalibrator` (split-conformal; per-quantile offset).
- `IsotonicQuantileCalibrator` (isotonic on the empirical CDF of
  prediction errors).

Each calibrator is testable standalone with hand-crafted
`(logits, y_true)` tensors; the architecture does NOT require
instantiating an Estimator to exercise the calibrator path.
`ConformalCalibrator.fit` raises
`TrainingError("non-monotone quantiles: <details>")` if the
calibrated quantile vector is not monotone increasing across the
calibration set (F9 contract). The non-monotone unit test feeds a
deliberately non-monotone tensor of shape `(N, len(quantiles))`,
e.g. `torch.tensor([[0.5, 0.3, 0.7]]).expand(64, 3)`, and asserts
the `TrainingError` raises with a message matching `r"non-monotone"`.

The calibrator is serialized to JSON inside the F4 `state.json` (each
calibrator's `serialize()` returns a JSON-compatible dict). The N1
save/load round-trip test includes at least one variant with
`calibration_strategy="temperature"` and asserts byte-equal
`predict_proba` output across the round trip; this exercises the
calibrator's `serialize` / `deserialize` path.

`ThresholdTuner` (binary classifier, when `threshold_tuning=True`)
fits on the calibration fold and picks the threshold maximizing
`threshold_metric` over a 101-point grid in [0, 1]. Stored as
`decision_threshold_` on the estimator.

## A9.1: EntityTimeSeriesSplit

The F10 cross-validation splitter is a STABLE public class.
Implementation contract:

```python
class EntityTimeSeriesSplit:
    def __init__(
        self,
        n_splits: int = 5,
        gap: int = 0,                  # measured in WINDOWS, not time-units
        max_train_size: int | None = None,
        lookback: int = 12,            # MUST match the estimator's
                                       # TabularToSequence.lookback so test
                                       # folds carry sufficient history
    ) -> None: ...

    def split(
        self,
        X: pd.DataFrame,
        y: ArrayLike | None = None,
        groups: ArrayLike | None = None,  # unused; entity grouping comes from id_col
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]: ...

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits
```

**`gap` is measured in windows**, matching `sklearn.model_selection.TimeSeriesSplit`'s
convention. Each entity's rows are sorted by time and chunked into
`n_splits + 1` time-ordered segments; for split `i`, the **train
indices** are segments `0..i`, with `gap` rows skipped between
train and test, and the **test indices** are segment `i+1`
extended LEFT by `lookback - 1` rows per entity from the preceding
train segment. The left-extension is the load-bearing part: without
it, `TabularToSequence.transform` on the test fold would see only
the test rows and pad every test entity to `lookback` length, which
destroys the temporal context the model was trained on. With the
extension, the test fold's `transform` sees `lookback - 1` rows of
real history per entity plus the test segment, and produces full
unpadded windows for every test prediction. The train and test
folds OVERLAP by `lookback - 1` rows per entity (rows in the
overlap are used as history-only at test time; no test target
spans into them). `max_train_size` caps the per-entity training
fold to its trailing-K rows (independent of `min_periods`).

Entities with fewer than `n_splits + 1 + gap + lookback - 1` rows
are dropped from that particular split with a `UserWarning`
(aggregated, one warning per `split()` call regardless of count).

The split function operates on the panel's `(entity_id, time)`
mapping; it does NOT call `TabularToSequence.transform` itself.
Callers compose `EntityTimeSeriesSplit` with `cross_val_score`
exactly as they would with sklearn's `TimeSeriesSplit`. The
`lookback` keyword on the splitter MUST match
`TFTClassifier(tabular_config__lookback=...)`; a helper
`EntityTimeSeriesSplit.from_estimator(estimator, **kwargs)`
constructor reads the lookback off a fitted or unfitted estimator
to remove the foot-gun.

## A10: Error hierarchy

```python
# src/seq_sklearn/errors.py
import sklearn.exceptions as sk_exc

class SeqSklearnError(Exception):
    """Root of the library exception hierarchy."""

class ConfigError(SeqSklearnError): ...
class DataContractError(SeqSklearnError): ...
class TrainingError(SeqSklearnError): ...
class PredictionError(SeqSklearnError): ...

class NotFittedError(SeqSklearnError, sk_exc.NotFittedError):
    """Raised when fit-requiring methods are called before fit.
    MRO order is LOAD-BEARING: SeqSklearnError first so library-side
    `except SeqSklearnError` catches; sklearn-side
    `except sklearn.exceptions.NotFittedError` also catches via the
    second parent. Both contracts are satisfied without preferential
    treatment in __str__ or pickling."""
```

`pydantic.ValidationError` is wrapped in `ConfigError` at the
`_build_config()` call site inside `fit`, not via
`@model_validator(mode="wrap")`. Source: `docs/research/pydantic_sklearn.md`.

## A11: Hardware and precision

```python
# src/seq_sklearn/hardware.py
from enum import IntEnum

class HardwareTier(IntEnum):
    CPU = 0
    PASCAL = 1
    VOLTA_TURING = 2
    AMPERE_ADA = 3
    HOPPER = 4
    BLACKWELL = 5

def detect() -> HardwareTier:
    """Pure detection. No side effects.

    Call sequence (the N1 mocked test patches this exact chain):
        1. if not torch.cuda.is_available(): return CPU
        2. major, _ = torch.cuda.get_device_capability(0)
        3. dispatch on `major` per N5:
              major == 6 -> PASCAL
              major == 7 -> VOLTA_TURING
              major == 8 -> AMPERE_ADA
              major == 9 -> HOPPER
              major >= 10 -> BLACKWELL
              else -> CPU (unsupported, warn once)

    Does NOT call torch.cuda.device_count() or torch.cuda.current_device().
    """
```

```python
# src/seq_sklearn/training/_precision.py
def resolve_precision(
    tier: HardwareTier,
    requested: Literal["bf16-mixed", "16-mixed", "32-true", "auto"],
) -> Literal["bf16-mixed", "16-mixed", "32-true"]:
    """Maps (tier, requested) to a concrete precision per N5.
    The dispatch decision lives here, not in hardware.detect(), so
    each function is independently unit-testable. The N1
    parametrized hardware-detect test exercises BOTH in sequence:
    one mock setup per tier patches torch.cuda calls, asserts the
    `HardwareTier` returned by `detect()`, then calls
    `resolve_precision(tier, 'auto')` and asserts the concrete
    precision. The combined test catches both detection bugs and
    dispatch bugs in one parametrized run."""
```

The auto-mapping is exactly per N5: `bf16-mixed` on CC>=8.0 (Ampere
and newer), `32` on everything else under `auto`. `16-mixed` is
selectable by explicit user opt-in but never picked by `auto` (the
TFT quantile-loss + softmax NaN history rules it out, per
`docs/research/pytorch.md`). The resolved precision is written into
the F4 `state.json` metadata as `precision_resolved`.

## A12: Documentation toolchain

mkdocs + mkdocs-material + mkdocstrings (python handler) +
griffe-pydantic. Resolved from Q12 in the requirements doc; the
implementation details are below. Source:
`docs/research/mkdocstrings.md`.

**Pinned versions** in the `[docs]` extra:

```toml
docs = [
    "mkdocs>=1.6,<2",                  # 2.0 was a hostile rewrite
    "mkdocs-material>=9.7,<10",
    "mkdocstrings[python]>=0.27",
    "griffe-pydantic>=1.3",            # pydantic v2 field-table rendering
    "mkdocs-gen-files>=0.5",
    "mkdocs-literate-nav>=0.6",
    "mkdocs-section-index>=0.3",
]
```

**Pydantic v2 rendering** uses `griffe-pydantic`, which adds
dedicated Fields / Validators / Config sections and respects
`inherited_members` for the `TFTConfig <- BaseModelConfig <- BaseTrainingConfig`
chain. Plain mkdocstrings-python renders fields as ordinary
attributes and surfaces neither validators nor `Field(description=...)`
metadata cleanly.

**Site layout**:

```
docs/
  index.md                           rewritten from README quickstart
  requirements.md                    hosted as-is
  architecture.md                    hosted as-is
  observability.md                   F11 event reference
  research/*.md                      hosted as a "Decisions" section
  guides/
    getting_started.md
    panel_data.md
    calibration.md
    optuna_search.md
    hardware_and_precision.md
  api/                               auto-generated by mkdocstrings + literate-nav
  examples/                          rendered from .py via mkdocs-gen-files
```

`mkdocs build --strict` gates on the `validation:` block (nav
omitted / not_found / absolute, links not_found / anchors / absolute
/ unrecognized) AND on plugin warnings including unresolved
mkdocstrings cross-references.

**mkdocstrings options recipe** (matches FastAPI's settings, adjusted
for this library):

```yaml
plugins:
  - mkdocstrings:
      handlers:
        python:
          options:
            show_signature_annotations: true
            show_root_full_path: false
            merge_init_into_class: true
            show_source: false
            inherited_members: true
            members_order: source
            docstring_style: google
            extensions:
              - griffe_pydantic
```

## A13: Performance baseline plan

Resolves Q15. v1 ships baselines for two public cells:

- **`(cpu-x86, torch-latest)`**: GitHub Actions `ubuntu-latest`
  runner. Free; covers every contributor PR.
- **`(t4, torch-latest)`**: Google Colab T4 captured by a nightly
  self-hosted runner. Cheapest Turing-tier CUDA option still widely
  available.

Optional contributor cells: `(ampere-a10, torch-latest)`,
`(hopper-h100, torch-latest)`. Baselines stored at
`tests/perf/_baselines/<cell>.json` with median + P95 step time,
peak memory, and inference latency per N7.

**Regression gate**: 15% on median step time, 10% on peak memory.
The CPU-cell regression fires on every PR but does NOT block merge
in v1 (the 5-minute PR-CI budget cannot absorb a full perf run); it
appears as a nightly alert. v2 may add a fast-cell gate.

## A14: Testing architecture

**`tests/conftest.py`** owns:

- The shared `SyntheticPanelGenerator` fixtures (function-scoped for
  mutation-sensitive cases, session-scoped for read-only large
  panels). The generator accepts `periods_per_entity=(min, max)` so
  one fixture call produces mixed-lookback panels; the
  variable-history e2e fixture pins `(1, 60)` and explicitly injects
  at least one 1-row entity and one 60-row entity to satisfy the F6
  coverage requirement without relying on sampling variance.
- `propagate_seq_sklearn_logger` (autouse) asserts
  `logging.getLogger("seq_sklearn").propagate is True` and attaches a
  handler that feeds `caplog`. F11-event-emission tests silently
  catch zero records without this.
- `strict_mode_globals` (function-scoped, autouse on
  `tests/unit/training/test_determinism.py`) snapshots
  `torch.are_deterministic_algorithms_enabled()`,
  `torch.backends.cudnn.deterministic`,
  `torch.backends.cudnn.benchmark`, and
  `os.environ.get("CUBLAS_WORKSPACE_CONFIG")` at setup, restores at
  teardown. `pytest-randomly` permutes test order; without
  restoration, Scenario B's preconditions become non-deterministic.
- The hypothesis profile registration:
  - `inner_loop` (default): `settings(deadline=2000, max_examples=50,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])`.
  - `nightly` (`HYPOTHESIS_PROFILE=nightly`):
    `settings(deadline=None, max_examples=500,
    suppress_health_check=[HealthCheck.too_slow,
    HealthCheck.data_too_large, HealthCheck.filter_too_much])`.
  - CI fails if neither profile is loaded.
- The `check_estimator` subset constants. sklearn 1.6 uses
  `parametrize_with_checks(estimators, expected_failed_checks=...)`,
  not the legacy `_xfail_checks` dict on the estimator. Source:
  `docs/research/sklearn.md`.

**Snapshot CI workflow** (resolves the snapshot enforcement deferral).
The GHA workflow `pr.yml` injects the PR author's type into the
shell environment because GitHub Actions does NOT expose
`$GITHUB_ACTOR_TYPE` as a built-in env var (the built-ins are
`$GITHUB_ACTOR` for the login string and `$GITHUB_TRIGGERING_ACTOR`;
author type is workflow-context-only). The injection in `pr.yml`:

```yaml
# .github/workflows/pr.yml (snapshot-guard job)
snapshot-guard:
  runs-on: ubuntu-latest
  env:
    PR_USER_TYPE: ${{ github.event.pull_request.user.type }}
  steps:
    - uses: actions/checkout@v4
      with: { fetch-depth: 0 }
    - run: bash scripts/check_snapshots.sh
```

`scripts/check_snapshots.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

SNAPSHOT_GLOB="tests/_snapshots/"
changed=$(git diff --name-only origin/main...HEAD)

# Only run further checks if snapshot files are part of the PR diff.
if ! echo "$changed" | grep -q "^${SNAPSHOT_GLOB}"; then
    exit 0
fi

# Bot author rejection regardless of marker.
if [ "${PR_USER_TYPE:-}" = "Bot" ]; then
    echo "FAIL: bot-authored PR modifying snapshots is not allowed"
    exit 1
fi

# If snapshot files changed alongside non-snapshot source files,
# require an explicit review marker in at least one commit message.
if echo "$changed" | grep -vq "^${SNAPSHOT_GLOB}"; then
    commit_msgs=$(git log origin/main..HEAD --format=%B)
    if ! echo "$commit_msgs" | grep -q "^SNAPSHOT_REVIEWED:"; then
        echo "FAIL: snapshot files modified without SNAPSHOT_REVIEWED: marker"
        exit 1
    fi
fi
```

The script inspects EVERY commit in the PR (`git log
origin/main..HEAD`), not just HEAD, so a mid-PR snapshot churn cannot
slip past. The pre-commit hook duplicates the marker check against
`git diff --cached --name-only` and `git log -1 --format=%B HEAD` to
catch the issue at commit time before it reaches CI.

**NaN-loss Variant B isolation** (resolves the deferral): the test
uses a function-scoped `tft_classifier_fresh` fixture that constructs
a fresh `TFTClassifier` and discards it after the test. Inf-injection
into model weights uses `.clone()` snapshotting before mutation and
restores in `finally`, so the model object is safe even if accidentally
shared across tests under a different scoping.

**Quickstart CI acceptance** (resolves the deferral):
`tests/e2e/test_quickstart.py` exercises the README quickstart end
to end with `seed=42, dgp_version=1` and asserts the binary classifier
hits the N1 threshold (accuracy >= 0.75). Not a smoke test. README
and test stay in sync because the test imports the same example file
the docs include.

**Test directory names**: `tests/_snapshots/` holds pinned artifacts
(matches the requirements doc N1 wording); `tests/snapshot/` holds
test code that consumes them. The two names are similar but
distinct; reviewer awareness is sufficient. The earlier proposal to
rename `_snapshots/` was rejected to avoid a doc-divergence from the
authoritative requirements doc.

## A15: Logging and observability

```python
# src/seq_sklearn/logging.py
from enum import Enum

class Event(str, Enum):
    TRAIN_GRAD_NORM = "train.grad_norm"
    TRAIN_EPOCH = "train.epoch"
    TRAIN_VAR_SELECTION_ENTROPY = "train.var_selection_entropy"
    TRAIN_ATTENTION_ENTROPY = "train.attention_entropy"
    TRAIN_HIDDEN_NORM = "train.hidden_norm"            # v3 emission path; xfail in v1 test
    TRAIN_NAN_STEP_SKIPPED = "train.nan_step_skipped"
    TRAIN_MIXED_PRECISION_DIVERGED = "train.mixed_precision_diverged"
    CALIBRATION_FIT = "calibration.fit"
    CALIBRATION_SMALL_SET = "calibration.small_set"
    OPTUNA_TRIAL_PRUNED = "optuna.trial_pruned"
    DATA_DUPLICATE_FLOOR_BREACH_COUNT = "data.duplicate_floor_breach_count"
    DATA_UNSEEN_CATEGORIES = "data.unseen_categories"
    HARDWARE_DETECT = "hardware.detect"

def emit(
    logger: logging.Logger,
    event: Event,
    level: int = logging.INFO,
    **payload: object,
) -> None:
    """Emit a structured record. The record carries `event` and
    `payload` inside `extra=`; this works with `caplog` without
    needing a custom LogRecord factory."""
```

Payload schemas live in `docs/observability.md` as the
single-source reference, mirroring the requirements F11 table. Each
F11 event has a required test in N1 that asserts emission + payload
keys via `caplog`. The `train.hidden_norm` event is recurrent-only
and carries `pytest.mark.xfail(strict=True)` in v1 since no
recurrent concrete model ships.

**Backbone-to-LightningModule instrumentation contract.** The
entropy events (`train.var_selection_entropy`,
`train.attention_entropy`) require the backbone's forward pass to
expose the source tensors to the LightningModule. The base
contract isolates the LightningModule from family-specific
introspection field names:

```python
# src/seq_sklearn/models/_backbone.py
@dataclass
class BackboneOutput:
    """Family-agnostic backbone output. Concrete families subclass
    this and add introspection fields. Generic training plumbing
    sees only `representation` and `padding_mask` plus the dict
    returned by `compute_training_metrics`. Plain `@dataclass`
    (not `Protocol`) so concrete families inherit via standard
    dataclass subclassing and pyright strict mode passes without
    `@runtime_checkable` ceremony."""
    representation: Tensor                   # (B, hidden_size)
    padding_mask: Tensor                     # (B, L); True = padding (ignore)

class BaseBackbone(nn.Module, ABC):
    @abstractmethod
    def forward(self, batch: dict[str, Tensor]) -> BackboneOutput: ...

    def compute_training_metrics(
        self, output: BackboneOutput
    ) -> dict[str, object]:
        """Return event-payload dicts keyed by F11 event name. Base
        returns {} so a backbone with no introspection (e.g. v3's
        recurrent base before any concrete model lands) emits
        nothing. Concrete backbones override to return one entry
        per event."""
        return {}
```

Transformer-family backbones (TFT in v1, PatchTST / TimesNet / TST
in v2) extend the dataclass:

```python
# src/seq_sklearn/models/transformer/_backbone.py
@dataclass
class TransformerBackboneOutput(BackboneOutput):
    var_selection_weights: Tensor            # (B, L, n_vars); softmaxed per timestep
    attention_weights: Tensor                # (B, n_heads, L, L); post-softmax, pre-V
    static_var_selection_weights: Tensor     # (B, n_static_vars); softmaxed across static
```

v3 recurrent backbones extend with `RecurrentBackboneOutput`
carrying `hidden_states` and a `var_selection_weights` field that
may be zero-valued if the model has no VSN.

`TFTBackbone.compute_training_metrics(output)` reduces the four
introspection tensors to the F11 payloads, applying the
`padding_mask` so padded timesteps do not corrupt the metrics:

```python
# src/seq_sklearn/models/transformer/tft/backbone.py
def compute_training_metrics(
    self, output: TransformerBackboneOutput
) -> dict[str, object]:
    mask = output.padding_mask                          # (B, L); True = padding
    valid = (~mask).float()                             # (B, L); 1 at valid timesteps
    valid_count = valid.sum().clamp_min(1.0)            # scalar; >= 1 per the mask
                                                        # invariant in A6

    # static_entropy: VSN softmaxes across n_static_vars per batch element;
    # no time axis so no mask is needed.
    sw = output.static_var_selection_weights            # (B, n_static_vars)
    static_h = -(sw * sw.clamp_min(1e-12).log()).sum(dim=-1)  # (B,)
    static_entropy = static_h.mean().item()

    # temporal_entropy: per-timestep VSN softmax across n_vars; padded
    # rows are zeroed BEFORE softmax (per A6 VSN spec), so post-softmax
    # they are uniform (max entropy). We mask them out of the mean.
    tw = output.var_selection_weights                   # (B, L, n_vars)
    temporal_h = -(tw * tw.clamp_min(1e-12).log()).sum(dim=-1)  # (B, L)
    temporal_entropy = ((temporal_h * valid).sum() / valid_count).item()

    # entropy_per_head: attention softmax across keys. Padded queries
    # produce zero attention rows (the nan_to_num pass in the
    # interpretable path; the SDPA fast path is not used here since
    # the introspection tensor is populated only on the interpretable
    # path). Mask padded queries out of the head-wise mean.
    aw = output.attention_weights                       # (B, H, L, L)
    attn_h = -(aw * aw.clamp_min(1e-12).log()).sum(dim=-1)  # (B, H, L)
    # broadcast mask over heads
    valid_h = valid.unsqueeze(1)                        # (B, 1, L)
    per_head = (attn_h * valid_h).sum(dim=(0, 2)) / valid_count
    entropy_per_head = per_head.tolist()                # list[float], length H

    return {
        "train.var_selection_entropy": {
            "static_entropy": static_entropy,
            "temporal_entropy": temporal_entropy,
        },
        "train.attention_entropy": {
            "entropy_per_head": entropy_per_head,
        },
    }
```

`_LightningModule.training_step` stashes the most recent
`BackboneOutput` on `self._last_train_output`;
`on_train_epoch_end` calls
`self.backbone.compute_training_metrics(self._last_train_output)`
and emits one event per returned key. The LightningModule never
reads family-specific attribute names; v3 recurrent models override
`compute_training_metrics` to emit `train.hidden_norm` (and
optionally `train.var_selection_entropy` if their VSN is non-trivial)
without touching `_LightningModule` code.

```python
# src/seq_sklearn/training/_lightning_module.py
def on_train_epoch_end(self) -> None:
    if self._last_train_output is None:
        return                                          # no successful batch this epoch
    payloads = self.backbone.compute_training_metrics(self._last_train_output)
    for event_name, payload in payloads.items():
        emit(self._logger, Event(event_name), **payload)
    # ... (other emissions: train.epoch, train.grad_norm, etc.)
```

Four named tests pin this contract. They share a common naming
convention with the plan to avoid the cross-doc drift the prior
Gemini pass surfaced:

- `test_on_train_epoch_end_skips_entropy_when_no_output` asserts
  `caplog` contains no entropy records when
  `_last_train_output is None` (e.g. the epoch ended with no
  successful training batches).
- `test_on_train_epoch_end_emits_events_from_compute_metrics`
  uses a `_DummyBackbone` subclass that overrides
  `compute_training_metrics` to return one synthetic payload
  (`{"train.var_selection_entropy": {"static_entropy": 1.0,
  "temporal_entropy": 0.5}}`); asserts `caplog` contains exactly
  one record with `record.event == "train.var_selection_entropy"`
  and the expected payload keys. Pins the
  `for event_name, payload in payloads.items(): emit(...)`
  delegation loop at the LightningModule unit level.
- `test_compute_training_metrics_ignores_padded_positions`
  constructs a `TransformerBackboneOutput` with uniform-distribution
  rows at padded timesteps and asserts the returned
  `temporal_entropy` and `entropy_per_head` equal hand-computed
  values on the unpadded slice (proving the time-axis mask is
  applied). Also asserts `static_entropy` is identical to the
  no-mask reference (proving the static branch correctly skips
  the mask).
- `test_base_backbone_compute_training_metrics_returns_empty`
  asserts `BaseBackbone.compute_training_metrics` defaults to `{}`
  so a v3 recurrent backbone that overrides nothing emits no events.

Under `accumulate_grad_batches > 1`, `_last_train_output` reflects
the FINAL micro-batch only, not an aggregate. v1 treats this as
representative sampling and documents the limitation in the
`docs/observability.md` payload reference. True per-epoch
aggregation across all training micro-batches is a v2 refinement
(non-blocking, but tracked).

## A15.1: AttentionOutput / RegressionAttentionOutput

`predict_with_attention` returns one of two frozen dataclasses
(BETA per the requirements stability table; field set may grow in
MINOR releases; tuple-unpacking is NOT supported). v1 fields:

```python
@dataclass(frozen=True, slots=True)
class AttentionOutput:
    """Returned by TFTClassifier.predict_with_attention."""
    predictions: np.ndarray                    # (N,) class indices or (N, K) for v1.1 multi-label logits
    probabilities: np.ndarray                  # (N, num_classes) post-softmax/sigmoid
    logits: np.ndarray                         # (N, num_classes); pre-activation
    var_selection_weights: np.ndarray          # (N, L, n_vars)
    static_var_selection_weights: np.ndarray   # (N, n_static_vars)
    attention_weights: np.ndarray              # (N, n_heads, L, L)
    padding_mask: np.ndarray                   # (N, L); True = padding (pass-through from preprocessing)
    entity_id: np.ndarray                      # (N,) for diagnostics

@dataclass(frozen=True, slots=True)
class RegressionAttentionOutput:
    """Returned by TFTRegressor.predict_with_attention."""
    predictions: np.ndarray                    # (N,) point or (N, len(quantiles)) when quantile mode
    quantiles_used: tuple[float, ...] | None   # the fit-time quantile vector, or None for point regression
    var_selection_weights: np.ndarray          # (N, L, n_vars)
    static_var_selection_weights: np.ndarray   # (N, n_static_vars)
    attention_weights: np.ndarray              # (N, n_heads, L, L)
    padding_mask: np.ndarray                   # (N, L); True = padding
    entity_id: np.ndarray                      # (N,) for diagnostics
```

Regression intentionally has no `logits` field. The classifier head
emits logits (pre-activation) and `AttentionOutput.logits` exposes
them for inspection; the regression head emits raw scalars (the
prediction itself, or per-quantile predictions), so the
`predictions` field already carries what `logits` would carry on a
classifier. Adding a `logits` field would be either a duplicate of
`predictions` or a leak of internal pre-projection state; both are
worse than the current contract.

Default return type is CPU `np.ndarray` for callable convenience;
the `device=` keyword on `predict_with_attention` flips to
on-device `Tensor` per A20 item 2.

A unit test asserts `dataclasses.fields(AttentionOutput)` matches
the v1 enumeration exactly; a MINOR release adding a field would
break the test and force a deliberate snapshot bump.

## A16: Optuna integration

```python
# src/seq_sklearn/tuning/suggest_params.py
def suggest_params(
    trial: optuna.Trial,
    model_class: type[BaseSequenceClassifier | BaseSequenceRegressor],
    base: BaseModelConfig | None = None,
) -> BaseModelConfig:
    """Sample a config from the per-model default search space.

    Closed under the F5 validity matrix by construction: samples
    task_type first (or reads from `base`), then samples each
    downstream field from the legal subset for that task type. A
    1000-iteration unit test using `optuna.create_study().ask()`
    asserts every sampled config passes the cross-field validator
    (FixedTrial would not actually sample the search space)."""
```

```python
# src/seq_sklearn/tuning/pruning.py
@contextmanager
def optuna_trial_guard(trial: optuna.Trial) -> Iterator[None]:
    """Wraps the user's `objective(trial)` body. Catches
    ConfigError and TrainingError and re-raises as optuna.TrialPruned
    with the original message. DataContractError, KeyboardInterrupt,
    and unexpected exceptions propagate so the study fails fast on
    genuine bugs.

    Default `study.optimize(catch=())` means non-TrialPruned
    exceptions terminate the study; the user is responsible for
    wrapping the objective with this context manager."""
```

Recommended objective shape:

```python
def _config_to_estimator_kwargs(config: BaseModelConfig) -> dict[str, object]:
    """Convert a pydantic config dump into the flat double-underscore
    kwargs an estimator's `__init__` accepts. The TFTConfig's
    nested `tabular_config` field becomes a `TabularConfigParams`
    adapter; other fields pass through unchanged."""
    raw = config.model_dump()
    tabular_dict = raw.pop("tabular_config")
    return {**raw, "tabular_config": TabularConfigParams(**tabular_dict)}

def objective(trial: optuna.Trial) -> float:
    with optuna_trial_guard(trial):
        config = suggest_params(trial, TFTClassifier, base=BASE_CONFIG)
        model = TFTClassifier(**_config_to_estimator_kwargs(config))
        model.fit(
            X_train, y_train,
            calibration_set=(X_cal, y_cal),
            optuna_trial=trial,           # threaded via fit, NOT __init__
        )
        return model.score(X_val, y_val)
```

`_config_to_estimator_kwargs` is documented here (not exported)
because the round-trip between a frozen pydantic dump and the
mutable BaseEstimator-adapter kwargs is non-obvious. v2 / v3
estimators with their own nested config adapters add similar
helpers; the shape stays the same.

The trial reaches `_LightningModule` via `fit`, not via the pydantic
config. `BaseSequenceEstimator.fit` accepts `optuna_trial:
optuna.Trial | None = None` as a keyword argument and forwards it
to the Trainer, which threads it into the LightningModule's
constructor. This preserves `BaseModelConfig`'s `extra="forbid"`
contract (Phase 7's test that unknown kwargs raise stays valid) and
keeps the trial out of `get_params` / `set_params` /
`save` / `load` (where a serialized `Trial` would be nonsense and a
pickle hazard).

**Pruning hook integration.** The Trainer passes `optuna_trial` to
`_LightningModule`'s constructor. `on_validation_epoch_end` pulls the
metric from `self.trainer.callback_metrics` (epoch-aggregated, not
last-batch), calls `trial.report(value, step=current_epoch)`, and
records the prune decision in `self._pending_prune` per A7's
deferred-raise pattern; `on_train_epoch_end` raises
`optuna.TrialPruned` at the END of the hook so the `train.epoch`
and entropy events still fire for the pruned epoch. The N1 pruning
test uses `MedianPruner(n_startup_trials=0, n_warmup_steps=0,
n_min_trials=1)` and asserts trial 2 prunes at epoch 0;
`n_min_trials=1` is mandatory or the test becomes flaky per
`docs/research/optuna.md`.

**`optuna-integration.PyTorchLightningPruningCallback` is NOT
shipped.** The upstream callback raises `TrialPruned` from
`on_validation_end`, which Lightning fires BEFORE
`on_train_epoch_end`; that skips the `train.epoch` and entropy
log events for the pruned epoch (the lifecycle bug A7's deferred-
raise pattern was designed to prevent). The library's native
`_pending_prune` machinery is the only supported path. The
`optuna-integration` package is still installed (its other
utilities may be referenced by callers) but
`PyTorchLightningPruningCallback` is not imported into the library
and not exposed in any example.

**Testing `suggest_params`.** Use a real `optuna.Study` via
`study.ask()` so each sampled trial draws from the search space.
`FixedTrial` is deterministic and only returns pre-loaded param
values, so a 1000-iteration `FixedTrial` loop would not actually
exercise the F5 validity matrix's downstream sampling logic. The
unit test pattern:

```python
def test_suggest_params_sweep_respects_validity_matrix():
    study = optuna.create_study()
    for _ in range(1000):
        trial = study.ask()
        config = suggest_params(trial, TFTClassifier)
        check_combo(
            config.task_type, config.loss_strategy,
            config.imbalance_strategy, config.calibration_strategy,
        )  # raises ValueError on any illegal cell
        study.tell(trial, 0.0)  # value irrelevant for this test
```

`FixedTrial` remains the right tool for testing
`optuna_trial_guard` (where the wrapped body should see `report` /
`should_prune` as no-ops without standing up a Study) and for any
test where the trial values are explicitly known.

Imports (from `docs/research/optuna.md`):

```python
import optuna
# No PyTorchLightningPruningCallback import; see the deferred-raise
# note above.
```

Pin: `optuna>=4.4,<5`, `optuna-integration>=4.4,<5` (the integration
package is installed for transitive dependency hygiene; the
specific pruning callback is not imported into seq-sklearn).

## A17: Save / load format

`save(path)` writes a directory at `path/` containing:

- `path/weights.safetensors`: tensor-only archive in the safetensors
  format. Holds the state dicts (backbone, head, any calibrator
  tensor state) plus tensorizable fit-state (`classes_`,
  `n_features_in_`, `n_outputs_`, `quantiles_` as a 1D tensor,
  `decision_threshold_` as a 0D tensor when present).
- `path/state.json`: human-readable JSON. Holds the pydantic config
  dump (`model_dump`), `feature_names_in_` (list of strings), the
  `tabular_to_sequence_state` (categorical-encoder vocabularies as
  arrays of strings; scaler statistics as floats), the calibrator's
  `serialize()` output, and the metadata block.

```python
def save(self, path: str | Path) -> None:
    path = Path(path); path.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file
    save_file(self._collect_tensors(), str(path / "weights.safetensors"))
    (path / "state.json").write_text(json.dumps(self._collect_state(), indent=2))

@classmethod
def load(cls, path: str | Path) -> "BaseSequenceEstimator":
    path = Path(path)
    from safetensors.torch import load_file
    weights = load_file(str(path / "weights.safetensors"))
    state = json.loads((path / "state.json").read_text())
    return cls._reconstruct(weights, state)   # UserWarning on version mismatch
```

No `torch.load` call anywhere in the save / load path (the library
never opts out of `weights_only=True`). No `trust=True` /
`weights_only=False` escape hatch exists in v1; a future model that
requires pickled state would be a MINOR bump with an explicit security
note. Source: `docs/research/pytorch.md`.

**Metadata block** (inside `state.json`):

```json
{
    "seq_sklearn_version": "...",
    "torch_version": "...",
    "cuda_version": "..." | null,
    "python_version": "...",
    "precision_resolved": "...",
    "feature_schema_fingerprint": "...",
    "created_at": "...",
    "schema_version": 1
}
```

**Schema versioning and migrations.**

```python
# src/seq_sklearn/serialization.py
Migration = Callable[[dict, dict], tuple[dict, dict]]

CURRENT_SCHEMA_VERSION: int = 1
OLDEST_SUPPORTED_SCHEMA_VERSION: int = 1   # v1 supports only itself
assert OLDEST_SUPPORTED_SCHEMA_VERSION <= CURRENT_SCHEMA_VERSION, \
    "OLDEST_SUPPORTED_SCHEMA_VERSION must be <= CURRENT_SCHEMA_VERSION"

# Registry: (from_version, to_version) -> migration function.
# Empty in v1; first entry lands when v1.1 needs to migrate v1 saves.
MIGRATIONS: dict[tuple[int, int], Migration] = {}

def _migrate(weights: dict, state: dict) -> tuple[dict, dict]:
    """Step the (weights, state) pair forward through MIGRATIONS
    until state['schema_version'] == CURRENT_SCHEMA_VERSION. Raises
    PredictionError if no path exists."""
    src = state.get("schema_version", 0)
    if src > CURRENT_SCHEMA_VERSION:
        raise PredictionError(
            f"checkpoint schema {src} newer than library "
            f"v1 (max {CURRENT_SCHEMA_VERSION}); upgrade seq-sklearn"
        )
    if src < OLDEST_SUPPORTED_SCHEMA_VERSION:
        raise PredictionError(
            f"checkpoint schema {src} older than oldest supported "
            f"({OLDEST_SUPPORTED_SCHEMA_VERSION})"
        )
    while src < CURRENT_SCHEMA_VERSION:
        try:
            step = MIGRATIONS[(src, src + 1)]
        except KeyError:
            raise PredictionError(
                f"no migration registered from schema {src} to {src + 1}"
            )
        weights, state = step(weights, state)
        post = state.get("schema_version", src)
        if post <= src:
            raise PredictionError(
                f"migration step ({src}, {src + 1}) did not advance "
                f"schema_version (got {post}, expected > {src}); the "
                f"migration callable must mutate state['schema_version'] "
                f"to a strictly larger value before returning"
            )
        src = post
    return weights, state
```

`_reconstruct` calls `_migrate` before consuming the dicts. The N1
save/load round-trip test exercises the no-op identity path
(v1 -> v1, `MIGRATIONS` is empty so `_migrate` short-circuits).

**Meta-test (`test_migrations_advance_schema_version`).** Because
`MIGRATIONS` is empty in v1, a "for every key in MIGRATIONS, assert
schema_version advances" test would be vacuously true and fail to
catch a future no-op registration. The meta-test instead exercises
`_migrate` with a TEST-LOCAL `MIGRATIONS` override:

```python
def test_migrate_detects_no_op_registration(monkeypatch):
    from seq_sklearn import serialization
    bad = {(1, 2): lambda w, s: (w, s)}    # forgets to bump schema_version
    monkeypatch.setattr(serialization, "MIGRATIONS", bad)
    monkeypatch.setattr(serialization, "CURRENT_SCHEMA_VERSION", 2)
    with pytest.raises(PredictionError, match="schema 1 to 2"):
        serialization._migrate({}, {"schema_version": 1})
```

`_migrate` advances `state["schema_version"]` by exactly one per
step, BUT the loop terminates only when `src == CURRENT_SCHEMA_VERSION`.
A no-op step would not advance `src`, the loop would spin forever.
The implementation MUST bound the loop by `CURRENT_SCHEMA_VERSION
- src + 1` iterations and raise `PredictionError("migration step
(X, Y) did not advance schema_version")` if the post-step value is
unchanged. This invariant is what the meta-test catches.

## A18: Dependencies and version pins

Concrete `pyproject.toml`:

```toml
[project]
name = "seq-sklearn"
requires-python = ">=3.12,<3.15"
dependencies = [
    "torch>=2.6,<3",
    "lightning>=2.6.1,<2.7",      # 2.6.2/2.6.3 yanked after PyPI compromise 2026-04-30
    "pydantic>=2.12,<3",          # 2.11.3 joblib regression; 2.12 fixes frozen pickle
    "pandas>=2.2",
    "numpy>=1.26",
    "scikit-learn>=1.6,<2",       # Tags API + __sklearn_tags__ instance method
    "optuna>=4.4,<5",
    "optuna-integration>=4.4,<5",
    "safetensors>=0.5",
]

[project.optional-dependencies]
onnx = ["onnx>=1.18", "onnxruntime>=1.21"]
mlflow = ["mlflow>=2.18"]
wandb = ["wandb>=0.19"]
docs = [
    "mkdocs>=1.6,<2",             # MkDocs 2.0 is a hostile rewrite
    "mkdocs-material>=9.7,<10",
    "mkdocstrings[python]>=0.27",
    "griffe-pydantic>=1.3",
    "mkdocs-gen-files>=0.5",
    "mkdocs-literate-nav>=0.6",
    "mkdocs-section-index>=0.3",
]
dev = [
    "ruff>=0.7",
    "pyright>=1.1.390",
    "pytest>=8.3",
    "pytest-cov>=6.0",
    "pytest-randomly>=3.16",
    "pytest-rerunfailures>=15.0",
    "pytest-benchmark>=5.0",
    "hypothesis>=6.120",
    "pre-commit>=4.0",
]
```

Upper-bound rationales mirror the N3 requirements text.

## A19: CI workflow

`.github/workflows/pr.yml` (under 5-minute target per N2). All six
jobs declare no `needs:` dependencies in the workflow YAML, so GHA
runs them in parallel. Wall-clock is `max` of the per-job times.

```
lint:            ruff check + ruff format --check         (~10s)
type:            pyright (strict mode)                    (~30s)
test-unit:       pytest -m "not slow and not perf and not gpu"
                 with --cov gate                          (~120s)
test-deploy:     build wheel + install in clean venv +
                 minimal fit/predict                      (~60s)
docs:            mkdocs build --strict                    (~10s)
snapshot-guard:  scripts/check_snapshots.sh               (~5s)
```

Wall-clock equals `max(10, 30, 120, 60, 10, 5) = 120s` (~2 minutes),
dominated by `test-unit`. Well under the 5-minute N2 budget. The
docs build is fast (`docs/research/mkdocstrings.md` order-of-magnitude
estimate ~3s on a 20-module library; v1 ships fewer modules so the
build runs faster still).

`.github/workflows/nightly.yml`:

```
full-matrix:     {linux, macos, windows} x {3.12, 3.13, 3.14}
perf:            pytest -m perf with regression gate
                 against tests/perf/_baselines/cpu-x86.json
gpu:             self-hosted T4 when available
```

## A20: Open questions for the implementation phase

These open points arise during implementation, not during the
architecture phase:

1. **OneCycleLR `total_steps` derivation**: needs `max_epochs *
   steps_per_epoch` up-front. Lightning supports both
   `epochs/steps_per_epoch` and `total_steps`. Pick at implementation
   time based on what plays nicely with `accumulate_grad_batches`.
2. **Variable-selection-weight return shape for
   `predict_with_attention`**: CPU numpy array vs. on-device tensor.
   Default plan: CPU numpy for callable convenience, on-device on
   opt-in via a `device=` argument; pin during implementation.
3. **Native ONNX `Attention` op (opset 23)**: PyTorch issue #149662
   tracks landing the native op. Not stable as of torch 2.12. Watch
   list; once it ships, we can drop the math-backend forcing in the
   ONNX-export path.
4. **`griffe-pydantic` v1.3 rendering of `Mapping[str, int]`**: the
   `categorical_embed_dims` field uses `Mapping` for hashability;
   confirm rendering at implementation time. If it falls back to
   plain `dict` rendering, the doc field stays useful but less
   structured.
5. **`resume_path` user-facing surface**: F5 promises restore of
   model weights, optimizer state, scheduler state, and RNG state
   from a checkpoint. A7's `RngStateCallback` handles the RNG side.
   **RESOLVED**: per requirements F5,
   `BaseSequenceEstimator.__init__(..., resume_path: str | Path |
   None = None, ...)`. The estimator stores `resume_path` as an
   instance attribute and forwards it to the Trainer at `fit` time
   via `pl_trainer.fit(ckpt_path=resume_path)`. `clone()` semantics
   are clean (a clone with the same `resume_path` resumes from the
   same checkpoint deterministically). `fit` does NOT accept
   `resume_path` as a keyword.

## Addressed

Round 1 (design-review swarm):

- **Snapshot CI shell-script bug (arch r1-C1 / qa r1-I6).** Replaced
  the non-existent `$GITHUB_ACTOR_TYPE` env var with explicit
  injection in `pr.yml` (`env: PR_USER_TYPE: ${{ github.event.pull_request.user.type }}`)
  and updated `scripts/check_snapshots.sh` to read `$PR_USER_TYPE`.
- **`compute_three_way_split` signature (arch r1-C2 / qa r1-I3).**
  Added `window_time_index: np.ndarray` parameter; the function now
  uses it to identify the last `cal_fraction` windows per entity
  without relying on input pre-sort. Documented the return value
  for `calibration_set_provided=True` (empty cal_idx).
- **Save/load migration framework (arch r1-C3).** Added a
  `MIGRATIONS: dict[tuple[int, int], Callable]` registry inside
  `serialization.py` with explicit `_migrate` dispatch, schema-too-new
  / schema-too-old error paths, and a meta-test for accidental no-op
  registrations.
- **A20 expansion (arch r1-C4).** Added two more open items
  (`EntityTimeSeriesSplit` semantics, `resume_path` user-facing
  surface) so the architecture phase is complete on its declared
  deliverable list.
- **`_LightningModule` test fixture pattern (qa r1-C1).** Specified
  the `make_test_module` helper that stubs `module._trainer =
  MagicMock(...)` directly to let `on_validation_epoch_end` access
  `self.trainer.callback_metrics` without a real `pl.Trainer`.
- **NaNLossGuard callback hook (qa r1-C2).** Pinned to
  `on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)`
  with `outputs` carrying the scalar loss when `automatic_optimization=True`.
- **GradScalerWatchdog CPU testability (qa r1-C3).** Specified the
  `hasattr(trainer.precision_plugin, "scaler")` guard and the
  no-op-on-CPU behavior, plus the mocked-scaler test path.
- **AttentionOutput / RegressionAttentionOutput field enumeration
  (qa r1-C4).** New A15.1 section with frozen-dataclass field lists
  and a field-set unit test.
- **MedianPruner `n_min_trials=1` discrepancy (qa r1-C5).** Updated
  the requirements doc N1 pruning test to include `n_min_trials=1`
  with a rationale note; architecture and requirements now align.
- **Clone-safe nested dict (arch r1-I1).** Added explicit
  `copy.deepcopy` requirement in `__init__` for the nested-config
  attribute so `sklearn.base.clone` produces an independent
  estimator under joblib threading.
- **Shared-V tensor reshape walkthrough (arch r1-I2).** Spelled out
  the 4-step reshape contract in A6 with explicit shape comments.
- **`_LightningModule` construction order (arch r1-I3).** Added a
  5-step construction-sequence block in A7 documenting how the
  Estimator builds `config_`, curries `optimizer_factory` /
  `scheduler_factory`, and passes them to the LightningModule.
- **ConformalCalibrator TrainingError contract (arch r1-I4).** A9
  now states `ConformalCalibrator.fit` raises `TrainingError` on
  non-monotone quantiles, with the test input shape pinned to
  `(N, len(quantiles))`.
- **A3 INTERNAL-tier note (arch r1-I5).** Added a sentence noting
  the recurrent-skeleton symbols are intentionally absent from the
  public re-export list.
- **CI wall-clock math (arch r1-I6).** Clarified that all six jobs
  run in parallel; wall-clock is `max(...) = 120s`. Rewrote the
  prior "Total wall-clock under 4 minutes" sentence which was
  ambiguous.
- **Snapshot directory naming collision (arch r1-I7).** Rolled back
  the `tests/snapshot_data/` rename; architecture now matches the
  authoritative requirements doc on `tests/_snapshots/`.
- **`compute_three_way_split` empty-cal_idx return (arch r1-I8).**
  Documented the return shape when `calibration_set_provided=True`
  and when `calibration_strategy="none"` + `threshold_tuning=False`.
- **detect / resolve_precision combined test (qa r1-I1).** A11
  docstring now states the N1 parametrized test exercises BOTH
  functions in sequence per tier, not independently.
- **Backbone-to-LightningModule entropy hooks (qa r1-I2).** Added
  the `BackboneOutput` dataclass contract in A15 (originally a NamedTuple; refactored to `@dataclass` in the post-Gemini pass): backbone
  forward returns `(representation, var_selection_weights,
  attention_weights, static_var_selection_weights)`, and the
  LightningModule stashes the most recent output for entropy
  computation in `on_train_epoch_end`.
- **`optuna_trial_guard` TrainingError test (qa r1-I4).** Updated
  the requirements N1 pruning entry to add a TrainingError variant
  alongside the ConfigError variant.
- **Shared-V correctness test (qa r1-I5).** A6 walkthrough now
  declares the test that counts V-projection weights (1, not H).
- **Classifier/Regressor mixin shape (arch r1-NITPICK 1).**
  Clarified as nested classes on `TransformerSequenceEstimator`
  with an example `class TFTClassifier(TransformerSequenceEstimator.Classifier, BaseSequenceClassifier)`.
- **Style line 1233 / line 1247 (style r1-I1, r1-I2).** Rewritten
  in place during the CI wall-clock and A20 edits.

## Deferred

Round 1 (design-review swarm):

- **`griffe-pydantic` v1.3 `Mapping[str, int]` rendering check
  (arch r1-NITPICK 2).** Easier to verify in 30 seconds against the
  v1.3 release than to research now; kept in A20 item 4 for the
  implementation phase.
- **`pack_padded_sequence` bit-equality on B>1 (qa r1-NITPICK 2).**
  The mask-correctness test fixture runs at B=1 by default per N1;
  BLAS reordering on B>1 with mixed batch sizes is a known PyTorch
  quirk that the test explicitly avoids. Documented; no contract
  change needed.
- **`enable_strict_mode` cuBLAS env var subtlety (qa r1-NITPICK 3).**
  The test asserts env-var value, not cuBLAS behavior. Documented
  in the test rationale; the in-process limitation is by design.
- **A4 numbered-list scaffolding (style r1-NITPICK 2).** Parallel
  structure is intentional for a spec doc.
- **Bullet rationale tightening at line 881
  (style r1-NITPICK 1).** Minor; parenthetical kept as-is.

Round 2 (design-review swarm):

- **Shared-V interpretable score capture (arch r2-I1).** A6 now
  documents two forward paths on the attention module: a fast SDPA
  path for fit / predict / predict_proba and a manual-softmax path
  for predict_with_attention that materializes `attn_weights
  (B, H, L, L)` post-softmax pre-V. The shared-V correctness test
  asserts output equality between the two paths.
- **Entropy payload shapes (arch r2-I2 / r2-I3).** A15 now matches
  the F11 payload schema: `static_entropy` (scalar),
  `temporal_entropy` (scalar), `entropy_per_head` (list of length
  `n_heads`). The reduction axes are spelled out in code.
- **`_last_train_output` under grad accumulation (arch r2-I4).**
  Documented: under `accumulate_grad_batches > 1`, the stashed
  output reflects the final micro-batch only. Treated as
  representative sampling in v1; v2 may add proper aggregation.
- **`validation_step` self.log requirement (arch r2-I5).** A7's
  `validation_step` stub now carries the inline contract
  "MUST call self.log(self.val_metric_name, loss, on_step=False,
  on_epoch=True)" so `trainer.callback_metrics` populates for the
  Optuna pruning hook.
- **EntityTimeSeriesSplit semantics (arch r2-I6).** Promoted out of
  A20 into a new A9.1 section. `gap` measured in WINDOWS (matching
  sklearn TimeSeriesSplit). Drop-with-warning rule for entities
  shorter than `n_splits + 1 + gap` windows.
- **`MIGRATIONS` meta-test reachability (qa r2-I1).** A17 now
  describes `test_migrate_detects_no_op_registration`: monkeypatches
  a test-local MIGRATIONS dict with a no-op step, asserts
  `_migrate` detects non-advancement and raises
  `PredictionError("migration step (X, Y) did not advance
  schema_version")`. The implementation is required to bound the
  loop and detect non-advancement.
- **`on_train_epoch_end` None-guard test (qa r2-I2).** A15 names
  `test_on_train_epoch_end_skips_entropy_when_no_output`; the
  None-guard branch is exercised explicitly.
- **`RegressionAttentionOutput.logits` intentionality (qa r2-I3).**
  A15.1 documents that the regression case intentionally omits
  `logits`: the regression head emits raw scalars, and
  `predictions` carries what `logits` would for a classifier.
- **Migration alias ordering (arch r2-N2).** Reordered the
  `Migration = Callable[...]` line above `CURRENT_SCHEMA_VERSION`
  in the snippet.
- **Schema-version invariant assert (arch r2-N3).** Added
  `assert OLDEST_SUPPORTED_SCHEMA_VERSION <= CURRENT_SCHEMA_VERSION`
  at module scope.
- **`RegressionAttentionOutput.static_var_selection_weights` shape
  annotation (qa r2-N1).** Added the missing `# (N, n_static_vars)`
  comment.

Round 3 (design-review swarm):

- **`_migrate` snippet / meta-test contract alignment (arch r3-I1).**
  Added the non-advancement guard to the `_migrate` code snippet
  itself; the snippet now raises `PredictionError("migration step
  (X, Y) did not advance schema_version; ...")` if the post-step
  `state["schema_version"]` is unchanged. The
  `test_migrate_detects_no_op_registration` meta-test now exercises
  visible behavior rather than a prose-only invariant.

Gemini final pass (cross-family review on the implementation plan; surfaced architecture-level issues):

- **`_LightningModule` hardcoded transformer-specific attributes (gemini r1-C1, surfaced via impl-plan review).** A15's entropy emission code indexed `var_selection_weights`, `attention_weights`, and `static_var_selection_weights` directly on `self._last_train_output`. v3 `RecurrentBackboneOutput` (carrying `hidden_states` instead of `attention_weights`) would `AttributeError` on every recurrent training epoch, forcing a rewrite of the generic training plumbing. Refactored A15 to a Protocol + delegation: `BackboneOutput` is a Protocol carrying only `representation` and `padding_mask`; `BaseBackbone` declares `compute_training_metrics(output)` returning `{event_name: payload}` with a default empty implementation. `TFTBackbone` overrides the method. The LightningModule reads only the Protocol surface plus the returned dict. Transformer-family backbones extend the Protocol via `TransformerBackboneOutput`; recurrent-family backbones will extend via their own concrete output type without touching LightningModule code.
- **Entropy reductions dropped the padding mask (gemini r1-C2).** A15's prior reductions (`.mean(dim=-1).mean()` and `.mean(dim=(0, 2))`) averaged across padded timesteps which carry max-entropy uniform VSN rows and zero attention rows by construction (per A6's VSN zeroing + the interpretable-attention `nan_to_num` safety net). The result inflated `temporal_entropy` and deflated `entropy_per_head` by amounts that scale with the padding fraction (severe for short-tenure entities). A15 now adds `padding_mask` to the base `BackboneOutput` Protocol and the reduction is a masked mean (sum over valid positions divided by valid count). Three named tests cover: padded-position mask application, the `_last_train_output is None` early return, and the `BaseBackbone` default-empty return.
- **Optuna trial polluting pydantic config (gemini r1-C3).** A16's prior objective example dumped the trial into the pydantic params dict before calling `TFTClassifier(**...)`. That contradicted A4's `extra="forbid"` contract (Phase 7's test expects unknown kwargs to raise). Moved the trial out of `__init__` into the `fit` keyword: `BaseSequenceEstimator.fit(X, y, *, calibration_set=None, optuna_trial=None)`. The trial reaches `_LightningModule` via the Trainer; it is never serialized, never enters `get_params`, and cannot break `extra="forbid"`. Requirements F1's fit signature updated to match. `resume_path` stays on `__init__` per requirements F5 (A20 item 5 is now resolved that way).
- **`PyTorchLightningPruningCallback` removed (gemini r1-C4).** The upstream callback raises `TrialPruned` from `on_validation_end`, before `on_train_epoch_end` fires, defeating the `_pending_prune` deferred-raise pattern A7 introduced. A16 now explicitly states the callback is NOT shipped. The library's native pruning hook in `_LightningModule.on_validation_epoch_end` (which stashes the prune decision) plus `on_train_epoch_end` (which raises at the END after logging fires) is the only supported path.
- **`FixedTrial` sweep is vacuous (gemini r1-I1).** A16's prior test recommendation used a 1000-iteration `FixedTrial` sweep to assert validity-matrix conformance. `FixedTrial` only returns pre-loaded param values; it does not sample distributions, so the sweep would be vacuous without externally-supplied randomized param dicts. Replaced with `optuna.create_study().ask()` in the test pattern. `FixedTrial` remains correct for `optuna_trial_guard` tests where the trial's `report` / `should_prune` must be no-ops without standing up a Study.

Gemini final pass (cross-family review on architecture doc; prior pass):

- **Pydantic nested-config pattern corrected (gemini r1-C1).** A4
  step 3 originally specified the flatten-into-dict pattern, which
  `docs/research/sklearn.md` explicitly REJECTS (duplicates schema,
  loses validation grouping). Rewrote A4 to use the recommended
  BaseEstimator-adapter pattern: a thin `TabularConfigParams(BaseEstimator)`
  with `to_pydantic()` method, the outer estimator stores the
  adapter instance, sklearn's `get_params(deep=True)` recurses
  automatically, `set_params(tabular_config__lookback=6)` chains via
  the standard sklearn double-underscore traversal. Replaced the
  `copy.deepcopy` clone-safety note with `sklearn.base.clone`
  recursion since the adapter is itself a `BaseEstimator`.
- **`EntityTimeSeriesSplit` lookback history loss (gemini r1-C2).**
  The splitter's test fold previously yielded disjoint indices from
  train; `cross_val_score` would call
  `TabularToSequence.transform` on the test fold in isolation, so
  test entities lost their historical context and got heavily
  left-padded. Added a `lookback` constructor argument; the split
  function now extends the test fold LEFT by `lookback - 1` rows
  per entity from the preceding train segment. Train and test
  overlap by `lookback - 1` history-only rows. Added a
  `from_estimator` helper that reads the lookback off an estimator
  to remove the foot-gun.
- **Manual softmax NaN safety net (gemini r1-C3).** Added
  `torch.nan_to_num(attn_weights, nan=0.0)` to the interpretable
  attention path. The mask invariant guarantees this should never
  fire, but the defensive zeroing costs one O(BHLL) pass and
  prevents an upstream mask bug from silently corrupting the
  `AttentionOutput.attention_weights` payload.
- **Migration loop strict monotonicity (gemini r1-I1).** Changed
  the `_migrate` invariant check from `if post == src` to
  `if post <= src` so a faulty migration that mutates
  `schema_version` BACKWARDS raises immediately rather than
  spinning the loop forever.
- **Pruning order vs. logging events (gemini r1-I2).** Lightning
  2.6 fires `on_validation_epoch_end` BEFORE `on_train_epoch_end`;
  raising `optuna.TrialPruned` from the validation hook would skip
  the `train.epoch` / entropy structured-log events for the pruned
  epoch. Refactored to stash the prune decision on
  `self._pending_prune` during `on_validation_epoch_end` and raise
  at the END of `on_train_epoch_end`, so logging always fires
  before the prune aborts the run.
- **Context-vector prose ordering (gemini r1-N1).** Reordered the
  prose list of the four static-context vectors to put `c_h` before
  `c_c`, matching the `nn.LSTM(input, (h_0, c_0))` tuple order used
  in code. Annotated each entry with its `nn.LSTM` signature
  position.
- **Precision literal `"32"` → `"32-true"` (gemini r1-C3 carried
  from requirements pass).** Updated A4's `BaseTrainingConfig` Literal,
  A7's `Trainer(deterministic=True)` gate, A11's `resolve_precision`
  return type. Same Lightning 2.6 API mismatch as the requirements
  doc fix.
- **`os.cpu_count() or 1` (gemini r1-I1 carried).** Updated A4 and
  A7 references to add the None-handling fallback.
