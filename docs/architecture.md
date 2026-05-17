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
    _validity.py                    F5 validity-matrix cross-field validator (check_combo)
    _extras.py                      ExtraDict, _normalize_extras, _PROMOTED_KEYS_BY_FAMILY, extract_deprecated_extras (hyperparameter strategy)
    _adapters.py                    BaseEstimator adapters per pydantic config (TabularConfigParams, OptimizerParams, SchedulerParams, LossParams, SamplerParams, <Model>AdvancedParams)
    base.py                         BaseTrainingConfig, BaseModelConfig
    optimizer.py                    OptimizerConfig (family sub-config)
    scheduler.py                    SchedulerConfig (family sub-config)
    loss.py                         LossConfig (family sub-config)
    sampler.py                      SamplerConfig (family sub-config)
    tabular.py                      TabularToSequenceConfig
    tft.py                          TFTConfig + TFTAdvancedConfig
    recurrent.py                    RecurrentSequenceEstimatorConfig (v1 skeleton, INTERNAL)

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
        _estimator.py               _TFTEstimatorMixin (shared A4 __init__ / _config_kwargs / _build_tft_backbone)
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
    callbacks.py                    GradScalerWatchdog, EventEmitter, RngStateCallback
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
    suggest_params.py               suggest_params(trial, model_class, *, search_advanced, search_extras)
    pruning.py                      optuna_trial_guard context manager
    _alpha_keys.py                  curated per-family ALPHA-key enum lists (empty in v1)
    _config_to_estimator_kwargs.py  _config_to_estimator_kwargs + _ADAPTER_MAP_BY_CONFIG registry
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
`BaseSequence*` class. The TFT family additionally leads the MRO with
`_TFTEstimatorMixin` (in `models/transformer/tft/_estimator.py`), which
owns the shared A4-adapter `__init__`, `_config_cls=TFTConfig`,
`_config_kwargs`, and `_build_tft_backbone` so the ~30-param
constructor is not duplicated across the classifier and regressor (the
same single-source discipline as `data.splits.window_time_index`).
Example:
`class TFTClassifier(_TFTEstimatorMixin, TransformerSequenceEstimator.Classifier, BaseSequenceClassifier): ...`
The mixin overrides template methods the base class calls;
`super().__init__` resolves cooperatively through the composed MRO to
`BaseSequenceEstimator.__init__`.

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

**Authoritative source for hyperparameter exposure**: this section
documents the v1 estimator-config plumbing (frozen pydantic + sklearn
adapter pattern + validity matrix). A4 below and requirements F7 are
authoritative for the four-tier exposure architecture (family
sub-configs → main configs → advanced sub-configs → `extra` escape
hatch), the deprecation-alias contract, and the per-model default
search space (the last via A16 / the Phase 8 `suggest_params`
implementation); the code carries the verbatim schemas.
`docs/hyperparameter_strategy.md` holds the design rationale (why the
four-tier shape) and the living ALPHA → BETA → STABLE promotion
procedure only; it is not authoritative for the schemas or contracts.

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
   `set_params` protocol. Under the four-tier hyperparameter
   architecture (F7 / A4), the v1 TFT estimator stores SIX
   adapter instances (one per nested pydantic sub-config), each
   following the same pattern:

   - `tabular_config: TabularConfigParams` ← `TabularToSequenceConfig`
   - `optimizer: OptimizerParams` ← `OptimizerConfig`
   - `scheduler: SchedulerParams` ← `SchedulerConfig`
   - `loss: LossParams | None` ← `LossConfig` (defaults to None; the
     estimator injects `_DEFAULT_LOSS_FOR_TASK[task_type]` at
     `_build_config` time when omitted)
   - `sampler: SamplerParams` ← `SamplerConfig`
   - `advanced: TFTAdvancedParams` ← `TFTAdvancedConfig`

   Every adapter is a thin `BaseEstimator` whose fields mirror the
   pydantic schema 1:1 and exposes `to_pydantic()` constructing the
   frozen instance. Every adapter `__init__` uses the `*` keyword-
   only marker so adding a BETA field via the ALPHA → BETA promotion
   path does NOT shift positional arguments (the promotion contract
   is MINOR-additive only; without keyword-only the promotion is
   silently MAJOR-breaking).

   ```python
   # src/seq_sklearn/config/_adapters.py
   class TabularConfigParams(BaseEstimator):
       def __init__(
           self,
           *,                              # mandatory keyword-only
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

   # OptimizerParams, SchedulerParams, LossParams, SamplerParams,
   # TFTAdvancedParams follow the same shape (keyword-only init
   # + to_pydantic). Their pydantic schemas are the family
   # sub-configs specified later in this A4 section and carried
   # verbatim in src/seq_sklearn/config/.

   # The real class composes the shared TFT mixin + the Phase-6b
   # transformer mixin + the Phase-6a family shell; the __init__ body
   # below lives on _TFTEstimatorMixin (see A2 / models/transformer/tft).
   class TFTClassifier(
       _TFTEstimatorMixin,
       TransformerSequenceEstimator.Classifier,
       BaseSequenceClassifier,
   ):
       def __init__(
           self,
           *,                              # mandatory keyword-only
           task_type: Literal["binary", "multiclass", ...],
           tabular_config: TabularConfigParams | None = None,
           optimizer: OptimizerParams | None = None,
           scheduler: SchedulerParams | None = None,
           loss: LossParams | None = None,
           sampler: SamplerParams | None = None,
           advanced: TFTAdvancedParams | None = None,
           hidden_size: int = 128,
           # ... TFTConfig flat fields (model-shape only) mirrored
       ) -> None:
           # NOTE (Phase 6a Addressed): the sklearn.base.clone(...) calls
           # below are SUPERSEDED. They break sklearn.base.clone (which
           # rejects a constructor that modifies a param and already
           # deep-clones nested params). __init__ stores params verbatim
           # with the None-default-instance idiom; see the Phase 6a
           # ledger entry.
           self.task_type = task_type
           self.tabular_config = (
               sklearn.base.clone(tabular_config)
               if tabular_config is not None else TabularConfigParams()
           )
           self.optimizer = (
               sklearn.base.clone(optimizer)
               if optimizer is not None else OptimizerParams()
           )
           # ... (one clone-protected assignment per adapter)
           self.loss = sklearn.base.clone(loss) if loss is not None else None
           # ... etc.
   ```

   sklearn's `get_params(deep=True)` recurses through every adapter
   automatically (each adapter is a `BaseEstimator`), producing the
   canonical double-underscore flat keys
   (`tabular_config__lookback`, `optimizer__learning_rate`,
   `loss__strategy`, `scheduler__warmup_steps`, etc.).
   `set_params(optimizer__learning_rate=3e-4)` chains via standard
   sklearn traversal:
   `self.optimizer.set_params(learning_rate=3e-4)`. The frozen
   pydantic instance is built inside `fit` as
   `self.config_ = self._build_config()`, where `_build_config`
   reads from each adapter's `to_pydantic()` and threads the
   task-type-aware loss default if `self.loss is None`. The
   pydantic validity-matrix validator runs inside the nested
   `BaseModelConfig` construction; failures wrap into `ConfigError`
   at the `_build_config` call site (step 4 below).

   **Task-type-aware loss default**:

   ```python
   # src/seq_sklearn/models/_base.py inside _build_config:
   _DEFAULT_LOSS_FOR_TASK: dict[str, str] = {
       "binary": "cross_entropy",
       "multiclass": "cross_entropy",
       "multilabel": "cross_entropy",            # v1.1
       "regression_point": "mse",
       "regression_quantile": "pinball",
       "regression_multioutput": "mse",          # v1.1
   }

   def _build_config(self) -> BaseModelConfig:
       loss = self.loss
       if loss is None:
           loss = LossParams(
               strategy=_DEFAULT_LOSS_FOR_TASK[self.task_type]
           )
       # ... build the nested BaseModelConfig with loss.to_pydantic()
       # and the other adapter instances' to_pydantic() outputs.
   ```

   `LossConfig.strategy` has no default at the pydantic layer (legal
   value depends on `task_type` per F5); the injection here preserves
   the ergonomic `TFTClassifier(task_type="binary").fit(X, y)` call
   while keeping the schema strict. v1.1 entries in the map are
   present so v1.1 enablement is one-line later, but they are
   unreachable in v1: the F5 validity matrix (`check_combo`) rejects
   any v1.1 `task_type` with a "scheduled for v1.1" `ValidationError`
   before `_build_config` runs. Phase 1's
   `test_v1_task_type_rejects_multilabel_and_regression_multioutput`
   pins this guard.

   **Clone safety**: `sklearn.base.clone(estimator)` calls
   `type(estimator)(**estimator.get_params(deep=False))`. The
   shallow-params dict contains the SAME adapter instances by
   reference for every nested adapter field (`tabular_config`,
   `optimizer`, `scheduler`, `loss`, `sampler`, `advanced`). The
   outer `__init__` defends against aliasing by calling
   `sklearn.base.clone` on each incoming adapter before storing:
   `self.optimizer = sklearn.base.clone(optimizer) if optimizer
   is not None else OptimizerParams()` (and the analogous pattern
   for each of the six adapter slots). `sklearn.base.clone`
   recursively constructs a fresh adapter instance from the
   original's params; this is the sklearn-idiomatic alternative to
   `copy.deepcopy` and works under both joblib `prefer='threads'`
   and `prefer='processes'` (the joblib-process path pickles each
   estimator independently, so adapter aliasing collapses by
   construction). Phase 1's `test_adapters.py` asserts each
   adapter's clone produces an independent instance: six named
   per-adapter tests (one for each of `TabularConfigParams`,
   `OptimizerParams`, `SchedulerParams`, `LossParams`,
   `SamplerParams`, `TFTAdvancedParams`) plus
   `test_outer_estimator_clone_does_not_alias_adapter_instances`
   for the end-to-end contract.
4. **Cross-field validators wrap `pydantic.ValidationError` into
   `ConfigError`** at the `_build_config()` call site inside `fit`,
   not inside the validator itself (the validator stays a pure
   `@model_validator(mode="after")` returning the model).

**Family sub-configs** (per F7; the load-bearing field-name surface,
specified here and carried verbatim in `src/seq_sklearn/config/`):

```python
# src/seq_sklearn/config/optimizer.py
class OptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: Literal["adamw", "adam", "sgd"] = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    momentum: float = 0.9
    nesterov: bool = False
    extra: ExtraDict = ()

# src/seq_sklearn/config/scheduler.py
class SchedulerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: Literal["constant", "cosine_with_warmup", "one_cycle", "reduce_on_plateau"] = "cosine_with_warmup"
    warmup_steps: int = 100
    pct_start: float = 0.3
    div_factor: float = 25.0
    final_div_factor: float = 1e4
    plateau_factor: float = 0.5
    plateau_patience: int = 5
    plateau_threshold: float = 1e-4
    min_lr: float = 0.0
    extra: ExtraDict = ()

# src/seq_sklearn/config/loss.py
class LossConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    # No default: legal value depends on task_type per F5;
    # the estimator injects _DEFAULT_LOSS_FOR_TASK[task_type] when
    # the caller omits the LossParams adapter.
    strategy: Literal["cross_entropy", "focal", "mse", "mae", "huber", "pinball"]
    focal_gamma: float = 2.0
    focal_alpha: float | None = None
    huber_delta: float = 1.0
    label_smoothing: float = 0.0
    extra: ExtraDict = ()

# src/seq_sklearn/config/sampler.py
class SamplerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    strategy: Literal["none", "class_weighted", "oversample_minority", "undersample_majority"] = "none"
    oversample_ratio: float = 1.0
    replacement: bool = True
    extra: ExtraDict = ()
```

**`BaseTrainingConfig`** (cross-cutting + nested family sub-configs):

```python
class BaseTrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_size: int = 64
    max_epochs: int = 50
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
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
```

**`BaseModelConfig`** (shared across all concrete models):

```python
class BaseModelConfig(BaseTrainingConfig):
    task_type: Literal[
        "binary", "multiclass", "multilabel",
        "regression_point", "regression_quantile", "regression_multioutput",
    ]
    loss: LossConfig                         # nested; no default per F5
    sampler: SamplerConfig = Field(default_factory=SamplerConfig)
    calibration_strategy: Literal[
        "none", "temperature", "platt", "isotonic",
        "conformal", "isotonic_quantile",
    ] = "none"
    threshold_tuning: bool = False
    threshold_metric: Literal["f1", "balanced_accuracy", "youden_j"] = "f1"
    quantiles: tuple[float, ...] | None = None

    @model_validator(mode="after")
    def _check_validity_matrix(self) -> Self:
        from seq_sklearn.config._validity import check_combo
        check_combo(self.task_type, self.loss.strategy,
                    self.sampler.strategy, self.calibration_strategy)
        return self

    @model_validator(mode="after")
    def _check_quantiles_monotone(self) -> Self:
        if self.quantiles is None:
            return self
        q = self.quantiles
        if any(not (0.0 < v < 1.0) for v in q):
            raise ValueError(f"quantiles must lie in (0, 1); got {q}")
        if any(q[i] >= q[i + 1] for i in range(len(q) - 1)):
            raise ValueError(f"quantiles must be strictly increasing; got {q}")
        return self
```

The validity-matrix validator's call site reads `self.loss.strategy`
and `self.sampler.strategy` from the nested family configs;
`check_combo` itself still takes four strings (signature unchanged).

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
    categorical_embed_dims: CategoricalEmbedDims = ()  # see _normalize_embed_dims
    model_config = ConfigDict(extra="forbid", frozen=True)
```

`categorical_embed_dims` is stored as a sorted
`tuple[tuple[str, int], ...]` via a `BeforeValidator` that accepts
`dict` / `Mapping` / `tuple-of-tuples` input. This keeps the frozen
model hashable; the original `Mapping[str, int]` claim from earlier
drafts did not deliver hashability in pydantic v2 (a plain `dict`
in `__dict__` defeats `hash()`).

**`TFTConfig`**:

```python
class TFTConfig(BaseModelConfig):
    hidden_size: int = 128
    attention_heads: int = 4
    dropout: float = 0.1
    variable_selection_dropout: float = 0.1
    prediction_readout: Literal["last_valid", "mean_pool"] = "last_valid"
    tabular_config: TabularToSequenceConfig
    advanced: TFTAdvancedConfig = Field(default_factory=TFTAdvancedConfig)


class TFTAdvancedConfig(BaseModel):
    """BETA per requirements stability tiers; fields here may change
    defaults or be renamed without a MAJOR bump. Empty in v1 plus the
    extra escape hatch; populated as benchmark testing identifies
    needle-movers per the promotion path in docs/hyperparameter_strategy.md."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    extra: ExtraDict = ()
```

`TFTClassifier.__init__` accepts each nested sub-config as a single
adapter kwarg (`tabular_config: TabularConfigParams | None = None`,
`optimizer: OptimizerParams | None = None`, etc.) plus the
TFT-specific model-shape fields (`hidden_size`, `attention_heads`,
etc.) as flat kwargs. `get_params(deep=True)` recurses into every
adapter automatically, yielding double-underscore flat keys like
`tabular_config__lookback`, `optimizer__learning_rate`,
`loss__strategy`. `set_params(optimizer__learning_rate=3e-4)`
chains via the standard sklearn double-underscore traversal into
`self.optimizer.set_params(learning_rate=3e-4)`. The
`clf__optimizer__learning_rate` triple-underscore form chains
through `Pipeline` cleanly for the same reason.

`GridSearchCV(estimator=TFTClassifier(task_type="binary"),
param_grid={"optimizer__learning_rate": [1e-4, 3e-4, 1e-3]})` works
without further plumbing.

**`ExtraDict` escape hatch** (per F7). Every family
sub-config carries an `extra: ExtraDict` field. `ExtraDict` is
`tuple[tuple[str, ExtraValue], ...]` after the `BeforeValidator`
normalizes input; `ExtraValue` is restricted to
`str | int | float | bool | None` so JSON round-trip is type-
identical. Non-primitive values raise `TypeError` at construction.
The `_normalize_extras` validator, the `_PROMOTED_KEYS_BY_FAMILY`
registry, and the `extract_deprecated_extras` helper live in
`src/seq_sklearn/config/_extras.py`; the family factories
(`build_optimizer`, `build_scheduler`, `build_loss`, `build_sampler`)
route through `extract_deprecated_extras` instead of `dict(cfg.extra)`
so ALPHA → BETA promotion is one registry edit. The reserved-keys
collision check (preventing `extra=(("lr", 0.1),)` from colliding
with the typed `OptimizerConfig.learning_rate` kwarg at
`torch.optim.AdamW(...)` construction) lives at the CONFIG layer
as a `@model_validator(mode="after")` on each family sub-config,
NOT at the factory call site, so the validation fires at
`OptimizerConfig(...)` construction time and Phase 1 owns the test.
The factories (Phase 4) trust the validated configs and do not
re-check. Reserved-key sets keyed by `cfg.name`:
`_RESERVED_BY_OPTIMIZER["adamw"] = {"params", "lr", "weight_decay",
"betas", "eps"}` (analogous for `adam` and `sgd`); untyped torch
kwargs pass through `extra` unrestricted.

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
      AND cal_fraction > 0. cal_idx holds the tail cal_fraction windows
      per entity; it is empty only in the degenerate case where every
      entity is too short for round(cal_fraction * m) to reach 1, in
      which case a UserWarning is emitted.
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
  Empty-side rule: a side with zero variables (no categoricals and no
    reals, reachable per F2 where static/tv reals are optional) gets
    one synthetic learned variable (a zero-init nn.Parameter) so the
    VSN dimensionality is always >= 1. Its selection softmax is then a
    constant 1.0 and its var-selection entropy a constant 0; F11
    consumers should expect 0 entropy on a padded side, not a bug.
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
  The F3 data contract left-pads (padding is a leading block); pack
  consumes the LEADING `length` steps, so each row is gathered
  valid-first (stable argsort on the mask) before packing and scattered
  back to its original positions (inverse permutation) after. This
  left-pad -> pack bridge is the canonical handling; it depends on F3
  keeping padding a contiguous leading block.

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

  (b) Interpretable path (TFTBackbone always uses this): replaces SDPA
      with a manual softmax to capture the per-head score tensor.
      scores = (q @ k.transpose(-2, -1)) / sqrt(d)  # (B, H, L, L)
      scores = scores.masked_fill(~attn_mask_bool[:, None, None, :], -inf)
      attn_weights = scores.softmax(dim=-1)         # (B, H, L, L); post-softmax, pre-V
      # NaN safety: softmax over an all-(-inf) key row produces NaN.
      # nan_to_num zeros such a row so a fully-masked query contributes
      # nothing instead of poisoning v_broadcast. The A6
      # mask.any(dim=1).all() check keeps this off the TFTBackbone path;
      # direct callers of forward_interpretable are still covered. One
      # O(BHLL) pass.
      attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
      out_per_head = attn_weights @ v_broadcast     # (B, H, L, d)
      out = out_per_head.mean(dim=1)
      out = out_proj(out)
      attn_weights is returned alongside the output for TransformerBackboneOutput.

  TFTBackbone always routes through (b): F11 attention-entropy metrics
  consume attn_weights every training step, so fit / predict /
  predict_proba / predict_with_attention all use the interpretable
  path. The fast path (a) skips capture entirely and is reserved for
  future family backbones that do not expose per-step attention
  weights. The shared-V correctness test asserts equality of outputs between
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

**Model-boundary contract.** `TFTBackbone.forward` validates its
batch dict at the boundary and raises `PredictionError` for any of
four preconditions; a Phase 4 estimator that assembles batches from a
fitted `TabularToSequence` satisfies all four by construction (F3
scaling yields finite reals, the per-column encoder yields in-range
codes, F3 left-pads), so these guard a direct caller that bypasses
Phase 2:

1. NaN or inf in `time_varying_real` / `static_real` ->
   `PredictionError("<name> contains NaN or inf; ...")`.
2. A categorical index outside `[0, cardinality + 1)` (the embedding
   table width, `<unk>` at 0) ->
   `PredictionError("<name> column <i> has an index outside ...")`.
3. A `padding_mask` that is not a contiguous leading block (interior
   or leading-valid padding breaks the left-pad -> pack gather) ->
   `PredictionError("padding_mask is not a contiguous leading block; ...")`.
4. A window with zero valid timesteps (all padding) ->
   `PredictionError("window had zero valid timesteps after preprocessing ...")`.

Construction-time precondition: `InterpretableMultiHeadAttention`
raises `ValueError` if `hidden_size % n_heads != 0`. On the TFT path
the `TFTConfig` validator already enforces this, so it only guards a
direct v2-family construction of the shared-V attention module.

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
    recurrent_dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    recurrent_dropout_kind: Literal["weight_drop", "variational", "bernoulli"] = "weight_drop"
    hidden_init_strategy: Literal["zero", "learned", "per_entity"] = "zero"
    readout: Literal["last_valid", "mean_pool", "attention"] = "last_valid"
    bptt_window: int | None = Field(default=None, ge=1)
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
2a. Drops below-`min_periods_predict` windows from `train_idx` /
   `val_idx` via `data.splits.below_floor_mask` (the shared helper the
   estimator's calibration fold / predict path also uses). Those
   windows carry sentinel targets (`-1` classification / `NaN`
   regression) from `transform`; training on them raises in
   `_class_weights` (`torch.bincount(-1)`) or trips the F9
   non-finite-loss abort. If every train / val window is below-floor
   the Trainer raises `ConfigError` (symmetric with the estimator's
   empty-calibration-fold guard) rather than handing Lightning an
   empty loader.
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
     save_top_k=1)`, `GradScalerWatchdog` (mixed precision only),
     `EventEmitter`, `RngStateCallback`. The F9 non-finite-loss skip
     is NOT a callback (see below). Source: `docs/research/lightning.md`.
   - logger: pass-through. The library does NOT register a default
     logger; Lightning auto-attaches `TensorBoardLogger` unless
     explicitly suppressed. The Trainer passes `logger=False` by
     default; callers attach `MLFlowLogger` / `WandbLogger` / etc.
     manually.
5. Wraps the model in `_LightningModule`.
6. Calls `pl_trainer.fit(lightning_module, train_loader, val_loader)`.
7. Returns the fitted `_LightningModule`. Calibration is NOT the
   Trainer's job: `BaseSequenceEstimator.fit` builds the calibrator and
   runs the threshold tuner on the recomputed calibration fold
   (`_fit_calibrator` / `_post_fit`) after `Trainer.fit` returns.

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
        optuna_trial: optuna.trial.BaseTrial | None = None,
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
        self._pending_prune: tuple[int, float] | None = None
        self._last_train_output: BackboneOutput | None = None
        self._consecutive_nan = 0
        self.automatic_optimization = True   # v1 stays on automatic
        # F9 non-finite-loss skip + 3-step abort live in training_step
        # (a post-hoc callback fires after optimizer.step()).

    def training_step(self, batch, batch_idx) -> Tensor | None:
        # forward + loss. F9: if the loss is non-finite (NaN OR inf),
        # emit train.nan_step_skipped and RETURN None so Lightning
        # skips backward + optimizer.step() (no gradient update); the
        # 3rd consecutive non-finite step raises TrainingError. On a
        # finite step: reset the counter, store
        # self._last_train_output = backbone_out, self.log("train_loss"),
        # return the loss.
        # on_before_optimizer_step emits train.grad_norm (F11, DEBUG,
        # every step: step/grad_norm/lr). on_train_epoch_end emits
        # train.epoch with the full F11 payload
        # (epoch/train_loss/val_loss/val_metric from callback_metrics).
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
3. The `Trainer` wrapper builds the curried factories. Each factory
   takes its nested family sub-config (per the F5 bridge), not the
   root config, and the scheduler factory must also bind `monitor`
   and `total_steps` so the residual matches the
   `Callable[[optim.Optimizer], dict]` `_LightningModule` type:
   `optimizer_factory = partial(build_optimizer, config=self.config_.optimizer)` and
   `scheduler_factory = partial(build_scheduler, config=self.config_.scheduler,
   monitor=self._val_metric_name, total_steps=self._total_steps)`
   (`total_steps` is the accumulation-adjusted optimizer-step count;
   `None` for non-step schedulers).
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

- **F9 non-finite-loss skip (in `_LightningModule.training_step`, NOT
  a callback).** F9 requires a single NaN/inf loss step to skip with
  no gradient update. Under `automatic_optimization=True` the only
  mechanism that actually skips backward + `optimizer.step()` is
  returning `None` from `training_step`; a `Callback.on_train_batch_end`
  fires AFTER the optimizer step and cannot skip it (this was a
  Gemini-final-pass CRITICAL: the earlier `NaNLossGuard` callback
  counted but never skipped, so F9's "no gradient update" was
  violated). `training_step` therefore checks
  `torch.isfinite(loss).all()`: on a non-finite loss it increments
  `self._consecutive_nan`, emits `train.nan_step_skipped` at WARNING
  with the F11 payload (`step`, `consecutive_nan_count`), and returns
  `None`; the third consecutive non-finite step emits at ERROR with
  `aborting=True` and raises `TrainingError`. A finite step resets the
  counter, stashes `_last_train_output`, logs `train_loss`, and returns
  the loss. `NaNLossGuard` is removed (it could not satisfy F9).
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

**Note on parameter names**: `build_loss` takes string-valued
arguments (`loss_strategy: str`, the value of `loss.strategy`).
Callers in `_LightningModule._configure_loss` extract these from the
frozen pydantic config via the nested access pattern
(`cfg.loss.strategy`, `cfg.sampler.strategy`, etc., per the F5 bridge
table in `docs/requirements.md`). The `build_loss` function's
parameter names use the historical short-form
(`loss_strategy`) for back-compat with the validity-matrix
vocabulary; the call site reads from the nested family configs.

```python
def build_loss(
    task_type: str,
    loss_strategy: str,             # value of cfg.loss.strategy
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
`cfg.sampler.strategy == "class_weighted"` (display label
`imbalance_strategy="class_weighted"` per the F5 bridge). Binary
class-weighting uses
`BCEWithLogitsLoss(pos_weight=neg_count / pos_count)` derived from
the train fold; multiclass uses `CrossEntropyLoss(weight=per_class_weights)`.

## A9: Calibration pipeline

```python
class _Calibrator(Protocol):
    def fit(self, raw_output: Tensor, y_true: Tensor) -> None: ...
    def transform(self, raw_output: Tensor) -> Tensor: ...
    def serialize(self) -> dict[str, object]: ...
    @classmethod
    def deserialize(cls, blob: dict[str, object]) -> "_Calibrator": ...
```

`raw_output` is the model's raw output on the calibration fold: class
logits for a classification calibrator, the predicted-quantile matrix
for a regression calibrator (the argument was named `logits` in the
original draft; renamed in Phase 5 so the regression case is not
misleading). `transform` returns calibrated probabilities
(classification) / calibrated quantile values (regression), never
logits, so the estimator's `predict_proba` / `predict_quantiles`
consumes the return value directly.

Concrete classes:

- `TemperatureScaling` (single scalar T; LBFGS-optimized on cal-set NLL).
- `PlattScaling` (binary only; logistic regression on logits).
- `IsotonicCalibrator` (`sklearn.isotonic.IsotonicRegression` wrapper;
  multiclass fits one regressor per class).
- `ConformalCalibrator` (split-conformal; per-quantile offset).
- `IsotonicQuantileCalibrator` (isotonic on the empirical CDF of
  prediction errors).

Each calibrator is testable standalone with hand-crafted
`(raw_output, y_true)` tensors; the architecture does NOT require
instantiating an Estimator to exercise the calibrator path.
`ConformalCalibrator.fit` raises
`TrainingError("non-monotone quantiles: <details>")` if the
regressor's RAW predicted quantiles are not monotone non-decreasing
across the calibration set (F9 contract). The check runs on the raw
predictions BEFORE the per-quantile recentring offset, since the
offset would otherwise mask the crossing (a per-column offset can
re-center each column independently and hide an undertrained
regressor's quantile inversion). The non-monotone unit test feeds a
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
- `strict_mode_globals` (function-scoped, autouse on EVERY test under
  `tests/unit/training/`) snapshots
  `torch.are_deterministic_algorithms_enabled()`,
  `torch.backends.cudnn.deterministic`,
  `torch.backends.cudnn.benchmark`, and
  `os.environ.get("CUBLAS_WORKSPACE_CONFIG")` at setup, restores at
  teardown. It is NOT scoped to `test_determinism.py` alone:
  `enable_strict_mode` mutates true process globals, so under
  `pytest-randomly` ordering a determinism test running before a
  sibling callback/factory test leaks strict-mode state and the suite
  goes nondeterministically red. The snapshot is four attribute reads;
  the directory-wide isolation guarantee is worth that cost.
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

The override takes the base `BackboneOutput` (not the narrower
`TransformerBackboneOutput`) so it stays Liskov-compatible with
`BaseBackbone.compute_training_metrics`; pyright strict rejects a
contravariant parameter narrowing. It narrows internally with an
`isinstance` check and raises `TypeError` on a non-transformer output,
which is a programming error since this backbone only ever produces a
`TransformerBackboneOutput`:

```python
# src/seq_sklearn/models/transformer/tft/backbone.py
def compute_training_metrics(
    self, output: BackboneOutput
) -> dict[str, object]:
    if not isinstance(output, TransformerBackboneOutput):
        raise TypeError(
            "TFTBackbone.compute_training_metrics requires a "
            f"TransformerBackboneOutput, got {type(output).__name__}"
        )
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

    # entropy_per_head: attention softmax across keys. A padded query
    # that still has valid keys produces a normal (non-zero, non-NaN)
    # distribution, NOT a zero row; only an all-keys-masked query would
    # NaN, which the A6 mask invariant prevents on this path. Padded
    # queries are therefore excluded explicitly via the valid_h mask,
    # not assumed zero. The introspection tensor exists only on the
    # interpretable path (the SDPA fast path is not used here).
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
Gemini pass surfaced. The two `_LightningModule`-side tests
(`test_on_train_epoch_end_skips_entropy_when_no_output`,
`test_on_train_epoch_end_emits_events_from_compute_metrics`) land
with Phase 4: `_LightningModule` does not exist in Phase 3, so they
are deferred to the Phase 4 training-loop work. The two backbone-side
tests (`test_compute_training_metrics_ignores_padded_positions`,
`test_base_backbone_compute_training_metrics_returns_empty`) are
delivered in Phase 3.

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
    predictions: np.ndarray                    # (N,) class indices (map via estimator.classes_); (N, K) v1.1 multi-label
    probabilities: np.ndarray                  # (N, num_classes) post-softmax/sigmoid
    logits: np.ndarray                         # (N, head_out_dim): 1 for binary, num_classes else; pre-activation
    var_selection_weights: np.ndarray          # (N, L, n_vars)
    static_var_selection_weights: np.ndarray   # (N, n_static_vars)
    attention_weights: np.ndarray              # (N, n_heads, L, L)
    padding_mask: np.ndarray                   # (N, L); True = padding (pass-through from preprocessing)
    entity_id: np.ndarray                      # (N,) internal contiguous entity code, for diagnostics

@dataclass(frozen=True, slots=True)
class RegressionAttentionOutput:
    """Returned by TFTRegressor.predict_with_attention."""
    predictions: np.ndarray                    # (N,) point or (N, len(quantiles)) when quantile mode
    quantiles_used: tuple[float, ...] | None   # the fit-time quantile vector, or None for point regression
    var_selection_weights: np.ndarray          # (N, L, n_vars)
    static_var_selection_weights: np.ndarray   # (N, n_static_vars)
    attention_weights: np.ndarray              # (N, n_heads, L, L)
    padding_mask: np.ndarray                   # (N, L); True = padding
    entity_id: np.ndarray                      # (N,) internal contiguous entity code, for diagnostics
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
    *,
    search_advanced: bool = False,
    search_extras: bool = False,
) -> BaseModelConfig:
    """Sample a config from the per-model default search space.

    The default search space (`search_advanced=False`,
    `search_extras=False`) samples ONLY STABLE fields; the per-model
    default search space is defined by the `suggest_params`
    implementation specified in this section (Phase 8 deliverable).
    `search_advanced=True`
    additionally samples fields on the model's `<Model>AdvancedConfig`;
    in v1 those configs are empty so the flag is a no-op for v1.
    `search_extras=True` samples from the curated per-family
    ALPHA-key list in `src/seq_sklearn/tuning/_alpha_keys.py` (empty
    in v1 by design).

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
# Per-model adapter map: nested pydantic sub-config field name -> adapter class.
_TFT_ADAPTER_MAP: dict[str, type[BaseEstimator]] = {
    "tabular_config": TabularConfigParams,
    "optimizer": OptimizerParams,
    "scheduler": SchedulerParams,
    "loss": LossParams,
    "sampler": SamplerParams,
    "advanced": TFTAdvancedParams,
}

# Registry keyed by concrete config class. v2 / v3 estimators register
# their per-model adapter map here when they ship.
_ADAPTER_MAP_BY_CONFIG: dict[
    type[BaseModelConfig], dict[str, type[BaseEstimator]]
] = {
    TFTConfig: _TFT_ADAPTER_MAP,
}


def _adapter_map_for(config_cls: type[BaseModelConfig]) -> dict[str, type[BaseEstimator]]:
    return _ADAPTER_MAP_BY_CONFIG[config_cls]


def _config_to_estimator_kwargs(config: BaseModelConfig) -> dict[str, object]:
    """Convert a pydantic config dump into the kwargs an estimator's
    `__init__` accepts. Every nested sub-config in the model's adapter
    map is popped from the dump and wrapped as the matching adapter
    instance; other fields pass through unchanged.

    v2 / v3 estimators add their own per-model adapter map to
    `_ADAPTER_MAP_BY_CONFIG`; the helper itself is per-model dispatch."""
    raw = config.model_dump(mode="json")  # mode='json' pins the
                                          # extra-tuple shape on disk
    adapter_map = _adapter_map_for(type(config))
    kwargs: dict[str, object] = {}
    for field_name, adapter_cls in adapter_map.items():
        sub_dict = raw.pop(field_name)
        kwargs[field_name] = adapter_cls(**sub_dict)
    return {**raw, **kwargs}

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
helpers. The shape is the same: pop every sub-config dict from
`model_dump(mode="json")`, wrap each in the matching adapter, pass
the rest through. Phase 8's
`test_config_to_estimator_kwargs_round_trips_all_adapters` and
`test_config_to_estimator_kwargs_extra_tuple_type_survives` pin the
helper's behavior (one covers every adapter slot; the second pins
the `extra` tuple round-trip under `mode="json"`).

The trial reaches `_LightningModule` via `fit`, not via the pydantic
config. `BaseSequenceEstimator.fit` accepts `optuna_trial:
optuna.trial.BaseTrial | None = None` as a keyword argument and forwards it
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
            config.task_type, config.loss.strategy,
            config.sampler.strategy, config.calibration_strategy,
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
  format. Holds only the backbone and head state dicts. (Phase 6a
  reconciliation: the fit-state attributes are NOT tensorized, see
  below.)
- `path/state.json`: human-readable JSON. Holds the pydantic config
  dump (`model_dump`), the `__init__` hyperparameter snapshot (so a
  loaded estimator's `get_params` / `sklearn.base.clone` work),
  `feature_names_in_` (list of strings), the
  `tabular_to_sequence_state` (the transformer's `serialize()` output),
  the calibrator's `serialize()` output, the metadata block, and the
  fit-state attributes `classes_` / `n_outputs_` / `quantiles_` /
  `decision_threshold_` as JSON. Phase 6a reconciliation: the F1.1
  draft / the earlier bullet put these in `weights.safetensors` as
  tensors, but `classes_` is a `LabelEncoder` class vector that is
  frequently non-numeric (string labels) and does not tensorize; a
  calibrator's and the transformer's fitted state already live in
  `state.json`, so all non-tensor fit-state is co-located there for one
  coherent JSON contract. `weights.safetensors` is therefore strictly
  the neural weights. Covered by the byte-equal save/load round-trip
  tests.

```python
def save(self, path: str | Path) -> None:
    path = Path(path); path.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file
    save_file(self._collect_tensors(), str(path / "weights.safetensors"))
    (path / "state.json").write_text(json.dumps(self._collect_state(), indent=2))

@classmethod
def load(cls, path: str | Path) -> "BaseSequenceEstimator":
    weights, state = load_weights_and_state(path)  # UserWarning on version mismatch
    obj = cls(task_type=state["task_type"])
    # `loss` is the one adapter __init__ stores verbatim as None (F5
    # task-aware default injection must see "unspecified"); restore a
    # default LossParams before set_params so persisted `loss__*`
    # leaves have an object to route onto instead of failing
    # set_params on None.
    if obj.loss is None and any(k.startswith("loss__") for k in state["hyperparams"]):
        obj.loss = LossParams()
    obj.set_params(**state["hyperparams"])         # restores the 6 adapters + scalars
    # `_collect_state` dumps `config` with exclude={"tabular_config"}
    # (the transformer config is persisted once, as the authoritative
    # flat `tabular_config` key). A `_config_cls` that requires
    # `tabular_config` (the TFT family's `TFTConfig`) must have it
    # merged back before validation; `BaseModelConfig` has no such
    # field so this is a no-op for the base / dummy path.
    config_state = state["config"]
    if "tabular_config" in cls._config_cls.model_fields:
        config_state = {**config_state, "tabular_config": state["tabular_config"]}
    obj.config_ = cls._config_cls.model_validate(config_state)
    obj.transformer_ = TabularToSequence.deserialize(...)
    obj._restore_family_state(state)               # family hook: classes_ / quantiles_ / threshold
    backbone, head = obj._build_backbone_head(obj.config_, obj.transformer_)  # family hook
    # load_state_dict(weights) into backbone/head
    obj.calibrator_ = obj._build_calibrator_from(state["calibrator"])  # family hook
    return obj
```

There is no `_reconstruct` method; `load` inlines the reconstruction.
The Phase-7 override surface is the family hooks named above
(`_restore_family_state`, `_build_backbone_head`,
`_build_calibrator_from`, plus `_config_kwargs` on the build path), not
a single `_reconstruct` seam.

`save_weights_and_state` / `load_weights_and_state` in
`serialization.py` are low-level primitives: they persist and read
`state` verbatim and do NOT synthesize the metadata block.
`_collect_state` owns assembling `state`, and MUST include
`schema_version` (set to `CURRENT_SCHEMA_VERSION`) along with the rest
of the metadata block. A `state` missing `schema_version` round-trips
into the `_migrate` "older than oldest supported" path on load and
fails with a confusing error; the owner of correctness is
`_collect_state`, not the primitive.

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
# PEP-695 aliases; pyright strict rejects bare `dict` in the public
# Migration signature, so weights and state carry precise element types.
type WeightDict = dict[str, torch.Tensor]
type StateDict = dict[str, object]
type Migration = Callable[[WeightDict, StateDict], tuple[WeightDict, StateDict]]

CURRENT_SCHEMA_VERSION: int = 1
OLDEST_SUPPORTED_SCHEMA_VERSION: int = 1   # v1 supports only itself
assert OLDEST_SUPPORTED_SCHEMA_VERSION <= CURRENT_SCHEMA_VERSION, \
    "OLDEST_SUPPORTED_SCHEMA_VERSION must be <= CURRENT_SCHEMA_VERSION"

# Registry: (from_version, to_version) -> migration function.
# Empty in v1; first entry lands when v1.1 needs to migrate v1 saves.
MIGRATIONS: dict[tuple[int, int], Migration] = {}

def _migrate(weights: WeightDict, state: StateDict) -> tuple[WeightDict, StateDict]:
    """Step the (weights, state) pair forward through MIGRATIONS
    until state['schema_version'] == CURRENT_SCHEMA_VERSION. Raises
    PredictionError if no path exists."""
    # An absent schema_version defaults to 0 (pre-versioning checkpoint,
    # caught by the too-old branch). A present-but-non-int value is a
    # corrupt state.json and is rejected distinctly rather than silently
    # coerced. bool is an int subclass, so reject it too.
    if "schema_version" in state:
        raw_src = state["schema_version"]
        if isinstance(raw_src, bool) or not isinstance(raw_src, int):
            raise PredictionError(
                f"checkpoint schema_version must be an int, got {raw_src!r}"
            )
        src = raw_src
    else:
        src = 0
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
            ) from None
        weights, state = step(weights, state)
        post = state.get("schema_version", src)
        if not isinstance(post, int) or post <= src:
            raise PredictionError(
                f"migration step ({src}, {src + 1}) did not advance "
                f"schema_version from schema {src} to {src + 1} "
                f"(got {post}, expected > {src}); the migration callable "
                f"must mutate state['schema_version'] to a strictly larger "
                f"value before returning"
            )
        src = post
    return weights, state
```

`load_weights_and_state` calls `_migrate` before returning the dicts,
so `load` consumes already-migrated state. The N1
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
The implementation guarantees termination via a strictly-increasing
post-step check rather than an explicit iteration counter: each step
must raise `schema_version` above its entry value, so the loop runs
at most `CURRENT_SCHEMA_VERSION - src` times and a step that does not
advance it raises `PredictionError("migration step (X, Y) did not
advance schema_version ...")`. This invariant is what the meta-test
catches.

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
   **RESOLVED** (Phase 6b): default CPU `np.ndarray` for callable
   convenience; a `device=` keyword on `predict_with_attention` flips
   every field to a detached on-device `Tensor`. Implemented on the
   `TransformerSequenceEstimator.Classifier` / `.Regressor` mixins in
   `models/transformer/_base.py`; the A15.1 dataclass field type stays
   `np.ndarray` (the default) with the tensor path as documented BETA
   runtime behaviour.
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
6. **`build_sampler` factory shape**: A4 lists `build_sampler`
   alongside the other three family factories, but Phase 4a ships the
   sampler side as the raw index builders `oversample_minority` /
   `undersample_majority` (pure numpy), not a config-dispatching
   `build_sampler(config) -> Sampler`. **RESOLVED**: Phase 4b's
   Trainer constructs the `SubsetRandomSampler` /
   `WeightedRandomSampler` from these builders plus `SamplerConfig`;
   the `build_sampler` dispatch entry in A4 is the 4b Trainer's
   responsibility, not a standalone 4a factory.

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
  with an example `class TFTClassifier(_TFTEstimatorMixin, TransformerSequenceEstimator.Classifier, BaseSequenceClassifier)`.
- **Style line 1233 / line 1247 (style r1-I1, r1-I2).** Rewritten
  in place during the CI wall-clock and A20 edits.

Phase 3 doc-sync (post-Gemini final-pass code/doc drift audit):

- **A6 `forward()` precondition contract.** Rounds 1-2 added three
  runtime guards (NaN/inf on real inputs, categorical index
  out-of-range, non-contiguous padding) and round 3 added the
  `InterpretableMultiHeadAttention` `hidden_size % n_heads` check;
  only the pre-existing zero-valid guard was in A6. A6 now has a
  "Model-boundary contract" block enumerating all four `forward()`
  `PredictionError` preconditions plus the attention construction-time
  `ValueError`. DOC-ONLY: the design was sound (F3 / `TFTConfig`
  already constrain the estimator path; the guards protect a direct
  caller), so no Phase 4 design review was required.
- **A17 `schema_version` ownership.** The `def0cdc` serialization
  docstring stated the caller (`_collect_state`) owns assembling
  `schema_version` while the primitive persists `state` verbatim; A17
  prose did not. A17 now states this explicitly next to the `save`
  skeleton so a Phase 4 author does not ship a `_collect_state` that
  omits `schema_version` and hits the confusing "older than oldest
  supported" load failure. DOC-ONLY.

Phase 4a (training Claude swarm):

- **strict_mode_globals fixture leak (arch-opus r1-C1).**
  `tests/unit/training/conftest.py` fixture was fspath-scoped to
  `test_determinism.py`, so `enable_strict_mode`'s four process
  globals leaked into sibling training tests and the suite went
  nondeterministically red under default `pytest` (pytest-randomly).
  Fixture now wraps every training-unit test; A14 synced; verified
  green across repeated default-randomized full-suite runs.
- **legal_task_loss_pairs SSOT (arch-opus r1-I2).** `build_loss`'s
  hand-maintained legal-pair frozenset now derives from the single
  `_validity._LEGAL_CELLS` via `legal_task_loss_pairs()`; direct
  tests pin the v1.1 exclusion and the `include_v1_1` branch
  (qa-opus r2-C1/I1).
- **build_loss post-extras boundary (arch-opus r1-I1).** Documented
  as intentionally string-in; routing `cfg.loss` through
  `extract_deprecated_extras` is the Phase 4b `_LightningModule`
  call-site obligation.
- **cosine_with_warmup min_lr guard (code-sonnet r2-I).** `min_lr >=
  base_lr` raised an inverted (climbing) post-warmup schedule; now a
  `ConfigError`, symmetric to the constant+warmup guard.
- **class_weighted requires cross_entropy (arch r2-I2).** `build_loss`
  now rejects `class_weights` with a non-`cross_entropy` strategy
  (converse of the focal guard); unreachable on legal v1 configs but
  the factory boundary defends it.
- **sampling empty-labels (arch-sonnet r2-I2).** `_class_indices`
  raises a clear `ValueError` instead of an opaque `max([])`.
- **A7 scheduler-factory curry / NaNLossGuard ERROR emit
  (arch-sonnet r2-C1/I1).** A7 now shows the full
  `partial(build_scheduler, config=.scheduler, monitor=, total_steps=)`
  curry matching the `_LightningModule` callable type, and documents
  the ERROR-level abort emit the NaN guard performs before raising.
- **build_sampler shape (arch-sonnet r2-I3).** A20 item 6 records
  that 4a ships raw index builders and 4b's Trainer owns the
  `build_sampler` dispatch.
- **oversample majority kept whole (arch-sonnet r3-I).** At
  `oversample_ratio=1.0` the majority class now passes through
  unchanged (each index once) instead of being bootstrap-resampled,
  matching the docstring and standard oversampling; otherwise the
  Phase 4b `SubsetRandomSampler` would see duplicated/dropped majority
  indices. Test pins the unique-majority invariant.
- **cosine min_lr equal boundary (qa r3-I).** Added
  `test_cosine_with_warmup_min_lr_equal_base_lr_raises` so the `>=`
  guard cannot relax to `>` undetected.
- **DEFERRED, overlapping class-weight guards (arch-opus r3-I).** The
  focal-specific `class_weights` guard and the general
  non-`cross_entropy` guard overlap on the focal path. Kept both
  intentionally: the focal-specific message is more actionable for
  the common focal+weights mistake; the general guard defends every
  other non-cross_entropy strategy. Redundancy is for message
  clarity, not an oversight.

Phase 4b (training Claude swarm):

- **class_weighted train-fold weights (round 1, code/qa).**
  `Trainer._class_weights` derives the binary `pos_weight` and the
  multiclass inverse-frequency vector from `targets[train_idx]` so the
  held-out folds never leak into the weighting (A8 architecture.md:1208).
- **optuna_trial threading (round 1, A16).** `Trainer.fit` accepts
  `optuna_trial: optuna.trial.BaseTrial | None` and threads it into the
  `_LightningModule` constructor, never through the pydantic config
  (A16 ~1990-1994). Widened from `optuna.Trial` to
  `optuna.trial.BaseTrial` (only `.report` / `.should_prune` used;
  enables `FixedTrial` in tests); A7 ~1017 and A16 ~1991-1992 synced.
- **multiclass weight-vector vocabulary size (round 2 CRITICAL,
  code-reviewer x2).** The multiclass branch sized `n_classes` from
  `targets[train_idx].max()`, so a top class held entirely out of the
  train fold produced a `CrossEntropyLoss(weight=...)` shorter than K
  and an opaque first-forward shape mismatch. Vocabulary is now sized
  from the full transformed `targets` while frequencies are still
  counted only over the train fold (A8 ~1209). Test pins a 3-class
  fold excluding the top class still yields a length-3 vector.
- **all-positive / all-negative binary fold guard (round 2
  IMPROVEMENT, code-opus/qa).** The prior `if pos > 0 else 1.0` guarded
  only the all-negative fold; an all-positive fold gave `neg/pos = 0.0`,
  silently discarding every positive in `BCEWithLogitsLoss`. Both
  degenerate folds now log a warning naming the absent class and fall
  back to `pos_weight = 1.0`.
- **no-leakage mutation tests (round 2 CRITICAL, qa-opus).** The F5
  `targets[train_idx]` contract in `_class_weights` and `_train_sampler`
  was mutation-insensitive: every test passed `train_idx = arange(N)`,
  making the slice an identity no-op. Added strict-subset `train_idx`
  tests whose fold balance differs from the panel balance; they fail if
  the slice is replaced with `targets`. Production slicing unchanged.
- **_window_time_index precondition error path (round 2 IMPROVEMENT,
  code-sonnet/code-opus).** The round-1 monotone-non-decreasing
  `assert` had no test exercising the failure path. Added
  `test_window_time_index_non_monotone_raises`; the assert is unchanged.
- **_class_weights docstring authority (round 2 IMPROVEMENT, arch-opus).**
  Docstring cited "F5 requirements.md ~786-787" as the train-fold-only
  authority; the actual authority is A8 architecture.md:1208 (F5 only
  says "frequency-based per-class weights"). Citation corrected; added
  a line that the F5 validity matrix guarantees
  `loss.strategy == "cross_entropy"` on this branch and `build_loss`
  re-asserts.
- **DEFERRED, `_ModuleBuildSpec` collapse (arch-opus round 2
  IMPROVEMENT-1).** The `class_weights` / `optuna_trial` passthroughs
  are threaded as positional `_build_module` params. Collapsing them
  into a `_ModuleBuildSpec` now is premature abstraction with no second
  consumer (a 2-param seam); revisit if Phase 6 adds a third
  module-construction input.
- **`_window_time_index` precondition (arch round 3 IMPROVEMENT).**
  Converted the monotone-`entity_id` `assert` to a `ValueError` raise
  so `python -O` cannot strip the guard the docstring promises;
  `test_window_time_index_non_monotone_raises` repointed to
  `pytest.raises(ValueError)`.
- **fit() class-weights wiring pinned (qa round 3 IMPROVEMENT).**
  Added `test_fit_passes_train_idx_not_full_panel_to_class_weights`:
  spies the index `fit` hands `_class_weights`, asserting it is the
  train fold (strict subset), not `arange(N)`, so val/cal balance
  cannot leak into the loss weighting at the call site.
- **A7 constructor pseudocode sync (round 3 NITPICK).** Dropped the
  stale `self._consecutive_nan = 0`, added `self._last_train_output`,
  in the A7 `_LightningModule` skeleton so the spec matches the
  implementation. (NaN ownership later moved to `training_step`; see
  the Gemini-final-pass block below.)

Phase 4 (Gemini final-pass):

- **F9 non-finite-loss step is now actually skipped (gemini C1).**
  Gemini's cross-family pass found a real F9 violation 6 Claude
  rounds missed: `training_step` returned the loss unconditionally and
  `NaNLossGuard.on_train_batch_end` only counted post-hoc, so a single
  NaN/inf loss still ran backward + `optimizer.step()` (weights
  poisoned) because the callback fires after the optimizer step. F9
  requires single NaN steps to skip with no gradient update. Moved the
  non-finite skip + 3-consecutive abort into `training_step` (returns
  `None` so Lightning skips the update; broadened NaN -> non-finite to
  also catch mixed-precision `inf`); removed the `NaNLossGuard`
  callback (it is structurally unable to satisfy F9) and its Phase 4a
  tests; A7 reconciled. The earlier "NaNLossGuard callback hook (qa
  r1-C2)" / "scheduler-factory curry / NaNLossGuard ERROR emit"
  Addressed entries are superseded by this.
- **F11 `train.grad_norm` now emitted (gemini C3).** Added
  `on_before_optimizer_step` emitting `train.grad_norm` at DEBUG every
  step with the F11 payload (`step`, `grad_norm`, `lr`); the event was
  in the enum but never emitted.
- **F11 `train.epoch` payload completed (gemini C2).** The event was
  emitted with only `epoch`; F11:1195 requires `epoch`, `train_loss`,
  `val_loss`, `val_metric`. `training_step` now logs `train_loss` and
  `on_train_epoch_end` reads `train_loss`/`val_loss`/`val_metric` from
  `callback_metrics` (None when absent, e.g. validation not yet run).
- **DEFERRED, DataLoader sampler branch (gemini NITPICK).** Cosmetic
  `sampler is None` branch in `fit`; `sampler or SubsetRandomSampler(...)`
  would collapse it. Left as-is: two explicit branches read clearer
  than an `or`-defaulted sampler and the duplication is two lines.

Phase 4 (post-Gemini re-confirmation swarm):

- **`on_before_optimizer_step` no longer emits a spurious
  `train.grad_norm` on a skipped step (code-opus / code-sonnet
  IMPROVEMENT).** Both code reviewers verified against Lightning 2.6.1
  source that the hook fires unconditionally, including after a
  non-finite `training_step` returned `None` (only backward +
  `optimizer.step()` are gated, not the hook). The old docstring's "a
  NaN/inf step never reaches here" was false, and a skipped step
  recorded `train.grad_norm=0.0` (no gradient was computed). Added a
  `_consecutive_nan > 0` early-return guard (a finite step always has
  `_consecutive_nan == 0`, reset in `training_step`) and corrected the
  docstring; pinned with
  `test_on_before_optimizer_step_silent_after_skipped_step`.
- **End-to-end `train.epoch` assertion added (code-opus I2).** F11
  `train.epoch` was only exercised through a hand-driven
  `on_train_epoch_end`; added `test_fit_emits_train_epoch_with_f11_payload`
  asserting the record fires with the full F11:1195 payload through a
  real `pl.Trainer.fit`.

Phase 5 (calibration Claude swarm, round 1):

- **`_Calibrator` argument renamed `logits` -> `raw_output` (arch-opus
  I1 / arch-sonnet IMPROVEMENT-2 / code-sonnet).** The protocol param
  was named `logits` after the classifier case, misleading for the
  regression calibrators that receive a predicted-quantile matrix.
  Renamed across the A9 protocol, `_protocol.py`, and all five
  concrete calibrators before Phase 6 imports the protocol; the A9
  doc block is updated to match.
- **A9 conformal wording reconciled with the code (arch-opus I2).**
  A9 said the check is on the "calibrated" vector; the code checks the
  RAW predicted quantiles before the offset (the offset would mask the
  crossing). A9 reworded to state the raw-pre-offset check explicitly.
- **Module loggers use `__name__` (code-sonnet I2 / arch-sonnet
  NITPICK-1).** `classification.py` / `regression.py` / `threshold.py`
  switched from the hardcoded `"seq_sklearn.calibration"` to
  `logging.getLogger(__name__)` per `.claude/rules/python.md`; tests
  capture on the parent logger and child records propagate.
- **Empty-fold guard (code-opus I1).** `expected_calibration_error`
  and `mean_quantile_coverage` raise `ValueError` on a zero-row fold
  instead of silently returning `0.0` / `nan` (testing.md empty-input
  rule); two tests added.
- **Regression hypothesis property now covers `ConformalCalibrator`
  (code-sonnet I1 / qa-opus I1 / qa-sonnet I1).** The pre-sorted
  hypothesis input keeps the F9 guard quiet so both regression
  strategies are exercised per implementation_plan.md:780.
- **Mutation-pinning tests added (qa-sonnet I2 / qa N3).** A 2-quantile
  crossing test pins the `pred.shape[1] > 1` guard; a `cal_size == 100`
  boundary test pins the exclusive `< 100` small-set threshold.

Phase 5 (calibration Claude swarm, round 2):

- **Empty-fold guarded at the `fit` boundary, no partial state
  (arch-opus IMPROVEMENT / arch-sonnet IMPROVEMENT-1).** Round 1 added
  empty-fold `ValueError`s in `_metrics`, but those fire only after
  `fit` has run the optimizer and set `self._*`, leaving a
  partially-fitted instance after `fit` raised, and `ConformalCalibrator`
  raised a leaked numpy `IndexError` before reaching `_metrics`. Added
  a `_require_nonempty` guard at the top of every classification `fit`
  and an `arr.shape[0] == 0` guard in regression's `_as_pred_matrix`
  (its first call in `fit`/`transform`), so a zero-row fold raises a
  boundary `ValueError` (python.md "raise `ValueError` for bad data")
  before any state mutation. Tests pin the raise AND that the
  calibrator stays unfitted (`transform` -> `NotFittedError`) for all
  five calibrators.
- **Stale `(logits, y_true)` test docstring fixed (style-opus
  NITPICK).** `test_classification.py` module docstring now reads
  `(raw_output, y_true)` to match the round-1 rename.
- **Flat-adjacent-quantile test added (qa NITPICK).** Pins
  `np.diff(...) < 0.0` (ties are non-decreasing, not a crossing)
  against a `<= 0.0` mutation, per the F9 "non-decreasing" reading.

Phase 5 (calibration Claude swarm, round 3):

- **Flaky binary-calibrator hypothesis property de-flaked (qa-sonnet
  CRITICAL C1).** The property derived labels from the logit sign with
  a 20% flip; on small `n` that can produce a near-separable fold where
  `TemperatureScaling` / `PlattScaling` correctly raise `TrainingError`
  (LBFGS divergence), so the unconditional shape/finiteness assertion
  failed nondeterministically and hypothesis replayed the stored
  counterexample. Labels are now balanced alternating and uncorrelated
  with the logits, so every generated fold is well-posed and the
  property holds universally; the legitimate divergence path stays
  covered by `test_temperature_non_finite_input_raises_training_error`.
  `.hypothesis/` added to `.gitignore`. Verified 5x isolated (fresh
  hypothesis DB) + 3x randomized full-suite.
- **Multiclass empty-fold params added (arch-sonnet NITPICK).**
  `TemperatureScaling("multiclass")` and `IsotonicCalibrator("multiclass")`
  added to the empty-fold parametrization for symmetry.

Phase 5 (Gemini cross-family final-pass):

- **Calibrators are now explicitly CPU-internal (gemini CRITICAL x3,
  one root cause).** Gemini's cross-family pass caught what 4 same-family
  rounds missed (every test is CPU; N1 emphasises the CPU path): the
  calibrators are numpy / sklearn bound but did not normalize input
  device, so a CUDA `raw_output` from a GPU-trained backbone (the
  A11/N5 hardware path, wired in Phase 6) crashes the fit path, a
  cross-device op (`x / log_t.exp()` with a CPU `log_t`) for the LBFGS
  calibrators and `Tensor.numpy()` on CUDA for the
  isotonic/conformal/threshold paths. Added a `_cpu` boundary
  normalization at every public `fit` / `transform` across
  `classification.py`, `regression.py`, `threshold.py`, and the
  `_metrics` helpers; `transform` returns a CPU `float64` tensor and
  the A9 `_Calibrator` docstring now pins the CPU-internal +
  CPU-return contract (the estimator owns moving predictions back to
  its API/device). Pinned by `tests/unit/calibration/test_device.py`.

Phase 5 (Gemini-fix re-establishment swarm, round 5):

- **Device-normalization mutation-pinned on the CPU-default runner
  (qa-opus / arch-sonnet IMPROVEMENT).** The original device tests
  asserted only CPU-in / CPU-out, which cannot catch a dropped `_cpu`
  call on a CPU runner (the `pytest.mark.gpu` test is skipped in
  CPU-only CI). Added `test_transform_detaches_requires_grad_input_cpu_visible`:
  a `requires_grad` input can only return without a `grad_fn` if the
  `.detach().cpu()` boundary actually ran, so a dropped `_cpu`
  regresses loudly in default CI.
- **`ThresholdTuner` device path now exercised (code-sonnet /
  qa-sonnet IMPROVEMENT).** It is not a `_Calibrator` so it was absent
  from `_calibrator_cases()`; added a `requires_grad` CPU test and a
  CUDA leg in the `pytest.mark.gpu` test. The earlier ledger overclaim
  ("enforced where a GPU exists") is corrected: the contract is now
  pinned CPU-visibly for every calibrator and the tuner.
- **float32 input pinned for the regression calibrators (qa-sonnet
  IMPROVEMENT).** `test_calibrators_accept_float32_input` is now
  parametrized over all seven calibrator cases (was classification
  only), asserting float32 in -> finite float64 out.
- **Redundant `.detach()` after the `_cpu` boundary removed (code-opus
  / code-sonnet / arch-sonnet NITPICK).** `_cpu` (and the regression
  `.detach().cpu()` line) already detaches; the downstream `.detach()`
  calls were no-ops and removed for clarity.
- **Review-process labels removed from shipped code (style-sonnet
  IMPROVEMENT / NITPICK).** The `test_device.py` module docstring no
  longer names the review tool; the `_cpu` docstring and the gpu-test
  comment drop "Phase 6" process labels for plain behavioural prose.
- **Redundant `.astype(float)` dropped from `IsotonicCalibrator.serialize`
  (gemini IMPROVEMENT, rationale corrected).** Gemini claimed
  `ndarray.astype(float)` raises in numpy 2.0; that is false (the repo
  runs numpy 2.4.5 and the byte-equal round-trip tests are green on
  it). The call was merely redundant: `X_thresholds_` is already a
  `float64` array and `.tolist()` yields Python floats. Removed for
  clarity, not for the stated reason.

Phase 6a (estimator-shell implementation):

- **A17 needs encoder/scaler + TabularToSequence fitted-state
  serialization (cross-phase, user-approved).** The save/load layer
  requires the fitted transformer in `state.json`. Added
  `get_fitted_state` / `set_fitted_state` to `CategoricalEncoder` and
  the scalers (the `encoders.py` docstring already promised this
  contract) and `serialize` / `deserialize` to `TabularToSequence`.
  Deliberate Phase 2 touch reviewed by the 6a swarm; A5 / A17 updated
  to name `tabular_to_sequence_state` as the transformer's
  `serialize()` JSON dict. The fit-time `(id, time)` target map is NOT
  persisted (a model artifact must not carry training labels;
  `_aligned_target` already NaN-falls-back for unmapped rows, so
  predict on a reloaded model is unaffected).
- **A4 "clone each adapter inside `__init__`" reconciled with
  `sklearn.base.clone`.** The A4 step-3 draft (clone adapters in
  `__init__`) is incompatible with sklearn: `clone` re-checks param
  identity and rejects any constructor that "modifies a parameter",
  and `clone` already deep-clones every nested param before
  re-construction, so the in-`__init__` clone breaks clone and is
  redundant. `__init__` now stores params verbatim with the
  None-default-instance idiom; clone-safety is sklearn's job. A4 step 3
  is corrected to describe this; the A4 example block is illustrative
  only and its `clone(...)` lines are superseded by this entry.
- **F1.1 tag pseudo-code reconciled with the real sklearn 1.6+ API.**
  The F1.1 draft set `tags.input_tags.dataframe = True`; sklearn 1.6+
  `InputTags` has no `dataframe` field. The panel-DataFrame contract is
  expressed with the real fields: `input_tags.two_d_array = False`
  (not a plain numpy-array estimator) + `input_tags.allow_nan = False`
  + `target_tags.required = True` + `requires_fit = True` +
  `non_deterministic` flipped on mixed precision (N5). F1.1's tag block
  is updated to the real field set.
- **`_ScalarOutputLoss` loss-factory bridge (Phase 4 `losses.py`
  touch).** The `(B, 1)` binary / point head (the F1.1 `out_dim=1`
  contract) and the `(B,)` scalar-target losses
  (`BCEWithLogitsLoss` / `BinaryFocalLoss` / `MSELoss` / `L1Loss` /
  `HuberLoss`) need shape + dtype alignment; the factory now wraps
  those four families in `_ScalarOutputLoss` (flatten both to `(B,)`,
  cast target to the prediction dtype). Multiclass CE and pinball keep
  their `(B, K)` head and are unwrapped. The affected Phase 4 loss /
  trainer tests were updated to assert the wrapper's `.inner`.
- **`predict` / `predict_quantiles` median consistency.** A quantile
  regressor's `predict` returns the CALIBRATED median column (the same
  matrix `predict_quantiles` reports), not the raw median, so the
  point estimate and the median quantile agree.
- **Estimator owns the RNG seed thread (A7 / F5 / N1).** `fit` calls
  `torch.manual_seed(seed)` / `np.random.seed(seed)` before backbone
  init and the sampler so two same-seed fits in one process are
  bit-identical; the N4 deterministic-algorithm gate stays the
  Trainer's job. The `implementation_plan.md` Phase 6a module list is
  noted as also touching `data/encoders.py`, `data/tabular_to_sequence.py`,
  and `training/losses.py` for the above (beyond the `models/` files).

Phase 6a (estimator Claude swarm, round 1):

- **`load()` now restores the full `__init__` surface (code-opus/sonnet
  CRITICAL).** The first cut set only ~7 params, so a loaded
  estimator's `get_params(deep=True)` / `sklearn.base.clone` raised
  `AttributeError` (F1 breakage). `_collect_state` now persists a
  `hyperparams` snapshot (`get_params(deep=True)` minus the adapter
  objects, JSON-coerced); `load()` reconstructs via `cls(task_type=...)`
  + `set_params(**hyperparams)` so the six adapters and every scalar
  are restored. Verified: loaded `get_params`/`clone` work, predictions
  byte-equal.
- **`_build_config` Phase-7 seam (arch-opus CRITICAL).** The hardcoded
  kwarg block could not construct `TFTConfig` (required `tabular_config`
  + model-shape fields, `extra='forbid'`). Split into a shared kwarg
  dict + a `_config_kwargs()` family hook (`{}` for base/dummy; the TFT
  family overrides to add `tabular_config` + model-shape) so the base
  keeps one `_build_config`.
- **`window_time_index` unified with a precondition guard (arch-sonnet
  CRITICAL).** The estimator's private copy omitted the
  monotone-`entity_id` `ValueError` the Trainer's had, risking a
  silently-wrong calibration fold for a non-TTS caller. Moved to
  `data/splits.window_time_index` (single source, with the guard);
  `Trainer._window_time_index` is a thin delegating alias. Resolves the
  duplication IMPROVEMENT too.
- **A17 / F4 fit-state location reconciled (arch-opus CRITICAL).** The
  spec said `classes_`/`n_outputs_`/`quantiles_`/`decision_threshold_`
  go in `weights.safetensors`; `classes_` is a `LabelEncoder` vector
  that is commonly non-numeric (string labels) and does not tensorize.
  A17 + F4 amended: `weights.safetensors` is strictly backbone/head
  tensors; all fit-state is JSON in `state.json` alongside the
  calibrator/transformer state. Covered by the byte-equal round-trip tests.
- **`tabular_config` no longer double-serialized (arch-sonnet
  CRITICAL).** `_collect_state` dumps `config` with
  `exclude={"tabular_config"}` (no-op for `BaseModelConfig`; drops the
  embedded copy for the Phase-7 `TFTConfig`) so the flat
  `tabular_config` key is the single authoritative transformer-config
  dump.
- **Missing plan-mandated tests added.** `test_short_entity_predict`
  (below-floor entities -> NaN-filled prediction rows + exactly one
  aggregated breach log, the estimator predict path), the
  estimator-level `test_load_version_mismatch_warning`, the N1
  fit-state-attribute contract test (all F1.1 attrs + shapes/dtypes +
  `config_` frozen), and the all-six-adapter clone-no-alias test. The
  stale subprocess-load docstring is corrected.
- **IMPROVEMENTs resolved.** `feature_schema_fingerprint_` is now
  re-validated at predict (mismatch -> `DataContractError`, the F4
  intent); `_build_calibrator_from` reads `self.calibration_strategy`
  set from the loaded config (single source); the control-flow
  `assert` in the classifier threshold path is an explicit guard; the
  Phase-7 process-label comments in `_base.py` are reworded to plain
  rationale.

Phase 6a (estimator Claude swarm, round 2):

- **Sentinel targets excluded from the recomputed calibration fold
  (code-opus IMPROVEMENT).** When `min_periods_predict > min_periods`,
  an entity with `min_periods <= n < min_periods_predict` survives
  `TabularToSequence.fit` but `transform` injects a sentinel target
  (`-1` classification / `NaN` regression). `_calibration_fold`
  returned `batch["target"][cal_idx]` unfiltered, so the calibrator /
  threshold tuner could fit on sentinel labels. The recomputed-split
  branch now drops below-floor windows via `_below_floor_mask`. (The
  explicit-`calibration_set` branch is unaffected: its targets are the
  caller's real `y_cal`.)
- **Empty-calibration-fold config guard (arch-opus IMPROVEMENT).**
  `calibration_strategy != 'none'` or `threshold_tuning=True` with
  `cal_fraction=0` and no `calibration_set` previously failed deep in
  the calibrator with a message that never named the cause. `fit` now
  raises `ConfigError` naming `cal_fraction` at the boundary (F2).
- **Dead `calibration_strategy` state key removed (arch-sonnet
  IMPROVEMENT).** It round-trips through `hyperparams` and `load()`
  restores it via `set_params`; the duplicate top-level key in
  `_collect_state` was never read and is migration-confusing. Removed.
- **A17 / A7 doc drift fixed (arch-sonnet IMPROVEMENT).** The A17 `load`
  skeleton named a non-existent `_reconstruct` seam; rewritten to the
  inlined constructor + `set_params` + family-hook reconstruction the
  code actually uses, naming the Phase-7 override surface. A7 step 7
  incorrectly attributed calibration to the Trainer; corrected to state
  the estimator owns it after `Trainer.fit` returns.
- **qa / weak-assertion test gaps closed.** Regressor below-floor
  NaN-fill now has a behavioral test (`predict` + `predict_quantiles`,
  parallel to the classifier one); classifier `predict_proba` pre-fit
  `NotFittedError` is asserted at the shell level; `decision_threshold_`
  ABSENCE after `load` is asserted (binary-no-tuning and multiclass);
  the vacuous `Trainer._window_time_index is not None` assertion now
  checks delegation by output equality.
- **Style.** The recurring "tests prove correctness" reviewer-vouching
  prose and two rhetorical capitalized `AND`s in the ledger are
  reworded.

Phase 6a (estimator Claude swarm, round 3):

- **Mutation-sensitive test for the Round-2 `keep` sentinel filter
  (code-opus / qa-sonnet / qa-opus CRITICAL).** The Round-2 fix had
  100% line+branch but no test that would fail if `keep = cal_idx[...]`
  were reverted (every prior test reaching it used
  `min_periods_predict=1`, so the mask was all-False). Added
  `test_calibration_fold_drops_below_floor_sentinel_rows` and
  `test_calibration_fold_threshold_tuner_path_no_sentinel`: a panel
  where a below-floor entity provably lands in the recomputed
  `cal_idx`, with explicit preconditions (`below[cal_idx].any()` and
  the unfiltered fold carries `-1`) so reverting the filter fails the
  test. Both the calibrator and the threshold-tuner consumers of
  `_calibration_fold` are exercised, and the estimator-side fold
  alignment (`len == (~below)[cal_idx].sum()`) is asserted.
- **Empty post-filter fold raises a typed `ConfigError`
  (code-sonnet IMPROVEMENT).** When `keep` empties (every cal-fold
  entity below `min_periods_predict`) the calibrator previously raised
  a deep, unnamed `ValueError`. `_calibration_fold` now raises a
  `ConfigError` naming `min_periods_predict` / `cal_fraction`.
  `test_calibration_fold_all_below_floor_raises_configerror` covers it.
- **Empty-fold guard's calibrator operand tested (qa-sonnet / qa-opus
  IMPROVEMENT).** `test_calibrator_strategy_with_cal_fraction_zero_raises`
  isolates the `_make_calibrator() is not None` arm of `needs_fold`
  (the prior test only triggered the `threshold_tuning` arm).

Phase 6b (family-base implementation):

- **Phase-6a `_forward_backbone` seam (cross-phase, behaviour-preserving).**
  `predict_with_attention` needs the full `BackboneOutput` (the
  introspection tensors), not just `representation`, from the SAME
  forward pass `_predict_raw` uses (a second pass could disagree).
  `_base.py` extracts `_forward_backbone(X) -> (output, head, batch,
  below)`; `_predict_raw` now delegates to it. `_classifier.py` splits
  `_index_from_proba` (shared by `predict` and the mixin, whose A15.1
  `predictions` field is class indices); `_regressor.py` splits
  `_calibrate_raw` (shared by `_calibrated_matrix` and the mixin so the
  attention path reuses the identical calibrate-then-NaN-fill on its
  single pass). All three are pure refactors: the full Phase-6a +
  integration suite stays green unchanged.
- **`predict_with_attention.predictions` is class indices, not labels
  (A15.1 literal).** A15.1 annotates the classifier `predictions`
  field `# (N,) class indices`. The mixin returns the integer index
  vector (`_index_from_proba`), NOT the `classes_`-mapped labels
  `predict` returns; callers map via `est.classes_`. This keeps the
  `device=` path tensorisable (string labels do not tensorise) and
  follows the A15.1 "class indices" annotation (the v1.1 multi-label
  clause in the A15.1 source comment is deferred from the v1 code
  comment, not a contradiction). The family-base test asserts
  `classes_[predictions] == predict(X)` for consistency.
- **A20 item 2 RESOLVED: `device=` keyword.** `predict_with_attention`
  defaults to CPU `np.ndarray`; a `device=` keyword flips every field
  to a detached on-device `Tensor`. Pinned here per A20 item 2's
  "pin during implementation". The A15.1 dataclass field type stays
  `np.ndarray` (the default and common case); the `device` path's
  tensors are the documented BETA runtime behaviour.
- **Introspection tensors are NOT NaN-filled for below-floor entities.**
  The F NaN-in-output contract scopes NaN to the prediction surface
  (`probabilities` / `logits` / regression `predictions`), which the
  mixin NaN-fills exactly as the base predict path. The attention /
  variable-selection tensors are diagnostics, not predictions, and
  stay finite so a caller can still inspect what the model attended to
  for a short entity.

Phase 6b (family-base Claude swarm, round 1):

- **Classifier mixin GPU crash (code-sonnet CRITICAL).** The classifier
  `predict_with_attention` passed the raw `head(...)` output (on-device
  for a GPU-trained model) to `_proba_from_raw`, which calls `.numpy()`
  internally and raises on a CUDA tensor. Now `.detach().cpu()` first,
  mirroring the base `_predict_raw` contract and the Regressor mixin.
  The `_force_cpu` test harness masked this; a CPU/GPU parity gap.
- **Mutation-insensitive / uncovered seam paths (qa-sonnet /
  qa-opus CRITICAL).** The family-base "consistency" assertions
  (`out.probabilities == predict_proba(X)`) were vacuous: both sides
  now delegate to the same seam, so a shared bug passes. Added
  mutation-sensitive tests: classifier (calibration_strategy=
  "temperature") and regressor (calibrated) `predict_with_attention`
  vs the base path AND an independent fixed-representation oracle
  (monkeypatched backbone) so a shared-seam regression fails; the
  binary `threshold_tuning` index branch through `predict_with_attention`;
  the regressor below-floor NaN-fill (was classifier-only); and a
  `device=` numpy-vs-tensor value-equality check (was isinstance-only).
- **`AttentionOutput.logits` shape comment (code-opus / arch IMPROVEMENT).**
  The field comment said `(N, num_classes)`; the binary head emits
  `(N, 1)`. Corrected to `(N, head_out_dim): 1 for binary, num_classes
  else`. `probabilities` is `(N, num_classes)` and was already correct.
- **`entity_id` is the internal code (arch IMPROVEMENT).** The field
  carries the contiguous LabelEncoder code, not the original id. The
  dataclass comment now says so; decoding to the user-facing id is
  deferred to Phase 7 (it needs the transformer's id inverse, which
  `TFTClassifier` wires up).
- **Ledger `verbatim` claim corrected (arch IMPROVEMENT).** The
  predictions bullet no longer claims the code comment matches A15.1
  "verbatim" (the v1.1 multi-label clause is deferred from the v1 code
  comment); reworded to state the deferral explicitly.

Phase 6b (family-base Claude swarm, round 2):

- **Regressor independent-oracle (qa-opus CRITICAL).** Round 1 closed
  the shared-seam trap for the classifier only; the regressor half of
  the same `_calibrate_raw` seam still had no oracle (a `mat + 1.0`
  mutation survived the whole suite because every regressor
  `predict_with_attention` assertion compared against `predict` /
  `predict_quantiles`, which delegate to the same seam). Added
  `test_regressor_predict_with_attention_independent_oracle`: point
  mode, no calibrator, monkeypatched fixed representation, predictions
  checked against an independent `head(rep)` recomputation, with a
  different representation required to move the output.
- **A15.1 source snippet synced (arch-opus / arch-sonnet IMPROVEMENT).**
  The Round-1 fix corrected the `inference/attention.py` field comments
  but left the authoritative A15.1 doc snippet stale. Synced
  `logits` to `(N, head_out_dim): 1 for binary, num_classes else`,
  both `entity_id` lines to "internal contiguous entity code", and the
  classifier `predictions` line to name `estimator.classes_`. A15.1 is
  the contract a Phase-7 author reads first; it now leads the code.
- **A6.1 snippet validation bounds (arch NITPICK).** The
  `RecurrentSequenceEstimatorConfig` snippet now shows the shipped
  `Field(ge=0.0, lt=1.0)` on `recurrent_dropout` and
  `Field(default=None, ge=1)` on `bptt_window` so the validation
  contract is visible at the architecture layer.

Phase 6b (family-base Claude swarm, round 3):

- **Classifier threshold-branch independent oracle (qa-sonnet /
  qa-opus CRITICAL).** `_index_from_proba`'s `proba[:,1] >=
  decision_threshold_` branch was the last classifier shared-seam
  trap: `predict` and `predict_with_attention` both route through it,
  so a `>=` -> `<` mutation inverted both and the consistency
  assertion stayed green. Added
  `test_classifier_threshold_branch_independent_oracle`: a fitted
  `threshold_tuning` classifier, fixed representation, the index
  recomputed independently with `>=` and asserted against production
  for two sigmoid-extreme representations. Verified the test fails
  under the `>=` -> `<` mutation and passes on revert.
- **Calibrated regressor independent oracle (qa-sonnet CRITICAL /
  code-opus IMPROVEMENT).** The `calibrator_.transform` arm of
  `_calibrate_raw` (and the quantile-mode regressor path) had no
  oracle independent of `predict_quantiles`. Added
  `test_calibrated_quantile_regressor_independent_oracle`:
  `isotonic_quantile` calibration, fixed representation,
  `calibrator_.transform(head(rep))` recomputed independently and
  matched against `predict_with_attention`, with a different
  representation required to move the calibrated output. This also
  closes the code-opus quantile-mode-oracle IMPROVEMENT.
- **A15.1 snippet wording sync (arch-sonnet IMPROVEMENT).** The
  pre-existing `probabilities` / classifier `padding_mask` comment
  divergences between the A15.1 snippet and `inference/attention.py`
  are reconciled (the code now carries the doc's
  `post-softmax/sigmoid` wording and the `(pass-through from
  preprocessing)` provenance clause).
- **Classifier below-floor predictions parity (qa NITPICK).**
  `test_below_floor_nan_fill_matches_base` now also asserts
  `classes_[predictions] == predict(x_pred)` for the below-floor
  rows, pinning the shared `_index_from_proba` A2 (NaN -> index 0)
  behaviour on the attention path.

Gemini final-pass (Phase 1-6 integration, post-consensus):

- **`load()` crash when an explicit `loss` adapter was saved
  (Gemini CRITICAL, verified).** `__init__` stores `self.loss`
  verbatim as `None` (the one adapter not defaulted to an instance,
  because F5 task-aware loss-default injection must see "unspecified").
  When a real `loss=LossParams(...)` was given at save, `_hyperparams`
  persisted its `loss__*` leaves but dropped the bare `loss` object;
  `load()` did `cls(task_type=...)` (loss=None) then
  `set_params(loss__strategy=...)`, which sklearn routes to
  `None.set_params` -> `AttributeError`. Latent: every Phase-6 test
  uses the loss=None default, so the green suite never exercised it.
  Fixed in `load()`: instantiate a default `LossParams` before
  `set_params` when `obj.loss is None` and a `loss__` key is present.
- **Trainer trained on below-floor sentinel targets (Gemini CRITICAL,
  verified; cross-phase Phase-4 touch).** `TabularToSequence.transform`
  emits sentinel targets (`-1` classification / `NaN` regression) for
  entities with `min_periods <= n < min_periods_predict`. The Phase-6b
  fix dropped these from the estimator's recomputed calibration fold,
  but `Trainer.fit` still passed the unfiltered `train_idx` / `val_idx`
  to `_class_weights` (`torch.bincount` on `-1` raises) and the loss
  (NaN regression target trips the F9 abort). The Phase-6a ledger had
  noted this exposure existed in the frozen Phase-4 Trainer but left it
  unaddressed; the integration pass correctly re-raised it. Fixed (a
  deliberate frozen-Phase-4 touch): `Trainer._below_floor_mask` drops
  below-floor windows from `train_idx` / `val_idx` before class-weights
  / sampler / loaders. (The duplicated mask logic was subsequently
  hoisted to a shared `data/splits` helper, see the confirming-swarm
  block below.)
- **`optuna_trial` typed `object` contradicting A16 (Gemini
  IMPROVEMENT, verified).** `optuna` is a hard dependency (A18,
  pyproject.toml), so annotating `optuna_trial: object | None` and
  carrying a `# type: ignore[arg-type]` on the `Trainer.fit`
  delegation served no purpose and broke strict typing against A16's
  `optuna.trial.BaseTrial | None`. Fixed: `if TYPE_CHECKING: import
  optuna`, correct annotation, `# type: ignore` removed.
- **Opaque `state["hyperparams"]` cast (Gemini NITPICK).** Left as-is:
  the F4 schema invariants enforce structural presence and the
  cast-narrowing matches the established `load()` convention; Gemini
  itself offered "or leave as-is" for this reason.

Post-Gemini confirming swarm (Phase 1-6 integration):

- **`below_floor_mask` de-duplicated into `data/splits.py` (arch-opus /
  code-opus / qa-opus IMPROVEMENT).** The Trainer and estimator each
  carried a copy of the `count < min_periods_predict` rule, the exact
  drift hazard the shared `window_time_index` helper was factored to
  prevent. Hoisted to `seq_sklearn.data.splits.below_floor_mask`;
  `Trainer._below_floor_mask` and `BaseSequenceEstimator._below_floor_mask`
  are now thin adapters delegating to it (single source, cannot drift).
  Direct `splits` unit tests added.
- **Trainer empty-fold guard (code-opus / arch-opus IMPROVEMENT).** An
  all-below-floor train / val fold was silently handed to Lightning as
  an empty loader (EarlyStopping / checkpoint degrade on a never-logged
  `val_loss`). `Trainer.fit` now raises `ConfigError` after the filter
  when `train_idx` or `val_idx` is empty, symmetric with the estimator's
  empty-calibration-fold guard. `test_fit_all_below_floor_raises_configerror`
  covers it.
- **Classifier `bincount(-1)` arm tested (qa-sonnet / qa-opus
  IMPROVEMENT).** The C2 filter protects both the regression NaN->F9
  arm AND the classifier multiclass + class_weighted
  `torch.bincount(-1)` arm, but only the regressor arm had a regression
  test. Added `test_fit_filters_below_floor_classifier_class_weighted`
  (multiclass + class_weighted + below-floor entities).
- **C1 multi-leaf round-trip (qa-sonnet / qa-opus IMPROVEMENT).**
  `test_save_load_with_explicit_loss_adapter_round_trips` now saves a
  `LossParams` with non-default `label_smoothing` / `focal_gamma` and
  asserts every leaf (not just `strategy`) survives load and clone.
- **A7 / A17 doc sync (arch-sonnet IMPROVEMENT).** A7 gains the
  below-floor filter step (2a); the A17 `load()` pseudocode shows the
  `LossParams` pre-injection so a future family implementor reading the
  canonical reference does not reproduce the C1 bug.

Phase 7 (TFT concrete Claude swarm, round 1):

- **`_base.load()` re-merges `tabular_config` for a tabular-config-
  bearing `_config_cls` (Phase-6a A17 seam, Phase 7 first to hit it).**
  `_collect_state` dumps `config` with `exclude={"tabular_config"}` (the
  transformer config is the single authoritative flat key); `load()`
  validated `state["config"]` directly, which works for the dummy's
  `BaseModelConfig` (no such field) but fails for `TFTConfig` (required
  `tabular_config`). `load()` now merges it back, gated on
  `"tabular_config" in cls._config_cls.model_fields` (no-op for the
  base path). Pinned mutation-sensitively: the TFT clf/reg e2e save/
  load tests now assert `reloaded.config_ == est.config_` (and the
  nested `tabular_config`), not just byte-equal predictions, so a
  revert of the re-merge fails the suite.
- **`_TFTEstimatorMixin` shared module (code-opus / arch IMPROVEMENT).**
  The not-in-plan `models/transformer/tft/_estimator.py` holds the
  shared A4-adapter `__init__`, `_config_cls`, `_config_kwargs`,
  `_build_tft_backbone`; `TFTClassifier`/`TFTRegressor` are thin. This
  is the project's standing DRY discipline (window_time_index /
  below_floor_mask); the TYPE_CHECKING `_MixinBase` alias gives pyright
  the cooperative-`super` signature while the runtime base stays
  `object`. A1 / A2 / A4 reconciled to show the 3-base form + the new
  module.
- **`TFTRegressor` init/clone unit coverage (qa CRITICAL).** The
  regressor MRO diverges from the classifier after the shared mixin, so
  `test_classifier_init.py` did not transfer. Added
  `test_regressor_init.py` (mirrors it + a quantile-mode clone
  round-trip pinning `quantiles` / `task_type`).
- **Multiclass + empty-categorical e2e (qa IMPROVEMENT).** Added lean
  (1-epoch, hidden=8) TFT e2e cases: multiclass head/softmax through
  the real backbone, and the empty-categorical-side `_static_pad` /
  `_tv_pad` path wired through the estimator.

Phase 7 (TFT concrete Claude swarm, round 2):

- **`prediction_readout="mean_pool"` estimator-wiring e2e (qa-sonnet /
  qa-opus IMPROVEMENT).** Round 1 deferred this as a config
  passthrough; two qa agents correctly distinguished that storing the
  value verbatim is not the same as it flowing through
  `_config_kwargs -> TFTConfig -> _build_tft_backbone` to the backbone
  readout (a passthrough break would silently fall back to
  `last_valid` while the backbone unit test still passes). Resolved,
  not deferred: added `test_tft_classifier_mean_pool_readout_roundtrip`
  (asserts `config_.prediction_readout == "mean_pool"` and a byte-equal
  save/load). Supersedes the round-1 mean_pool Deferred entry.
- **Silent-NaN guards (qa NITPICK).** Added `np.isfinite(...)` to the
  multiclass `predict_proba` and the quantile `predict_quantiles`
  assertions so a NaN-producing path cannot satisfy shape + sum-to-1.

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


> Historical note: the fold-in ledger entries below describe edits
> made when `docs/hyperparameter_strategy.md` was the authoritative
> spec. That doc was subsequently demoted to rationale + promotion
> procedure only; its schemas, test rosters, and search-space table
> were stripped. Authority now lives in this doc (A4/A16),
> requirements F5/F7, the implementation plan's test rosters, and the
> code. Entries referencing the strategy doc as authoritative for the
> surface are historical.

Hyperparameter-strategy fold-in (Round 1):

- **A1 layout** updated to list `optimizer.py`, `scheduler.py`,
  `loss.py`, `sampler.py`, `_extras.py`, and `_adapters.py` (renamed
  from `_params_adapter.py`) under `config/`. The four-tier family
  sub-configs land alongside the existing `tabular.py` / `tft.py` /
  `recurrent.py`.
- **A4 header note** added. (Subsequently updated when the strategy
  doc was demoted: A4 + requirements F7 are authoritative for the
  four-tier architecture, contracts, and per-model default search
  space; the code carries verbatim schemas; the strategy doc holds
  the rationale and the ALPHA → BETA → STABLE promotion procedure
  only.)
- **A4 step 3 (adapter pattern)** generalized from a single
  `TabularConfigParams` example to the six v1 TFT adapters
  (`TabularConfigParams`, `OptimizerParams`, `SchedulerParams`,
  `LossParams`, `SamplerParams`, `TFTAdvancedParams`). Every adapter
  `__init__` carries the `*` keyword-only marker per the Gemini-pass
  finding on the strategy doc (positional-shift on BETA promotion
  would silently break callers). The clone-safety paragraph now
  covers all six adapter slots; the six per-adapter clone tests live
  in Phase 1's `test_adapters.py`.
- **A4 task-type-aware loss default** added: `_DEFAULT_LOSS_FOR_TASK`
  map and `_build_config` injection logic; `LossConfig.strategy`
  keeps no default at the pydantic layer (legal value depends on
  `task_type` per F5).
- **`BaseTrainingConfig` / `BaseModelConfig` snippets** rewritten to
  show the nested family-sub-config shape:
  `optimizer: OptimizerConfig`, `scheduler: SchedulerConfig`,
  `loss: LossConfig`, `sampler: SamplerConfig`. The validity-matrix
  validator call site now reads `self.loss.strategy` and
  `self.sampler.strategy`; `check_combo` signature is unchanged
  (still four strings).
- **`TFTConfig` snippet** updated to add `advanced:
  TFTAdvancedConfig = Field(default_factory=TFTAdvancedConfig)`.
  `TFTAdvancedConfig` defined inline with the `extra: ExtraDict`
  escape hatch; v1 ships empty otherwise (Tier 3 per F7 / this A4
  section).
- **`TabularToSequenceConfig.categorical_embed_dims`** corrected
  to `CategoricalEmbedDims = tuple[tuple[str, int], ...]` (sorted,
  hashable, via `BeforeValidator`). The earlier `Mapping[str, int]`
  claim did not actually deliver hashability under pydantic v2.
- **`ExtraDict` escape hatch subsection** added at end of A4:
  documents `ExtraValue` restriction, `_normalize_extras`
  validator, `_PROMOTED_KEYS_BY_FAMILY` registry,
  `extract_deprecated_extras` helper, and the per-family
  `_<NAME>_RESERVED` collision-detection sets.
- **A16 `suggest_params` signature** updated with `search_advanced`
  and `search_extras` keyword-only flags. Default behavior samples
  ONLY STABLE fields; the per-model default search space is defined
  by the A16 `suggest_params` implementation (Phase 8).
- **A16 `_config_to_estimator_kwargs`** generalized to handle six
  nested fields per the `_TFT_ADAPTER_MAP` pattern. The helper
  pops every sub-config dict from `model_dump(mode="json")` and
  wraps each in its matching adapter; `mode="json"` is the pinned
  serialization mode per the F7 save/load contract, exercised by
  `test_extra_dict_survives_json_roundtrip`.
- **A16 validity-matrix sweep example** updated to read
  `config.loss.strategy` and `config.sampler.strategy` from the
  nested family configs (signature of `check_combo` itself is
  unchanged).

Hyperparameter-strategy fold-in (Round 2):

- **Em dash slipped in A16 prose** (style r2-C1 / arch r2-C1).
  Replaced ` — ` with `: ` plus follow-on prose. The doc is
  em-dash-free.
- **Clone-safety test count off by one** (arch r2-C2 / qa r2-C1).
  Strategy doc was missing `test_tabular_config_params_clone_is_independent`
  in the named-tests table; six adapter slots were claimed but only
  five tests listed. Strategy doc table now lists six per-adapter
  clone tests (one per adapter); architecture A4 clone-safety
  paragraph and the Round 1 ledger entry rewritten to name each
  adapter explicitly.
- **`_config_to_estimator_kwargs` untested** (qa r2-C2). Two new
  named tests added to the Phase 8 test roster:
  `test_config_to_estimator_kwargs_round_trips_all_adapters` (every
  adapter slot survives) and
  `test_config_to_estimator_kwargs_extra_tuple_type_survives` (the
  `extra` tuple round-trips via `mode="json"`). A16 prose now names
  both tests as the helper's coverage.
- **A8 build_loss prose drift** (arch r2-I1). Added a bridge note
  at A8 acknowledging `loss_strategy: str` is the value of
  `cfg.loss.strategy` (display label per the F5 bridge in
  requirements). The function signature parameter name stays
  `loss_strategy` for back-compat with the validity-matrix
  vocabulary; the call site reads from the nested family configs.
- **Flat-kwargs ambiguity in strategy doc** (arch r2-I2). Tightened
  the strategy doc's "flat per-field kwargs DO NOT appear" to
  scope only fields living on nested sub-configs; model-shape
  fields on `<Model>Config` (`hidden_size`, `attention_heads`, etc.)
  ARE flat top-level kwargs.
- **`_adapter_map_for` undefined** (arch r2-I4). Architecture A16
  defines the registry inline: `_ADAPTER_MAP_BY_CONFIG:
  dict[type[BaseModelConfig], dict[str, type[BaseEstimator]]]` keyed
  by concrete config class, plus a one-line `_adapter_map_for` body
  looking up the registry.
- **`model_dump` mode inconsistency between strategy and arch**
  (qa r2-I1). The strategy doc's `_config_to_estimator_kwargs`
  code sample originally used bare `model_dump()`; updated to
  `model_dump(mode="json")` matching the architecture A16 sample
  and the `test_extra_dict_survives_json_roundtrip` contract.
- **v1.1 entries in `_DEFAULT_LOSS_FOR_TASK` unguarded** (qa r2-I2).
  Added a prose note that v1.1 entries are present so v1.1
  enablement is one-line, but they are unreachable in v1 because
  `check_combo` rejects v1.1 task types before `_build_config`
  runs. Named test
  `test_v1_task_type_rejects_multilabel_and_regression_multioutput`
  pins the guard.

Gemini three-doc final pass:

- **Reserved-keys collision check moved from factory to config layer**
  (gemini-qa r1-C1). The Gemini QA pass caught a phase-ordering bug:
  `test_adamw_reserved_keys_collision_raises` and
  `test_sgd_reserved_keys_collision_raises` were placed in Phase 1
  but the reserved-keys check lived in `build_optimizer` (Phase 4).
  Moved the check to a `@model_validator(mode="after")` on
  `OptimizerConfig` (and analogous validators on `SchedulerConfig`,
  `LossConfig`, `SamplerConfig`). A4's escape-hatch subsection now
  documents the config-layer check; the factory (Phase 4) trusts the
  validated config and does not re-check. Reserved sets are now
  `_RESERVED_BY_OPTIMIZER: dict[str, frozenset[str]]` keyed by
  `cfg.name`.
- **`extract_deprecated_extras` happy-path test added** (gemini-qa
  r1-I2). The strategy doc named tests for the warning path and the
  ambiguous-configuration error path but not for the unpromoted-key
  passthrough case. Added
  `test_extract_deprecated_extras_happy_path_passes_through` in
  Phase 1; it asserts unpromoted `extra` keys pass through unchanged
  with no `DeprecationWarning`.

Phase 2 (data-layer Claude swarm):

- **A9.1 `EntityTimeSeriesSplit.left_extension` draws from the
  pre-gap-trim segment.** `split()` builds `left_extension` from the
  preceding train segment before the `gap` clamp is applied, so the
  `gap` separation is honored on the train side but a gap-window row
  can still appear in the test fold's history-only prefix. A9.1's
  semantics permit this (the left-extension rows are unscored context
  per "no test target spans the overlap"), and there is no reachable
  impact until a Phase 4+ splitter-consumer scores predictions. The
  Phase 4 splitter-consumer review must verify those rows are masked
  from loss before scoring.

Phase 4a (training Claude swarm):

- **Loss-module NaN/inf-input test (arch-opus r1-I3).** Deferred:
  out of scope by design. F2/F3 own NaN-in-features (raise
  `DataContractError` at the data-contract layer, requirements.md
  1113-1124); runtime NaN-loss is `_LightningModule.training_step`'s
  job per F9 (the non-finite skip + 3-consecutive abort; superseded
  the `NaNLossGuard` callback in the Gemini-final-pass reconciliation).
  The loss
  modules in `losses.py` are deliberately not input validators under
  the F5 design, so a NaN-input-raises test there would assert a
  contract the spec assigns elsewhere.

Phase 4 (post-Gemini re-confirmation swarm):

- **`train.epoch` suppressed on an all-non-finite epoch (arch-opus
  IMP-1).** Deferred: correct as designed. When every step of an epoch
  is non-finite, `_last_train_output is None`, so `on_train_epoch_end`
  emits no `train.epoch`. F11:1195 specifies `train.epoch` as an
  end-of-epoch summary of a completed training pass; an all-skipped
  epoch had no pass, and `train.nan_step_skipped` (WARNING, every step)
  plus the 3-consecutive `TrainingError` already make the condition
  observable. Emitting `train.epoch` with four `None` payload fields
  would be a fabricated summary. Revisit only if an operator metric
  explicitly needs a per-epoch skipped-only marker.
- **F11 payload-shape mismatches on `train.mixed_precision_diverged`,
  `optuna.trial_pruned`, and the `train.var_selection_entropy` epoch
  key (pre-existing).** Deferred to Phase 9: these are governed by the
  requirements.md F11 event-payload table and are not introduced by the
  Phase 4 work (they predate it on the v1 -> v3 logging surface). The
  Phase 9 F11-table conformance test owns a single systematic
  pass over every `Event` payload against the spec table; piecemeal
  fixes here would duplicate that effort and risk drift.
- **F9 Optuna path raises `optuna.TrialPruned`, not `TrainingError`
  (code-sonnet question).** Not a finding: correct per A16. The
  3-consecutive-non-finite abort raises `TrainingError`; translation to
  `optuna.TrialPruned` when an Optuna trial is active is the Phase 8
  `optuna_trial_guard`'s job (A16:1893+), not `training_step`'s.
  `_LightningModule` stays Optuna-agnostic on the abort path by design.

Phase 5 (calibration Claude swarm, round 1):

- **Single-class calibration fold not guarded (code-opus IMPROVEMENT-3).**
  Deferred to Phase 6: a degenerate single-class calibration fold is the
  estimator's F2 three-way-split concern, not the calibrator's. No Phase 5
  spec line mandates a symmetric guard (F9 mandates only the conformal
  non-monotone check); the Phase 6 estimator owns calibration-fold
  composition and is where the guard, if any, belongs.
- **`mean_quantile_coverage` is a single mean scalar (arch-opus I3).**
  Deferred: requirements.md F11:1201 specifies the `calibration.fit`
  payload carries a single `pre_coverage` / `post_coverage` scalar. The
  per-quantile-column comparison is documented as the calibrator's own
  job and is out of the F11 payload contract; richer per-column
  diagnostics would exceed the spec'd payload shape.
- **No explicit shape validator on the classification calibrators
  (arch-opus I4).** Deferred: classification calibrators disambiguate
  binary vs multiclass by the `task` constructor argument, not by input
  shape; the Phase 6 estimator controls the logit tensor it passes. No
  spec line mandates a regression-style `_as_pred_matrix` guard on the
  classification side.
- **NaN/inf-input-raises untested on the non-LBFGS calibrators and the
  metrics helpers (qa-opus).** Deferred: mirrors the Phase 4a
  `losses.py` deferral. Runtime NaN ownership is the F2/F3
  data-contract layer (`DataContractError`) plus F9 in the training
  loop; calibrators are not input validators, and the calibration fold
  is model output on validated data. A NaN-input-raises test here would
  assert a contract the spec assigns elsewhere.
- **Platt `and`-vs-`or` divergence mutation untested (qa-sonnet I3).**
  Deferred: the joint LBFGS fit over `(a, b)` diverges both parameters
  together on a non-finite calibration set; isolating exactly one
  requires monkeypatching `LBFGS.step` to pin a runtime state the
  optimizer cannot actually produce. The guard
  `not (isfinite(a) and isfinite(b))` is correct (raise if either is
  non-finite); a contrived test would not exercise a reachable path.
- **Double-JSON-roundtrip idempotency untested (qa-opus).** Deferred:
  the single round-trip already asserts byte-equality via `torch.equal`
  after `json.dumps`/`json.loads` (N1 exact-equality), which proves the
  serialize/deserialize inverse is exact; a second round-trip is
  implied transitively.
- **`fit`-side `_as_pred_matrix` shape-mismatch message untested
  (qa-opus).** Deferred: `fit` and `transform` call the identical
  `_as_pred_matrix` helper; the transform-side shape-mismatch test
  already exercises that exact code path and message.
- **`ThresholdTuner` uses a plain `logger.info`, not a structured
  `Event` (code-opus I2 / arch).** Deferred: the requirements.md F11
  table defines no `threshold.*` event; emitting a structured record
  would invent an unspecified event. The plain INFO line is consistent
  with that deliberate F11 omission (threshold tuning's durable output
  is `decision_threshold_` on the estimator, not a log event).

Phase 5 (Gemini cross-family final-pass):

- **`pre_ece` computed in float64, not the model's float32 (gemini
  IMPROVEMENT I1).** Deferred: `pre_ece` and `post_ece` are both
  computed in float64, so the pre/post comparison stays
  apples-to-apples (the only contract on these fields). The
  float32->float64 activation delta is orders of magnitude below the
  15-bin ECE resolution, and `pre_ece` is a logged diagnostic, not a
  contract output. Measuring both legs in one precision is the correct
  choice; matching the model's float32 only on the pre leg would skew
  the comparison.
- **No upper bound on the fitted temperature (gemini IMPROVEMENT
  I4).** Deferred: a large finite `T` is mathematically valid
  recalibration, temperature scaling is monotone, preserves
  argmax/ranking/AUC, and only flattens overconfidence (the intended
  effect). No spec line mandates an upper bound, and any cutoff
  (Gemini suggested `1e4`) would reject legitimate strong recalibration
  of a heavily overconfident network. The existing guard correctly
  rejects only non-finite / non-positive `T` (the genuine divergence).
- **Isotonic knot-refit byte-equality "drift" (gemini IMPROVEMENT
  I3).** Not a finding: refuted. sklearn sets
  `IsotonicRegression.X_thresholds_[0] == X_min_` and `[-1] == X_max_`,
  so refitting on the persisted knots reproduces identical clipping
  bounds. The `torch.equal` JSON round-trip tests
  (`test_isotonic_*_roundtrip`) pass and the Phase 5 qa swarm
  experimentally measured max-abs-diff `0.0` across the round trip.
  No code change.

Phase 6a (estimator Claude swarm, round 1):

- **Random-split calibration-fold seam not estimator-tested
  (qa-opus/sonnet IMPROVEMENT).** Deferred: no Phase 6a deliverable
  test names a random-split calibration test, and two architecture
  reviewers verified the logic is deterministically correct,
  `splits._random_split` uses `np.arange(n)` with no RNG, so the
  estimator's `compute_three_way_split` recomputation yields the exact
  fold the Trainer held out for both `time_ordered` and `random`. The
  default-path (time_ordered) calibration round-trip is covered;
  adding a random-split estimator test is a non-core robustness
  addition, not a correctness gap.
- **Double / triple panel windowing (arch IMPROVEMENT).** Deferred:
  `Trainer.fit` transforms `X`, the estimator transforms again for the
  calibration fold and again per `predict`. Caching the fold between
  the Trainer and the estimator requires the frozen Phase-4 Trainer to
  expose its held-out batch + `cal_idx` (a Phase-4 API change outside
  the Phase 6a module list); v1 panels are small and this is a
  performance, not correctness, concern. Revisit when a Trainer-seam
  refactor is in scope.

Phase 6a (estimator Claude swarm, round 2):

- **`_calibration_fold` recomputed twice when both a calibrator and
  `threshold_tuning` are set (arch-opus IMPROVEMENT-2).** Deferred:
  `_fit_calibrator` and the classifier `_post_fit` each call
  `_calibration_fold` (a transform + forward over the panel). Same
  rationale as the double/triple-windowing deferral above, a small-panel
  perf cost with no correctness impact; threading the computed
  `(raw, targets)` through both call sites is the same Trainer-seam
  refactor and is revisited together.
- **Encoder-vocab persistence unverified at the integration boundary
  (qa-opus IMPROVEMENT).** Deferred: every estimator e2e/subprocess
  panel is all-numeric, so the byte-equal reload proves persisted
  scaler-stats but not categorical encoder vocab. The TTS
  encoder serialize/deserialize is 100% line+branch covered at the
  unit level (`tests/unit/data`); a categorical integration panel is a
  non-core robustness addition, not a correctness gap.
- **`predict` returns class 0 for below-floor binary entities
  (code-sonnet I1).** Deferred: not a defect. requirements.md scopes
  the NaN-in-output contract to `predict_proba` / `predict_quantiles`
  (a label array cannot carry NaN). `predict` calls `predict_proba`
  internally, so the aggregated `min_periods_predict_breach` WARNING
  still fires for `predict` callers via `transform`; the signal is not
  lost.
- **Verbose clone-wiring comment in `_base.py` (style-opus
  IMPROVEMENT-2).** Deferred: the comment documents the non-obvious
  sklearn-contract reason the A4-draft clone-in-`__init__` was dropped
  (a regression guard); trimming it risks losing the rationale.
  style-opus rated it defer-acceptable.

Phase 6a (estimator Claude swarm, round 3):

- **Trainer-vs-estimator cal-fold cross-check under sentinel-drop
  (qa-opus I2).** Deferred: the new mutation-sensitive test asserts the
  estimator-side invariant (the recomputed fold is exactly
  `cal_idx` minus below-floor windows). Cross-checking that this equals
  the Trainer's actual held-out cal rows requires the frozen Phase-4
  Trainer to expose its held-out batch + `cal_idx`, the same Phase-4
  API change deferred for the double/triple-windowing item. The split
  is a pure deterministic function of identical inputs (verified by two
  architecture reviewers across rounds 1-3), so the estimator-side
  assertion is sufficient for v1; revisit with the Trainer-seam
  refactor.

Phase 6b (family-base Claude swarm, round 1):

- **`entity_id` decode to original id (arch IMPROVEMENT).** Deferred to
  Phase 7: the diagnostics field carries the internal contiguous code;
  decoding to the user-facing id needs the transformer's id inverse,
  which `TFTClassifier` / `TFTRegressor` wire up in Phase 7. The v1
  contract (documented in the dataclass comment and A15.1) is the
  internal code; no v1 requirement asks for the decoded id here.
- **`_predict_raw` head no-grad-scope mutation test (qa-opus I3).**
  Deferred: the head runs under `torch.no_grad()` and the returned
  tensor is immediately `.detach()`-ed, so the grad scope is
  immaterial to every observable output (predictions, save/load,
  determinism are all value-level and already pinned). A test that
  fails only on the grad graph would assert an internal detail with no
  user-visible contract; low value for the maintenance cost.

Post-Gemini confirming swarm (Phase 1-6 integration):

- **Factor the C1 loss-restore into a base classmethod (arch-opus
  IMPROVEMENT-3).** Deferred to Phase 7: the `if obj.loss is None ...`
  restore is inline in `load()`. A17 designates `_restore_family_state`
  / `_build_backbone_head` / `_build_calibrator_from` as the family
  override surface, NOT `load()` itself, and no v1 family overrides
  `load()`. Extracting a `_restore_default_loss_adapter` hook is
  speculative until a Phase-7 family actually overrides `load`; revisit
  then so the abstraction is shaped by a real second caller rather than
  guessed. The A17 pseudocode now documents the step so a Phase-7
  author cannot miss it.

Phase 7 (TFT concrete Claude swarm, round 1):

- **`prediction_readout="mean_pool"` TFT e2e: SUPERSEDED.** Round 1
  deferred this; Round 2 resolved it (see the round-2 Addressed block:
  `test_tft_classifier_mean_pool_readout_roundtrip`). The round-1
  deferral reason (config passthrough, low signal) was wrong: it is the
  estimator's `_config_kwargs` wiring, not a pure passthrough.
- **Regressor empty-categorical e2e (qa-opus IMPROVEMENT, round 2).**
  Deferred: the empty-categorical cardinality wiring lives in the
  SHARED `_TFTEstimatorMixin._build_tft_backbone`, already pinned end
  to end by `test_tft_classifier_no_categorical_columns`. `TFTRegressor`
  differs from `TFTClassifier` only in the head + family base (not the
  backbone cardinality path), and both regressor modes (point +
  quantile) already round-trip via the e2e. A regressor-side empty-cat
  e2e would re-exercise identical shared mixin code for a different
  head; near-zero marginal signal for the ~1-2 min/pass cost. Revisit
  if a family-specific empty-side branch ever appears.
- **`_restore_default_loss_adapter` base classmethod (arch-opus
  IMPROVEMENT-3, re-evaluated in Phase 7).** Still deferred: Phase 7
  composes a mixin but does NOT override `load()`, so there is still no
  second caller to shape the abstraction. The inline `load()`
  loss-restore + the now-also-inline `tabular_config` re-merge remain
  the single call site; extracting a hook is premature until a v2/v3
  family overrides `load`. The A17 pseudocode documents both steps so a
  future author cannot miss them.
