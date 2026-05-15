# Hyperparameter exposure strategy

## Requirements

This document specifies the architecture for hyperparameter exposure
across seq-sklearn's lifetime. It is graded against:

1. **Benchmark headroom**: deep-sequence libraries are made or broken by
   their ability to compete on standard benchmarks. Benchmark performance
   is dominated by hyperparameter tuning. The library must let users tune
   comprehensively without us shipping every knob on day one.
2. **First-use ergonomics**: a new user reading the README quickstart
   should see a small, well-documented set of hyperparameters. Field
   counts on `TFTConfig` directly affect the `mkdocstrings` rendered
   field-table size (per architecture A12 / griffe-pydantic); a kitchen-
   sink page is a documented usability cost.
3. **Forward additivity**: new hyperparameters discovered through
   internal benchmark testing must land additively (MINOR bumps under
   the project's semver, per `docs/requirements.md` versioning policy)
   without breaking existing callers, search-space code, save / load
   artifacts, or documentation.
4. **No box-in-a-corner**: nothing in v1 may force a MAJOR bump to
   expose a hyperparameter that future benchmark testing shows is
   important.
5. **sklearn-compatible mutation**: `set_params(optimizer__learning_rate=0.001)`
   must work through nested sub-configs without further plumbing.
6. **Optuna-compatible sampling**: `suggest_params` must continue to
   produce configs closed under the F5 validity matrix; expanding the
   search space to additional hyperparameters is an opt-in operation.
7. **Promotion discipline**: when an internal benchmark identifies a
   hyperparameter that moves the needle, the process to expose it must
   be documented, low-cost, and produce a CHANGELOG entry.
8. **No silent breakage on promotion**: when an `extra` dict key is
   promoted to a typed field, the dict path keeps working (as a
   documented deprecation alias) so existing callers see no behavior
   change.
9. **Save / load type fidelity**: anything passed via the `extra` dict
   escape hatch must survive `model_dump()` → JSON → `model_validate()`
   round-trip with identical types. Save / load failures or silent
   type coercion are prohibited.

CRITICAL findings against this design must trace to one of the above.

## Context

seq-sklearn ships TFT in v1 plus six more deep-sequence models in v2 / v3
(PatchTST, TimesNet, TST, LSTM, GRU, LSTM-FCN). Each is performance-
sensitive in different ways:

- TFT: VSN gating, attention head count, dropout taxonomy
  (recurrent / variational / weight-drop), prediction readout
- PatchTST: patch length, stride, channel-independent vs. mixed
- TimesNet: FFT period detection threshold, top-k periods, 2D
  conv kernel
- TST: positional encoding choice (sinusoidal vs. learned), normalization
  (LayerNorm vs. BatchNorm), classifier head pooling
- LSTM / GRU: bidirectional, dropout kind (weight-drop default per
  research), hidden-init strategy, BPTT window
- LSTM-FCN: branch width, FCN kernel sizes, branch fusion

Plus loss-side and optimizer-side hyperparameters that move benchmarks
across every model family:

- Optimizer: AdamW betas / eps, SGD nesterov / momentum, weight-decay
  schedule
- Scheduler: OneCycleLR `pct_start` / `div_factor` / `final_div_factor`,
  ReduceLROnPlateau `factor` / `threshold` / `patience`, cosine
  `min_lr`
- Loss: focal `alpha` (currently hardcoded `None`), label smoothing,
  pinball loss `delta` (Huber-quantile)
- Sampler: `replacement` flag, per-class weight strategy

Forty-plus hyperparameters across the library is a conservative
estimate. The flat-with-inheritance config shape (currently
`BaseTrainingConfig` 18 fields, `BaseModelConfig` +10, `TFTConfig` +6,
verified against `src/seq_sklearn/config/*.py`) hits sixty fields by v3
if we just keep extending. That is the box-in-a-corner risk we are
designing against.

## Current state (post-Phase 1)

The Phase 1 work ships these pydantic v2 frozen models:

```
BaseTrainingConfig (18 fields, verified at src/seq_sklearn/config/base.py:27-47)
├── learning_rate, weight_decay
├── batch_size, max_epochs
├── optimizer: Literal["adamw", "adam", "sgd"]
├── scheduler: Literal["constant", "cosine_with_warmup", "one_cycle", "reduce_on_plateau"]
├── warmup_steps, gradient_clip_val, accumulate_grad_batches, precision
├── early_stopping_patience, val_check_interval
├── val_fraction, cal_fraction, val_split_strategy
├── num_workers, pin_memory, seed, verbose
└── (frozen, extra=forbid)

BaseModelConfig(BaseTrainingConfig)  (+10 fields)
├── task_type, loss_strategy
├── imbalance_strategy, calibration_strategy
├── threshold_tuning, threshold_metric
├── focal_gamma, huber_delta
├── quantiles, oversample_ratio
└── (validity-matrix + quantiles-monotone + val-cal-sum validators)

TFTConfig(BaseModelConfig)  (+6 fields)
├── hidden_size, attention_heads
├── dropout, variable_selection_dropout
├── prediction_readout
├── tabular_config: TabularToSequenceConfig
└── (heads-divide-hidden validator)
```

Total: 18 + 10 + 6 = **34 fields** on the TFT user-facing surface. The
adapter pattern `TabularConfigParams` (A4 step 3) mirrors
`TabularToSequenceConfig` for sklearn mutation; flat training / model
fields land directly on the estimator's `__init__`.

The failure modes if we keep extending this shape:

- **Family-of-options scattering**: optimizer-specific fields
  (`weight_decay`, `learning_rate`) sit alongside optimizer-agnostic
  fields (`precision`, `seed`). Adding AdamW `betas` and `eps` plus
  SGD `momentum` and `nesterov` plus OneCycleLR `pct_start` etc.
  produces no namespace.
- **No promotion mechanism**: a new hyperparameter discovered via
  benchmark testing either lands as a top-level field (permanent;
  expensive to remove) or stays internal (untunable; benchmark
  contribution wasted).
- **Optuna search-space discoverability**: `suggest_params` produces a
  flat namespace; users cannot easily target "only the architecture
  hyperparameters" or "only the optimizer hyperparameters" without
  hand-listing field names.

## Proposal: four-tier hierarchy

```
Tier 1: Family sub-configs (nested, factored by concern)
    OptimizerConfig, SchedulerConfig, LossConfig, SamplerConfig
    + matching BaseEstimator adapters for sklearn mutation

Tier 2: Main model config (the README-quickstart surface)
    BaseTrainingConfig (cross-cutting fields)
    BaseModelConfig (task / loss / imbalance / calibration nested)
    <Model>Config extends with model-shape fields

Tier 3: Advanced model sub-config (BETA, opt-in)
    <Model>AdvancedConfig (experimental knobs gated as BETA)

Tier 4: `extra: dict[str, ExtraValue]` escape hatch
    Every sub-config has it; ALPHA-tier knobs pass through here
```

### Tier 1: family sub-configs

Each family-of-options collapses into a sub-config that owns its name,
its tunable defaults, and its escape hatch.

```python
# src/seq_sklearn/config/optimizer.py
class OptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Literal["adamw", "adam", "sgd"] = "adamw"
    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    # AdamW / Adam:
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = Field(default=1e-8, gt=0.0)
    # SGD:
    momentum: float = Field(default=0.9, ge=0.0, lt=1.0)
    nesterov: bool = False
    # ALPHA escape hatch; see Tier 4 below
    extra: ExtraDict = ()


# src/seq_sklearn/config/scheduler.py
class SchedulerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Literal["constant", "cosine_with_warmup", "one_cycle", "reduce_on_plateau"] = (
        "cosine_with_warmup"
    )
    warmup_steps: int = Field(default=100, ge=0)
    # OneCycleLR:
    pct_start: float = Field(default=0.3, gt=0.0, lt=1.0)
    div_factor: float = Field(default=25.0, gt=0.0)
    final_div_factor: float = Field(default=1e4, gt=0.0)
    # ReduceLROnPlateau:
    plateau_factor: float = Field(default=0.5, gt=0.0, lt=1.0)
    plateau_patience: int = Field(default=5, ge=1)
    plateau_threshold: float = Field(default=1e-4, gt=0.0)
    # Cosine:
    min_lr: float = Field(default=0.0, ge=0.0)
    extra: ExtraDict = ()


# src/seq_sklearn/config/loss.py
class LossConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # No default: legal value depends on task_type; the F5 validity
    # matrix on BaseModelConfig gates the (task_type, strategy) pair.
    strategy: Literal["cross_entropy", "focal", "mse", "mae", "huber", "pinball"]
    focal_gamma: float = Field(default=2.0, gt=0.0)
    focal_alpha: float | None = None        # ALPHA→BETA candidate
    huber_delta: float = Field(default=1.0, gt=0.0)
    label_smoothing: float = Field(default=0.0, ge=0.0, lt=1.0)
    extra: ExtraDict = ()


# src/seq_sklearn/config/sampler.py
class SamplerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: Literal["none", "class_weighted", "oversample_minority", "undersample_majority"] = (
        "none"
    )
    oversample_ratio: float = Field(default=1.0, gt=0.0)
    replacement: bool = True
    extra: ExtraDict = ()
```

### Tier 2: main configs

`BaseTrainingConfig` and `BaseModelConfig` shrink. Cross-cutting fields
(precision, seed, val_fraction, etc.) stay flat; family-of-options fields
nest.

```python
class BaseTrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_size: int = Field(default=64, ge=1)
    max_epochs: int = Field(default=50, ge=1)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    gradient_clip_val: float | None = None
    accumulate_grad_batches: int = Field(default=1, ge=1)
    precision: Literal["bf16-mixed", "16-mixed", "32-true", "auto"] = "auto"
    early_stopping_patience: int = Field(default=10, ge=1)
    val_check_interval: float = Field(default=1.0, gt=0.0)
    val_fraction: float = Field(default=0.1, ge=0.0, lt=1.0)
    cal_fraction: float = Field(default=0.1, ge=0.0, lt=1.0)
    val_split_strategy: Literal["time_ordered", "random"] = "time_ordered"
    num_workers: int | None = None
    pin_memory: bool | None = None
    seed: int = 42
    verbose: bool = True


class BaseModelConfig(BaseTrainingConfig):
    task_type: Literal[...]
    loss: LossConfig                         # nested; no default since strategy depends on task
    sampler: SamplerConfig = Field(default_factory=SamplerConfig)
    calibration_strategy: Literal[
        "none", "temperature", "platt", "isotonic", "conformal", "isotonic_quantile"
    ] = "none"
    threshold_tuning: bool = False
    threshold_metric: Literal["f1", "balanced_accuracy", "youden_j"] = "f1"
    quantiles: tuple[float, ...] | None = None

    @model_validator(mode="after")
    def _check_validity_matrix(self) -> Self:
        check_combo(
            self.task_type,
            self.loss.strategy,
            self.sampler.strategy,
            self.calibration_strategy,
        )
        return self
```

### Tier 3: advanced model sub-config

Each concrete model gets a paired `<Model>AdvancedConfig`. Empty in v1;
populated as internal benchmark testing identifies needle-movers.

```python
# src/seq_sklearn/config/tft.py
class TFTAdvancedConfig(BaseModel):
    """BETA per requirements stability tiers. Fields here may change
    defaults or be renamed without a MAJOR bump; consult the CHANGELOG.
    Pass an explicit instance to opt into experimental knobs.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Populated as benchmark testing identifies wins. v1 ships empty
    # plus the `extra` escape hatch.
    extra: ExtraDict = ()


class TFTConfig(BaseModelConfig):
    hidden_size: int = Field(default=128, ge=1)
    attention_heads: int = Field(default=4, ge=1)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    variable_selection_dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    prediction_readout: Literal["last_valid", "mean_pool"] = "last_valid"
    tabular_config: TabularToSequenceConfig
    advanced: TFTAdvancedConfig = Field(default_factory=TFTAdvancedConfig)
```

The `advanced` field is non-optional (defaults to the empty
`TFTAdvancedConfig()` instance) so the access pattern
`cfg.advanced.<field>` is type-stable without `Optional` plumbing.

### Tier 4: `extra` escape hatch

Every sub-config (`OptimizerConfig`, `SchedulerConfig`, `LossConfig`,
`SamplerConfig`, `<Model>AdvancedConfig`) carries an `extra` field. This
is the ALPHA-tier landing zone: untyped beyond a JSON-safe value union,
undocumented except in the CHANGELOG, discoverable only through
experimentation.

To satisfy Requirements 8 (no silent breakage) and 9 (save / load type
fidelity), `extra` is:

```python
# src/seq_sklearn/config/_extras.py
ExtraValue = str | int | float | bool | None
ExtraDict = Annotated[
    tuple[tuple[str, ExtraValue], ...],
    BeforeValidator(_normalize_extras),
]


def _normalize_extras(v: object) -> tuple[tuple[str, ExtraValue], ...]:
    """Coerce dict / Mapping / iterable-of-pairs input to a sorted
    hashable tuple. Mirrors the pattern at
    src/seq_sklearn/config/tabular.py::_normalize_embed_dims.

    Raises TypeError for values outside the documented ExtraValue
    union so unserializable types fail at construction, not at save
    time.
    """
    if v in (None, (), {}):
        return ()
    items = (
        list(v.items()) if isinstance(v, Mapping)
        else list(v)
    )
    for k, val in items:
        if not isinstance(k, str):
            raise TypeError(f"extra key must be str, got {type(k).__name__}")
        # `bool` is a subclass of `int`, so the four-type tuple covers all
        # five documented ExtraValue types without redundancy.
        if not isinstance(val, (str, int, float, type(None))):
            raise TypeError(
                f"extra value for key {k!r} must be one of "
                f"str/int/float/bool/None; got {type(val).__name__}. "
                "Nested structures and custom objects are not supported "
                "in the `extra` escape hatch."
            )
    return tuple(sorted(items))
```

Properties:

- **Restricted value type** (`str | int | float | bool | None`):
  guaranteed JSON round-trip without coercion. A `tuple` input raises
  `TypeError`; if a power user needs a tuple-valued knob, that is the
  signal to promote the knob to a typed field.
- **Stored as sorted `tuple[tuple[str, ExtraValue], ...]`**: hashable
  (so the frozen pydantic model stays hashable, per the
  `categorical_embed_dims` precedent at
  `src/seq_sklearn/config/tabular.py`).
- **Construction-time validation**: invalid types fail at config build,
  not on save. A user passing a `numpy.ndarray` sees the failure
  immediately.
- **Save / load mode pin**: the save path uses
  `config.model_dump(mode="json")`, which serializes tuples as JSON
  arrays. On `model_validate()`, the `BeforeValidator` reconstructs
  the sorted tuple of tuples from the nested-list input. This mode is
  pinned so a future change to `mode="python"` (which preserves Python
  tuples) does not silently shift the on-disk schema; the
  `test_extra_dict_survives_json_roundtrip` test pins the
  `mode="json"` path.

Access via `dict(cfg.optimizer.extra)` round-trips the contents. Factory
sites consume the tuple directly or convert as needed:

```python
# src/seq_sklearn/config/optimizer.py
# Reserved set contains ONLY keys that collide with positional or typed
# kwargs at the build call site. Untyped torch kwargs (maximize, foreach,
# capturable, differentiable, fused, dampening, etc.) are the legitimate
# ALPHA-tier passthrough use case; the escape hatch must NOT block them.
# When such a key gets promoted to a typed field via the ALPHA→BETA
# process, it moves into this reserved set and the typed field handles it.
_RESERVED_BY_OPTIMIZER: dict[str, frozenset[str]] = {
    "adamw": frozenset({"params", "lr", "weight_decay", "betas", "eps"}),
    "adam":  frozenset({"params", "lr", "weight_decay", "betas", "eps"}),
    "sgd":   frozenset({"params", "lr", "weight_decay", "momentum", "nesterov"}),
}


class OptimizerConfig(BaseModel):
    # ... fields as documented above ...

    @model_validator(mode="after")
    def _check_extra_not_reserved(self) -> Self:
        """Reserved-keys collision check lives at the config layer (not at
        build_optimizer time) so the error surfaces at construction. Phase
        1 tests the check via `OptimizerConfig(...)` raising
        ValidationError; build_optimizer (Phase 4) trusts the validator
        and does not re-check."""
        reserved = _RESERVED_BY_OPTIMIZER[self.name]
        extra_keys = {k for k, _ in self.extra}
        clashes = reserved & extra_keys
        if clashes:
            raise ValueError(
                f"extra keys collide with typed {self.name} kwargs: "
                f"{sorted(clashes)}. Set the typed OptimizerConfig field "
                "directly."
            )
        return self


# src/seq_sklearn/training/optimizers.py (Phase 4)
def build_optimizer(cfg: OptimizerConfig, params) -> torch.optim.Optimizer:
    # cfg has already been validated at construction; build_optimizer
    # trusts the config-layer validators (reserved-keys collision,
    # extras normalization, etc.) and does not duplicate them.
    extra = extract_deprecated_extras(cfg, "optimizer")
    if cfg.name == "adamw":
        return torch.optim.AdamW(
            params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
            betas=cfg.betas, eps=cfg.eps, **extra,
        )
    # ... sgd / adam analogous, each consuming the same OptimizerConfig
```

`SchedulerConfig`, `LossConfig`, and `SamplerConfig` carry analogous
`_check_extra_not_reserved` model_validators (each with its own narrow
reserved set against its own family's typed fields). Name collisions
raise `ValidationError` at config construction; untyped torch /
lightning / optuna kwargs pass through `extra` unrestricted. Phase 1
owns these tests because the validators ship with the configs in Phase
1; the factories (Phase 4) consume the already-validated configs.

### Deprecation alias helper

Manual `extra.pop` + `warnings.warn` logic in every factory is
error-prone (a maintainer promoting a field might forget the fallback
route, silently breaking existing user configurations). The library
ships a single helper that every factory routes through:

```python
# src/seq_sklearn/config/_extras.py
_PROMOTED_KEYS_BY_FAMILY: dict[str, dict[str, str]] = {
    "optimizer": {
        # Populated as ALPHA keys are promoted to typed fields.
        # Format: "<extra-key>": "<typed-field-name>".
        # Example after a future promotion:
        #     "amsgrad": "amsgrad",
    },
    "scheduler": {},
    "loss": {},
    "sampler": {},
}


def extract_deprecated_extras(
    cfg: BaseModel,
    family: str,
) -> dict[str, ExtraValue]:
    """Return `extra` as a dict, routing any promoted keys to the typed
    field and emitting a `DeprecationWarning` per route.

    Every family factory (build_optimizer, build_scheduler, build_loss,
    build_sampler) calls this helper instead of `dict(cfg.extra)` so the
    deprecation-alias contract lands once. A maintainer promoting an
    ALPHA key to a typed field adds one entry to
    _PROMOTED_KEYS_BY_FAMILY and the alias behavior fires automatically.
    """
    extra = dict(cfg.extra)
    promoted = _PROMOTED_KEYS_BY_FAMILY[family]
    for extra_key, typed_name in promoted.items():
        if extra_key in extra:
            existing_typed = getattr(cfg, typed_name)
            warnings.warn(
                f"Passing {extra_key!r} via {type(cfg).__name__}.extra is "
                f"deprecated; use the typed {type(cfg).__name__}.{typed_name} "
                f"field. The dict path remains a permanent alias.",
                DeprecationWarning,
                stacklevel=3,
            )
            # Promoted fields must have an explicit default (the registry
            # meta-test enforces this). The asymmetric check below: if the
            # typed value differs from its default, the caller set BOTH
            # the typed field AND the extra key, which is ambiguous. If
            # the typed value equals its default, the extra value wins
            # (existing callers see no behavior change).
            typed_default = cfg.model_fields[typed_name].default
            if existing_typed != typed_default:
                raise ConfigError(
                    f"{extra_key!r} provided via both extra and the typed "
                    f"{typed_name} field; remove one."
                )
            extra.pop(extra_key)  # consumed; typed field carries the value
    return extra
```

A meta-test asserts that every entry in `_PROMOTED_KEYS_BY_FAMILY` has a
matching typed field on the family config, so a maintainer cannot
register a promotion without the typed field existing. This catches the
"forgot the typed field" failure mode at CI rather than at first user
report.

### User-facing surface: estimator `__init__` kwargs

Per architecture A4 step 3, the estimator's `__init__` accepts a kwarg
per pydantic field through the BaseEstimator-adapter pattern. Under the
nested-config proposal, the estimator stores adapter instances for each
sub-config; callers see the standard sklearn double-underscore traversal
for nesting.

**Keyword-only adapter constructors.** Every adapter `__init__` uses the
`*` keyword-only marker. Without it, adding a BETA field via the
ALPHA → BETA promotion path would shift positional arguments and break
existing callers using `OptimizerParams("adamw", 0.01)` positionally,
making a MINOR-additive change into a MAJOR break. The marker is a hard
requirement for the promotion-path contract to hold.

```python
# src/seq_sklearn/config/_adapters.py
class OptimizerParams(BaseEstimator):
    def __init__(
        self,
        *,                                  # <-- mandatory: keyword-only
        name: Literal["adamw", "adam", "sgd"] = "adamw",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        momentum: float = 0.9,
        nesterov: bool = False,
        extra: ExtraDict = (),
    ) -> None:
        self.name = name
        # ... etc.
```

The keyword-only marker applies to every adapter:
`TabularConfigParams`, `OptimizerParams`, `SchedulerParams`,
`LossParams`, `SamplerParams`, every `<Model>AdvancedParams`. The
existing `TabularConfigParams` at
`src/seq_sklearn/config/_params_adapter.py:65` is amended to keyword-
only as part of the refactor.

**Usage:**

```python
# Construction. `loss` is optional: when omitted, _build_config injects
# a task-type-aware default (see "Task-type-aware loss default" below).
clf = TFTClassifier(
    task_type="binary",
    loss=LossParams(strategy="cross_entropy"),
    tabular_config=TabularConfigParams(),
)

# Or omit loss; _build_config supplies the default:
clf = TFTClassifier(task_type="binary", tabular_config=TabularConfigParams())

# Tune via standard sklearn double-underscore traversal:
clf.set_params(optimizer__learning_rate=3e-4)
clf.set_params(loss__strategy="focal", loss__focal_gamma=1.5)
clf.set_params(optimizer__extra=(("amsgrad", True),))   # ALPHA passthrough

# get_params produces flat sklearn-compatible keys:
clf.get_params(deep=True)["loss__strategy"]   # "focal"
clf.get_params(deep=True)["optimizer__learning_rate"]  # 3e-4
```

The estimator's `__init__` signature carries every nested sub-config
as a single adapter kwarg (e.g. `loss: LossParams | None = None`).
Fields that LIVE on a nested sub-config (e.g.
`OptimizerConfig.learning_rate`, `LossConfig.strategy`) DO NOT appear
flat at the top level; callers reach them via the adapter. Fields on
the main `<Model>Config` that are NOT nested under a sub-config (e.g.
TFT's `hidden_size`, `attention_heads`, `dropout`,
`variable_selection_dropout`, `prediction_readout`) DO appear flat
at the top level. These are model-architecture knobs documented on
`<Model>Config` directly. The split matches the v1 Phase 1 convention
for `tabular_config: TabularConfigParams | None = None`. The
trade-off is explicit: shallower top-level surface (~7 adapter kwargs
plus the model-shape flat kwargs versus 34 fully-flat kwargs today)
at the cost of one level of `LossParams(...)` wrapping for nested
fields. The sklearn search-CV ecosystem already uses double-underscore
traversal so power users tune via `set_params` which is unchanged
from today.

### Task-type-aware loss default

`LossConfig.strategy` has no default because legal values depend on
`task_type` (per F5). On the estimator's `__init__` side, `loss:
LossParams | None = None` is the type hint, and `_build_config` injects
a task-type-aware default when the caller omits `loss=`. The default
map matches the F5 "neutral" choice per task:

```python
# src/seq_sklearn/models/_base.py inside BaseSequenceEstimator._build_config:
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
        loss = LossParams(strategy=_DEFAULT_LOSS_FOR_TASK[self.task_type])
    # ... build the nested BaseModelConfig with loss.to_pydantic()
```

This preserves three contracts simultaneously: the type hint stays
truthful (`loss=None` is a legal call), `LossConfig.strategy` keeps no
default (legal values are task-dependent so a top-level default would
be wrong half the time), and `TFTClassifier(task_type="binary").fit(X, y)`
works without explicit loss specification (because the task implies
the default). The named test
`test_loss_default_injection_per_task_type` (added below) parametrizes
over every v1 task type and asserts the injected default is the one in
the map.

### `_config_to_estimator_kwargs` recursive update

Architecture A16 currently defines `_config_to_estimator_kwargs(config)`
that pops a single nested field (`tabular_config`) from
`config.model_dump()` and wraps it as a `TabularConfigParams` adapter.
Under the proposal, the helper extends to handle six nested fields per
model (`tabular_config`, `optimizer`, `scheduler`, `loss`, `sampler`,
`advanced`):

```python
# Documented adapter map per concrete model class.
_TFT_ADAPTER_MAP: dict[str, type[BaseEstimator]] = {
    "tabular_config": TabularConfigParams,
    "optimizer": OptimizerParams,
    "scheduler": SchedulerParams,
    "loss": LossParams,
    "sampler": SamplerParams,
    "advanced": TFTAdvancedParams,
}

# Registry keyed by concrete config class. v2 / v3 estimators register
# their own per-model adapter map here when they ship.
_ADAPTER_MAP_BY_CONFIG: dict[
    type[BaseModelConfig], dict[str, type[BaseEstimator]]
] = {
    TFTConfig: _TFT_ADAPTER_MAP,
}


def _adapter_map_for(config_cls: type[BaseModelConfig]) -> dict[str, type[BaseEstimator]]:
    return _ADAPTER_MAP_BY_CONFIG[config_cls]


def _config_to_estimator_kwargs(config: BaseModelConfig) -> dict[str, object]:
    # mode="json" pins the on-disk extra-tuple shape (per the
    # test_extra_dict_survives_json_roundtrip contract); the adapter
    # __init__ receives nested lists for tuples and the
    # _normalize_extras BeforeValidator reconstructs them.
    raw = config.model_dump(mode="json")
    kwargs: dict[str, object] = {}
    adapter_map = _adapter_map_for(type(config))
    # Iterate the adapter map (not `raw`) so we never mutate `raw`
    # while iterating; matches the canonical A16 implementation.
    for field_name, adapter_cls in adapter_map.items():
        sub_dict = raw.pop(field_name)
        kwargs[field_name] = adapter_cls(**sub_dict)
    return {**raw, **kwargs}
```

This generalizes A16's one-level helper to the full nested shape.
Architecture A16 will be updated alongside the implementation refactor.

## Promotion path: ALPHA → BETA → STABLE

A new hyperparameter discovered via internal benchmark testing goes
through three stages with a documented promotion gate at each step.

### Stage 1: ALPHA (escape hatch)

A maintainer identifies a knob worth tunable exposure. First exposure
is via the appropriate sub-config's `extra` tuple:

```python
config = TFTConfig(
    task_type="binary",
    loss=LossConfig(strategy="cross_entropy"),
    optimizer=OptimizerConfig(
        name="adamw",
        extra=(("amsgrad", True),),  # ALPHA: passed to torch.optim.AdamW
    ),
    tabular_config=TabularToSequenceConfig(id_col="id", time_col="t"),
)
```

CHANGELOG entry: "Added support for `amsgrad` AdamW flag via
`OptimizerConfig.extra`. ALPHA: documentation in CHANGELOG only."

### Stage 2: BETA (typed field on advanced sub-config or family config)

Benchmark testing on the synthetic DGP plus at least one external
dataset shows the hyperparameter moves the headline metric by at least
the threshold documented in `docs/benchmarks.md` (see "Benchmark gate
metrics" below). Promote to a typed field, either on the appropriate
family sub-config (`OptimizerConfig`) or on the model's
`<Model>AdvancedConfig`:

```python
class OptimizerConfig(BaseModel):
    name: Literal["adamw", ...]
    learning_rate: float = ...
    weight_decay: float = ...
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = False    # NEW: promoted ALPHA -> BETA
    momentum: float = ...
    nesterov: bool = False
    extra: ExtraDict = ()
```

CHANGELOG entry: "Promoted `amsgrad` from `OptimizerConfig.extra` to a
typed `OptimizerConfig.amsgrad: bool` field (BETA). Benchmark: ECE
improvement of 1.4% on F6 multiclass DGP. Defaults to `False`
(neutral); existing callers unaffected. BETA: defaults may change
without a MAJOR bump."

**Deprecation alias contract**: when a key is promoted, the
`extra`-dict path keeps working indefinitely as an alias. The
The `extract_deprecated_extras` helper (specified above in
"Deprecation alias helper") detects the promoted key and routes the
value to the typed field, emitting a `DeprecationWarning`. The
promotion itself is a one-line registry edit:

```python
# In src/seq_sklearn/config/_extras.py, the maintainer's only edit:
_PROMOTED_KEYS_BY_FAMILY["optimizer"]["amsgrad"] = "amsgrad"
```

The helper handles `warnings.warn`, the `extra.pop`, the typed-field
routing, and the "both paths set" `ConfigError`. The maintainer does
NOT write per-factory `if "amsgrad" in extra: ...` code; that pattern
was the bug Gemini's r1-I1 finding fixed.

This is the deprecation alias model: the dict path NEVER becomes a
silent no-op. The behavior contract is preserved indefinitely; the
warning is the only change. If the maintainers later decide the alias
is no longer worth carrying, removal requires a MAJOR bump per the
project's semver policy (and at least one MINOR cycle of
`DeprecationWarning`).

**Promoted-field default constraint**: every key registered in
`_PROMOTED_KEYS_BY_FAMILY` must map to a typed field with an explicit
default. Fields with no default (e.g. `LossConfig.strategy`) have
`FieldInfo.default == PydanticUndefined`, which makes the helper's
"both paths set" detection ambiguous. The
`test_extract_deprecated_extras_meta_promoted_keys_exist` meta-test
asserts both that the typed field exists AND that
`field_info.default is not PydanticUndefined` for every promoted key.

### Stage 3: STABLE (no further promotion needed)

After two MINOR releases at BETA without breaking changes, the field
becomes STABLE by default. Stability table in `docs/requirements.md`
adds an entry. No further code changes; the promotion is documentary.

CHANGELOG entry: "`OptimizerConfig.amsgrad` graduates from BETA to
STABLE. Default value (`False`) is now stability-guaranteed; removal
requires a MAJOR bump."

### Promotion gate criteria

Each promotion requires:

1. **Benchmark evidence**: a documented improvement in the metric
   defined for the affected `target_kind` (see "Benchmark gate
   metrics" below) on the F6 synthetic DGP at the canonical seed triple
   plus at least one external dataset.
2. **Default-neutral landing**: the default value must preserve
   existing behavior so existing callers are unaffected.
3. **CHANGELOG rationale**: the entry names the benchmark, the
   improvement, and the field's tier.
4. **Optuna search-space update**: `suggest_params` adds the field to
   its sampling space (gated by an `advanced=True` flag in v1; see
   Optuna section below).
5. **Test addition**: at least one unit test covers the promoted typed
   field plus one test exercises the deprecation alias.

### Benchmark gate metrics

Headline metrics per `target_kind` for promotion gate purposes. These
are v1.0 starting thresholds, revisited once `docs/benchmarks.md`
populates with external-dataset numbers:

| `target_kind` | Headline metric | Improvement threshold |
|---|---|---|
| `binary` | accuracy on F6 DGP | ≥ 0.5 percentage points |
| `multiclass` | macro-F1 on F6 DGP | ≥ 0.5 percentage points |
| `regression_point` | R² on F6 DGP | ≥ 0.01 absolute |
| `regression_quantile` | empirical coverage gap on 80% interval | ≥ 0.01 absolute |
| (calibration) | ECE on calibration fold | ≥ 0.5 percentage points |

The metric choices (accuracy, macro-F1, R², ECE) match the N1
acceptance test families per `docs/requirements.md` lines 1377-1392;
the absolute pass thresholds in N1 (e.g. accuracy >= 0.75) are
release-gate thresholds, while the deltas in this table are
promotion-gate thresholds. Both are calibrated against the F6 DGP
three-seed median.

"External dataset" means any dataset outside the synthetic DGP for
which the maintainers commit reproducible benchmark numbers to
`docs/benchmarks.md` (a v1.x deliverable; v1.0 ships the file as a
skeleton with section headers and no benchmark numbers yet, per the
migration plan row). External-dataset evidence may be deferred until
`docs/benchmarks.md` populates; until then, the F6 DGP three-seed-
median improvement is the sole gate.

## sklearn adapter pattern interaction

Per architecture A4 step 3, every frozen pydantic config has a paired
`BaseEstimator` adapter that mirrors its fields and exposes
`get_params` / `set_params` to sklearn. Phase 1 ships
`TabularConfigParams` for `TabularToSequenceConfig`.

The four-tier proposal adds five more adapters under
`src/seq_sklearn/config/_adapters.py` (renamed from
`_params_adapter.py` to reflect the broader scope):

- `OptimizerParams(BaseEstimator)` ← `OptimizerConfig`
- `SchedulerParams(BaseEstimator)` ← `SchedulerConfig`
- `LossParams(BaseEstimator)` ← `LossConfig`
- `SamplerParams(BaseEstimator)` ← `SamplerConfig`
- `<Model>AdvancedParams(BaseEstimator)` ← `<Model>AdvancedConfig`

Each adapter follows the existing pattern at
`src/seq_sklearn/config/_params_adapter.py:23-127`:

- Mutable `__init__` mirroring every pydantic field name and input
  type 1:1
- `to_pydantic()` constructing the frozen instance with
  `ValidationError` propagating to the caller
- Class-level type annotations so pyright carries the Literal narrowing
  through instance access (matches the `scaling_real` / `scaling_static_real`
  pattern at `src/seq_sklearn/config/_params_adapter.py:48-63`)
- Clone safety: the outer estimator's `__init__` calls
  `sklearn.base.clone(adapter)` on every incoming adapter so two
  estimators sharing an adapter instance produce independent
  configurations under `set_params`.

`get_params(deep=True)` recurses through each adapter, producing flat
double-underscore keys (`optimizer__learning_rate`, `optimizer__extra`,
`loss__focal_gamma`, etc.). `GridSearchCV` / `RandomizedSearchCV` /
`Pipeline` compose without further plumbing.

## Optuna search-space interaction

`suggest_params` continues to produce a `BaseModelConfig` (or subclass)
that is closed under the F5 validity matrix. The expansion to advanced /
nested fields is gated by two opt-in flags. The signature change is a
MINOR-bump-compatible additive change (per requirements F7); the
proposal updates F7 explicitly in the migration plan:

```python
def suggest_params(
    trial: optuna.Trial,
    model_class: type[BaseSequenceClassifier | BaseSequenceRegressor],
    base: BaseModelConfig | None = None,
    *,
    search_advanced: bool = False,
    search_extras: bool = False,
) -> BaseModelConfig:
    """Sample a config from the per-model default search space.

    By default samples only the STABLE fields enumerated in the
    "Default search space" table below. `search_advanced=True` also
    samples `<Model>AdvancedConfig` BETA fields. `search_extras=True`
    allows sampling from documented ALPHA `extra` keys per family
    (curated; not arbitrary keys)."""
```

### Default search space per model

The default search space (`search_advanced=False, search_extras=False`)
samples ONLY the fields listed below. v1 ships only TFT; v2 / v3 add
their entries when their concrete models ship.

| Model | Family sub-configs sampled | Main config fields sampled | Notes |
|---|---|---|---|
| `TFTClassifier` / `TFTRegressor` | `optimizer.learning_rate` (log-uniform 1e-5..1e-2); `optimizer.weight_decay` (log-uniform 1e-7..1e-2); `scheduler.name`; `scheduler.warmup_steps` (int 0..500); `loss.strategy` (legal cells gated by F5); `loss.focal_gamma` (uniform 1..3, when `loss.strategy=focal`); `sampler.strategy` (legal cells gated by F5) | `hidden_size` (categorical 32/64/128/256); `attention_heads` (categorical 1/2/4/8 gated by `hidden_size % heads == 0`); `dropout` (uniform 0..0.5); `variable_selection_dropout` (uniform 0..0.5); `prediction_readout` (categorical) | `tabular_config` fields are sampled only when explicitly requested by the caller per F7 |

The table is the authoritative reference; if `suggest_params` samples a
field not listed, that is a bug. Phase 8 names a regression test that
runs `suggest_params` with default flags and asserts every sampled
field belongs to the table.

`search_advanced=True` additionally samples fields on the model's
`<Model>AdvancedConfig`. In v1 those configs are empty, so the flag is
a no-op for v1; the flag exists to lock the surface for future BETA
fields without an additional MINOR bump at first use.

`search_extras=True` samples from the curated per-family ALPHA-key
list. The curated list is a maintainer-controlled enum in
`src/seq_sklearn/tuning/_alpha_keys.py`; passing the flag samples a
subset of those keys per call. v1 ships the file with empty enums.

## Stability-tier mapping

Hyperparameter stability tiers map to the existing stability tiers in
`docs/requirements.md` "Per-module stability tiers":

| Hyperparameter tier | Stability mapping | Removal cost |
|---|---|---|
| STABLE | STABLE in main config / family sub-config | MAJOR bump required |
| BETA | BETA in `<Model>AdvancedConfig` or family sub-config | MINOR bump + DeprecationWarning cycle |
| ALPHA | INTERNAL: `extra` tuple on any sub-config | No version bump; CHANGELOG-only |

Stability tiers are surfaced in three places:

1. The pydantic field's docstring carries the tier marker.
2. `docs/requirements.md` Per-module stability tiers table gets a new
   sub-section listing BETA hyperparameters per model.
3. `docs/api/` mkdocstrings render shows BETA fields under a "BETA"
   header per the griffe-pydantic integration (architecture A12).

## Tier mapping for v2 / v3 models

Model-architecture knobs (load-bearing for the model's behavior) land
on the main `<Model>Config`. Experimental knobs land on
`<Model>AdvancedConfig`. Family-of-options knobs (optimizer / scheduler /
loss / sampler) inherit from the Tier 1 sub-configs.

### Worked example: PatchTST (v2)

```python
class PatchTSTConfig(BaseModelConfig):
    # Main config: load-bearing architecture knobs (STABLE)
    hidden_size: int = Field(default=128, ge=1)
    attention_heads: int = Field(default=4, ge=1)
    n_layers: int = Field(default=3, ge=1)
    patch_length: int = Field(default=16, ge=1)            # NEW per PatchTST
    patch_stride: int = Field(default=8, ge=1)             # NEW per PatchTST
    channel_independent: bool = True                       # NEW per PatchTST
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    tabular_config: TabularToSequenceConfig
    advanced: PatchTSTAdvancedConfig = Field(
        default_factory=PatchTSTAdvancedConfig,
    )


class PatchTSTAdvancedConfig(BaseModel):
    """BETA. v2 ships empty; populated as benchmark testing identifies
    needle-movers (e.g. RevIN normalization, channel mixing weights)."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    extra: ExtraDict = ()
```

`patch_length`, `patch_stride`, and `channel_independent` are STABLE on
the main config because they are documented architecture knobs in the
PatchTST paper and reference implementations; users pick them like
they pick `hidden_size`. Future PatchTST-specific knobs (RevIN
normalization, gated mixing weights) land in `PatchTSTAdvancedConfig`.

The other v2 / v3 models follow the same partitioning:

- **TimesNet**: `n_periods_top_k` (STABLE), `mask_aware_fft` (STABLE);
  advanced reserved for FFT-detection threshold tuning.
- **TST**: `positional_encoding: Literal["sinusoidal", "learned"]`
  (STABLE); `normalization: Literal["layer", "batch"]` (STABLE);
  advanced reserved for ConvTran-style positional refinement.
- **LSTM / GRU**: `bidirectional` (STABLE), `recurrent_dropout_kind`
  (STABLE per recurrent skeleton in A6.1), `bptt_window` (STABLE);
  advanced reserved for AWD-LSTM-style weight-tying experiments.
- **LSTM-FCN**: `branch_width` (STABLE), `fcn_kernel_sizes` (STABLE);
  advanced reserved for branch-fusion variants.

## Migration plan for Phase 1

The Phase 1 work currently on branch `phase-1-foundation` ships the
flat config shape. The refactor reorganizes:

### Source-file moves and additions

| File | Action |
|---|---|
| `src/seq_sklearn/config/_extras.py` | NEW: `ExtraValue`, `ExtraDict`, `_normalize_extras` |
| `src/seq_sklearn/config/optimizer.py` | NEW: `OptimizerConfig` |
| `src/seq_sklearn/config/scheduler.py` | NEW: `SchedulerConfig` |
| `src/seq_sklearn/config/loss.py` | NEW: `LossConfig` |
| `src/seq_sklearn/config/sampler.py` | NEW: `SamplerConfig` |
| `src/seq_sklearn/config/base.py` | EDIT: shrink BaseTrainingConfig / BaseModelConfig; nest family configs; `_check_validity_matrix` reads `self.loss.strategy` and `self.sampler.strategy` |
| `src/seq_sklearn/config/tft.py` | EDIT: add `TFTAdvancedConfig` + `advanced` field on `TFTConfig` |
| `src/seq_sklearn/config/_validity.py` | EDIT: `check_combo` signature unchanged (still 4 strings); call site update only |
| `src/seq_sklearn/config/_adapters.py` | RENAME from `_params_adapter.py`; add 5 more adapter classes |

### Test-file moves and additions

| File | Action |
|---|---|
| `tests/unit/config/test_base.py` | EDIT: rewrite `_legal_kwargs` to use nested sub-config constructors |
| `tests/unit/config/test_validity_matrix.py` | EDIT: rewrite parametrized `BaseModelConfig` construction to use nested `LossConfig(strategy=loss)`, `SamplerConfig(strategy=imb)`; parametrize IDs change shape (documented below) |
| `tests/unit/config/test_params_adapter.py` | RENAME to `test_adapters.py`; add tests per adapter |
| `tests/unit/config/test_optimizer.py` | NEW |
| `tests/unit/config/test_scheduler.py` | NEW |
| `tests/unit/config/test_loss.py` | NEW |
| `tests/unit/config/test_sampler.py` | NEW |
| `tests/unit/config/test_extras.py` | NEW: covers `_normalize_extras` type validation, hash stability, JSON round-trip |

### Architecture and requirements document updates

| File | Action |
|---|---|
| `docs/architecture.md` A4 (~lines 240-470) | REWRITE: document the four-tier hierarchy; A4 step 3 generalizes to multi-adapter recursion |
| `docs/architecture.md` A16 (~lines 1601-1737) | EDIT: `_config_to_estimator_kwargs` updated to handle multi-nested fields per the adapter-map pattern; `suggest_params` signature gains `search_advanced` / `search_extras` |
| `docs/architecture.md` A20 | ADD: open question for `extra` curated-alpha-key list |
| `docs/requirements.md` F7 | EDIT: `suggest_params` signature updated to include the two flags; FixedTrial vs. study.ask() guidance unchanged |
| `docs/requirements.md` "TFT hyperparameters" (~lines 1326-1341) | EDIT: replace flat enumeration with nested-config reference |
| `docs/requirements.md` Per-module stability tiers (~lines 240-256) | EDIT: add hyperparameter-tier sub-section |
| `docs/implementation_plan.md` Phase 1 | EDIT: add the new modules and test files; update effort estimate |
| `docs/benchmarks.md` | NEW (skeleton): headline metrics per task_type; populated as external benchmarks land |

### Named tests added in the Phase 1 refactor

The test-file table above lists scope; this table names every test
function the refactor must land so a Phase 9 verification pass can grep
for each name. Tests that ship with the refactor itself are marked
"Phase 1"; tests deferred until the matching feature lands are marked
"Phase 8" or "first promotion".

| Test file | Test function | When | Intent |
|---|---|---|---|
| `tests/unit/config/test_adapters.py` | `test_tabular_config_params_clone_is_independent` | Phase 1 | `sklearn.base.clone(TabularConfigParams(...))` produces an independent instance; mutating the clone via `set_params` does not affect the original. Carries forward and renames the pre-existing `test_params_adapter.py` clone test under the renamed `test_adapters.py`. |
| `tests/unit/config/test_adapters.py` | `test_optimizer_params_clone_is_independent` | Phase 1 | `sklearn.base.clone(OptimizerParams(...))` produces an independent instance; mutating the clone via `set_params` does not affect the original. |
| `tests/unit/config/test_adapters.py` | `test_scheduler_params_clone_is_independent` | Phase 1 | Same isolation contract for `SchedulerParams`. |
| `tests/unit/config/test_adapters.py` | `test_loss_params_clone_is_independent` | Phase 1 | Same isolation contract for `LossParams`. |
| `tests/unit/config/test_adapters.py` | `test_sampler_params_clone_is_independent` | Phase 1 | Same isolation contract for `SamplerParams`. |
| `tests/unit/config/test_adapters.py` | `test_tft_advanced_params_clone_is_independent` | Phase 1 | Same isolation contract for `TFTAdvancedParams`. |
| `tests/unit/config/test_adapters.py` | `test_outer_estimator_clone_does_not_alias_adapter_instances` | Phase 6a | `sklearn.base.clone` on the outer estimator produces fresh adapter instances; mutating one does not affect the cloned estimator. |
| `tests/unit/config/test_extras.py` | `test_extra_dict_rejects_non_primitive_value` | Phase 1 | `OptimizerConfig(extra={"k": numpy.array([1])})` raises `TypeError` naming the offending type. |
| `tests/unit/config/test_extras.py` | `test_extra_dict_round_trips_each_primitive_type` | Phase 1 | Construct `OptimizerConfig(extra={"s": "v", "i": 1, "f": 1.5, "b": True, "n": None})`, call `model_dump(mode="json")`, serialize / deserialize JSON, `model_validate()`, assert tuple equality with the input. |
| `tests/unit/config/test_extras.py` | `test_extra_dict_stored_as_sorted_tuple` | Phase 1 | Two constructions with reversed key order produce identical stored tuples and identical hashes. |
| `tests/unit/config/test_extras.py` | `test_extra_dict_survives_json_roundtrip` | Phase 1 | A config with `extra=(("flag", True), ("count", 3))` round-trips through `model_dump(mode="json")` + `json.dumps` + `json.loads` + `model_validate` with identical types. |
| `tests/unit/config/test_extras.py` | `test_extract_deprecated_extras_mock_promotion_emits_warning` | Phase 1 | Mock-promote a fake key into `_PROMOTED_KEYS_BY_FAMILY` (test-local monkeypatch) and a corresponding typed field on a stub config; pass `extra=(("fake", True),)`; assert `DeprecationWarning` matches `r"deprecated.*fake"` and the typed field carries the value. Pins the alias machinery in Phase 1 even though no real promotion has occurred yet. |
| `tests/unit/config/test_extras.py` | `test_extra_path_after_promotion_emits_deprecation_warning` | First promotion (deferred) | Lands when the first ALPHA → BETA promotion occurs; constructs a config with `extra=(("amsgrad", True),)` after `amsgrad` has been promoted; asserts `DeprecationWarning` matching `r"deprecated.*amsgrad"` against the real production-config field. |
| `tests/unit/config/test_extras.py` | `test_extract_deprecated_extras_meta_promoted_keys_exist` | Phase 1 | For every family in `_PROMOTED_KEYS_BY_FAMILY`, every promoted-key value names a real typed field on the family config. Catches a maintainer who registers a promotion without adding the typed field. |
| `tests/unit/config/test_extras.py` | `test_extract_deprecated_extras_both_typed_and_extra_raises_config_error` | Phase 1 | Caller sets a promoted typed field AND passes the same key via `extra`; assert `ConfigError`. Pins the ambiguous-configuration contract. |
| `tests/unit/config/test_adapters.py` | `test_all_adapters_have_keyword_only_init` | Phase 1 | Inspect every adapter's `__init__` signature via `inspect.signature(...)`; assert every parameter except `self` has `kind == POSITIONAL_OR_KEYWORD` with `default != Parameter.empty` AND that `*` keyword-only marker is present (the first non-self parameter has `kind == KEYWORD_ONLY`). Pins the promotion-path positional-shift contract. |
| `tests/unit/models/test_loss_default_injection.py` | `test_loss_default_injection_per_task_type` | Phase 6a | Parametrize over v1 task types only (`binary`, `multiclass`, `regression_point`, `regression_quantile`; v1.1 entries in the `_DEFAULT_LOSS_FOR_TASK` map are out of scope until v1.1 ships); assert `_DEFAULT_LOSS_FOR_TASK[task]` matches the injected `LossConfig.strategy` when the estimator is constructed without `loss=`. Pins the task-type-aware default contract from the "Task-type-aware loss default" subsection. |
| `tests/unit/config/test_optimizer.py` | `test_adamw_reserved_keys_collision_raises` | Phase 1 | `OptimizerConfig(name="adamw", extra=(("lr", 0.1),))` raises `ValidationError` from the `_check_extra_not_reserved` model_validator at config construction. |
| `tests/unit/config/test_optimizer.py` | `test_sgd_reserved_keys_collision_raises` | Phase 1 | Same as above for SGD; `OptimizerConfig(name="sgd", extra=(("momentum", 0.5),))` raises `ValidationError` because `momentum` collides with the typed SGD field. |
| `tests/unit/config/test_extras.py` | `test_extract_deprecated_extras_happy_path_passes_through` | Phase 1 | Pass an `OptimizerConfig` with `extra=(("amsgrad", True),)` BEFORE any promotion has occurred (the registry is empty for `optimizer`); assert `extract_deprecated_extras` returns `{"amsgrad": True}` unchanged with no `DeprecationWarning`. Pins the non-promoted-key passthrough contract. |
| `tests/unit/config/test_optimizer.py` | `test_default_construction_uses_documented_defaults` | Phase 1 | `OptimizerConfig()` produces `name="adamw"`, `learning_rate=1e-3`, etc. matching the documented table. |
| `tests/unit/config/test_scheduler.py` | `test_default_construction_uses_documented_defaults` | Phase 1 | Same shape for scheduler defaults. |
| `tests/unit/config/test_loss.py` | `test_construction_requires_strategy` | Phase 1 | `LossConfig()` without `strategy=` raises `ValidationError` per the no-default contract. |
| `tests/unit/config/test_sampler.py` | `test_default_strategy_is_none` | Phase 1 | `SamplerConfig()` produces `strategy="none"` per F5 default. |
| `tests/unit/config/test_tft.py` | `test_tft_config_advanced_field_is_not_none_by_default` | Phase 1 | `TFTConfig(...)` constructed without `advanced=` has `cfg.advanced` non-`None` and `isinstance(cfg.advanced, TFTAdvancedConfig)`. |
| `tests/unit/config/test_tft.py` | `test_tft_advanced_config_default_construction_succeeds` | Phase 1 | `TFTAdvancedConfig()` validates with no arguments; `cfg.extra == ()`. |
| `tests/unit/tuning/test_suggest_params.py` | `test_suggest_params_default_flags_exclude_advanced_fields` | Phase 8 | Run 100 trials with `search_advanced=False, search_extras=False`; assert every sampled config has `advanced.extra == ()` and no key in the curated ALPHA list appears. |
| `tests/unit/tuning/test_suggest_params.py` | `test_suggest_params_search_advanced_true_accepts_flag` | Phase 8 | Run with `search_advanced=True` on TFT (v1 advanced is empty); assert no exception and the flag is honored (no advanced fields sampled because the config carries none in v1). |
| `tests/unit/tuning/test_suggest_params.py` | `test_suggest_params_sweeps_only_default_fields` | Phase 8 | Run 1000 trials with default flags; assert every sampled field name appears in the "Default search space per model" table. |
| `tests/unit/tuning/test_config_to_estimator_kwargs.py` | `test_config_to_estimator_kwargs_round_trips_all_adapters` | Phase 8 | Construct a `BaseModelConfig` with non-default values in every adapter sub-config (including a non-empty `extra` tuple), call `_config_to_estimator_kwargs`, assert every field name maps to the correct adapter class and the scalar fields pass through unchanged. |
| `tests/unit/tuning/test_config_to_estimator_kwargs.py` | `test_config_to_estimator_kwargs_extra_tuple_type_survives` | Phase 8 | Call `_config_to_estimator_kwargs` on a config with `optimizer=OptimizerConfig(extra=(("amsgrad", True),))`; assert the resulting `OptimizerParams` adapter's `extra` attribute equals `(("amsgrad", True),)` (post-`BeforeValidator` reconstruction), verifying that `mode="json"` serialization round-trips correctly through adapter construction. |

The `Phase 1` rows ship with the four-tier refactor itself. The
`Phase 6a` and `Phase 8` rows ship when their containing phases land
per `docs/implementation_plan.md`. The "first promotion" row is the
deferred-deprecation-warning case from the Round 1 ledger.

### Test parametrize-ID stability

The validity-matrix test parametrization is over the Cartesian product
of `TASK_TYPES * LOSS_STRATEGIES * IMBALANCE_STRATEGIES *
CALIBRATION_STRATEGIES` (per requirements F5). The pytest parameter IDs
today look like `binary-cross_entropy-class_weighted-none`. After the
refactor, the test body constructs nested configs but the
parametrization tuple itself is unchanged (still 4 strings). The
pytest parameter IDs remain stable; baseline test caches and CI
snapshots are unaffected.

### Estimated cost

- 5 new sub-config / extras modules: ~60 lines each = ~300 lines src
- 5 new adapter classes: ~80 lines each = ~400 lines src
- Phase 1 test rewrites: ~250 lines diff
- 5 new test files (one per new module): ~80 lines each = ~400 lines tests
- Architecture A4 / A16 rewrite: ~200 lines diff
- Requirements F7 / TFT-hyperparameters edits: ~50 lines diff
- Implementation plan Phase 1 update: ~50 lines diff
- New `docs/benchmarks.md` skeleton: ~50 lines

Total: ~1,700 lines of net additions across src, tests, and docs.
Coverage gate (85% line / 80% branch) holds: every new src module has
its corresponding test file (1:1 mapping), and the per-module
coverage-by-line of the new tests targets 90%+ following the Phase 1
precedent.

## Alternatives considered

### Alternative A: keep the flat config, rely only on `extra` dicts

Cost: low (one `extra: ExtraDict` field on `BaseModelConfig`). Risk:
hyperparameters that warrant a typed field stay in the dict forever,
because there is no documented promotion path. Discoverability
craters; benchmark wins go unshipped.

Rejected: solves only Tier 4. The promotion ramp (ALPHA → BETA →
STABLE) is the actual contribution.

### Alternative B: keep the flat config, add `<Model>AdvancedConfig` only

Cost: medium (one new sub-config per model, plus the `advanced` field
on each `<Model>Config`). Risk: family-of-options scattering persists.
Adding AdamW `betas` and SGD `momentum` to the flat `BaseTrainingConfig`
still produces a 60-field surface by v3.

Rejected: half-fix. Family-of-options nesting is the bigger source of
scaling pain.

### Alternative C: full hierarchical config with no escape hatch

Cost: high (all of the proposed tier-1/2/3 work, plus discipline to
type every new field rather than ever using a dict). Risk: ALPHA-tier
experimentation requires shipping a release; benchmark testing slows
down.

Rejected: ergonomic regression for the maintainer. The `extra` tuple
adds ~3 lines per sub-config and unlocks rapid experimentation.

### Alternative D: defer everything to v2

Cost: zero now; high later (full refactor of estimator constructors,
search spaces, save / load schema once v1 ships). Risk: v1 callers
build code against the flat config; v2 either breaks them or carries
the flat config forever as a compatibility layer.

Rejected: kicks the can. Phase 1 is the lowest-cost moment to do this
because no callers exist yet.

## Open questions for the implementation phase

(Resolved questions are now in "Addressed".)

1. **Curated ALPHA-key enumeration**: `src/seq_sklearn/tuning/_alpha_keys.py`
   is the maintainer-controlled list of `extra` keys eligible for
   `search_extras=True` sampling. v1 ships empty; the population
   process is left to implementation time (likely: a maintainer adds
   keys here when they land an ALPHA passthrough they want
   Optuna-tunable). Pin the population workflow at implementation
   time.

## Addressed

Round 1 (design-review swarm):

- **`LossConfig.strategy` no default + nested-adapter recursion (arch
  r1-C1).** Added "User-facing surface" subsection documenting the
  adapter-kwarg pattern (one adapter kwarg per nested sub-config; flat
  per-field kwargs do NOT appear at the top level). Added
  "`_config_to_estimator_kwargs` recursive update" subsection with the
  `_TFT_ADAPTER_MAP` pattern; updates `docs/architecture.md` A16
  accordingly.
- **No-op-after-deprecation contradiction (arch r1-C2).** Rewrote the
  promotion-path text: the dict path becomes a permanent deprecation
  alias that routes values to the typed field with a
  `DeprecationWarning`. No silent no-op. Removal of the alias itself
  requires a MAJOR bump per the project semver policy.
- **F7 signature drift (arch r1-C3 / qa r1-I1).** Migration plan now
  has an explicit row updating `docs/requirements.md` F7 to add the
  `search_advanced` / `search_extras` keyword arguments. F7 also gets
  its flat training-field enumeration replaced with the nested-config
  reference.
- **Default search-space contents undefined (arch r1-C4).** Added the
  "Default search space per model" table enumerating exactly which
  fields fall into the default sample set, plus the regression-test
  requirement.
- **Phase 1 test rewrites unaddressed (qa r1-C1).** Migration plan
  table now lists every existing test file with its required edit and
  every new test file with its scope. Parametrize-ID stability note
  added.
- **`extra` dict immutable after construction (qa r1-C2).** `ExtraDict`
  refactored to `tuple[tuple[str, ExtraValue], ...]` (sorted, hashable,
  immutable) with the `_normalize_extras` BeforeValidator mirroring
  the `categorical_embed_dims` pattern.
- **`extra` value JSON round-trip fidelity (qa r1-C3 / arch r1-I1 /
  r1-I2).** Restricted `ExtraValue` to `str | int | float | bool |
  None`. Tuples / nested dicts / custom objects raise `TypeError` at
  construction. Round-trip through `model_dump()` and `model_validate()`
  is guaranteed type-identical.
- **Adapter clone-safety tests missing (qa r1-C4).** Added six
  named clone-safety tests under `tests/unit/config/test_adapters.py`
  in the migration plan's test-file table, one per adapter:
  `test_tabular_config_params_clone_is_independent`,
  `test_optimizer_params_clone_is_independent`,
  `test_scheduler_params_clone_is_independent`,
  `test_loss_params_clone_is_independent`,
  `test_sampler_params_clone_is_independent`,
  `test_tft_advanced_params_clone_is_independent`. The
  `TabularConfigParams` clone test was added in the Round 2 sweep
  per the architecture-doc fold-in finding (cross-doc count drift).
- **Deprecation-warning test missing (qa r1-C5).** Added the
  `test_extra_path_after_promotion_emits_deprecation_warning` test
  to the migration plan (lands once the first ALPHA→BETA promotion
  occurs; v1 ships the deprecation-alias machinery itself with a
  unit test that mocks the promotion to exercise the path).
- **Tier mapping for v2 / v3 models (arch r1-I4).** Added the "Tier
  mapping for v2 / v3 models" subsection with a worked PatchTST
  example and one-line treatment for the other v2 / v3 entries.
- **Benchmark gate metrics undefined (arch r1-I6).** Added the
  "Benchmark gate metrics" subsection with per-`target_kind` headline
  metrics and improvement thresholds tied to N1 acceptance numbers.
- **ALPHA passthrough name collision (arch r1-I8).** Spec'd the
  `_<NAME>_RESERVED` frozenset guard per family factory; collisions
  raise `ConfigError`, not the upstream torch `TypeError`.
- **Phase 1 test parametrize-ID stability (qa r1-I5 / r1-I5).** The
  validity-matrix parametrization tuple is unchanged (still 4
  strings); only the test body constructs nested configs. IDs stay
  stable.
- **Coverage gate continuity (qa r1-I4).** Estimated cost section now
  enumerates new src AND new test files (1:1 mapping), targets 90%+
  per-module coverage per Phase 1 precedent.
- **`suggest_params` flag tests (qa r1-I1).** Migration-plan test-file
  table includes `test_suggest_params_default_flags_exclude_advanced_fields`
  and `test_suggest_params_search_advanced_true_accepts_flag`
  under Phase 8 (already on the implementation plan; this proposal
  references them).
- **`TFTAdvancedConfig` empty-default test (qa r1-I2).** Migration-plan
  test additions include `test_tft_config_advanced_field_is_not_none_by_default`
  and `test_tft_advanced_config_default_construction_succeeds`.
- **`extra` serialization type inventory test (qa r1-I3).**
  `tests/unit/config/test_extras.py` covers the full type inventory
  per the JSON-roundtrip requirement.
- **Open question 1 (`extra` dict typing) (arch r1-I1).** Resolved:
  `ExtraValue = str | int | float | bool | None`. Removed from open
  questions.
- **Open question 2 (`<Model>AdvancedConfig` non-optional vs. None
  default).** Resolved: non-optional with empty default; documented
  in Tier 3.
- **Open question 3 (promotion-test scaffolding).** Deferred (see
  Deferred section).
- **Open question 4 (CHANGELOG schema).** Deferred (see Deferred
  section).
- **Field count miscount 33 → 34 (arch r1-N2).** Corrected.
- **Pre-populated Deferred section (arch r1-N3).** Done.
- **"60 fields churn signal" rephrased (arch r1-N1).** Now cites the
  griffe-pydantic rendering cost in Requirement 2.
- **`LossConfig.strategy` no-default comment (arch r1-N4).** Inline
  comment added at the field declaration.

## Deferred

Round 1 (design-review swarm):

- **Promotion-test scaffolding that parses CHANGELOG for tier
  transitions** (arch open q3). Useful but costs CHANGELOG-format
  parsing and a brittle test surface. Deferred to a v1.x maintenance
  PR once the first ALPHA→BETA promotion actually occurs.
- **CHANGELOG-schema "Hyperparameter changes" subsection** (arch
  open q4). The current Keep-a-Changelog `Added` / `Changed` /
  `Removed` sections already accommodate tier-transition entries via
  prose. Adding a dedicated subsection is documentary polish; if
  promotion volume grows, revisit at v1.x.
- **SGD-only field guard for OptimizerConfig (qa r1-N3).** Setting
  `momentum=0.99` with `name="adamw"` is silently ignored by
  `build_optimizer`. A warning would help but the field-level
  Literal-gating is non-trivial under pydantic v2 (the discriminator
  pattern complicates `set_params`). Deferred until benchmark
  testing reveals the silent misconfiguration causes real bugs.
- **Default scheduler-name preservation test (qa r1-N2).** The
  default `SchedulerConfig.name="cosine_with_warmup"` matches the
  Phase 1 `BaseTrainingConfig.scheduler` default. Pinning this as a
  test is documentary; the equality is enforced by the migration
  plan's source edits.

Gemini final pass (cross-family review):

- **Escape-hatch deadlock via over-broad reserved set (gemini r1-C1).**
  The original `_ADAMW_RESERVED` listed `maximize`, `foreach`,
  `capturable`, `differentiable`, `fused`, which `OptimizerConfig`
  exposes no typed fields for. Users were blocked from passing them via
  `extra` AND had no typed alternative. Shrunk the reserved set to keys
  that actually collide with typed-field kwargs at the
  `build_optimizer` call site: `{"params", "lr", "weight_decay",
  "betas", "eps"}` for AdamW / Adam, `{"params", "lr", "weight_decay",
  "momentum", "nesterov"}` for SGD. Untyped torch kwargs pass through
  `extra` unrestricted, which is the legitimate ALPHA-tier use case.
  Pinned by `test_adamw_reserved_keys_collision_raises` (uses `"lr"`
  as the collision key, which IS in the new shrunken reserved set) and
  `test_sgd_reserved_keys_collision_raises`.
- **Positional argument break on BETA promotion (gemini r1-C2).** Every
  adapter `__init__` now uses the `*` keyword-only marker. Without it,
  promoting a new BETA field would shift positional arguments and break
  callers using `OptimizerParams("adamw", 0.01)` positionally, turning
  a MINOR-additive change into a MAJOR break. The existing
  `TabularConfigParams` at
  `src/seq_sklearn/config/_params_adapter.py:65` is amended to
  keyword-only as part of the refactor. Added the meta-test
  `test_all_adapters_have_keyword_only_init` that introspects every
  adapter's signature.
- **Loss-default ergonomics under sklearn instantiation (gemini r1-C3,
  graded down to IMPROVEMENT after verification).** The proposal showed
  `loss=LossParams(strategy=...)` as required in the user example but
  `loss: LossParams | None = None` in the adapter signature, with no
  specified behavior for the `None` case. Added the "Task-type-aware
  loss default" subsection: `_build_config` injects a
  `_DEFAULT_LOSS_FOR_TASK[task_type]` default when the caller omits
  `loss=`. Preserves three contracts at once (type hint truthful,
  `LossConfig.strategy` keeps no default, ergonomic fit-without-loss
  works). Named test
  `test_loss_default_injection_per_task_type` lives in Phase 6a (lands
  with `BaseSequenceEstimator._build_config`).
- **Brittle deprecation alias enforcement (gemini r1-I1).** Added the
  `extract_deprecated_extras` helper plus the
  `_PROMOTED_KEYS_BY_FAMILY` registry. Every family factory calls the
  helper instead of `dict(cfg.extra)`; promotion of an ALPHA key adds
  one entry to the registry and the alias behavior fires automatically.
  Meta-test `test_extract_deprecated_extras_meta_promoted_keys_exist`
  catches "registered promotion without typed field" at CI rather than
  at first user report. A second test,
  `test_extract_deprecated_extras_both_typed_and_extra_raises_config_error`,
  pins the ambiguous-configuration contract (setting both the typed
  field and the extra key raises `ConfigError`). Stage-2 BETA promotion
  example rewritten to show the one-line registry edit (`_PROMOTED_KEYS_BY_FAMILY["optimizer"]["amsgrad"] = "amsgrad"`)
  instead of the inline `extra.pop` + `warnings.warn` pattern that
  predated the helper.
- **Empty advanced-config `get_params` noise (gemini r1-I2).** Trade-off
  acknowledged: `get_params(deep=True)` produces `advanced__extra = ()`
  per estimator. The slot-existence benefit (locks the BETA-promotion
  destination from day one) outweighs the one-key noise in
  GridSearchCV output. No suppression hook in sklearn currently; if
  Optuna or GridSearchCV iteration grows unwieldy in practice, revisit
  via the `_get_param_names` override.
- **NITPICK hallucinated (gemini r1-N1).** Gemini flagged a typo at
  line 364 claiming "the test pins the `mode='json'` path" should
  reference `config.model_dump(mode='json')`. The doc at line 369
  already does this; the line 374-375 reference is to the test name.
  No change needed.

Round 2 (design-review swarm):

- **Em dash slipped in at line 544 (style r2-C1).** Replaced with a
  colon. The doc is now em-dash-free.
- **Clone-safety test names not anchored in the migration-plan body
  (qa r2-C1 / arch r2-I3).** Added the "Named tests added in the
  Phase 1 refactor" subsection enumerating all 21 named test
  functions by file + name + landing phase. The migration-plan body
  is now self-contained: a reader can verify Round 1's CRITICAL
  fixes without reading the Addressed ledger.
- **Fabricated "N1 acceptance threshold" citation in the
  benchmark-gate-metrics table (arch r2-I1).** Dropped the "Source"
  column; rewrote the surrounding prose to acknowledge these are
  v1.0 starting thresholds. The connection to N1 (same metric
  families) is now stated in prose without claiming the deltas come
  from N1.
- **Stale A16 line range (arch r2-I2).** Corrected to 1601-1737.
- **`suggest_params` / `TFTAdvancedConfig` / deprecation tests
  anchored only in the ledger (qa r2-I1, r2-I2, r2-I3).** The new
  "Named tests added in the Phase 1 refactor" subsection anchors
  every previously-ledger-only test in the migration-plan body.
- **`model_dump` serialization mode for `extra` unspecified (qa
  r2-I4).** Pinned to `mode="json"` with a corresponding named test
  (`test_extra_dict_survives_json_roundtrip`).
- **Sorted-key contract test not named (qa r2-N1).** Added
  `test_extra_dict_stored_as_sorted_tuple` to the named-tests table.
- **SGD reserved-key collision test not named (qa r2-N2).** Added
  `test_sgd_reserved_keys_collision_raises` to the named-tests
  table.
- **Redundant `bool` in isinstance check (arch r2-N1).** Dropped
  `bool` from the isinstance tuple with a comment noting `bool` is
  an `int` subclass.
- **Skeleton vs. empty phrasing reconciliation (arch r2-N2).** The
  benchmark-gate-metrics section now reads "v1.0 ships the file as a
  skeleton with section headers and no benchmark numbers yet."
