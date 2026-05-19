# Phase 1 refactor plan: flat-config to four-tier architecture

## Requirements

This plan maps the current Phase 1 implementation (on branch
`phase-1-foundation`) to the post-fold-in target documented in
`docs/requirements.md`, `docs/architecture.md`, and
`docs/implementation_plan.md`. It is graded against:

1. **End-state correctness**: after the refactor, the Phase 1 code
   matches every load-bearing claim in requirements F1-F11 / N1-N7,
   architecture A1-A20 (especially A1 layout, A4 config schemas, A8
   build-loss bridge, A16 Optuna integration), and implementation plan
   Phase 1 module + test lists.
2. **No silent test loss**: every passing test in the current Phase 1
   either continues to pass with identical intent, OR is intentionally
   rewritten with a one-line rationale in this plan's "Test rewrite
   manifest." A test that disappears without justification is a
   correctness gap.
3. **All named Phase 1 tests land**: the 24 Phase-1-mandatory tests
   in the `docs/implementation_plan.md` Phase 1 test roster become
   real test functions. Each one has a documented home (file + scope).
4. **Gates green at the end**: `ruff check`, `ruff format --check`,
   `pyright` (strict on `src/`, relaxed on `tests/` per
   `pyrightconfig.json`), `pytest --cov` line >= 85% / branch >= 80%
   on `src/seq_sklearn/`. Mid-refactor checkpoints may temporarily
   fail per-commit gates so long as the final PR end-state is green.
5. **No new architectural deltas**: the refactor implements what is
   already documented. Anything that would require a doc change first
   is out of scope; surface it as a discovery, do not silently land it.
6. **Bisectable**: each commit within the refactor PR ships a coherent
   logical step so future bisection on regressions narrows to one
   commit rather than the whole PR.
7. **Reversible**: the refactor produces a single PR that can be
   reverted via `git revert <merge-commit>` without leaving the tree
   in an inconsistent state. No partial migration to clean up later.

CRITICAL findings against this plan must trace to one of the above.

## Current state (snapshot)

Branch: `phase-1-foundation`. Last green commit on this branch passes
1195 tests + 288 xfail (the v1.1 task-type cells, by design) with 97%
line / 96% branch coverage on the modules below.

### Source files (10 modules)

```
src/seq_sklearn/
  errors.py                    SeqSklearnError hierarchy
  logging.py                   Event enum + emit() helper
  _validate.py                 check_y, check_columns
  hardware.py                  HardwareTier IntEnum + detect()
  config/
    __init__.py
    _domains.py                TASK_TYPES, LOSS_STRATEGIES, etc.
    _validity.py               check_combo()
    base.py                    BaseTrainingConfig, BaseModelConfig (flat)
    tabular.py                 TabularToSequenceConfig
    tft.py                     TFTConfig (no advanced)
    _params_adapter.py         TabularConfigParams only
```

### Test files (10 files)

```
tests/
  conftest.py                  propagate_seq_sklearn_logger fixture
  unit/
    __init__.py
    test_errors.py
    test_logging.py
    test_validate.py
    test_hardware_detect.py
    config/
      __init__.py
      test_base.py             flat-kwargs construction
      test_tabular.py
      test_tft.py              flat-field validators
      test_validity_matrix.py  Cartesian-product sweep on check_combo + BaseModelConfig
      test_params_adapter.py   TabularConfigParams clone + to_pydantic
```

### Key gaps versus target

The current Phase 1 implements the **flat-config** shape from the
pre-fold-in design. It is missing:

- Family sub-configs: no `OptimizerConfig`, `SchedulerConfig`,
  `LossConfig`, `SamplerConfig`.
- Escape-hatch infrastructure: no `ExtraDict`, `_normalize_extras`,
  `_PROMOTED_KEYS_BY_FAMILY`, `extract_deprecated_extras`.
- Advanced sub-config: no `TFTAdvancedConfig`; `TFTConfig` has no
  `advanced` field.
- Five of the six required adapters: only `TabularConfigParams`
  exists. Missing `OptimizerParams`, `SchedulerParams`, `LossParams`,
  `SamplerParams`, `TFTAdvancedParams`.
- Keyword-only marker on the existing `TabularConfigParams.__init__`.
- Config-layer reserved-keys collision check
  (`_check_extra_not_reserved` model_validator on each family
  sub-config).
- `_DEFAULT_LOSS_FOR_TASK` map (lives in Phase 6a's
  `models/_base.py`, not Phase 1; noted here only because Phase 1's
  `LossConfig.strategy` ships with no default and the injection
  logic is Phase 6a's concern).

The current `base.py` `BaseTrainingConfig` and `BaseModelConfig`
carry the flat-field shape that needs to nest under the family
sub-configs.

## Target state

After the refactor, the Phase 1 source tree matches the architecture
A1 layout exactly:

```
src/seq_sklearn/
  errors.py                    (unchanged)
  logging.py                   (unchanged)
  _validate.py                 (unchanged)
  hardware.py                  (unchanged)
  config/
    __init__.py                (unchanged)
    _domains.py                (unchanged)
    _validity.py               (unchanged signature; call sites in base.py update)
    _extras.py                 NEW: ExtraValue, ExtraDict, _normalize_extras,
                                    _PROMOTED_KEYS_BY_FAMILY,
                                    extract_deprecated_extras
    optimizer.py               NEW: OptimizerConfig + _RESERVED_BY_OPTIMIZER
    scheduler.py               NEW: SchedulerConfig + _RESERVED_BY_SCHEDULER
    loss.py                    NEW: LossConfig + _RESERVED_BY_LOSS
    sampler.py                 NEW: SamplerConfig + _RESERVED_BY_SAMPLER
    base.py                    REWRITE: nested family sub-configs;
                                        validator reads self.loss.strategy etc.
    tabular.py                 (unchanged; CategoricalEmbedDims already
                                correct per Phase 1)
    tft.py                     EDIT: add TFTAdvancedConfig + advanced field
    _adapters.py               RENAME from _params_adapter.py;
                                ADD: 5 more adapters + keyword-only marker
                                on existing TabularConfigParams
```

Test tree:

```
tests/
  conftest.py                  (unchanged)
  unit/
    test_errors.py             (unchanged)
    test_logging.py            (unchanged)
    test_validate.py           (unchanged)
    test_hardware_detect.py    (unchanged)
    config/
      test_base.py             REWRITE: nested-config construction;
                                add v1.1-task-type-rejection test
      test_tabular.py          (unchanged)
      test_tft.py              EDIT: add advanced-field default tests
      test_validity_matrix.py  REWRITE: parametrize body uses nested configs;
                                tuple parametrization unchanged
      test_adapters.py         RENAME from test_params_adapter.py;
                                ADD 5 more clone-safety tests +
                                test_all_adapters_have_keyword_only_init
      test_extras.py           NEW: 10 named tests
      test_optimizer.py        NEW
      test_scheduler.py        NEW
      test_loss.py             NEW
      test_sampler.py          NEW
```

Net change: +5 new source modules, +5 new test modules, 1 source-file
rename (`_params_adapter.py` to `_adapters.py`) plus the test-file mirror
rename (`test_params_adapter.py` to `test_adapters.py`).

## Module diff in detail

### NEW: `src/seq_sklearn/config/_extras.py`

Per architecture A4:

- `ExtraValue = str | int | float | bool | None` type alias.
- `ExtraDict = Annotated[tuple[tuple[str, ExtraValue], ...], BeforeValidator(_normalize_extras)]`.
- `_normalize_extras(v: object) -> tuple[tuple[str, ExtraValue], ...]`
  validator: accepts `None` / `Mapping` / iterable-of-pairs; rejects
  non-primitive values with a documented `TypeError`; returns sorted
  tuple of tuples.
- `_PROMOTED_KEYS_BY_FAMILY: dict[str, dict[str, str]]` registry
  (empty in v1; populated as ALPHA → BETA promotions occur).
- `extract_deprecated_extras(cfg, family) -> tuple[cfg, dict[str, ExtraValue]]`
  helper per architecture A4. Returns the cfg (a `model_copy` with any
  promoted value routed onto the typed field) plus the cleaned extra
  dict; emits a `DeprecationWarning` per promoted key; raises
  `ConfigError` on the both-paths ambiguous-configuration case.

Dependencies: stdlib, pydantic, and `seq_sklearn.errors.ConfigError`
(raised on the both-paths ambiguous-configuration case).

### NEW: `src/seq_sklearn/config/optimizer.py`

```python
class OptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Literal["adamw", "adam", "sgd"] = "adamw"
    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = Field(default=1e-8, gt=0.0)
    momentum: float = Field(default=0.9, ge=0.0, lt=1.0)
    nesterov: bool = False
    extra: ExtraDict = ()

    @model_validator(mode="after")
    def _check_extra_not_reserved(self) -> Self:
        reserved = _RESERVED_BY_OPTIMIZER[self.name]
        extra_keys = {k for k, _ in self.extra}
        clashes = reserved & extra_keys
        if clashes:
            raise ValueError(
                f"extra keys collide with typed {self.name} kwargs: "
                f"{sorted(clashes)}. Set the typed OptimizerConfig "
                "field directly."
            )
        return self
```

Plus the module-level `_RESERVED_BY_OPTIMIZER` dict.

Dependencies: `_extras.py`.

### NEW: `src/seq_sklearn/config/scheduler.py`

Same `ConfigDict(extra="forbid", frozen=True)` shape as
`optimizer.py`. The schema below matches the authoritative source
(architecture A4 and `src/seq_sklearn/config/scheduler.py`; field list
and defaults also trace to requirements F5, requirements.md:755-766).
Do not re-derive the defaults; mirror the code exactly:

```python
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

    @model_validator(mode="after")
    def _check_extra_not_reserved(self) -> Self:
        reserved = _RESERVED_BY_SCHEDULER[self.name]
        # ... same shape as OptimizerConfig
```

Plus the module-level `_RESERVED_BY_SCHEDULER` dict keyed by `name`.
The `constant`-rejects-`warmup_steps` interaction (requirements F5 /
arch I10) is NOT enforced here in Phase 1; see the Deferred section.

Dependencies: `_extras.py`.

### NEW: `src/seq_sklearn/config/loss.py`

```python
class LossConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # No default: legal value depends on task_type per F5; the
    # estimator's _build_config (Phase 6a) injects the default.
    strategy: Literal["cross_entropy", "focal", "mse", "mae", "huber", "pinball"]
    focal_gamma: float = Field(default=2.0, gt=0.0)
    focal_alpha: float | None = None
    huber_delta: float = Field(default=1.0, gt=0.0)
    label_smoothing: float = Field(default=0.0, ge=0.0, lt=1.0)
    extra: ExtraDict = ()

    @model_validator(mode="after")
    def _check_extra_not_reserved(self) -> Self:
        reserved = _RESERVED_BY_LOSS[self.strategy]
        # ... same shape as OptimizerConfig
```

### NEW: `src/seq_sklearn/config/sampler.py`

Same shape. `SamplerConfig(strategy, oversample_ratio, replacement, extra)`.

### REWRITE: `src/seq_sklearn/config/base.py`

`BaseTrainingConfig` loses flat optimizer/scheduler fields, gains
nested family sub-configs:

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
```

`BaseModelConfig` loses flat loss / imbalance fields, gains nested
`loss` and `sampler`:

```python
class BaseModelConfig(BaseTrainingConfig):
    task_type: Literal[...]
    loss: LossConfig                         # no default
    sampler: SamplerConfig = Field(default_factory=SamplerConfig)
    calibration_strategy: Literal[...] = "none"
    threshold_tuning: bool = False
    threshold_metric: Literal[...] = "f1"
    quantiles: tuple[float, ...] | None = None

    @model_validator(mode="after")
    def _check_validity_matrix(self) -> Self:
        check_combo(
            self.task_type,
            self.loss.strategy,        # CHANGED: nested access
            self.sampler.strategy,     # CHANGED: nested access
            self.calibration_strategy,
        )
        return self

    @model_validator(mode="after")
    def _check_quantiles_monotone(self) -> Self:
        # unchanged
        ...

    @model_validator(mode="after")
    def _check_val_cal_sum(self) -> Self:
        # unchanged
        ...
```

Fields removed (now live on family sub-configs):
- `learning_rate`, `weight_decay` → `OptimizerConfig`
- `optimizer: Literal[...]` (flat name) → `OptimizerConfig.name`
- `scheduler: Literal[...]` (flat name) → `SchedulerConfig.name`
- `warmup_steps` → `SchedulerConfig.warmup_steps`
- `loss_strategy` → `LossConfig.strategy`
- `imbalance_strategy` → `SamplerConfig.strategy`
- `focal_gamma`, `huber_delta` → `LossConfig`
- `oversample_ratio` → `SamplerConfig.oversample_ratio`

### EDIT: `src/seq_sklearn/config/tft.py`

Add `TFTAdvancedConfig` class. Add `advanced:
TFTAdvancedConfig = Field(default_factory=TFTAdvancedConfig)` field
to `TFTConfig`. No other field changes (TFT-specific architecture
fields stay flat per architecture A4).

```python
class TFTAdvancedConfig(BaseModel):
    """BETA per requirements stability tiers. Empty in v1."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    extra: ExtraDict = ()


class TFTConfig(BaseModelConfig):
    hidden_size: int = Field(default=128, ge=1)
    attention_heads: int = Field(default=4, ge=1)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    variable_selection_dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    prediction_readout: Literal["last_valid", "mean_pool"] = "last_valid"
    tabular_config: TabularToSequenceConfig
    advanced: TFTAdvancedConfig = Field(default_factory=TFTAdvancedConfig)

    @model_validator(mode="after")
    def _check_heads_divide_hidden(self) -> Self:
        # unchanged
        ...
```

### RENAME + EXTEND: `src/seq_sklearn/config/_params_adapter.py` → `_adapters.py`

Rename the file. Add the `*` keyword-only marker to the existing
`TabularConfigParams.__init__`. Add five more adapter classes
following the same shape (mutable mirror of the pydantic schema,
`to_pydantic()` builder):

- `OptimizerParams` ← `OptimizerConfig`
- `SchedulerParams` ← `SchedulerConfig`
- `LossParams` ← `LossConfig`
- `SamplerParams` ← `SamplerConfig`
- `TFTAdvancedParams` ← `TFTAdvancedConfig`

Every adapter `__init__` carries `*` keyword-only so positional
callers cannot collide with future BETA promotion field additions.

## Test rewrite manifest

### Existing tests: continuity

| File | Action | Continuity |
|---|---|---|
| `test_errors.py` | unchanged | all assertions stable |
| `test_logging.py` | unchanged | all assertions stable |
| `test_validate.py` | unchanged | all assertions stable |
| `test_hardware_detect.py` | unchanged | all assertions stable |
| `test_tabular.py` | unchanged | the `CategoricalEmbedDims` work in the Phase 1 PR already nailed this |

### Existing tests: rewrites (with rationale)

| File | Rewrite | Rationale |
|---|---|---|
| `test_base.py` | Replace flat `_legal_kwargs` (returning `task_type`, `loss_strategy`, etc.) with nested `_legal_kwargs` (returning `task_type`, `loss=LossConfig(strategy="cross_entropy")`). Add `test_v1_task_type_rejects_multilabel_and_regression_multioutput`. | `BaseModelConfig` no longer accepts flat `loss_strategy` per the new schema. Tests that constructed via the flat shape must use nested construction. Intent preserved: every previous test still pins the same invariant (frozen, extra=forbid, validity matrix, quantile validator, val+cal sum). |
| `test_tft.py` | Add `test_tft_config_advanced_field_is_not_none_by_default` and `test_tft_advanced_config_default_construction_succeeds`. Existing tests on attention_heads divides hidden_size, dropout bounds, prediction_readout literal stay. | New `advanced` field needs default-value test coverage per architecture A4. |
| `test_validity_matrix.py` | The parametrization tuple stays 4 strings (`task_type, loss_strategy, imbalance_strategy, calibration_strategy`). The test body construction changes from `BaseModelConfig(task_type=t, loss_strategy=l, ...)` to `BaseModelConfig(task_type=t, loss=LossConfig(strategy=l), sampler=SamplerConfig(strategy=i), ...)`. | `check_combo` signature is unchanged so the parametrize-IDs match the previous run (pytest cache stays stable; the four-string parametrize tuple is unchanged). |
| `test_params_adapter.py` → `test_adapters.py` | Rename file. Existing `TabularConfigParams` tests rename to `test_tabular_config_params_*`. Add five more per-adapter clone tests (one per new adapter). Add `test_all_adapters_have_keyword_only_init` introspecting every adapter's `__init__` signature. | The single-adapter contract generalizes to six adapters; the introspection test pins the `*` keyword-only marker per Gemini-pass finding. |

### New tests by file

`tests/unit/config/test_extras.py` (10 named tests):

1. `test_extra_dict_rejects_non_primitive_value`: `numpy.ndarray` value raises `TypeError`.
2. `test_extra_dict_round_trips_each_primitive_type`: `str | int | float | bool | None` survive `model_dump(mode="json")` then JSON then `model_validate()` with identical types. Outer container is `OptimizerConfig` to exercise the full pydantic field path.
3. `test_extra_dict_stored_as_sorted_tuple`: two constructions with reversed key order produce identical stored tuples and identical hashes.
4. `test_extra_dict_survives_json_roundtrip`: an `OptimizerConfig(extra=(("flag", True), ("count", 3)))` (outer container named so the pydantic field path, not a bare `_normalize_extras` call, is exercised) survives full `model_dump(mode="json")` + `json.dumps` + `json.loads` + `OptimizerConfig.model_validate` cycle with `cfg == reconstructed`.
5. `test_extract_deprecated_extras_meta_promoted_keys_exist`: every entry in `_PROMOTED_KEYS_BY_FAMILY` names a real typed field on the corresponding pydantic family config (stub config uses a `BaseModel` subclass to match the helper's `cfg.model_fields` access path).
6. `test_extract_deprecated_extras_both_typed_and_extra_raises_config_error`: caller sets both the typed field and the extra key; `ConfigError` raises.
7. `test_extract_deprecated_extras_happy_path_passes_through` (Gemini pass): unpromoted keys pass through unchanged, no `DeprecationWarning`.
8. `test_extract_deprecated_extras_mock_promotion_emits_warning`: monkeypatch a fake key into the registry and a typed field on a pydantic `BaseModel` subclass; assert `DeprecationWarning` matches `r"deprecated.*"` and the typed field carries the value.
9. `test_extra_dict_rejects_non_string_key`: `OptimizerConfig(extra=((1, "v"),))` raises `TypeError` per `_normalize_extras` documented behavior.
10. `test_normalize_extras_accepts_none_produces_empty_tuple`: `OptimizerConfig(extra=None)` produces `cfg.extra == ()`. Pins the sentinel-default path.

`tests/unit/config/test_optimizer.py` (4 named tests):

1. `test_default_construction_uses_documented_defaults`: `OptimizerConfig()` produces `name="adamw"`, `learning_rate=1e-3`, etc.
2. `test_adamw_reserved_keys_collision_raises`: `OptimizerConfig(name="adamw", extra=(("lr", 0.1),))` raises `ValidationError` from `_check_extra_not_reserved` with `match=r"lr"` (per requirements F7 / `_check_extra_not_reserved` ConfigError-message contract).
3. `test_sgd_reserved_keys_collision_raises`: same for SGD with the `momentum` collision; `match=r"momentum"`.
4. `test_extra_field_rejected`: `OptimizerConfig(undocumented_field=1)` raises `ValidationError` per `ConfigDict(extra="forbid")`.

`tests/unit/config/test_scheduler.py` (3 named tests):

1. `test_default_construction_uses_documented_defaults`: `SchedulerConfig()` produces `name="cosine_with_warmup"`, `warmup_steps=100`, etc.
2. `test_reserved_keys_collision_raises`: a scheduler name with reserved typed fields (e.g. `name="cosine_with_warmup"`, `extra=(("warmup_steps", 50),)`) raises `ValidationError`; `match=r"warmup_steps"`.
3. `test_extra_field_rejected`: `SchedulerConfig(undocumented_field=1)` raises `ValidationError`.

`tests/unit/config/test_loss.py` (3 named tests):

1. `test_construction_requires_strategy`: `LossConfig()` without `strategy=` raises `ValidationError`.
2. `test_reserved_keys_collision_raises`: `LossConfig(strategy="focal", extra=(("focal_gamma", 2.5),))` raises `ValidationError`; `match=r"focal_gamma"`.
3. `test_extra_field_rejected`: `LossConfig(strategy="cross_entropy", undocumented_field=1)` raises `ValidationError`.

`tests/unit/config/test_sampler.py` (3 named tests):

1. `test_default_strategy_is_none`: `SamplerConfig()` produces `strategy="none"`, `replacement=True`, `oversample_ratio=1.0`.
2. `test_reserved_keys_collision_raises`: `SamplerConfig(strategy="oversample_minority", extra=(("oversample_ratio", 2.0),))` raises `ValidationError`; `match=r"oversample_ratio"`.
3. `test_extra_field_rejected`: `SamplerConfig(undocumented_field=1)` raises `ValidationError`.

`tests/unit/config/test_adapters.py` (renamed; 6 clone-safety + 1 meta-test + 5 to-pydantic tests for new adapters; the two carried-over to-pydantic tests on `TabularConfigParams` stay):

1. `test_tabular_config_params_clone_is_independent` (carried over from `test_params_adapter.py`).
2. `test_optimizer_params_clone_is_independent` (new).
3. `test_scheduler_params_clone_is_independent` (new).
4. `test_loss_params_clone_is_independent` (new).
5. `test_sampler_params_clone_is_independent` (new).
6. `test_tft_advanced_params_clone_is_independent` (new).
7. `test_all_adapters_have_keyword_only_init`: introspects every adapter's `__init__` via `inspect.signature(cls)`; asserts EVERY non-self parameter has `kind == Parameter.KEYWORD_ONLY` (not merely `POSITIONAL_OR_KEYWORD` with a default). Pins the `*` keyword-only marker so a BETA promotion shifting positional argument order cannot silently land.
8. `test_optimizer_params_to_pydantic_produces_correct_config` (new): `OptimizerParams(name="sgd", learning_rate=0.01).to_pydantic()` returns an `OptimizerConfig` with `name == "sgd"` and `learning_rate == 0.01`.
9. `test_scheduler_params_to_pydantic_produces_correct_config` (new): analogous.
10. `test_loss_params_to_pydantic_produces_correct_config` (new): analogous.
11. `test_sampler_params_to_pydantic_produces_correct_config` (new): analogous.
12. `test_tft_advanced_params_to_pydantic_produces_correct_config` (new): analogous.

Plus the existing `test_to_pydantic_produces_frozen_config` and `test_to_pydantic_propagates_validation_errors` on `TabularConfigParams` (carried over from `test_params_adapter.py`, intent unchanged).

`tests/unit/config/test_validity_matrix.py` (rewrite + 1 new invariant):

In addition to the body rewrite (nested-config construction), add:

- `test_family_config_strategy_literals_match_domains`: parametrized over `(OptimizerConfig, "name", OPTIMIZER_NAMES)`, `(SchedulerConfig, "name", SCHEDULER_NAMES)`, `(LossConfig, "strategy", LOSS_STRATEGIES)`, `(SamplerConfig, "strategy", IMBALANCE_STRATEGIES)`. Asserts `typing.get_args(cls.model_fields[field].annotation) == tuple(domain)` for each. Pins the Literal-to-domain invariant so the F5 error path stays stable: if a family Literal drifts from `_domains.py`, an illegal-cell parametrize value would fail at sub-config construction (pydantic Literal error) instead of at `BaseModelConfig._check_validity_matrix` (F5 `ConfigError`), changing the documented failure-mode contract.

`tests/unit/config/test_base.py` (rewrite + 2 new):

In addition to the rewrite (nested-config construction), add:

- `test_v1_task_type_rejects_multilabel_and_regression_multioutput` (per architecture A4 fold-in; documented in `docs/implementation_plan.md`).
- `test_nested_base_model_config_model_dump_json_round_trips`: build a `BaseModelConfig` with non-default `optimizer`, `scheduler`, `loss`, `sampler` (each carrying a non-empty `extra=(("flag", True),)`); call `cfg.model_dump(mode="json")`; round-trip via `json.dumps` then `json.loads` then `BaseModelConfig.model_validate(...)`; assert the reconstructed config equals the original. Pins the on-disk serialization contract per requirements N1 save/load (mode="json").

**Total named tests in the Phase 1 manifest: 40.** Breakdown:
- 23 from the implementation plan's Phase 1 test roster.
- 1 from the architecture A4 fold-in (`test_v1_task_type_rejects_multilabel_and_regression_multioutput`).
- 16 added during this refactor plan's swarm review to close per-family extra-forbid coverage, per-family `_check_extra_not_reserved` coverage, the nested JSON round-trip path, the Literal-to-domain invariant, the adapter `to_pydantic()` paths on the five new adapters, the non-string-key rejection branch, and the `extra=None` sentinel path.

The 16 additions are within the F7 / N1 mandatory-test-gate coverage spirit. They do not change the architecture; they close coverage gaps that the swarm flagged as CRITICAL or IMPROVEMENT for the four-tier hyperparameter contract.

## Execution sequencing

Single PR. Within the PR, commits are ordered for cognitive load
and bisectability. Each commit is validated against a checkpoint
before the next one starts. Failures at a checkpoint are fixed
before moving on; if the failure is structural, this plan is
revisited before continuing.

### Commit 1: extras infrastructure

**Files**: `src/seq_sklearn/config/_extras.py` (new),
`tests/unit/config/test_extras.py` (new).

**Dependencies**: none (stdlib + pydantic only).

**Checkpoint**:
- Full project gates run on the commit: `ruff check`, `ruff format --check`, `pyright` strict on `src/`, `pytest`. (Per-commit full gates so bisection stays useful; commit 3's narrative and the validation matrix below agree on this policy.)
- `pytest tests/unit/config/test_extras.py`: all 10 named tests pass.

**Risk**: the `BeforeValidator` interplay with `frozen=True` is the
same idiom used by `TabularToSequenceConfig.categorical_embed_dims`
already; precedent is proven.

### Commit 2: family sub-configs (optimizer, scheduler, loss, sampler)

**Files**: `src/seq_sklearn/config/optimizer.py`,
`src/seq_sklearn/config/scheduler.py`,
`src/seq_sklearn/config/loss.py`,
`src/seq_sklearn/config/sampler.py`,
`tests/unit/config/test_optimizer.py`,
`tests/unit/config/test_scheduler.py`,
`tests/unit/config/test_loss.py`,
`tests/unit/config/test_sampler.py` (all new).

**Dependencies**: `_extras.py` (from commit 1).

**Checkpoint**:
- Full project gates: `ruff check`, `ruff format --check`, `pyright` strict on `src/`, `pytest`.
- `pytest tests/unit/config/test_{optimizer,scheduler,loss,sampler}.py`: all green.
- 13 new named tests pass (4 in `test_optimizer.py`, 3 each in `test_scheduler.py`, `test_loss.py`, `test_sampler.py`).

**Risk**: the `_check_extra_not_reserved` model_validator depends
on a `name` field for `OptimizerConfig` / `SchedulerConfig` (which
keyed dict to use) and on the static reserved set. For `LossConfig`
the validator keys off `strategy`; for `SamplerConfig` off
`strategy` too. Each family carries its own `_RESERVED_BY_<NAME>`
dict in the same file as the config.

### Commit 3: TFTAdvancedConfig + TFTConfig advanced field

**Files**: `src/seq_sklearn/config/tft.py` (edit),
`tests/unit/config/test_tft.py` (edit; add 2 tests).

**Dependencies**: `_extras.py` (for `TFTAdvancedConfig.extra:
ExtraDict`). Does NOT depend on the `base.py` rewrite yet. TFTConfig
still inherits from the current flat `BaseModelConfig` shape at this
point; the existing `test_tft.py` `_minimal_tft` helper still
constructs via the flat `loss_strategy="cross_entropy"` kwarg. The
helper rewrite to nested `loss=LossConfig(strategy="cross_entropy")`
is deferred to commit 4 (which is the commit that breaks the flat
shape).

**Checkpoint**:
- Full project gates green (ruff / pyright / pytest).
- `test_tft_config_advanced_field_is_not_none_by_default` passes.
- `test_tft_advanced_config_default_construction_succeeds` passes.
- All previous `test_tft.py` tests still pass (the advanced field
  is purely additive at this stage; `_minimal_tft` still uses
  flat kwargs and works because the base.py rewrite has not
  landed yet).

**Risk**: low. `TFTConfig` gets one new field with a default factory;
no existing field is removed yet. The `base.py` rewrite (commit 4)
will later force the `_minimal_tft` helper rewrite.

### Commit 4: nested base.py rewrite

**Files**: `src/seq_sklearn/config/base.py` (rewrite, major change),
`tests/unit/config/test_base.py` (rewrite),
`tests/unit/config/test_validity_matrix.py` (rewrite),
`tests/unit/config/test_tft.py` (edit `_minimal_tft` helper and the
one test that builds via flat kwargs).

**Dependencies**: `_extras.py`, `optimizer.py`, `scheduler.py`,
`loss.py`, `sampler.py` (all from commits 1-2).

**This is the riskiest commit.** It changes:
- `BaseTrainingConfig` fields (flat to nested); breaks any code that
  reads `cfg.learning_rate` (must become `cfg.optimizer.learning_rate`).
- `BaseModelConfig` fields (flat to nested); breaks `cfg.loss_strategy`
  (must become `cfg.loss.strategy`), `cfg.imbalance_strategy`, etc.
- `_check_validity_matrix` validator call site (reads from nested).
- Test bodies on `test_base.py`, `test_validity_matrix.py`, and
  `test_tft.py`'s `_minimal_tft` helper. The `_minimal_tft` helper
  currently constructs `TFTConfig(task_type=..., loss_strategy="cross_entropy", ...)`;
  after this commit it MUST construct
  `TFTConfig(task_type=..., loss=LossConfig(strategy="cross_entropy"), ...)`.
  Without this update, every `test_tft.py` test that calls `_minimal_tft`
  hits a pydantic `ValidationError` (unknown `loss_strategy` kwarg under
  `extra="forbid"`).

**Checkpoint**:
- Full project gates green (ruff / pyright / pytest).
- All 288 xfail markers still fire (v1.1 task types still rejected).
- Validity-matrix parametrize IDs are stable (pytest cache survives).
- New tests pass: `test_v1_task_type_rejects_multilabel_and_regression_multioutput`,
  `test_nested_base_model_config_model_dump_json_round_trips`,
  `test_family_config_strategy_literals_match_domains`.
- `TFTConfig` (which inherits from `BaseModelConfig`) still constructs
  cleanly because the inherited fields shifted but every test
  construction (including `_minimal_tft`) was updated in this commit.

**Risk**: this is the high-blast-radius commit. Specific mitigations:
- Before committing, search for stale references via:
  `grep -rn "loss_strategy\|imbalance_strategy" src/ tests/ --include=*.py`
  Acceptable remaining hits are limited to (a) function-parameter names
  and error-message string literals in `_validity.py` (those are the
  F5 display labels, preserved by contract per the requirements F5
  bridge table), and (b) `pytest.mark.parametrize` IDs in
  `test_validity_matrix.py` that use the same display-label strings.
  Every other hit must be addressed.
- Run `pyright` aggressively; strict mode catches missed field
  renames as `reportAttributeAccessIssue`.

### Commit 5: adapter rename + extension

**Files**: `git mv src/seq_sklearn/config/_params_adapter.py
src/seq_sklearn/config/_adapters.py`, then EDIT to add five more
adapter classes + `*` keyword-only marker on existing
`TabularConfigParams.__init__`; `git mv
tests/unit/config/test_params_adapter.py
tests/unit/config/test_adapters.py`, then EDIT.

**Dependencies**: all configs in place (`_extras.py`, family
sub-configs, base.py rewrite, tft.py with advanced).

**Checkpoint**:
- Full project gates green (ruff / pyright / pytest).
- All 14 `test_adapters.py` tests pass: 6 clone-safety, 1 keyword-only
  meta-test, 5 `to_pydantic` round-trip tests for the new adapters,
  and the 2 carried-over `TabularConfigParams` tests
  (`test_to_pydantic_produces_frozen_config`,
  `test_to_pydantic_propagates_validation_errors`) with intent
  preserved post-rename. Of these 14, 12 are named in the Phase 1
  manifest; the 2 carried-over tests pre-date this refactor.

**Risk**: low. The existing adapter pattern is proven; the five new
adapters follow the same recipe. The keyword-only marker is the only
behavioral change to `TabularConfigParams`; no caller passes its
fields positionally so this is non-breaking.

### Commit 6: final sweep

**Files**: none new; this commit runs full gates and fixes any
end-state lint / format / pyright residue.

**Actions**:
- `ruff format .` (apply any auto-formatting).
- Re-read every changed file end-to-end for missed prose updates
  (CLAUDE.md anti-tell rules: no em dashes introduced; no banned
  vocabulary in new docstrings).
- `pyright` final pass with no errors.
- `pytest --cov` final pass: line >= 85%, branch >= 80%, all
  Phase 1 tests pass, all 288 v1.1 xfails still fire.

**Checkpoint**: PR-ready end state. Open the PR.

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Stale `loss_strategy` / `imbalance_strategy` references in non-test code | Medium | `AttributeError` / pyright `reportAttributeAccessIssue` when downstream code reads a renamed field (e.g. `cfg.loss_strategy` after it moved to `cfg.loss.strategy`) | Pyright strict mode + the pre-commit-4 grep script (documented in commit 4) surface every hit before the commit lands |
| Pydantic v2 nested-validator order subtleties | Low | Pydantic v2 validates nested submodels before the parent's `mode='after'` validator runs; confirmed stable as of pydantic 2.12 | Pinned in `pyproject.toml`; no v2.11 fallback |
| `_check_extra_not_reserved` model_validator interaction with `BeforeValidator` on `extra` | Low | `BeforeValidator` runs at validation time (parses input → stored tuple); `model_validator(mode="after")` runs after all field validators complete. Order is well-defined | Each family sub-config tests the collision case explicitly |
| Test parametrize IDs change | Medium | The validity-matrix parametrize tuple stays 4 strings; only the test body changes | The `check_combo` signature is unchanged, so the 4-string parametrize tuple (and the pytest cache) stays stable across the refactor |
| Hashability regression on nested configs | Low | Every sub-config is `frozen=True` with `extra: ExtraDict` (a sorted tuple, hashable); pydantic's default hash should compose | Add `hash(cfg)` assertion to one test per nested-config layer if pyright signals trouble |
| Coverage drops below 85% | Low | Every new src module ships with a 1:1 test module (the Phase 1 module/test pairing); net additions: ~700 lines src + ~600 lines tests | `pytest --cov` at commits 1, 2, 4, 5 to catch drift early |
| Import cycle when adapters reference family configs | Low | Each adapter imports its specific pydantic config; family configs do not import adapters. Acyclic | Verify with `python -c "import seq_sklearn.config.adapters"` at commit 5 |
| `_DEFAULT_LOSS_FOR_TASK` injection misplaced | NA in Phase 1 | The map and injection logic live in Phase 6a's `models/_base.py`, not Phase 1. `LossConfig.strategy` shipping with no default is by design; callers must pass `loss=LossConfig(strategy=...)` until Phase 6a lands the estimator-side injection | Out of scope for Phase 1; the test `test_construction_requires_strategy` pins the no-default contract |

## Validation matrix

What runs at each checkpoint:

Policy: full project gates run after every commit so bisection narrows
to one commit. The "narrowest subset that must pass" column names the
tests most directly load-bearing for that commit (those that would
catch a commit-specific regression).

| Checkpoint | ruff check | ruff format | pyright | pytest (full) | Narrowest subset that must pass |
|---|---|---|---|---|---|
| After commit 1 | full project | full project | strict on `src/` | green | `test_extras.py` (10 named tests) |
| After commit 2 | full project | full project | strict on `src/` | green | `test_{optimizer,scheduler,loss,sampler}.py` (13 named tests) |
| After commit 3 | full project | full project | strict on `src/` | green | `test_tft.py` |
| After commit 4 | full project | full project | strict on `src/` | green | `test_base.py`, `test_validity_matrix.py`, `test_tft.py` (validity-matrix parametrize IDs stay stable; 288 xfails still fire) |
| After commit 5 | full project | full project | strict on `src/` | green | `test_adapters.py` (14 tests: 12 named in the manifest + 2 carried-over `TabularConfigParams` `to_pydantic` tests) |
| Final PR gate | full project | full project | strict on `src/` | green + `--cov` line >= 85 / branch >= 80 | all of the above |

## Out of scope for this refactor

These appear in requirements / architecture / plan but ship in later
phases. Listed here only to prevent scope creep during the refactor.

- `models/_base.py` `BaseSequenceEstimator` (Phase 6a). Where
  `_DEFAULT_LOSS_FOR_TASK` lives. Where the task-type-aware
  injection happens.
- `models/_base.py` `__sklearn_tags__` (Phase 6a). The flat-tag block.
- `training/optimizers.py` `build_optimizer` (Phase 4). Consumes
  `OptimizerConfig`. Does NOT re-check the reserved-keys collision
  (config-layer validator owns that; Gemini-pass fix).
- `training/_lightning_module.py` (Phase 4). The `_pending_prune`
  pattern.
- `tuning/suggest_params.py` with `search_advanced` / `search_extras`
  flags (Phase 8).
- `tuning/_alpha_keys.py` and `tuning/_config_to_estimator_kwargs.py`
  (Phase 8).
- `inference/attention.py` `AttentionOutput` / `RegressionAttentionOutput`
  (Phase 6b).

The refactor lands the configuration layer only. Every downstream
consumer (estimators, training, inference, tuning) follows in its
respective phase.

## Effort estimate

- Commit 1 (extras + 10 tests): 1.5-2.5 hours.
- Commit 2 (4 family sub-configs + 13 tests across 4 files): 3-4 hours.
- Commit 3 (TFTAdvancedConfig + 2 tests): 30 minutes.
- Commit 4 (base.py rewrite + test_base / test_validity_matrix / test_tft
  body rewrites + 3 new tests): 3-4 hours (the highest-risk commit;
  budget time for stale-reference cleanup).
- Commit 5 (adapter rename + 5 new adapters + 12 tests): 2-3 hours.
- Commit 6 (final sweep + PR prep): 1 hour.

Total: 11-15 hours of focused engineering work.

The implementation plan's Phase 1 estimate is 5-7 days (post-fold-in
bump). This refactor consumes part of that budget; the remainder
covers any iteration on review feedback and the original Phase 1
acceptance gates.

## Acceptance criteria

The refactor is done when ALL of the following hold:

1. `git status` shows the source tree matches the target-state layout
   above (no orphaned files, no missing files).
2. Final `pytest --cov` reports >= 85% line / >= 80% branch on
   `src/seq_sklearn/`.
3. Final `pyright` reports 0 errors in strict mode on `src/`.
4. Final `ruff check` and `ruff format --check` are clean.
5. All 40 named tests in this plan's manifest pass.
6. All previously-passing Phase 1 tests still pass (or are
   intentionally rewritten with documented rationale in this plan's
   "Test rewrite manifest").
7. The 288 v1.1 task-type `xfail` markers still fire (regression
   safeguard).
8. `grep -rn "_params_adapter" src/ tests/ --include=*.py` returns
   ZERO matches outside docstrings and comments (the file was
   renamed to `_adapters.py`; any remaining hit signals a missed
   import or reference).

   Separately, `grep -rn "loss_strategy\|imbalance_strategy" src/
   tests/ --include=*.py` may return matches, but ONLY in:
   - `src/seq_sklearn/config/_validity.py` parameter names and
     error-message string literals (these are the F5 display labels
     preserved by contract; see requirements F5 bridge table);
   - `tests/unit/config/test_validity_matrix.py` `pytest.mark.parametrize`
     IDs and parameter strings that mirror the F5 display labels;
   - test parametrize IDs and string-form assertion `match=` patterns
     that mirror the F5 display-label vocabulary.

   Any other hit (e.g. an attribute access `cfg.loss_strategy` or a
   docstring referring to a no-longer-existing field) is a missed
   rename and must be addressed before merge.
9. The git history within the PR has 6 logical commits matching the
   sequencing above (rebasable / squashable if the maintainer prefers
   one commit on merge).

## Addressed

> Historical note: this is the frozen swarm-review ledger for the
> completed Phase 1 refactor. Entries below cite
> `docs/hyperparameter_strategy.md` line numbers (e.g. `:176-193`,
> `:1051`) that were valid at review time. That doc was subsequently
> demoted and slimmed; those line references are historical. The
> authoritative schemas now live in architecture A4 and
> `src/seq_sklearn/config/`.

Round 1 swarm tally: 12 CRITICAL, 14 IMPROVEMENT, 11 NITPICK across
architecture-reviewer, qa-test-coverage, and style-reviewer. Findings
resolved in this revision:

**Architecture-reviewer**
- C1: `test_tft.py` `_minimal_tft` helper constructs via flat
  `loss_strategy="cross_entropy"` kwarg; after the base.py rewrite at
  commit 4 it would raise `ValidationError`. Resolved by adding
  `test_tft.py` to commit 4's file set with explicit rationale in the
  commit narrative.
- C2: Acceptance criterion 8's grep was unsatisfiable because
  `_validity.py` keeps `loss_strategy` / `imbalance_strategy` as
  function-parameter names and error-message string literals per the
  F5 bridge contract. Resolved by rewriting criterion 8 to (a) require
  zero `_params_adapter` matches outside docstrings, and (b) enumerate
  the F5-display-label carve-outs as acceptable for `loss_strategy` /
  `imbalance_strategy` matches.
- C3: Literal-to-domain drift could change the F5 failure-mode contract
  (sub-config Literal error vs. `BaseModelConfig` ConfigError). Resolved
  by adding `test_family_config_strategy_literals_match_domains` to
  `test_validity_matrix.py`.
- I1: `_extras.py` dependencies line now names `seq_sklearn.errors.ConfigError`.
- I2: Validation matrix and commit 3 narrative now both state full
  project gates per commit (bisection policy).
- I5: Test-count derivation now explicitly enumerates the 23 + 1 + 16
  arithmetic and the source of each addition.
- I7: Module-count line corrected to "+5 new source modules" instead of
  "+6", with explicit accounting for the source-file rename plus the
  test-file mirror rename.

**qa-test-coverage**
- C1: `_check_extra_not_reserved` had no collision tests on Scheduler /
  Loss / Sampler. Resolved by adding `test_reserved_keys_collision_raises`
  to each of `test_scheduler.py`, `test_loss.py`, `test_sampler.py`.
- C2: nested `BaseModelConfig` JSON round-trip was untested. Resolved
  by adding `test_nested_base_model_config_model_dump_json_round_trips`
  to `test_base.py`, with non-default values on every nested family
  sub-config including non-empty `extra` tuples.
- C3: `extra="forbid"` propagation on each family sub-config was
  unverified. Resolved by adding `test_extra_field_rejected` to each
  of `test_optimizer.py`, `test_scheduler.py`, `test_loss.py`,
  `test_sampler.py`.
- I2: `to_pydantic()` on the five new adapters was untested. Resolved
  by adding one `test_{family}_params_to_pydantic_produces_correct_config`
  per new adapter in `test_adapters.py`.
- I4: `_normalize_extras(None)` sentinel path was untested. Resolved by
  adding `test_normalize_extras_accepts_none_produces_empty_tuple` to
  `test_extras.py`.
- I6: `test_all_adapters_have_keyword_only_init` assertion sharpened to
  require EVERY non-self parameter has `kind == Parameter.KEYWORD_ONLY`
  (not merely `POSITIONAL_OR_KEYWORD` with a default).
- I7: Commit 3 vs commit 4 ambiguity on `_minimal_tft` resolved by
  explicit narrative: commit 3 leaves the helper untouched; commit 4
  rewrites it as part of the base.py shape change.
- I1: `_extras.py` test gains `test_extra_dict_rejects_non_string_key`
  (architecture I4 / QA's overlapping concern).

**style-reviewer**
- All 6 CRITICAL em dashes resolved: 18 list-item separators replaced
  with `: ` (test-name to description), and 5 prose em dashes replaced
  with periods, semicolons, or commas per surrounding-clause grammar.

Net effect on the test count: 24 named tests (round 1) became 40 named
tests (round 2). All 16 additions trace to one of the CRITICAL or
IMPROVEMENT findings above; none represent a new architectural delta
beyond the four-tier spec already documented in `docs/requirements.md`,
`docs/architecture.md`, and `docs/implementation_plan.md` (with the
rationale in `docs/hyperparameter_strategy.md`).

### Round 2

Round 2 swarm tally: 0 CRITICAL, 5 IMPROVEMENT, 6 NITPICK. Style and
architecture APPROVE; QA REQUEST_CHANGES on two doc-sync items. All
three Round 1 CRITICALs verified resolved by both architecture and QA
reviewers. Round 2 findings resolved in this revision:

- **QA IMP-1 / Arch I2** (commit 5 checkpoint undercounted
  `test_adapters.py` as 12 when 14 tests run): resolved. The commit 5
  checkpoint and the validation matrix now state 14 tests (12 named in
  the manifest + 2 carried-over `TabularConfigParams` `to_pydantic`
  tests pre-dating this refactor).
- **QA IMP-2** (extras test 4 `test_extra_dict_survives_json_roundtrip`
  did not name its outer pydantic container, so a bare `_normalize_extras`
  call could satisfy it without exercising the pydantic field path):
  resolved. The test now constructs an `OptimizerConfig` and asserts
  `cfg == reconstructed`.
- **QA IMP-3** (SchedulerConfig spec too thin to verify the defaults
  asserted by `test_default_construction_uses_documented_defaults`):
  resolved. The SchedulerConfig spec block now carries the full field
  list, Literal menu, and defaults sourced from requirements F5
  (requirements.md:755-766).
- **Arch N1** (risk-register row put a mitigation summary in the Impact
  column): resolved by rewriting the row so Impact describes the
  consequence and Mitigation holds the pyright + grep safeguard.

### Round 3

Round 3 swarm tally: 1 CRITICAL, 0 IMPROVEMENT, 1 NITPICK. QA and
style APPROVE; architecture REQUEST_CHANGES on one CRITICAL introduced
by the Round 2 QA-IMP-3 fix.

- **Arch C1 (Round 3)**: the SchedulerConfig spec block added in
  Round 2 used hand-derived `plateau_*` defaults
  (`plateau_factor=0.1`, `plateau_patience=10`) and omitted
  `plateau_threshold` and `min_lr`, contradicting the authoritative
  schema at `docs/hyperparameter_strategy.md:176-193`. Resolved by
  replacing the block with a verbatim copy of the strategy-doc schema
  (`plateau_factor=0.5`, `plateau_patience=5`, plus
  `plateau_threshold=1e-4` and the cosine `min_lr=0.0` field) and
  adding an explicit "do not re-derive; mirror the strategy doc"
  instruction above the block. Root cause: the Round 2 fix sourced
  defaults from the requirements F5 prose menu, which describes
  behavior but does not enumerate every sub-field default; the
  strategy doc's code block is the authoritative schema and should
  have been the source.
- **QA N1 (Round 3, carried)**: the target-state tree comment said
  `test_extras.py NEW: 8 named tests` while the body specifies 10.
  Resolved by correcting the tree comment to 10.

- **Arch I6**: Validator-order test on `BaseModelConfig` (assert
  `_check_validity_matrix` fires before `_check_quantiles_monotone`
  before `_check_val_cal_sum`). Deferred. Reason: pydantic v2 runs
  `model_validator(mode="after")` in declaration order; the current
  declaration order in `base.py` is correct and the order is preserved
  literally by the rewrite. A test pinning the order would catch a
  future reordering, but the cost of writing a single-purpose
  "validator order" test outweighs the benefit when the order is
  visible in the source. Re-raise if pydantic ever introduces
  non-declaration-order validator dispatch.
- **QA I3**: `hash(cfg)` assertion at the composed `BaseModelConfig`
  level. Deferred. Reason: the risk register already covers this with
  a conditional "add `hash(cfg)` assertion if pyright signals trouble"
  mitigation. Adding the assertion eagerly bloats the test surface
  without catching a known failure; if a downstream Optuna trial
  dedup path needs hashability, the failure surfaces there with a
  clear traceback.
- **QA I5**: `constant` scheduler rejecting `warmup_steps` validator.
  Deferred. Reason: the behavior IS documented (requirements F5 /
  arch I10, requirements.md:763 and 2102-2104: "`constant`. No
  schedule. `warmup_steps` ignored; setting it raises `ConfigError`"),
  but the enforcement layer is not pinned. It could live on
  `SchedulerConfig` (config layer), in `check_combo` (validity layer),
  or in `build_optimizer` (Phase 4). Adding the test in this refactor
  would pre-commit to the SchedulerConfig home without an upstream
  decision. Surface as a discovery for the Phase 4 design pass.

- **Arch I1 (discovery, not a plan defect)**: the
  `test_all_adapters_have_keyword_only_init` assertion in
  `docs/hyperparameter_strategy.md:1051` is internally contradictory:
  it requires every non-self parameter to be `POSITIONAL_OR_KEYWORD`
  with a default AND the first non-self parameter to be
  `KEYWORD_ONLY` (a parameter cannot be both). This refactor plan
  resolves the contradiction by specifying the sharper, correct
  assertion (every non-self parameter is `KEYWORD_ONLY`). Per
  acceptance criterion 5 (no silent doc deltas), the upstream
  strategy-doc wording is NOT edited inside this refactor PR; it is
  surfaced here as a one-line follow-up fix to
  `docs/hyperparameter_strategy.md:1051` to re-sync that doc with the
  test this refactor will actually write. Tracked for a separate
  strategy-doc edit; does not block this plan's consensus because the
  refactor's own assertion is unambiguous and correct.
- **Arch C3 sub-concern**: the architecture reviewer noted that
  pinning the validator-order at the `BaseModelConfig` level is
  related to C3 (Literal-to-domain drift). The Literal/domain
  invariant test (added) covers the C3 failure-mode-contract concern
  directly; the validator-order test is the deferred Arch I6 above
  and does not affect the F5 contract.
- **Round 1 NITPICKs** (architecture N1-N6, QA N1-N4) and **Round 2
  NITPICKs** (architecture N2-N3 on count-callout placement and effort
  double-statement, QA N1-N3 on naming `task_type` in the round-trip
  test, the equality-assertion form for the primitive round-trip test,
  and a second scheduler-name collision case): optional polish. Arch
  N1 from Round 2 was addressed (not deferred). The remainder are not
  material to consensus; the test specs are unambiguous enough for an
  implementer to write correct tests, and any picked-but-valid value
  (e.g. `task_type="binary"` for the round-trip test) satisfies the
  contract.
