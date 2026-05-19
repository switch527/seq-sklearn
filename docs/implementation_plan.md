# Implementation plan: seq-sklearn v1

## Requirements

This plan orders the v1 build into testable phases such that:

1. **Every phase produces an independently testable artifact.** The next
   phase can be paused, the work to date verified, and a regression
   bisected to a single phase.
2. **Earlier phases lock contracts later phases depend on.** Tensor
   shapes, mask polarity, the LightningModule constructor signature, the
   pydantic adapter shape, the `BackboneOutput` dataclass, the save /
   load schema version, the validity matrix domains. Anything that
   ships in v2 / v3 must inherit these contracts unchanged. Any
   contract that has to change once a second model lands is rework.
3. **No phase ships code without the tests that gate it.** Coverage
   gate (85% line, 80% branch on `src/seq_sklearn/`, per N1) holds at
   every phase boundary, not at the end of v1.
4. **The order respects the dependency DAG.** Foundation primitives
   first, pure data primitives second, tensor components third,
   training plumbing fourth, estimator shells fifth, concrete TFT
   sixth, polish last. No phase calls forward into a phase that has
   not landed.
5. **Each phase is a PR-sized unit of work.** A phase produces one
   reviewable commit (or short stack of commits), passes the swarm,
   merges, then the next phase starts on the new main.

This plan is graded against:

- The v1 contract in `docs/requirements.md` (F1 through F11, N1
  through N7).
- The architecture in `docs/architecture.md` (A1 through A20).
- The library philosophy in CLAUDE.md (rule 3: every new code path
  has tests; coverage delta cannot decrease).

CRITICAL findings against this plan must trace back to one of the
above. Anything else is IMPROVEMENT or NITPICK.

## Goals

1. **Build-test rhythm**: every phase ends with a passing test suite
   that covers the code added in the phase. The dev inner loop
   (`pytest -m "not slow and not perf and not gpu"`) stays green at
   every boundary.
2. **Minimize rework across v1 → v1.1 → v2 → v3**: the architectural
   seams that absorb future models are locked in v1 and exercised by
   the recurrent skeleton (A6.1). Any future model is a new concrete
   class plus a per-model search-space block, not an infrastructure
   rewrite.
3. **End-to-end signal early**: a smoke skeleton runs a degenerate
   fit / predict round-trip on a dummy backbone before TFT-specific
   code lands. This catches estimator-shell contract bugs before they
   get tangled with TFT block bugs.
4. **Bisectable failures**: every phase is small enough that a CI
   regression bisects to a single phase. Phases that touch more than
   ~600 lines of `src/` get split.

## Principles

- **Bottom-up by default.** Pure helpers, then pure data, then tensor
  components, then training, then estimator shells. The exception is
  the smoke skeleton in Phase 6, which forces an early end-to-end
  validation against the contracts the bottom-up phases produced.
- **Dummies for the not-yet-implemented.** Phases 3 through 5 use
  `_DummyBackbone`, `_DummyHead`, `_LossReturningScalar` (per A7's
  `make_test_module` fixture pattern) so tests do not require TFT
  blocks. Phase 7 substitutes the real TFT components into the same
  contracts.
- **Decouple at construction time.** The Trainer wraps an Estimator's
  pydantic config into curried `optimizer_factory` /
  `scheduler_factory` callables and hands them plus the backbone /
  head / loss modules to the LightningModule (A7 construction order
  block). The LightningModule never reads the Estimator. This makes
  every LightningModule test a 6-line construction.
- **Lock the data contract once.** `TabularToSequence.transform`
  emits a stable `dict[str, Tensor]` (A5). Every model family
  consumes this dict. v2 PatchTST / TimesNet / TST and v3 LSTM / GRU /
  LSTM-FCN inherit the same dict shape.
- **Frozen pydantic + mutable adapter once.** The
  BaseEstimator-adapter pattern (A4 step 3) is the canonical way to
  reconcile sklearn's mutation contract with pydantic's frozen
  contract. Every estimator follows it. The pattern is validated in
  Phase 1 before any concrete estimator exists.
- **Coverage delta only goes up.** Every PR closes its own coverage
  obligations. Phase boundaries do not carry test debt forward.

## Phased plan

### Phase 0: Scaffold

**Goal**: a working dev environment that lints, type-checks, and
runs an empty test suite green.

**Modules**:

- `pyproject.toml` per A18, with dependency pins and the `dev`,
  `onnx`, `mlflow`, `wandb`, `docs` extras.
- `src/seq_sklearn/__init__.py` (empty `__all__`).
- `src/seq_sklearn/py.typed` (PEP 561 marker).
- `tests/__init__.py`, `tests/conftest.py` (skeleton; fixtures added
  in later phases).
- `tests/unit/`, `tests/integration/`, `tests/e2e/`, `tests/deploy/`,
  `tests/perf/`, `tests/snapshot/`, `tests/_snapshots/` directories.
- `ruff.toml` / `pyproject.toml` ruff config matching the
  CLAUDE.md style rules (no em dashes, banned vocabulary cannot
  appear in `# noqa`-able forms).
- `pyrightconfig.json` strict mode.
- `.pre-commit-config.yaml` running ruff and pyright on every
  commit, plus the snapshot-marker check from A14.
- `.github/workflows/pr.yml` (lint, type, test-unit, test-deploy,
  docs, snapshot-guard jobs per A19; tests jobs are no-ops until
  Phase 1 lands real tests; the docs job is a no-op script until
  Phase 12 lands `mkdocs.yml`, then flips to `mkdocs build
  --strict`).
- `.github/workflows/nightly.yml` skeleton per A19.
- `scripts/check_snapshots.sh` per A14.
- `CHANGELOG.md`, `CONTRIBUTING.md`, `SECURITY.md`,
  `CODE_OF_CONDUCT.md`, `LICENSE` (Apache-2.0), the GitHub issue
  and PR templates per N3.

**Dependencies**: none.

**Deliverable tests**: `pytest` collects zero items and exits 0;
`ruff check .` and `pyright` pass on the empty source tree.

**Done when**: a PR opened from a clean clone shows all CI jobs
green and the wheel builds. Wall-clock under the N2 5-minute
budget verified empirically.

### Phase 1: Foundation primitives

**Goal**: every pure-Python module that the rest of the library
imports. No torch dependency at the import surface of any module in
this phase except `hardware.py`, which uses `torch.cuda` for
detection but stays a pure function.

**Modules**:

The hyperparameter-strategy refactor (per architecture A4 /
requirements F7) lands the four-tier config hierarchy in Phase 1. The module roster expands to include
family sub-configs (`optimizer.py`, `scheduler.py`, `loss.py`,
`sampler.py`), the `extras` infrastructure (`_extras.py` with
`ExtraDict`, `_normalize_extras`, `_PROMOTED_KEYS_BY_FAMILY`,
`extract_deprecated_extras`), and the renamed multi-adapter
module (`_adapters.py`).

- `src/seq_sklearn/errors.py` (A10): full exception hierarchy
  including `NotFittedError` with the MRO-load-bearing dual parent.
- `src/seq_sklearn/logging.py` (A15): `Event` enum + `emit` helper.
- `src/seq_sklearn/_validate.py`: `check_y`, `check_columns` per F1.1
  and F2. v1 rejects multi-output `y` with the documented message.
- `src/seq_sklearn/hardware.py` (A11): `HardwareTier` IntEnum +
  `detect()` pure function. Detection avoids `torch.cuda.device_count`
  / `current_device` per A11.
- `src/seq_sklearn/config/_domains.py` (F5): `TASK_TYPES`,
  `LOSS_STRATEGIES`, `IMBALANCE_STRATEGIES`, `CALIBRATION_STRATEGIES`
  as tuple constants. Single source of truth.
- `src/seq_sklearn/config/_validity.py`: `check_combo(task_type,
  loss_strategy, imbalance_strategy, calibration_strategy)` per the
  F5 matrix. Signature unchanged (still four strings); the call
  site inside `BaseModelConfig._check_validity_matrix` reads from
  the nested family configs (`self.loss.strategy`,
  `self.sampler.strategy`). Returns `None` on legal cells, raises
  `ValueError` on illegal cells with a message naming legal
  alternatives. Pure function so the Optuna search-space sampler
  and the cross-field validator share it.
- `src/seq_sklearn/config/_extras.py` (new per architecture A4):
  `ExtraValue` restricted-union type, `ExtraDict` annotated tuple,
  `_normalize_extras` BeforeValidator, `_PROMOTED_KEYS_BY_FAMILY`
  registry (empty in v1; populated as ALPHA → BETA promotions
  occur), `extract_deprecated_extras` helper. Single source of
  truth for the escape-hatch contract.
- `src/seq_sklearn/config/optimizer.py` (new per architecture A4):
  `OptimizerConfig` family sub-config (`name`, `learning_rate`,
  `weight_decay`, `betas`, `eps`, `momentum`, `nesterov`, `extra`).
  Frozen pydantic, `extra="forbid"`.
- `src/seq_sklearn/config/scheduler.py` (new per architecture A4):
  `SchedulerConfig` family sub-config (`name`, `warmup_steps`,
  OneCycleLR / ReduceLROnPlateau / cosine sub-fields, `extra`).
- `src/seq_sklearn/config/loss.py` (new per architecture A4):
  `LossConfig` family sub-config (`strategy` with NO default per
  F5, `focal_gamma`, `focal_alpha`, `huber_delta`,
  `label_smoothing`, `extra`).
- `src/seq_sklearn/config/sampler.py` (new per architecture A4):
  `SamplerConfig` family sub-config (`strategy`,
  `oversample_ratio`, `replacement`, `extra`).
- `src/seq_sklearn/config/base.py`: `BaseTrainingConfig`,
  `BaseModelConfig` (A4). Pydantic v2 frozen models. Both nest
  the family sub-configs (`optimizer: OptimizerConfig`,
  `scheduler: SchedulerConfig`, `loss: LossConfig`, `sampler:
  SamplerConfig`); cross-cutting fields (precision, seed,
  val/cal_fraction, etc.) stay flat per architecture A4 (Tier 2).
  `BaseModelConfig._check_validity_matrix` calls
  `check_combo` reading `self.loss.strategy` and
  `self.sampler.strategy` from the nested family configs.
- `src/seq_sklearn/config/tabular.py`:
  `TabularToSequenceConfig` (A4). `categorical_embed_dims` is
  `CategoricalEmbedDims = tuple[tuple[str, int], ...]` with a
  `BeforeValidator` that accepts dict / Mapping / tuple input
  (the earlier `Mapping[str, int]` claim did not deliver
  hashability in pydantic v2).
- `src/seq_sklearn/config/tft.py`: `TFTConfig` per A4 plus the new
  `TFTAdvancedConfig` (Tier 3, BETA, ships empty in v1 plus the
  `extra` escape hatch). `TFTConfig.advanced:
  TFTAdvancedConfig = Field(default_factory=TFTAdvancedConfig)`
  per architecture A4. TFT-specific cross-field validators
  (`attention_heads divides hidden_size`) live here. The base-
  class validators (`quantiles strictly increasing in (0, 1)`,
  validity-matrix check) live on `BaseModelConfig`.
- `src/seq_sklearn/config/_adapters.py` (renamed from
  `_params_adapter.py` per architecture A4): six BaseEstimator
  adapters, one per nested pydantic sub-config. Every adapter
  `__init__` carries the `*` keyword-only marker (mandatory per A4;
  without it the ALPHA → BETA promotion path silently breaks
  positional callers).
  Adapters: `TabularConfigParams` ← `TabularToSequenceConfig`,
  `OptimizerParams` ← `OptimizerConfig`, `SchedulerParams` ←
  `SchedulerConfig`, `LossParams` ← `LossConfig`, `SamplerParams`
  ← `SamplerConfig`, `TFTAdvancedParams` ← `TFTAdvancedConfig`.
  Each has `to_pydantic()` constructing the frozen instance.

**Dependencies**: Phase 0.

**Deliverable tests** (all under `tests/unit/`):

- `tests/unit/test_errors.py`: each subclass raises and catches at
  the right MRO level; `NotFittedError` is catchable as both
  `SeqSklearnError` and `sklearn.exceptions.NotFittedError`.
- `tests/unit/test_logging.py`: `emit` produces a record whose
  `extra` keys land as attributes on the `LogRecord`
  (`record.event`, `record.payload`), confirming the `caplog`
  contract.
- `tests/unit/test_validate.py`: `check_y` rejects 2-D y with the
  documented message; `check_columns` raises `DataContractError`
  on missing columns, duplicate `(id, time)`, object-dtype time
  columns, tz mixing.
- `tests/unit/test_hardware_detect.py` (one of N1's required
  tests): parametrized over the six tiers + CPU; each parametrized
  case patches `torch.cuda.is_available` and
  `torch.cuda.get_device_capability` and asserts the returned
  `HardwareTier`. Combined with `resolve_precision` from Phase 4
  in a follow-up test once Phase 4 lands.
- `tests/unit/config/test_validity_matrix.py` (one of N1's required tests):
  parametrized over the Cartesian product of the four domain
  enumerations from `_domains.py` minus the legal cells listed in F5.
  Every illegal cell raises and the error message names both the
  offending fields and a legal alternative. The test body constructs
  configs via the nested shape: pass
  `LossConfig(strategy=loss_strategy)` and
  `SamplerConfig(strategy=imbalance_strategy)` as sub-configs to
  `BaseModelConfig`. The parametrization tuple stays 4 strings (the
  `check_combo` signature is unchanged) so pytest IDs are stable. v1.1 task-type rows carry
  `pytest.mark.xfail(strict=True)` so a future change that
  accidentally accepts a v1.1 task type fails CI rather than
  silently converting to a pass.
- `tests/unit/config/test_extras.py` (new per architecture A4; lands
  in Phase 1): covers `_normalize_extras` type validation, hash
  stability, JSON round-trip via `model_dump(mode="json")`, and
  the deprecation-alias machinery. Named tests:
  `test_extra_dict_rejects_non_primitive_value`,
  `test_extra_dict_round_trips_each_primitive_type`,
  `test_extra_dict_stored_as_sorted_tuple`,
  `test_extra_dict_survives_json_roundtrip`,
  `test_extract_deprecated_extras_meta_promoted_keys_exist`,
  `test_extract_deprecated_extras_both_typed_and_extra_raises_config_error`,
  `test_extract_deprecated_extras_happy_path_passes_through`
  (asserts unpromoted `extra` keys pass through unchanged with no
  `DeprecationWarning`),
  `test_extract_deprecated_extras_mock_promotion_emits_warning`
  (Phase 1 mock variant of the deprecation-warning test; the real
  post-promotion test
  `test_extra_path_after_promotion_emits_deprecation_warning`
  is deferred until the first ALPHA→BETA promotion).
- `tests/unit/config/test_optimizer.py` (new per architecture A4):
  `test_default_construction_uses_documented_defaults` asserts
  `OptimizerConfig()` produces the documented field defaults;
  `test_adamw_reserved_keys_collision_raises` and
  `test_sgd_reserved_keys_collision_raises` assert
  `OptimizerConfig(name=..., extra=(("collision_key", value),))`
  raises `ValidationError` from the `_check_extra_not_reserved`
  model_validator at config construction (per architecture A4; the
  check lives on the config, not on `build_optimizer`, so Phase 1
  owns the test).
- `tests/unit/config/test_scheduler.py` (new per architecture A4):
  `test_default_construction_uses_documented_defaults` asserts
  `SchedulerConfig()` produces the documented field defaults.
- `tests/unit/config/test_loss.py` (new per architecture A4):
  `test_construction_requires_strategy` asserts `LossConfig()`
  without `strategy=` raises `ValidationError` (no default per F5).
- `tests/unit/config/test_sampler.py` (new per architecture A4):
  `test_default_strategy_is_none` asserts `SamplerConfig()`
  produces `strategy="none"`.
- `tests/unit/config/test_base.py`: legal `BaseModelConfig`
  constructions succeed; mutation post-construction raises
  (frozen); `extra="forbid"` rejects unknown fields. The
  `quantiles strictly increasing in (0, 1)` cross-field validator
  fires on non-monotone or out-of-(0, 1) values; the test
  parametrizes over malformed quantile vectors. Includes
  `test_v1_task_type_rejects_multilabel_and_regression_multioutput`
  pinning the v1.1-unreachable guard per the architecture A4
  fold-in note.
- `tests/unit/config/test_tabular.py`: same as above for
  `TabularToSequenceConfig`. The hashability assertion confirms
  `hash(config)` succeeds against the `CategoricalEmbedDims`
  tuple form.
- `tests/unit/config/test_tft.py`: legal `TFTConfig` constructions
  succeed; `attention_heads` not dividing `hidden_size` raises
  via the TFT-specific cross-field validator; mutation
  post-construction raises. The quantile validator is tested in
  `test_base.py` (it lives on `BaseModelConfig`, not
  `TFTConfig`). Per architecture A4:
  `test_tft_config_advanced_field_is_not_none_by_default` and
  `test_tft_advanced_config_default_construction_succeeds` pin
  the `advanced` slot's non-None empty-default contract.
- `tests/unit/config/test_adapters.py` (renamed from
  `test_params_adapter.py` per architecture A4): every adapter
  composes with `sklearn.base.clone`. Six named per-adapter
  clone-safety tests:
  `test_tabular_config_params_clone_is_independent`,
  `test_optimizer_params_clone_is_independent`,
  `test_scheduler_params_clone_is_independent`,
  `test_loss_params_clone_is_independent`,
  `test_sampler_params_clone_is_independent`,
  `test_tft_advanced_params_clone_is_independent`. Plus
  `test_all_adapters_have_keyword_only_init` introspecting every
  adapter's `__init__` signature for the `*` marker.
  `get_params(deep=False)` returns the flat field dict;
  `set_params(...)` mutates in place; `to_pydantic()` builds the
  frozen pydantic instance and re-raises pydantic
  `ValidationError` as the caller's context allows.

**Done when**: all unit tests pass; coverage on `src/seq_sklearn/`
hits 90%+ on the modules in this phase (it should, they are pure
Python with no fallback branches).

### Phase 2: Synthetic data and preprocessing

**Goal**: end-to-end "panel DataFrame in, batched tensors out".
This is the input contract every model family consumes; locking it
here pays off across v2 and v3.

**Modules**:

- `src/seq_sklearn/data/synthetic/_rng.py`: single-`Generator`
  threading helper per F6 step 1.
- `src/seq_sklearn/data/synthetic/generator.py`: full DGP per F6.
  `dgp_version=1` stamped on every generator instance. Canonical
  seed triple `(42, 137, 9999)`.
- `src/seq_sklearn/data/encoders.py`: `CategoricalEncoder` (per-column
  vocabulary + `<unk>` index 0), scaler factories (`standard`,
  `robust`, `quantile_uniform`, `none`). Each scaler is a pure-Python
  wrapper around scikit-learn's primitives so save / load JSON-
  serializes the statistics directly.
- `src/seq_sklearn/data/tabular_to_sequence.py`: `TabularToSequence`
  per A5 / F3. The output dict's tensor shapes are locked here.
  `feature_schema_fingerprint` computed and exposed on the fitted
  transformer.
- `src/seq_sklearn/data/splits.py`: `compute_three_way_split` per A5.
  Pure function. Three return shapes per F2 / A5: cal-fold present,
  cal-fold empty when calibration set is external, cal-fold empty
  when calibration is disabled.
- `src/seq_sklearn/model_selection/split.py`:
  `EntityTimeSeriesSplit` per A9.1. Constructor takes `lookback` and
  the `from_estimator` classmethod reads it off an estimator. The
  test fold extends LEFT by `lookback - 1` rows per entity (the
  Gemini-pass fix).

**Dependencies**: Phase 1 (`errors`, `logging`, `_validate`,
`config.tabular`).

**Deliverable tests**:

- `tests/unit/data/test_rng.py`: same seed produces the same byte
  sequence across two `Generator` constructions; reordering steps
  changes output.
- `tests/unit/data/test_synthetic_generator.py`: three-seed median
  threshold sanity (the panels generated at `(42, 137, 9999)` produce
  the documented signal patterns); the F6 step-6 fallback when the
  post-clip vector has fewer than 3 non-zero entries is exercised
  by a synthetic seed that triggers it. The N1
  `test_dgp_version_bump_regression` lives here: generate at
  `dgp_version=1`, change to a hypothetical `dgp_version=2` via
  monkeypatch, assert outputs differ.
- `tests/unit/data/test_encoders.py`: unseen category at transform
  time maps to `<unk>` index 0 without raising (N1 unseen-category
  robustness test); each scaler round-trips through inverse_transform
  to within `atol=1e-6`.
- `tests/unit/data/test_tabular_to_sequence.py`:
  - Variable-history paths: a 1-row entity emits a window of length
    `lookback` with `padding_mask` set on `lookback - 1` positions.
  - The `(id, time)` uniqueness check rejects duplicates with
    `DataContractError`.
  - Object-dtype time column raises `DataContractError`.
  - `feature_schema_fingerprint` is stable across two
    independently-fitted transformers on the same X / dtypes.
  - The mask polarity is `True = padding (ignore)` (assert by
    inspecting a known-short entity).
  - Hypothesis property test: for any panel produced by the
    synthetic generator with `periods_per_entity=(1, 60)`, the
    output `time_varying_real` tensor has shape
    `(n_windows, lookback, len(time_varying_real_cols))`.
- `tests/unit/data/test_splits.py`: `compute_three_way_split` on a
  pinned panel produces train / val / cal folds whose pairwise
  intersections on `(entity_id, window_time_index)` tuples are
  empty (the F2 "disjoint by `(id, time)`" contract, stronger than
  disjoint integer index arrays); the calibration fold is exactly
  the last `cal_fraction` rows per entity; the random policy emits
  a `UserWarning` when more than one entity is present; the
  `calibration_set_provided=True AND cal_fraction > 0` path raises
  `ConfigError` at the split-function level. The estimator-shell
  variant of this `ConfigError` (the F2 fit-time check) is tested
  in Phase 6. `test_calibration_none_threshold_false_collapses_cal_into_train`
  asserts the F2 collapse rule: when `calibration_strategy="none"`
  and `threshold_tuning=False`, `cal_idx` is empty and `train_idx`
  contains the rows that would otherwise have been calibration.
- `tests/unit/model_selection/test_entity_time_series_split.py`:
  `n_splits` splits produced; train and test folds overlap by
  exactly `lookback - 1` rows per entity in the time-ordered
  policy; `gap` measured in windows (an N1-style assertion);
  `from_estimator` reads the lookback off an estimator; entities
  shorter than `n_splits + 1 + gap + lookback - 1` rows are dropped
  with one aggregated `UserWarning`.

**Done when**: a small e2e helper (in tests, not in src) can take a
DGP-generated panel through `TabularToSequence.fit_transform` and
produce the documented dict shape. The integration test
`tests/integration/test_synth_to_tensors.py` covers this.

### Phase 3: Tensor primitives (no fit loop)

**Goal**: every `nn.Module` building block the TFT backbone needs,
plus the save / load framework. No training infrastructure yet.

**Modules**:

- `src/seq_sklearn/serialization.py`: safetensors + JSON save / load
  per A17. `MIGRATIONS` registry empty in v1; `_migrate` enforces
  strict-monotone advancement.
- `src/seq_sklearn/models/_backbone.py`: `BackboneOutput`
  dataclass with the family-agnostic fields (`representation`,
  `padding_mask`) and `BaseBackbone(nn.Module, ABC)` declaring
  `forward(batch) -> BackboneOutput` and a default
  `compute_training_metrics(output) -> {}` that returns no event
  payloads. Concrete backbones override the metric method to emit
  family-specific entropy / norm events; the LightningModule reads
  only the dataclass surface (the two base fields) plus the returned
  dict per architecture A15. Plain `@dataclass` (not `Protocol`) so
  family-specific extensions inherit via standard dataclass
  subclassing and pyright strict mode passes. Lands in Phase 3
  because `TFTBackbone` is the first consumer.
- `src/seq_sklearn/models/_layers.py`: layer factory per A6 / F4. v1
  returns standard PyTorch layers.
- `src/seq_sklearn/models/_attention.py`: mask polarity flip helper
  (`padding_mask` → `attn_mask`). One module, one function. Every
  attention call site routes through it.
- `src/seq_sklearn/models/_heads.py`: `ClassificationHead`,
  `RegressionHead` per A6.
- `src/seq_sklearn/models/transformer/_positional.py`: sinusoidal
  and learned positional encodings.
- `src/seq_sklearn/models/transformer/_interpretable_attention.py`:
  shared-V multi-head attention per A6. Two forward paths: fast
  (SDPA, returns representation only) and interpretable (manual
  softmax, returns representation + `attn_weights`). The N1
  shared-V correctness test (fast vs. interpretable equality
  within float tolerance) lives here.
- `src/seq_sklearn/models/transformer/_backbone.py`:
  `TransformerBackboneOutput` extending `BackboneOutput` with
  `var_selection_weights`, `attention_weights`, and
  `static_var_selection_weights` per architecture A15. Every
  transformer-family backbone (TFT in v1; PatchTST / TimesNet /
  TST in v2) returns this concrete type. Lands in Phase 3 because
  `TFTBackbone` returns it.
- `src/seq_sklearn/models/transformer/tft/blocks.py`: `VSN`, `GRN`,
  `GLU`, `AddNorm` per A6 / F4 / v1 TFT block list.
- `src/seq_sklearn/models/transformer/tft/backbone.py`: `TFTBackbone`
  extends `BaseBackbone`, composes the TFT blocks, and returns
  `TransformerBackboneOutput`. The LSTM init tuple is `(c_h, c_c)`
  (the research-pass correction).
  `pack_padded_sequence(enforce_sorted=False)` for variable-length
  handling. Overrides `compute_training_metrics(output)` to emit
  `train.var_selection_entropy` and `train.attention_entropy`
  payloads. The reductions apply the `padding_mask` so padded
  timesteps (which carry max-entropy uniform VSN rows by
  construction and zero attention rows after `nan_to_num`) do not
  bias the metrics.

**Dependencies**: Phases 1 and 2. Phase 2's
`TabularToSequence.transform` output dict is the backbone's input
contract; the integration test below consumes it directly. The
configs in Phase 1 (`TFTConfig`, `TabularToSequenceConfig`)
parametrize every module in this phase.

**Deliverable tests**:

- `tests/unit/test_serialization.py`: the no-op identity migration
  path (v1 → v1, empty `MIGRATIONS`); the schema-too-new and
  schema-too-old error paths; the `test_migrate_detects_no_op_registration`
  meta-test per A17 (monkeypatched MIGRATIONS with a no-op step
  raises `PredictionError` naming non-advancement); writing and
  reading a `weights.safetensors` + `state.json` pair round-trips
  a small tensor + metadata block byte-identical;
  `test_load_emits_version_mismatch_warning` (one of N1's required
  tests): write a `state.json` with a fake older
  `seq_sklearn_version`, call the low-level reload helper, assert
  exactly one `UserWarning` whose message contains "version
  mismatch" and both versions. The estimator-side integrated
  variant (with a full state dict) is covered in Phase 6.
- `tests/unit/models/test_layers.py`: every factory call returns a
  `torch.nn.Module` instance of the expected type with the expected
  shape parameter.
- `tests/unit/models/test_attention_mask_helper.py`: the polarity
  flip turns `padding_mask` (True = padding) into `attn_mask`
  (True = participate); idempotent under double flip.
- `tests/unit/models/test_heads.py`: each head produces the right
  output shape for binary / multiclass / point / quantile
  configurations; gradients flow.
- `tests/unit/models/transformer/test_interpretable_attention.py`:
  - Shared-V correctness: the V-projection weight count is 1, not
    `n_heads`, asserted by enumerating named parameters with the
    expected key (per A6).
  - Fast path output equals interpretable path output within
    `atol=1e-5, rtol=1e-5` on a fixed input.
  - Mask correctness: forward an entity unpadded vs. padded to
    `lookback`; the valid-position slice byte-equals via
    `torch.equal` in `model.eval()` with deterministic algorithms
    enabled (one of the N1 mask tests).
  - NaN safety: forward a synthetic batch where the mask
    invariant pre-condition holds; assert no NaN appears in the
    interpretable path output (regression guard).
- `tests/unit/models/transformer/tft/test_blocks.py`: each TFT
  block (VSN, GRN, GLU, AddNorm) forwards a (B, L, hidden) tensor
  to a (B, L, hidden) tensor without shape regressions.
  `test_vsn_mask_correctness` (one of the N1 mask tests per the
  "every variable-length-aware layer" clause): under
  `model.eval()` + `torch.use_deterministic_algorithms(True)` +
  fixed seed, forward an unpadded entity through the VSN, then
  forward the same entity padded to `lookback`; assert
  `torch.equal` on the valid-position slice.
  Hypothesis property test: for any `(batch, seq_len, hidden_size)`
  within valid ranges, GRN / VSN output shape equals input shape.
- `tests/unit/models/transformer/tft/test_backbone.py`: the full
  backbone consumes the `TabularToSequence` output dict and emits
  a `TransformerBackboneOutput` whose six fields (the two base
  dataclass fields plus the four transformer-family introspection
  fields) have the documented shapes; the `(c_h, c_c)` ordering is
  asserted on a multi-entity batch (`B >= 2`) with deliberately
  distinct static features so a swap in `c_h` / `c_c` would
  surface (a `B=1` symmetric init cannot distinguish the swap);
  the test patches `nn.LSTM`'s `forward` and inspects the tuple
  positionally. A one-row entity produces a non-NaN
  representation. `test_mean_pool_readout_mask_correctness` (the
  third N1 mask test): with `prediction_readout="mean_pool"`,
  forward an unpadded entity and the same entity padded; assert
  `torch.equal` on the scalar readout.
  `test_compute_training_metrics_ignores_padded_positions`:
  construct a `TransformerBackboneOutput` with uniform-distribution
  rows at padded timesteps and zero attention rows at padded
  queries; assert the returned `temporal_entropy` and
  `entropy_per_head` equal the values computed by hand on the
  unpadded slice (proving the time-axis mask is applied in the
  reduction per architecture A15). The test also asserts
  `static_entropy` is identical to the value computed without the
  mask (the static branch has no time axis and must skip the mask;
  this pins the boundary).
- `tests/unit/models/test_backbone_base.py`:
  `test_base_backbone_compute_training_metrics_returns_empty`:
  a minimal `BaseBackbone` subclass that does not override
  `compute_training_metrics` returns `{}` so a v3 recurrent
  backbone that overrides nothing emits no events. This pins the
  v3-friendly default behavior. The test name matches the
  reference in architecture A15 verbatim to prevent cross-doc
  drift.

**Done when**: a small integration test in
`tests/integration/test_tabular_to_backbone.py` runs a synthetic
panel through `TabularToSequence` → `TFTBackbone` and asserts the
output `BackboneOutput` shape. No training, no estimator yet.

### Phase 4 (split into 4a and 4b): Training plumbing

The Trainer wraps Lightning. The LightningModule machinery (
`_pending_prune`, BackboneOutput stashing) lands here, including
the `optuna_trial` constructor argument; Phase 8 then wires the
user-supplied `optuna.Trial` through the Trainer into that
argument. F5's `resume_path` (model weights, optimizer state,
scheduler state, RNG state restore) is implemented here: the
Trainer accepts `resume_path` from its caller (the estimator)
and forwards it as `pl_trainer.fit(ckpt_path=resume_path)`. The
`RngStateCallback` round-trips RNG state per A14 / N4. The
caller-facing surface is pinned in Phase 6 to the
`BaseSequenceEstimator.__init__` constructor argument per
requirements F5; A20 item 5's open question is resolved that way.

**Goal**: the Lightning side. Every callback, the LightningModule,
the Trainer wrapper. No estimator shell; tests use the
`make_test_module` fixture.

**Modules** (Phase 4a):

- `src/seq_sklearn/training/_determinism.py`: `enable_strict_mode()`
  per N4. Idempotent.
- `src/seq_sklearn/training/_precision.py`: `resolve_precision(tier,
  requested)` per A11.
- `src/seq_sklearn/training/losses.py`: `build_loss()` dispatch per
  A8 / F5.
- `src/seq_sklearn/training/optimizers.py`: `build_optimizer()` per
  F5.
- `src/seq_sklearn/training/schedulers.py`: `build_scheduler()` per
  F5. `constant + warmup_steps` raises `ConfigError`. OneCycleLR
  total-steps derivation per A20 item 1 pinned here.
- `src/seq_sklearn/training/sampling.py`: oversample / undersample
  per F5.
- `src/seq_sklearn/training/callbacks.py`: `NaNLossGuard`,
  `GradScalerWatchdog`, `EventEmitter`, `RngStateCallback` per A7.
- `tests/_test_models/_dummy_modules.py` (test-only, NOT under
  `src/`): `_DummyBackbone`, `_DummyHead`, `_LossReturningScalar`.
  These are the small fixtures referenced in `make_test_module`
  per A14. The full `_DummySequenceClassifier` / `_DummySequenceRegressor`
  composition lands in Phase 6a; this phase ships the building
  blocks so the LightningModule unit tests in 4b can construct an
  isolated module without standing up an estimator.

**Modules** (Phase 4b; depends on 4a):

- `src/seq_sklearn/training/_lightning_module.py`: `_LightningModule`
  per A7. `_pending_prune` deferred-raise pattern. The
  `BackboneOutput`-stash in `training_step` per A15.
  `on_train_epoch_end` calls
  `self.backbone.compute_training_metrics(self._last_train_output)`
  to obtain a `{event_name: payload}` dict, then emits one event
  per entry. The module reads only the base `representation` /
  `padding_mask` fields directly; family-specific introspection
  attributes are NOT accessed here. This is the v1 → v3
  cross-family abstraction (architecture A15); v3 recurrent
  backbones override `compute_training_metrics` to emit
  `train.hidden_norm` without touching the LightningModule.
- `src/seq_sklearn/training/trainer.py`: `Trainer` wrapper per A7.

**Dependencies**: Phases 1, 2, 3.

**Deliverable tests** (Phase 4a):

- `tests/unit/training/conftest.py`: declares the
  `strict_mode_globals` autouse fixture per A14, scoped explicitly
  to `test_determinism.py` via a marker check inside the fixture
  body (`if "test_determinism" in request.node.fspath.basename: ...`)
  so the snapshot / restore overhead does not apply to the other
  training-unit tests in the same directory. The fixture
  snapshots `torch.are_deterministic_algorithms_enabled()`,
  `torch.backends.cudnn.deterministic`,
  `torch.backends.cudnn.benchmark`, and
  `os.environ.get("CUBLAS_WORKSPACE_CONFIG")` at setup and
  restores at teardown. Phase 6 declares a parallel fixture in
  `tests/integration/conftest.py` so the same-process determinism
  integration test is isolated under `pytest-randomly` (see Phase
  6 below).
- `tests/unit/training/test_determinism.py`: both N1 scenarios
  (env var unset, env var pre-set to a non-default); idempotency
  on a second call.
- `tests/unit/training/test_precision_resolution.py`
  (`test_hardware_detect_and_resolve_precision_combined`):
  parametrized over the six tiers; for each tier, mock
  `torch.cuda.is_available` and
  `torch.cuda.get_device_capability`, call `detect()`, then call
  `resolve_precision(tier, "auto")`, and assert both return
  values in a single parametrized run per A11. Phase 1's
  `test_hardware_detect.py` covers detection in isolation; this
  test covers the combined contract. Each phase owns its own test
  file (no cross-phase touch-backs).
- `tests/unit/training/test_losses.py`: `build_loss` dispatches to
  the right concrete class per the F5 loss-class table; illegal
  combos raise `ConfigError`. Hypothesis property test over
  `(N, num_classes)` logit tensors asserts the loss output is
  scalar and finite for each legal `(task_type, loss_strategy)`
  cell.
- `tests/unit/training/test_optimizers.py`,
  `tests/unit/training/test_schedulers.py`: legal combos build the
  right concrete instance; `scheduler="constant"` with
  `warmup_steps > 0` raises `ConfigError`.
- `tests/unit/training/test_sampling.py`: oversample produces a
  longer index array than the input; undersample produces a
  shorter one; ratios respected.
- `tests/unit/training/test_callbacks.py`: each callback exercised
  in isolation against a `pl.LightningModule` constructed via
  `make_test_module`:
  - `NaNLossGuard`: 3 consecutive NaN losses raise `TrainingError`
    with `batch_idx` in the log payload (N1 Variant A: monkey-patch
    the loss module). N1 Variant B (inject `Inf` into model
    weights at step 0) is deferred to Phase 7 because A14 pins it
    to a `tft_classifier_fresh` fixture that requires the real
    TFTClassifier. Phase 4's coverage gate counts Variant A only;
    Phase 7 closes Variant B.
  - `GradScalerWatchdog`: `hasattr(precision_plugin, 'scaler')` is
    False on CPU and the callback is a no-op; injected fake plugin
    with a decreasing `scaler.get_scale()` raises after 3 decreases.
  - `EventEmitter`: emits records with `extra={"event", "payload"}`
    that `caplog` captures.
  - `RngStateCallback`: snapshot under `on_save_checkpoint` survives
    the round trip under `on_load_checkpoint`. The N4 RNG-state
    restore test (load a checkpoint, assert bit-exact RNG state
    restoration) lives here.
**Deliverable tests** (Phase 4b; depends on 4a):

- `tests/unit/training/test_lightning_module.py`: the
  `make_test_module` fixture builds a module with the
  `_DummyBackbone` (extending `BaseBackbone` from Phase 3, with
  its default empty `compute_training_metrics`) / `_DummyHead` /
  `_LossReturningScalar` fixtures landed in 4a. Three named tests
  cover the LightningModule's hook surface:
  `test_on_train_epoch_end_skips_entropy_when_no_output` asserts
  the None-guard branch (`_last_train_output is None`);
  `test_on_train_epoch_end_emits_events_from_compute_metrics`
  uses a `_DummyBackbone` subclass that overrides
  `compute_training_metrics` to return a single synthetic payload
  `{"train.var_selection_entropy": {"static_entropy": 1.0,
  "temporal_entropy": 0.5}}` and asserts `caplog` contains exactly
  one record with `record.event == "train.var_selection_entropy"`
  and the expected payload keys (pins the delegation loop
  `for event_name, payload in payloads.items(): emit(...)` at the
  unit level, independent of Phase 9's broad parametrized sweep);
  `test_on_train_epoch_end_deferred_prune_raises_after_logging`
  asserts `optuna.TrialPruned` raises from the deferred-raise path
  at the END of the hook (after the emit loop has fired). `on_validation_epoch_end` records the
  prune decision; `on_train_epoch_end` raises
  `optuna.TrialPruned` from the deferred-raise path at the end of
  the hook (after `train.epoch` and entropy logging has already
  fired), confirming the `_pending_prune` pattern. The trial in
  these tests is `optuna.trial.FixedTrial` with
  `should_prune` monkey-patched to return True (per the architecture
  research brief, `FixedTrial` exposes the same surface as
  `Trial`); no real `Study` is instantiated.
- `tests/unit/training/test_trainer.py`: the Trainer constructs a
  `pl.Trainer` with the right callbacks attached and the right
  precision; the `deterministic=True` gate flips on
  `precision="32-true"` and a set seed. The `resume_path` path
  threads through to `pl_trainer.fit(ckpt_path=...)` and the
  `RngStateCallback` restore fires on resume (asserted by checking
  Python / numpy / torch RNG state matches the saved snapshot
  byte-exact).

**Done when**: an integration test in
`tests/integration/test_training_smoke.py` runs the Trainer on a
`_DummyBackbone` + `_DummyHead` + `_LossReturningScalar` against a
synthetic panel for 1 epoch. No real estimator yet.

### Phase 5: Calibration

**Goal**: every calibrator, the threshold tuner. The BETA
`AttentionOutput` / `RegressionAttentionOutput` dataclasses move
to Phase 6 (they are consumed by `predict_with_attention` on the
estimator base, not by calibration).

**Modules**:

- `src/seq_sklearn/calibration/_protocol.py`: `_Calibrator` Protocol
  per A9.
- `src/seq_sklearn/calibration/classification.py`:
  `TemperatureScaling`, `PlattScaling`, `IsotonicCalibrator`.
- `src/seq_sklearn/calibration/regression.py`: `ConformalCalibrator`,
  `IsotonicQuantileCalibrator`. The non-monotone TrainingError path
  is exercised here.
- `src/seq_sklearn/calibration/threshold.py`: `ThresholdTuner`.

**Dependencies**: Phases 1, 3 (for serialization helpers).

**Deliverable tests**:

- One test file per calibrator, each constructed with hand-crafted
  `(logits, y_true)` tensors so no estimator is required.
- `ConformalCalibrator` non-monotone test (N1): a deliberately
  non-monotone tensor of shape `(N, len(quantiles))` raises
  `TrainingError` matching `r"non-monotone"`.
- The calibrator's `serialize` / `deserialize` round-trip is
  byte-identical AFTER passing through `json.dumps` and
  `json.loads` (the JSON serialization is the actual save / load
  path). The test asserts on the JSON-roundtripped state, not on
  the raw dict, so float-precision loss in JSON encoding surfaces.
- `ThresholdTuner` produces a float threshold in [0, 1] on a
  synthetic logits / labels pair.
- Hypothesis property test over `(N, num_classes)` logit tensors:
  for each calibration strategy, the calibrated output shape
  equals the input shape and values stay finite.

**Done when**: every calibrator passes its standalone test; the
JSON round-trip on each calibrator's state is byte-equal.

### Phase 6 (split into 6a and 6b): Estimator base shells + recurrent skeleton + smoke skeleton

**Goal**: `BaseSequenceEstimator`, `BaseSequenceClassifier`,
`BaseSequenceRegressor`, `TransformerSequenceEstimator`, and the
INTERNAL `RecurrentSequenceEstimator` abstract base. A synthetic
concrete estimator (test-only) exercises the full shell end-to-end
before TFT specifics land. The `resume_path` surface decision (A20
item 5) is pinned here per requirements F5:
`BaseSequenceEstimator.__init__(..., resume_path: str | Path | None
= None, ...)`. The estimator stores `resume_path` as an instance
attribute and passes it through to the Trainer at `fit` time; the
Trainer's `ckpt_path=` forwarding from Phase 4 receives the path.

The phase ships as two sub-PRs to stay within the ~600 lines-of-
`src/` per-PR guideline. 6b depends on 6a (the family bases inherit
from `BaseSequenceEstimator` and the recurrent skeleton compose
test imports it).

**Modules** (Phase 6a):

- `src/seq_sklearn/models/_base.py`: `BaseSequenceEstimator`.
  Implements the fit / predict shell, plumbs `TabularToSequence`,
  builds the LightningModule via the curried-factory pattern per
  A7, owns the save / load round trip, threads the seed.
  `__init__(..., resume_path: str | Path | None = None, ...)` per
  requirements F5; the resume path stays an instance attribute and
  forwards to the Trainer at fit time as
  `pl_trainer.fit(ckpt_path=resume_path)`.
  `fit(X, y, *, calibration_set=None, optuna_trial=None)` accepts
  the trial as a keyword (NOT a pydantic config field, NOT an
  `__init__` argument) so `extra="forbid"` stays intact and the
  trial cannot enter `get_params` or `save` / `load`. The F2
  `calibration_set + cal_fraction` conflict check fires here at
  `fit` time (in addition to the split-function check exercised
  in Phase 2). `__sklearn_tags__` is implemented here as a base
  instance method so every concrete model inherits the family-wide
  tag block (`input_tags.dataframe = True`, `requires_fit = True`,
  etc.) per F1.1; concrete classes override only when a tag must
  flip (e.g. v3 recurrent models flip `non_deterministic = True`).
  `_build_config` injects the task-type-aware loss default per
  architecture A4: when `self.loss is None`, build a
  `LossParams(strategy=_DEFAULT_LOSS_FOR_TASK[self.task_type])`
  so `TFTClassifier(task_type="binary").fit(X, y)` works without
  explicit loss specification while keeping `LossConfig.strategy`
  no-default at the pydantic layer. The `BackboneOutput` dataclass
  base and the abstract `BaseBackbone` class with the default
  `compute_training_metrics() -> {}` already landed in Phase 3 at
  `models/_backbone.py`; `BaseSequenceEstimator` imports
  `BaseBackbone` for type annotations only.
- `src/seq_sklearn/models/_classifier.py`: `BaseSequenceClassifier`
  overlays the `predict_proba`, `predict`, classifier-specific
  fit-state attributes (`classes_`).
- `src/seq_sklearn/models/_regressor.py`: `BaseSequenceRegressor`
  overlays the `predict`, `predict_quantiles`, regressor-specific
  fit-state attributes (`quantiles_`). The three F1 / N1
  `predict_quantiles` error paths (point-mode `PredictionError`,
  pre-fit `NotFittedError`, off-vector `ValueError`) raise from
  this class.
- `tests/_test_models/_dummy_estimator.py` (test-only, NOT under
  src): `_DummySequenceClassifier`, `_DummySequenceRegressor`
  composing a trivial 1-layer linear backbone, a real
  `ClassificationHead` / `RegressionHead`, the real loss / optimizer
  / scheduler, and the real Trainer. The dummy backbone extends
  the transformer family and returns a `TransformerBackboneOutput`
  with non-trivial (small random init) `var_selection_weights`,
  `attention_weights`, and `static_var_selection_weights` so the
  entropy-emission and attention-extraction code paths produce
  non-degenerate values through the smoke estimator (zero-filled
  tensors would make the entropy reduction silently degenerate).
  It overrides `compute_training_metrics` to emit one synthetic
  payload so the `_LightningModule` delegation loop exercises a
  non-empty dict (the test path added in Phase 4b below). The
  dummy estimator stays in `tests/_test_models/` after Phase 7 as
  a fast fixture for LightningModule and estimator-base unit tests.

**Modules** (Phase 6b; depends on 6a):

- `src/seq_sklearn/models/transformer/_base.py`:
  `TransformerSequenceEstimator` family base with the nested
  `Classifier` / `Regressor` mixin classes per A2. The
  `TransformerBackboneOutput` dataclass landed in Phase 3 at
  `models/transformer/_backbone.py` (it had to, since `TFTBackbone`
  returns it); this family base imports it for the mixin classes'
  type annotations.
- `src/seq_sklearn/models/recurrent/_base.py`: abstract
  `RecurrentSequenceEstimator` per A6.1. INTERNAL in v1. Landing
  the abstract base here (alongside the 6a shells) locks the
  v1 → v3 cross-family contract before TFT specifics land in
  Phase 7, per future-proofing item 7.
- `src/seq_sklearn/config/recurrent.py`:
  `RecurrentSequenceEstimatorConfig` skeleton per A6.1. INTERNAL.
- `src/seq_sklearn/inference/attention.py`: `AttentionOutput`,
  `RegressionAttentionOutput` per A15.1. Moved here from Phase 5
  because `predict_with_attention` is implemented on the
  `BaseSequenceClassifier` / `BaseSequenceRegressor`, not on any
  calibration module.

**Dependencies**: Phases 1-5.

**Deliverable tests** (Phase 6a):

- `tests/unit/models/test_base_estimator.py`: `fit` builds
  `self.config_` (frozen pydantic) from the mutable attributes;
  `get_params(deep=True)` produces the canonical
  `tabular_config__lookback` flat keys;
  `set_params(tabular_config__lookback=6)` chains via standard
  sklearn double-underscore;
  `clone(estimator)` produces an independent instance whose
  mutation does not aliase the original. The F1.1 punted-method
  contract is asserted here:
  `with pytest.raises(NotImplementedError, match="partial_fit is
  not supported in seq-sklearn v1")` for each of `partial_fit`,
  `fit_predict`, `fit_transform`, so a future message change
  surfaces as a test failure.
  `test_sklearn_tags_base_values` calls
  `_DummySequenceClassifier().__sklearn_tags__()` and asserts
  field-level values: `tags.input_tags.dataframe is True`,
  `tags.input_tags.two_d_array is False`,
  `tags.input_tags.allow_nan is False`,
  `tags.target_tags.required is True`, `tags.requires_fit is True`,
  `tags.non_deterministic is False`. A tag-value regression
  surfaces immediately at the unit level, not only when
  `check_estimator` happens to probe the affected field.
- `tests/unit/models/test_loss_default_injection.py` (per
  hyperparameter-strategy fold-in; Phase 6a since it ships with
  `BaseSequenceEstimator._build_config`):
  `test_loss_default_injection_per_task_type` parametrizes over v1
  task types (`binary`, `multiclass`, `regression_point`,
  `regression_quantile`); for each, construct
  `_DummySequenceClassifier(task_type=task)` without `loss=`,
  call `_build_config`, assert the injected
  `LossConfig.strategy == _DEFAULT_LOSS_FOR_TASK[task]`.
- `tests/unit/config/test_adapters.py` (extends Phase 1's
  clone-safety roster):
  `test_outer_estimator_clone_does_not_alias_adapter_instances`
  per hyperparameter-strategy fold-in. Constructs a
  `_DummySequenceClassifier` with explicit adapter instances,
  calls `sklearn.base.clone`, asserts each cloned adapter is a
  fresh instance (not aliased to the original).
- `tests/unit/models/test_short_entity_predict.py` (one of N1's
  required tests): predict on a panel with three entities below
  `min_periods_predict` plus one above-floor entity; assert each
  below-floor row has output shape `(num_classes,)` (or `(1,)` for
  binary, `(quantiles,)` for quantile mode) filled with NaN (not
  scalar, not zero-filled); assert `caplog` contains exactly one
  `data.duplicate_floor_breach_count` record regardless of how many
  entities fell below the floor.
- `tests/unit/models/test_regressor_error_paths.py` (three N1
  required tests at the base-class level; the file lives under
  `tests/unit/models/` rather than `tests/unit/models/transformer/tft/`
  because the error paths originate on `BaseSequenceRegressor`,
  not on TFT specifically. The concrete instance under test is
  the `_DummySequenceRegressor` from the smoke skeleton):
  - `test_predict_quantiles_on_point_mode_raises`: call
    `predict_quantiles()` on a point-mode regressor; assert
    `PredictionError` naming `predict`.
  - `test_predict_quantiles_before_fit_raises`: call
    `predict_quantiles()` before `fit`; assert `NotFittedError`
    (catchable as both `seq_sklearn.errors.NotFittedError` and
    `sklearn.exceptions.NotFittedError`).
  - `test_predict_quantiles_off_vector_raises`: call
    `predict_quantiles(quantiles=[0.99])` when fit-time vector is
    `[0.1, 0.5, 0.9]`; assert `ValueError` whose message lists the
    fit-time vector.
- `tests/integration/test_dummy_estimator_e2e.py`: fit the
  `_DummySequenceClassifier` on a synthetic binary panel for 2
  epochs; `predict_proba` returns valid probabilities;
  `save(tmp_path)` writes the two-file layout; reload in the SAME
  process and re-predict; predictions byte-equal. A second variant
  in the same file (`test_save_load_with_temperature_calibration`)
  fits with `calibration_strategy="temperature"`, saves, reloads
  same-process, and asserts byte-equal `predict_proba` so the
  calibrator's `serialize` / `deserialize` is exercised through
  the integrated save / load path (N1 requirement).
- `tests/integration/conftest.py` (Phase 6 owns the integration
  test directory's conftest): declares
  `integration_strict_mode_globals`, a function-scoped autouse
  fixture that snapshots the four determinism flags
  (`torch.are_deterministic_algorithms_enabled()`,
  `torch.backends.cudnn.deterministic`,
  `torch.backends.cudnn.benchmark`,
  `os.environ.get("CUBLAS_WORKSPACE_CONFIG")`) at setup and
  restores at teardown. Required because `pytest-randomly` can
  order a `tests/unit/` test that disables determinism (e.g. a
  hypothesis test) before the integration-determinism test; without
  the fixture the integration test inherits contaminated global
  state and the bit-identical assertion passes or fails by
  accident. The fixture body checks `request.node.get_closest_marker
  ("determinism")` and restores only on tests carrying that mark
  (the determinism integration test marks itself
  `@pytest.mark.determinism` per N1's mark taxonomy).
- `tests/integration/test_dummy_estimator_same_process_determinism.py`
  (one of N1's required tests; marked `@pytest.mark.determinism`
  so the integration-conftest fixture above kicks in): fit
  `_DummySequenceClassifier(seed=42)` twice in the same process
  with no intervening global-state changes; assert `torch.equal`
  on both `predict_proba` outputs. Exercises the in-process
  determinism contract independent of the subprocess-load path.
- `tests/integration/test_dummy_estimator_subprocess_load.py`:
  the same fit + save, but the reload happens in a fresh
  subprocess. (One of the N1 required tests; pinned here.) An
  additional variant
  (`test_load_version_mismatch_warning_via_estimator`) mutates
  `seq_sklearn_version` in the saved `state.json` to a fake older
  value and reloads through the estimator's `load` classmethod
  (not just the low-level helper in Phase 3), asserting exactly
  one `UserWarning` whose message names "version mismatch" and
  both versions.
- `tests/unit/models/test_calibration_set_fit_conflict.py`: call
  `fit(X, y, calibration_set=(X_cal, y_cal))` on a
  `_DummySequenceClassifier` with `cal_fraction=0.1`; assert
  `ConfigError` raised at the estimator-shell `fit` entry, not
  just at the inner `compute_three_way_split` call.
- `tests/integration/test_three_way_split_correctness.py`
  (`test_estimator_split_folds_disjoint_by_id_time`): one of N1's
  required tests. Fit `_DummySequenceClassifier` on a synthetic
  panel where each entity has at least 20 rows; expose the
  internal `(train_idx, val_idx, cal_idx)` triple via a test hook
  on the estimator; assert pairwise empty intersection on
  `(entity_id, time)` tuples (the N1 "disjoint by (id, time)"
  clause); assert the last `cal_fraction` rows per entity sit in
  `cal_idx`; assert the preceding `val_fraction` rows sit in
  `val_idx`. A second case
  (`test_estimator_split_collapse_when_no_calibration`) sets
  `calibration_strategy="none"` and `threshold_tuning=False` and
  asserts `cal_idx` is empty while `train_idx` includes the rows
  that would otherwise have been calibration. Phase 2 covers the
  split helper in isolation; this integration test exercises the
  same contract through `estimator.fit`, closing the N1 "Fit on a
  panel" phrasing.
- `tests/integration/test_dummy_estimator_e2e.py` second variant
  (`test_val_split_random_warning_via_estimator`): call
  `_DummySequenceClassifier.fit(X, y)` on a multi-entity panel
  with `val_split_strategy="random"`; assert exactly one
  `UserWarning` containing both "panel" and "random" propagates
  through the estimator's `fit` (not just from the split helper).
  Phase 2 covered the helper-level warning emission; this test
  closes the estimator-shell-level F10 / N1 contract.
**Deliverable tests** (Phase 6b; depends on 6a):

- `tests/unit/models/transformer/test_family_base.py`: the
  `TransformerSequenceEstimator.Classifier` mixin composes with
  `BaseSequenceClassifier` and adds the attention-extraction hook
  without breaking the base contract.
- `tests/unit/inference/test_attention_output.py`: field set of
  each dataclass matches the v1 enumeration exactly; a future
  MINOR-release field addition will break this test and force a
  deliberate snapshot bump. Tuple-unpacking via `__iter__` raises
  `TypeError`.
- `tests/unit/models/recurrent/test_skeleton_composes.py` (the v1
  contract-lock test per A6.1; landing here in 6b, before Phase 7's
  TFT, validates the v1 → v3 cross-family contract): an inline
  no-op concrete subclass that fills the three abstract methods
  composes with `BaseSequenceEstimator`'s shell and runs
  `fit` / `predict` against a tiny synthetic panel.

**Done when**: a `_DummySequenceClassifier` round-trips through fit
+ save + cross-process load + predict on a synthetic binary panel.
The infrastructure is proven before TFT-specific code lands.

### Phase 7: TFT concrete estimators

**Goal**: `TFTClassifier`, `TFTRegressor`. v1's first real model.
The recurrent skeleton already landed in Phase 6, so this phase
contains only TFT-specific code.

**Modules**:

- `src/seq_sklearn/models/transformer/tft/classifier.py`:
  `TFTClassifier(TransformerSequenceEstimator.Classifier,
  BaseSequenceClassifier)`. `__init__` follows the BaseEstimator-
  adapter pattern (A4 step 3). Inherits the family-wide
  `__sklearn_tags__` from `BaseSequenceEstimator` (Phase 6a);
  overrides only if a TFT-specific tag flip is needed (none in
  v1).
- `src/seq_sklearn/models/transformer/tft/regressor.py`:
  `TFTRegressor`. Same shape.

**Dependencies**: Phases 1-6.

**Deliverable tests**:

- `tests/integration/test_tft_classifier_e2e.py`: fit a binary
  classifier on a small synthetic panel (n_entities=20,
  periods=24, signal_strength=0.7, seed=42) for 5 epochs; assert
  predict / predict_proba / save / load round-trip; assert
  `predict_with_attention` returns an `AttentionOutput` with the
  documented fields and shapes.
- `tests/integration/test_tft_regressor_e2e.py`: same for point
  and quantile regressors; quantile mode asserts the
  `predict_quantiles` contract.
- `tests/unit/models/transformer/tft/test_classifier_init.py`:
  `TFTClassifier.__init__` mirrors every `TFTConfig` field;
  unknown kwargs raise; the adapter-pattern `tabular_config__`
  flat keys appear in `get_params(deep=True)`.
- `tests/integration/test_tft_nan_loss_variant_b.py` (the N1
  Variant B test; the `tft_classifier_fresh` fixture per A14):
  construct a fresh `TFTClassifier`, snapshot its weights via
  `.clone()`, inject `Inf` into the weights at step 0, fit; assert
  `TrainingError` raised from the natural NaN-propagation path;
  restore weights in `finally`. Variant A landed in Phase 4
  against the dummy estimator; this closes the requirement.
- `tests/integration/test_gpu_cpu_parity.py` (one of N1's required
  nightly tests, marked `@pytest.mark.gpu` and
  `@pytest.mark.slow`): fit `TFTClassifier(precision="32-true",
  seed=42)` on CPU and on GPU (skip when no CUDA); assert
  `predict_proba` agrees within `atol=1e-5, rtol=1e-5`. Picked up
  by the nightly workflow's `gpu` job per A19.

**Done when**: `TFTClassifier(seed=42).fit(X, y).predict(X)`
produces sane outputs on a synthetic panel. The N1 Variant B
NaN-loss test passes against the real TFT.

### Phase 8: Public API + Optuna integration

**Goal**: the public surface compiles, exports the documented
symbols, and Optuna integration works end-to-end.

**Modules**:

- `src/seq_sklearn/__init__.py`: full re-export list per A3.
- `src/seq_sklearn/tuning/suggest_params.py` per A16 (signature
  includes `search_advanced: bool = False` and `search_extras: bool
  = False` keyword-only flags per hyperparameter-strategy fold-in).
  Default flags sample ONLY STABLE fields enumerated in the strategy
  doc's "Default search space per model" table; `search_advanced=True`
  also samples BETA fields on `<Model>AdvancedConfig` (v1 ships
  empty so the flag is a no-op for v1); `search_extras=True` samples
  from `src/seq_sklearn/tuning/_alpha_keys.py` (empty in v1).
  Closed under the F5 validity matrix regardless of flag values.
- `src/seq_sklearn/tuning/_alpha_keys.py` (new per architecture A4):
  curated per-family ALPHA-key enum lists. v1 ships empty;
  maintainers populate as ALPHA passthroughs land.
- `src/seq_sklearn/tuning/_estimator_bridge.py` (new per
  architecture A16): `config_to_estimator_kwargs` helper + the
  `_ADAPTER_MAP_BY_CONFIG` registry + `_TFT_ADAPTER_MAP` per the
  A16 spec. Pops every nested sub-config from
  `config.model_dump(mode="json")` and wraps in the matching
  adapter; v2 / v3 register their per-model adapter maps here.
- `src/seq_sklearn/tuning/pruning.py`: `optuna_trial_guard` context
  manager per A16. The LightningModule's `optuna_trial` constructor
  argument and the `_pending_prune` deferred-raise pattern already
  landed in Phase 4. This phase wires `BaseSequenceEstimator.fit`'s
  `optuna_trial` keyword through to the Trainer and into the
  LightningModule's constructor. The library does NOT ship
  `optuna-integration.PyTorchLightningPruningCallback`: that
  callback raises `TrialPruned` from `on_validation_end`, before
  Lightning fires `on_train_epoch_end`, which would skip the
  `train.epoch` and entropy events on the pruned epoch (the exact
  lifecycle bug A7's deferred-raise pattern was introduced to
  prevent). The native `_pending_prune` path is the only supported
  pruning hook.

**Dependencies**: Phases 1-7.

**Deliverable tests**:

- `tests/unit/tuning/test_suggest_params.py`: a 1000-iteration
  sweep using a real `optuna.Study` via `study.ask()`
  (NOT `FixedTrial`, which is deterministic and would not exercise
  the search space). Each sampled trial drives `suggest_params`;
  every returned config passes `check_combo` from the validity
  matrix. One of N1's required tests. The architecture A16 code
  block is the canonical pattern. Includes
  `test_suggest_params_default_flags_exclude_advanced_fields`,
  `test_suggest_params_search_advanced_true_accepts_flag`, and
  `test_suggest_params_sweeps_only_default_fields` per the
  hyperparameter-strategy fold-in (pins the per-flag sampling
  behavior).
  `FixedTrial` remains the right tool for the
  `optuna_trial_guard` tests below, where the wrapped objective
  body needs `report` / `should_prune` to be no-ops without
  standing up a `Study`.
- `tests/unit/tuning/test_estimator_bridge.py` (new per
  hyperparameter-strategy fold-in):
  `test_config_to_estimator_kwargs_round_trips_all_adapters`
  exercises every adapter slot;
  `test_config_to_estimator_kwargs_extra_tuple_type_survives`
  pins the `mode="json"` round-trip of `extra` tuples through
  adapter construction. Both load-bearing for the Optuna helper.
- `tests/unit/tuning/test_pruning.py`: a `ConfigError` raised
  inside the guard converts to `optuna.TrialPruned`; a
  `TrainingError` does the same; `DataContractError` and
  `KeyboardInterrupt` propagate.
- `tests/integration/test_optuna_pruning_e2e.py`: the N1 pruning
  test with `MedianPruner(n_startup_trials=0, n_warmup_steps=0,
  n_min_trials=1)`. Two trials, three epochs. Assert trial 2 prunes
  at epoch 0. A second variant asserts metric routing
  (`trial.intermediate_values[epoch]` matches the logged
  validation metric per N1).
- `tests/unit/test_public_api_surface.py`: every symbol in the A3
  `__all__` is importable; no extra public symbols leak.
- `tests/unit/tuning/test_fit_optuna_trial_kwarg_routing.py`: call
  `estimator.fit(X, y, optuna_trial=FixedTrial({}))` on the
  `_DummySequenceClassifier`; assert `"optuna_trial"` is absent
  from `estimator.get_params(deep=True)`; assert the trial reached
  the LightningModule by inspecting an exposed test hook on the
  estimator. Pins the F1 contract that the trial is a `fit` keyword
  only, never a config field. A future refactor that accidentally
  puts `optuna_trial` back into `__init__` or into the pydantic
  config fails this test.

**Done when**: the documented Optuna example
(`docs/examples/optuna_search.py`) runs end-to-end on a synthetic
panel and produces a pruned trial.

### Phase 9: check_estimator contract + acceptance thresholds

**Goal**: every required N1 test is implemented and passes.
Coverage gate verified.

**Modules**: tests only. `tests/conftest.py` is extended.

- `tests/conftest.py`: `EXPECTED_FAILED_CHECKS`,
  `EXPECTED_PASSING_CHECKS` (separated per the Gemini-pass fix),
  `propagate_seq_sklearn_logger` autouse, hypothesis profile
  registration (`inner_loop`, `nightly`).
- `tests/unit/test_check_estimator.py`: `parametrize_with_checks`
  over `(TFTClassifier(), TFTRegressor())` with
  `expected_failed_checks=EXPECTED_FAILED_CHECKS`. The meta-test
  asserts each `EXPECTED_PASSING_CHECKS` entry is in the collected
  check IDs.
- `tests/e2e/test_acceptance_thresholds.py`: the three-seed median
  thresholds from N1. Marked `@pytest.mark.slow`; runs nightly.
- `tests/e2e/test_calibration_coverage_per_strategy.py`: one test
  per `calibration_strategy` value asserting the band from N1.
  Marked `@pytest.mark.slow`; runs nightly.
- `tests/e2e/test_imbalance_smoke.py`: one test per
  `imbalance_strategy` per N1. Marked `@pytest.mark.slow`; runs
  nightly. The 5-minute PR-CI budget (N2) excludes these via
  `-m "not slow"`; `tests/e2e/test_quickstart.py` is the only e2e
  test that runs on every PR and is intentionally NOT marked
  `slow`. The implementer MUST NOT add `@pytest.mark.slow` to the
  quickstart for "consistency" because doing so would silently
  drop the N1 quickstart-in-CI requirement.
- `tests/snapshot/test_tft_snapshot.py`: one snapshot per task
  type. Pin DGP version, seed, config. Stored in
  `tests/_snapshots/`. The `--snapshot-update` flag refreshes them;
  the pre-commit hook enforces the `SNAPSHOT_REVIEWED:` marker.
  Marked `@pytest.mark.slow` (a TFT forward pass at default config
  on a synthetic panel exceeds the 2s `slow` threshold per N1);
  runs nightly. The per-PR CI budget at N2 stays under 5 minutes
  because the snapshot test is excluded by `-m "not slow"`.
- `tests/e2e/test_quickstart.py`: imports the README quickstart
  file and asserts the N1 binary classifier accuracy threshold
  (>= 0.75) per A14.
- `tests/unit/test_fit_state_attributes.py`: the F1.1 attribute
  contract is satisfied post-fit on both classifier and regressor;
  the `decision_threshold_` presence / absence parametrization per
  A2 is included.
- `tests/unit/test_structured_log_events.py`: parametrized over the
  F11 event table; one assertion per event; `train.hidden_norm` is
  xfailed (strict) per the recurrent-only marker.

**Dependencies**: Phases 1-8.

**Deliverable tests**: see modules above.

**Done when**: `pytest --cov=seq_sklearn --cov-branch` reports >=85%
line and >=80% branch coverage with no failing tests under the
inner-loop profile. The phase-by-phase N1 test assignment in this
plan IS the checklist; the Phase 9 PR description references this
plan's phase entries to confirm every item is implemented.

### Phase 10: ONNX export + deploy tests

**Goal**: `export_onnx` works, the wheel installs cleanly, the
deploy smoke test passes.

**Modules**:

- Add `export_onnx` to `BaseSequenceEstimator` per F1. Routes
  attention through the math backend via
  `torch.nn.attention.sdpa_kernel(SDPBackend.MATH)` per N5.
- `tests/integration/test_onnx_parity.py`: export at opset 20 with
  `dynamo=True`; load in onnxruntime; predict on a fixed batch
  with masked variable-length entities; assert agreement within
  `atol=1e-4, rtol=1e-4` per N1.
- `tests/deploy/test_wheel_install.py`: build the wheel, install
  in a clean venv (via `tox` or `nox`), import, fit / predict on
  a tiny panel.
- `tests/deploy/test_restricted_op_surface.py`: static-analysis
  check that the backbone uses only the documented ONNX-safe
  PyTorch op surface (N1 item).

**Dependencies**: Phases 1-9.

**Deliverable tests**: see above.

**Done when**: `pytest tests/deploy/` passes against a freshly
built wheel; ONNX parity holds within tolerance.

### Phase 11: Performance baselines

**Goal**: the perf-benchmark cells from A13 have checked-in
baselines.

**Modules**:

- `tests/perf/test_train_step_time.py`,
  `tests/perf/test_peak_memory.py`,
  `tests/perf/test_inference_latency.py` per N7.
- `tests/perf/_baselines/cpu-x86.json`,
  `tests/perf/_baselines/t4.json` per A13.
- `tests/perf/conftest.py`: pytest-benchmark wiring; the regression
  gate (15% on median step time, 10% on peak memory).

**Dependencies**: Phases 1-10.

**Deliverable tests**: nightly only. The PR-CI perf job warns; it
does not block.

**Done when**: the nightly job produces a baseline JSON for the
two public cells and the regression gate fires correctly on a
contrived 20% slowdown.

### Phase 12: Documentation and release prep

**Goal**: docs site builds, examples run, release artifacts ready.

**Modules**:

- `mkdocs.yml` with the A12 plugin recipe (`griffe-pydantic`,
  `mkdocs-material`, `mkdocs-gen-files`, `mkdocs-literate-nav`,
  `mkdocs-section-index`).
- `docs/index.md` (rewritten from README quickstart).
- `docs/observability.md` (F11 event-payload reference).
- `docs/guides/`: getting_started, panel_data, calibration,
  optuna_search, hardware_and_precision per A12.
- `docs/examples/optuna_search.py`,
  `docs/examples/mlflow_search.py` (F7 and F11 references).
- `docs/api/`: auto-generated via mkdocstrings; the literate-nav
  index drives the table of contents.
- `README.md` rewritten with the one-screen quickstart that
  `tests/e2e/test_quickstart.py` imports.
- `CHANGELOG.md` updated with the v1.0.0 entry.
- The `mkdocs build --strict` job in the PR workflow turns from a
  skeleton into a real gate.

**Dependencies**: Phases 1-11.

**Deliverable tests**: `mkdocs build --strict` exits 0; the
docstring-examples test (`pytest --doctest-modules src/seq_sklearn/`
per N1) passes; the quickstart in CI passes.

**Done when**: the docs site renders, the API reference shows the
pydantic field tables via griffe-pydantic, and the release
checklist (acceptance criteria 1-11 in requirements) is complete.

## Future-proofing for v2 / v3

The phases above lock contracts that v2 and v3 inherit unchanged:

1. **`BaseSequenceEstimator` template-method shape** (Phase 6).
   Every future model overrides `_build_module`, `_loss_function`,
   `_head`. No change to the base class shell.
2. **`TabularToSequence.transform` output dict** (Phase 2). Every
   future model consumes the same dict. PatchTST patches the
   `time_varying_real` tensor before its attention block; TimesNet
   reshapes the same tensor period-wise; TST consumes it directly;
   v3 recurrent models pass it through `pack_padded_sequence`.
3. **`BackboneOutput` dataclass base + `compute_training_metrics`
   delegation** (Phase 3). The base dataclass carries only the
   family-agnostic fields (`representation`, `padding_mask`);
   `BaseBackbone.compute_training_metrics(output)` returns a
   `{event_name: payload}` dict that the LightningModule emits
   without reading family-specific attributes. Non-TFT transformers
   in v2 extend `TransformerBackboneOutput` (same four
   introspection fields) and either inherit TFT's metric
   implementation or override; PatchTST without VSN sets
   `var_selection_weights` to a one-hot constant. Recurrent models
   in v3 extend the base with `RecurrentBackboneOutput` (carrying
   `hidden_states`) and override `compute_training_metrics` to
   emit `train.hidden_norm`. The LightningModule code never
   changes.
4. **Validity matrix data-driven** (Phase 1). Adding a new
   `task_type` (e.g. v1.1 `multilabel`) is a one-line addition to
   `_domains.py` plus matrix rows in `_validity.py`. No code change
   in `BaseModelConfig`, `suggest_params`, or any concrete model.
5. **Save / load schema versioning** (Phase 3). `MIGRATIONS` is
   empty in v1. v1.1 multi-output adds the first migration; v2's
   new model classes add their concrete-model-specific state but
   inherit the schema version. The `_migrate` strict-monotone
   invariant catches drift.
6. **Layer factory single chokepoint** (Phase 3). v2's FP8 pass
   swaps `nn.Linear` and `nn.LayerNorm` for Transformer Engine
   equivalents inside `models/_layers.py` only. Every call site
   inherits.
7. **Recurrent skeleton compose test** (Phase 7). A no-op concrete
   subclass exercises `BaseSequenceEstimator.fit` / `predict` via
   the recurrent abstract base. v3's first concrete LSTM ports
   the abstract methods to real bodies; no shell change.
8. **`HardwareTier` enumerates all six tiers in v1** (Phase 1). v1
   branches only on the first four (CPU, Pascal, Volta/Turing,
   Ampere/Ada), but Hopper and Blackwell exist in the enum so v2's
   FP8 path is a one-line dispatch addition in `_precision.py`.
9. **Family-base nested classes** (Phase 6). `TransformerSequenceEstimator`
   ships in v1. v2 adds `PatchTSTBackbone`, `TimesNetBackbone`,
   `TSTBackbone` under `models/transformer/`. Each composes with the
   same nested-mixin pattern.

## Risk register

- **R1: pydantic adapter pattern is novel** (Phase 1). The
  BaseEstimator adapter for nested pydantic configs has no
  surveyed precedent per `docs/research/pydantic_sklearn.md`.
  Mitigation: Phase 1 includes a focused
  `test_adapters.py` covering `clone`, `get_params(deep=True)`,
  `set_params` chaining, joblib threading, and
  pickle. If the pattern proves unworkable, the fallback is the
  flatten-into-dict pattern from the Gemini-rejected A4 v1, taking
  the rework hit before any concrete estimator is written.
- **R2: ONNX op surface (Phase 10) - RESOLVED by design.** The TFT
  backbone's predict/export forward uses the interpretable
  manual-softmax attention, NOT `F.scaled_dot_product_attention`, so
  the SDPA-backend export hazard does not apply to the exported
  graph; `sdpa_kernel(SDPBackend.MATH)` is retained around the
  export only as a no-op defense-in-depth. The real guard is the
  architecture A21 restricted-op-surface allowlist (enumerated by
  static analysis, not back-derived) plus the deploy test
  `tests/deploy/test_restricted_op_surface.py`, which fails at first
  pin on any op outside the surface (e.g. a future
  `ScaledDotProductAttention`/`Loop`). A20 item 3 tracks the native
  ONNX `Attention` op (opset 23, PyTorch issue #149662) as a future
  optimization, not a correctness dependency.
- **R3: Lightning 2.6 yanked-version policy churn** (Phase 0).
  `lightning>=2.6.1,<2.7` skips the 2.6.2 / 2.6.3 PyPI compromise.
  If 2.6.4 reverses behavior, the pin holds; a CHANGELOG entry
  documents the upper bound bump.
- **R4: BLAS reordering on B>1 batches** (Phase 3). The
  mask-correctness test runs at B=1 by default; B>1 introduces
  reduction-order drift that `torch.equal` does not tolerate.
  Mitigation: the test documents the B=1 contract; B>1
  mask-correctness moves to `atol=1e-5` if a future regression
  forces it (deferred per the Round 1 design-review nitpick).
- **R5: `_last_train_output` under grad accumulation** (Phase 4).
  `accumulate_grad_batches > 1` makes the stashed output the final
  micro-batch only, not an aggregate. Mitigation: documented in
  `docs/observability.md`; true per-epoch aggregation is a v2
  refinement.
- **R6: snapshot churn** (Phase 9). A PyTorch / CUDA point release
  might shift accumulation order and break snapshot byte-equality.
  Mitigation: snapshots run only on the CPU FP32 deterministic
  path; the `SNAPSHOT_REVIEWED:` marker forces human review on
  every refresh.
- **R7: A user reintroducing
  `optuna-integration.PyTorchLightningPruningCallback` in their
  own objective body** (Phase 8). The library no longer ships the
  callback (Gemini r1-C4: the callback raises `TrialPruned` from
  `on_validation_end`, which Lightning fires before
  `on_train_epoch_end`, skipping the `train.epoch` and entropy
  events on the pruned epoch). Risk: a user copying a pre-2026
  Optuna + Lightning tutorial would add the callback to their
  Trainer and silently drop the events. Mitigation: the library's
  Optuna example (`docs/examples/optuna_search.py`) demonstrates
  the native `_pending_prune` path only; A16's prose explicitly
  states the upstream callback is not shipped and why. No code
  enforcement is feasible because the user's Trainer is theirs to
  configure; the protection is documentation.
- **R8: docs build flakiness from `griffe-pydantic` 1.3 rendering
  of `Mapping[str, int]`** (Phase 12). Mitigation: A20 item 4
  tracks the verification; the docs job runs `mkdocs build
  --strict` and a render regression fails CI.

## PR workflow per phase

Each phase is one PR (or a short stack). Every PR:

1. Branches off `main` after the prior phase merges.
2. Includes both `src/` changes and the `tests/` that gate them.
3. Adds a one-line note to this plan's "Phase status" section
   below at PR-open time.
4. Runs the `/review` Claude swarm to consensus.
5. Runs `/gemini-final-pass code <diff-spec>` after consensus,
   following the gating rules in the gemini-final-pass skill.
6. Merges only when all of: CI green, coverage delta non-negative,
   swarm consensus, Gemini-pass surfaces no new CRITICAL.
7. Updates this plan's "Phase status" section to mark the phase
   complete and writes one line on what changed if the
   implementation diverged from the plan.

Phases that span more than ~600 lines of `src/` split into
sub-PRs along internal seams (e.g. Phase 4 splits along
losses + optimizers + schedulers / callbacks / lightning_module /
trainer).

## Phase status

| Phase | Status | PR | Notes |
|---|---|---|---|
| 0 | not started | | scaffold + tooling |
| 1 | not started | | foundation primitives (incl. `TFTConfig`) |
| 2 | not started | | synthetic data + preprocessing |
| 3 | not started | | tensor primitives + serialization |
| 4a | not started | | determinism, precision, losses, optimizers, schedulers, sampling, callbacks |
| 4b | not started | | LightningModule + Trainer + `resume_path` plumbing |
| 5 | not started | | calibration (calibrators + threshold tuner) |
| 6a | not started | | `BaseSequenceEstimator` + classifier + regressor + smoke skeleton + base tests |
| 6b | not started | | transformer + recurrent family bases + `AttentionOutput` + family tests |
| 7 | not started | | TFT concrete (classifier + regressor) |
| 8 | not started | | public API + Optuna |
| 9 | not started | | check_estimator + acceptance + snapshots |
| 10 | not started | | ONNX + deploy |
| 11 | not started | | perf baselines |
| 12 | not started | | docs + release prep |

## Estimated effort

Order-of-magnitude only. Each phase counts the gated `src/` +
`tests/` work, not docs or research. Effort is "engineer-days on a
single experienced engineer working full-time."

| Phase | Estimate |
|---|---|
| 0 | 1 day |
| 1 | 5-7 days (post-hyperparameter-strategy fold-in: six new sub-config modules, six adapter classes, _extras machinery, expanded test roster) |
| 2 | 3-4 days |
| 3 | 4-5 days |
| 4a (determinism + precision + losses + optimizers + schedulers + sampling + callbacks) | 2-3 days |
| 4b (LightningModule + Trainer + resume_path plumbing) | 2-3 days |
| 5 | 2-3 days |
| 6a (BaseSequenceEstimator + classifier + regressor + smoke skeleton + base tests) | 2-3 days |
| 6b (transformer + recurrent family bases + AttentionOutput + family tests) | 2-3 days |
| 7 | 3-4 days |
| 8 | 2-3 days (post-fold-in: `_alpha_keys.py`, `_estimator_bridge.py` registry, suggest_params flag plumbing) |
| 9 | 3-4 days |
| 10 | 2-3 days |
| 11 | 1-2 days |
| 12 | 3-4 days |
| **Total** | **37-52 days** |

Calendar estimate is wider because of swarm-review cycles and
Gemini-pass wait time. A three-month v1 ship is achievable.

## Addressed

Round 1 (design-review swarm):

- **Phase 3 dependency declaration contradicted its deliverable
  (arch r1-C1).** Rewrote Phase 3 Dependencies as "Phases 1 and 2"
  with an explicit note that `TabularToSequence.transform`'s output
  dict is the backbone's input contract.
- **F5 `resume_path` had no phase assignment (arch r1-C2).** Added
  to Phase 4 (Trainer plumbing under both surface options) with
  the surface decision pinned in Phase 6 to
  `fit(X, y, *, resume_path=None)` per sklearn convention.
  Phase 4's `test_trainer.py` covers the `pl_trainer.fit(ckpt_path=...)`
  threading and the `RngStateCallback` RNG-state restore.
- **N1 `predict_quantiles` three error paths missing (arch r1-C3
  / qa r1-C6).** Added
  `tests/unit/models/transformer/tft/test_regressor_error_paths.py`
  to Phase 6 with three named tests
  (`test_predict_quantiles_on_point_mode_raises`,
  `test_predict_quantiles_before_fit_raises`,
  `test_predict_quantiles_off_vector_raises`).
- **N1 short-entity NaN-shape + single-aggregated-warning test
  missing (arch r1-C4 / qa r1-C5).** Added
  `tests/unit/models/test_short_entity_predict.py` to Phase 6.
- **N1 save/load version-mismatch `UserWarning` test missing (arch
  r1-C5 / qa r1-C1).** Added at two layers:
  `test_load_emits_version_mismatch_warning` in Phase 3 (low-level
  helper) and `test_load_version_mismatch_warning_via_estimator` in
  Phase 6 (estimator-level reload).
- **VSN and mean-pool mask correctness not scheduled (qa r1-C2).**
  Phase 3's `test_blocks.py` now schedules `test_vsn_mask_correctness`
  and `test_backbone.py` schedules
  `test_mean_pool_readout_mask_correctness`. Three N1 mask tests
  in total now (attention, VSN, mean pool).
- **NaN-loss Variant B fixture mismatch in Phase 4 (qa r1-C3).**
  Phase 4 now covers Variant A only (against the dummy estimator);
  Variant B moved to Phase 7 as
  `tests/integration/test_tft_nan_loss_variant_b.py` using the
  real `TFTClassifier` per the A14 `tft_classifier_fresh` fixture.
- **Same-process determinism test missing (qa r1-C4).** Added
  `tests/integration/test_dummy_estimator_same_process_determinism.py`
  to Phase 6.
- **Sub-phase 3a `config/tft.py` placement was a smell (arch
  r1-I8).** Moved `TFTConfig` to Phase 1 alongside the other
  pydantic configs. Phase 3 lost its sub-phase; Phase 1 gained the
  `test_tft.py` cross-field validator test.
- **AttentionOutput dataclasses placed in Phase 5 with no
  functional coupling (arch r1-I2).** Moved to Phase 6 alongside
  the estimator base shells that expose `predict_with_attention`.
- **Smoke skeleton dummy backbone did not lock the BackboneOutput
  contract (arch r1-I1).** Specified that the dummy backbone
  extends the transformer family and returns a full
  `TransformerBackboneOutput` with non-trivial introspection
  tensors so the entropy-emission and attention-extraction code
  paths execute through the smoke estimator.
- **`partial_fit` / `fit_predict` `NotImplementedError` stable
  message test missing (arch r1-I3).** Added to Phase 6's
  `test_base_estimator.py`.
- **GPU/CPU parity test had no phase home (arch r1-I5 / qa
  r1-I6).** Added `tests/integration/test_gpu_cpu_parity.py` to
  Phase 7 (after TFT lands), marked `gpu` and `slow`.
- **Phase 4 Optuna pruning wiring vs. Phase 8 dangling (arch
  r1-I6).** Rewrote Phase 8 to clarify: Phase 4 lands the
  LightningModule's `optuna_trial` constructor argument and the
  deferred-raise pattern; Phase 8 wires the Trainer to thread the
  user's trial into the constructor and exposes the
  `optuna_trial_guard` context manager.
- **Recurrent skeleton landed in same phase as TFT, weakening the
  v1 → v3 contract-lock claim (arch r1-I9).** Moved
  `RecurrentSequenceEstimator` (abstract class), `config/recurrent.py`,
  and the compose test from Phase 7 to Phase 6 so the cross-family
  contract locks before TFT specifics land.
- **Phase 0 docs CI job had no `mkdocs.yml` until Phase 12 (arch
  r1-I9).** Phase 0 entry now notes the docs job is a no-op script
  until Phase 12 lands the config, at which point it flips to
  `mkdocs build --strict`.
- **`strict_mode_globals` autouse fixture deferred to Phase 9 but
  required in Phase 4 (qa r1-I1).** Moved to Phase 4's local
  `tests/unit/training/conftest.py`.
- **Hardware-detect + resolve_precision combined test
  underspecified (qa r1-I2).** Phase 1 now owns `test_hardware_detect.py`
  (detection only) and Phase 4 owns `test_precision_resolution.py`
  with the combined parametrized run. No cross-phase touch-back.
- **Property-based tests only in Phase 2 (qa r1-I3).** Added
  hypothesis property tests in Phase 3 (GRN / VSN shape), Phase 4
  (loss output finiteness), and Phase 5 (calibrator output shape).
- **Three-disjoint-by-(id, time) assertion not explicit (qa
  r1-I5).** Phase 2 `test_splits.py` description rewritten to
  assert pairwise empty intersection on `(entity_id,
  window_time_index)` tuples, not just on integer index arrays.
- **`calibration_set + cal_fraction` conflict tested at the wrong
  layer (qa r1-I7).** Phase 2 retains the split-function-level
  test; Phase 6 adds
  `tests/unit/models/test_calibration_set_fit_conflict.py` for
  the estimator-shell-level check.
- **Phase 6 smoke skeleton calibration-included save/load not
  scheduled (qa r1-I9).** Added
  `test_save_load_with_temperature_calibration` as a second variant
  in Phase 6's `test_dummy_estimator_e2e.py`.
- **xfail-strict not specified on v1.1 validity-matrix rows (qa
  r1-N1).** Phase 1 test now names `pytest.mark.xfail(strict=True)`.
- **Phase 3 `(c_h, c_c)` ordering test under-specified for B>1 (qa
  r1-N2).** Phase 3 `test_backbone.py` description now requires a
  multi-entity batch (`B >= 2`) with distinct static features so
  a swap surfaces.
- **Phase 9 e2e tests slow-mark unstated (qa r1-N3).** Phase 9
  e2e test entries now annotate `slow` explicitly; only the
  quickstart runs on every PR.
- **Phase 5 byte-identical calibrator round-trip ambiguous (qa
  r1-N4).** Phase 5 now specifies the round trip is through
  `json.dumps` + `json.loads`, so JSON float-precision loss
  surfaces.
- **Title Case H1 (style r1-I2).** Rewritten to sentence case.
- **Rhetorical "? NO:" self-answering pattern (style r1-I1).** The
  Phase 6 dummy estimator description now states the long-lived
  fixture role declaratively.
- **Phase 4 module-count vs. estimate honesty (arch r1-NITPICK 1).**
  Effort table split Phase 4 into 4a / 4b.
- **Phase 1 effort estimate inconsistency.** Bumped from 2-3 days
  to 3-4 days to absorb the added `test_tft.py` work from
  the config-tft move.

Round 2 (design-review swarm):

- **`resume_path` surface contradicted requirements F5 (arch
  r2-C1).** Round 1's pin to `fit(X, y, *, resume_path=None)`
  contradicted requirements.md F5 line 677, which says
  "constructor argument". Re-pinned to
  `BaseSequenceEstimator.__init__(..., resume_path=None, ...)`.
  Phase 4 preamble updated to describe one path (Trainer accepts
  the path from the estimator and forwards via `ckpt_path=`)
  rather than two.
- **Phase status table stale notes (arch r2-I1).** Refreshed
  Phase 5 ("calibrators + threshold tuner"), Phase 6 ("split into
  6a / 6b"), and Phase 7 ("TFT concrete only") rows.
- **Phase 6 grew past the ~600-line guideline (arch r2-I2).**
  Split into 6a (BaseSequenceEstimator + classifier + regressor +
  smoke skeleton + base tests) and 6b (transformer + recurrent
  family bases + `AttentionOutput` + family tests). 6b depends
  on 6a; landing order is fixed.
- **`_DummyBackbone` / `_DummyHead` / `_LossReturningScalar`
  source location undeclared (arch r2-I3).** Added
  `tests/_test_models/_dummy_modules.py` as a Phase 4 deliverable
  so the LightningModule unit tests can use the fixtures directly.
- **Phase 4 LightningModule test trial source unspecified (arch
  r2-I4).** Added explicit "uses `optuna.trial.FixedTrial` with
  `should_prune` monkey-patched" note to the test description.
- **Phase 7 effort estimate inflated (arch r2-I5).** Reduced
  Phase 7 to 3-4 days; 6a / 6b each 2-3 days. Total unchanged.
- **Same-process determinism integration test had no fixture
  isolation under `pytest-randomly` (qa r2-C1).** Added
  `tests/integration/conftest.py` to Phase 6 with
  `integration_strict_mode_globals` autouse fixture, scoped to
  tests carrying `@pytest.mark.determinism`. The determinism
  integration test now carries that mark.
- **`tests/snapshot/test_tft_snapshot.py` had no `slow` mark
  policy (qa r2-C2).** Phase 9 entry now annotates
  `@pytest.mark.slow` explicitly and notes the test runs nightly.
  Adds a complementary "implementer MUST NOT mark quickstart slow"
  note to the e2e block to prevent accidental drift.
- **`test_regressor_error_paths.py` misplaced under `tft/` (qa
  r2-I1).** Moved file to `tests/unit/models/test_regressor_error_paths.py`;
  the error paths originate on `BaseSequenceRegressor`, not on TFT.
  The dummy regressor is the instance under test.
- **Phase 4a / 4b sub-PR coverage gap (qa r2-I2).** Phase 4
  body and effort table both reflect the split; transient
  coverage delta is acknowledged: Phase 4a closes 7 modules with
  their tests, Phase 4b closes 2 modules
  (`_lightning_module.py`, `trainer.py`) with their tests. Both
  PRs ship with their own coverage gates met.
- **`strict_mode_globals` fixture scope ambiguity (qa r2-I3).**
  Phase 4 entry now spells out the fixture scopes itself to
  `test_determinism.py` via a `request.node.fspath.basename`
  check inside the body, not via filename-bound autouse (which
  pytest does not provide directly).
- **`val_split_strategy="random"` warning at estimator-shell
  level (qa r2-I4).** Added second variant
  `test_val_split_random_warning_via_estimator` in Phase 6's
  `test_dummy_estimator_e2e.py` covering the F10 contract at the
  estimator-shell level (Phase 2 already tested the split helper).
- **F2 collapse case not named (qa r2-I5).** Added
  `test_calibration_none_threshold_false_collapses_cal_into_train`
  to Phase 2's `test_splits.py` description.
- **`on_train_epoch_end` deferred-raise timing (qa r2-N1).**
  Phase 4 test description rewritten to assert
  `optuna.TrialPruned` raises from the deferred-raise path at
  the END of the hook, not from `should_prune()` directly.
- **Quickstart slow-mark prohibition note (qa r2-N2).** Added
  the explicit "MUST NOT add `@pytest.mark.slow` to the
  quickstart" line in Phase 9.
- **Mask correctness test seed precondition (qa r2-N3).** Phase 3
  `test_interpretable_attention.py` description already states
  "fixed seed" alongside `model.eval()` and
  `torch.use_deterministic_algorithms(True)`; no doc change
  needed but confirmed.
- **Hedging adverb stacking (style r2-N1).** Rewrote
  "A realistic three-month v1 ship is plausible." as "A
  three-month v1 ship is achievable."

Round 3 (design-review swarm):

- **Effort table total math drift (arch r3-I1).** The Round 2
  rebalance increased Phase 1 and split Phase 6 into 6a/6b
  without updating the **Total** row. Re-summed to 35-48 days.
- **Phase 4 body did not demarcate 4a from 4b (arch r3-I2 / qa
  r3-I1).** Phase 4 modules and tests blocks now carry explicit
  `**Modules** (Phase 4a):` / `**Modules** (Phase 4b; depends on
  4a):` and matching deliverable-test demarcation.
- **Phase 6 body did not demarcate 6a from 6b (arch r3-I2 / qa
  r3-I2).** Same treatment applied to Phase 6 modules and tests
  blocks. The preamble's "phase grew" paragraph collapsed into a
  single-line note now that the body carries the structural
  split.
- **Phase 6 preamble double-named Trainer plumbing (arch r3-N1).**
  Tightened to "the Trainer's `ckpt_path=` forwarding from Phase 4
  receives the path."
- **Phase 4a callbacks row needed a handoff note (arch r3-N2).**
  Inline annotation added: 4a ships `RngStateCallback`; 4b's
  `test_trainer.py` exercises the RNG-state restore end-to-end.
- **Phase 9 done-when "meta-doc or checklist" ambiguity (qa
  r3-N1).** Rewrote as "the phase-by-phase N1 test assignment in
  this plan IS the checklist; the Phase 9 PR description
  references this plan's phase entries to confirm every item is
  implemented."
- **Phase 9 root-cause: `prediction_step` default `1` -> `0`
  (cross-phase, assumption changed since Phase 1).** The Phase 9
  accuracy assertions surfaced that the v1 TFT learned nothing on
  F6 (~0.68 AUC). Deep-dive root cause: `TabularToSequenceConfig`'s
  default `prediction_step=1` re-aligned the already-contemporaneous
  F6 panel into a 1-step forecast, contradicting the library's
  classification identity. Fix: default `0` (contemporaneous);
  `>0` opt-in forecast. With it, TFT reaches ~0.94 AUC (baseline
  ~0.98), so N1 thresholds hold unchanged (no `dgp_version` bump,
  no Phase 7 model change). Secondary: `predict` must return rows
  in input order, not the internal `(id, time)` sort order (F1
  contract). requirements F3/item-8/F6/N1 and architecture
  A4/A5/Phase-9-ledger are revised in sync with this entry. Because
  this changes a Phase-1 assumption it runs the full governance
  pipeline: doc revision -> Claude design-review consensus ->
  Gemini design consensus -> point-for-point refactor plan ->
  Claude consensus -> code -> Claude /review consensus -> Gemini
  code consensus. The Phase 9 quickstart / acceptance `xfail`s lift
  in the code stage once the fix lands.
  - **Mandatory tests the S4 refactor plan and S6 code stage MUST
    include (named here so they are not skipped):**
    1. `test_default_prediction_step_is_contemporaneous`
       (`tests/unit/data/test_tabular_to_sequence.py`): assert
       `TabularToSequenceConfig().prediction_step == 0` AND that the
       default transform's emitted target for a window ending at row
       `t` equals `label[t]`, not `label[t+1]`, on a fixed
       ground-truth 3-entity panel.
    2. Default-is-0 guard at BOTH layers:
       `TabularToSequenceConfig().prediction_step == 0` and the
       `TabularConfigParams` adapter default == 0. The named existing
       functions asserting the old default `1` -
       `test_tabular.py::test_tabular_to_sequence_config_defaults` and
       `test_adapters.py::test_tabular_config_params_defaults` (or the
       specific default-asserting functions/params the S4 grep
       identifies) - are updated in place, not parametrized-around or
       silently left.
    3. `tests/integration/test_predict_row_order.py::test_predict_output_row_order_shuffled_panel`
       (concrete new file): a multi-entity panel with **string** entity
       ids AND row-shuffled input; assert `predict` / `predict_proba` /
       `predict_quantiles` each align element-wise to `X.index` (the
       int-id synthetic panels coincidentally hide the bug, so this
       adversarial case is mandatory).
    4. Fast (<30s) signal-reachability integration test
       `tests/integration/test_contemporaneous_signal_reachable.py::test_signal_reachable_default_config_floor`:
       default-config `TFTClassifier` on a small F6 binary panel clears
       a loose floor (provisional `>=0.70`; S4 pins the exact floor and
       panel size against a measured run) so a regression to
       `prediction_step=1` is caught in the inner loop, not only the
       nightly slow e2e, before the quickstart xfail is lifted.
    5. `tests/unit/data/test_tabular_to_sequence.py::test_prediction_step_horizon_edge_clamp`:
       with `prediction_step>0`, a window whose target row would fall
       past the entity's last period has its target row clamped to
       `min(window_end+prediction_step, n_rows-1)` and is NOT dropped,
       so the one-window-per-row count holds for `prediction_step>0`
       too (verifies the F3 horizon-edge contract). S4/S5 DECISION
       (refactor_prediction_step.md Step 6): the F6 generator's
       vestigial `prediction_step` skip-guard is REMOVED entirely
       (not reworked into forecast-aligned emission), so the prior
       "parity with generator forecast mode" clause is MOOT and
       dropped. Test #5 therefore pins ONLY the kept forecasting path
       (`TabularToSequence.prediction_step>0` horizon-edge clamp). The
       generator-removal completeness + tail-trim count assertion is
       NOT bundled here; it is the dedicated test #12, decoupled
       (S5 R2) so neither test can be silently omitted while the
       other is written.
    6. `tests/integration/test_predict_row_order.py::test_below_min_periods_predict_entity_nan_filled_preserves_count`:
       a panel with one sub-`min_periods_predict` entity (N>1 rows) and
       one above-floor entity; assert `len(predict(X)) == len(X)`,
       every row of the sub-floor entity is NaN (N NaN rows, not a
       single collapsed row, not dropped), the above-floor rows are
       finite, all `X.index`-aligned, and exactly one aggregated breach
       warning fired. This pins the F1 per-row NaN-fill cardinality
       invariant (req `min_periods_predict` per-row clause) that a
       drop-the-entity regression would otherwise break silently while
       still passing the row-order test #3.
    7. `tests/integration/test_predict_row_order.py::test_calibration_fold_alignment_unsorted_x_cal`
       (Gemini S3 G-C2): the fixture MUST be mispairing-SENSITIVE, NOT
       a separable fold (separability masks a label mispairing). Use a
       `calibration_set=(X_cal, y_cal)` with string entity ids in a
       NON-`(id_col,time_col)` order so the internal sort is a
       non-identity permutation, and a fold where the calibrated
       output depends on correct (prediction,label) pairing
       (e.g. labels constructed so the correctly-paired temperature/
       Platt fit is numerically distinct from the mispaired fit).
       Assert BOTH: (a) the correctly-paired calibrator/threshold
       equals an independently computed oracle (concrete form: the
       fitted binary `decision_threshold_` is within 1e-6 of the
       oracle threshold, or post-calibration ECE <= the N1 band); AND
       (b) it differs measurably from the result the sorted-vs-caller
       mispairing would produce (assert the two are NOT close), so the
       test fails if `_calibration_fold` pairs `transform(x_cal)`-
       sorted outputs with caller-order `y_cal`.
    8. `tests/integration/test_predict_row_order.py::test_predict_with_attention_row_order_shuffled_panel`
       (Gemini S3 G-C3): shuffled string-id panel; assert
       `predict_with_attention` / `predict_with_states` returns a
       dataclass whose EVERY per-row field is aligned to `X.index`,
       enumerated with no ellipsis - `AttentionOutput`: `predictions`,
       `probabilities`, `logits`, `var_selection_weights`,
       `static_var_selection_weights`, `attention_weights`,
       `padding_mask`, `entity_id`; `RegressionAttentionOutput`:
       `predictions`, `var_selection_weights`,
       `static_var_selection_weights`, `attention_weights`,
       `padding_mask`, `entity_id`. Separately assert
       `RegressionAttentionOutput.quantiles_used` is shuffle-INVARIANT
       (identical regardless of input row order; it is fit-time
       metadata, never permuted). Same `input_row_order` permutation
       as the array predict surfaces.
    9. `tests/unit/data/test_tabular_to_sequence.py::test_transform_input_row_order_is_stateless`
       (Gemini S3 G-C1 structural invariant + S5 R1 value-oracle):
       three assertions. (a) VALUE ORACLE: on a small KNOWN string-id
       panel deliberately NOT in `(id_col,time_col)` order (hand-built,
       e.g. ids `["b","a","b","a"]` with explicit time values), assert
       the returned `input_row_order` tensor equals a hand-computed
       expected permutation (compute the post-sort emission order on
       paper, then the inverse via argsort, and hardcode that exact
       index vector in the test) AND that indexing the emission-order
       `entity_id`/`target` arrays by `input_row_order` reproduces the
       original caller row order. This is a direct unit oracle on the
       permutation itself, not only the downstream array/below-floor
       checks of #3/#6 (which would pass even for a subtly wrong but
       self-consistent permutation). (b) STATELESS: `transform` sets NO
       `input_row_order`* attribute on the transformer instance -
       snapshot `set(vars(tts))` (and `getattr(tts, "__dict__", {})`)
       before and after two successive `transform` calls on
       different-order panels; the attribute set is unchanged. (c) NO
       CROSS-CALL LEAK: the two calls' returned `input_row_order`
       tensors are independent, so a stateful `self.input_row_order_`
       implementation (which would race under concurrent predict)
       fails this even in serial execution. (d) ERROR PATHS (S5 R2,
       testing.md one-error-path-per-public-function): two
       `pytest.raises(DataContractError)` cases in the same file -
       (d1) a DIRECT unit call to the module-private
       `_restore_permutation(emitted_pos, n)` helper (the Step 2c
       TESTABILITY SEAM) with a deliberately short `emitted_pos`
       (e.g. `_restore_permutation([0, 1], 3)`) so the
       `len != n` `DataContractError` raise is exercised by a real
       call rather than an unconstructable data path (the raise is
       unreachable through `transform()` on the real predict path by
       Step 2c's own argument, so a data-driven test would be
       vacuous); (d2) a caller `X` already containing the private
       `_POS` sentinel column triggers the Step 2d collision raise
       BEFORE the sort/assign. Both new raises are the only safety
       net for the row-order contract and must be exercised by a
       call that actually triggers them, not assumed.
    10. Phase-1-8 re-validation obligation: audit every test/usage
       constructing `TabularToSequenceConfig` / `TabularConfigParams`
       WITHOUT an explicit `prediction_step` (grep across `tests/` and
       `src/`); any that implicitly relied on the old `=1` windowing
       (target alignment, below-floor counts, sentinel positions,
       synth->tensor shapes) is reviewed and updated. The full
       Phases 1-9 gate (+3 randomized) must stay green post-change.
       CLOSED only when the commit body records the verbatim grep hit
       count AND a per-hit affected/not-affected classification for
       every hit (no unclassified hit; refactor_prediction_step.md
       Step 8 completeness done-criterion).
    11. `tests/integration/test_predict_row_order.py::test_calibration_fold_internal_split_sorted_consistent`
       (S5 R2, Gemini G-C2 sibling path): the DEFAULT calibration
       path - fit with `cal_fraction>0` and NO explicit
       `calibration_set` on a shuffled non-`(id_col,time_col)`-order
       string-id panel - exercises the internal-split branch
       `_base.py:359-379`, where `_raw_outputs(batch)` and
       `batch["target"]` are both indexed by the same sorted-space
       `keep` (`keep = cal_idx[~_below_floor_mask(batch)[cal_idx]]`,
       `_base.py:369`; `keep` is the below-floor-filtered subset of
       the sorted-space `cal_idx`). FIXTURE MUST BE NON-DEGENERATE
       (sensitivity clause, mirrors test #7): a binary +
       threshold-tuning panel with at least two distinct per-entity
       class-label patterns so the fitted `decision_threshold_` is
       non-trivial (assert it is neither 0.0 nor 1.0); a degenerate
       single-class or all-zero-feature fixture would yield identical
       calibrators under BOTH the correct and the regressed
       implementation and pass vacuously. Then assert the fitted
       calibrator/`decision_threshold_` (or post-calibration ECE) on
       the shuffled panel is IDENTICAL (within 1e-6) to fitting the
       same model on the same panel pre-sorted by `(id_col,
       time_col)`, proving the internal-split branch never has
       `input_row_order` applied and stays sorted-space
       self-consistent. This pins the branch test #7 does not reach
       (test #7 uses an explicit `calibration_set`, so it exercises
       only `_base.py:355-358`); a regression that threads
       `input_row_order` through the internal-split branch would
       silently mis-pair the most common calibration path while
       #1-#10 stay green.
    12. `tests/unit/data/test_synthetic_generator.py::test_generator_has_no_prediction_step_and_emits_n_periods`
       (S5 R2, decoupled from #5): a DEDICATED test, separate from
       the `TabularToSequence` clamp test #5, so neither can be
       silently omitted while the other is written. Assert
       `"prediction_step"` is absent from
       `inspect.signature(SyntheticPanelGenerator).parameters`, that
       constructing with `prediction_step=` raises `TypeError`, and
       that generation on a fixed panel emits exactly `n_periods`
       (not `n_periods - 1`) windows per entity, including that a
       single-period entity now appears (entity-id SET change, not
       just count; refactor_prediction_step.md Step 6b consequence
       (ii)). Test #5 retains ONLY the
       `TabularToSequence.prediction_step>0` horizon-edge clamp; its
       prior bundled generator-removal companion clause is moved here.

## Deferred

Round 1 (design-review swarm):

- **Docstring examples tested only in Phase 12 (qa r1-I4).**
  Phase 12 owns the `pytest --doctest-modules` gate. The cost of
  running it per-phase is real (it imports every module), and the
  N1 requirement only calls for the gate to exist, not for
  per-phase enforcement. If a Phase 1-3 docstring has a wrong
  example, the Phase 12 gate catches it before release.
- **Structured-log event tests fixture grounding (qa r1-I8).**
  Phase 9's `test_structured_log_events.py` is the consolidating
  pass; the design detail of whether each event's test uses stubs
  or real TFT inference is left to the Phase 9 implementer. The
  entropy events need non-trivial introspection tensors, which
  the smoke skeleton already provides (per the BackboneOutput
  fix above), so the dummy estimator is a viable fixture for
  most events.
- **MIGRATIONS meta-test still partially open per feedback memory
  (qa r1-I10).** The Phase 3 schedule explicitly names
  `test_migrate_detects_no_op_registration` via `monkeypatch`,
  which is the architecture's prescribed approach. The Round 2
  feedback-memory note flagged this as a doc-clarity issue, not
  an implementation gap; no further action needed in the plan.
- **Phase 5 `robust` and `robustness` token matches (style r1-N1,
  r1-N2).** Both uses are technical (scaler strategy name and
  test-concept name); not vague-praise vocabulary.
- **Bold density in the plan (style r1-N3).** All 97 bold spans
  are structural labels (`**Goal**`, `**Modules**`, etc.), not
  prose emphasis. Plan structure relies on them for fast
  scanning.

Round 3 (design-review swarm):

- **MIGRATIONS meta-test vacuous-pass risk (qa r3-N2).** Already
  addressed structurally: A17's `test_migrate_detects_no_op_registration`
  uses `monkeypatch` to inject a synthetic non-empty MIGRATIONS
  dict, so the test exercises the non-advancement path
  meaningfully even when the production MIGRATIONS dict is empty.
  No additional plan-body note needed.

Gemini final pass (cross-family review on the implementation plan):

- **`_LightningModule` hardcoded transformer-specific attributes
  (gemini r1-C1).** Architecture A15's entropy-emission code
  directly indexed `var_selection_weights`, `attention_weights`,
  and `static_var_selection_weights` on `self._last_train_output`.
  v3's `RecurrentBackboneOutput` (with `hidden_states` instead of
  `attention_weights`) would raise `AttributeError` on every
  recurrent training epoch, forcing a rewrite of the generic
  training module. Refactored to a Protocol + delegation: A15
  defines `BackboneOutput` as a Protocol carrying only the
  family-agnostic fields (`representation`, `padding_mask`);
  `BaseBackbone` declares
  `compute_training_metrics(output) -> dict[event_name, payload]`
  with a default empty implementation. `TFTBackbone` overrides
  the method to emit the two TFT-specific events. The
  LightningModule reads only the Protocol surface plus the
  returned dict; v3 recurrent backbones override the method to
  emit `train.hidden_norm` without touching LightningModule code.
  Plan Phase 3 now lands `models/_backbone.py` (the Protocol and
  abstract base) and `models/transformer/_backbone.py`
  (the transformer-family concrete dataclass); Phase 4b's
  LightningModule consumes them by import.
- **Entropy reductions dropped the padding mask (gemini r1-C2).**
  Padded timesteps carry uniform `1/n_vars` VSN rows (after softmax
  over zero pre-softmax inputs per A6) and zero attention rows
  (after `nan_to_num` per A6). Averaging across all timesteps
  inflated `temporal_entropy` (max-entropy padded rows pull up the
  mean) and deflated `entropy_per_head` (zero-entropy padded
  query rows pull down the mean). A15 now adds `padding_mask` to
  the base `BackboneOutput` Protocol and computes masked means in
  `compute_training_metrics`. Phase 3's `test_backbone.py` adds
  `test_compute_training_metrics_ignores_padded_positions` to
  prove the mask is applied.
- **Optuna trial polluting pydantic config (gemini r1-C3).** A16's
  prior objective example put `params["optuna_pruning_trial"] =
  trial` into the kwargs passed to `TFTClassifier(**...)`, which
  would crash at construction time because `optuna_pruning_trial`
  is not a `TFTConfig` field and `extra="forbid"` is set. Moved
  the trial out of `__init__` into the `fit` keyword:
  `BaseSequenceEstimator.fit(X, y, *, calibration_set=None,
  optuna_trial=None)` per requirements F1 (updated to match). The
  trial reaches `_LightningModule` via the Trainer; it is never
  serialized, never enters `get_params`, and cannot break
  `extra="forbid"`. `resume_path` stays on `__init__` per
  requirements F5; it is not a `fit` keyword.
- **`PyTorchLightningPruningCallback` reintroduces the
  validation-hook-raise lifecycle bug (gemini r1-C4).** The
  upstream callback raises `TrialPruned` from `on_validation_end`,
  which Lightning fires BEFORE `on_train_epoch_end`; that skips
  the `train.epoch` and entropy events for the pruned epoch (the
  exact bug A7's `_pending_prune` deferred-raise pattern was
  designed to prevent). Removed `PyTorchLightningPruningCallback`
  from Phase 8's module list. The library's native
  `_pending_prune` path is the only supported pruning hook.
- **`FixedTrial` sweep is vacuous (gemini r1-I1).** Switched
  Phase 8's `test_suggest_params.py` from a 1000-iteration
  `FixedTrial` sweep to `optuna.create_study().ask()` so each
  trial actually samples the search space. `FixedTrial` remains
  the right tool for the `optuna_trial_guard` tests below where
  the wrapped body needs `report` / `should_prune` to be no-ops.
- **`__sklearn_tags__` on concrete `TFTClassifier` forces
  duplication (gemini r1-I2).** Moved the method to
  `BaseSequenceEstimator` (Phase 6a) with the family-wide tag
  block; concrete classes override only when a tag must flip
  (e.g. v3 recurrent models override to set
  `non_deterministic = True`).
- **`quantiles` validator placement (gemini r1-I3).** The
  `quantiles` field lives on `BaseModelConfig` per A4. Moved the
  cross-field validator (strictly increasing, in `(0, 1)`) from
  `TFTConfig` to `BaseModelConfig` and the test from
  `test_tft.py` to `test_base.py`. v2 quantile
  regressors (PatchTST quantile, TimesNet quantile) inherit the
  validator without duplication.

Post-Gemini design-review swarm (Round 1 verification):

- **`_backbone.py` vs `_base.py` divergence (arch r1-C1).** A1
  package layout now lists `models/_backbone.py` and
  `models/transformer/_backbone.py`. A15's code comments updated
  to reference the same paths. Phase 6a's `BaseSequenceEstimator`
  imports `BaseBackbone` from `_backbone.py` for type annotations
  only; the Protocol / abstract base does not live in `_base.py`.
- **`resume_path` self-contradiction (arch r1-C2).** Phase 6a's
  Modules block dropped `resume_path=None` from the `fit`
  signature; the keyword stays on `__init__` per requirements F5.
  Architecture A20 item 5 marked RESOLVED. The Gemini-pass entry
  in architecture's Addressed section updated to reflect the
  `__init__`-only surface.
- **Requirements F7 stale design (arch r1-C3).** Updated F7's
  pruning hook description to the native `_LightningModule`
  deferred-raise pattern; switched `suggest_params` test guidance
  to `optuna.create_study().ask()`; updated the N3 dep-table
  rationale. Requirements and architecture are now aligned.
- **Test name mismatch on the empty-metrics test (qa r1-C1).**
  `test_base_backbone_compute_training_metrics_returns_empty` is
  the canonical name; architecture A15 updated to match the plan.
- **LightningModule delegation loop untested in Phase 4b (qa
  r1-C2).** Added
  `test_on_train_epoch_end_emits_events_from_compute_metrics` to
  Phase 4b: a `_DummyBackbone` subclass overrides
  `compute_training_metrics` to return one synthetic payload; the
  test asserts `caplog` contains exactly one matching record.
- **BaseModelConfig snippet missing the quantiles validator (arch
  r1-I1).** Added the `_check_quantiles_monotone` model validator
  inline in A4's `BaseModelConfig` block.
- **Protocol-vs-dataclass mixing (arch r1-I2).** Refactored
  `BackboneOutput` from `Protocol` to a plain `@dataclass`;
  `TransformerBackboneOutput` extends it via standard dataclass
  inheritance. Pyright strict mode passes without
  `@runtime_checkable` ceremony.
- **R7 stale smoke-test mitigation (arch r1-I3).** Rewrote R7 to
  describe the real risk (a user importing the upstream callback
  themselves) and the actual mitigation (documentation in
  examples and A16).
- **Dummy-backbone description mis-typed (arch r1-I4).** Phase 6a
  dummy backbone now described as returning
  `TransformerBackboneOutput` (not the base `BackboneOutput`); the
  smoke estimator extends the transformer family for testing the
  entropy-emission path.
- **A6 line 625 reference (arch r1-I5).** Changed "BackboneOutput"
  to "TransformerBackboneOutput" in the interpretable-attention
  return note.
- **`flatten_double_underscore` undefined (arch r1-I6).** A16 now
  defines `config_to_estimator_kwargs(config)` inline, with the
  pydantic-dump-to-adapter-instance round trip spelled out for
  the `tabular_config` nested field.
- **Optuna trial routing untested at unit level (qa r1-I1).**
  Added `tests/unit/tuning/test_fit_optuna_trial_kwarg_routing.py`
  in Phase 8: asserts `optuna_trial` is absent from `get_params`
  and reaches the LightningModule. Pins the F1 contract that the
  trial is `fit`-only.
- **`__sklearn_tags__` field values untested (qa r1-I2).** Added
  `test_sklearn_tags_base_values` to Phase 6a's
  `test_base_estimator.py`: asserts every documented field value
  on the `Tags` dataclass returned by `BaseSequenceEstimator.__sklearn_tags__()`.
- **Future-proofing item 3 stale wording (arch r1-N2).** Rewrote
  the item to describe the `compute_training_metrics` delegation
  pattern instead of a "named tuple" shape; the v3 extension
  description now matches the Protocol-superseded-by-dataclass
  abstraction.
- **`entropy_per_head` broadcast simplification (arch r1-N1).**
  Dropped the explicit `.expand_as(...)` on the masked-count
  divisor; broadcasting handles the scalar / `(H,)` shape.
- **`static_entropy` no-mask branch assertion (qa r1-N1).** The
  `test_compute_training_metrics_ignores_padded_positions`
  description now also asserts `static_entropy` equals the
  reference value computed without the mask, confirming the
  static branch correctly skips the time-axis mask.

Post-Gemini design-review swarm (Round 2 verification):

- **Phase 3 body still said `BackboneOutput` Protocol (arch
  r2-C1).** Rewrote the Phase 3 module entry to "`BackboneOutput`
  dataclass" matching architecture A15 and the post-refactor
  contract. Also updated the preamble at line 12 ("named-tuple" →
  "dataclass") and the Phase 3 backbone test description ("Protocol
  fields" → "dataclass fields").
- **gemini-r1-C3 ledger entry stale `resume_path` (arch r2-C2).**
  Removed `resume_path=None` from the fit-signature quoted in the
  ledger; the keyword stays on `__init__` per requirements F5.
- **`suggest_params` docstring stale (arch r2-I1).** Updated A16's
  `suggest_params` docstring to reference
  `optuna.create_study().ask()` instead of `FixedTrial-built`.
- **Preamble item 2 stale shape (arch r2-I2).** "BackboneOutput
  named-tuple" → "BackboneOutput dataclass" in the preamble's
  locked-contracts list.
- **arch-r1 ledger entry "BackboneOutput named-tuple" (arch
  r2-I3).** Updated to "BackboneOutput dataclass" with a parenthetical
  note explaining the original NamedTuple was refactored.
- **arch r1 qa-I2 ledger entry referenced wrong type (arch
  r2-I3).** Updated the BackboneOutput reference in the
  architecture qa r1-I2 Addressed entry to dataclass + the
  TransformerBackboneOutput note.
- **smoke-skeleton ledger entry mis-typed (arch r2-I4).** Updated
  the Round 1 entry to say `TransformerBackboneOutput` instead of
  `BackboneOutput` for the dummy backbone's return type.
- **A15 "Three named tests" count stale (qa r2-I1).** Architecture
  A15 now enumerates FOUR named tests, adding
  `test_on_train_epoch_end_emits_events_from_compute_metrics` so
  the LightningModule delegation loop is documented in A15 (it was
  in the plan's Phase 4b body but absent from the architecture's
  contract enumeration).
- **`static_entropy` clause missing from Phase 3 body (qa
  r2-I2).** Phase 3's `test_compute_training_metrics_ignores_padded_positions`
  description now includes the `static_entropy` no-mask assertion;
  matches A15.
- **N1 three-way split estimator-level test missing (qa r2-I3).**
  Added
  `tests/integration/test_three_way_split_correctness.py` to
  Phase 6a with two named tests
  (`test_estimator_split_folds_disjoint_by_id_time`,
  `test_estimator_split_collapse_when_no_calibration`). Phase 2
  covers the split helper in isolation; this integration test
  closes the N1 "Fit on a panel" clause through `estimator.fit`.
- **Phase 4b duplicate sentence (arch r2-N1).** Removed the
  trailing `test_on_train_epoch_end_skips_entropy_when_no_output`
  sentence; the test is already named in the three-named-tests
  block above.

Hyperparameter-strategy fold-in (Round 1):

- **Phase 1 module roster expanded.** Added six new module entries
  (`_extras.py`, `optimizer.py`, `scheduler.py`, `loss.py`,
  `sampler.py`, `_adapters.py` renamed from `_params_adapter.py`)
  plus updated descriptions for `base.py` (nested sub-configs),
  `tabular.py` (`CategoricalEmbedDims` tuple form), and `tft.py`
  (`TFTAdvancedConfig` + `advanced` field). The single-adapter
  `TabularConfigParams`-only design from the previous Phase 1 spec
  is superseded by the six-adapter family per architecture A4.
- **Phase 1 deliverable tests expanded.** Renamed
  `test_params_adapter.py` to `test_adapters.py` with six per-
  adapter clone-safety tests plus
  `test_all_adapters_have_keyword_only_init`. Added new test files
  per family sub-config (`test_optimizer.py`, `test_scheduler.py`,
  `test_loss.py`, `test_sampler.py`) and `test_extras.py` for the
  escape-hatch machinery and deprecation-alias contract. Updated
  `test_validity_matrix.py` to construct via the nested shape
  (`LossConfig`, `SamplerConfig` sub-configs); parametrize IDs
  stay 4 strings so pytest caches are stable. Added
  `test_v1_task_type_rejects_multilabel_and_regression_multioutput`
  to `test_base.py` pinning the v1.1-unreachable guard.
  Added `test_tft_config_advanced_field_is_not_none_by_default`
  and `test_tft_advanced_config_default_construction_succeeds`
  to `test_tft.py` pinning the `advanced` slot contract.
- **Phase 6a `_build_config` loss-default injection.** `_build_config`
  now injects `LossParams(strategy=_DEFAULT_LOSS_FOR_TASK[task_type])`
  when `self.loss is None`. New test
  `tests/unit/models/test_loss_default_injection.py::test_loss_default_injection_per_task_type`
  parametrizes over v1 task types and asserts the injected
  default. The corresponding clone-safety test
  `test_outer_estimator_clone_does_not_alias_adapter_instances`
  also lives in Phase 6a (extends the Phase 1 per-adapter clone
  roster with the outer-estimator-level guarantee).
- **Phase 8 Optuna module and test additions.** Phase 8 now ships
  `_alpha_keys.py` (empty enum lists in v1) and
  `_estimator_bridge.py` (helper + adapter map registry
  per A16 fold-in). `test_suggest_params.py` gains the per-flag
  sampling tests; `test_estimator_bridge.py` (new)
  pins the round-trip and `extra`-tuple `mode="json"` contracts.
- **F7 signature change cascaded through Phase 8.** `suggest_params`
  carries `search_advanced` and `search_extras` keyword-only flags;
  the per-model default search space is defined by the Phase 8
  `suggest_params` implementation (architecture A16).

Gemini three-doc final pass:

- **Phase 1 reserved-keys test placement contract clarified**
  (gemini-qa r1-C1). `test_adamw_reserved_keys_collision_raises` and
  `test_sgd_reserved_keys_collision_raises` test the
  `_check_extra_not_reserved` model_validator on `OptimizerConfig`
  (which lives in Phase 1's `src/seq_sklearn/config/optimizer.py`),
  NOT the build-factory check at `build_optimizer` (Phase 4). The
  reserved-keys collision is a config-layer invariant per
  architecture A4; Phase 1 ownership is correct.
- **`test_extract_deprecated_extras_happy_path_passes_through` added**
  to Phase 1's `test_extras.py` roster (gemini-qa r1-I2). Pins the
  unpromoted-key passthrough contract that the prior test roster
  missed.
- **F7 / requirements N1 paragraph updated** (gemini-arch r1-I1 +
  gemini-qa r1-I1). The `suggest_params` signature in F7 now uses
  the strict type hints matching architecture A16. The N1 Phase 8
  carve-out now names both the `test_suggest_params_*` trio AND the
  `test_config_to_estimator_kwargs_*` pair.

S5 refactor-plan consensus (refactor_prediction_step.md Step 6):

- **Forecast-aligned synthetic emission deferred.** The considered
  alternative to removing the F6 generator's vestigial
  `prediction_step` (make `generator.prediction_step>0` emit
  genuinely forecast-aligned targets so forecast-mode synthetic data
  exists) is deferred: it is net-new functionality outside v1 scope,
  not required by any N1 threshold, and the kept forecasting path is
  `TabularToSequence.prediction_step>0` (test #5). Revisit only if a
  future requirement needs forecast-mode synthetic panels.

S7 code-review (S6 implementation, deferred items):

- **Two calibration bands re-derivation - RESOLVED (post-pipeline,
  user-directed, Claude /review-gated).** The `temperature` ECE and
  `conformal` coverage bands were pinned against the buggy
  `prediction_step=1` seed-42 value. Re-derived against the corrected
  contemporaneous regime via a 5-seed sweep (42, 137, 9999, 7, 2024):
  temperature ECE = [0.061, 0.034, 0.035, 0.050, 0.057] (mean 0.047,
  max 0.061); conformal coverage = [0.856, 0.831, 0.825, 0.850,
  0.842] (min 0.825, max 0.856; conservative over-coverage on an 80%
  nominal interval). Re-pinned: `_CLF_ECE_BANDS["temperature"]`
  0.05 -> 0.07; `_REG_COVERAGE_BANDS["conformal"]` (0.75, 0.85) ->
  (0.75, 0.88). Both new constants EQUAL this file's existing
  `platt`/`isotonic` ECE band (0.07) and `isotonic_quantile` upper
  coverage band (0.88) respectively - temperature/conformal are now
  held to the same bar the project already accepts for the adjacent
  strategies, with headroom over the measured 5-seed extremes. The
  two `xfail(strict=False)` cases and the `_PS_REFACTOR_BAND_XFAIL`
  constant were removed; both are hard assertions again and pass at
  seed 42. The user directed this re-pin and required a Claude
  /review round on the quality-gate constant change before commit
  (no unilateral weakening). The /review round reached consensus
  (code-opus/sonnet + qa-opus/sonnet all APPROVE, 0 CRITICAL): the
  bands still catch a genuinely broken model (near-random ECE
  0.12-0.20 >> the 0.07 ceiling) and the safety-critical conformal
  LOWER bound 0.75 is unchanged (under-coverage still caught; only
  the benign over-coverage ceiling moved).
- **Calibration ECE/coverage measured in-sample + thin temperature
  headroom (deferred IMPROVEMENT).** The /review qa pass noted ECE
  and coverage are evaluated on the training panel (an in-sample
  proxy) and the temperature band has ~0.009 headroom over the
  5-seed max. Both are PRE-EXISTING properties shared by ALL five
  band-sensitive cases (platt, isotonic, conformal,
  isotonic_quantile too), NOT introduced by this re-pin, and neither
  creates false-pass risk (a broken model fails in-sample too).
  Deferred: a holdout-split calibration evaluation + headroom review
  is a separate calibration-methodology improvement, out of scope
  for closing the prediction_step deferral.
- **Explicit `calibration_set` below-floor exclusion (Gemini S8 +
  arch-opus R3-I1 extension) - RESOLVED.** Decision: the explicit-
  `calibration_set` branch is caller-owned and does NOT floor-filter
  short-history (`< min_periods_predict`) entities; this is the
  final v1 contract, not a gap. Rationale: the targets are the
  caller's real `y_cal` (not `transform` sentinels, so no sentinel-
  label hazard), and the recomputed-fold drop is correctly scoped to
  the internal-split branch. The deliberate asymmetry is now pinned
  by `test_explicit_calibration_set_keeps_below_floor_entity` (an
  all-below-floor explicit set still calibrates, unlike the
  internal-split branch which raises). The related fail-fast
  `len(x_cal) == len(y_cal)` guard is IMPLEMENTED in
  `_calibration_fold` (raises `DataContractError` before transform
  instead of failing late in the calibrator/tuner fit), pinned by
  `test_explicit_calibration_set_length_mismatch_raises`. The
  `_calibration_fold` docstring states the explicit-vs-recomputed
  asymmetry explicitly.
