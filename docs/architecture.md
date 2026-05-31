# seq-sklearn architecture

This document is the v1 architecture spec: the package layout, the
public API surface, and the per-section anchors (`A1` through `A21`)
that source-code module docstrings cite. Two sections are LOAD-BEARING
for CI gates and are preserved verbatim:

- `A3: Public-API surface` is parsed by
  `tests/unit/test_public_api_surface.py` to extract the canonical
  `__all__`.
- `A21: ONNX restricted op surface` is copied verbatim by
  `tests/deploy/test_restricted_op_surface.py` as the `ONNX_SAFE_OPS`
  allowlist.

For "why" answers (TFT choice, determinism model, attention
semantics), see `docs/explanation/`. For the live API reference, see
`docs/reference/api.md`. For benchmark-suite-specific design, see
`docs/benchmark_suite_design.md` and `docs/benchmark_suite_phase_log.md`.

## A1: Package layout

```
src/seq_sklearn/
  __init__.py                       re-exports public API (per A3)
  errors.py                         SeqSklearnError hierarchy (A10)
  hardware.py                       detect(), HardwareTier (A11 / N5)
  _validate.py                      input validation helpers (internal)
  serialization.py                  safetensors + JSON I/O (A17 / F4)
  logging.py                        Event enum, emit() helper (A15 / F11)

  config/
    _domains.py                     TASK_TYPES, LOSS_STRATEGIES (F5)
    _validity.py                    F5 validity-matrix cross-field check
    _extras.py                      ExtraDict + promotion bookkeeping
    adapters.py                     {Tabular,Optimizer,Scheduler,Loss,
                                    Sampler,TFTAdvanced}Params (A4 / A3)
    base.py                         BaseTrainingConfig, BaseModelConfig
    {optimizer,scheduler,loss,sampler}.py  family sub-configs
    tabular.py                      TabularToSequenceConfig
    tft.py                          TFTConfig + TFTAdvancedConfig
    recurrent.py                    RecurrentSequenceEstimatorConfig
                                    (v1 skeleton, INTERNAL)

  data/
    tabular_to_sequence.py          TabularToSequence (A5)
    splits.py                       compute_three_way_split (A5 / F2)
    encoders.py                     CategoricalEncoder, scalers
    synthetic/{generator,_rng}.py   SyntheticPanelGenerator (F6)

  models/
    _layers.py                      Linear/LayerNorm/Embedding factory (F4)
    _backbone.py                    BackboneOutput, BaseBackbone (A15)
    _base.py                        BaseSequenceEstimator
    _classifier.py                  BaseSequenceClassifier
    _regressor.py                   BaseSequenceRegressor
    _heads.py                       ClassificationHead, RegressionHead (A6)
    _attention.py                   mask-polarity flip helper (A6)
    transformer/
      _backbone.py                  TransformerBackboneOutput (A15)
      _base.py                      TransformerSequenceEstimator
      _interpretable_attention.py   shared-V interpretable MHA (TFT)
      _positional.py                positional encoders
      tft/
        backbone.py                 TFTBackbone (A6)
        blocks.py                   VSN, GRN, GLU, AddNorm
        _estimator.py               _TFTEstimatorMixin (A4 plumbing)
        {classifier,regressor}.py   TFTClassifier, TFTRegressor
    recurrent/
      _base.py                      RecurrentSequenceEstimator
                                    (abstract, INTERNAL in v1; A6.1)

  training/
    trainer.py                      Trainer (Lightning wrapper, A7)
    _lightning_module.py            _LightningModule (A7)
    _determinism.py                 enable_strict_mode() (N4)
    _precision.py                   resolve_precision(tier, requested) (N5)
    callbacks.py                    GradScalerWatchdog, EventEmitter,
                                    RngStateCallback
    losses.py                       build_loss() dispatch (A8 / F5)
    optimizers.py                   build_optimizer() (F5)
    schedulers.py                   build_scheduler() (F5)
    sampling.py                     oversample/undersample helpers (F5)

  calibration/
    _protocol.py                    _Calibrator Protocol
    classification.py               Temperature, Platt, Isotonic
    regression.py                   Conformal, IsotonicQuantile
    threshold.py                    ThresholdTuner

  inference/
    attention.py                    AttentionOutput,
                                    RegressionAttentionOutput (A15.1)
    onnx.py                         ONNX-traceable forward wrapper (A21)

  model_selection/
    split.py                        EntityTimeSeriesSplit

  tuning/
    suggest_params.py               Optuna-curated search spaces (A16)
    pruning.py                      optuna_trial_guard ctx manager
    _alpha_keys.py                  per-family ALPHA-key enum lists
    _estimator_bridge.py            config_to_estimator_kwargs
```

The benchmark suite lives outside `src/`:

```
benchmarks/                         Cross-family comparator harness
  protocol/                         lookback, split, lag_featurize,
                                    raw_mts reshape (TSC channel
                                    one-hot per B39 / D-B12.6)
  metrics/                          bootstrap CIs (B13 / B21 BCa)
  experiments/                      raw_loss, ensemble, training_time,
                                    hpo_uplift, ensemble_lift
  adapters/                         seq_sklearn / gbm / tsc families
  report/                           Markdown rendering + bootstrap
                                    rollup aggregators
  datasets/                         synthetic_panel, c_mapss_fd001,
                                    uea_mtsc, amex_default
```

Tests mirror the src layout under `tests/{unit,integration,e2e,deploy,
perf,snapshot}/`.

## A2: Class hierarchy

```
BaseSequenceEstimator (sklearn.base.BaseEstimator)
├── BaseSequenceClassifier (sklearn.base.ClassifierMixin)
│   └── TransformerSequenceEstimator (mixin) + _TFTEstimatorMixin
│       └── TFTClassifier
└── BaseSequenceRegressor (sklearn.base.RegressorMixin)
    └── TransformerSequenceEstimator (mixin) + _TFTEstimatorMixin
        └── TFTRegressor

RecurrentSequenceEstimator (abstract, INTERNAL in v1)
```

The transformer / recurrent split lives one level above the concrete
TFT classes so v2 models (PatchTST, TimesNet, TST) and v3 models
(LSTM, GRU, LSTM-FCN) plug in without restructuring.

## A3: Public-API surface

```python
# src/seq_sklearn/__init__.py
from seq_sklearn.models.transformer.tft.classifier import TFTClassifier
from seq_sklearn.models.transformer.tft.regressor import TFTRegressor
from seq_sklearn.data.tabular_to_sequence import TabularToSequence
from seq_sklearn.model_selection.split import EntityTimeSeriesSplit
from seq_sklearn.config.tft import TFTConfig
from seq_sklearn.config.tabular import TabularToSequenceConfig
from seq_sklearn.config.adapters import (
    TabularConfigParams, OptimizerParams, SchedulerParams,
    LossParams, SamplerParams, TFTAdvancedParams,
)
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
    "TabularConfigParams", "OptimizerParams", "SchedulerParams",
    "LossParams", "SamplerParams", "TFTAdvancedParams",
    "EntityTimeSeriesSplit",
    "HardwareTier", "detect",
    "SeqSklearnError", "ConfigError", "DataContractError",
    "TrainingError", "PredictionError", "NotFittedError",
    "AttentionOutput", "RegressionAttentionOutput",
    "suggest_params", "optuna_trial_guard",
]
```

The six `*Params` adapter classes are the sklearn-compatible
nested-config surface (A4 step 3) and are STABLE in v1: they are the
only way to construct a configured estimator (`tabular_config=...`,
etc.). Everything else (the family pydantic sub-configs at
`seq_sklearn.config.{optimizer,scheduler,loss,sampler}`, anything
reachable only via an underscore-prefixed module) is INTERNAL per
the requirements stability rules. The `RecurrentSequenceEstimator`
class and `RecurrentSequenceEstimatorConfig` (A6.1) are INTERNAL-tier
in v1 by design and are absent from this re-export list; v3 promotes
both to STABLE.

## A4: Configuration schemas

Four tiers (validated by the F5 validity matrix in
`config/_validity.py`):

1. **Family sub-configs** (`config/{optimizer,scheduler,loss,sampler}.py`):
   frozen pydantic models, one per family of choice. INTERNAL.
2. **Main configs** (`config/{tabular,tft}.py`): compose the family
   sub-configs into a single estimator config. STABLE.
3. **Adapter classes** (`config/adapters.py`): sklearn-compatible
   `BaseEstimator` adapters mirroring each pydantic config, so an
   estimator accepts `tabular_config=TabularConfigParams(...)` instead
   of forcing the caller to import pydantic. STABLE; these are the
   only path for constructing configured estimators.
4. **Advanced sub-configs** (`{Model}AdvancedConfig`): rarely-tuned
   knobs (warmup ratios, attention dropout splits). STABLE but
   opt-in via `advanced=...`.

`extra` escape hatch on the main configs accepts unknown keys with a
deprecation warning, promoting load-bearing ones to first-class
slots in subsequent minor releases.

## A5: Data pipeline

`TabularToSequence` (sklearn-compatible transformer) consumes an F2
panel `DataFrame` `(id_col, time_col, *feature_cols, target_col)` and
emits a tensor batch + a `padding_mask`. Periods within each entity
are ordinal (consecutive rows = consecutive periods). The transformer
fits a `CategoricalEncoder` (sorted-unique → int) and per-real-column
scalers, persists them in `categorical_encoder_` and `scaler_` so
`save` / `load` round-trip. `compute_three_way_split` carves the val
and calibration folds from the train block by time-ordered fraction.

`EntityTimeSeriesSplit` (`model_selection/split.py`) is the
sklearn-compatible CV splitter: expanding-window over time per entity
with no future-period leakage. Per-entity min-rows floor is
`n_splits + 1 + gap + lookback - 1`; entities below are dropped from
the split with a `UserWarning`.

## A6: TFTBackbone

`models/transformer/tft/backbone.py` composes the standard TFT
block pipeline against the canonical Lim et al. (arXiv:1912.09363) +
Google Research TF1 reference: static encoders → static VSN → four
context GRNs `(c_s, c_h, c_c, c_e)` → past VSN gated by `c_s` →
`LSTM(x, (c_h, c_c))` over packed variable-length sequences →
post-LSTM GLU + AddNorm against the pre-LSTM VSN output → `c_e`
enrichment GRN → interpretable shared-V self-attention → post-
attention GLU + AddNorm → position-wise FFN GRN + GLU + AddNorm →
readout. Blocks are in `blocks.py` (GLU, AddNorm, GRN, VSN); the
interpretable attention is in `transformer/_interpretable_attention.py`.

### A6.1: Recurrent family skeleton

`models/recurrent/_base.py` ships an abstract `RecurrentSequenceEstimator`
sufficient to land the v3 LSTM/GRU/LSTM-FCN classes without library-
internal churn. Class is INTERNAL in v1 and absent from A3.

## A7: Training pipeline

`training/trainer.py` wraps `pl.Trainer` and accepts the resolved
precision + hardware tier (A11 / N5). `_lightning_module.py` is the
LightningModule that owns step-level logging (A15) and forwards the
backbone output to the head. Callbacks (`callbacks.py`):
`GradScalerWatchdog` (NaN-loss escalation), `EventEmitter` (A15
events), `RngStateCallback` (resume contract).

## A8: Loss factory

`training/losses.py::build_loss(LossConfig, *, task_type, num_classes)`
returns a `torch.nn.Module` from the F5 LOSS_STRATEGIES enum:
`cross_entropy`, `focal`, `mse`, `huber`, `quantile`. The quantile
loss is the canonical pinball-loss family; `quantile_alphas` lives on
the `LossConfig`.

## A9: Calibration pipeline

`calibration/classification.py` ships `TemperatureScaling`, `PlattScaling`,
`IsotonicCalibrator`. `calibration/regression.py` ships `ConformalCalibrator`
and `IsotonicQuantileCalibrator`. Calibration is fit on the held-out
calibration fold the trainer carved via `compute_three_way_split`;
the F5 validity matrix gates `calibration_strategy` against `task_type`.

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
    second parent."""
```

`pydantic.ValidationError` is wrapped in `ConfigError` at the
`_build_config()` call site inside `fit`, not via
`@model_validator(mode="wrap")`.

## A11: Hardware and precision

```python
# src/seq_sklearn/hardware.py
class HardwareTier(IntEnum):
    CPU = 0
    PASCAL = 1
    VOLTA_TURING = 2
    AMPERE_ADA = 3
    HOPPER = 4
    BLACKWELL = 5
```

`detect()` returns the current tier from the CUDA compute capability;
`training/_precision.py::resolve_precision(tier, requested)` selects
the trainer precision (`16-mixed`, `bf16-mixed`, `32-true`) per tier.
The N5 precision matrix is the authoritative table.

## A12: Documentation toolchain

Sphinx + numpydoc + autosummary/autodoc + autodoc-pydantic +
sphinx-gallery + PyData Sphinx Theme + sphinx-copybutton + MyST +
sphinx-sitemap, hosted on Read the Docs. Diátaxis IA:
`tutorial/ how-to/ reference/ explanation/ about/ design/`. Build
gate is `sphinx-build -W --keep-going` in CI; doctest blocks are
executed under `sphinx-build -b doctest`.

Pinned `[docs]` extra (must equal `pyproject.toml [docs]`; checked
by `tests/docs/test_docs_extra_matches_spec.py`):

```toml
docs = [
    "sphinx>=8.1,<9",
    "pydata-sphinx-theme>=0.16,<0.17",
    "numpydoc>=1.8,<2",
    "myst-parser>=4.0,<5",
    "sphinx-copybutton>=0.5,<0.6",
    "sphinx-sitemap>=2.6,<3",
    "sphinx-gallery>=0.18,<0.19",
    "autodoc-pydantic>=2.2,<3",
    "matplotlib>=3.9,<4",
]
```

## A13: Performance baseline plan

`tests/perf/` carries the N7 absolute-budget gates
(`test_n7_absolute.py`) and the relative-baseline tracker. The N7
budgets are listed verbatim in `docs/requirements.md`.

## A14: Testing architecture

Five tiers:
- `tests/unit/` — module-local correctness and contracts.
- `tests/integration/` — multi-module flows (fit → predict round-trip,
  config → estimator construction).
- `tests/e2e/` — end-to-end runs against the synthetic DGP
  (`test_quickstart.py` pins the N1 acceptance threshold).
- `tests/deploy/` — ONNX export + parity (A21).
- `tests/perf/` — A13 / N7 budgets.
- `tests/snapshot/` — pinned numerical snapshots that catch silent
  drift from refactors.

The cross-tier policy gates live under `tests/docs/` (style residue,
toolchain-leftover sweep, snippet-execution ratio).

## A15: Logging and observability

`logging.py::Event` enumerates every named event the library emits
on the `seq_sklearn` logger. The catalog is the authoritative source
for what's observable in production; `docs/reference/observability.md`
documents each event's payload schema. Events fall into three
buckets: `train.*` (per-batch + per-epoch), `predict.*` (per-call,
including degenerate paths), `lifecycle.*` (init, save, load, ONNX
export). Schema is frozen at the BETA tier in v1; payload field
additions are minor-version compatible, removals are major-only.

### A15.1: Inference outputs

`inference/attention.py` ships two frozen dataclasses:

- `AttentionOutput` — variable-selection weights + temporal-attention
  weights returned by `TFTClassifier.predict_with_attention`.
- `RegressionAttentionOutput` — same for `TFTRegressor`.

Both are BETA in v1 (schema may evolve in a minor release with a
deprecation window).

## A16: Optuna integration

`tuning/suggest_params.py::suggest_params(trial, model_class, *,
search_advanced, search_extras)` builds a library-curated search
space per model family. `tuning/pruning.py::optuna_trial_guard` is a
context manager that converts `TrialPruned` to graceful failure and
records the prune reason. Native in-training pruning is wired via the
estimator's `optuna_trial=trial` kwarg (the trainer raises
`optuna.TrialPruned` from inside the callback).

## A17: Save / load format

`serialization.py` writes a directory of `safetensors` + JSON:
`weights.safetensors`, `state.json`, `tabular_to_sequence.json`,
`calibration.json`, `meta.json` (library version, dgp_version, etc.).
The format is language-agnostic and pickle-free; the round-trip
contract is pinned in `tests/integration/test_save_load_roundtrip.py`.

## A18: Dependencies and version pins

Runtime deps (see `pyproject.toml` for current pins): PyTorch 2.x,
pytorch-lightning, pydantic v2, numpy 2.x, pandas 2.x, scikit-learn
≥ 1.6, scipy. Optional `[onnx]` adds onnx + onnxscript. Optional
`[benchmarks]` adds aeon, lightgbm, catboost, xgboost, optuna. Optional
`[docs]` carries the Sphinx stack.

## A19: CI workflow

Per-PR jobs: ruff, pyright, pytest (CPU; full unit + integration +
e2e + deploy + snapshot suite), docs build (`sphinx-build -W
--keep-going` + `-b doctest`). Nightly: `pytest -m gpu` and
`pytest -m slow` against the perf budgets when the runner is the
reference SKU. ONNX deploy job runs `test_restricted_op_surface.py`
+ `test_onnx_parity.py`.

## A20: Open questions

Tracked in `CHANGELOG.md` (per-release) and GitHub issues. Active
deferrals: `D-B38.1` (wire the sufficient-stats bootstrap fast path
into the 5 benchmark aggregators behind a per-experiment flag,
gated on a consumer surfacing the perf ceiling). The benchmark-
suite phase log (`docs/benchmark_suite_phase_log.md`) carries the
full deferral history with merge SHAs.

## A21: ONNX restricted op surface (Phase 10)

requirements.md N1 (ONNX parity) delegates to the architecture
phase: "The architecture phase enumerates the restricted PyTorch op
surface the backbone is allowed to use; ops outside the surface are
caught by a static-analysis check in the deploy job." This
discharges that forward reference. The set below is derived by
STATIC ANALYSIS of the modules the exported graph traces (the
two-step `torch.export(strict=False)` + `torch.onnx.export(
dynamo=True, opset_version=20)` of `_OnnxForward`, i.e.
`head(backbone(batch))`), NOT by running an export. Phase 10's
`tests/deploy/test_restricted_op_surface.py` copies this set
verbatim into `ONNX_SAFE_OPS` and asserts the exported graph's
op_type set (recursing subgraphs) is a subset; growth requires a
deliberate edit here and in that test.

Permitted ONNX op_types:

`{Add, Sub, Mul, Div, Pow, Sqrt, MatMul, Gemm, LSTM, Softmax,
Sigmoid, Tanh, Elu, Erf, ReduceMean, ReduceSum,
LayerNormalization, Gather, GatherND, GatherElements,
ArgMax, Range, Greater, GreaterOrEqual, Less, Where, Expand, Cast,
Concat, Slice, Reshape, Transpose, Squeeze, Unsqueeze, Shape,
Constant, ConstantOfShape, Identity, Flip, Neg, Equal, And, Not,
IsNaN, IsInf, Mod, Split}`

Per-op static-analysis provenance (so each entry is auditable;
verified against the actual exported graph in Phase 10 - the static
guess mis-named some ONNX ops, e.g. `chunk` lowers to `Split` not
`Slice`, and the rank-3 `gather`/scatter lowers to BOTH `GatherND`
and `GatherElements` (form-dependent) but never `ScatterElements`,
which is the kind of discrepancy the deploy test is designed to
surface and which is reconciled here, NOT by silently widening to
an unknown op):

- `nn.Linear` -> `Gemm`/`MatMul`/`Add`; `nn.LayerNorm` ->
  `LayerNormalization`; `nn.Embedding` -> `Gather`; `nn.LSTM` ->
  `LSTM` (the gather-preserving export path keeps the eager
  `nn.LSTM`, only the `pack_padded_sequence` wrapper is dropped).
- GLU/GRN/VSN/AddNorm blocks -> `Sigmoid`/`Tanh`/`Elu`/`Mul`/
  `Add`/`Split`(`chunk`)/`Softmax`/`ReduceSum`/`Reshape`/`Expand`/
  `Concat`/`Unsqueeze`/`Cast`/`Not`/`Slice`.
- Interpretable attention (`_interpretable_attention.py:90-96`,
  the export path; the SDPA fast path is NOT used) ->
  `MatMul`/`Transpose`/`Div`/`Sqrt`/`Where`(masked_fill)/`Softmax`/
  `ReduceMean`; `torch.nan_to_num` (UNCONDITIONAL on the export
  path) -> `IsNaN`/`IsInf`/`Where` (it sanitizes NaN AND +/-inf).
- `_run_lstm` export path (no `argsort`/`sort`: the F3 left-pad
  stable valid-first permutation is the modular ROLL
  `(arange + (L - lengths)) % L`) -> `Range`/`Add`/`Sub`/`Mod`,
  rank-3 `gather`/scatter -> `GatherND`/`GatherElements`, `Not`/`Cast`
  (`~mask`/`.int`), `ReduceSum` (`lengths`).
- `_readout` -> `Flip` (`valid.flip`), `ArgMax`, `Gather`,
  `ReduceSum`/`Div` (mean_pool).
- DELIBERATELY EXCLUDED (their appearance is the regression the
  deploy test must catch): `ScaledDotProductAttention`, `Attention`
  (the SDPA path is off the export graph), `Loop`, `If`, `Scan`
  (the strict=False guard-specialization erases the data-dependent
  control flow), `NonZero`.
