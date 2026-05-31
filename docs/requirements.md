# seq-sklearn requirements (v1)

This document is the v1 contract: scope, stability tiers, and the
per-section requirements (`F1`-`F11` functional, `N1`-`N7`
non-functional) that source-code module docstrings cite. The
`N7: Performance budgets` section is preserved verbatim and is
the authoritative reference for the budgets `tests/perf/
test_n7_absolute.py` enforces.

For the technical architecture (`A1`-`A21`), see
`docs/architecture.md`. For the live API reference, see
`docs/reference/`. For benchmark-suite requirements, see
`docs/benchmark_suite_design.md`.

## Scope

seq-sklearn provides sklearn-compatible deep sequence models for
supervised classification and regression on multivariate
time-series panels. v1 ships a Temporal Fusion Transformer (Lim
et al., 2021) adapted to supervised tasks: classifier + regressor
(point and quantile), the full `fit` / `predict` / `predict_proba`
contract, sklearn pipeline-compatibility, calibrated probabilities,
conformal quantile regression, ONNX export, and the variable-
selection + temporal-attention surfaces as typed dataclasses.

## Non-goals

- Multi-horizon forecasting (the canonical TFT use case). v1 is
  supervised classification + standard regression only.
- Probabilistic forecasting with prediction intervals over a
  forecast horizon. v1 quantile regression is a single-step
  contemporaneous quantile head.
- Unsupervised representation learning, anomaly detection,
  changepoint detection.
- Foundation-model-style pre-training. v1 trains from scratch on
  the caller's data.

## Roadmap

| Version | Scope |
|---|---|
| **v1** | TFT classifier + regressor; full `TabularToSequence` pipeline; Optuna; all library-wide infrastructure detailed in this doc |
| v1.1 | Multi-output regression, multi-label classification (architectural constraints already in v1) |
| v2 | PatchTST, TimesNet, TST (transformer family completion) |
| v3 | LSTM, GRU, LSTM-FCN (recurrent family) |

Versions are additive. v2 does not require waiting for v3; v3 does
not require waiting for v2. The library-wide infrastructure (sklearn
API, `TabularToSequence`, training pipeline, Optuna integration,
calibration, hardware/precision policy, error contract, repo
hygiene) is built ONCE in v1 and reused by every subsequent model.

## Architectural philosophy

Three layers, separated cleanly: (1) a sklearn-facing estimator
contract (`fit` / `predict` / `predict_proba` / `predict_quantiles`
/ `score` / `get_params` / `set_params` / `save` / `load`); (2) a
pydantic-validated config layer (`config/`); (3) a torch-side
backbone + Lightning trainer. The estimator wraps the trainer; the
trainer wraps the backbone; the config layer crosses all three.

## Versioning and stability

Semantic versioning strictly. `MAJOR.MINOR.PATCH`:

- **MAJOR** on any breaking change to the public API surface
  (renamed or removed public class, removed public method, narrowed
  argument type, new required argument).
- **MINOR** on additive changes (new model class, new method, new
  argument with a default, new acceptable value in a `Literal`).
  Behavior-changing default values bump MINOR with a CHANGELOG entry.
- **PATCH** on bug fixes that do not change behavior beyond fixing
  the bug.

The **public API** is exactly what `seq_sklearn/__init__.py`
re-exports (the architecture A3 `__all__`), plus what's documented
under `docs/reference/`, plus module attributes reached without a
leading underscore in the import path. Anything reachable only
through an underscore-prefixed module is internal.

### Per-module stability tiers (v1)

| Tier | Module / Symbol | Notes |
|---|---|---|
| STABLE | `TFTClassifier`, `TFTRegressor`: `fit`, `predict`, `predict_proba`, `predict_quantiles`, `score`, `get_params`, `set_params`, `save`, `load` | sklearn-contract methods; breaking change requires MAJOR |
| STABLE | `TabularToSequence`: fit/transform/inverse_transform | |
| STABLE | `seq_sklearn.hardware.detect`, `HardwareTier` | enum values may be added; existing values stable |
| STABLE | `seq_sklearn.model_selection.EntityTimeSeriesSplit` | |
| STABLE | `seq_sklearn.config.adapters.{TabularConfigParams, OptimizerParams, SchedulerParams, LossParams, SamplerParams, TFTAdvancedParams}` | sklearn-compatible nested-config adapters (architecture A4 step 3); the only way to construct a configured estimator. Field additions follow the F7 ALPHA→BETA→STABLE promotion path |
| BETA | `TFTClassifier.export_onnx`, `TFTRegressor.export_onnx` | depends on `[onnx]` extra; export shape may evolve |
| BETA | `predict_with_attention`, `AttentionOutput`, `RegressionAttentionOutput` | fields may be added in MINOR releases; consult attribute access, not tuple position |
| ALPHA | `seq_sklearn.tuning.suggest_params` default search space | search-space defaults may change without MINOR bump; pass an explicit search space for stable behavior |
| INTERNAL | `seq_sklearn._*` modules | not part of the public API |
| INTERNAL | `seq_sklearn.config.{optimizer, scheduler, loss, sampler}` family pydantic sub-config modules | frozen configs reached via the adapter `.to_pydantic()` call. The STABLE surface is the `*Params` adapters at `seq_sklearn.config.adapters` |

### Per-hyperparameter stability tiers

| Tier | Location | Removal / change cost |
|---|---|---|
| STABLE | Typed field on main config or family sub-config | MAJOR bump required |
| BETA | Typed field on `<Model>AdvancedConfig` or freshly-promoted family sub-config field | MINOR bump + `DeprecationWarning` cycle |
| ALPHA | Entry in a sub-config's `extra: dict[str, ...]` escape hatch | No version bump; CHANGELOG entry only |

### Deprecation policy

Removal of public functionality requires at least one MINOR release
emitting `DeprecationWarning` before removal. The warning message
names the replacement.

## Data shape

One row per entity per period. Mandatory columns: `id_col` (entity),
`time_col` (period; ordinal integer or sortable timestamp),
`target_col`. Feature columns split into real and categorical; both
are passed via the `TabularToSequence` config.

## Functional requirements (library-wide)

### F1: sklearn-compatible estimator contract

The estimator implements the sklearn 1.6+ tagged-contract surface
the panel shape allows: `fit(X, y)`, `predict(X)`,
`predict_proba(X)` (classifier), `predict_quantiles(X)` (regressor
with quantile loss), `score(X, y)`, `get_params(deep=True)`,
`set_params(**kwargs)`, `save(path)`, `load(path)`, plus the
introspection surface `predict_with_attention(X)` returning a frozen
`AttentionOutput` / `RegressionAttentionOutput` (BETA).

Caller-input-row-order restore: every prediction surface returns
rows in the caller's input `X` row order, not the transform's
internal `(id, time)` sort order. Pinned by
`tests/integration/test_predict_row_order.py`.

### F1.1: sklearn fit-state attributes and tags

Trailing-underscore fit-state attrs (`classes_`, `n_classes_`,
`feature_names_in_`, `n_features_in_`); `__sklearn_tags__` exposes
the panel-specific signal (input is a panel `DataFrame`, not a 2-D
array). The pinned `check_estimator` subset is enumerated in the
F1.1 test under `tests/unit/`.

### F2: Input data contract

The panel `DataFrame` schema:

- Index: arbitrary (the F1 row-order restore handles non-default).
- Columns: `(id_col, time_col, *feature_real_cols, *feature_categorical_cols, target_col)` exactly per the config.
- Periods within each entity are ordinal: consecutive rows for the
  same entity = consecutive periods (the transform DOES NOT
  resample irregular time).
- Variable-length entities are accepted; the F3 left-pad convention
  handles padding semantics.

Data-contract violations raise `DataContractError`.

### F3: TabularToSequence preprocessing

Variable-length entities are **left-padded** to the configured
`lookback`. `padding_mask` is `True` where padded. Real features
pass through a per-column scaler (`StandardScaler` default);
categorical features pass through `CategoricalEncoder` (sorted-
unique → int). State (`scaler_`, `categorical_encoder_`,
`cardinalities_`) is persisted at fit so `save` / `load` round-trips
to identical predictions.

### F4: Model abstraction

Backbones expose a uniform `forward(batch) -> BackboneOutput`
contract. The `BackboneOutput` dataclass carries the latent
representation + family-specific introspection state (TFT carries
variable-selection weights + attention scores). Heads
(`ClassificationHead`, `RegressionHead`) map the backbone output
to the target. Every `nn.Linear`, `nn.LayerNorm`, `nn.Embedding` is
built through the F4 layer factory so weight init is deterministic
and `save` / `load` is byte-stable.

### F5: Training pipeline

Trainer wraps `pl.Trainer`. The F5 validity matrix
(`config/_validity.py`) gates the cross-product of
`(task_type, loss_strategy, imbalance_strategy, calibration_strategy)`
cells: every combination is either ALLOWED (exhaustively tested) or
REJECTED with a typed `ConfigError` at fit-time. Loss factories,
optimizer factories, scheduler factories, and sampler builders all
dispatch off the matrix.

### F6: Synthetic data generators

`SyntheticPanelGenerator` is the deterministic DGP for tests +
examples. Byte-reproducible per seed; the `dgp_version` (currently
`1`) pins the structural sampling order. F1.1 / N1 acceptance
thresholds are pinned per `dgp_version`; a procedural change bumps
the version.

### F7: Hyperparameter tuning compatibility (Optuna first-class)

`suggest_params(trial, model_class, *, search_advanced, search_extras)`
builds a library-curated search space. The estimator accepts
`optuna_trial=trial` for in-training pruning; `optuna_trial_guard`
is the context manager that turns `TrialPruned` into a graceful
failure with the prune reason recorded.

### F8: Error contract

The architecture A10 hierarchy is the single source of truth for
exception types. Every internal failure raises one of
`{ConfigError, DataContractError, TrainingError, PredictionError,
NotFittedError}`; uncaught exceptions are a CI failure.

### F9: Numerical contracts

`tests/snapshot/` carries pinned numerical snapshots that catch
silent drift from refactors. The strict-determinism toggle
(`enable_strict_mode()` per N4) makes every snapshot reproducible
across runs.

### F10: Cross-validation strategy

`EntityTimeSeriesSplit` (sklearn-compatible) is the canonical CV
splitter: per-entity expanding-window over time, no future-period
leakage. Used by both the library's calibration carve-out and the
benchmark suite's outer CV.

### F11: Logging strategy

Structured events on the `seq_sklearn` logger via `Event` enum.
Schema is documented in `docs/reference/observability.md`; event
payloads are JSON-serializable. Schema is BETA in v1.

## Non-functional requirements

### N1: Testing

The `tests/` tree mirrors `src/`. `tests/e2e/test_quickstart.py`
imports `examples/quickstart.py` and asserts the binary classifier
three-seed median accuracy threshold (≥ 0.75 per the
F6-pinned DGP). `tests/deploy/` discharges the ONNX-parity
requirement: an ONNX export round-trip yields predictions within
the documented tolerance. Coverage gates: 85% line, 80% branch on
the non-slow / non-perf subset.

### N2: CI and review automation

Per-PR: ruff, pyright strict, pytest (CPU full unit + integration +
e2e + deploy + snapshot suite), docs build (`sphinx-build -W
--keep-going` + `-b doctest`). Nightly: `pytest -m gpu` + `pytest -m
slow` against perf budgets on the reference SKU. The dual-model
swarm review loop is invoked via `/design-review`, `/review`, and
`/gemini-final-pass` per `CLAUDE.md`.

### N3: Repository hygiene (open-source standard)

`README.md`, `LICENSE`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
`SECURITY.md`, `CHANGELOG.md` at repo root. `pyproject.toml` is the
single source of dep + extras + version. `.gitignore` covers
`_build/`, `examples_gallery/`, `.pytest_cache/`, etc.

### N4: Reproducibility

`enable_strict_mode()` sets every torch / numpy / python RNG to a
deterministic state and disables cudnn benchmarking. The strict-mode
contract is pinned by `tests/integration/test_strict_determinism.py`:
two runs with the same seed produce byte-identical predictions.

### N5: Hardware and precision

`HardwareTier` (architecture A11) enumerates `{CPU, PASCAL,
VOLTA_TURING, AMPERE_ADA, HOPPER, BLACKWELL}`. The N5 precision
matrix pins per-tier defaults:

| Tier | Default precision |
|---|---|
| CPU | `32-true` |
| PASCAL | `32-true` (no native fp16 / bf16) |
| VOLTA_TURING | `16-mixed` |
| AMPERE_ADA | `bf16-mixed` |
| HOPPER | `bf16-mixed` |
| BLACKWELL | `bf16-mixed` |

The trainer honors `requested` precision when valid for the tier
and warns + downgrades otherwise.

### N6: Documentation

Sphinx + numpydoc + autosummary/autodoc + autodoc-pydantic +
sphinx-gallery + PyData Sphinx Theme + MyST, on Read the Docs.
Diátaxis IA: `tutorial / how-to / reference / explanation / about /
design`. Every documentation code snippet is CI-executed.

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
2. `pytest -m "not slow and not perf"` passes with the 85% line /
   80% branch coverage gates.
3. `tests/deploy/` smoke test passes against the built wheel.
4. The `/design-review` loop has reached consensus on changes since
   the previous release.
5. The `style-reviewer` agent reports zero CRITICAL findings.
6. `CHANGELOG.md` is updated.
7. A release-candidate wheel installs from TestPyPI and runs a
   minimal end-to-end script.

### v1-specific criteria (TFT release)

8. All F1-F11 requirements are implemented and tested.
9. All N1-N7 requirements are met. N7's absolute budgets are
   discharged by `tests/perf/test_n7_absolute.py` (marked `gpu`
   + `slow`, excluded from PR and nightly CPU/GPU CI), run manually
   on an A100/T4/4090 as a release-checklist step.
10. Two quickstart examples exist and pass in CI:
    - A binary classifier on synthetic monthly data recovers
      accuracy ≥ 0.75 on the three-seed median (N1).
    - A quantile regressor recovers empirical coverage on the
      nominal 80% interval in `[0.75, 0.85]` after conformal
      calibration.
11. The Gemini final-pass review against this requirements doc and
    the architecture doc surfaces no new CRITICAL findings (for
    v1.0.0 this was deferred per the Claude-only consensus, per the
    release checklist note).
