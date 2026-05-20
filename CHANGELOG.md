# Changelog

All notable changes to seq-sklearn are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - YYYY-MM-DD

First stable release. The public API surface (architecture A3) is now
covered by the SemVer stability guarantee. See
[`versioning`](versioning.md) for the per-tier
contract and [`release_checklist`](release_checklist.md)
for the criteria 1-11 release procedure.

### Added

- **F1 / sklearn estimator contract.** `TFTClassifier` and
  `TFTRegressor` (point + quantile) with `fit` / `predict` /
  `predict_proba` / `predict_quantiles` / `predict_with_attention` /
  `score` / `get_params` / `set_params` / `save` / `load` /
  `export_onnx`. Passes the sklearn 1.6
  `parametrize_with_checks` suite with documented expected-fails for
  the panel-shape mismatch points (Phase 9 N1 gate).
- **F2 / `TabularToSequence` preprocessing.** Schema-declarative
  panel → tensor batch transform, shared by every model. Periods are
  ordinal within each entity (consecutive rows = consecutive periods),
  per the F2 contract.
- **F3 / left-padded mask convention.** Variable-length entities are
  left-padded; the `padding_mask` channel is `True` where padded.
- **F4 / safetensors + JSON serialization.** `save` / `load` write a
  language-agnostic, pickle-free format. ONNX is the separate
  cross-runtime export.
- **F5 / resume from checkpoint.** Model weights, optimizer state,
  scheduler state, and RNG state round-trip per the resume contract.
- **F6 / `SyntheticPanelGenerator`.** Deterministic data-generating
  process for tests and examples; the v1 quickstart consumes it.
- **F7 / Optuna integration.** `suggest_params` for library-curated
  search spaces, `optuna_trial_guard` for graceful failure, native
  in-training pruning via `optuna_trial=trial` on the estimator.
- **F8 / hardware detection.** `seq_sklearn.hardware.detect()` and
  `HardwareTier`; the trainer reads the tier and chooses precision
  defaults automatically.
- **F9 / `EntityTimeSeriesSplit`.** Per-entity time-expanding-window
  cross-validation, sklearn-compatible, that does not leak future
  periods across folds (the F2 multi-entity-random-split risk
  reconciled).
- **F10 / calibration.** Temperature/isotonic calibration on a
  held-out fold; conformal correction for the quantile regressor.
- **F11 / structured event logging.** Named, schema-stable events on
  the `seq_sklearn` logger; the catalog is in
  [`reference/observability`](../reference/observability.md).
- **N1 / quickstart in CI.** `tests/e2e/test_quickstart.py` imports
  `examples/quickstart.py` and asserts the binary-classifier
  three-seed median accuracy threshold (≥ 0.75).
- **N5 / hardware-and-precision matrix.** Strict-mode determinism
  contract; the canonical reference is at
  [`reference/determinism`](../reference/determinism.md).
- **N6 / docs site.** Sphinx + numpydoc + autosummary +
  autodoc-pydantic + sphinx-gallery + PyData Sphinx Theme + MyST,
  hosted on Read the Docs (Phase 12 R1 user-ratified stack).
  Diátaxis IA: tutorial / how-to / reference / explanation / about /
  design. Every documentation code snippet is CI-executed (Phase 12
  R3, four mechanisms).
- **`seq_sklearn` package façade.** The architecture-A3 public-API
  re-export block is wired in `__init__.py`; `__version__` reads
  from `importlib.metadata.version("seq-sklearn")` so the library
  version is single-sourced from `pyproject.toml`.
  `tests/unit/test_public_api_surface.py` owns the façade going
  forward (Phase 12 R13/PE.4; the unowned Phase 8↔12 seam that hid
  the missing re-export for 11 phases now has a CI gate). The S7-R1
  pass extended the façade from 18 to 24 names by promoting the six
  `*Params` adapter classes to STABLE; they are the only documented
  way to construct a configured estimator (`tabular_config=...`,
  `optimizer=...`, etc.) and so belong on the public surface.
  Module renamed from `_adapters.py` to `adapters.py` accordingly.
- **24 STABLE public names** at `seq_sklearn.*` after Phase 12 S7-R1:
  the 18 originals plus `TabularConfigParams`, `OptimizerParams`,
  `SchedulerParams`, `LossParams`, `SamplerParams`, and
  `TFTAdvancedParams` (`docs/reference/api.md` autoclasses each).
- **Phase-0 scaffold.** `pyproject.toml`, src/tests tree, ruff +
  pyright + pytest config, `.github/workflows/pr.yml` and
  `nightly.yml`, `scripts/check_snapshots.sh`,
  `scripts/check_perf_baselines.sh`, Apache-2.0 LICENSE,
  CONTRIBUTING, SECURITY, CODE_OF_CONDUCT, GitHub templates.
- **Python support: 3.12, 3.13, 3.14** (matches N3 "three
  most-recent releases" rule; 3.11 dropped before v1 ships).

### N7 absolute-budget validation (criterion 9)

The four N7 numbers below are recorded at release time from two
manual runs per [the release checklist](release_checklist.md):
`SEQ_SKLEARN_N7_GPU=1 pytest -m "gpu and slow" tests/perf/test_n7_absolute.py`
on an A100/T4/4090 for the three GPU/training numbers, and
`SEQ_SKLEARN_N7_CPU=1 pytest -m slow tests/perf/test_n7_absolute.py::test_n7_cpu_inference_latency`
on the release-reference CPU for the inference-latency number.
Both env-var gates exist so the strict per-batch budgets never
assert incidentally on a non-reference device. Each `*_INFER_MS`
budget is the *whole 1024-window batch* (not per-sample), matching
the requirements wording verbatim.

- GPU peak memory: TBD (target `< 8 GB`)
- Training wall-clock: TBD (target `< 30 min`)
- Inference latency, GPU: TBD (target `< 10 ms` per 1024-window batch)
- Inference latency, CPU: TBD (target `< 100 ms` per 1024-window batch)

### Governance

- Phase 9 (`check_estimator` + acceptance thresholds): merged.
- Phase 10 (ONNX export + deploy tests): merged.
- Phase 11 (perf-baseline regression gate): merged.
- Phase 12 (docs + release prep): 4-round design-review consensus +
  R13/PE.4 delta + Claude-only S7 consensus (Gemini deferred per
  user direction).
