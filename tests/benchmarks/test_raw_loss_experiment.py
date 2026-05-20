"""Phase B5 raw-loss experiment e2e tests.

Drives `run_raw_loss` against a synthetic 2-dataset x 2-model
matrix using fake adapters whose `fit` and `predict` are constant-
output stubs, so the test does NOT require torch / GPU. The asserts
cover the B5 + B7.2 contracts:

- Every (dataset, model, seed, fold) cell produces a shard + sentinel
  (with metric columns populated on success or `skipped_reason`
  populated on skip).
- A second `run_raw_loss` invocation against the same `output_root`
  skips every already-complete cell via the sentinel; `cells_attempted`
  is 0 on the second call.
- A task-type-mismatch cell (binary dataset / regression model) emits
  a `task_type_mismatch` skip reason and no metrics.
- A `regression_quantile` cell emits the B5-followup deferral skip
  reason.
- An adapter that raises at `fit` produces a `skipped_reason` row
  but the rest of the run continues.
- The manifest concatenates back via `load_run` and the leaderboard
  renderer produces non-empty Markdown.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np
import pandas as pd
import pytest
from benchmarks.adapters._base import (
    ProbaUnsupportedError,
    QuantilesUnsupportedError,
    SeqSklearnAdapter,
)
from benchmarks.config import (
    BenchmarkConfig,
    DatasetSpec,
    ExperimentSpec,
    ModelSpec,
    TaskType,
)
from benchmarks.datasets._base import PanelDataset
from benchmarks.experiments import (
    RawLossExperimentResult,
    build_run_environment,
    run_raw_loss,
)
from benchmarks.manifest import (
    is_cell_complete,
    list_completed_keys,
    load_run,
)
from benchmarks.registry import (
    register_adapter_factory,
    register_dataset,
    register_model,
)
from benchmarks.report.raw_loss import (
    rank_by_primary_loss,
    render_leaderboard_markdown,
)

_ZERO_SHA = "0" * 64


# --- Fake datasets ----------------------------------------------------------


def _make_binary_panel(n_entities: int = 4, n_periods: int = 10) -> PanelDataset:
    """Synthetic 4 entity x 10 period binary classification panel.

    Target signal: `y = 1 iff x_feat > 0`. The fake adapter doesn't
    need to learn this; the driver only cares that the metric
    layer receives a coherent (y_true, y_pred, y_proba) shape.
    """
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(0)
    for entity in range(n_entities):
        for t in range(n_periods):
            x = float(rng.normal(0.0, 1.0))
            rows.append(
                {
                    "entity_id": f"e_{entity}",
                    "period": t,
                    "x_feat": x,
                    "y": int(x > 0),
                }
            )
    panel = pd.DataFrame(rows)
    spec = DatasetSpec(
        name="fake_binary",
        task_type="binary",
        access_tier="OPEN",
        size_tier="small",
        balance="balanced",
        modality="numeric",
        source_uri="https://example.test/binary.csv",
        integrity_sha256=_ZERO_SHA,
        archive_basename="binary.csv",
        entity_col="entity_id",
        time_col="period",
        target_col="y",
        feature_real_cols=("x_feat",),
        feature_categorical_cols=(),
        lookback=2,
        positive_label=1,
        citation="synthetic; test only",
    )
    return PanelDataset(spec=spec, panel=panel, y=panel["y"].to_numpy())


def _make_regression_panel(n_entities: int = 4, n_periods: int = 10) -> PanelDataset:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(1)
    for entity in range(n_entities):
        for t in range(n_periods):
            x = float(rng.normal(0.0, 1.0))
            rows.append(
                {
                    "entity_id": f"e_{entity}",
                    "period": t,
                    "x_feat": x,
                    "y": 2.0 * x + 0.5,
                }
            )
    panel = pd.DataFrame(rows)
    spec = DatasetSpec(
        name="fake_regression_point",
        task_type="regression_point",
        access_tier="OPEN",
        size_tier="small",
        balance="balanced",
        modality="numeric",
        source_uri="https://example.test/reg.csv",
        integrity_sha256=_ZERO_SHA,
        archive_basename="reg.csv",
        entity_col="entity_id",
        time_col="period",
        target_col="y",
        feature_real_cols=("x_feat",),
        feature_categorical_cols=(),
        lookback=2,
        citation="synthetic; test only",
    )
    return PanelDataset(spec=spec, panel=panel, y=panel["y"].to_numpy())


def _make_regression_quantile_panel() -> PanelDataset:
    base = _make_regression_panel()
    spec = base.spec.model_copy(
        update={
            "name": "fake_regression_quantile",
            "task_type": "regression_quantile",
            "archive_basename": "regq.csv",
            "source_uri": "https://example.test/regq.csv",
        }
    )
    return PanelDataset(spec=spec, panel=base.panel, y=base.y)


# --- Fake adapters ----------------------------------------------------------


@dataclass
class _ConstantBinaryAdapter:
    """Predicts the majority class with a constant 0.7/0.3 probability."""

    name: ClassVar[str] = "fake_constant_binary"
    family: ClassVar[str] = "sklearn_passthrough"
    task_types: ClassVar[tuple[TaskType, ...]] = ("binary",)
    supports_proba: ClassVar[bool] = True

    spec: DatasetSpec
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    _majority: int = field(default=0, init=False)

    def fit(self, panel: pd.DataFrame, y: np.ndarray) -> Self:
        # Majority class from training data.
        if len(y) == 0:
            self._majority = 0
        else:
            values, counts = np.unique(y, return_counts=True)
            self._majority = int(values[counts.argmax()])
        return self

    def predict(self, panel: pd.DataFrame) -> np.ndarray:
        return np.full(len(panel), self._majority, dtype=np.int64)

    def predict_proba(self, panel: pd.DataFrame) -> np.ndarray:
        # 0.7 on the predicted majority class; 0.3 on the other.
        proba = np.full((len(panel), 2), 0.3)
        proba[:, self._majority] = 0.7
        return proba

    def predict_quantiles(self, panel: pd.DataFrame) -> np.ndarray:
        raise QuantilesUnsupportedError(
            f"{self.name}: classifier does not produce quantile predictions"
        )


@dataclass
class _ConstantRegressorAdapter:
    """Predicts the training-mean target on every row."""

    name: ClassVar[str] = "fake_constant_regressor"
    family: ClassVar[str] = "sklearn_passthrough"
    task_types: ClassVar[tuple[TaskType, ...]] = ("regression_point",)
    supports_proba: ClassVar[bool] = False

    spec: DatasetSpec
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    _mean: float = field(default=0.0, init=False)

    def fit(self, panel: pd.DataFrame, y: np.ndarray) -> Self:
        self._mean = float(np.mean(y)) if len(y) > 0 else 0.0
        return self

    def predict(self, panel: pd.DataFrame) -> np.ndarray:
        return np.full(len(panel), self._mean, dtype=np.float64)

    def predict_proba(self, panel: pd.DataFrame) -> np.ndarray:
        raise ProbaUnsupportedError(
            f"{self.name}: regression adapter does not produce probabilities"
        )

    def predict_quantiles(self, panel: pd.DataFrame) -> np.ndarray:
        raise QuantilesUnsupportedError(
            f"{self.name}: point-regression adapter does not produce quantiles"
        )


@dataclass
class _CrashingAdapter:
    """Always raises at `fit`; exercises the adapter-error skip path."""

    name: ClassVar[str] = "fake_crashing"
    family: ClassVar[str] = "sklearn_passthrough"
    task_types: ClassVar[tuple[TaskType, ...]] = ("binary",)
    supports_proba: ClassVar[bool] = True

    spec: DatasetSpec
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    def fit(self, panel: pd.DataFrame, y: np.ndarray) -> Self:
        raise RuntimeError("fake_crashing: synthetic adapter failure")

    def predict(self, panel: pd.DataFrame) -> np.ndarray:
        raise RuntimeError("not reached")

    def predict_proba(self, panel: pd.DataFrame) -> np.ndarray:
        raise RuntimeError("not reached")

    def predict_quantiles(self, panel: pd.DataFrame) -> np.ndarray:
        raise QuantilesUnsupportedError(self.name)


@dataclass
class _RuntimeProbaErrorAdapter:
    """Declares supports_proba=True at the spec but raises
    `ProbaUnsupportedError` at runtime.

    Pre-R2 this would abort the run via the narrowed-except chain
    (compute_all raises ValueError on `y_proba=None`); post-R2 the
    typed `probabilities_required_but_runtime_unavailable` skip
    catches it.
    """

    name: ClassVar[str] = "fake_runtime_proba_error"
    family: ClassVar[str] = "sklearn_passthrough"
    task_types: ClassVar[tuple[TaskType, ...]] = ("binary",)
    supports_proba: ClassVar[bool] = True

    spec: DatasetSpec
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    def fit(self, panel: pd.DataFrame, y: np.ndarray) -> Self:
        return self

    def predict(self, panel: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(panel), dtype=np.int64)

    def predict_proba(self, panel: pd.DataFrame) -> np.ndarray:
        raise ProbaUnsupportedError(
            f"{self.name}: runtime ProbaUnsupportedError despite supports_proba=True"
        )

    def predict_quantiles(self, panel: pd.DataFrame) -> np.ndarray:
        raise QuantilesUnsupportedError(self.name)


@dataclass
class _PredictCrashingAdapter:
    """Fits cleanly but raises `RuntimeError` at `predict`."""

    name: ClassVar[str] = "fake_predict_crashing"
    family: ClassVar[str] = "sklearn_passthrough"
    task_types: ClassVar[tuple[TaskType, ...]] = ("binary",)
    supports_proba: ClassVar[bool] = True

    spec: DatasetSpec
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    def fit(self, panel: pd.DataFrame, y: np.ndarray) -> Self:
        return self

    def predict(self, panel: pd.DataFrame) -> np.ndarray:
        raise RuntimeError("fake_predict_crashing: synthetic predict-time crash")

    def predict_proba(self, panel: pd.DataFrame) -> np.ndarray:
        raise RuntimeError("not reached")

    def predict_quantiles(self, panel: pd.DataFrame) -> np.ndarray:
        raise QuantilesUnsupportedError(self.name)


@dataclass
class _NaNRegressorAdapter:
    """Regression adapter that returns nan in the first 2 y_pred rows.

    Exercises the float-y_pred branch of `_strip_below_floor_rows`
    (which `_ConstantRegressorAdapter` doesn't reach because its
    predictions are all-finite).
    """

    name: ClassVar[str] = "fake_nan_regressor"
    family: ClassVar[str] = "sklearn_passthrough"
    task_types: ClassVar[tuple[TaskType, ...]] = ("regression_point",)
    supports_proba: ClassVar[bool] = False

    spec: DatasetSpec
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    _mean: float = field(default=0.0, init=False)

    def fit(self, panel: pd.DataFrame, y: np.ndarray) -> Self:
        self._mean = float(np.mean(y)) if len(y) > 0 else 0.0
        return self

    def predict(self, panel: pd.DataFrame) -> np.ndarray:
        out = np.full(len(panel), self._mean, dtype=np.float64)
        out[: min(2, len(panel))] = np.nan
        return out

    def predict_proba(self, panel: pd.DataFrame) -> np.ndarray:
        raise ProbaUnsupportedError(self.name)

    def predict_quantiles(self, panel: pd.DataFrame) -> np.ndarray:
        raise QuantilesUnsupportedError(self.name)


@dataclass
class _NotFittedAdapter:
    """Always raises `NotFittedError` at `predict`.

    Confirms the R2 narrowed except includes `NotFittedError` and
    routes the cell through the adapter-error skip path.
    """

    name: ClassVar[str] = "fake_not_fitted"
    family: ClassVar[str] = "sklearn_passthrough"
    task_types: ClassVar[tuple[TaskType, ...]] = ("binary",)
    supports_proba: ClassVar[bool] = True

    spec: DatasetSpec
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    def fit(self, panel: pd.DataFrame, y: np.ndarray) -> Self:
        return self

    def predict(self, panel: pd.DataFrame) -> np.ndarray:
        from seq_sklearn import NotFittedError as _NotFittedError

        raise _NotFittedError("fake_not_fitted: synthetic NotFittedError")

    def predict_proba(self, panel: pd.DataFrame) -> np.ndarray:
        raise RuntimeError("not reached")

    def predict_quantiles(self, panel: pd.DataFrame) -> np.ndarray:
        raise QuantilesUnsupportedError(self.name)


@dataclass
class _QuantileRegressorAdapter:
    """Declares `regression_quantile` so the B5-followup skip fires."""

    name: ClassVar[str] = "fake_quantile_regressor"
    family: ClassVar[str] = "sklearn_passthrough"
    task_types: ClassVar[tuple[TaskType, ...]] = (
        "regression_point",
        "regression_quantile",
    )
    supports_proba: ClassVar[bool] = False

    spec: DatasetSpec
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    def fit(self, panel: pd.DataFrame, y: np.ndarray) -> Self:
        return self

    def predict(self, panel: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(panel), dtype=np.float64)

    def predict_proba(self, panel: pd.DataFrame) -> np.ndarray:
        raise ProbaUnsupportedError(self.name)

    def predict_quantiles(self, panel: pd.DataFrame) -> np.ndarray:
        return np.zeros((len(panel), 3), dtype=np.float64)


@dataclass
class _NoProbaClassifierAdapter:
    """Classifier-applicable but supports_proba=False (KNN-DTW shape)."""

    name: ClassVar[str] = "fake_no_proba_classifier"
    family: ClassVar[str] = "tsc"
    task_types: ClassVar[tuple[TaskType, ...]] = ("binary", "multiclass")
    supports_proba: ClassVar[bool] = False

    spec: DatasetSpec
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    def fit(self, panel: pd.DataFrame, y: np.ndarray) -> Self:
        return self

    def predict(self, panel: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(panel), dtype=np.int64)

    def predict_proba(self, panel: pd.DataFrame) -> np.ndarray:
        raise ProbaUnsupportedError(self.name)

    def predict_quantiles(self, panel: pd.DataFrame) -> np.ndarray:
        raise QuantilesUnsupportedError(self.name)


@dataclass
class _NaNProbaAdapter:
    """Returns nan rows in y_proba for the first 2 test rows."""

    name: ClassVar[str] = "fake_nan_proba_classifier"
    family: ClassVar[str] = "seq_sklearn"
    task_types: ClassVar[tuple[TaskType, ...]] = ("binary",)
    supports_proba: ClassVar[bool] = True

    spec: DatasetSpec
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    _majority: int = field(default=0, init=False)

    def fit(self, panel: pd.DataFrame, y: np.ndarray) -> Self:
        if len(y) > 0:
            values, counts = np.unique(y, return_counts=True)
            self._majority = int(values[counts.argmax()])
        return self

    def predict(self, panel: pd.DataFrame) -> np.ndarray:
        return np.full(len(panel), self._majority, dtype=np.int64)

    def predict_proba(self, panel: pd.DataFrame) -> np.ndarray:
        proba = np.full((len(panel), 2), 0.3)
        proba[:, self._majority] = 0.7
        # Inject nan in the first 2 rows: the harness must strip
        # them before the metric layer.
        proba[: min(2, len(panel)), :] = np.nan
        return proba

    def predict_quantiles(self, panel: pd.DataFrame) -> np.ndarray:
        raise QuantilesUnsupportedError(self.name)


@dataclass
class _HarnessBugAdapter:
    """Raises `ValueError` (a harness-bug signal) at fit time.

    Pre-R1, this would be swallowed as "adapter_error"; post-R1, the
    narrowed except passes ValueError through so a config / data
    drift surfaces loudly.
    """

    name: ClassVar[str] = "fake_harness_bug"
    family: ClassVar[str] = "sklearn_passthrough"
    task_types: ClassVar[tuple[TaskType, ...]] = ("binary",)
    supports_proba: ClassVar[bool] = True

    spec: DatasetSpec
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    def fit(self, panel: pd.DataFrame, y: np.ndarray) -> Self:
        raise ValueError("fake_harness_bug: ValueError must propagate")

    def predict(self, panel: pd.DataFrame) -> np.ndarray:
        raise RuntimeError("not reached")

    def predict_proba(self, panel: pd.DataFrame) -> np.ndarray:
        raise RuntimeError("not reached")

    def predict_quantiles(self, panel: pd.DataFrame) -> np.ndarray:
        raise QuantilesUnsupportedError(self.name)


# --- Fixtures ----------------------------------------------------------------


def _register_fake_datasets(panels: list[PanelDataset]) -> None:
    for panel_dataset in panels:

        def _loader(_root: Path, _captured: PanelDataset = panel_dataset) -> PanelDataset:
            return _captured

        register_dataset(panel_dataset.spec, _loader)


def _register_fake_models() -> None:
    for adapter_cls in (
        _ConstantBinaryAdapter,
        _ConstantRegressorAdapter,
        _CrashingAdapter,
        _QuantileRegressorAdapter,
        _NoProbaClassifierAdapter,
        _NaNProbaAdapter,
        _HarnessBugAdapter,
        _RuntimeProbaErrorAdapter,
        _PredictCrashingAdapter,
        _NaNRegressorAdapter,
        _NotFittedAdapter,
    ):
        spec = ModelSpec(
            name=adapter_cls.name,
            family=adapter_cls.family,
            task_types=adapter_cls.task_types,
            supports_proba=adapter_cls.supports_proba,
            reason="synthetic; test fixture only",
        )
        register_model(spec)

        def _factory(
            *,
            spec: DatasetSpec,
            hyperparameters: dict[str, Any] | None = None,
            _cls: type = adapter_cls,
        ) -> SeqSklearnAdapter:
            return _cls(spec=spec, hyperparameters=hyperparameters or {})

        register_adapter_factory(adapter_cls.name, _factory)


@pytest.fixture
def fake_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> list[PanelDataset]:  # pyright: ignore[reportUnusedFunction]
    """Register the synthetic datasets + models for the test run.

    `isolated_registry` (autouse) restores the registries on teardown.
    """
    del monkeypatch  # placeholder for future env knobs
    panels = [
        _make_binary_panel(),
        _make_regression_panel(),
        _make_regression_quantile_panel(),
    ]
    _register_fake_datasets(panels)
    _register_fake_models()
    return panels


def _make_config(
    *,
    datasets: tuple[str, ...],
    models: tuple[str, ...],
    seeds: tuple[int, ...],
    output_dir: Path,
    cache_dir: Path,
) -> BenchmarkConfig:
    return BenchmarkConfig(
        datasets=datasets,
        models=models,
        experiments=(ExperimentSpec(kind="raw_loss", seeds=seeds),),
        output_dir=output_dir,
        cache_dir=cache_dir,
    )


# --- Tests -------------------------------------------------------------------


def test_run_raw_loss_emits_shards_and_sentinels_for_each_cell(
    fake_registry: list[PanelDataset], tmp_path: Path
) -> None:
    del fake_registry
    config = _make_config(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        seeds=(0,),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    result = run_raw_loss(config, output_root=tmp_path / "out", env=env)
    assert isinstance(result, RawLossExperimentResult)
    # The default splitter uses n_splits=5; the panel has 10 periods
    # per 4 entities so every fold has rows.
    assert result.cells_attempted == 5
    keys = list_completed_keys(tmp_path / "out")
    assert len(keys) == 5
    manifest = load_run(tmp_path / "out")
    assert len(manifest) == 5
    # The metrics column is populated (constant adapter -> log_loss is
    # finite but non-trivial).
    assert bool(manifest["log_loss"].notna().all())
    assert bool(manifest["skipped_reason"].isna().all())


def test_run_raw_loss_is_resumable_via_sentinel_check(
    fake_registry: list[PanelDataset], tmp_path: Path
) -> None:
    del fake_registry
    config = _make_config(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        seeds=(0,),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    first = run_raw_loss(config, output_root=tmp_path / "out", env=env)
    assert first.cells_attempted == 5
    # Re-run; every cell is already complete via the sentinel and the
    # adapter is NOT invoked again.
    second = run_raw_loss(config, output_root=tmp_path / "out", env=env)
    assert second.cells_attempted == 0
    assert second.cells_already_complete == 5


def test_task_type_mismatch_emits_typed_skip_reason(
    fake_registry: list[PanelDataset], tmp_path: Path
) -> None:
    del fake_registry
    # Binary dataset + regression-only model.
    config = _make_config(
        datasets=("fake_binary",),
        models=("fake_constant_regressor",),
        seeds=(0,),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    result = run_raw_loss(config, output_root=tmp_path / "out", env=env)
    assert result.cells_attempted == 0
    assert result.cells_skipped_task_mismatch == 5
    manifest = load_run(tmp_path / "out")
    assert bool(manifest["skipped_reason"].notna().all())
    assert bool(manifest["skipped_reason"].str.startswith("task_type_mismatch").all())
    # Skipped rows carry None for metrics.
    assert bool(manifest["log_loss"].isna().all())


def test_regression_quantile_emits_b5_followup_skip_reason(
    fake_registry: list[PanelDataset], tmp_path: Path
) -> None:
    del fake_registry
    config = _make_config(
        datasets=("fake_regression_quantile",),
        models=("fake_constant_regressor",),
        seeds=(0,),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    result = run_raw_loss(config, output_root=tmp_path / "out", env=env)
    # Model declares task_types=("regression_point",) so the cell is
    # task-mismatch first, NOT quantile-followup. To reach the
    # followup branch we'd need a model whose task_types include
    # regression_quantile; the seq-sklearn TFTRegressor adapter does,
    # but it's not in the fake registry. The fixture-driven test
    # still pins the precedence (task mismatch wins over the
    # followup), which is the correct skip-reason layering.
    assert result.cells_skipped_task_mismatch == 5
    assert result.cells_skipped_quantile_followup == 0


def test_adapter_crash_emits_skipped_reason_and_does_not_abort_run(
    fake_registry: list[PanelDataset], tmp_path: Path
) -> None:
    del fake_registry
    # The crashing adapter declares task_types=("binary",) so it IS
    # applicable to the binary dataset and the run must reach `fit`,
    # which raises.
    config = _make_config(
        datasets=("fake_binary",),
        models=("fake_crashing", "fake_constant_binary"),
        seeds=(0,),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    result = run_raw_loss(config, output_root=tmp_path / "out", env=env)
    # The constant adapter's cells run cleanly; the crashing
    # adapter's cells are recorded as skipped.
    assert result.cells_attempted == 5
    assert result.cells_skipped_adapter_error == 5
    manifest = load_run(tmp_path / "out")
    crashed = manifest.loc[manifest["model_name"] == "fake_crashing"]
    healthy = manifest.loc[manifest["model_name"] == "fake_constant_binary"]
    assert len(crashed) == 5
    assert len(healthy) == 5
    assert bool(crashed["skipped_reason"].str.contains("adapter_error: RuntimeError").all())
    assert bool(healthy["skipped_reason"].isna().all())


def test_leaderboard_renders_nonempty_markdown(
    fake_registry: list[PanelDataset], tmp_path: Path
) -> None:
    del fake_registry
    config = _make_config(
        datasets=("fake_binary", "fake_regression_point"),
        models=("fake_constant_binary", "fake_constant_regressor"),
        seeds=(0, 1),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    run_raw_loss(config, output_root=tmp_path / "out", env=env)
    manifest = load_run(tmp_path / "out")
    md = render_leaderboard_markdown(manifest)
    assert "Raw-loss leaderboard" in md
    assert "fake_binary" in md
    assert "fake_regression_point" in md
    # Skipped (task-mismatch) cells are footnoted.
    assert "Skipped cells" in md


def test_rank_by_primary_loss_orders_by_log_loss_for_classification(
    fake_registry: list[PanelDataset], tmp_path: Path
) -> None:
    del fake_registry
    config = _make_config(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        seeds=(0,),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    run_raw_loss(config, output_root=tmp_path / "out", env=env)
    manifest = load_run(tmp_path / "out")
    entries = rank_by_primary_loss(manifest)
    assert len(entries) == 1
    assert entries[0].primary_metric == "log_loss"
    assert entries[0].n_cells_evaluated == 5
    assert entries[0].n_skipped == 0


def test_resolve_seeds_raises_when_no_raw_loss_experiment(
    fake_registry: list[PanelDataset], tmp_path: Path
) -> None:
    del fake_registry
    # Config carries only an `ensemble` experiment; raw_loss driver
    # must refuse cleanly.
    config = BenchmarkConfig(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        experiments=(ExperimentSpec(kind="ensemble", seeds=(0,)),),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    with pytest.raises(ValueError, match="no raw_loss"):
        run_raw_loss(config, output_root=tmp_path / "out", env=env)


def test_run_requires_cache_dir(fake_registry: list[PanelDataset], tmp_path: Path) -> None:
    del fake_registry
    config = BenchmarkConfig(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        experiments=(ExperimentSpec(kind="raw_loss", seeds=(0,)),),
        output_dir=tmp_path / "out",
        cache_dir=None,
    )
    env = build_run_environment(profile="smoke")
    with pytest.raises(ValueError, match="cache_dir"):
        run_raw_loss(config, output_root=tmp_path / "out", env=env)


def test_already_complete_cells_keep_their_metrics_after_resume(
    fake_registry: list[PanelDataset], tmp_path: Path
) -> None:
    del fake_registry
    config = _make_config(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        seeds=(0,),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    run_raw_loss(config, output_root=tmp_path / "out", env=env)
    first_manifest = load_run(tmp_path / "out")
    # Verify each cell sentinel exists.
    for _, row in first_manifest.iterrows():
        key = "{ds}__{m}__seed_{s}__default__fold_{f}".format(
            ds=row["dataset_name"],
            m=row["model_name"],
            s=row["seed"],
            f=row["fold_index"],
        )
        assert is_cell_complete(tmp_path / "out", key)

    # Second invocation reuses the manifest verbatim.
    run_raw_loss(config, output_root=tmp_path / "out", env=env)
    second_manifest = load_run(tmp_path / "out")
    assert len(first_manifest) == len(second_manifest)
    # Compare a deterministic field: the started_at_utc carries from
    # the first run because we did NOT re-write the shard. (Run
    # IDs across the two invocations differ, but the resumed cells
    # carry the run_id of the FIRST invocation, not the second.)
    assert first_manifest["run_id"].tolist() == second_manifest["run_id"].tolist()


# --- R1 / qa-C1: regression_quantile B5-followup branch actually fires. ---


def test_regression_quantile_followup_branch_fires_for_applicable_model(
    fake_registry: list[PanelDataset], tmp_path: Path
) -> None:
    del fake_registry
    config = _make_config(
        datasets=("fake_regression_quantile",),
        models=("fake_quantile_regressor",),
        seeds=(0,),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    result = run_raw_loss(config, output_root=tmp_path / "out", env=env)
    # The model declares task_types=(regression_point, regression_quantile)
    # so the cell IS applicable; the driver then emits the followup
    # skip reason instead of computing metrics.
    assert result.cells_attempted == 0
    assert result.cells_skipped_task_mismatch == 0
    assert result.cells_skipped_quantile_followup == 5
    manifest = load_run(tmp_path / "out")
    assert bool(manifest["skipped_reason"].str.startswith("regression_quantile_b5_followup").all())


# --- R1 / arch-C2 part 2: classifier dataset + supports_proba=False model. ---


def test_classifier_dataset_with_no_proba_model_emits_typed_skip(
    fake_registry: list[PanelDataset], tmp_path: Path
) -> None:
    del fake_registry
    config = _make_config(
        datasets=("fake_binary",),
        models=("fake_no_proba_classifier",),
        seeds=(0,),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    result = run_raw_loss(config, output_root=tmp_path / "out", env=env)
    assert result.cells_attempted == 0
    assert result.cells_skipped_proba_unavailable == 5
    assert result.cells_skipped_adapter_error == 0
    manifest = load_run(tmp_path / "out")
    assert bool(
        manifest["skipped_reason"].str.startswith("probabilities_required_for_classification").all()
    )


# --- R1 / arch-C2: harness-side ValueError propagates instead of being
# swallowed as an "adapter_error". ---


def test_harness_value_error_propagates_through_narrowed_except(
    fake_registry: list[PanelDataset], tmp_path: Path
) -> None:
    del fake_registry
    config = _make_config(
        datasets=("fake_binary",),
        models=("fake_harness_bug",),
        seeds=(0,),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    with pytest.raises(ValueError, match="ValueError must propagate"):
        run_raw_loss(config, output_root=tmp_path / "out", env=env)


# --- R1 / arch-C1: NaN-row strip before metrics. ---


def test_nan_proba_rows_are_stripped_and_counted(
    fake_registry: list[PanelDataset], tmp_path: Path
) -> None:
    del fake_registry
    config = _make_config(
        datasets=("fake_binary",),
        models=("fake_nan_proba_classifier",),
        seeds=(0,),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    result = run_raw_loss(config, output_root=tmp_path / "out", env=env)
    # Every cell should succeed; the 2 nan rows per fold are stripped
    # before the metric call.
    assert result.cells_attempted == 5
    assert result.cells_skipped_adapter_error == 0
    manifest = load_run(tmp_path / "out")
    # The dropped count is recorded on every successful row.
    assert bool(manifest["n_below_floor_dropped"].notna().all())
    assert bool((manifest["n_below_floor_dropped"] == 2).all())
    # And the metric columns are finite.
    assert bool(manifest["log_loss"].notna().all())


# --- R1 / qa-C2, C3, C4: renderer coverage. ---


def test_render_leaderboard_markdown_returns_no_results_on_empty_manifest() -> None:
    md = render_leaderboard_markdown(pd.DataFrame())
    assert md.startswith("# Raw-loss leaderboard")
    assert "_No results" in md


def test_render_from_dir_loads_manifest_and_renders_markdown(
    fake_registry: list[PanelDataset], tmp_path: Path
) -> None:
    from benchmarks.report.raw_loss import render_from_dir

    del fake_registry
    config = _make_config(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        seeds=(0,),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    run_raw_loss(config, output_root=tmp_path / "out", env=env)
    md = render_from_dir(tmp_path / "out")
    assert "Raw-loss leaderboard" in md
    assert "fake_binary" in md
    # The classification primary loss must surface in the header.
    assert "log_loss" in md


def test_regression_point_block_renders_rmse_in_header(
    fake_registry: list[PanelDataset], tmp_path: Path
) -> None:
    del fake_registry
    config = _make_config(
        datasets=("fake_regression_point",),
        models=("fake_constant_regressor",),
        seeds=(0,),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    run_raw_loss(config, output_root=tmp_path / "out", env=env)
    manifest = load_run(tmp_path / "out")
    md = render_leaderboard_markdown(manifest)
    # The header for a regression_point block is "ranked by rmse";
    # without this assert, the dataset name could appear only in the
    # skipped-cells footnote and the regression block could be missing.
    assert "ranked by rmse" in md


# --- R1 / qa-C6: B1.5 excluded datasets are silently skipped. ---


def test_excluded_dataset_is_not_loaded_or_benchmarked(
    fake_registry: list[PanelDataset], tmp_path: Path
) -> None:
    del fake_registry
    # Register an excluded dataset whose loader would crash if called;
    # the driver must skip it before invoking the loader.
    loader_invocations: list[str] = []

    def _exploding_loader(_root: Path) -> PanelDataset:
        loader_invocations.append("called")
        raise RuntimeError("excluded dataset loader should not be invoked")

    excluded_spec = DatasetSpec(
        name="fake_excluded",
        task_type="binary",
        access_tier="OPEN",
        size_tier="small",
        balance="balanced",
        modality="numeric",
        source_uri="https://example.test/exc.csv",
        integrity_sha256=_ZERO_SHA,
        archive_basename="exc.csv",
        entity_col="entity_id",
        time_col="period",
        target_col="y",
        feature_real_cols=("x_feat",),
        feature_categorical_cols=(),
        lookback=2,
        excluded=True,
        exclusion_reason="rubric_audit_negative; test fixture",
        citation="synthetic; test only",
    )
    register_dataset(excluded_spec, _exploding_loader)

    config = _make_config(
        datasets=("fake_excluded",),
        models=("fake_constant_binary",),
        seeds=(0,),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    out_root = tmp_path / "out"
    # The CLI invariant: output_root exists before the driver runs.
    out_root.mkdir(parents=True, exist_ok=True)
    result = run_raw_loss(config, output_root=out_root, env=env)
    assert loader_invocations == []  # loader never invoked
    assert result.cells_attempted == 0
    assert result.cells_skipped_task_mismatch == 0
    assert result.cells_skipped_adapter_error == 0
    # The manifest is empty (no shards written for the excluded
    # dataset; the skip is at the dataset-level boundary).
    manifest = load_run(out_root)
    assert manifest.empty


# --- R1 / qa-I1: "all" sentinel expansion. ---


def test_all_sentinel_expands_to_registered_names(
    fake_registry: list[PanelDataset],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_registry
    # `datasets=["all"]` resolves to every registered dataset at run
    # time. The autouse `isolated_registry` snapshot includes the
    # production datasets registered at import time (`amex_default`,
    # `c_mapss_fd001`), which would trigger real loaders here.
    # Monkeypatch `list_datasets` to return only the synthetic
    # entries so the assertion is bounded to the fake registry.
    monkeypatch.setattr(
        "benchmarks.experiments.raw_loss.list_datasets",
        lambda: ("fake_binary", "fake_regression_point"),
    )
    config = BenchmarkConfig(
        datasets=("all",),
        models=("fake_constant_binary",),
        experiments=(ExperimentSpec(kind="raw_loss", seeds=(0,)),),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    result = run_raw_loss(config, output_root=tmp_path / "out", env=env)
    manifest = load_run(tmp_path / "out")
    # Both fake datasets get cells (binary applicable, regression skip).
    seen_datasets = set(manifest["dataset_name"].unique())
    assert "fake_binary" in seen_datasets
    assert "fake_regression_point" in seen_datasets
    # Only the binary cells actually attempted.
    assert result.cells_attempted == 5


# --- R1 / arch-I2: loader-skip when every cell already complete. ---


def test_loader_is_not_called_on_full_resume(
    fake_registry: list[PanelDataset], tmp_path: Path
) -> None:
    del fake_registry
    config = _make_config(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        seeds=(0,),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    # First run materializes the manifest.
    run_raw_loss(config, output_root=tmp_path / "out", env=env)

    # Re-register the binary dataset with an exploding loader; if
    # the driver invokes it on the second run, the test fails
    # loudly. `isolated_registry` (autouse) restores the original
    # loader on teardown.
    import benchmarks.registry.datasets as _ds_reg

    def _exploding_loader(_root: Path) -> PanelDataset:
        raise RuntimeError("loader must not be invoked on full resume")

    _ds_reg._LOADERS["fake_binary"] = _exploding_loader  # pyright: ignore[reportPrivateUsage]

    second_result = run_raw_loss(config, output_root=tmp_path / "out", env=env)
    assert second_result.cells_attempted == 0
    assert second_result.cells_already_complete == 5
    manifest = load_run(tmp_path / "out")
    assert len(manifest) == 5


# --- R1 / qa-I3: _resolve_git_sha failure modes. ---


def test_resolve_git_sha_returns_unknown_on_subprocess_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmarks.experiments.raw_loss import _resolve_git_sha

    def _raise(_args: list[str], **_kw: Any) -> None:
        raise OSError("simulated: git binary missing")

    monkeypatch.setattr("benchmarks.experiments.raw_loss.subprocess.run", _raise)
    assert _resolve_git_sha(Path("/")) == "unknown"


def test_resolve_git_sha_returns_unknown_on_non_zero_returncode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmarks.experiments.raw_loss import _resolve_git_sha

    class _Completed:
        def __init__(self) -> None:
            self.returncode = 128
            self.stdout = ""

    def _fake_run(*_a: Any, **_kw: Any) -> _Completed:
        return _Completed()

    monkeypatch.setattr("benchmarks.experiments.raw_loss.subprocess.run", _fake_run)
    assert _resolve_git_sha(Path("/")) == "unknown"


# --- R1 / qa-I5 + arch-I4: register_adapter and register_adapter_factory
# error paths. ---


def test_register_adapter_factory_raises_for_unregistered_model() -> None:
    from benchmarks.registry import ModelNotRegisteredError, register_adapter_factory

    def _factory(*, spec: Any, hyperparameters: Any = None) -> Any:
        return None

    with pytest.raises(ModelNotRegisteredError):
        register_adapter_factory("ghost_model", _factory)


def test_register_adapter_factory_rejects_conflicting_factory() -> None:
    from benchmarks.registry import (
        register_adapter_factory,
        register_model,
    )

    spec = ModelSpec(
        name="fake_conflict_model",
        family="sklearn_passthrough",
        task_types=("binary",),
        supports_proba=True,
        reason="synthetic; test only",
    )
    register_model(spec)

    def _factory_a(*, spec: Any, hyperparameters: Any = None) -> Any:
        return None

    def _factory_b(*, spec: Any, hyperparameters: Any = None) -> Any:
        return None

    register_adapter_factory("fake_conflict_model", _factory_a)
    with pytest.raises(ValueError, match="already registered"):
        register_adapter_factory("fake_conflict_model", _factory_b)


def test_register_adapter_rolls_back_spec_when_factory_registration_fails() -> None:
    # `register_adapter` registers spec + factory atomically; if the
    # factory registration raises, the spec must be rolled back so a
    # later registration under the same name does not collide.
    import benchmarks.registry.models as _models
    from benchmarks.registry import list_models, register_adapter

    spec = ModelSpec(
        name="fake_rollback_model",
        family="sklearn_passthrough",
        task_types=("binary",),
        supports_proba=True,
        reason="synthetic; test only",
    )

    # Force the factory registration to fail by pre-poisoning the
    # factory map with a conflicting entry under the same name. This
    # requires the spec to be present already, so do the poisoning
    # in two steps that match `register_adapter`'s precondition.
    def _placeholder(**_: Any) -> Any:
        return None

    _models._REGISTRY[spec.name] = spec  # pyright: ignore[reportPrivateUsage]
    _models._ADAPTER_FACTORIES[spec.name] = _placeholder  # pyright: ignore[reportPrivateUsage]
    # Now remove the spec so register_adapter's `is_new_spec` is True;
    # the factory map still has an entry under the same name so the
    # second `register_adapter_factory` call will trigger the
    # conflict guard.
    del _models._REGISTRY[spec.name]  # pyright: ignore[reportPrivateUsage]

    def _new_factory(*, spec: Any, hyperparameters: Any = None) -> Any:
        return None

    with pytest.raises(ValueError, match="already registered"):
        register_adapter(spec, _new_factory)
    # Rolled back: the spec is no longer in the registry after the
    # failure (otherwise a second registration would mis-fire).
    assert spec.name not in list_models()


# --- R1 / qa-I6: rank_by_primary_loss all-skipped sentinel row. ---


def test_rank_by_primary_loss_all_skipped_emits_nan_entry() -> None:
    # Construct a manifest manually with one (dataset, model) group
    # whose every row is skipped; the renderer must emit a sentinel
    # entry with nan loss and n_cells_evaluated=0.
    manifest = pd.DataFrame(
        [
            {
                "dataset_name": "ds_a",
                "model_name": "m_a",
                "task_type": "binary",
                "seed": 0,
                "fold_index": 0,
                "skipped_reason": "task_type_mismatch: ...",
                "log_loss": None,
            },
            {
                "dataset_name": "ds_a",
                "model_name": "m_a",
                "task_type": "binary",
                "seed": 0,
                "fold_index": 1,
                "skipped_reason": "task_type_mismatch: ...",
                "log_loss": None,
            },
        ]
    )
    entries = rank_by_primary_loss(manifest)
    assert len(entries) == 1
    assert math.isnan(entries[0].primary_loss_mean)
    assert entries[0].n_cells_evaluated == 0
    assert entries[0].n_skipped == 2


# --- R1 / qa-I7: long skipped_reason gets truncated in the footnote. ---


def test_long_skipped_reason_is_truncated_in_footnote() -> None:
    long_reason = "adapter_error: RuntimeError: " + ("x" * 200)
    manifest = pd.DataFrame(
        [
            {
                "dataset_name": "ds_a",
                "model_name": "m_a",
                "task_type": "binary",
                "seed": 0,
                "fold_index": 0,
                "skipped_reason": long_reason,
                "log_loss": None,
            }
        ]
    )
    md = render_leaderboard_markdown(manifest)
    # The truncation guard inserts "..." at the 117-char mark; the
    # full 230-char reason must NOT appear verbatim.
    assert long_reason not in md
    assert "..." in md


# --- R1 / qa-I9 + B8.1 named test: split_fingerprint stable across seeds. ---


def test_split_fingerprint_is_stable_across_seeds(
    fake_registry: list[PanelDataset], tmp_path: Path
) -> None:
    del fake_registry
    config = _make_config(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        seeds=(0, 1, 2),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    run_raw_loss(config, output_root=tmp_path / "out", env=env)
    manifest = load_run(tmp_path / "out")
    # B8.1: per-dataset, the split_fingerprint must be identical
    # across every seed (the splitter is seed-invariant; only the
    # model's training seed changes).
    per_dataset = manifest.groupby("dataset_name")["split_fingerprint"].nunique()
    assert (per_dataset == 1).all()


# --- R1 / qa-N2: _format_metric direct branches. ---


def test_format_metric_branches() -> None:
    from benchmarks.report.raw_loss import _format_metric

    assert _format_metric(None) == ""
    assert _format_metric(float("nan")) == ""
    assert _format_metric(1.23456) == "1.2346"
    assert _format_metric("hello") == "hello"


# --- R2 / arch-CRITICAL: runtime ProbaUnsupportedError emits a typed
# skip instead of aborting the run. ---


def test_runtime_proba_unsupported_emits_typed_skip(
    fake_registry: list[PanelDataset], tmp_path: Path
) -> None:
    del fake_registry
    config = _make_config(
        datasets=("fake_binary",),
        models=("fake_runtime_proba_error",),
        seeds=(0,),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    # Pre-R2: this would raise ValueError from compute_all and abort
    # the run. Post-R2: the cell is skipped with the typed reason.
    result = run_raw_loss(config, output_root=tmp_path / "out", env=env)
    assert result.cells_attempted == 0
    assert result.cells_skipped_proba_runtime_unavailable == 5
    assert result.cells_skipped_adapter_error == 0
    manifest = load_run(tmp_path / "out")
    assert bool(
        manifest["skipped_reason"]
        .str.startswith("probabilities_required_but_runtime_unavailable")
        .all()
    )


# --- R2 / code-IMP: a predict-time adapter crash routes to
# adapter_error (the existing test only exercises fit-time). ---


def test_predict_time_runtime_error_routes_to_adapter_error(
    fake_registry: list[PanelDataset], tmp_path: Path
) -> None:
    del fake_registry
    config = _make_config(
        datasets=("fake_binary",),
        models=("fake_predict_crashing",),
        seeds=(0,),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    result = run_raw_loss(config, output_root=tmp_path / "out", env=env)
    assert result.cells_attempted == 0
    assert result.cells_skipped_adapter_error == 5
    manifest = load_run(tmp_path / "out")
    assert bool(manifest["skipped_reason"].str.contains("adapter_error: RuntimeError").all())


# --- R2 / arch-I1: NotFittedError is in the narrowed except set. ---


def test_not_fitted_error_is_caught_as_adapter_error(
    fake_registry: list[PanelDataset], tmp_path: Path
) -> None:
    del fake_registry
    config = _make_config(
        datasets=("fake_binary",),
        models=("fake_not_fitted",),
        seeds=(0,),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    # Pre-R2: NotFittedError -> ValueError-shaped (NotFittedError
    # subclasses sklearn.exceptions.NotFittedError which itself
    # subclasses ValueError) -> falls outside narrow set -> aborts.
    # Post-R2: caught explicitly, routes to adapter_error.
    result = run_raw_loss(config, output_root=tmp_path / "out", env=env)
    assert result.cells_attempted == 0
    assert result.cells_skipped_adapter_error == 5
    manifest = load_run(tmp_path / "out")
    assert bool(manifest["skipped_reason"].str.contains("NotFittedError").all())


# --- R2 / qa-I-C: _strip_below_floor_rows for float y_pred (regression). ---


def test_strip_below_floor_rows_regression_nan_pred(
    fake_registry: list[PanelDataset], tmp_path: Path
) -> None:
    del fake_registry
    config = _make_config(
        datasets=("fake_regression_point",),
        models=("fake_nan_regressor",),
        seeds=(0,),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    result = run_raw_loss(config, output_root=tmp_path / "out", env=env)
    assert result.cells_attempted == 5
    assert result.cells_skipped_adapter_error == 0
    manifest = load_run(tmp_path / "out")
    # The 2 nan rows per fold are stripped before the metric layer.
    assert bool(manifest["n_below_floor_dropped"].notna().all())
    assert bool((manifest["n_below_floor_dropped"] == 2).all())
    # And the metric columns are finite.
    assert bool(manifest["rmse"].notna().all())


# --- R2 / qa-I-B: register_adapter happy path. ---


def test_register_adapter_returns_spec_on_success() -> None:
    from benchmarks.registry import (
        get_adapter_factory,
        list_models,
        register_adapter,
    )

    spec = ModelSpec(
        name="fake_happy_path_model",
        family="sklearn_passthrough",
        task_types=("binary",),
        supports_proba=True,
        reason="synthetic; test only",
    )

    def _factory(
        *,
        spec: DatasetSpec,
        hyperparameters: dict[str, Any] | None = None,
    ) -> SeqSklearnAdapter:
        del hyperparameters
        return _ConstantBinaryAdapter(spec=spec)  # type: ignore[return-value]

    returned = register_adapter(spec, _factory)
    assert returned is spec
    assert "fake_happy_path_model" in list_models()
    assert get_adapter_factory("fake_happy_path_model") is _factory


# --- R2 / qa-I-D: rank_by_primary_loss skips regression_quantile groups. ---


def test_rank_by_primary_loss_skips_regression_quantile_groups() -> None:
    manifest = pd.DataFrame(
        [
            {
                "dataset_name": "ds_a",
                "model_name": "m_a",
                "task_type": "regression_quantile",
                "seed": 0,
                "fold_index": 0,
                "skipped_reason": None,
                "rmse": 0.5,
                "log_loss": None,
            }
        ]
    )
    entries = rank_by_primary_loss(manifest)
    # regression_quantile rows are not in `_PRIMARY_LOSS`; skipped.
    assert entries == []


# --- R2 / qa-I-E: build_run_environment honors explicit library_repo_root. ---


def test_build_run_environment_honors_explicit_library_repo_root(
    tmp_path: Path,
) -> None:
    # An empty tmp_path is not a git checkout; `_resolve_git_sha`
    # returns "unknown" for it. The explicit-root branch must
    # actually route through this path (not fall back to the
    # module-relative default which IS a git checkout).
    env = build_run_environment(profile="smoke", library_repo_root=tmp_path)
    assert env.library_git_sha == "unknown"


# --- R2 / arch-I2: int|None column dtype is Int64 in shards. ---


def test_int_or_none_columns_are_int64_nullable_dtype(
    fake_registry: list[PanelDataset], tmp_path: Path
) -> None:
    del fake_registry
    config = _make_config(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        seeds=(0,),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    run_raw_loss(config, output_root=tmp_path / "out", env=env)
    manifest = load_run(tmp_path / "out")
    # n_below_floor_dropped is `int | None` -> nullable Int64.
    assert str(manifest["n_below_floor_dropped"].dtype) == "Int64"
    # peak_cuda_bytes is also int | None and unset on this CPU run.
    assert str(manifest["peak_cuda_bytes"].dtype) == "Int64"
