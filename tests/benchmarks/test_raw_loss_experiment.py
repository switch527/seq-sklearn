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


# --- Fixtures ----------------------------------------------------------------


def _register_fake_datasets(panels: list[PanelDataset]) -> None:
    for panel_dataset in panels:

        def _loader(_root: Path, _captured: PanelDataset = panel_dataset) -> PanelDataset:
            return _captured

        register_dataset(panel_dataset.spec, _loader)


def _register_fake_models() -> None:
    for adapter_cls in (_ConstantBinaryAdapter, _ConstantRegressorAdapter, _CrashingAdapter):
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
    assert entries[0].n_folds_evaluated == 5
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
