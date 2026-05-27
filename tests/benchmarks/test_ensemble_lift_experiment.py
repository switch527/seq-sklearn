"""Phase B11 ensemble-lift experiment tests (B6.2.5).

The ensemble-lift driver reads B5's per-cell predictions shards,
builds GBM-only and GBM+seq averaged ensembles per (dataset, seed,
fold), computes per-dataset Δloss, and runs Wilcoxon signed-rank
across datasets. These tests pin the end-to-end flow against the
synthetic fake registry plus a fresh `seq_sklearn`-family
constant adapter (so the e2e join actually completes; the
existing `NaNProbaAdapter` fake in `_fakes.py` returns NaN proba
which would block the ensemble's log_loss computation).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Self, cast

import numpy as np
import pandas as pd
import pytest
from benchmarks.adapters._base import (
    QuantilesUnsupportedError,
    SeqSklearnAdapter,
)
from benchmarks.config import BenchmarkConfig, DatasetSpec, ExperimentSpec, TaskType
from benchmarks.experiments import (
    EnsembleLiftExperimentResult,
    build_run_environment,
    run_ensemble_lift,
    run_raw_loss,
)
from benchmarks.registry.models import register_adapter
from benchmarks.report.ensemble_lift import render_ensemble_lift_markdown

from tests.benchmarks._fakes import (
    register_all_fakes_and_get_panels,
)


@dataclass
class _ConstantSeqClassifierAdapter:
    """A `family="seq_sklearn"` test adapter that mirrors
    `ConstantBinaryAdapter` but registers under the seq family so
    the ensemble-lift driver routes it into the seq half of the
    GBM+seq ensemble. Returns deterministic 0.6/0.4 probabilities
    so log_loss is well-defined."""

    name: ClassVar[str] = "fake_seq_constant_classifier"
    family: ClassVar[str] = "seq_sklearn"
    task_types: ClassVar[tuple[TaskType, ...]] = ("binary",)
    supports_proba: ClassVar[bool] = True

    spec: DatasetSpec
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    _majority: int = field(default=0, init=False)

    def fit(self, panel: pd.DataFrame, y: np.ndarray) -> Self:
        del panel
        if len(y) > 0:
            values, counts = np.unique(y, return_counts=True)
            self._majority = int(values[counts.argmax()])
        return self

    def predict(self, panel: pd.DataFrame) -> np.ndarray:
        return np.full(len(panel), self._majority, dtype=np.int64)

    def predict_proba(self, panel: pd.DataFrame) -> np.ndarray:
        proba = np.full((len(panel), 2), 0.4)
        proba[:, self._majority] = 0.6
        return proba

    def predict_quantiles(self, panel: pd.DataFrame) -> np.ndarray:
        del panel
        raise QuantilesUnsupportedError(self.name)


def _register_fake_seq_classifier() -> None:
    """Register the fake seq_sklearn-family classifier in the
    registry. The conftest's `isolated_registry` autouse fixture
    restores both the model + adapter-factory dicts after each
    test."""
    from benchmarks.config import ModelSpec

    spec = ModelSpec(
        name="fake_seq_constant_classifier",
        family="seq_sklearn",
        task_types=("binary",),
        supports_proba=True,
        reason="ensemble-lift test fixture",
    )

    def _factory(
        *,
        spec: DatasetSpec,
        hyperparameters: dict[str, Any] | None = None,
    ) -> SeqSklearnAdapter:
        return cast(
            SeqSklearnAdapter,
            _ConstantSeqClassifierAdapter(spec=spec, hyperparameters=hyperparameters or {}),
        )

    register_adapter(spec, _factory)


# --- e2e flow ----------------------------------------------------------------


def test_run_ensemble_lift_raises_when_no_ensemble_lift_experiment(
    tmp_path: Path,
) -> None:
    """The driver refuses a config that doesn't declare its
    experiment kind. Matches the B7/B8 contract."""
    register_all_fakes_and_get_panels()
    _register_fake_seq_classifier()
    config = BenchmarkConfig(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        experiments=(ExperimentSpec(kind="raw_loss", seeds=(0,)),),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    (tmp_path / "out").mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="no ensemble_lift"):
        run_ensemble_lift(config, output_root=tmp_path / "out", env=env)


def test_run_ensemble_lift_raises_when_b5_manifest_empty(
    tmp_path: Path,
) -> None:
    """The driver refuses cleanly when no B5 manifest exists."""
    register_all_fakes_and_get_panels()
    _register_fake_seq_classifier()
    config = BenchmarkConfig(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        experiments=(ExperimentSpec(kind="ensemble_lift", seeds=(0,)),),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    (tmp_path / "out").mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="manifest"):
        run_ensemble_lift(config, output_root=tmp_path / "out", env=env)


def test_run_ensemble_lift_no_gbm_cells_surfaces_footnote(tmp_path: Path) -> None:
    """A manifest with only seq cells produces a `no_gbm_predictions`
    sentinel row so the report footnote lists the gap."""
    register_all_fakes_and_get_panels()
    _register_fake_seq_classifier()
    config = BenchmarkConfig(
        datasets=("fake_binary",),
        models=("fake_seq_constant_classifier",),
        experiments=(
            ExperimentSpec(kind="raw_loss", seeds=(0,)),
            ExperimentSpec(kind="ensemble_lift", seeds=(0,)),
        ),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    output_root = tmp_path / "out"
    run_raw_loss(config, output_root=output_root, env=env)
    result = run_ensemble_lift(config, output_root=output_root, env=env)
    assert len(result.rows) == 1
    row = result.rows[0]
    assert row.dataset_name == "fake_binary"
    assert row.delta_loss_mean is None
    assert row.no_gbm_predictions is True


def test_run_ensemble_lift_no_seq_cells_surfaces_footnote(tmp_path: Path) -> None:
    """Mirror: a manifest with only GBM cells produces a
    `no_seq_predictions` sentinel."""
    register_all_fakes_and_get_panels()
    _register_fake_seq_classifier()
    config = BenchmarkConfig(
        datasets=("fake_binary",),
        models=("lightgbm_classifier",),
        experiments=(
            ExperimentSpec(kind="raw_loss", seeds=(0,)),
            ExperimentSpec(kind="ensemble_lift", seeds=(0,)),
        ),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    output_root = tmp_path / "out"
    run_raw_loss(config, output_root=output_root, env=env)
    result = run_ensemble_lift(config, output_root=output_root, env=env)
    assert len(result.rows) == 1
    row = result.rows[0]
    assert row.delta_loss_mean is None
    assert row.no_seq_predictions is True


def test_run_ensemble_lift_paired_cells_produce_delta_and_wilcoxon(
    tmp_path: Path,
) -> None:
    """The headline path: both GBM and seq families have OK cells
    on the same (seed, fold) pairs. The driver produces a Δloss
    row + oracle bound + Wilcoxon result."""
    register_all_fakes_and_get_panels()
    _register_fake_seq_classifier()
    config = BenchmarkConfig(
        datasets=("fake_binary",),
        models=("lightgbm_classifier", "fake_seq_constant_classifier"),
        experiments=(
            ExperimentSpec(kind="raw_loss", seeds=(0,)),
            ExperimentSpec(kind="ensemble_lift", seeds=(0,)),
        ),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    output_root = tmp_path / "out"
    run_raw_loss(config, output_root=output_root, env=env)
    result = run_ensemble_lift(config, output_root=output_root, env=env)
    assert isinstance(result, EnsembleLiftExperimentResult)
    assert result.seq_family == "seq_sklearn"
    assert result.baseline_family == "gbm"
    assert len(result.rows) == 1
    row = result.rows[0]
    assert row.dataset_name == "fake_binary"
    assert row.n_cells_paired > 0
    assert row.loss_gbm_only_mean is not None
    assert row.loss_gbm_plus_seq_mean is not None
    assert row.delta_loss_mean is not None
    # Oracle <= both individual losses by construction.
    assert row.oracle_loss_mean is not None
    assert row.oracle_loss_mean <= row.loss_gbm_only_mean
    assert row.oracle_loss_mean <= row.loss_gbm_plus_seq_mean
    # Wilcoxon over a single dataset: scipy returns a statistic +
    # p-value even at n=1 (degenerate but defined). family_size=1
    # so holm-adjusted equals raw.
    assert result.wilcoxon.n_datasets == 1
    assert result.wilcoxon.family_size == 1
    if result.wilcoxon.p_value is not None:
        assert result.wilcoxon.holm_adjusted_p_value == result.wilcoxon.p_value


def test_render_ensemble_lift_markdown_renders_delta_and_wilcoxon(
    tmp_path: Path,
) -> None:
    """The renderer turns the structured result into a Markdown
    report with the executive summary, the per-dataset table, and
    the Wilcoxon block."""
    register_all_fakes_and_get_panels()
    _register_fake_seq_classifier()
    config = BenchmarkConfig(
        datasets=("fake_binary",),
        models=("lightgbm_classifier", "fake_seq_constant_classifier"),
        experiments=(
            ExperimentSpec(kind="raw_loss", seeds=(0,)),
            ExperimentSpec(kind="ensemble_lift", seeds=(0,)),
        ),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    output_root = tmp_path / "out"
    run_raw_loss(config, output_root=output_root, env=env)
    result = run_ensemble_lift(config, output_root=output_root, env=env)
    md = render_ensemble_lift_markdown(result)
    assert "# Ensemble-lift report" in md
    assert "fake_binary" in md
    assert "Per-dataset Δloss" in md
    assert "Wilcoxon signed-rank" in md


def test_render_ensemble_lift_markdown_empty_result_shape() -> None:
    """A result with zero rows still renders a valid Markdown
    document (no-results block + skipped Wilcoxon)."""
    from benchmarks.experiments.ensemble_lift import WilcoxonResult

    result = EnsembleLiftExperimentResult(
        run_id="r1",
        seq_family="seq_sklearn",
        baseline_family="gbm",
        rows=(),
        wilcoxon=WilcoxonResult(
            statistic=None,
            p_value=None,
            holm_adjusted_p_value=None,
            n_datasets=0,
            family_size=1,
        ),
    )
    md = render_ensemble_lift_markdown(result)
    assert "No paired" in md
    assert "Wilcoxon skipped" in md
