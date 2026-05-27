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

import benchmarks.adapters  # type: ignore[reportUnusedImport]  # noqa: F401  --  side-effect import: registers the GBM family for `lightgbm_classifier`
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
from benchmarks.experiments.ensemble_lift import (
    PerDatasetLift,
    WilcoxonResult,
    _wilcoxon_across_datasets,
)
from benchmarks.predictions import write_predictions
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
    # Wilcoxon over a single dataset is informational only: the
    # driver short-circuits to the None path so the report flags
    # the result rather than quoting a meaningless p-value.
    assert result.wilcoxon.n_datasets == 1
    assert result.wilcoxon.family_size == 1
    assert result.wilcoxon.statistic is None
    assert result.wilcoxon.p_value is None
    assert result.wilcoxon.holm_adjusted_p_value is None


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


# --- Wilcoxon aggregation (R1 fixes) -----------------------------------------


def _row(*, dataset_name: str, delta: float | None) -> PerDatasetLift:
    """Build a PerDatasetLift carrying just the fields the
    Wilcoxon aggregator reads."""
    return PerDatasetLift(
        dataset_name=dataset_name,
        task_type="binary",
        primary_loss_column="log_loss",
        n_cells_paired=1 if delta is not None else 0,
        delta_loss_mean=delta,
    )


def test_wilcoxon_at_n_one_short_circuits_to_skip_path() -> None:
    """Paired signed-rank needs n>=2; n=1 returns the None path
    so the renderer flags the result as informational."""
    result = _wilcoxon_across_datasets([_row(dataset_name="a", delta=0.1)], family_size=1)
    assert result.statistic is None
    assert result.p_value is None
    assert result.holm_adjusted_p_value is None
    assert result.n_datasets == 1


def test_wilcoxon_at_n_two_returns_real_statistic() -> None:
    """The n>=2 branch hits scipy.stats.wilcoxon; family_size=1
    so the Holm-adjusted value equals the raw p-value."""
    rows = [_row(dataset_name="a", delta=0.1), _row(dataset_name="b", delta=0.05)]
    result = _wilcoxon_across_datasets(rows, family_size=1)
    assert result.statistic is not None
    assert result.p_value is not None
    assert result.holm_adjusted_p_value == result.p_value
    assert result.n_datasets == 2


def test_wilcoxon_filters_none_deltas() -> None:
    """None deltas are dropped before scipy sees them; the
    aggregator counts only the populated entries."""
    rows = [
        _row(dataset_name="a", delta=None),
        _row(dataset_name="b", delta=0.2),
        _row(dataset_name="c", delta=-0.1),
    ]
    result = _wilcoxon_across_datasets(rows, family_size=1)
    assert result.n_datasets == 2


def test_wilcoxon_all_zero_deltas_returns_neutral_p_one() -> None:
    """An all-zero diff vector triggers the early-return branch
    that surfaces statistic=0, p=1 (scipy.stats.wilcoxon would
    raise on an all-zero input under some scipy versions)."""
    rows = [_row(dataset_name="a", delta=0.0), _row(dataset_name="b", delta=0.0)]
    result = _wilcoxon_across_datasets(rows, family_size=1)
    assert result.statistic == 0.0
    assert result.p_value == 1.0
    assert result.holm_adjusted_p_value == 1.0
    assert result.n_datasets == 2


def test_wilcoxon_family_size_gt_one_applies_holm() -> None:
    """family_size>1 routes raw p through holm_correction; the
    adjusted value is >= the raw value (Holm is monotone non-
    decreasing on the smallest p)."""
    rows = [_row(dataset_name="a", delta=0.1), _row(dataset_name="b", delta=0.05)]
    result = _wilcoxon_across_datasets(rows, family_size=3)
    assert result.p_value is not None
    assert result.holm_adjusted_p_value is not None
    assert result.holm_adjusted_p_value >= result.p_value


# --- _join_predictions guards (R1 fixes) -------------------------------------


def _make_proba_shard(
    tmp_path: Path,
    *,
    key: str,
    rows: int,
    n_classes: int = 2,
) -> None:
    """Drop a fake predictions shard at `tmp_path/predictions/{key}.parquet`."""
    panel_row_index = np.arange(rows, dtype=np.int64)
    y_true = np.zeros(rows, dtype=np.float64)
    y_pred = np.zeros(rows, dtype=np.float64)
    y_proba = np.full((rows, n_classes), 1.0 / n_classes, dtype=np.float64)
    write_predictions(
        tmp_path,
        key,
        panel_row_index=panel_row_index,
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
    )


def test_join_predictions_returns_none_on_y_true_mismatch(tmp_path: Path) -> None:
    """The y_true cross-check at `_join_predictions` returns None
    when two cells on the same panel rows disagree on labels (a
    B5 contract violation; the driver skips the (seed, fold) pair
    rather than averaging mis-labelled rows)."""
    from benchmarks.experiments.ensemble_lift import _join_predictions

    panel_row_index = np.arange(4, dtype=np.int64)
    y_true_a = np.zeros(4, dtype=np.float64)
    y_true_b = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    proba = np.full((4, 2), 0.5, dtype=np.float64)
    write_predictions(
        tmp_path,
        "ds__m_a__seed_0__default__fold_0",
        panel_row_index=panel_row_index,
        y_true=y_true_a,
        y_pred=np.zeros(4),
        y_proba=proba,
    )
    write_predictions(
        tmp_path,
        "ds__m_b__seed_0__default__fold_0",
        panel_row_index=panel_row_index,
        y_true=y_true_b,
        y_pred=np.zeros(4),
        y_proba=proba,
    )
    cells = pd.DataFrame(
        [
            {"cell_key": "ds__m_a__seed_0__default__fold_0"},
            {"cell_key": "ds__m_b__seed_0__default__fold_0"},
        ]
    )
    assert _join_predictions(cells=cells, output_root=tmp_path) is None


def test_join_predictions_returns_none_on_proba_column_count_mismatch(
    tmp_path: Path,
) -> None:
    """If two cells disagree on the proba column count (a class-
    label encoding regression), the driver returns None rather
    than averaging dim-mismatched probability matrices."""
    from benchmarks.experiments.ensemble_lift import _join_predictions

    _make_proba_shard(tmp_path, key="ds__m_a__seed_0__default__fold_0", rows=4, n_classes=2)
    _make_proba_shard(tmp_path, key="ds__m_b__seed_0__default__fold_0", rows=4, n_classes=3)
    cells = pd.DataFrame(
        [
            {"cell_key": "ds__m_a__seed_0__default__fold_0"},
            {"cell_key": "ds__m_b__seed_0__default__fold_0"},
        ]
    )
    assert _join_predictions(cells=cells, output_root=tmp_path) is None


def test_join_predictions_inner_join_drops_non_overlapping_rows(
    tmp_path: Path,
) -> None:
    """The inner join intersects panel_row_index across cells.
    This is the load-bearing invariant the row-set fix relies on:
    `_per_dataset_lift` then intersects gbm_only with gbm_plus_seq
    using the same panel rows so Δloss is computed over the same
    sample population."""
    from benchmarks.experiments.ensemble_lift import _join_predictions

    proba_a = np.full((4, 2), 0.5, dtype=np.float64)
    proba_b = np.full((3, 2), 0.4, dtype=np.float64)
    proba_b[:, 1] = 0.6
    write_predictions(
        tmp_path,
        "ds__m_a__seed_0__default__fold_0",
        panel_row_index=np.arange(4, dtype=np.int64),
        y_true=np.zeros(4, dtype=np.float64),
        y_pred=np.zeros(4),
        y_proba=proba_a,
    )
    write_predictions(
        tmp_path,
        "ds__m_b__seed_0__default__fold_0",
        panel_row_index=np.array([1, 2, 3], dtype=np.int64),
        y_true=np.zeros(3, dtype=np.float64),
        y_pred=np.zeros(3),
        y_proba=proba_b,
    )
    cells = pd.DataFrame(
        [
            {"cell_key": "ds__m_a__seed_0__default__fold_0"},
            {"cell_key": "ds__m_b__seed_0__default__fold_0"},
        ]
    )
    out = _join_predictions(cells=cells, output_root=tmp_path)
    assert out is not None
    idx, y_true, averaged = out
    assert idx.tolist() == [1, 2, 3]
    assert y_true.shape == (3,)
    assert averaged.shape == (3, 2)
    np.testing.assert_allclose(averaged, (proba_a[1:4] + proba_b) / 2)


# --- renderer footnote arms (R1 fixes) ---------------------------------------


def test_render_incomplete_neither_arm_lists_neither_reason() -> None:
    """A row with BOTH `no_gbm_predictions` and
    `no_seq_predictions` set surfaces the 'neither GBM nor seq'
    incomplete-footnote arm."""
    result = EnsembleLiftExperimentResult(
        run_id="r1",
        seq_family="seq_sklearn",
        baseline_family="gbm",
        rows=(
            PerDatasetLift(
                dataset_name="orphan_ds",
                task_type="binary",
                primary_loss_column="log_loss",
                n_cells_paired=0,
                no_gbm_predictions=True,
                no_seq_predictions=True,
            ),
        ),
        wilcoxon=WilcoxonResult(
            statistic=None,
            p_value=None,
            holm_adjusted_p_value=None,
            n_datasets=0,
            family_size=1,
        ),
    )
    md = render_ensemble_lift_markdown(result)
    assert "neither GBM nor seq cells in manifest" in md
    assert "orphan_ds" in md


def test_render_incomplete_paired_but_no_overlap_arm_lists_overlap_reason() -> None:
    """When both families are PRESENT but no (seed, fold) pair
    yielded a join (predictions shards missing / proba mismatch),
    the renderer lands on the fourth incomplete-footnote arm."""
    result = EnsembleLiftExperimentResult(
        run_id="r1",
        seq_family="seq_sklearn",
        baseline_family="gbm",
        rows=(
            PerDatasetLift(
                dataset_name="overlap_ds",
                task_type="binary",
                primary_loss_column="log_loss",
                n_cells_paired=0,
                no_gbm_predictions=False,
                no_seq_predictions=False,
            ),
        ),
        wilcoxon=WilcoxonResult(
            statistic=None,
            p_value=None,
            holm_adjusted_p_value=None,
            n_datasets=0,
            family_size=1,
        ),
    )
    md = render_ensemble_lift_markdown(result)
    assert "GBM + seq cells present but no" in md
    assert "overlap_ds" in md


# --- regression task path (R1 fixes) -----------------------------------------


def test_per_cell_primary_loss_regression_point_hand_oracle() -> None:
    """RMSE matches the hand-computed value to 1e-9."""
    from benchmarks.experiments.ensemble_lift import _per_cell_primary_loss

    y_true = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y_pred = np.array([1.5, 1.5, 3.5], dtype=np.float64)
    expected = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    got = _per_cell_primary_loss(
        task_type="regression_point", y_true=y_true, proba=None, y_pred=y_pred
    )
    assert got is not None
    np.testing.assert_allclose(got, expected, atol=1e-9)


def test_per_sample_oracle_loss_regression_point_takes_min_per_row() -> None:
    """The regression oracle is sqrt(mean(min(sqerr_a, sqerr_b)))."""
    from benchmarks.experiments.ensemble_lift import _per_sample_oracle_loss

    y_true = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    pred_a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    pred_b = np.array([3.0, 1.0, 1.0], dtype=np.float64)
    sqerr_a = (y_true - pred_a) ** 2
    sqerr_b = (y_true - pred_b) ** 2
    expected = float(np.sqrt(np.minimum(sqerr_a, sqerr_b).mean()))
    got = _per_sample_oracle_loss(
        task_type="regression_point",
        y_true=y_true,
        proba_a=None,
        proba_b=None,
        pred_a=pred_a,
        pred_b=pred_b,
    )
    assert got is not None
    np.testing.assert_allclose(got, expected, atol=1e-9)


# --- pydantic result-type contract (R1 fixes) --------------------------------


def test_result_types_are_frozen_and_forbid_extra_fields() -> None:
    """The three result types ship as frozen extra='forbid'
    pydantic models so downstream tooling cannot silently
    introduce new fields. Mutation must raise, extra field
    construction must raise."""
    wilcoxon = WilcoxonResult(
        statistic=0.0,
        p_value=1.0,
        holm_adjusted_p_value=1.0,
        n_datasets=2,
        family_size=1,
    )
    row = PerDatasetLift(
        dataset_name="ds",
        task_type="binary",
        primary_loss_column="log_loss",
        n_cells_paired=1,
    )
    result = EnsembleLiftExperimentResult(
        run_id="r",
        seq_family="seq_sklearn",
        baseline_family="gbm",
        rows=(row,),
        wilcoxon=wilcoxon,
    )

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        # The model is frozen + extra="forbid"; passing an
        # unknown field via model_validate exercises the extra-
        # forbid contract without tripping pyright on a literal
        # call-site kwarg.
        WilcoxonResult.model_validate(
            {
                "statistic": 0.0,
                "p_value": 1.0,
                "holm_adjusted_p_value": 1.0,
                "n_datasets": 2,
                "family_size": 1,
                "unknown_field": "x",
            }
        )
    with pytest.raises(ValidationError):
        result.run_id = "mutated"  # type: ignore[misc]
