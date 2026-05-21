"""Phase B7 training-time experiment e2e tests (B6.3).

`run_training_time` aggregates the B5 raw-loss manifest under
`output_root` and returns per-(dataset, model, hardware_tier)
counters. The CLI (`benchmarks.run`) is responsible for rendering
`output_root/training_time.md` from the same manifest via
`benchmarks.report.training_time.render_from_dir`. These tests
exercise the driver against the synthetic fake-registry and
verify the renderer's output through `render_from_dir` directly.
"""

from pathlib import Path

import pytest
from benchmarks.config import BenchmarkConfig, ExperimentSpec
from benchmarks.datasets._base import PanelDataset
from benchmarks.experiments import (
    TrainingTimeExperimentResult,
    build_run_environment,
    run_raw_loss,
    run_training_time,
)
from benchmarks.report.training_time import render_from_dir as render_training_time_from_dir

from tests.benchmarks._fakes import register_all_fakes_and_get_panels


@pytest.fixture
def fake_registry() -> list[PanelDataset]:
    """Register the synthetic registry for the e2e flow."""
    return register_all_fakes_and_get_panels()


def _make_config(
    *,
    datasets: tuple[str, ...],
    models: tuple[str, ...],
    seeds: tuple[int, ...],
    output_dir: Path,
    cache_dir: Path,
    experiment_kinds: tuple[str, ...] = ("raw_loss", "training_time"),
) -> BenchmarkConfig:
    return BenchmarkConfig(
        datasets=datasets,
        models=models,
        experiments=tuple(
            ExperimentSpec(kind=k, seeds=seeds)  # type: ignore[arg-type]
            for k in experiment_kinds
        ),
        output_dir=output_dir,
        cache_dir=cache_dir,
    )


def test_run_training_time_aggregates_from_b5_manifest(
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
    output_root = tmp_path / "out"
    run_raw_loss(config, output_root=output_root, env=env)
    result = run_training_time(config, output_root=output_root, env=env)
    assert isinstance(result, TrainingTimeExperimentResult)
    # Single (dataset, model, hardware_tier) group with 5 OK cells.
    assert result.groups_evaluated == 1
    assert result.groups_fully_skipped == 0
    # The driver returns counters only; rendering goes through the
    # report helper (which the CLI calls). Verify the renderer
    # consumes the same manifest cleanly.
    md = render_training_time_from_dir(output_root)
    assert "Training-time report" in md
    assert "fake_binary" in md
    assert "fake_constant_binary" in md


def test_run_training_time_raises_when_no_training_time_experiment(
    fake_registry: list[PanelDataset], tmp_path: Path
) -> None:
    del fake_registry
    # Config carries only `raw_loss`; the training-time driver
    # refuses cleanly.
    config = _make_config(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        seeds=(0,),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        experiment_kinds=("raw_loss",),
    )
    env = build_run_environment(profile="smoke")
    output_root = tmp_path / "out"
    run_raw_loss(config, output_root=output_root, env=env)
    with pytest.raises(ValueError, match="no training_time"):
        run_training_time(config, output_root=output_root, env=env)


def test_run_training_time_raises_when_b5_manifest_is_empty(
    fake_registry: list[PanelDataset], tmp_path: Path
) -> None:
    del fake_registry
    # No raw_loss run before training_time; manifest is empty.
    config = _make_config(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        seeds=(0,),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    (tmp_path / "out").mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="manifest"):
        run_training_time(config, output_root=tmp_path / "out", env=env)


def test_run_training_time_aggregates_across_datasets(
    fake_registry: list[PanelDataset], tmp_path: Path
) -> None:
    del fake_registry
    config = _make_config(
        datasets=("fake_binary", "fake_regression_point"),
        models=("fake_constant_binary", "fake_constant_regressor"),
        seeds=(0,),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    output_root = tmp_path / "out"
    run_raw_loss(config, output_root=output_root, env=env)
    result = run_training_time(config, output_root=output_root, env=env)
    # 2 datasets * 1 applicable model each = 2 groups evaluated;
    # the cross-product (binary x regressor + regression x binary)
    # produces task_type_mismatch skips that land in fully-skipped
    # groups.
    assert result.groups_evaluated == 2
    assert result.groups_fully_skipped == 2
    md = render_training_time_from_dir(output_root)
    assert "fake_binary" in md
    assert "fake_regression_point" in md
    assert "Skipped groups" in md
