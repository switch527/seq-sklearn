"""Phase B8 HPO-uplift experiment e2e tests (B6.4).

`run_hpo_uplift` drives the tuned arm against the registered HPO
search-space modules; the default arm is the B5 manifest. These
tests register a tiny fake HPO space for the `sklearn_passthrough`
family so the synthetic fakes can exercise the full B5 -> B8 flow
without requiring the seq_sklearn TFT estimator at smoke-test
speed.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
import pytest
from benchmarks.config import BenchmarkConfig, DatasetSpec, ExperimentSpec, ModelSpec
from benchmarks.datasets._base import PanelDataset
from benchmarks.experiments import (
    HPOUpliftExperimentResult,
    build_run_environment,
    run_hpo_uplift,
    run_raw_loss,
)
from benchmarks.hpo._base import HPO_REGISTRY, HPOSpace, register_hpo_space
from benchmarks.manifest import load_run
from benchmarks.report.hpo_uplift import (
    aggregate_hpo_uplift,
    render_from_dir,
    render_hpo_uplift_markdown,
)

from tests.benchmarks._fakes import register_all_fakes_and_get_panels


def _passthrough_sampler(
    trial: optuna.trial.BaseTrial,
    model_spec: ModelSpec,
    dataset_spec: DatasetSpec,
) -> dict[str, Any]:
    """Fake HPO sampler for the `sklearn_passthrough` family.

    The fake adapters ignore `hyperparameters`, so the sampled
    `learning_rate` is a structural placeholder: it forces the
    Optuna study to actually take a sample per trial. The trial's
    value space is wide enough that the TPE sampler explores
    multiple cells across 3 trials.
    """
    del model_spec, dataset_spec
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
    }


@pytest.fixture
def fake_registry_and_hpo() -> Iterator[list[PanelDataset]]:
    """Register the synthetic dataset/model registry AND a fake HPO
    space for the `sklearn_passthrough` family. The HPO registry is
    process-global, so the cleanup function unregisters at the end
    of every test that uses this fixture."""
    panels = register_all_fakes_and_get_panels()
    space = HPOSpace(
        family="sklearn_passthrough",
        search_space_size=1,
        description="fake passthrough space; one learning_rate float",
    )
    register_hpo_space(space, _passthrough_sampler)
    yield panels
    # The registry is module-global; tear it down so a later test
    # (test_hpo_seq_sklearn.py's `test_get_hpo_space_unknown_family_raises_typed`,
    # in particular) doesn't see the fake registration. The
    # `register_hpo_space` API has no `unregister` to keep its
    # surface minimal, so we mutate the dict directly here.
    HPO_REGISTRY.pop("sklearn_passthrough", None)
    from benchmarks.hpo._base import _HPO_SAMPLERS

    _HPO_SAMPLERS.pop("sklearn_passthrough", None)


def _make_config(
    *,
    datasets: tuple[str, ...],
    models: tuple[str, ...],
    seeds: tuple[int, ...],
    output_dir: Path,
    cache_dir: Path,
    n_trials: int = 3,
    timeout_seconds: float = 30.0,
    experiment_kinds: tuple[str, ...] = ("raw_loss", "hpo_uplift"),
) -> BenchmarkConfig:
    experiments = []
    for kind in experiment_kinds:
        if kind == "hpo_uplift":
            experiments.append(
                ExperimentSpec(
                    kind=kind,  # type: ignore[arg-type]
                    seeds=seeds,
                    n_trials=n_trials,
                    timeout_seconds=timeout_seconds,
                )
            )
        else:
            experiments.append(ExperimentSpec(kind=kind, seeds=seeds))  # type: ignore[arg-type]
    return BenchmarkConfig(
        datasets=datasets,
        models=models,
        experiments=tuple(experiments),
        output_dir=output_dir,
        cache_dir=cache_dir,
    )


# --- e2e flow ----------------------------------------------------------------


def test_run_hpo_uplift_writes_tuned_rows(
    fake_registry_and_hpo: list[PanelDataset], tmp_path: Path
) -> None:
    del fake_registry_and_hpo
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
    result = run_hpo_uplift(config, output_root=output_root, env=env)
    assert isinstance(result, HPOUpliftExperimentResult)
    # Single (dataset, model, seed) group with 5 folds.
    assert result.cells_attempted == 5
    assert result.cells_skipped_hpo_family_not_registered == 0
    # The manifest now carries both default and tuned variants.
    manifest = load_run(output_root)
    assert (manifest["variant"] == "default").any()
    assert (manifest["variant"] == "tuned").any()


def test_run_hpo_uplift_raises_when_no_hpo_uplift_experiment(
    fake_registry_and_hpo: list[PanelDataset], tmp_path: Path
) -> None:
    del fake_registry_and_hpo
    config = _make_config(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        seeds=(0,),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        experiment_kinds=("raw_loss",),
    )
    env = build_run_environment(profile="smoke")
    with pytest.raises(ValueError, match="no hpo_uplift"):
        run_hpo_uplift(config, output_root=tmp_path / "out", env=env)


def test_run_hpo_uplift_skips_unregistered_family(
    fake_registry_and_hpo: list[PanelDataset], tmp_path: Path
) -> None:
    """When a model's family has no registered HPO space, the
    driver records a typed `skipped_reason` instead of crashing."""
    del fake_registry_and_hpo
    # Unregister the passthrough HPO space so `fake_constant_binary`
    # has no search space available.
    HPO_REGISTRY.pop("sklearn_passthrough", None)
    from benchmarks.hpo._base import _HPO_SAMPLERS

    _HPO_SAMPLERS.pop("sklearn_passthrough", None)

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
    result = run_hpo_uplift(config, output_root=output_root, env=env)
    assert result.cells_attempted == 0
    assert result.cells_skipped_hpo_family_not_registered == 5


def test_run_hpo_uplift_smoke_profile_with_zero_budget_records_skip(
    fake_registry_and_hpo: list[PanelDataset], tmp_path: Path
) -> None:
    """A smoke profile without an n_trials override produces 0
    completed trials; the driver records a typed skip and writes
    the row so the report can footnote it."""
    del fake_registry_and_hpo
    # Config with no n_trials override -> profile default kicks in
    # (smoke -> 0 trials).
    config = _make_config(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        seeds=(0,),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        n_trials=0,
        timeout_seconds=0.1,  # >0 required by the ExperimentSpec validator
    )
    env = build_run_environment(profile="smoke")
    output_root = tmp_path / "out"
    run_raw_loss(config, output_root=output_root, env=env)
    result = run_hpo_uplift(config, output_root=output_root, env=env)
    assert result.cells_attempted == 0
    assert result.cells_skipped_hpo_budget_zero == 5


def test_run_hpo_uplift_resume_skips_completed_cells(
    fake_registry_and_hpo: list[PanelDataset], tmp_path: Path
) -> None:
    del fake_registry_and_hpo
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
    # First pass: 5 cells attempted.
    first = run_hpo_uplift(config, output_root=output_root, env=env)
    assert first.cells_attempted == 5
    # Second pass: all cells already complete (sentinel hit).
    second = run_hpo_uplift(config, output_root=output_root, env=env)
    assert second.cells_attempted == 0
    assert second.cells_already_complete == 5


# --- report renderer --------------------------------------------------------


def test_render_hpo_uplift_empty_manifest_returns_no_results() -> None:
    md = render_hpo_uplift_markdown(pd.DataFrame())
    assert md.startswith("# HPO-uplift report")
    assert "_No results" in md


def test_render_hpo_uplift_renders_delta_row_after_e2e(
    fake_registry_and_hpo: list[PanelDataset], tmp_path: Path
) -> None:
    del fake_registry_and_hpo
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
    run_hpo_uplift(config, output_root=output_root, env=env)
    md = render_from_dir(output_root, reference_model="fake_constant_binary")
    assert "HPO-uplift report" in md
    assert "fake_binary" in md
    assert "fake_constant_binary" in md
    # Δ column present.
    assert "delta" in md
    # Search-space size column reflects the registered passthrough
    # space's `search_space_size=1`.
    assert "search_space_size" in md


def test_aggregate_hpo_uplift_returns_empty_on_empty_manifest() -> None:
    assert aggregate_hpo_uplift(pd.DataFrame()) == []


def test_aggregate_hpo_uplift_emits_default_only_sentinel(
    fake_registry_and_hpo: list[PanelDataset], tmp_path: Path
) -> None:
    """If only the default arm ran, the aggregator emits a
    `default_only=True` sentinel row so the renderer's footnote
    can list the (dataset, model) group."""
    del fake_registry_and_hpo
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
    manifest = load_run(output_root)
    rows = aggregate_hpo_uplift(manifest)
    assert len(rows) == 1
    row = rows[0]
    assert row.default_only is True
    assert row.delta is None
    assert row.n_cells_paired == 0
