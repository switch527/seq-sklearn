"""Phase B8 HPO-uplift experiment e2e tests (B6.4).

`run_hpo_uplift` drives the tuned arm against the registered HPO
search-space modules; the default arm is the B5 manifest. These
tests register a tiny fake HPO space for the `sklearn_passthrough`
family so the synthetic fakes can exercise the full B5 -> B8 flow
without requiring the seq_sklearn TFT estimator at smoke-test
speed.
"""

from pathlib import Path
from typing import Any

# Eager import of the adapter family so the `isolated_registry`
# autouse fixture snapshots `_ADAPTER_FACTORIES` AFTER every
# B10/B12 factory is registered. Without this, under
# `pytest-randomly` reordering, if this file runs before any
# test that triggers `benchmarks.adapters` as a side-effect,
# the snapshot would not contain the TSC factories. The
# B12 test below pops `rocket_classifier` and re-registers a
# fake; teardown would then restore an empty snapshot,
# permanently deleting the TSC adapters for the rest of the
# session.
import benchmarks.adapters  # noqa: F401  # pyright: ignore[reportUnusedImport]
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
def fake_registry_and_hpo() -> list[PanelDataset]:
    """Register the synthetic dataset/model registry AND a fake HPO
    space for the `sklearn_passthrough` family. The autouse
    `isolated_registry` fixture in conftest.py snapshots+restores
    `HPO_REGISTRY` (the dual-dict was collapsed in B9 arch-I5),
    so this fixture does not need a manual teardown."""
    panels = register_all_fakes_and_get_panels()
    space = HPOSpace(
        family="sklearn_passthrough",
        search_space_size=1,
        description="fake passthrough space; one learning_rate float",
    )
    register_hpo_space(space, _passthrough_sampler)
    return panels


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
    # has no search space available; conftest's `isolated_registry`
    # restores both dicts after the test.
    HPO_REGISTRY.pop("sklearn_passthrough", None)

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


# --- qa-C1: regression_quantile early skip ----------------------------------


def test_run_hpo_uplift_skips_regression_quantile_cells(
    fake_registry_and_hpo: list[PanelDataset], tmp_path: Path
) -> None:
    """B5-followup carryover: regression_quantile cells skip cleanly
    with the typed `regression_quantile_b5_followup` reason. The
    counter at the driver's tally site must NOT lump them into
    `cells_skipped_adapter_error` (which would silently merge
    unfitted cells into the Δ table)."""
    del fake_registry_and_hpo
    # Register a passthrough HPO space for sklearn_passthrough so
    # the quantile model is otherwise eligible.
    config = _make_config(
        datasets=("fake_regression_quantile",),
        models=("fake_quantile_regressor",),
        seeds=(0,),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    output_root = tmp_path / "out"
    run_raw_loss(config, output_root=output_root, env=env)
    result = run_hpo_uplift(config, output_root=output_root, env=env)
    # All 5 folds skip with the quantile-followup reason; nothing
    # else (adapter_error in particular) should fire.
    assert result.cells_attempted == 0
    assert result.cells_skipped_quantile_followup == 5
    assert result.cells_skipped_adapter_error == 0
    # The written rows carry the typed reason.
    manifest = load_run(output_root)
    tuned_rows = manifest.loc[manifest["variant"] == "tuned"]
    assert (
        tuned_rows["skipped_reason"]
        .astype(str)
        .str.startswith("regression_quantile_b5_followup")
        .all()
    )


# --- qa-C2: hpo_all_trials_pruned -------------------------------------------


def _always_raising_sampler(
    trial: optuna.trial.BaseTrial, model_spec: Any, dataset_spec: Any
) -> dict[str, Any]:
    """Sampler whose every trial body raises `ValueError` (the
    legacy "illegal cell" signal the driver converts to
    `TrialPruned`)."""
    del trial, model_spec, dataset_spec
    raise ValueError("forced prune for test")


def test_run_hpo_uplift_records_all_trials_pruned_when_sampler_always_raises(
    tmp_path: Path,
) -> None:
    """Force every trial to prune (sampler raises ValueError ->
    TrialPruned) so `best_hyperparameters` stays None. The driver
    must record the cell with `hpo_all_trials_pruned` and zero
    completed trials, never crash on a None-config refit."""
    register_all_fakes_and_get_panels()
    # Replace the default passthrough HPO space with one whose
    # sampler always prunes.
    HPO_REGISTRY.pop("sklearn_passthrough", None)
    space = HPOSpace(
        family="sklearn_passthrough",
        search_space_size=1,
        description="always-pruning fake",
    )
    register_hpo_space(space, _always_raising_sampler)

    config = _make_config(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        seeds=(0,),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        n_trials=2,
        timeout_seconds=30.0,
    )
    env = build_run_environment(profile="smoke")
    output_root = tmp_path / "out"
    run_raw_loss(config, output_root=output_root, env=env)
    result = run_hpo_uplift(config, output_root=output_root, env=env)
    # All 5 folds get the all-pruned skip; no cell attempts a refit.
    assert result.cells_attempted == 0
    assert result.cells_skipped_hpo_all_trials_pruned == 5


# --- qa-C4: tuned_only sentinel ---------------------------------------------


def test_aggregate_hpo_uplift_emits_tuned_only_sentinel() -> None:
    """A manifest with only `variant="tuned"` OK rows must yield a
    `tuned_only=True` sentinel so the renderer footnote can list
    the (dataset, model) group. arch-I3 ALSO requires the sentinel
    to carry a populated `tuned_mean` so the Friedman matrix
    includes the cell."""
    manifest = pd.DataFrame(
        [
            {
                "library_git_sha": "abc",
                "run_id": "r1",
                "started_at_utc": "2026-01-01T00:00:00+00:00",
                "dataset_name": "ds_x",
                "model_name": "m_y",
                "seed": 0,
                "variant": "tuned",
                "task_type": "binary",
                "fold_index": 0,
                "n_folds": 1,
                "split_fingerprint": "0" * 64,
                "l_resolved": 8,
                "hardware_tier": "cpu",
                "profile": "smoke",
                "hpo_tier": "custom",
                "skipped_reason": None,
                "log_loss": 0.42,
                "hpo_n_trials_completed": 2,
                "hpo_search_space_size": 1,
            }
        ]
    )
    rows = aggregate_hpo_uplift(manifest)
    assert len(rows) == 1
    row = rows[0]
    assert row.tuned_only is True
    assert row.delta is None
    assert row.tuned_mean == pytest.approx(0.42)


# --- qa-I1: _resolve_hpo_budget override branches ---------------------------


def test_resolve_hpo_budget_n_trials_only_override_sets_custom_tier() -> None:
    from benchmarks.config import ExperimentSpec
    from benchmarks.experiments.hpo_uplift import (
        _resolve_hpo_budget,  # pyright: ignore[reportPrivateUsage]
    )

    specs = [ExperimentSpec(kind="hpo_uplift", seeds=(0,), n_trials=5)]
    n_trials, timeout_seconds, tier, n_startup_trials, n_jobs = _resolve_hpo_budget(specs, "smoke")
    assert n_trials == 5
    # smoke profile default timeout is 0.0; override flips tier.
    assert tier == "custom"
    # timeout untouched -> smoke default (0.0).
    assert timeout_seconds == 0.0
    # n_startup_trials not overridden -> None (use Optuna default).
    assert n_startup_trials is None
    assert n_jobs is None


def test_resolve_hpo_budget_n_jobs_override_sets_custom_tier() -> None:
    from benchmarks.config import ExperimentSpec
    from benchmarks.experiments.hpo_uplift import (
        _resolve_hpo_budget,  # pyright: ignore[reportPrivateUsage]
    )

    specs = [ExperimentSpec(kind="hpo_uplift", seeds=(0,), hpo_n_jobs=2)]
    (
        _n_trials,
        _timeout_seconds,
        tier,
        _n_startup_trials,
        n_jobs,
    ) = _resolve_hpo_budget(specs, "smoke")
    # smoke profile defaults: n_trials=0, timeout=0.0; n_jobs override
    # alone flips tier to custom.
    assert tier == "custom"
    assert n_jobs == 2


def test_resolve_hpo_budget_timeout_only_override_sets_custom_tier() -> None:
    from benchmarks.config import ExperimentSpec
    from benchmarks.experiments.hpo_uplift import (
        _resolve_hpo_budget,  # pyright: ignore[reportPrivateUsage]
    )

    specs = [ExperimentSpec(kind="hpo_uplift", seeds=(0,), timeout_seconds=60.0)]
    n_trials, timeout_seconds, tier, n_startup_trials, n_jobs = _resolve_hpo_budget(
        specs, "standard"
    )
    # n_trials untouched -> standard default (25).
    assert n_trials == 25
    assert timeout_seconds == 60.0
    assert tier == "custom"
    assert n_startup_trials is None
    assert n_jobs is None


def test_resolve_hpo_budget_unknown_profile_falls_back_to_zero() -> None:
    from benchmarks.experiments.hpo_uplift import (
        _resolve_hpo_budget,  # pyright: ignore[reportPrivateUsage]
    )

    n_trials, timeout_seconds, tier, n_startup_trials, n_jobs = _resolve_hpo_budget(
        [], "nonexistent_profile"
    )
    assert n_trials == 0
    assert timeout_seconds == 0.0
    assert tier == "none"
    assert n_startup_trials is None
    assert n_jobs is None


# --- qa-I4: reference_model absent from matrix ------------------------------


def test_aggregate_hpo_uplift_emits_paired_but_no_valid_loss_sentinel() -> None:
    """R2 qa-I1 + arch-I4: when both default and tuned arms have OK
    rows for the same (seed, fold) key but every paired primary
    loss is missing (e.g., a refit crashed mid-fit so
    log_loss is None on the tuned row), the aggregator emits a
    `paired_but_no_valid_loss=True` sentinel. The footnote then
    surfaces the group; previously the row vanished from both the
    per-dataset block AND the footnote (silent data loss)."""
    rows = [
        # Default OK row with log_loss=None to simulate a missing
        # primary loss on a non-skipped cell.
        {
            "library_git_sha": "abc",
            "run_id": "r1",
            "started_at_utc": "2026-01-01T00:00:00+00:00",
            "dataset_name": "ds_x",
            "model_name": "m_y",
            "seed": 0,
            "variant": "default",
            "task_type": "binary",
            "fold_index": 0,
            "n_folds": 1,
            "split_fingerprint": "0" * 64,
            "l_resolved": 8,
            "hardware_tier": "cpu",
            "profile": "smoke",
            "hpo_tier": "none",
            "skipped_reason": None,
            "log_loss": None,
        },
        {
            "library_git_sha": "abc",
            "run_id": "r1",
            "started_at_utc": "2026-01-01T00:00:00+00:00",
            "dataset_name": "ds_x",
            "model_name": "m_y",
            "seed": 0,
            "variant": "tuned",
            "task_type": "binary",
            "fold_index": 0,
            "n_folds": 1,
            "split_fingerprint": "0" * 64,
            "l_resolved": 8,
            "hardware_tier": "cpu",
            "profile": "smoke",
            "hpo_tier": "custom",
            "skipped_reason": None,
            "log_loss": None,
        },
    ]
    manifest = pd.DataFrame(rows)
    out = aggregate_hpo_uplift(manifest)
    assert len(out) == 1
    row = out[0]
    assert row.paired_but_no_valid_loss is True
    assert row.delta is None
    assert row.n_cells_paired == 1
    # The footnote must render the typed reason for the group.
    md = render_hpo_uplift_markdown(manifest)
    assert "both arms present" in md
    assert "every paired row had missing primary loss" in md


def test_render_hpo_uplift_markdown_renders_pairwise_block_when_matrix_has_three_models() -> None:
    """R2 qa-I3: the Friedman pairwise table only renders when
    `friedman.pairwise` is non-empty (>= 3 models in the matrix
    per scipy's friedmanchisquare minimum). The default e2e
    fixture has 1 model so the pairwise renderer's table-emit
    branch is untested unless a hand-crafted manifest reaches
    it. Pin the block + Holm column by passing three (model,
    dataset) cells with two datasets each."""
    base = {
        "library_git_sha": "abc",
        "run_id": "r1",
        "started_at_utc": "2026-01-01T00:00:00+00:00",
        "task_type": "binary",
        "fold_index": 0,
        "n_folds": 1,
        "seed": 0,
        "split_fingerprint": "0" * 64,
        "l_resolved": 8,
        "hardware_tier": "cpu",
        "profile": "smoke",
        "hpo_tier": "custom",
        "variant": "tuned",
        "skipped_reason": None,
    }
    rows: list[dict[str, Any]] = []
    losses = {
        ("m_ref", "ds_a"): 0.50,
        ("m_ref", "ds_b"): 0.55,
        ("m_a", "ds_a"): 0.40,
        ("m_a", "ds_b"): 0.45,
        ("m_b", "ds_a"): 0.30,
        ("m_b", "ds_b"): 0.35,
    }
    for (model_name, dataset_name), loss in losses.items():
        row = {**base, "model_name": model_name, "dataset_name": dataset_name, "log_loss": loss}
        rows.append(row)
    manifest = pd.DataFrame(rows)
    md = render_hpo_uplift_markdown(manifest, reference_model="m_ref")
    # The Friedman block renders the χ² stat + the Holm pairwise
    # table; the latter has one row per non-reference model.
    assert "Matrix-level Friedman + Holm" in md
    assert "Friedman χ²" in md
    assert "Holm-adjusted pairwise" in md
    assert "m_ref vs m_a" in md
    assert "m_ref vs m_b" in md


def test_time_to_best_seconds_returns_none_on_no_best_trial() -> None:
    """R2 code-I1 / qa-I2: `_time_to_best_seconds` short-circuits
    to None when no trial completed (best_trial_number is None) or
    when the study has no trials at all. Pin the no-best branch
    without spinning up a study with synthetic timestamps."""
    from benchmarks.experiments.hpo_uplift import (
        _time_to_best_seconds,  # pyright: ignore[reportPrivateUsage]
    )

    empty_study = optuna.create_study(direction="minimize")
    assert _time_to_best_seconds(empty_study, None) is None
    assert _time_to_best_seconds(empty_study, 0) is None


def test_render_hpo_uplift_markdown_absent_reference_model_emits_skip_note(
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
    md = render_from_dir(output_root, reference_model="absent_model_name")
    # The Friedman block emits a skip note when the reference is
    # not in the matrix.
    assert "Friedman omnibus skipped" in md
    assert "absent_model_name" in md


# --- B12 optional-dep skip wire-up ------------------------------------------


def test_run_hpo_uplift_records_optional_dep_skip_when_aeon_missing(
    fake_registry_and_hpo: list[PanelDataset], tmp_path: Path
) -> None:
    """B12 touch points (g)-(j): an HPO cell whose adapter raises
    `OptionalDependencyMissingError` at trial-setup time skips
    cleanly via the typed `optional_dep_missing` reason rather
    than burning the budget on a missing-dep error that would
    never resolve, and the summary counter is wired through.

    Uses a fake TSC adapter that raises the typed error at fit
    time + a fake TSC HPO space so the test doesn't depend on
    aeon being installed."""
    from dataclasses import dataclass, field
    from typing import ClassVar, Self, cast

    import numpy as np
    from benchmarks.adapters._base import (
        OptionalDependencyMissingError,
        QuantilesUnsupportedError,
        SeqSklearnAdapter,
    )
    from benchmarks.config import TaskType
    from benchmarks.registry.models import register_adapter_factory

    del fake_registry_and_hpo  # the fake registry is loaded; we add a TSC fake too

    @dataclass
    class _FakeTSCAdapter:
        # Re-uses the `rocket_classifier` registration name so the
        # B12 TSC HPO sampler dispatches to a real per-classifier
        # sampler; the conftest's `isolated_registry` fixture
        # restores the real adapter after the test.
        name: ClassVar[str] = "rocket_classifier"
        family: ClassVar[str] = "tsc"
        task_types: ClassVar[tuple[TaskType, ...]] = ("binary",)
        supports_proba: ClassVar[bool] = True

        spec: DatasetSpec
        hyperparameters: dict[str, Any] = field(default_factory=dict)

        def fit(self, panel: pd.DataFrame, y: np.ndarray) -> Self:
            del panel, y
            raise OptionalDependencyMissingError("aeon")

        def predict(self, panel: pd.DataFrame) -> np.ndarray:
            return np.zeros(len(panel))

        def predict_proba(self, panel: pd.DataFrame) -> np.ndarray:
            return np.full((len(panel), 2), 0.5)

        def predict_quantiles(self, panel: pd.DataFrame) -> np.ndarray:
            del panel
            raise QuantilesUnsupportedError(self.name)

    def _factory(
        *,
        spec: DatasetSpec,
        hyperparameters: dict[str, Any] | None = None,
    ) -> SeqSklearnAdapter:
        return cast(
            SeqSklearnAdapter,
            _FakeTSCAdapter(spec=spec, hyperparameters=hyperparameters or {}),
        )

    # The real `rocket_classifier` factory is already registered
    # (B12 ships it); pop it so we can install the fake one.
    # The conftest's `isolated_registry` autouse fixture restores
    # `_ADAPTER_FACTORIES` on teardown.
    from benchmarks.registry import models as _models_reg

    _models_reg._ADAPTER_FACTORIES.pop("rocket_classifier", None)  # type: ignore[attr-defined]
    register_adapter_factory("rocket_classifier", _factory)

    config = _make_config(
        datasets=("fake_binary",),
        models=("rocket_classifier",),
        seeds=(0,),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    output_root = tmp_path / "out"
    # `run_raw_loss` runs first and also surfaces the typed skip
    # (B5 touch points); the HPO uplift then iterates over the
    # tuned arm and uses B8's wire-up to produce the same shape.
    run_raw_loss(config, output_root=output_root, env=env)
    result = run_hpo_uplift(config, output_root=output_root, env=env)
    assert result.cells_skipped_optional_dep_missing > 0
    assert result.cells_skipped_adapter_error == 0
    # Manifest: every tuned row carries the typed skip reason.
    manifest = load_run(output_root)
    tuned_tsc = manifest.loc[
        (manifest["variant"] == "tuned") & (manifest["model_name"] == "rocket_classifier")
    ]
    assert len(tuned_tsc) > 0
    for _, row in tuned_tsc.iterrows():
        assert str(row["skipped_reason"]).startswith("optional_dep_missing: aeon")
