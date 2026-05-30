"""Phase B13 bootstrap-rollup aggregator e2e tests.

The aggregator reads B5 manifest + per-cell predictions shards,
re-resolves entity_id via the registered loader, and emits a
RollupRow per (dataset, model, task_type). These tests cover the
classification + regression paths, mixed-skip + all-skipped
sentinels, the freshness fingerprint contract, the loader
row-count drift gate (Gemini-C1), and the row-count ceiling
gate (Gemini-C2).
"""

import importlib.metadata
from pathlib import Path
from typing import Any

import benchmarks.adapters  # type: ignore[reportUnusedImport]  # noqa: F401  --  side-effect import
import pytest
from benchmarks.bootstrap_manifest import load_rollup, rollup_path
from benchmarks.config import BenchmarkConfig, ExperimentSpec
from benchmarks.experiments import build_run_environment, run_raw_loss
from benchmarks.report.bootstrap_rollup import (
    RawRollupError,
    aggregate_bootstrap_rollup,
)
from benchmarks.run_manifest import build_run_manifest

from tests.benchmarks._fakes import register_all_fakes_and_get_panels


def _make_config(
    *,
    datasets: tuple[str, ...],
    models: tuple[str, ...],
    output_dir: Path,
    cache_dir: Path,
    bootstrap_rollup_enabled: bool = True,
    bootstrap_n_resamples: int | None = None,
) -> BenchmarkConfig:
    return BenchmarkConfig(
        datasets=datasets,
        models=models,
        experiments=(
            ExperimentSpec(
                kind="raw_loss",
                seeds=(0,),
                bootstrap_rollup_enabled=bootstrap_rollup_enabled,
                bootstrap_n_resamples=bootstrap_n_resamples,
            ),
        ),
        output_dir=output_dir,
        cache_dir=cache_dir,
    )


def _build_run_manifest_for(config: BenchmarkConfig, output_root: Path) -> Any:
    return build_run_manifest(
        config=config,
        run_id="test-run",
        library_git_sha="0" * 40,
        profile="smoke",
        hardware_tier="cpu",
        output_root=output_root,
    )


# --- classification happy path ---------------------------------------------


def test_aggregate_bootstrap_rollup_classification_emits_rollup_row(
    tmp_path: Path,
) -> None:
    """e2e with fake_binary + fake_constant_binary; assert one
    RollupRow with finite (mean, ci_lo, ci_hi), positive
    n_entities, n_rows, and bootstrap_rng_algorithm == "PCG64"."""
    register_all_fakes_and_get_panels()
    config = _make_config(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    run_raw_loss(config, output_root=output_root, env=env)
    manifest = _build_run_manifest_for(config, output_root)
    rows = aggregate_bootstrap_rollup(config, output_root=output_root, env=env, manifest=manifest)
    assert len(rows) == 1
    row = rows[0]
    assert row.dataset_name == "fake_binary"
    assert row.model_name == "fake_constant_binary"
    assert row.task_type == "binary"
    assert row.primary_metric == "log_loss"
    assert row.n_rows > 0
    assert row.n_entities > 0
    assert row.primary_metric_mean is not None
    assert row.primary_metric_ci_lo is not None
    assert row.primary_metric_ci_hi is not None
    assert row.bootstrap_rng_algorithm == "PCG64"
    assert row.bootstrap_skipped_reason is None
    # B21 R1 qa-I1 closure: pin the audit fields end-to-end so a
    # mutation that replaces the late-bound constant reference with
    # a hardcoded string in the aggregator fails immediately. The
    # fallback reason is type-checked against the documented values
    # so a small-N happy-path fixture that triggers `p0_at_edge`
    # (BCa fallback to percentile on concentrated distributions)
    # still passes.
    assert row.bootstrap_ci_method == "bca"
    assert row.bootstrap_ci_fallback_reason in (None, "p0_at_edge", "a_overshoot")


# --- regression path ------------------------------------------------------


def test_aggregate_bootstrap_rollup_regression_emits_rollup_row(
    tmp_path: Path,
) -> None:
    """Regression path: primary_metric == "rmse" and the bootstrap
    primitive's per-resample sqrt produces a finite RMSE CI."""
    register_all_fakes_and_get_panels()
    config = _make_config(
        datasets=("fake_regression_point",),
        models=("fake_constant_regressor",),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    run_raw_loss(config, output_root=output_root, env=env)
    manifest = _build_run_manifest_for(config, output_root)
    rows = aggregate_bootstrap_rollup(config, output_root=output_root, env=env, manifest=manifest)
    assert len(rows) == 1
    row = rows[0]
    assert row.task_type == "regression_point"
    assert row.primary_metric == "rmse"
    assert row.primary_metric_mean is not None
    assert row.primary_metric_mean >= 0  # RMSE is non-negative


# --- all-skipped sentinel --------------------------------------------------


def test_aggregate_bootstrap_rollup_all_cells_skipped_emits_sentinel(
    tmp_path: Path,
) -> None:
    """A (dataset, model) where every cell skipped emits a
    sentinel RollupRow with mean / ci_lo / ci_hi = None and
    n_cells_evaluated = 0."""
    register_all_fakes_and_get_panels()
    # `fake_crashing` triggers an adapter_error on every cell.
    config = _make_config(
        datasets=("fake_binary",),
        models=("fake_crashing",),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    run_raw_loss(config, output_root=output_root, env=env)
    manifest = _build_run_manifest_for(config, output_root)
    rows = aggregate_bootstrap_rollup(config, output_root=output_root, env=env, manifest=manifest)
    assert len(rows) == 1
    row = rows[0]
    assert row.n_cells_evaluated == 0
    assert row.n_skipped_cells > 0
    assert row.primary_metric_mean is None
    assert row.primary_metric_ci_lo is None
    assert row.primary_metric_ci_hi is None
    assert row.bootstrap_skipped_reason is not None


# --- rng_algorithm + numpy_version pinned ---------------------------------


def test_aggregate_bootstrap_rollup_records_numpy_version_as_string(
    tmp_path: Path,
) -> None:
    """`bootstrap_numpy_version` equals the live
    `importlib.metadata.version("numpy")` and is a non-empty
    string."""
    register_all_fakes_and_get_panels()
    config = _make_config(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    run_raw_loss(config, output_root=output_root, env=env)
    manifest = _build_run_manifest_for(config, output_root)
    rows = aggregate_bootstrap_rollup(config, output_root=output_root, env=env, manifest=manifest)
    assert rows[0].bootstrap_numpy_version == importlib.metadata.version("numpy")
    assert rows[0].bootstrap_numpy_version


# --- manifest_fingerprint recorded ----------------------------------------


def test_aggregate_bootstrap_rollup_records_manifest_fingerprint(
    tmp_path: Path,
) -> None:
    """Every emitted RollupRow carries the live manifest's
    fingerprint at aggregation time (Gemini-C3)."""
    register_all_fakes_and_get_panels()
    config = _make_config(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    run_raw_loss(config, output_root=output_root, env=env)
    manifest = _build_run_manifest_for(config, output_root)
    rows = aggregate_bootstrap_rollup(config, output_root=output_root, env=env, manifest=manifest)
    expected = manifest.fingerprint()
    assert all(row.manifest_fingerprint == expected for row in rows)


# --- profile + per-experiment override ------------------------------------


def test_aggregate_bootstrap_rollup_smoke_profile_halves_n_resamples(
    tmp_path: Path,
) -> None:
    """When `profile="smoke"` and no ExperimentSpec override is
    set, n_resamples = 5_000."""
    register_all_fakes_and_get_panels()
    config = _make_config(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    run_raw_loss(config, output_root=output_root, env=env)
    manifest = _build_run_manifest_for(config, output_root)
    rows = aggregate_bootstrap_rollup(config, output_root=output_root, env=env, manifest=manifest)
    assert rows[0].bootstrap_n_resamples == 5_000


def test_aggregate_bootstrap_rollup_experiment_spec_override_takes_precedence(
    tmp_path: Path,
) -> None:
    """An ExperimentSpec(bootstrap_n_resamples=2000) override
    beats the profile default (smoke=5000)."""
    register_all_fakes_and_get_panels()
    config = _make_config(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        bootstrap_n_resamples=2_000,
    )
    env = build_run_environment(profile="smoke")
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    run_raw_loss(config, output_root=output_root, env=env)
    manifest = _build_run_manifest_for(config, output_root)
    rows = aggregate_bootstrap_rollup(config, output_root=output_root, env=env, manifest=manifest)
    assert rows[0].bootstrap_n_resamples == 2_000


# --- loader row-count drift gate (Gemini-C1) ------------------------------


def test_aggregate_bootstrap_rollup_loader_row_count_drift_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loader non-determinism: if the re-loaded panel is SHORTER
    than the predictions shard's max(panel_row_index)+1, the
    aggregator raises `RawRollupError` rather than silently
    mis-attributing entities."""
    register_all_fakes_and_get_panels()
    config = _make_config(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    run_raw_loss(config, output_root=output_root, env=env)

    # Monkey-patch the `_LOADERS` dict entry to a loader that
    # returns a TRUNCATED panel. The `isolated_registry` autouse
    # fixture's snapshot restores the original on teardown.
    from benchmarks.datasets._base import PanelDataset
    from benchmarks.registry import datasets as _datasets_reg

    original = _datasets_reg._LOADERS["fake_binary"]  # type: ignore[reportPrivateUsage]

    def _truncating_loader(_root: Path) -> PanelDataset:
        ds = original(_root)
        return PanelDataset(spec=ds.spec, panel=ds.panel.head(2), y=ds.y[:2])

    monkeypatch.setitem(
        _datasets_reg._LOADERS,  # type: ignore[reportPrivateUsage]
        "fake_binary",
        _truncating_loader,
    )
    manifest = _build_run_manifest_for(config, output_root)
    with pytest.raises(RawRollupError, match="row-count drift"):
        aggregate_bootstrap_rollup(config, output_root=output_root, env=env, manifest=manifest)


# --- row-count ceiling gate (Gemini-C2) -----------------------------------


def test_aggregate_bootstrap_rollup_row_count_ceiling_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N * n_resamples > BOOTSTRAP_ROW_COUNT_CEILING raises
    `RawRollupError` (defensive OOM gate). B14 hoisted the
    constant from `bootstrap_rollup` to the shared
    `_bootstrap_aggregate` module; the importing module's binding
    is what the aggregator reads at call time."""
    register_all_fakes_and_get_panels()
    config = _make_config(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    run_raw_loss(config, output_root=output_root, env=env)

    # Lower the ceiling so the tiny fake-binary panel trips it.
    import benchmarks.report.bootstrap_rollup as _module

    monkeypatch.setattr(_module, "BOOTSTRAP_ROW_COUNT_CEILING", 1)
    manifest = _build_run_manifest_for(config, output_root)
    with pytest.raises(RawRollupError, match="ceiling"):
        aggregate_bootstrap_rollup(config, output_root=output_root, env=env, manifest=manifest)


# --- empty manifest raises ------------------------------------------------


def test_aggregate_bootstrap_rollup_empty_manifest_raises(tmp_path: Path) -> None:
    register_all_fakes_and_get_panels()
    config = _make_config(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = _build_run_manifest_for(config, output_root)
    with pytest.raises(ValueError, match="empty"):
        aggregate_bootstrap_rollup(config, output_root=output_root, env=env, manifest=manifest)


# --- rollup written to disk ------------------------------------------------


def test_aggregate_bootstrap_rollup_writes_shard_to_disk(tmp_path: Path) -> None:
    register_all_fakes_and_get_panels()
    config = _make_config(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    run_raw_loss(config, output_root=output_root, env=env)
    manifest = _build_run_manifest_for(config, output_root)
    aggregate_bootstrap_rollup(config, output_root=output_root, env=env, manifest=manifest)
    assert rollup_path(output_root).exists()
    rows = load_rollup(output_root)
    assert len(rows) >= 1
