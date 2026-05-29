"""Phase B14 D-B13.2 render_from_dir dispatch tests for training_time.

Pins the CI-variant CLI seam: rollup-present, fingerprint
matching, sentinel handling, manifest absent/corrupt cases.
"""

from pathlib import Path

from benchmarks.bootstrap_manifest import (
    TrainingTimeRollupRow,
    training_time_aggregator_failed_sentinel_path,
    training_time_rollup_path,
    write_training_time_rollup,
)
from benchmarks.config import BenchmarkConfig, ExperimentSpec
from benchmarks.experiments import build_run_environment, run_raw_loss
from benchmarks.report.training_time import render_from_dir
from benchmarks.run_manifest import (
    build_run_manifest,
    run_manifest_path,
    write_run_manifest,
)

from tests.benchmarks._fakes import register_all_fakes_and_get_panels


def _setup_b5(tmp_path: Path) -> Path:
    register_all_fakes_and_get_panels()
    config = BenchmarkConfig(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        experiments=(
            ExperimentSpec(kind="raw_loss", seeds=(0,)),
            ExperimentSpec(kind="training_time", seeds=(0,)),
        ),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    run_raw_loss(config, output_root=output_root, env=env)
    manifest = build_run_manifest(
        config=config,
        run_id="b14-tt-rt-cli",
        library_git_sha="0" * 40,
        profile="smoke",
        hardware_tier="cpu",
        output_root=output_root,
    )
    write_run_manifest(output_root, manifest)
    return output_root


def _rollup_row(
    *, manifest_fingerprint: str = "anything"
) -> TrainingTimeRollupRow:
    return TrainingTimeRollupRow(
        dataset_name="fake_binary",
        model_name="fake_constant_binary",
        hardware_tier="cpu",
        task_type="binary",
        primary_metric="wall_seconds",
        n_seeds=1,
        n_cells_evaluated=5,
        n_skipped_cells=0,
        primary_loss_mean=1.05,
        primary_loss_ci_lo=0.98,
        primary_loss_ci_hi=1.12,
        bootstrap_seed=42,
        bootstrap_n_resamples=10_000,
        bootstrap_confidence=0.95,
        bootstrap_rng_algorithm="PCG64",
        bootstrap_numpy_version="2.3.0",
        bootstrap_skipped_reason=None,
        manifest_fingerprint=manifest_fingerprint,
    )


def test_render_from_dir_falls_back_silently_when_rollup_file_absent(
    tmp_path: Path,
) -> None:
    output_root = _setup_b5(tmp_path)
    md = render_from_dir(output_root)
    assert "wall_seconds [95% CI]" not in md


def test_render_from_dir_surfaces_aggregator_failed_sentinel(
    tmp_path: Path,
) -> None:
    output_root = _setup_b5(tmp_path)
    training_time_aggregator_failed_sentinel_path(output_root).write_text(
        "RawRollupError", encoding="utf-8"
    )
    md = render_from_dir(output_root)
    assert "Bootstrap aggregator failed" in md
    assert "RawRollupError" in md


def test_render_from_dir_renders_ci_when_rollup_present_and_fingerprint_matches(
    tmp_path: Path,
) -> None:
    from benchmarks.run_manifest import load_run_manifest

    output_root = _setup_b5(tmp_path)
    manifest = load_run_manifest(output_root)
    write_training_time_rollup(
        output_root, [_rollup_row(manifest_fingerprint=manifest.fingerprint())]
    )

    md = render_from_dir(output_root)
    assert "wall_seconds [95% CI]" in md


def test_render_from_dir_falls_back_on_stale_rollup(
    tmp_path: Path,
) -> None:
    output_root = _setup_b5(tmp_path)
    write_training_time_rollup(
        output_root, [_rollup_row(manifest_fingerprint="oldfingerprint")]
    )
    md = render_from_dir(output_root)
    assert "Bootstrap rollup is stale" in md


def test_render_from_dir_freshness_check_skipped_when_manifest_corrupt(
    tmp_path: Path,
) -> None:
    output_root = _setup_b5(tmp_path)
    write_training_time_rollup(output_root, [_rollup_row()])
    run_manifest_path(output_root).write_bytes(b"{not valid json")

    md = render_from_dir(output_root)
    assert "wall_seconds [95% CI]" in md
    assert "Bootstrap freshness check skipped" in md


def test_render_from_dir_renders_ci_without_freshness_footnote_when_manifest_absent(
    tmp_path: Path,
) -> None:
    output_root = _setup_b5(tmp_path)
    write_training_time_rollup(output_root, [_rollup_row()])
    run_manifest_path(output_root).unlink()

    md = render_from_dir(output_root)
    assert "wall_seconds [95% CI]" in md
    assert "Bootstrap rollup is stale" not in md
    assert "Bootstrap freshness check skipped" not in md


def test_render_from_dir_falls_back_silently_when_rollup_file_empty(
    tmp_path: Path,
) -> None:
    output_root = _setup_b5(tmp_path)
    write_training_time_rollup(output_root, [])
    assert training_time_rollup_path(output_root).exists()

    md = render_from_dir(output_root)
    assert "wall_seconds [95% CI]" not in md
