"""Phase B13 raw_loss leaderboard renderer tests (CI variant).

Pins the dispatch-by-rollup-presence (Gemini-C3), freshness
fingerprint check, std-variant fallback, "Bootstrap aggregator
failed" footnote, partial-fold `*` flag (Gemini-I3), and
separate "Bootstrap skipped" footnote (Gemini-I4 / qa-I4).
"""

from pathlib import Path

import benchmarks.adapters  # type: ignore[reportUnusedImport]  # noqa: F401
import pandas as pd
from benchmarks.bootstrap_manifest import RollupRow
from benchmarks.config import BenchmarkConfig, ExperimentSpec
from benchmarks.experiments import build_run_environment, run_raw_loss
from benchmarks.manifest import load_run
from benchmarks.report.raw_loss import (
    render_leaderboard_markdown,
    render_leaderboard_markdown_with_ci,
)

from tests.benchmarks._fakes import register_all_fakes_and_get_panels


def _run_b5(tmp_path: Path) -> pd.DataFrame:
    register_all_fakes_and_get_panels()
    config = BenchmarkConfig(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        experiments=(ExperimentSpec(kind="raw_loss", seeds=(0,)),),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    run_raw_loss(config, output_root=output_root, env=env)
    return load_run(output_root)


def _rollup_row(
    *,
    dataset_name: str = "fake_binary",
    model_name: str = "fake_constant_binary",
    task_type: str = "binary",
    primary_metric: str = "log_loss",
    n_seeds: int = 1,
    n_cells_evaluated: int = 5,
    n_skipped_cells: int = 0,
    n_rows: int = 40,
    n_entities: int = 4,
    primary_loss_mean: float | None = 0.234,
    primary_loss_ci_lo: float | None = 0.221,
    primary_loss_ci_hi: float | None = 0.247,
    bootstrap_skipped_reason: str | None = None,
    manifest_fingerprint: str = "fp1",
) -> RollupRow:
    return RollupRow(
        dataset_name=dataset_name,
        model_name=model_name,
        task_type=task_type,
        primary_metric=primary_metric,
        n_seeds=n_seeds,
        n_cells_evaluated=n_cells_evaluated,
        n_skipped_cells=n_skipped_cells,
        n_rows=n_rows,
        n_entities=n_entities,
        primary_loss_mean=primary_loss_mean,
        primary_loss_ci_lo=primary_loss_ci_lo,
        primary_loss_ci_hi=primary_loss_ci_hi,
        bootstrap_seed=42,
        bootstrap_n_resamples=5_000,
        bootstrap_confidence=0.95,
        bootstrap_rng_algorithm="PCG64",
        bootstrap_numpy_version="2.3.0",
        bootstrap_skipped_reason=bootstrap_skipped_reason,
        manifest_fingerprint=manifest_fingerprint,
    )


def test_render_leaderboard_with_ci_renders_mean_and_interval(tmp_path: Path) -> None:
    manifest = _run_b5(tmp_path)
    rollup = [_rollup_row()]
    md = render_leaderboard_markdown_with_ci(manifest, rollup)
    assert "log_loss [95% CI]" in md
    assert "0.2340 [0.2210, 0.2470]" in md


def test_render_leaderboard_without_ci_falls_back_to_scalar(tmp_path: Path) -> None:
    manifest = _run_b5(tmp_path)
    md = render_leaderboard_markdown_with_ci(manifest, [])
    # No CI column, std variant column header is present.
    assert "log_loss (mean ± std)" in md
    assert "[95% CI]" not in md


def test_render_leaderboard_with_ci_drops_std_column_when_ci_present(
    tmp_path: Path,
) -> None:
    manifest = _run_b5(tmp_path)
    md = render_leaderboard_markdown_with_ci(manifest, [_rollup_row()])
    assert "log_loss [95% CI]" in md
    assert "(mean ± std)" not in md


def test_render_leaderboard_with_ci_falls_back_on_manifest_fingerprint_mismatch(
    tmp_path: Path,
) -> None:
    """The CI variant emits the std fallback + a footnote when
    the rollup's manifest_fingerprint doesn't match the live
    manifest's (Gemini-C3)."""
    manifest = _run_b5(tmp_path)
    rollup = [_rollup_row(manifest_fingerprint="stale_fp")]
    md = render_leaderboard_markdown_with_ci(
        manifest, rollup, expected_manifest_fingerprint="live_fp"
    )
    assert "Bootstrap rollup is stale" in md
    assert "(mean ± std)" in md  # std fallback rendered
    assert "[95% CI]" not in md


def test_render_leaderboard_with_ci_emits_aggregator_failed_footnote_on_wrapper_caught(
    tmp_path: Path,
) -> None:
    """When the CLI wrapper caught a RawRollupError, the
    renderer falls back to std + emits the failure footnote
    (Gemini-C2 wrapper case)."""
    manifest = _run_b5(tmp_path)
    md = render_leaderboard_markdown_with_ci(manifest, [], aggregator_error_class="RawRollupError")
    assert "Bootstrap aggregator failed" in md
    assert "RawRollupError" in md
    assert "(mean ± std)" in md  # std fallback


def test_render_leaderboard_with_ci_partial_fold_appends_asterisk(
    tmp_path: Path,
) -> None:
    """The CI cell appends `*` when n_cells_evaluated <
    n_seeds * n_folds (Gemini-I3 visual flag)."""
    manifest = _run_b5(tmp_path)
    # Fake panel produces 5 folds in smoke profile; force the
    # rollup row to claim only 2 of them were OK.
    rollup = [_rollup_row(n_cells_evaluated=2)]
    md = render_leaderboard_markdown_with_ci(manifest, rollup)
    assert "0.2340 [0.2210, 0.2470]*" in md


def test_render_leaderboard_with_ci_surfaces_rollup_skipped_in_separate_footnote(
    tmp_path: Path,
) -> None:
    """A rollup row with bootstrap_skipped_reason set renders in
    a "Bootstrap skipped" sub-table SEPARATE from the per-cell
    skipped footnote (Gemini-I4 / qa-I4)."""
    manifest = _run_b5(tmp_path)
    rollup = [
        _rollup_row(),  # OK
        _rollup_row(
            model_name="fake_loader_failed",
            n_cells_evaluated=0,
            n_skipped_cells=5,
            primary_loss_mean=None,
            primary_loss_ci_lo=None,
            primary_loss_ci_hi=None,
            bootstrap_skipped_reason="loader_failed: KeyError: no_such_dataset",
        ),
    ]
    md = render_leaderboard_markdown_with_ci(manifest, rollup)
    assert "Bootstrap skipped" in md
    assert "fake_loader_failed" in md


def test_render_leaderboard_markdown_keeps_std_signature(tmp_path: Path) -> None:
    """The legacy `render_leaderboard_markdown(manifest)` keeps
    its signature unchanged for backwards compatibility."""
    manifest = _run_b5(tmp_path)
    md = render_leaderboard_markdown(manifest)
    assert "log_loss (mean ± std)" in md


def test_render_from_dir_passes_freshness_fingerprint_at_cli_seam(
    tmp_path: Path,
) -> None:
    """R1 code-I1: `render_from_dir` must pass the live manifest's
    fingerprint to the CI variant so Gemini-C3 freshness check
    fires in CLI path, not just unit tests. With a fresh rollup +
    matching manifest, the CI variant renders."""
    from benchmarks.bootstrap_manifest import write_rollup
    from benchmarks.config import BenchmarkConfig, ExperimentSpec
    from benchmarks.experiments import build_run_environment, run_raw_loss
    from benchmarks.report.raw_loss import render_from_dir
    from benchmarks.run_manifest import build_run_manifest, write_run_manifest

    register_all_fakes_and_get_panels()
    config = BenchmarkConfig(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        experiments=(ExperimentSpec(kind="raw_loss", seeds=(0,)),),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    run_raw_loss(config, output_root=output_root, env=env)
    manifest = build_run_manifest(
        config=config,
        run_id="r1",
        library_git_sha="0" * 40,
        profile="smoke",
        hardware_tier="cpu",
        output_root=output_root,
    )
    write_run_manifest(output_root, manifest)
    fresh_fp = manifest.fingerprint()
    write_rollup(output_root, [_rollup_row(manifest_fingerprint=fresh_fp)])

    md = render_from_dir(output_root)
    assert "log_loss [95% CI]" in md  # CI variant active


def test_render_from_dir_falls_back_on_stale_rollup_at_cli_seam(
    tmp_path: Path,
) -> None:
    """R1 code-I1 + Gemini-C3: a rollup with a STALE manifest_
    fingerprint (e.g., from a prior run with different library
    SHA) triggers the std fallback + "rollup is stale" footnote
    when invoked through `render_from_dir`."""
    from benchmarks.bootstrap_manifest import write_rollup
    from benchmarks.config import BenchmarkConfig, ExperimentSpec
    from benchmarks.experiments import build_run_environment, run_raw_loss
    from benchmarks.report.raw_loss import render_from_dir
    from benchmarks.run_manifest import build_run_manifest, write_run_manifest

    register_all_fakes_and_get_panels()
    config = BenchmarkConfig(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        experiments=(ExperimentSpec(kind="raw_loss", seeds=(0,)),),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    run_raw_loss(config, output_root=output_root, env=env)
    manifest = build_run_manifest(
        config=config,
        run_id="r1",
        library_git_sha="0" * 40,
        profile="smoke",
        hardware_tier="cpu",
        output_root=output_root,
    )
    write_run_manifest(output_root, manifest)
    # Stale rollup carries a fingerprint that does NOT match the
    # live manifest's.
    write_rollup(output_root, [_rollup_row(manifest_fingerprint="stale_fp_xxx")])

    md = render_from_dir(output_root)
    assert "Bootstrap rollup is stale" in md
    assert "log_loss (mean ± std)" in md  # std fallback active


def test_render_from_dir_surfaces_aggregator_failed_sentinel(
    tmp_path: Path,
) -> None:
    """R1 code-I1: when the CLI wrapper dropped a
    `bootstrap_aggregator_failed.txt` sentinel (Gemini-C2 wrapper
    case), `render_from_dir` reads the sentinel and surfaces the
    'Bootstrap aggregator failed: <class>' footnote in the std
    fallback."""
    from benchmarks.config import BenchmarkConfig, ExperimentSpec
    from benchmarks.experiments import build_run_environment, run_raw_loss
    from benchmarks.report.raw_loss import render_from_dir

    register_all_fakes_and_get_panels()
    config = BenchmarkConfig(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        experiments=(ExperimentSpec(kind="raw_loss", seeds=(0,)),),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    run_raw_loss(config, output_root=output_root, env=env)
    (output_root / "bootstrap_aggregator_failed.txt").write_text("RawRollupError", encoding="utf-8")
    md = render_from_dir(output_root)
    assert "Bootstrap aggregator failed" in md
    assert "RawRollupError" in md
