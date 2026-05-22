"""Phase B9 cross-experiment report assembly tests.

Pins that `assemble_report` reads the four per-experiment Markdown
files + the run manifest under `output_root`, emits a single
`report.md` with the right section ordering, and handles missing
per-experiment files gracefully.
"""

from pathlib import Path

import pytest
from benchmarks.config import BenchmarkConfig, ExperimentSpec
from benchmarks.report.render import (
    REPORT_FILENAME,
    assemble_report,
    render_report_from_dir,
)
from benchmarks.run_manifest import (
    EnvironmentFingerprint,
    build_run_manifest,
    write_run_manifest,
)


def _env() -> EnvironmentFingerprint:
    return EnvironmentFingerprint(
        platform="Linux-test",
        python_version="3.14.0",
        cpu_model="test-cpu",
        gpu_model="test-gpu",
        cuda_runtime="12.6",
        cuda_driver="590.48.01",
        package_versions={"seq-sklearn": "0.0.0", "torch": "2.6"},
    )


def _seed_manifest(tmp_path: Path) -> Path:
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    config = BenchmarkConfig(
        datasets=("ds_a",),
        models=("m_x",),
        experiments=(ExperimentSpec(kind="raw_loss"),),
        output_dir=out,
    )
    manifest = build_run_manifest(
        config,
        run_id="r-id",
        library_git_sha="abcdef0",
        profile="smoke",
        hardware_tier="cpu",
        output_root=out,
        started_at_utc="2026-05-22T00:00:00+00:00",
        completed_at_utc="2026-05-22T00:05:00+00:00",
        environment=_env(),
    )
    write_run_manifest(out, manifest)
    return out


# --- assemble_report -------------------------------------------------------


def test_assemble_report_renders_single_h1_and_experiment_sections(
    tmp_path: Path,
) -> None:
    out = _seed_manifest(tmp_path)
    # Seed each per-experiment Markdown with a recognizable line.
    (out / "leaderboard.md").write_text("# Leaderboard\n\ncontent A\n")
    (out / "pairwise.md").write_text("# Pairwise\n\ncontent B\n")
    (out / "training_time.md").write_text("# Training time\n\ncontent C\n")
    (out / "hpo_uplift.md").write_text("# HPO uplift\n\ncontent D\n")
    md = assemble_report(out)
    # One top-level H1 + four section H2 headings + run-metadata
    # H2 + methodology H2 = 6 H2 lines.
    assert md.count("\n# ") == 0  # body has no extra H1 (the leading H1 is the title)
    assert md.startswith("# seq-sklearn benchmark report")
    # Section headings in the canonical B5 -> B8 order.
    raw_loss_idx = md.find("Raw loss leaderboard")
    pairwise_idx = md.find("Ensemble complementarity")
    tt_idx = md.find("Training-time table")
    hpo_idx = md.find("HPO uplift + Friedman/Holm")
    assert -1 < raw_loss_idx < pairwise_idx < tt_idx < hpo_idx
    # Per-experiment content quoted verbatim (the H1 from the
    # source is stripped).
    assert "content A" in md
    assert "content B" in md
    assert "content C" in md
    assert "content D" in md
    # The per-experiment H1 lines are NOT in the assembled body
    # (the assembler strips them).
    assert "# Leaderboard" not in md
    assert "# Pairwise" not in md


def test_assemble_report_missing_files_render_as_notice(tmp_path: Path) -> None:
    out = _seed_manifest(tmp_path)
    # Only leaderboard.md exists; the other three should render
    # as one-line notices.
    (out / "leaderboard.md").write_text("# Leaderboard\n\nL\n")
    md = assemble_report(out)
    assert "content L" not in md  # sanity: we did not write that string
    assert "L\n" in md  # leaderboard quoted
    assert "`pairwise.md` not present in this manifest." in md
    assert "`training_time.md` not present in this manifest." in md
    assert "`hpo_uplift.md` not present in this manifest." in md


def test_assemble_report_executive_summary_names_run_id_and_profile(
    tmp_path: Path,
) -> None:
    out = _seed_manifest(tmp_path)
    md = assemble_report(out)
    assert "Run `r-id`" in md
    assert "profile `smoke`" in md
    assert "hardware tier `cpu`" in md
    assert "Library + benchmarks SHA `abcdef0`" in md
    # Datasets + models from the manifest.
    assert "ds_a" in md
    assert "m_x" in md


def test_assemble_report_run_metadata_block_lists_env_fingerprint(
    tmp_path: Path,
) -> None:
    out = _seed_manifest(tmp_path)
    md = assemble_report(out)
    assert "## Run metadata" in md
    assert "Python: `3.14.0`" in md
    assert "CPU: `test-cpu`" in md
    assert "GPU: `test-gpu`" in md
    assert "CUDA runtime: `12.6`" in md
    assert "CUDA driver: `590.48.01`" in md
    assert "`seq-sklearn`: `0.0.0`" in md


def test_assemble_report_methodology_footer_present(tmp_path: Path) -> None:
    out = _seed_manifest(tmp_path)
    md = assemble_report(out)
    assert "## Methodology" in md
    assert "docs/explanation/benchmarks/" in md


def test_assemble_report_missing_manifest_raises(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="no run manifest"):
        assemble_report(out)


def test_assemble_report_accepts_manifest_override(tmp_path: Path) -> None:
    """The `manifest=` kwarg lets a caller inject a manifest
    without loading from disk; useful for tests / mid-run
    rendering."""
    out = tmp_path / "out"
    out.mkdir(parents=True)
    config = BenchmarkConfig(
        datasets=("ds_a",),
        models=("m_x",),
        experiments=(ExperimentSpec(kind="raw_loss"),),
        output_dir=out,
    )
    manifest = build_run_manifest(
        config,
        run_id="override-id",
        library_git_sha="sha",
        profile="standard",
        hardware_tier="gpu_single",
        output_root=out,
        environment=_env(),
    )
    md = assemble_report(out, manifest=manifest)
    assert "Run `override-id`" in md
    # Manifest never landed on disk; the JSON loader was not used.
    from benchmarks.run_manifest import run_manifest_path

    assert not run_manifest_path(out).exists()


# --- render_report_from_dir convenience --------------------------------------


def test_render_report_from_dir_wraps_assemble_report(tmp_path: Path) -> None:
    out = _seed_manifest(tmp_path)
    (out / "leaderboard.md").write_text("# Leaderboard\n\nbody\n")
    md = render_report_from_dir(out)
    assert md.startswith("# seq-sklearn benchmark report")
    assert "body" in md


def test_report_filename_constant_is_report_md() -> None:
    assert REPORT_FILENAME == "report.md"
