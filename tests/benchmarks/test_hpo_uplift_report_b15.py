"""Phase B15 D-B13.3 HPO-uplift renderer tests (CI variant).

Pins the 4-source footnote precedence + the B15-specific source 5
(loss-dropout disclosure) + the no_paired_cells routing through
the shared render_rollup_skipped_footnote + the partial-fold
asterisk + the render_from_dir dispatch chain.
"""

from pathlib import Path

import pandas as pd
import pytest
from benchmarks.bootstrap_manifest import (
    HPOUpliftRollupRow,
    hpo_uplift_aggregator_failed_sentinel_path,
    hpo_uplift_rollup_path,
    write_hpo_uplift_rollup,
)
from benchmarks.config import BenchmarkConfig, ExperimentSpec
from benchmarks.experiments import build_run_environment, run_raw_loss
from benchmarks.report.hpo_uplift import (
    render_from_dir,
    render_hpo_uplift_markdown_with_ci,
)
from benchmarks.run_manifest import (
    build_run_manifest,
    run_manifest_path,
    write_run_manifest,
)

from tests.benchmarks._fakes import register_all_fakes_and_get_panels


def _make_manifest_with_uplift_row(
    *,
    delta: float | None = 0.20,
    n_cells_paired: int = 4,
) -> pd.DataFrame:
    """B5/B8 manifest fixture: 2 seeds x 2 folds x default+tuned
    for fake_binary / fake_constant_binary, with controllable
    per-cell loss so the derived Δ surfaces predictably in the
    UpliftRow."""
    rows = []
    # Default arm: log_loss = 0.50; tuned: 0.30 -> Δ = 0.20 per pair.
    tuned_loss = 0.50 - delta if delta is not None else 0.50
    for s in (0, 1):
        for f in (0, 1):
            rows.append(
                {
                    "library_git_sha": "0" * 40,
                    "run_id": "b15-rt",
                    "started_at_utc": "2026-05-30T00:00:00+00:00",
                    "dataset_name": "fake_binary",
                    "model_name": "fake_constant_binary",
                    "task_type": "binary",
                    "seed": s,
                    "fold_index": f,
                    "variant": "default",
                    "log_loss": 0.50,
                    "skipped_reason": None,
                }
            )
            rows.append(
                {
                    "library_git_sha": "0" * 40,
                    "run_id": "b15-rt",
                    "started_at_utc": "2026-05-30T00:00:00+00:00",
                    "dataset_name": "fake_binary",
                    "model_name": "fake_constant_binary",
                    "task_type": "binary",
                    "seed": s,
                    "fold_index": f,
                    "variant": "tuned",
                    "log_loss": tuned_loss,
                    "skipped_reason": None,
                }
            )
    return pd.DataFrame(rows)


def _rollup_row(
    *,
    dataset_name: str = "fake_binary",
    model_name: str = "fake_constant_binary",
    task_type: str = "binary",
    primary_metric_mean: float | None = 0.20,
    primary_metric_ci_lo: float | None = 0.15,
    primary_metric_ci_hi: float | None = 0.25,
    bootstrap_skipped_reason: str | None = None,
    manifest_fingerprint: str = "feedbeef" * 8,
    n_seeds: int = 2,
    n_folds: int = 2,
    n_cells_paired: int = 4,
    n_skipped_cells: int = 0,
) -> HPOUpliftRollupRow:
    return HPOUpliftRollupRow(
        dataset_name=dataset_name,
        model_name=model_name,
        task_type=task_type,
        primary_metric="delta",
        primary_loss_column="log_loss",
        n_seeds=n_seeds,
        n_folds=n_folds,
        n_cells_paired=n_cells_paired,
        n_skipped_cells=n_skipped_cells,
        primary_metric_mean=primary_metric_mean,
        primary_metric_ci_lo=primary_metric_ci_lo,
        primary_metric_ci_hi=primary_metric_ci_hi,
        bootstrap_seed=42,
        bootstrap_n_resamples=10_000,
        bootstrap_confidence=0.95,
        bootstrap_rng_algorithm="PCG64",
        bootstrap_numpy_version="2.3.0",
        bootstrap_skipped_reason=bootstrap_skipped_reason,
        manifest_fingerprint=manifest_fingerprint,
    )


# --- Footnote-source precedence (B15.4) -------------------------------------


def test_render_hpo_uplift_with_ci_renders_mean_and_interval() -> None:
    """Happy path: matching fingerprint -> CI variant."""
    manifest = _make_manifest_with_uplift_row()
    rollup = [_rollup_row()]
    md = render_hpo_uplift_markdown_with_ci(
        manifest,
        rollup,
        expected_manifest_fingerprint="feedbeef" * 8,
    )
    assert "Δ [95% CI]" in md
    assert "0.2000 [0.1500, 0.2500]" in md


def test_render_hpo_uplift_without_ci_falls_back_to_scalar() -> None:
    """Empty rollup -> std variant silently (no CI header)."""
    manifest = _make_manifest_with_uplift_row()
    md = render_hpo_uplift_markdown_with_ci(manifest, [])
    assert "Δ [95% CI]" not in md
    # Std variant uses bare `delta` column header.
    assert "delta" in md


def test_render_hpo_uplift_with_ci_drops_old_delta_header_when_ci_present() -> None:
    """The per-dataset table header uses `Δ [95% CI]` when active;
    the bare-delta header should NOT appear in the per-dataset
    block table header row."""
    manifest = _make_manifest_with_uplift_row()
    rollup = [_rollup_row()]
    md = render_hpo_uplift_markdown_with_ci(manifest, rollup)
    assert "Δ [95% CI]" in md
    # The pre-CI header `| delta |` (cell exactly equal to delta)
    # must not appear in the dataset-block table header.
    lines = md.splitlines()
    # Find the dataset-block table header.
    dataset_idx = next(
        i for i, line in enumerate(lines) if line.startswith("### fake_binary")
    )
    table_header = lines[dataset_idx + 2]
    assert "Δ [95% CI]" in table_header
    assert "| delta |" not in table_header


def test_render_hpo_uplift_with_ci_falls_back_on_manifest_fingerprint_mismatch() -> None:
    """Stale rollup -> std + footnote."""
    manifest = _make_manifest_with_uplift_row()
    rollup = [_rollup_row(manifest_fingerprint="oldfingerprint")]
    md = render_hpo_uplift_markdown_with_ci(
        manifest, rollup, expected_manifest_fingerprint="newfingerprint"
    )
    assert "Bootstrap rollup is stale" in md
    assert "Δ [95% CI]" not in md


def test_render_hpo_uplift_with_ci_surfaces_aggregator_failed_footnote_on_wrapper_caught() -> None:
    """aggregator_error_class -> std + footnote."""
    manifest = _make_manifest_with_uplift_row()
    md = render_hpo_uplift_markdown_with_ci(
        manifest, [], aggregator_error_class="RawRollupError"
    )
    assert "Bootstrap aggregator failed" in md
    assert "RawRollupError" in md


def test_render_hpo_uplift_with_ci_manifest_unreadable_appends_freshness_footnote() -> None:
    """manifest_unreadable=True -> CI body + freshness-skipped footnote."""
    manifest = _make_manifest_with_uplift_row()
    md = render_hpo_uplift_markdown_with_ci(
        manifest, [_rollup_row()], manifest_unreadable=True
    )
    assert "Δ [95% CI]" in md
    assert "Bootstrap freshness check skipped" in md


# --- Source 5: partial coverage footnote -----------------------------------


def test_render_hpo_uplift_with_ci_surfaces_skipped_cells_footnote_when_n_skipped_cells_nonzero() -> None:
    """qa-R2-C1 closure: source 5 fires when n_skipped_cells > 0
    even on otherwise-clean rollup row."""
    manifest = _make_manifest_with_uplift_row()
    rollup = [_rollup_row(n_skipped_cells=2, n_cells_paired=4, n_seeds=2, n_folds=3)]
    md = render_hpo_uplift_markdown_with_ci(manifest, rollup)
    assert "Partial coverage" in md
    assert "2 / 4" in md


def test_render_hpo_uplift_with_ci_surfaces_skipped_cells_footnote_without_asterisk_when_no_pairing_asymmetry() -> None:
    """R3 qa-I3 closure: source 5 fires INDEPENDENTLY of the
    partial-fold asterisk. Fixture: n_cells_paired == n_seeds * n_folds
    (no asterisk) AND n_skipped_cells > 0 (footnote should fire)."""
    manifest = _make_manifest_with_uplift_row()
    rollup = [_rollup_row(n_seeds=2, n_folds=2, n_cells_paired=4, n_skipped_cells=1)]
    md = render_hpo_uplift_markdown_with_ci(manifest, rollup)
    # CI cell does NOT end with `*` (no pairing asymmetry).
    assert "0.2000 [0.1500, 0.2500]*" not in md
    assert "0.2000 [0.1500, 0.2500]" in md
    # But the partial coverage footnote IS present.
    assert "Partial coverage" in md
    assert "1 / 4" in md


def test_render_hpo_uplift_with_ci_surfaces_both_freshness_and_skipped_cells_footnotes_when_both_apply() -> None:
    """R3 qa-I2 closure: source 4 (manifest_unreadable) and source 5
    (n_skipped_cells > 0) co-occur."""
    manifest = _make_manifest_with_uplift_row()
    rollup = [_rollup_row(n_skipped_cells=1, n_cells_paired=4, n_seeds=2, n_folds=2)]
    md = render_hpo_uplift_markdown_with_ci(manifest, rollup, manifest_unreadable=True)
    assert "Bootstrap freshness check skipped" in md
    assert "Partial coverage" in md


# --- Partial-fold asterisk + missing-row ----------------------------------


def test_render_hpo_uplift_with_ci_marks_partial_fold_cell_with_asterisk() -> None:
    """Pairing asymmetry: n_cells_paired=4 < n_seeds * n_folds=6
    -> CI cell ends with `*`."""
    manifest = _make_manifest_with_uplift_row()
    rollup = [_rollup_row(n_seeds=2, n_folds=3, n_cells_paired=4, n_skipped_cells=0)]
    md = render_hpo_uplift_markdown_with_ci(manifest, rollup)
    assert "0.2000 [0.1500, 0.2500]*" in md


def test_render_hpo_uplift_with_ci_renders_no_ci_when_rollup_missing_for_row() -> None:
    """Manifest has a (dataset, model) pair but the rollup has NO
    matching row -> `(no CI)` cell sentinel."""
    manifest = _make_manifest_with_uplift_row()
    rollup = [_rollup_row(model_name="other_model")]
    md = render_hpo_uplift_markdown_with_ci(manifest, rollup)
    assert "(no CI)" in md


# --- no_paired_cells sentinel routes via shared helper --------------------


def test_render_hpo_uplift_with_ci_surfaces_no_paired_cells_sentinel_in_rollup_skipped_footnote() -> None:
    """qa-R2-I4: the no_paired_cells sentinel routes through the
    shared render_rollup_skipped_footnote helper. The other three
    sentinels flow through the existing _render_footnote.

    Note: the `no_paired_cells` sentinel rollup row carries
    primary_loss_*=None; the renderer routes it to the
    "Bootstrap skipped" footnote with the EXACT reason string."""
    manifest = _make_manifest_with_uplift_row()
    sentinel_row = _rollup_row(
        primary_metric_mean=None,
        primary_metric_ci_lo=None,
        primary_metric_ci_hi=None,
        n_cells_paired=0,
        bootstrap_skipped_reason="no_paired_cells",
    )
    rollup = [_rollup_row(), sentinel_row]
    md = render_hpo_uplift_markdown_with_ci(manifest, rollup)
    assert "Bootstrap skipped" in md
    assert "no_paired_cells" in md


# --- render_from_dir dispatch tests ----------------------------------------


def _setup_b8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Provide an output_root with run_manifest + a synthetic
    manifest the renderer's load_run sees (both default + tuned
    variant rows so build_hpo_uplift_report sees a valid Δ row)."""
    register_all_fakes_and_get_panels()
    config = BenchmarkConfig(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        experiments=(
            ExperimentSpec(kind="raw_loss", seeds=(0,)),
            ExperimentSpec(kind="hpo_uplift", seeds=(0,)),
        ),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    # Run B5 so the results/ directory exists (the renderer's
    # rollup_path check still wants something on disk).
    run_raw_loss(config, output_root=output_root, env=env)
    manifest = build_run_manifest(
        config=config,
        run_id="b15-rt-cli",
        library_git_sha="0" * 40,
        profile="smoke",
        hardware_tier="cpu",
        output_root=output_root,
    )
    write_run_manifest(output_root, manifest)

    # Patch load_run inside the renderer module so build_hpo_uplift_report
    # sees a manifest with both arms.
    import benchmarks.report.hpo_uplift as _module

    fake_manifest = _make_manifest_with_uplift_row()
    monkeypatch.setattr(_module, "load_run", lambda _root: fake_manifest)  # type: ignore[misc]
    return output_root


def test_render_from_dir_falls_back_silently_when_rollup_file_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = _setup_b8(tmp_path, monkeypatch)
    md = render_from_dir(output_root)
    assert "Δ [95% CI]" not in md


def test_render_from_dir_surfaces_aggregator_failed_sentinel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = _setup_b8(tmp_path, monkeypatch)
    hpo_uplift_aggregator_failed_sentinel_path(output_root).write_text(
        "RawRollupError", encoding="utf-8"
    )
    md = render_from_dir(output_root)
    assert "Bootstrap aggregator failed" in md
    assert "RawRollupError" in md


def test_render_from_dir_renders_ci_when_rollup_present_and_fingerprint_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmarks.run_manifest import load_run_manifest

    output_root = _setup_b8(tmp_path, monkeypatch)
    manifest = load_run_manifest(output_root)
    rollup_row = _rollup_row(manifest_fingerprint=manifest.fingerprint())
    write_hpo_uplift_rollup(output_root, [rollup_row])

    md = render_from_dir(output_root)
    assert "Δ [95% CI]" in md


def test_render_from_dir_falls_back_on_stale_rollup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = _setup_b8(tmp_path, monkeypatch)
    rollup_row = _rollup_row(manifest_fingerprint="oldfingerprint")
    write_hpo_uplift_rollup(output_root, [rollup_row])

    md = render_from_dir(output_root)
    assert "Bootstrap rollup is stale" in md


def test_render_from_dir_freshness_check_skipped_when_manifest_corrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = _setup_b8(tmp_path, monkeypatch)
    rollup_row = _rollup_row(manifest_fingerprint="anything")
    write_hpo_uplift_rollup(output_root, [rollup_row])
    run_manifest_path(output_root).write_bytes(b"{not valid json")

    md = render_from_dir(output_root)
    assert "Δ [95% CI]" in md
    assert "Bootstrap freshness check skipped" in md


def test_render_from_dir_renders_ci_without_freshness_footnote_when_manifest_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = _setup_b8(tmp_path, monkeypatch)
    rollup_row = _rollup_row(manifest_fingerprint="anything")
    write_hpo_uplift_rollup(output_root, [rollup_row])
    run_manifest_path(output_root).unlink()

    md = render_from_dir(output_root)
    assert "Δ [95% CI]" in md
    assert "Bootstrap rollup is stale" not in md
    assert "Bootstrap freshness check skipped" not in md


def test_render_from_dir_falls_back_silently_when_rollup_file_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = _setup_b8(tmp_path, monkeypatch)
    write_hpo_uplift_rollup(output_root, [])
    assert hpo_uplift_rollup_path(output_root).exists()

    md = render_from_dir(output_root)
    assert "Δ [95% CI]" not in md


# silence unused import (pytest is consumed by the fixtures).
_ = pytest
