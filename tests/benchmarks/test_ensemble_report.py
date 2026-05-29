"""Phase B14 D-B13.1 pairwise CI renderer tests.

Pins the 4-source footnote precedence (B14.4 mirror of B13):
1. aggregator_error_class -> std + "aggregator failed" footnote
2. not rollup -> std silently
3. fingerprint mismatch -> std + "stale rollup" footnote
4. CI body + optional manifest-unreadable footnote

Plus the render_from_dir dispatch tests for the CLI seam.
"""

from pathlib import Path

import pandas as pd
from benchmarks.bootstrap_manifest import (
    PairwiseRollupRow,
    pairwise_aggregator_failed_sentinel_path,
    pairwise_rollup_path,
    write_pairwise_rollup,
)
from benchmarks.config import BenchmarkConfig, ExperimentSpec
from benchmarks.experiments import build_run_environment, run_ensemble, run_raw_loss
from benchmarks.report.ensemble import (
    render_from_dir,
    render_pairwise_markdown_with_ci,
)
from benchmarks.run_manifest import (
    build_run_manifest,
    run_manifest_path,
    write_run_manifest,
)

from tests.benchmarks._fakes import register_all_fakes_and_get_panels


def _make_manifest() -> pd.DataFrame:
    """A minimal pairwise manifest with one (dataset, A, B)
    classification pair across 2 seeds x 1 fold."""
    rows = []
    for s in (0, 1):
        rows.append(
            {
                "library_git_sha": "0" * 40,
                "run_id": "b14-rt-test",
                "started_at_utc": "2026-05-30T00:00:00+00:00",
                "dataset_name": "fake_binary",
                "model_a": "a",
                "model_b": "b",
                "seed": s,
                "fold_index": 0,
                "task_type": "binary",
                "skipped_reason": None,
                "n_samples": 100,
                "n11": 40, "n10": 10, "n01": 15, "n00": 35,
                "yule_q": 0.7, "phi": 0.5,
                "disagreement_rate": 0.25,
                "double_fault_rate": 0.1,
                "pearson_pred_corr": 0.6,
                "spearman_pred_corr": 0.55,
                "pearson_error_corr": 0.30,
            }
        )
    return pd.DataFrame(rows)


def _make_rollup_row(
    *,
    dataset_name: str = "fake_binary",
    model_a: str = "a",
    model_b: str = "b",
    task_type: str = "binary",
    primary_loss_mean: float | None = 0.95,
    primary_loss_ci_lo: float | None = 0.88,
    primary_loss_ci_hi: float | None = 1.02,
    bootstrap_skipped_reason: str | None = None,
    manifest_fingerprint: str = "feedface" * 8,
    n_cells_evaluated: int = 2,
    n_seeds: int = 2,
) -> PairwiseRollupRow:
    return PairwiseRollupRow(
        dataset_name=dataset_name,
        model_a=model_a,
        model_b=model_b,
        task_type=task_type,
        primary_metric="complementarity_score",
        n_seeds=n_seeds,
        n_cells_evaluated=n_cells_evaluated,
        n_skipped_cells=0,
        primary_loss_mean=primary_loss_mean,
        primary_loss_ci_lo=primary_loss_ci_lo,
        primary_loss_ci_hi=primary_loss_ci_hi,
        bootstrap_seed=42,
        bootstrap_n_resamples=10_000,
        bootstrap_confidence=0.95,
        bootstrap_rng_algorithm="PCG64",
        bootstrap_numpy_version="2.3.0",
        bootstrap_skipped_reason=bootstrap_skipped_reason,
        manifest_fingerprint=manifest_fingerprint,
    )


# --- Footnote-source precedence (B14.4) -------------------------------------


def test_render_pairwise_with_ci_renders_mean_and_interval() -> None:
    """Happy path: a non-empty rollup + matching fingerprint -> CI variant."""
    manifest = _make_manifest()
    rollup = [_make_rollup_row()]
    md = render_pairwise_markdown_with_ci(
        manifest,
        rollup,
        expected_manifest_fingerprint="feedface" * 8,
    )
    assert "complementarity_score [95% CI]" in md
    assert "0.9500 [0.8800, 1.0200]" in md


def test_render_pairwise_without_ci_falls_back_to_scalar() -> None:
    """Empty rollup -> std variant silently (no CI header)."""
    manifest = _make_manifest()
    md = render_pairwise_markdown_with_ci(manifest, [])
    assert "complementarity_score [95% CI]" not in md
    # Std variant: scalar complementarity_score column header.
    assert "complementarity_score" in md


def test_render_pairwise_with_ci_drops_old_score_header_when_ci_present() -> None:
    """Stage-1 qa-N1 mirror for B6: the per-dataset table header
    uses the `[95% CI]` form when the CI variant is active. The
    Top-N summary table keeps its original `complementarity_score`
    header (it's a separate aggregation table). Pin the dataset-
    block header specifically rather than the whole document."""
    manifest = _make_manifest()
    rollup = [_make_rollup_row()]
    md = render_pairwise_markdown_with_ci(manifest, rollup)
    assert "complementarity_score [95% CI]" in md
    # Locate the dataset-block header row and assert it carries
    # the CI form (not the bare-mean form).
    lines = md.splitlines()
    dataset_header_idx = next(
        i for i, line in enumerate(lines) if line.startswith("### fake_binary")
    )
    # The header row is two lines after the `### ...` heading
    # (the heading, a blank line, then the table header row).
    table_header = lines[dataset_header_idx + 2]
    assert "complementarity_score [95% CI]" in table_header
    assert "| complementarity_score |" not in table_header


def test_render_pairwise_with_ci_falls_back_on_manifest_fingerprint_mismatch() -> None:
    """Stale rollup (fingerprint mismatch) -> std + footnote."""
    manifest = _make_manifest()
    rollup = [_make_rollup_row(manifest_fingerprint="oldfingerprint")]
    md = render_pairwise_markdown_with_ci(
        manifest,
        rollup,
        expected_manifest_fingerprint="newfingerprint",
    )
    assert "Bootstrap rollup is stale" in md
    assert "complementarity_score [95% CI]" not in md


def test_render_pairwise_with_ci_surfaces_aggregator_failed_footnote_on_wrapper_caught() -> None:
    """aggregator_error_class set -> std + 'aggregator failed' footnote."""
    manifest = _make_manifest()
    md = render_pairwise_markdown_with_ci(
        manifest, [], aggregator_error_class="RawRollupError"
    )
    assert "Bootstrap aggregator failed" in md
    assert "RawRollupError" in md


def test_render_pairwise_with_ci_regression_sentinel_produces_bootstrap_skipped_footnote() -> None:
    """Round-1 qa-C2 cross-pin: a regression PairwiseRollupRow
    with bootstrap_skipped_reason="regression_complementarity_undefined"
    surfaces in the 'Bootstrap skipped' footnote with the EXACT
    reason string."""
    manifest = _make_manifest()
    regression_row = _make_rollup_row(
        task_type="regression_point",
        primary_loss_mean=None,
        primary_loss_ci_lo=None,
        primary_loss_ci_hi=None,
        bootstrap_skipped_reason="regression_complementarity_undefined",
        n_cells_evaluated=0,
    )
    md = render_pairwise_markdown_with_ci(manifest, [regression_row, _make_rollup_row()])
    assert "Bootstrap skipped" in md
    assert "regression_complementarity_undefined" in md


def test_render_pairwise_with_ci_manifest_unreadable_appends_freshness_footnote() -> None:
    """manifest_unreadable=True -> CI body + 'freshness check skipped' footnote."""
    manifest = _make_manifest()
    md = render_pairwise_markdown_with_ci(
        manifest, [_make_rollup_row()], manifest_unreadable=True
    )
    assert "complementarity_score [95% CI]" in md
    assert "Bootstrap freshness check skipped" in md


def test_render_pairwise_with_ci_surfaces_rollup_skipped_in_separate_footnote() -> None:
    """Stage-4 qa-C1 closure: the 'Bootstrap skipped' footnote
    for the pairwise rollup uses the B6-specific column-headers
    `Dataset | Model A | Model B | Reason`. Pins the header_labels
    argument passed to render_rollup_skipped_footnote so a copy-
    paste from the B5 (`Dataset | Model |`) shape would fail."""
    manifest = _make_manifest()
    rollup = [
        _make_rollup_row(),  # normal CI row
        _make_rollup_row(
            model_a="bad_a",
            model_b="bad_b",
            primary_loss_mean=None,
            primary_loss_ci_lo=None,
            primary_loss_ci_hi=None,
            bootstrap_skipped_reason="loader_failed: missing predictions",
            n_cells_evaluated=0,
        ),
    ]
    md = render_pairwise_markdown_with_ci(manifest, rollup)
    assert "Bootstrap skipped" in md
    # Footnote table header carries Model A and Model B columns
    # (not the B5 single-Model header).
    lines = md.splitlines()
    skipped_idx = next(i for i, line in enumerate(lines) if line == "### Bootstrap skipped")
    # Table header is two lines below the section heading.
    table_header = lines[skipped_idx + 2]
    assert "Model A" in table_header
    assert "Model B" in table_header
    # The reason string surfaces verbatim.
    assert "loader_failed: missing predictions" in md


def test_render_pairwise_with_ci_marks_partial_fold_cell_with_asterisk() -> None:
    """Stage-4 qa-C2 closure: when rollup_row.n_cells_evaluated <
    n_seeds * n_folds, the CI cell ends with `*`. The format_ci_cell
    asterisk had no coverage prior to this test."""
    # Manifest has 2 folds (fold_index 0 + 1) for one pair.
    rows = []
    for s in (0,):
        for f in (0, 1):
            rows.append(
                {
                    "library_git_sha": "0" * 40,
                    "run_id": "b14-partial",
                    "started_at_utc": "2026-05-30T00:00:00+00:00",
                    "dataset_name": "fake_binary",
                    "model_a": "a",
                    "model_b": "b",
                    "seed": s,
                    "fold_index": f,
                    "task_type": "binary",
                    "skipped_reason": None,
                    "n_samples": 100,
                    "n11": 40, "n10": 10, "n01": 15, "n00": 35,
                    "yule_q": 0.7, "phi": 0.5,
                    "disagreement_rate": 0.25,
                    "double_fault_rate": 0.1,
                    "pearson_pred_corr": 0.6,
                    "spearman_pred_corr": 0.55,
                    "pearson_error_corr": 0.30,
                }
            )
    manifest = pd.DataFrame(rows)
    # rollup_row.n_cells_evaluated=1 < n_seeds=1 * n_folds=2 = 2 -> partial=True
    rollup = [_make_rollup_row(n_cells_evaluated=1, n_seeds=1)]
    md = render_pairwise_markdown_with_ci(manifest, rollup)
    assert "0.9500 [0.8800, 1.0200]*" in md


def test_render_pairwise_with_ci_renders_no_ci_when_rollup_missing_for_pair() -> None:
    """Stage-4 code-I2 closure: when the manifest has a pair but
    the rollup has NO row covering it, the CI cell renders as the
    `(no CI)` sentinel and the rest of the table is unaffected."""
    manifest = _make_manifest()
    # Rollup covers a DIFFERENT pair, leaving (a, b) uncovered.
    rollup = [_make_rollup_row(model_a="other", model_b="pair")]
    md = render_pairwise_markdown_with_ci(manifest, rollup)
    assert "(no CI)" in md
    # The dataset-block table still uses the CI header (the
    # variant is active even though one row is missing).
    assert "complementarity_score [95% CI]" in md


# --- render_from_dir dispatch tests ----------------------------------------


def _setup_b5_ensemble(tmp_path: Path) -> Path:
    register_all_fakes_and_get_panels()
    # Need two classification models for at least one pairwise pair.
    config = BenchmarkConfig(
        datasets=("fake_binary",),
        models=("fake_constant_binary", "fake_nan_proba_classifier"),
        experiments=(
            ExperimentSpec(kind="raw_loss", seeds=(0,)),
            ExperimentSpec(kind="ensemble", seeds=(0,)),
        ),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    env = build_run_environment(profile="smoke")
    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    run_raw_loss(config, output_root=output_root, env=env)
    run_ensemble(config, output_root=output_root, env=env)
    manifest = build_run_manifest(
        config=config,
        run_id="b14-rt-cli",
        library_git_sha="0" * 40,
        profile="smoke",
        hardware_tier="cpu",
        output_root=output_root,
    )
    write_run_manifest(output_root, manifest)
    return output_root


def test_render_from_dir_falls_back_silently_when_rollup_file_absent(
    tmp_path: Path,
) -> None:
    """No rollup file present -> std variant silently."""
    output_root = _setup_b5_ensemble(tmp_path)
    md = render_from_dir(output_root)
    assert "complementarity_score [95% CI]" not in md


def test_render_from_dir_surfaces_aggregator_failed_sentinel(
    tmp_path: Path,
) -> None:
    """When the CLI wrapper dropped the failure sentinel, the
    renderer reads it and surfaces the 'aggregator failed'
    footnote in the std fallback."""
    output_root = _setup_b5_ensemble(tmp_path)
    pairwise_aggregator_failed_sentinel_path(output_root).write_text(
        "RawRollupError", encoding="utf-8"
    )
    md = render_from_dir(output_root)
    assert "Bootstrap aggregator failed" in md
    assert "RawRollupError" in md


def test_render_from_dir_renders_ci_when_rollup_present_and_fingerprint_matches(
    tmp_path: Path,
) -> None:
    """Rollup file present + matching fingerprint -> CI variant."""
    from benchmarks.run_manifest import load_run_manifest

    output_root = _setup_b5_ensemble(tmp_path)
    manifest = load_run_manifest(output_root)
    rollup_row = _make_rollup_row(manifest_fingerprint=manifest.fingerprint())
    write_pairwise_rollup(output_root, [rollup_row])

    md = render_from_dir(output_root)
    assert "complementarity_score [95% CI]" in md


def test_render_from_dir_falls_back_on_stale_rollup(
    tmp_path: Path,
) -> None:
    """Rollup file present + mismatched fingerprint -> std + footnote."""
    output_root = _setup_b5_ensemble(tmp_path)
    rollup_row = _make_rollup_row(manifest_fingerprint="oldfingerprint")
    write_pairwise_rollup(output_root, [rollup_row])

    md = render_from_dir(output_root)
    assert "Bootstrap rollup is stale" in md


def test_render_from_dir_freshness_check_skipped_when_manifest_corrupt(
    tmp_path: Path,
) -> None:
    """Rollup present + corrupt run_manifest -> CI variant + footnote."""
    output_root = _setup_b5_ensemble(tmp_path)
    rollup_row = _make_rollup_row(manifest_fingerprint="anything")
    write_pairwise_rollup(output_root, [rollup_row])
    run_manifest_path(output_root).write_bytes(b"{not valid json")

    md = render_from_dir(output_root)
    assert "complementarity_score [95% CI]" in md
    assert "Bootstrap freshness check skipped" in md


def test_render_from_dir_renders_ci_without_freshness_footnote_when_manifest_absent(
    tmp_path: Path,
) -> None:
    """Rollup present + run_manifest absent -> silent CI variant
    (distinct from corrupt manifest case)."""
    output_root = _setup_b5_ensemble(tmp_path)
    rollup_row = _make_rollup_row(manifest_fingerprint="anything")
    write_pairwise_rollup(output_root, [rollup_row])
    # Delete the run_manifest written during setup so the test
    # exercises the absent-not-corrupt path.
    run_manifest_path(output_root).unlink()

    md = render_from_dir(output_root)
    assert "complementarity_score [95% CI]" in md
    assert "Bootstrap rollup is stale" not in md
    assert "Bootstrap freshness check skipped" not in md


def test_render_from_dir_falls_back_silently_when_rollup_file_empty(
    tmp_path: Path,
) -> None:
    """Rollup parquet exists but loads to zero rows -> std variant."""
    output_root = _setup_b5_ensemble(tmp_path)
    write_pairwise_rollup(output_root, [])
    assert pairwise_rollup_path(output_root).exists()

    md = render_from_dir(output_root)
    assert "complementarity_score [95% CI]" not in md


