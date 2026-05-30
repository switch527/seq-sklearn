"""Phase B16 D-B13.4 ensemble-lift renderer tests.

24 named tests:
- 15 footnote-precedence + behavior tests (B16.6 design)
- 6 render_from_dir dispatch tests (B16.6 design)
- 3 Stage 3 R1 closure tests (manifest-absent dispatch,
  empty-rollup fallback, dead-predicate mutation pin)
"""

from pathlib import Path

import pytest
from benchmarks.bootstrap_manifest import (
    EnsembleLiftRollupRow,
    ensemble_lift_aggregator_failed_sentinel_path,
    write_ensemble_lift_rollup,
)
from benchmarks.experiments.ensemble_lift import (
    EnsembleLiftExperimentResult,
    PerDatasetLift,
    WilcoxonResult,
)
from benchmarks.report.ensemble_lift import (
    render_ensemble_lift_markdown_with_ci,
    render_from_dir,
)


def _make_lift_result(rows: tuple[PerDatasetLift, ...]) -> EnsembleLiftExperimentResult:
    return EnsembleLiftExperimentResult(
        run_id="b16-render-test",
        seq_family="seq_sklearn",
        baseline_family="gbm",
        rows=rows,
        wilcoxon=WilcoxonResult(
            statistic=1.0,
            p_value=0.10,
            holm_adjusted_p_value=0.10,
            n_datasets=len(rows),
            family_size=1,
        ),
    )


def _make_per_dataset_row(
    *,
    dataset_name: str = "ds_one",
    task_type: str = "binary",
    primary_loss_column: str = "log_loss",
    n_cells_paired: int = 4,
    loss_gbm_only_mean: float | None = 0.50,
    loss_gbm_plus_seq_mean: float | None = 0.30,
    delta_loss_mean: float | None = 0.20,
    delta_loss_std: float | None = 0.01,
    oracle_loss_mean: float | None = 0.25,
    oracle_delta_loss_mean: float | None = 0.25,
    no_gbm_predictions: bool = False,
    no_seq_predictions: bool = False,
) -> PerDatasetLift:
    return PerDatasetLift(
        dataset_name=dataset_name,
        task_type=task_type,
        primary_loss_column=primary_loss_column,
        n_cells_paired=n_cells_paired,
        loss_gbm_only_mean=loss_gbm_only_mean,
        loss_gbm_plus_seq_mean=loss_gbm_plus_seq_mean,
        delta_loss_mean=delta_loss_mean,
        delta_loss_std=delta_loss_std,
        oracle_loss_mean=oracle_loss_mean,
        oracle_delta_loss_mean=oracle_delta_loss_mean,
        no_gbm_predictions=no_gbm_predictions,
        no_seq_predictions=no_seq_predictions,
    )


def _make_rollup_row(
    *,
    dataset_name: str = "ds_one",
    task_type: str = "binary",
    primary_loss_column: str = "log_loss",
    n_seeds: int = 2,
    n_folds: int = 2,
    n_cells_paired: int = 4,
    n_pair_grid: int | None = None,
    n_oracle_cells_paired: int | None = None,
    n_skipped_cells: int = 0,
    primary_metric_mean: float | None = 0.20,
    primary_metric_ci_lo: float | None = 0.15,
    primary_metric_ci_hi: float | None = 0.25,
    oracle_metric_mean: float | None = 0.10,
    oracle_metric_ci_lo: float | None = 0.08,
    oracle_metric_ci_hi: float | None = 0.12,
    bootstrap_skipped_reason: str | None = None,
    manifest_fingerprint: str = "f" * 64,
) -> EnsembleLiftRollupRow:
    # B19 / D-B16.7: when n_pair_grid is not supplied, default to
    # n_cells_paired so the renderer's partial-flag computation
    # matches the pre-B19 symmetric-roster happy-path semantics.
    if n_pair_grid is None:
        n_pair_grid = n_cells_paired
    # B20 / D-B16.1: when n_oracle_cells_paired is not supplied, default
    # to n_cells_paired (every paired cell has a computable oracle_loss)
    # so the renderer's oracle-partial flag does not fire on the
    # symmetric-roster default. Mirrors the B19 n_pair_grid None-fallback
    # pattern: the schema field stays `int = Field(ge=0)` with no Python
    # default; the fallback lives in the factory body only.
    if n_oracle_cells_paired is None:
        n_oracle_cells_paired = n_cells_paired
    return EnsembleLiftRollupRow(
        dataset_name=dataset_name,
        task_type=task_type,
        primary_metric="delta_loss",
        primary_loss_column=primary_loss_column,
        n_seeds=n_seeds,
        n_folds=n_folds,
        n_cells_paired=n_cells_paired,
        n_pair_grid=n_pair_grid,
        n_oracle_cells_paired=n_oracle_cells_paired,
        n_skipped_cells=n_skipped_cells,
        primary_metric_mean=primary_metric_mean,
        primary_metric_ci_lo=primary_metric_ci_lo,
        primary_metric_ci_hi=primary_metric_ci_hi,
        oracle_metric_mean=oracle_metric_mean,
        oracle_metric_ci_lo=oracle_metric_ci_lo,
        oracle_metric_ci_hi=oracle_metric_ci_hi,
        bootstrap_seed=42,
        bootstrap_n_resamples=10_000,
        bootstrap_numpy_version="2.3.0",
        bootstrap_skipped_reason=bootstrap_skipped_reason,
        manifest_fingerprint=manifest_fingerprint,
    )


# =============================================================================
# Footnote-precedence + behavior (15 tests)
# =============================================================================


def test_render_ensemble_lift_with_ci_renders_mean_and_interval() -> None:
    """The CI variant renders `Δloss [95% CI]` with mean + bracketed
    interval; the bare Δloss column header is replaced."""
    result = _make_lift_result((_make_per_dataset_row(),))
    rollup = [_make_rollup_row()]
    md = render_ensemble_lift_markdown_with_ci(result, rollup)
    assert "Δloss [95% CI]" in md
    assert "0.2000 [0.1500, 0.2500]" in md


def test_render_ensemble_lift_without_ci_falls_back_to_scalar() -> None:
    """Empty rollup -> std variant silently (source 2 of the
    precedence). The bare Δloss + Δstd columns surface."""
    result = _make_lift_result((_make_per_dataset_row(),))
    md = render_ensemble_lift_markdown_with_ci(result, [])
    assert "Δloss [95% CI]" not in md
    assert "Δstd" in md


def test_render_ensemble_lift_with_ci_drops_delta_loss_mean_header_when_ci_present() -> None:
    """qa-I2 two-column-to-one pin (header direction):
    the bare `Δloss` standalone column header is ABSENT
    from the per-dataset table header row AND
    `Δloss [95% CI]` IS present."""
    result = _make_lift_result((_make_per_dataset_row(),))
    rollup = [_make_rollup_row()]
    md = render_ensemble_lift_markdown_with_ci(result, rollup)
    # Find the table header row and assert the bare "Δloss" column
    # header is absent in favor of the combined "Δloss [95% CI]".
    header_line = next(line for line in md.splitlines() if line.startswith("| dataset |"))
    header_cells = [c.strip() for c in header_line.split("|") if c.strip()]
    assert "Δloss" not in header_cells
    assert "Δloss [95% CI]" in header_cells


def test_render_ensemble_lift_with_ci_drops_delta_loss_std_column_when_ci_present() -> None:
    """qa-I2 two-column-to-one pin: `Δstd` column header is ABSENT
    from the table header row AND `Δloss [95% CI]` IS present."""
    result = _make_lift_result((_make_per_dataset_row(),))
    rollup = [_make_rollup_row()]
    md = render_ensemble_lift_markdown_with_ci(result, rollup)
    header_line = next(line for line in md.splitlines() if line.startswith("| dataset |"))
    header_cells = [c.strip() for c in header_line.split("|") if c.strip()]
    assert "Δstd" not in header_cells
    assert "Δloss [95% CI]" in header_cells


def test_render_ensemble_lift_with_ci_falls_back_on_manifest_fingerprint_mismatch() -> None:
    """Source 3 of the precedence: rollup fingerprint mismatch ->
    std variant + 'stale' footnote."""
    result = _make_lift_result((_make_per_dataset_row(),))
    rollup = [_make_rollup_row(manifest_fingerprint="a" * 64)]
    md = render_ensemble_lift_markdown_with_ci(
        result, rollup, expected_manifest_fingerprint="b" * 64
    )
    assert "Δloss [95% CI]" not in md
    assert "Bootstrap rollup is stale" in md
    assert "Δstd" in md


def test_render_ensemble_lift_with_ci_surfaces_aggregator_failed_footnote_on_wrapper_caught() -> (
    None
):
    """Source 1 of the precedence: aggregator_error_class set ->
    std variant + 'Bootstrap aggregator failed' footnote naming
    the error class."""
    result = _make_lift_result((_make_per_dataset_row(),))
    md = render_ensemble_lift_markdown_with_ci(result, [], aggregator_error_class="RawRollupError")
    assert "Bootstrap aggregator failed" in md
    assert "RawRollupError" in md
    assert "Δloss [95% CI]" not in md


def test_render_ensemble_lift_with_ci_manifest_unreadable_appends_freshness_footnote() -> None:
    """Source 4 of the precedence: CI body renders; when
    `manifest_unreadable=True`, append a freshness-skipped
    footnote at the bottom."""
    result = _make_lift_result((_make_per_dataset_row(),))
    rollup = [_make_rollup_row()]
    md = render_ensemble_lift_markdown_with_ci(result, rollup, manifest_unreadable=True)
    assert "Δloss [95% CI]" in md
    assert "freshness check skipped" in md


def test_render_ensemble_lift_with_ci_surfaces_skipped_cells_footnote_when_n_skipped_cells_nonzero() -> (
    None
):
    """Source 5 footnote pin: a rollup row with
    n_skipped_cells > 0 surfaces the 'Partial coverage'
    footnote."""
    result = _make_lift_result((_make_per_dataset_row(),))
    rollup = [_make_rollup_row(n_skipped_cells=2)]
    md = render_ensemble_lift_markdown_with_ci(result, rollup)
    assert "Partial coverage" in md
    assert "2 / 4" in md


def test_render_ensemble_lift_with_ci_surfaces_skipped_cells_footnote_without_asterisk_when_no_pairing_asymmetry() -> (
    None
):
    """Source 5 vs partial-fold asterisk independence: when
    n_cells_paired == n_seeds * n_folds, the CI cell carries no
    asterisk even if n_skipped_cells > 0."""
    result = _make_lift_result((_make_per_dataset_row(),))
    rollup = [_make_rollup_row(n_skipped_cells=2, n_cells_paired=4, n_seeds=2, n_folds=2)]
    md = render_ensemble_lift_markdown_with_ci(result, rollup)
    # Find the CI cell line; assert no asterisk on the value.
    ci_line = next(line for line in md.splitlines() if "0.2000 [0.1500, 0.2500]" in line)
    assert "*" not in ci_line


def test_render_ensemble_lift_with_ci_surfaces_both_freshness_and_skipped_cells_footnotes_when_both_apply() -> (
    None
):
    """Source 4 + source 5 co-occurrence: both freshness skipped
    AND partial coverage footnotes appear when both apply."""
    result = _make_lift_result((_make_per_dataset_row(),))
    rollup = [_make_rollup_row(n_skipped_cells=2)]
    md = render_ensemble_lift_markdown_with_ci(result, rollup, manifest_unreadable=True)
    assert "freshness check skipped" in md
    assert "Partial coverage" in md


def test_render_ensemble_lift_with_ci_marks_partial_fold_cell_with_asterisk() -> None:
    """When `n_cells_paired < n_pair_grid`, the CI cell
    receives a trailing asterisk per the shared `format_ci_cell`
    contract. B19 / D-B16.7 closure: the expected count is now
    `n_pair_grid` (intersection cardinality), not
    `n_seeds * n_folds` (union)."""
    result = _make_lift_result((_make_per_dataset_row(),))
    rollup = [_make_rollup_row(n_cells_paired=3, n_pair_grid=4, n_seeds=2, n_folds=2)]
    md = render_ensemble_lift_markdown_with_ci(result, rollup)
    ci_line = next(line for line in md.splitlines() if "0.2000 [0.1500, 0.2500]" in line)
    assert "*" in ci_line


def test_render_ensemble_lift_with_ci_renders_no_ci_when_rollup_missing_for_dataset() -> None:
    """The result has a dataset but the rollup has no row for it
    -> the table renders a `(no CI)` cell rather than dropping
    the row."""
    result = _make_lift_result((_make_per_dataset_row(dataset_name="missing_in_rollup"),))
    rollup = [_make_rollup_row(dataset_name="other_dataset")]
    md = render_ensemble_lift_markdown_with_ci(result, rollup)
    assert "missing_in_rollup" in md
    assert "(no CI)" in md


def test_render_ensemble_lift_with_ci_no_gbm_predictions_routes_to_incomplete_block() -> None:
    """qa-I1: the pre-existing B11 incomplete-block path receives
    the `no_gbm_predictions` PerDatasetLift row; the shared
    rollup-skipped helper is NOT used for this sentinel."""
    incomplete_row = _make_per_dataset_row(
        dataset_name="no_gbm_ds",
        loss_gbm_only_mean=None,
        loss_gbm_plus_seq_mean=None,
        delta_loss_mean=None,
        delta_loss_std=None,
        oracle_loss_mean=None,
        oracle_delta_loss_mean=None,
        no_gbm_predictions=True,
        n_cells_paired=0,
    )
    result = _make_lift_result((incomplete_row,))
    rollup = [
        _make_rollup_row(
            dataset_name="no_gbm_ds",
            primary_metric_mean=None,
            primary_metric_ci_lo=None,
            primary_metric_ci_hi=None,
            n_cells_paired=0,
            bootstrap_skipped_reason="no_gbm_predictions",
        )
    ]
    md = render_ensemble_lift_markdown_with_ci(result, rollup)
    assert "Incomplete datasets" in md
    assert "no GBM cells in manifest" in md
    # The shared rollup-skipped helper is NOT engaged for this
    # sentinel.
    assert "Bootstrap skipped" not in md


def test_render_ensemble_lift_with_ci_no_seq_predictions_routes_to_incomplete_block() -> None:
    """R2 qa-I1 symmetric mirror: `no_seq_predictions` routes
    through the same incomplete-block path."""
    incomplete_row = _make_per_dataset_row(
        dataset_name="no_seq_ds",
        loss_gbm_only_mean=None,
        loss_gbm_plus_seq_mean=None,
        delta_loss_mean=None,
        delta_loss_std=None,
        oracle_loss_mean=None,
        oracle_delta_loss_mean=None,
        no_seq_predictions=True,
        n_cells_paired=0,
    )
    result = _make_lift_result((incomplete_row,))
    rollup = [
        _make_rollup_row(
            dataset_name="no_seq_ds",
            primary_metric_mean=None,
            primary_metric_ci_lo=None,
            primary_metric_ci_hi=None,
            n_cells_paired=0,
            bootstrap_skipped_reason="no_seq_predictions",
        )
    ]
    md = render_ensemble_lift_markdown_with_ci(result, rollup)
    assert "Incomplete datasets" in md
    assert "no seq cells in manifest" in md
    assert "Bootstrap skipped" not in md


def test_render_ensemble_lift_with_ci_all_cells_skipped_routes_to_shared_render_rollup_skipped_footnote() -> (
    None
):
    """The only sentinel that uses the new shared-helper path:
    `all_cells_skipped_in_manifest` surfaces via
    `render_rollup_skipped_footnote`, NOT through the
    PerDatasetLift incomplete block."""
    result = _make_lift_result((_make_per_dataset_row(),))
    rollup = [
        _make_rollup_row(),  # valid CI row keeps the table populated
        _make_rollup_row(
            dataset_name="all_skipped_ds",
            primary_metric_mean=None,
            primary_metric_ci_lo=None,
            primary_metric_ci_hi=None,
            n_cells_paired=0,
            bootstrap_skipped_reason="all_cells_skipped_in_manifest",
        ),
    ]
    md = render_ensemble_lift_markdown_with_ci(result, rollup)
    assert "Bootstrap skipped" in md
    assert "all_cells_skipped_in_manifest" in md
    assert "all_skipped_ds" in md


# =============================================================================
# render_from_dir dispatch (6 tests)
# =============================================================================


def test_render_from_dir_falls_back_silently_when_rollup_file_absent(tmp_path: Path) -> None:
    """No rollup file present -> std variant rendered silently."""
    result = _make_lift_result((_make_per_dataset_row(),))
    md = render_from_dir(tmp_path, lift_result=result)
    assert "Δloss [95% CI]" not in md
    assert "Δstd" in md


def test_render_from_dir_surfaces_aggregator_failed_sentinel(tmp_path: Path) -> None:
    """Failure sentinel present -> CI variant with the aggregator-
    failed footnote pointing at the sentinel's content."""
    sentinel = ensemble_lift_aggregator_failed_sentinel_path(tmp_path)
    sentinel.write_text("RawRollupError", encoding="utf-8")
    result = _make_lift_result((_make_per_dataset_row(),))
    md = render_from_dir(tmp_path, lift_result=result)
    assert "Bootstrap aggregator failed" in md
    assert "RawRollupError" in md


def test_render_from_dir_renders_ci_when_rollup_present_and_fingerprint_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rollup present + freshness OK -> CI variant body."""
    fp = "c" * 64
    rows = [_make_rollup_row(manifest_fingerprint=fp)]
    write_ensemble_lift_rollup(tmp_path, rows)

    # Stub the run_manifest layer so freshness check passes.
    import benchmarks.report.ensemble_lift as _module

    def _rm_path(_root: Path) -> Path:
        return tmp_path / "rm.json"

    monkeypatch.setattr(_module, "run_manifest_path", _rm_path)
    (tmp_path / "rm.json").write_text("{}", encoding="utf-8")

    class _FakeManifest:
        def fingerprint(self) -> str:
            return fp

    def _load_match(_root: Path) -> _FakeManifest:
        return _FakeManifest()

    monkeypatch.setattr(_module, "load_run_manifest", _load_match)

    result = _make_lift_result((_make_per_dataset_row(),))
    md = render_from_dir(tmp_path, lift_result=result)
    assert "Δloss [95% CI]" in md


def test_render_from_dir_falls_back_on_stale_rollup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rollup present but fingerprint mismatch -> std variant +
    stale footnote."""
    rows = [_make_rollup_row(manifest_fingerprint="d" * 64)]
    write_ensemble_lift_rollup(tmp_path, rows)

    import benchmarks.report.ensemble_lift as _module

    def _rm_path(_root: Path) -> Path:
        return tmp_path / "rm.json"

    monkeypatch.setattr(_module, "run_manifest_path", _rm_path)
    (tmp_path / "rm.json").write_text("{}", encoding="utf-8")

    class _FakeManifest:
        def fingerprint(self) -> str:
            return "e" * 64

    def _load_mismatch(_root: Path) -> _FakeManifest:
        return _FakeManifest()

    monkeypatch.setattr(_module, "load_run_manifest", _load_mismatch)

    result = _make_lift_result((_make_per_dataset_row(),))
    md = render_from_dir(tmp_path, lift_result=result)
    assert "Bootstrap rollup is stale" in md
    assert "Δloss [95% CI]" not in md


def test_render_from_dir_renders_ci_with_freshness_skipped_when_run_manifest_unreadable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rollup present + run_manifest corrupt -> CI variant with the
    freshness-skipped footnote (source 4)."""
    rows = [_make_rollup_row(manifest_fingerprint="f" * 64)]
    write_ensemble_lift_rollup(tmp_path, rows)

    import benchmarks.report.ensemble_lift as _module

    def _rm_path(_root: Path) -> Path:
        return tmp_path / "rm.json"

    monkeypatch.setattr(_module, "run_manifest_path", _rm_path)
    (tmp_path / "rm.json").write_text("{not valid json", encoding="utf-8")

    def _raise(_root: Path) -> object:
        raise ValueError("simulated corrupt manifest")

    monkeypatch.setattr(_module, "load_run_manifest", _raise)

    result = _make_lift_result((_make_per_dataset_row(),))
    md = render_from_dir(tmp_path, lift_result=result)
    assert "Δloss [95% CI]" in md
    assert "freshness check skipped" in md


def test_render_from_dir_failure_sentinel_takes_precedence_over_rollup_file(
    tmp_path: Path,
) -> None:
    """When BOTH the failure sentinel AND a rollup file exist on
    disk, the failure sentinel wins (source 1 has the highest
    precedence)."""
    sentinel = ensemble_lift_aggregator_failed_sentinel_path(tmp_path)
    sentinel.write_text("RawRollupError", encoding="utf-8")
    write_ensemble_lift_rollup(tmp_path, [_make_rollup_row()])

    result = _make_lift_result((_make_per_dataset_row(),))
    md = render_from_dir(tmp_path, lift_result=result)
    assert "Bootstrap aggregator failed" in md
    assert "RawRollupError" in md
    assert "Δloss [95% CI]" not in md


# --- Stage 3 R1 closures ---------------------------------------------------


def test_render_from_dir_renders_ci_without_freshness_footnote_when_manifest_absent(
    tmp_path: Path,
) -> None:
    """Closes Stage 3 R1 CRITICAL: rollup file present and
    non-empty BUT `run_manifest.json` absent -> CI variant
    renders with NO freshness-skipped footnote AND NO stale
    footnote (expected_manifest_fingerprint stays None,
    manifest_unreadable stays False)."""
    write_ensemble_lift_rollup(tmp_path, [_make_rollup_row()])
    # Do NOT write run_manifest.json into tmp_path.

    result = _make_lift_result((_make_per_dataset_row(),))
    md = render_from_dir(tmp_path, lift_result=result)
    assert "Δloss [95% CI]" in md
    assert "freshness check skipped" not in md
    assert "Bootstrap rollup is stale" not in md


def test_render_from_dir_falls_back_silently_when_rollup_file_empty(
    tmp_path: Path,
) -> None:
    """Closes Stage 3 R1 qa-I2: an EMPTY rollup file present on
    disk (zero-row parquet) routes to the std fallback path
    silently, matching the absent-file behavior. Pins the
    `if rollup:` false arm at the dispatch site."""
    write_ensemble_lift_rollup(tmp_path, [])

    result = _make_lift_result((_make_per_dataset_row(),))
    md = render_from_dir(tmp_path, lift_result=result)
    assert "Δloss [95% CI]" not in md
    assert "Δstd" in md
    assert "freshness check skipped" not in md


def test_render_ensemble_lift_with_ci_renders_no_ci_when_rollup_row_has_skipped_reason_with_complete_delta_loss() -> (
    None
):
    """Closes Stage 3 R1 code-I1 / qa-I1 dead-predicate gap +
    Stage 3 R2 qa CRITICAL (mutation-insensitivity): a complete
    `PerDatasetLift` row (`delta_loss_mean` is not None) paired
    with a rollup row that DOES carry a non-None
    `bootstrap_skipped_reason` AND VALID NUMERIC CI VALUES
    renders `(no CI)` in the CI cell. If the second predicate
    of the OR guard at `_render_complete_table_with_ci:148`
    were removed, the code would fall through to
    `format_ci_cell` with the row's real numeric values and
    surface `0.4000 [0.3500, 0.4500]` instead of `(no CI)`;
    the bare-equality `not in md` assertion on the numeric
    string then kills the mutation."""
    result = _make_lift_result((_make_per_dataset_row(),))
    # B27 / D-B26.2 closure: post-B27 the EnsembleLiftRollupRow
    # CI-sentinel validator rejects "all-set metrics + populated
    # bootstrap_skipped_reason" via normal construction. This
    # mutation pin deliberately constructs the invalid shape to
    # verify the renderer's defensive OR guard, so use
    # `model_construct` to bypass both validators (also the
    # B23 row-count validator).
    rollup = [
        EnsembleLiftRollupRow.model_construct(
            dataset_name="ds_one",
            task_type="binary",
            primary_metric="delta_loss",
            primary_loss_column="log_loss",
            n_seeds=2,
            n_folds=2,
            n_cells_paired=4,
            n_pair_grid=4,
            n_oracle_cells_paired=4,
            n_skipped_cells=0,
            primary_metric_mean=0.40,
            primary_metric_ci_lo=0.35,
            primary_metric_ci_hi=0.45,
            oracle_metric_mean=0.10,
            oracle_metric_ci_lo=0.08,
            oracle_metric_ci_hi=0.12,
            bootstrap_seed=42,
            bootstrap_n_resamples=10_000,
            bootstrap_numpy_version="2.3.0",
            bootstrap_skipped_reason="all_cells_skipped_in_manifest",
            manifest_fingerprint="f" * 64,
        ),
    ]
    md = render_ensemble_lift_markdown_with_ci(result, rollup)
    # The PerDatasetLift's ds_one is in the table but its CI cell
    # is "(no CI)" because the rollup row sentinel suppresses it.
    assert "ds_one" in md
    assert "(no CI)" in md
    # The full numeric interval that the rollup row carries must
    # NOT render in the table cell; this kills the mutation where
    # the second predicate is removed.
    assert "0.4000 [0.3500, 0.4500]" not in md
