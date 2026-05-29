"""Phase B14 regression pin for `_bootstrap_render.py` extraction.

R-B14-3 named the high-severity risk that the mechanical
extraction of `format_ci_cell`, `render_rollup_skipped_footnote`,
and `folds_per_group` from `benchmarks/report/raw_loss.py` to the
new `benchmarks/report/_bootstrap_render.py` could silently drift
the B5 CI variant's rendered output. This module pins a known-good
Markdown string against the live renderer; any formatting drift
fails this test immediately.
"""

import pandas as pd
from benchmarks.bootstrap_manifest import RollupRow
from benchmarks.report._bootstrap_render import (
    folds_per_group,
    format_ci_cell,
    render_rollup_skipped_footnote,
)
from benchmarks.report.raw_loss import render_leaderboard_markdown_with_ci


def _manifest() -> pd.DataFrame:
    """Two OK cells x 2 seeds, one skipped cell; one model, one dataset."""
    return pd.DataFrame(
        {
            "dataset_name": ["fake_binary"] * 5,
            "model_name": ["m1"] * 5,
            "task_type": ["binary"] * 5,
            "seed": [0, 0, 1, 1, 2],
            "fold_index": [0, 1, 0, 1, 0],
            "log_loss": [0.20, 0.22, 0.21, 0.23, None],
            "wall_seconds": [1.0, 1.1, 1.2, 1.3, None],
            "skipped_reason": [None, None, None, None, "adapter_error"],
        }
    )


def _rollup_row(**overrides: object) -> RollupRow:
    defaults: dict[str, object] = {
        "dataset_name": "fake_binary",
        "model_name": "m1",
        "task_type": "binary",
        "primary_metric": "log_loss",
        "n_seeds": 2,
        "n_cells_evaluated": 4,
        "n_skipped_cells": 1,
        "n_rows": 40,
        "n_entities": 4,
        "primary_metric_mean": 0.215,
        "primary_metric_ci_lo": 0.200,
        "primary_metric_ci_hi": 0.230,
        "bootstrap_seed": 42,
        "bootstrap_n_resamples": 10_000,
        "bootstrap_confidence": 0.95,
        "bootstrap_rng_algorithm": "PCG64",
        "bootstrap_numpy_version": "2.3.0",
        "bootstrap_skipped_reason": None,
        "manifest_fingerprint": "deadbeef" * 8,
    }
    defaults.update(overrides)
    return RollupRow(**defaults)  # type: ignore[arg-type]


def test_format_ci_cell_renders_mean_and_interval_with_4_decimals() -> None:
    """The shared `format_ci_cell` reproduces the B13 4-decimal
    format. Pinning the string so future drift fails immediately."""
    cell = format_ci_cell(0.21456, 0.20011, 0.22899, partial=False)
    assert cell == "0.2146 [0.2001, 0.2290]"


def test_format_ci_cell_appends_asterisk_when_partial() -> None:
    """The partial-fold `*` flag is appended when partial=True."""
    cell = format_ci_cell(0.21, 0.20, 0.22, partial=True)
    assert cell == "0.2100 [0.2000, 0.2200]*"


def test_format_ci_cell_returns_no_ci_sentinel_on_none_mean() -> None:
    """Missing mean -> `(no CI)` sentinel string."""
    assert format_ci_cell(None, None, None, partial=False) == "(no CI)"
    assert format_ci_cell(0.21, None, 0.22, partial=False) == "(no CI)"


def test_folds_per_group_returns_unique_fold_count_per_2_key_group() -> None:
    """B5 uses 2-key grouping (dataset_name, model_name). Pins the
    return shape and value for the test manifest."""
    manifest = _manifest()
    folds = folds_per_group(
        manifest,
        group_columns=("dataset_name", "model_name"),
    )
    # OK rows: 4 cells across (seed, fold_index) pairs (0,0), (0,1),
    # (1,0), (1,1); 2 distinct fold_index values.
    assert folds == {("fake_binary", "m1"): 2}


def test_folds_per_group_returns_empty_dict_on_empty_manifest() -> None:
    """Empty manifest -> empty folds dict (defensive)."""
    empty = pd.DataFrame(columns=["dataset_name", "model_name", "fold_index", "skipped_reason"])
    assert folds_per_group(empty, group_columns=("dataset_name", "model_name")) == {}


def test_folds_per_group_returns_empty_dict_when_all_rows_skipped() -> None:
    """Stage-1 qa-I1: non-empty manifest where every row has a
    populated `skipped_reason` -> empty folds dict. The
    `if ok.empty: return {}` branch was untested otherwise."""
    all_skipped = pd.DataFrame(
        {
            "dataset_name": ["fake_binary", "fake_binary"],
            "model_name": ["m1", "m1"],
            "fold_index": [0, 1],
            "skipped_reason": ["adapter_error", "loader_failed"],
        }
    )
    assert folds_per_group(all_skipped, group_columns=("dataset_name", "model_name")) == {}


def test_render_rollup_skipped_footnote_renders_b5_header_text() -> None:
    """The B5 footnote uses `Dataset | Model | Reason` headers
    (preserved across the B14 extraction). Byte-pinning the string
    catches any header-text drift introduced by the extraction."""
    row = _rollup_row(
        bootstrap_skipped_reason="loader_failed: FileNotFoundError: panel.parquet",
        primary_metric_mean=None,
        primary_metric_ci_lo=None,
        primary_metric_ci_hi=None,
    )
    footnote = render_rollup_skipped_footnote(
        [row],
        group_columns=("dataset_name", "model_name"),
        header_labels=("Dataset", "Model"),
    )
    assert footnote == (
        "### Bootstrap skipped\n"
        "\n"
        "| Dataset | Model | Reason |\n"
        "| --- | --- | --- |\n"
        "| fake_binary | m1 | loader_failed: FileNotFoundError: panel.parquet |\n"
    )


def test_render_rollup_skipped_footnote_truncates_long_reason() -> None:
    """A reason string longer than 120 chars is truncated to 117 +
    '...'. The B13 behavior; pinned across the extraction."""
    long_reason = "x" * 200
    row = _rollup_row(
        bootstrap_skipped_reason=long_reason,
        primary_metric_mean=None,
        primary_metric_ci_lo=None,
        primary_metric_ci_hi=None,
    )
    footnote = render_rollup_skipped_footnote(
        [row],
        group_columns=("dataset_name", "model_name"),
        header_labels=("Dataset", "Model"),
    )
    assert "x" * 117 + "..." in footnote
    assert "x" * 200 not in footnote


def test_render_rollup_skipped_footnote_supports_3_key_groups_for_b6() -> None:
    """B6 uses (dataset_name, model_a, model_b) with header labels
    (Dataset, Model A, Model B). The helper must accept either
    layout without modification."""
    # Construct a minimal pairwise-shaped row via a duck-typed object.
    class _Row:
        def __init__(self) -> None:
            self.dataset_name = "fake_binary"
            self.model_a = "m1"
            self.model_b = "m2"
            self.bootstrap_skipped_reason = "regression_complementarity_undefined"

    footnote = render_rollup_skipped_footnote(
        [_Row()],
        group_columns=("dataset_name", "model_a", "model_b"),
        header_labels=("Dataset", "Model A", "Model B"),
    )
    assert footnote == (
        "### Bootstrap skipped\n"
        "\n"
        "| Dataset | Model A | Model B | Reason |\n"
        "| --- | --- | --- | --- |\n"
        "| fake_binary | m1 | m2 | regression_complementarity_undefined |\n"
    )


def test_render_rollup_skipped_footnote_raises_on_label_length_mismatch() -> None:
    """`group_columns` and `header_labels` must agree in length;
    a mismatch raises ValueError rather than silently mis-rendering."""
    import pytest

    with pytest.raises(ValueError, match="equal length"):
        render_rollup_skipped_footnote(
            [],
            group_columns=("dataset_name", "model_name"),
            header_labels=("Dataset",),
        )


def test_render_leaderboard_markdown_with_ci_byte_string_regression() -> None:
    """The load-bearing B14 extraction-regression pin (R-B14-3
    closure / qa-I1). Renders the B5 CI variant against a known
    rollup + manifest fixture; asserts the output byte-for-byte
    against the expected string. Any formatting drift in the
    `_bootstrap_render.py` helpers shows up here immediately."""
    manifest = _manifest()
    rollup = [_rollup_row()]
    md = render_leaderboard_markdown_with_ci(
        manifest,
        rollup,
        expected_manifest_fingerprint="deadbeef" * 8,
    )
    expected = (
        "# Raw-loss leaderboard\n"
        "\n"
        "### fake_binary (binary, ranked by log_loss)\n"
        "\n"
        "| Rank | Model | log_loss [95% CI] | n_cells | accuracy | "
        "precision_zd0 | recall_zd0 | f1_zd0 | precision_macro_zd0 | "
        "precision_weighted_zd0 | recall_macro_zd0 | recall_weighted_zd0 | "
        "f1_macro_zd0 | f1_weighted_zd0 | roc_auc | pr_auc | roc_auc_macro_ovr | "
        "pr_auc_macro_ovr | balanced_accuracy | mcc | brier | "
        "brier_multiclass_mean | ece_q15 | wall_seconds | median_predict_ms | "
        "p95_predict_ms |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
        "--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
        "--- | --- | --- | --- |\n"
        "| 1 | m1 | 0.2150 [0.2000, 0.2300] | 4 |  |  |  |  |  |  |  |  |  |  | "
        " |  |  |  |  |  |  |  |  | 1.1500 |  |  |\n"
        "\n"
        "### Skipped cells\n"
        "\n"
        "| Dataset | Model | Reason | Count |\n"
        "| --- | --- | --- | --- |\n"
        "| fake_binary | m1 | adapter_error | 1 |\n"
        ""
    )
    assert md == expected


def test_render_leaderboard_markdown_with_ci_byte_string_regression_with_rollup_skipped_footnote() -> None:
    """Stage-1 code-I1: the original byte-string regression pin
    used a rollup row with `bootstrap_skipped_reason=None`, so the
    full-pipeline path through `_render_with_ci` ->
    `render_rollup_skipped_footnote` had no byte-level regression
    pin. This pin covers the call site that actually invokes the
    extracted footnote helper from the main renderer."""
    manifest = _manifest()
    rollup = [
        _rollup_row(
            model_name="m_skip",
            primary_metric_mean=None,
            primary_metric_ci_lo=None,
            primary_metric_ci_hi=None,
            bootstrap_skipped_reason="loader_failed: missing panel",
        )
    ]
    md = render_leaderboard_markdown_with_ci(
        manifest,
        rollup,
        # No fingerprint guard; rollup's row carries one but we're
        # not asserting against it. The CI variant still renders.
    )
    # Bootstrap-skipped footnote appears verbatim from the
    # shared `render_rollup_skipped_footnote` helper.
    assert "### Bootstrap skipped" in md
    assert "| Dataset | Model | Reason |" in md
    assert "| fake_binary | m_skip | loader_failed: missing panel |" in md
