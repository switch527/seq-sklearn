"""Phase B17 / D-B16.5 byte-identity pins + Guard B source-tree grep.

This file holds the permanent regression guards for the rename:
Guard A lives in `test_bootstrap_manifest.py` and pins the
schema-field invariant; the tests here pin both the renderer-
side CI cell shape (one byte-identity pin per CI-variant
renderer family) AND the source-tree absence of any stray
`primary_loss_*` reference outside the explicit whitelist.

The file was created during the B17 rename rollout but its
guards remain load-bearing across future phases. Tests here
prove:

1. The four CI-variant renderers (B6 pairwise, B7 training-time,
   B15 HPO-uplift, B16 ensemble-lift) successfully render against
   a deterministic fixture AND the output carries the expected
   `mean [lo, hi]*` CI cell shape with the renamed fields read
   correctly. A rename miss in any renderer would either fail
   AttributeError on the rollup-row read OR drop the CI cell to
   `(no CI)` because `format_ci_cell(None, None, None)` returns
   that sentinel.
2. Guard B: a source-tree scan asserts zero non-whitelisted
   references to `primary_loss_mean`, `primary_loss_ci_lo`,
   `primary_loss_ci_hi` anywhere under `benchmarks/` or `tests/`.
   `benchmarks/report/raw_loss.py` is whitelisted as a whole
   file because the unrelated `LeaderboardEntry` schema and its
   sort keys, formatters, and docstrings reference the same
   names; B5 rollup-row regressions inside `raw_loss.py` are
   independently covered by Guard A on `*RollupRow` schemas
   plus pyright on `extra="forbid"` attribute reads.

The B5 family is independently pinned by the pre-existing
`tests/benchmarks/test_bootstrap_render_regression.py` byte-pin
that already runs against `render_leaderboard_markdown_with_ci`;
no new B5 pin lives here.
"""

import re
from pathlib import Path

import pandas as pd
import pytest
from benchmarks.bootstrap_manifest import (
    EnsembleLiftRollupRow,
    HPOUpliftRollupRow,
    PairwiseRollupRow,
    TrainingTimeRollupRow,
)
from benchmarks.experiments.ensemble_lift import (
    EnsembleLiftExperimentResult,
    PerDatasetLift,
    WilcoxonResult,
)
from benchmarks.report.ensemble import render_pairwise_markdown_with_ci
from benchmarks.report.ensemble_lift import render_ensemble_lift_markdown_with_ci
from benchmarks.report.hpo_uplift import render_hpo_uplift_markdown_with_ci
from benchmarks.report.training_time import render_training_time_markdown_with_ci

# Matches the CI cell shape `mean [lo, hi]*` with a MANDATORY
# trailing asterisk for the partial-coverage flag. Every fixture
# in this file is configured so the rollup row's
# `n_cells_paired < n_seeds * n_folds` (or
# `n_cells_evaluated < n_seeds * n_folds` for the B5-shaped
# schemas), which means `format_ci_cell` is called with
# `partial=True` and appends the trailing `*`. Requiring the
# asterisk closes the qa-R1-I2 mutation-sensitivity gap: a
# hardwired `partial=False` at any of the four renderer call
# sites would drop the asterisk and fail this regex, even
# though the numeric interval would still render correctly.
# The four floats are formatted via `format_ci_cell` with
# `%.4f`.
_CI_CELL_RE = re.compile(r"-?\d+\.\d{4} \[-?\d+\.\d{4}, -?\d+\.\d{4}\]\*")


def _make_pairwise_manifest() -> pd.DataFrame:
    """Manifest fixture for the B6 pairwise renderer.

    Two paired models on a single dataset with one OK cell so the
    pairwise table can render at least one row. Shape mirrors
    `tests/benchmarks/test_ensemble_report.py::_make_manifest`."""
    rows: list[dict[str, object]] = []
    for s in (0, 1):
        rows.append(
            {
                "library_git_sha": "0" * 40,
                "run_id": "b17-pin",
                "started_at_utc": "2026-05-30T00:00:00+00:00",
                "dataset_name": "ds_one",
                "model_a": "model_a",
                "model_b": "model_b",
                "seed": s,
                "fold_index": 0,
                "task_type": "binary",
                "skipped_reason": None,
                "n_samples": 100,
                "n11": 40,
                "n10": 10,
                "n01": 15,
                "n00": 35,
                "yule_q": 0.7,
                "phi": 0.5,
                "disagreement_rate": 0.25,
                "double_fault_rate": 0.1,
                "pearson_pred_corr": 0.6,
                "spearman_pred_corr": 0.55,
                "pearson_error_corr": 0.30,
                "complementarity_score": 0.40,
                "model_a_log_loss": 0.45,
                "model_b_log_loss": 0.50,
            }
        )
    return pd.DataFrame(rows)


def _make_pairwise_rollup() -> list[PairwiseRollupRow]:
    """Single-row rollup for the B6 fixture.

    `n_cells_paired < n_seeds * n_folds = 1` is not provable on a
    1-cell fixture; the partial-coverage flag fires on
    `n_cells_evaluated < n_seeds * n_folds`. Set n_seeds=2,
    n_folds=1 so expected=2 > evaluated=1 -> asterisk."""
    return [
        PairwiseRollupRow(
            dataset_name="ds_one",
            model_a="model_a",
            model_b="model_b",
            task_type="binary",
            primary_metric="complementarity_score",
            n_seeds=2,
            n_cells_evaluated=1,
            n_skipped_cells=1,
            primary_metric_mean=0.40,
            primary_metric_ci_lo=0.35,
            primary_metric_ci_hi=0.45,
            bootstrap_seed=42,
            bootstrap_n_resamples=10_000,
            bootstrap_numpy_version="2.3.0",
            bootstrap_skipped_reason=None,
            manifest_fingerprint="a" * 64,
        ),
    ]


def test_render_pairwise_byte_identity_post_rename() -> None:
    """B17 R-B17-2 pin for the B6 pairwise CI renderer: the
    rendered output reads the renamed `primary_metric_*` fields
    from the rollup row and surfaces the standard `mean [lo, hi]`
    CI cell shape. A rename miss inside `ensemble.py` would
    AttributeError (caught by pyright already) OR render `(no CI)`
    silently (caught by this pin)."""
    manifest = _make_pairwise_manifest()
    rollup = _make_pairwise_rollup()
    md = render_pairwise_markdown_with_ci(manifest, rollup)
    # CI cell shape must appear; matches `0.4000 [0.3500, 0.4500]*`.
    assert _CI_CELL_RE.search(md), (
        f"pairwise renderer CI cell shape missing; output starts: {md[:400]!r}"
    )
    # Sentinel: the rename must not have left a (no CI) cell on a
    # row whose rollup has a valid numeric mean.
    assert "(no CI)" not in md
    # B21 / D-B21.1 deferral: the new audit fields are parquet-shard
    # columns only and must NOT surface in the rendered markdown.
    assert "bootstrap_ci_method" not in md
    assert "bootstrap_ci_fallback_reason" not in md
    # B22 / D-B22.1 deferral: per_fold_cis is parquet-only.
    assert "per_fold_cis" not in md


def _make_training_time_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset_name": "ds_one",
                "model_name": "model_a",
                "task_type": "binary",
                "seed": 0,
                "fold_index": 0,
                "skipped_reason": None,
                "wall_seconds": 12.5,
                "peak_rss_bytes": 1_000_000.0,
                "peak_cuda_bytes": None,
                "hardware_tier": "cpu",
                "log_loss": 0.30,
            },
        ]
    )


def _make_training_time_rollup() -> list[TrainingTimeRollupRow]:
    return [
        TrainingTimeRollupRow(
            dataset_name="ds_one",
            model_name="model_a",
            hardware_tier="cpu",
            task_type="binary",
            primary_metric="wall_seconds",
            n_seeds=2,
            n_cells_evaluated=1,
            n_skipped_cells=1,
            primary_metric_mean=12.5,
            primary_metric_ci_lo=10.0,
            primary_metric_ci_hi=15.0,
            bootstrap_seed=42,
            bootstrap_n_resamples=10_000,
            bootstrap_numpy_version="2.3.0",
            bootstrap_skipped_reason=None,
            manifest_fingerprint="b" * 64,
        ),
    ]


def test_render_training_time_byte_identity_post_rename() -> None:
    """B17 R-B17-2 pin for the B7 training-time CI renderer."""
    manifest = _make_training_time_manifest()
    rollup = _make_training_time_rollup()
    md = render_training_time_markdown_with_ci(manifest, rollup)
    assert _CI_CELL_RE.search(md), (
        f"training-time renderer CI cell shape missing; output starts: {md[:400]!r}"
    )
    assert "(no CI)" not in md
    # B21 / D-B21.1 deferral: new audit fields must NOT surface in the markdown.
    assert "bootstrap_ci_method" not in md
    assert "bootstrap_ci_fallback_reason" not in md
    # B22 / D-B22.1 deferral: per_fold_cis is parquet-only.
    assert "per_fold_cis" not in md


def _make_hpo_uplift_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset_name": "ds_one",
                "model_name": "model_a",
                "task_type": "binary",
                "seed": 0,
                "fold_index": 0,
                "variant": "default",
                "skipped_reason": None,
                "log_loss": 0.50,
                "hpo_search_space_size": 5,
                "hpo_tier": "smoke",
            },
            {
                "dataset_name": "ds_one",
                "model_name": "model_a",
                "task_type": "binary",
                "seed": 0,
                "fold_index": 0,
                "variant": "tuned",
                "skipped_reason": None,
                "log_loss": 0.30,
                "hpo_search_space_size": 5,
                "hpo_tier": "smoke",
                "hpo_n_trials_completed": 10,
                "hpo_time_to_best_seconds": 5.0,
            },
        ]
    )


def _make_hpo_uplift_rollup() -> list[HPOUpliftRollupRow]:
    return [
        HPOUpliftRollupRow(
            dataset_name="ds_one",
            model_name="model_a",
            task_type="binary",
            primary_metric="delta",
            primary_loss_column="log_loss",
            n_seeds=2,
            n_folds=1,
            n_cells_paired=1,
            n_skipped_cells=1,
            primary_metric_mean=0.20,
            primary_metric_ci_lo=0.15,
            primary_metric_ci_hi=0.25,
            bootstrap_seed=42,
            bootstrap_n_resamples=10_000,
            bootstrap_numpy_version="2.3.0",
            bootstrap_skipped_reason=None,
            manifest_fingerprint="c" * 64,
        ),
    ]


def test_render_hpo_uplift_byte_identity_post_rename() -> None:
    """B17 R-B17-2 pin for the B15 HPO-uplift CI renderer."""
    manifest = _make_hpo_uplift_manifest()
    rollup = _make_hpo_uplift_rollup()
    md = render_hpo_uplift_markdown_with_ci(manifest, rollup)
    assert _CI_CELL_RE.search(md), (
        f"hpo-uplift renderer CI cell shape missing; output starts: {md[:400]!r}"
    )
    assert "(no CI)" not in md
    # B21 / D-B21.1 deferral: new audit fields must NOT surface in the markdown.
    assert "bootstrap_ci_method" not in md
    assert "bootstrap_ci_fallback_reason" not in md
    # B22 / D-B22.1 deferral: per_fold_cis is parquet-only.
    assert "per_fold_cis" not in md


def _make_ensemble_lift_result() -> EnsembleLiftExperimentResult:
    return EnsembleLiftExperimentResult(
        run_id="b17-pin",
        seq_family="seq_sklearn",
        baseline_family="gbm",
        rows=(
            PerDatasetLift(
                dataset_name="ds_one",
                task_type="binary",
                primary_loss_column="log_loss",
                n_cells_paired=1,
                loss_gbm_only_mean=0.50,
                loss_gbm_plus_seq_mean=0.30,
                delta_loss_mean=0.20,
                delta_loss_std=0.01,
                oracle_loss_mean=0.25,
                oracle_delta_loss_mean=0.25,
            ),
        ),
        wilcoxon=WilcoxonResult(
            statistic=1.0,
            p_value=0.10,
            holm_adjusted_p_value=0.10,
            n_datasets=1,
            family_size=1,
        ),
    )


def _make_ensemble_lift_rollup() -> list[EnsembleLiftRollupRow]:
    return [
        EnsembleLiftRollupRow(
            dataset_name="ds_one",
            task_type="binary",
            primary_metric="delta_loss",
            primary_loss_column="log_loss",
            n_seeds=2,
            n_folds=1,
            n_cells_paired=1,
            # B19 / D-B16.7 closure: set n_pair_grid > n_cells_paired so
            # the partial=True flag fires and the byte-pin's mandatory
            # trailing-asterisk regex (B17 R2 mutation-sensitivity
            # tightening) keeps matching.
            n_pair_grid=2,
            # B20 / D-B16.1 closure: matching n_cells_paired=1 so the
            # main Δloss CI cell asterisk continues to fire under
            # n_cells_paired < n_pair_grid (B19 R2 contract). The oracle
            # CI cell on the same row is an independent column; the
            # B17 byte-pin regex only matches the main Δloss column.
            n_oracle_cells_paired=2,
            n_skipped_cells=1,
            primary_metric_mean=0.20,
            primary_metric_ci_lo=0.15,
            primary_metric_ci_hi=0.25,
            oracle_metric_mean=0.10,
            oracle_metric_ci_lo=0.08,
            oracle_metric_ci_hi=0.12,
            bootstrap_seed=42,
            bootstrap_n_resamples=10_000,
            bootstrap_numpy_version="2.3.0",
            bootstrap_skipped_reason=None,
            manifest_fingerprint="d" * 64,
        ),
    ]


def test_render_ensemble_lift_byte_identity_post_rename() -> None:
    """B17 R-B17-2 pin for the B16 ensemble-lift CI renderer."""
    result = _make_ensemble_lift_result()
    rollup = _make_ensemble_lift_rollup()
    md = render_ensemble_lift_markdown_with_ci(result, rollup)
    assert _CI_CELL_RE.search(md), (
        f"ensemble-lift renderer CI cell shape missing; output starts: {md[:400]!r}"
    )
    assert "(no CI)" not in md
    # B21 / D-B21.1 deferral: new audit fields must NOT surface in the markdown.
    assert "bootstrap_ci_method" not in md
    assert "bootstrap_ci_fallback_reason" not in md
    # B22 / D-B22.1 deferral: per_fold_cis is parquet-only.
    assert "per_fold_cis" not in md
    assert "bootstrap_oracle_ci_fallback_reason" not in md


# --- Guard B: source-tree grep invariant -----------------------------------


_FORBIDDEN_NAMES = (
    "primary_loss_mean",
    "primary_loss_ci_lo",
    "primary_loss_ci_hi",
)

# Whitelisted as a whole file because the unrelated
# `LeaderboardEntry` schema (B5 std-variant leaderboard row) and
# its sort keys, formatters, and docstrings reference
# `primary_loss_mean`/`primary_loss_std`. B5 rollup-row
# regressions inside this file are caught by Guard A on the 5
# `*RollupRow` classes plus pyright on `extra="forbid"`
# attribute reads.
_WHITELISTED_FILES = {
    "benchmarks/report/raw_loss.py",
    # test_raw_loss_experiment.py:1245 asserts on
    # `LeaderboardEntry.primary_loss_mean` post-aggregation; this
    # is the std-variant leaderboard surface, not the rollup row.
    "tests/benchmarks/test_raw_loss_experiment.py",
    # This very test file mentions the forbidden names inside
    # whitelist constants and docstrings; whitelist it so the
    # guard does not flag itself.
    "tests/benchmarks/test_b17_byte_identity_pins.py",
}


def _repo_root() -> Path:
    """Repository root: parent of `tests/`."""
    return Path(__file__).resolve().parents[2]


def test_no_stray_primary_loss_references_in_source_tree() -> None:
    """B17 R-B17-5 Guard B: zero non-whitelisted references to
    `primary_loss_mean`, `primary_loss_ci_lo`, or
    `primary_loss_ci_hi` anywhere under `benchmarks/` or
    `tests/`. The whitelist covers `benchmarks/report/raw_loss.py`
    (`LeaderboardEntry` lives there) and
    `tests/benchmarks/test_raw_loss_experiment.py` (asserts on
    `LeaderboardEntry.primary_loss_mean`).

    A future sed sweep that misses a `model_dump()` dict-key
    assertion (which pyright cannot catch) fails this test."""
    root = _repo_root()
    hits: dict[str, list[tuple[int, str]]] = {}
    for src_dir in ("benchmarks", "tests"):
        for path in (root / src_dir).rglob("*.py"):
            rel = str(path.relative_to(root))
            if rel in _WHITELISTED_FILES:
                continue
            text = path.read_text(encoding="utf-8")
            for lineno, line in enumerate(text.splitlines(), start=1):
                for name in _FORBIDDEN_NAMES:
                    if name in line:
                        hits.setdefault(rel, []).append((lineno, line))
                        break
    assert hits == {}, (
        "Guard B (B17 R-B17-5) found stray primary_loss_* references "
        f"outside the whitelist: {hits}; rename to primary_metric_* "
        "per D-B16.5 or extend the whitelist if the reference is on "
        "LeaderboardEntry or its tests."
    )


@pytest.mark.parametrize("whitelisted", sorted(_WHITELISTED_FILES))
def test_guard_b_whitelist_files_exist(whitelisted: str) -> None:
    """B17 R-B17-5 Guard B sanity: the whitelist entries must
    correspond to existing files in the repo. A future repo
    reorganization that deletes one of the whitelisted files
    would leave a stale whitelist; this test fires immediately
    rather than silently masking a real stray reference."""
    path = _repo_root() / whitelisted
    assert path.exists(), (
        f"Guard B whitelist entry {whitelisted!r} does not exist; "
        "prune the whitelist or fix the path."
    )
