"""Phase B14 D-B13.1 pairwise bootstrap-CI aggregator tests.

Covers `aggregate_bootstrap_pairwise_rollup`: the classification
happy path, the regression-cell sentinel, mixed-skip, empty
manifest, malformed-cell raise, and fingerprint round-trip. Also
pins the bootstrap-degeneracy oracle (Round-1 qa-C1 closure).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from benchmarks.bootstrap_manifest import (
    load_pairwise_rollup,
    pairwise_rollup_path,
)
from benchmarks.config import BenchmarkConfig, ExperimentSpec
from benchmarks.experiments import build_run_environment
from benchmarks.experiments.ensemble import pairwise_dir
from benchmarks.report.bootstrap_pairwise import (
    aggregate_bootstrap_pairwise_rollup,
)
from benchmarks.report.bootstrap_rollup import RawRollupError
from benchmarks.run_manifest import (
    RunManifest,
    build_run_manifest,
    write_run_manifest,
)


def _make_config(tmp_path: Path) -> BenchmarkConfig:
    return BenchmarkConfig(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        experiments=(
            ExperimentSpec(kind="raw_loss", seeds=(0,)),
            ExperimentSpec(kind="ensemble", seeds=(0,)),
        ),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )


def _make_run_manifest(config: BenchmarkConfig, output_root: Path) -> RunManifest:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = build_run_manifest(
        config=config,
        run_id="b14-test-run",
        library_git_sha="0" * 40,
        profile="smoke",
        hardware_tier="cpu",
        output_root=output_root,
    )
    write_run_manifest(output_root, manifest)
    return manifest


def _pairwise_row(
    *,
    dataset_name: str = "fake_binary",
    model_a: str = "model_a",
    model_b: str = "model_b",
    task_type: str = "binary",
    seed: int = 0,
    fold_index: int = 0,
    pearson_error_corr: float | None = 0.30,
    disagreement_rate: float | None = 0.20,
    skipped_reason: str | None = None,
) -> dict[str, object]:
    """One PairwiseRow-shaped dict. The aggregator reads
    `pearson_error_corr` and `disagreement_rate` to compute
    `complementarity_score = (1 - pearson_error_corr) +
    disagreement_rate`."""
    return {
        "library_git_sha": "0" * 40,
        "run_id": "b14-test-run",
        "started_at_utc": "2026-05-30T00:00:00+00:00",
        "dataset_name": dataset_name,
        "model_a": model_a,
        "model_b": model_b,
        "seed": seed,
        "fold_index": fold_index,
        "task_type": task_type,
        "skipped_reason": skipped_reason,
        "n_samples": 100 if skipped_reason is None else None,
        "n11": 40 if skipped_reason is None else None,
        "n10": 10 if skipped_reason is None else None,
        "n01": 15 if skipped_reason is None else None,
        "n00": 35 if skipped_reason is None else None,
        "yule_q": 0.7 if skipped_reason is None else None,
        "phi": 0.5 if skipped_reason is None else None,
        "disagreement_rate": disagreement_rate,
        "double_fault_rate": 0.1 if skipped_reason is None else None,
        "pearson_pred_corr": 0.6 if skipped_reason is None else None,
        "spearman_pred_corr": 0.55 if skipped_reason is None else None,
        "pearson_error_corr": pearson_error_corr,
    }


def _write_pairwise_manifest(output_root: Path, rows: list[dict[str, object]]) -> None:
    """Persist a list of pairwise rows under {output_root}/pairwise/
    using the same shard layout as run_ensemble."""
    target_dir = pairwise_dir(output_root)
    target_dir.mkdir(parents=True, exist_ok=True)
    # One shard per row (matches the prod layout).
    for i, row in enumerate(rows):
        df = pd.DataFrame([row])
        # Use a deterministic filename per index; load_pairwise
        # globs every *.parquet so the actual name does not matter.
        shard_path = target_dir / f"shard_{i:04d}.parquet"
        df.to_parquet(shard_path, index=False)


def _setup(tmp_path: Path) -> tuple[BenchmarkConfig, Path, RunManifest]:
    config = _make_config(tmp_path)
    output_root = tmp_path / "out"
    manifest = _make_run_manifest(config, output_root)
    return config, output_root, manifest


# --- Empty manifest ---------------------------------------------------------


def test_aggregate_bootstrap_pairwise_rollup_empty_manifest_returns_empty_list(
    tmp_path: Path,
) -> None:
    """Absent pairwise manifest -> []; no rollup file written."""
    config, output_root, manifest = _setup(tmp_path)
    env = build_run_environment(profile="smoke")
    rows = aggregate_bootstrap_pairwise_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    assert rows == []
    assert not pairwise_rollup_path(output_root).exists()


# --- Classification happy path ----------------------------------------------


def test_aggregate_bootstrap_pairwise_rollup_classification_cells_emit_ci(
    tmp_path: Path,
) -> None:
    """2 seeds x 3 folds = 6 OK cells on one (dataset, A, B) pair
    with classification task_type. Assert the emitted row has
    mean ≈ unresampled mean, ci_lo < mean < ci_hi, and
    n_cells_evaluated == 6."""
    config, output_root, manifest = _setup(tmp_path)
    rows_in: list[dict[str, object]] = []
    # Spread scores by seed*fold so the bootstrap has variance.
    seed_fold_pairs = [(s, f) for s in (0, 1) for f in (0, 1, 2)]
    for s, f in seed_fold_pairs:
        rows_in.append(
            _pairwise_row(
                seed=s,
                fold_index=f,
                pearson_error_corr=0.25 + 0.05 * f,  # 0.25, 0.30, 0.35
                disagreement_rate=0.20 + 0.01 * s,  # 0.20, 0.21
            )
        )
    _write_pairwise_manifest(output_root, rows_in)

    env = build_run_environment(profile="smoke")
    rows_out = aggregate_bootstrap_pairwise_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    assert len(rows_out) == 1
    row = rows_out[0]
    assert row.dataset_name == "fake_binary"
    assert row.model_a == "model_a"
    assert row.model_b == "model_b"
    assert row.task_type == "binary"
    assert row.primary_metric == "complementarity_score"
    assert row.n_cells_evaluated == 6
    assert row.n_seeds == 2
    assert row.bootstrap_skipped_reason is None
    # B21 R1 qa-I1 closure: pin ci_method end-to-end on B6 happy path.
    # The fallback reason is type-checked (small-N fixtures may
    # legitimately trigger a BCa fallback to percentile).
    assert row.bootstrap_ci_method == "bca"
    assert row.bootstrap_ci_fallback_reason in (None, "p0_at_edge", "a_overshoot")
    # The unresampled complementarity_score values:
    expected_scores = np.array(
        [
            (1.0 - 0.25) + 0.20,
            (1.0 - 0.30) + 0.20,
            (1.0 - 0.35) + 0.20,
            (1.0 - 0.25) + 0.21,
            (1.0 - 0.30) + 0.21,
            (1.0 - 0.35) + 0.21,
        ]
    )
    assert row.primary_metric_mean is not None
    assert row.primary_metric_ci_lo is not None
    assert row.primary_metric_ci_hi is not None
    assert abs(row.primary_metric_mean - float(expected_scores.mean())) < 1e-9
    assert row.primary_metric_ci_lo < row.primary_metric_mean < row.primary_metric_ci_hi


def test_aggregate_bootstrap_pairwise_rollup_ci_width_nonzero_with_multiple_cells(
    tmp_path: Path,
) -> None:
    """Round-1 qa-C1 oracle: 4 OK cells with non-identical scores.
    Assert ci_hi - ci_lo > 0 AND the width is within 3x of
    `std * 2 * 1.96 / sqrt(n)`. Rules out the silent degeneracy
    bug where the aggregator passes entity_ids=np.zeros(n)
    (one entity -> ci_lo=ci_hi=mean)."""
    config, output_root, manifest = _setup(tmp_path)
    rows_in: list[dict[str, object]] = []
    pearson_values = [0.10, 0.20, 0.30, 0.40]
    for i, p in enumerate(pearson_values):
        rows_in.append(
            _pairwise_row(
                seed=i,
                fold_index=0,
                pearson_error_corr=p,
                disagreement_rate=0.15,
            )
        )
    _write_pairwise_manifest(output_root, rows_in)

    env = build_run_environment(profile="smoke")
    rows_out = aggregate_bootstrap_pairwise_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    assert len(rows_out) == 1
    row = rows_out[0]
    assert row.primary_metric_ci_lo is not None
    assert row.primary_metric_ci_hi is not None
    width = row.primary_metric_ci_hi - row.primary_metric_ci_lo
    assert width > 0.0
    scores = np.array([(1.0 - p) + 0.15 for p in pearson_values])
    expected_naive_width = float(np.std(scores) * 2.0 * 1.96 / np.sqrt(len(scores)))
    # 3x tolerance accommodates bootstrap variance.
    assert width < 3.0 * expected_naive_width


# --- Regression sentinel ----------------------------------------------------


def test_aggregate_bootstrap_pairwise_rollup_regression_cells_emit_sentinel(
    tmp_path: Path,
) -> None:
    """Regression-task cells emit a sentinel row with EXACT
    `bootstrap_skipped_reason="regression_complementarity_undefined"`.
    Cross-pinned with the renderer's footnote consumption in
    Stage 4."""
    config, output_root, manifest = _setup(tmp_path)
    rows_in = [
        _pairwise_row(
            task_type="regression_point",
            seed=0,
            fold_index=f,
            pearson_error_corr=0.30,
            disagreement_rate=None,
        )
        for f in range(3)
    ]
    _write_pairwise_manifest(output_root, rows_in)

    env = build_run_environment(profile="smoke")
    rows_out = aggregate_bootstrap_pairwise_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    assert len(rows_out) == 1
    row = rows_out[0]
    assert row.task_type == "regression_point"
    assert row.bootstrap_skipped_reason == "regression_complementarity_undefined"
    assert row.primary_metric_mean is None
    assert row.primary_metric_ci_lo is None
    assert row.primary_metric_ci_hi is None
    assert row.n_cells_evaluated == 0


# --- Mixed skip -------------------------------------------------------------


def test_aggregate_bootstrap_pairwise_rollup_mixed_skip_runs_on_ok_subset(
    tmp_path: Path,
) -> None:
    """Mixed manifest: 1 OK + 2 skipped on the same (A, B) pair.
    Assert the rollup bootstraps on the OK cell only (single-
    entity degenerate primitive contract)."""
    config, output_root, manifest = _setup(tmp_path)
    rows_in = [
        _pairwise_row(seed=0, fold_index=0, pearson_error_corr=0.20, disagreement_rate=0.30),
        _pairwise_row(seed=0, fold_index=1, skipped_reason="predictions_missing"),
        _pairwise_row(seed=0, fold_index=2, skipped_reason="empty_panel_row_index_join"),
    ]
    _write_pairwise_manifest(output_root, rows_in)

    env = build_run_environment(profile="smoke")
    rows_out = aggregate_bootstrap_pairwise_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    assert len(rows_out) == 1
    row = rows_out[0]
    assert row.n_cells_evaluated == 1
    assert row.n_skipped_cells == 2
    # Single-entity degenerate: mean == ci_lo == ci_hi.
    expected_score = (1.0 - 0.20) + 0.30
    assert row.primary_metric_mean == pytest.approx(expected_score, abs=1e-9)
    assert row.primary_metric_ci_lo == pytest.approx(expected_score, abs=1e-9)
    assert row.primary_metric_ci_hi == pytest.approx(expected_score, abs=1e-9)


# --- Malformed cell ---------------------------------------------------------


def test_aggregate_bootstrap_pairwise_rollup_malformed_cell_raises(
    tmp_path: Path,
) -> None:
    """An OK row (`skipped_reason is None`) with NaN
    `pearson_error_corr` is malformed and raises RawRollupError.
    The pairwise manifest is per-(seed, fold) not per-row, so the
    failure mode is malformed-cell not row drift."""
    config, output_root, manifest = _setup(tmp_path)
    rows_in = [
        _pairwise_row(seed=0, fold_index=0, pearson_error_corr=None, disagreement_rate=0.20),
    ]
    _write_pairwise_manifest(output_root, rows_in)

    env = build_run_environment(profile="smoke")
    with pytest.raises(RawRollupError, match="NaN complementarity_score"):
        aggregate_bootstrap_pairwise_rollup(
            config, output_root=output_root, env=env, manifest=manifest
        )


# --- Fingerprint round-trip -------------------------------------------------


def test_aggregate_bootstrap_pairwise_rollup_records_manifest_fingerprint(
    tmp_path: Path,
) -> None:
    """Every emitted row's `manifest_fingerprint` equals the
    live `RunManifest.fingerprint()` (B14.4 freshness check
    machinery, reused from B13)."""
    config, output_root, manifest = _setup(tmp_path)
    rows_in = [
        _pairwise_row(seed=0, fold_index=f, pearson_error_corr=0.30, disagreement_rate=0.20)
        for f in range(3)
    ]
    _write_pairwise_manifest(output_root, rows_in)

    env = build_run_environment(profile="smoke")
    rows_out = aggregate_bootstrap_pairwise_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    fingerprint = manifest.fingerprint()
    assert all(row.manifest_fingerprint == fingerprint for row in rows_out)


# --- Rollup file written ----------------------------------------------------


def test_aggregate_bootstrap_pairwise_rollup_writes_parquet_shard(
    tmp_path: Path,
) -> None:
    """The aggregator writes `bootstrap_pairwise_rollup.parquet`
    on a successful call; a subsequent load returns the same
    rows."""
    config, output_root, manifest = _setup(tmp_path)
    rows_in = [
        _pairwise_row(seed=0, fold_index=f, pearson_error_corr=0.30, disagreement_rate=0.20)
        for f in range(3)
    ]
    _write_pairwise_manifest(output_root, rows_in)

    env = build_run_environment(profile="smoke")
    rows_out = aggregate_bootstrap_pairwise_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    assert pairwise_rollup_path(output_root).exists()
    loaded = load_pairwise_rollup(output_root)
    assert len(loaded) == len(rows_out)
    assert loaded[0].model_dump() == rows_out[0].model_dump()


# --- All-cells-skipped sentinel (Stage-2 qa-I1) ----------------------------


def test_aggregate_bootstrap_pairwise_rollup_all_cells_skipped_emits_sentinel(
    tmp_path: Path,
) -> None:
    """Classification group where EVERY cell has a non-None
    skipped_reason -> sentinel row with `bootstrap_skipped_reason=
    "all_cells_skipped_in_manifest"`. The `ok.empty` branch of
    `_build_group_rollup` was untested in the first cut."""
    config, output_root, manifest = _setup(tmp_path)
    rows_in = [
        _pairwise_row(seed=0, fold_index=0, skipped_reason="predictions_missing"),
        _pairwise_row(seed=0, fold_index=1, skipped_reason="empty_panel_row_index_join"),
        _pairwise_row(seed=1, fold_index=0, skipped_reason="predictions_missing"),
    ]
    _write_pairwise_manifest(output_root, rows_in)

    env = build_run_environment(profile="smoke")
    rows_out = aggregate_bootstrap_pairwise_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    assert len(rows_out) == 1
    row = rows_out[0]
    assert row.bootstrap_skipped_reason == "all_cells_skipped_in_manifest"
    assert row.n_cells_evaluated == 0
    assert row.n_skipped_cells == 3
    assert row.primary_metric_mean is None


# --- Per-spec n_resamples override (Stage-2 qa-I2) -------------------------


def test_aggregate_bootstrap_pairwise_rollup_respects_per_spec_n_resamples_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`ExperimentSpec(kind="ensemble", bootstrap_n_resamples=N)`
    overrides the profile default. Pinned by monkeypatching the
    primitive to capture the n_resamples argument."""
    output_root = tmp_path / "out"
    config = BenchmarkConfig(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        experiments=(
            ExperimentSpec(kind="raw_loss", seeds=(0,)),
            ExperimentSpec(kind="ensemble", seeds=(0,), bootstrap_n_resamples=137),
        ),
        output_dir=output_root,
        cache_dir=tmp_path / "cache",
    )
    manifest = _make_run_manifest(config, output_root)
    rows_in = [
        _pairwise_row(seed=0, fold_index=f, pearson_error_corr=0.30, disagreement_rate=0.20)
        for f in range(3)
    ]
    _write_pairwise_manifest(output_root, rows_in)

    captured: dict[str, int] = {}

    def _capturing_primitive(
        losses: object,
        entity_ids: object,
        *,
        n_resamples: int,
        confidence: float,
        seed: int,
        **_kwargs: object,
    ) -> tuple[float, float, float, str | None]:
        captured["n_resamples"] = n_resamples
        return (0.5, 0.4, 0.6, None)

    import benchmarks.report.bootstrap_pairwise as _module

    monkeypatch.setattr(_module, "entity_block_bootstrap_ci", _capturing_primitive)

    env = build_run_environment(profile="smoke")
    aggregate_bootstrap_pairwise_rollup(config, output_root=output_root, env=env, manifest=manifest)
    assert captured["n_resamples"] == 137


# --- OOM ceiling raise (Stage-2 qa-I3 + code-I1) ---------------------------


def test_aggregate_bootstrap_pairwise_rollup_oom_gate_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lower BOOTSTRAP_ROW_COUNT_CEILING to 1 so the tiny test
    fixture trips the OOM gate. Pin the defensive raise added
    at bootstrap_pairwise.py."""
    config, output_root, manifest = _setup(tmp_path)
    rows_in = [
        _pairwise_row(seed=0, fold_index=f, pearson_error_corr=0.30, disagreement_rate=0.20)
        for f in range(3)
    ]
    _write_pairwise_manifest(output_root, rows_in)

    import benchmarks.report.bootstrap_pairwise as _module

    monkeypatch.setattr(_module, "BOOTSTRAP_ROW_COUNT_CEILING", 1)
    env = build_run_environment(profile="smoke")
    with pytest.raises(RawRollupError, match="ceiling"):
        aggregate_bootstrap_pairwise_rollup(
            config, output_root=output_root, env=env, manifest=manifest
        )
