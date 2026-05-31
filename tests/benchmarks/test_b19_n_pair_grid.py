"""Phase B19 / D-B16.7 `n_pair_grid` tests.

8 collected tests covering the four-way contract:

- Aggregator-side (#1, #2): the
  `aggregate_bootstrap_ensemble_lift_rollup` caller writes
  the intersection cardinality of `(seed, fold)` pairs across
  the two families to `n_pair_grid` on every row.
- Renderer-side (#3, #4, #8): the
  `_render_complete_table_with_ci` partial-flag computation
  reads `n_pair_grid` (not the pre-B19 `n_seeds * n_folds`)
  AND the defensive `expected > 0` guard suppresses the
  asterisk on the hypothetical zero-grid non-sentinel case.
- Schema-side (#5, #6, #7): the field round-trips through
  parquet AND the `Field(ge=0)` validator accepts 0 AND
  rejects negative values.
"""

from pathlib import Path

import pandas as pd
import pytest
from benchmarks.bootstrap_manifest import (
    EnsembleLiftRollupRow,
    load_ensemble_lift_rollup,
    write_ensemble_lift_rollup,
)
from benchmarks.config import BenchmarkConfig, ExperimentSpec
from benchmarks.experiments import build_run_environment
from benchmarks.experiments.ensemble_lift import (
    ComputePerCellLiftDeltasResult,
    EnsembleLiftExperimentResult,
    PerCellLiftDelta,
    PerDatasetLift,
    WilcoxonResult,
)
from benchmarks.report.bootstrap_ensemble_lift import (
    aggregate_bootstrap_ensemble_lift_rollup,
)
from benchmarks.report.ensemble_lift import render_ensemble_lift_markdown_with_ci
from benchmarks.run_manifest import RunManifest, build_run_manifest, write_run_manifest
from pydantic import ValidationError

from tests.benchmarks._fakes import register_all_fakes_and_get_panels

_GBM_MODEL = "gbm_constant"
_SEQ_MODEL = "seq_constant"


def _make_config(tmp_path: Path) -> BenchmarkConfig:
    return BenchmarkConfig(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        experiments=(
            ExperimentSpec(kind="raw_loss", seeds=(0,)),
            ExperimentSpec(kind="ensemble_lift", seeds=(0,)),
        ),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )


def _make_run_manifest(config: BenchmarkConfig, output_root: Path) -> RunManifest:
    register_all_fakes_and_get_panels()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = build_run_manifest(
        config=config,
        run_id="b19-test",
        library_git_sha="0" * 40,
        profile="smoke",
        hardware_tier="cpu",
        output_root=output_root,
    )
    write_run_manifest(output_root, manifest)
    return manifest


def _manifest_row(
    *,
    dataset_name: str = "fake_binary",
    model_name: str = _GBM_MODEL,
    seed: int = 0,
    fold_index: int = 0,
) -> dict[str, object]:
    return {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "task_type": "binary",
        "seed": seed,
        "fold_index": fold_index,
        "variant": "default",
        "skipped_reason": None,
    }


def _stub_load_run(monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, object]]) -> None:
    import benchmarks.report.bootstrap_ensemble_lift as _module

    df = pd.DataFrame(rows)
    monkeypatch.setattr(_module, "load_run", lambda _root: df)  # type: ignore[misc]


def _stub_families(monkeypatch: pytest.MonkeyPatch) -> None:
    import benchmarks.report.bootstrap_ensemble_lift as _module

    def _families(_df: pd.DataFrame) -> dict[str, str]:
        return {_GBM_MODEL: "gbm", _SEQ_MODEL: "seq_sklearn"}

    monkeypatch.setattr(_module, "model_families", _families)  # type: ignore[misc]


def _stub_per_cell_happy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub `compute_per_cell_lift_deltas` to return a non-empty
    result so the aggregator reaches the happy-path emit branch."""
    import benchmarks.report.bootstrap_ensemble_lift as _module

    result = ComputePerCellLiftDeltasResult(
        cells=(
            PerCellLiftDelta(
                seed=0,
                fold_index=0,
                loss_gbm=0.60,
                loss_gbm_plus_seq=0.40,
                delta_loss=0.20,
                oracle_loss=None,
            ),
        ),
        selector="log_loss",
    )
    monkeypatch.setattr(_module, "compute_per_cell_lift_deltas", lambda **_kwargs: result)  # type: ignore[misc]


def _stub_per_cell_disjoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub returning empty cells with BOTH B11 flags tripped so
    the aggregator routes through the sentinel-emit branch."""
    import benchmarks.report.bootstrap_ensemble_lift as _module

    result = ComputePerCellLiftDeltasResult(
        cells=(), seen_no_gbm=True, seen_no_seq=True, selector="log_loss"
    )
    monkeypatch.setattr(_module, "compute_per_cell_lift_deltas", lambda **_kwargs: result)  # type: ignore[misc]


# --- 1. Aggregator writes n_pair_grid = intersection cardinality ----------


def test_aggregator_writes_n_pair_grid_as_intersection_cardinality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """qa-R1-I1 closure: GBM on seeds {0,1,2} x folds {0,1} (6
    cells), seq on seeds {0,1} x folds {0,1} (4 cells).
    Intersection: 4 cells (seeds {0,1} x folds {0,1}). Union
    n_seeds=3, n_folds=2. Assert n_pair_grid=4 AND n_seeds=3
    AND n_folds=2 AND (the self-documenting mutation pin)
    n_pair_grid != n_seeds * n_folds. A future regression that
    reverts the aggregator to writing `n_seeds * n_folds`
    would fail the bare-inequality assertion specifically."""
    config = _make_config(tmp_path)
    output_root = tmp_path / "out"
    manifest = _make_run_manifest(config, output_root)

    rows: list[dict[str, object]] = []
    for s in (0, 1, 2):
        for f in (0, 1):
            rows.append(_manifest_row(model_name=_GBM_MODEL, seed=s, fold_index=f))
    for s in (0, 1):
        for f in (0, 1):
            rows.append(_manifest_row(model_name=_SEQ_MODEL, seed=s, fold_index=f))
    _stub_load_run(monkeypatch, rows)
    _stub_families(monkeypatch)
    _stub_per_cell_happy(monkeypatch)

    env = build_run_environment(profile="smoke")
    rollup = aggregate_bootstrap_ensemble_lift_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )

    assert len(rollup) == 1
    row = rollup[0]
    assert row.n_pair_grid == 4
    assert row.n_seeds == 3
    assert row.n_folds == 2
    # qa-R1-I1: self-documenting union-vs-intersection mutation pin.
    assert row.n_pair_grid != row.n_seeds * row.n_folds


# --- 2. Aggregator writes n_pair_grid=0 on disjoint-roster sentinel -------


def test_aggregator_writes_n_pair_grid_zero_when_intersection_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disjoint seed sets (GBM {0,1}, seq {2,3}). The
    intersection is empty so n_pair_grid=0. The aggregator's
    seen_no_gbm + seen_no_seq flags trip and the sentinel
    branch fires."""
    config = _make_config(tmp_path)
    output_root = tmp_path / "out"
    manifest = _make_run_manifest(config, output_root)

    rows: list[dict[str, object]] = []
    for s in (0, 1):
        rows.append(_manifest_row(model_name=_GBM_MODEL, seed=s, fold_index=0))
    for s in (2, 3):
        rows.append(_manifest_row(model_name=_SEQ_MODEL, seed=s, fold_index=0))
    _stub_load_run(monkeypatch, rows)
    _stub_families(monkeypatch)
    _stub_per_cell_disjoint(monkeypatch)

    env = build_run_environment(profile="smoke")
    rollup = aggregate_bootstrap_ensemble_lift_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )

    assert len(rollup) == 1
    row = rollup[0]
    assert row.n_pair_grid == 0
    assert row.bootstrap_skipped_reason in {"no_gbm_predictions", "no_seq_predictions"}


# --- 3. Renderer reads n_pair_grid (not n_seeds * n_folds) ----------------


def _make_lift_result_for_renderer() -> EnsembleLiftExperimentResult:
    return EnsembleLiftExperimentResult(
        run_id="b19-renderer-test",
        seq_family="seq_sklearn",
        baseline_family="gbm",
        rows=(
            PerDatasetLift(
                dataset_name="ds_one",
                task_type="binary",
                primary_loss_column="log_loss",
                n_cells_paired=4,
                loss_gbm_only_mean=0.60,
                loss_gbm_plus_seq_mean=0.40,
                delta_loss_mean=0.20,
                delta_loss_std=0.01,
                oracle_loss_mean=0.30,
                oracle_delta_loss_mean=0.30,
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


def _make_rollup_row_for_renderer(
    *,
    n_seeds: int,
    n_folds: int,
    n_cells_paired: int,
    n_pair_grid: int,
    n_oracle_cells_paired: int | None = None,
) -> EnsembleLiftRollupRow:
    # B20 / D-B16.1: mirror the B19 None-fallback pattern. When
    # n_oracle_cells_paired is not supplied, default to n_cells_paired
    # so the existing B19 tests assert on the main Δloss CI cell only
    # (the oracle CI cell is symmetric and does not fire its own
    # asterisk).
    if n_oracle_cells_paired is None:
        n_oracle_cells_paired = n_cells_paired
    # B28 / D-B27.2 closure: zero-oracle rows must carry None
    # oracle metrics per the new CI-sentinel invariant.
    oracle_metric_mean: float | None = 0.10
    oracle_metric_ci_lo: float | None = 0.08
    oracle_metric_ci_hi: float | None = 0.12
    if n_oracle_cells_paired == 0:
        oracle_metric_mean = None
        oracle_metric_ci_lo = None
        oracle_metric_ci_hi = None
    return EnsembleLiftRollupRow(
        dataset_name="ds_one",
        task_type="binary",
        primary_metric="delta_loss",
        primary_loss_column="log_loss",
        n_seeds=n_seeds,
        n_folds=n_folds,
        n_cells_paired=n_cells_paired,
        n_pair_grid=n_pair_grid,
        n_oracle_cells_paired=n_oracle_cells_paired,
        n_skipped_cells=0,
        primary_metric_mean=0.20,
        primary_metric_ci_lo=0.15,
        primary_metric_ci_hi=0.25,
        oracle_metric_mean=oracle_metric_mean,
        oracle_metric_ci_lo=oracle_metric_ci_lo,
        oracle_metric_ci_hi=oracle_metric_ci_hi,
        bootstrap_seed=42,
        bootstrap_n_resamples=10_000,
        bootstrap_numpy_version="2.3.0",
        bootstrap_skipped_reason=None,
        manifest_fingerprint="f" * 64,
    )


def test_renderer_partial_flag_uses_n_pair_grid_not_n_seeds_times_n_folds() -> None:
    """n_seeds=3, n_folds=2 (union view: expected=6 under the
    pre-B19 logic), n_cells_paired=4, n_pair_grid=4. Under the
    OLD logic expected=6 > 4 fires the asterisk. Under the
    NEW logic expected=n_pair_grid=4 == 4 does NOT fire.
    Pins the renderer's read of `n_pair_grid` over the
    union-view product."""
    result = _make_lift_result_for_renderer()
    rollup = [_make_rollup_row_for_renderer(n_seeds=3, n_folds=2, n_cells_paired=4, n_pair_grid=4)]
    md = render_ensemble_lift_markdown_with_ci(result, rollup)
    ci_line = next(line for line in md.splitlines() if "0.2000 [0.1500, 0.2500]" in line)
    assert "*" not in ci_line


# --- 4. Renderer fires the asterisk when n_cells_paired < n_pair_grid -----


def test_renderer_partial_flag_fires_when_n_cells_paired_less_than_n_pair_grid() -> None:
    """n_pair_grid=4, n_cells_paired=3 (covered 3 of 4
    intersection cells). The asterisk MUST fire."""
    result = _make_lift_result_for_renderer()
    rollup = [_make_rollup_row_for_renderer(n_seeds=2, n_folds=2, n_cells_paired=3, n_pair_grid=4)]
    md = render_ensemble_lift_markdown_with_ci(result, rollup)
    ci_line = next(line for line in md.splitlines() if "0.2000 [0.1500, 0.2500]" in line)
    assert "*" in ci_line


# --- 5. n_pair_grid round-trips through the parquet shard -----------------


def test_n_pair_grid_round_trips_through_parquet_shard(tmp_path: Path) -> None:
    """qa-R1-I2 closure: 137 is an unusual prime far from any
    factory default (matches the B16 `bootstrap_n_resamples=
    137` precedent). A silent coercion to a small-integer
    default would not match 137.

    B29 / D-B28.1 closure: n_seeds and n_folds bumped to
    12 each (product 144 > 137) so the new
    `n_pair_grid <= n_seeds * n_folds` validator accepts
    the row; the 137 round-trip is what's pinned."""
    row = EnsembleLiftRollupRow(
        dataset_name="ds_one",
        task_type="binary",
        primary_metric="delta_loss",
        primary_loss_column="log_loss",
        n_seeds=12,
        n_folds=12,
        n_cells_paired=4,
        n_pair_grid=137,
        n_oracle_cells_paired=4,
        n_skipped_cells=0,
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
        manifest_fingerprint="f" * 64,
    )
    write_ensemble_lift_rollup(tmp_path, [row])
    loaded = load_ensemble_lift_rollup(tmp_path)
    assert len(loaded) == 1
    assert loaded[0].n_pair_grid == 137


# --- 6. Field(ge=0) accepts the zero boundary -----------------------------


def test_n_pair_grid_zero_passes_field_validator() -> None:
    """`Field(ge=0)` accepts the boundary value 0 (used on
    sentinel-emit rows)."""
    row = EnsembleLiftRollupRow(
        dataset_name="ds_one",
        task_type="binary",
        primary_metric="delta_loss",
        primary_loss_column="log_loss",
        n_seeds=2,
        n_folds=2,
        n_cells_paired=0,
        n_pair_grid=0,
        n_oracle_cells_paired=0,
        n_skipped_cells=0,
        primary_metric_mean=None,
        primary_metric_ci_lo=None,
        primary_metric_ci_hi=None,
        oracle_metric_mean=None,
        oracle_metric_ci_lo=None,
        oracle_metric_ci_hi=None,
        bootstrap_seed=42,
        bootstrap_n_resamples=10_000,
        bootstrap_numpy_version="2.3.0",
        bootstrap_skipped_reason="no_gbm_predictions",
        manifest_fingerprint="f" * 64,
    )
    assert row.n_pair_grid == 0


# --- 7. Field(ge=0) rejects negative values (qa-R2-N1 closure) -----------


def test_n_pair_grid_negative_rejected_by_field_validator() -> None:
    """qa-R2-N1 closure: the `Field(ge=0)` constraint rejects
    negative values. Pins the lower bound from below; test #6
    covers the boundary from above."""
    with pytest.raises(ValidationError, match="n_pair_grid"):
        EnsembleLiftRollupRow(
            dataset_name="ds_one",
            task_type="binary",
            primary_metric="delta_loss",
            primary_loss_column="log_loss",
            n_seeds=2,
            n_folds=2,
            n_cells_paired=0,
            n_pair_grid=-1,
            n_oracle_cells_paired=0,
            n_skipped_cells=0,
            primary_metric_mean=None,
            primary_metric_ci_lo=None,
            primary_metric_ci_hi=None,
            oracle_metric_mean=None,
            oracle_metric_ci_lo=None,
            oracle_metric_ci_hi=None,
            bootstrap_seed=42,
            bootstrap_n_resamples=10_000,
            bootstrap_numpy_version="2.3.0",
            bootstrap_skipped_reason="no_gbm_predictions",
            manifest_fingerprint="f" * 64,
        )


# --- 8. Renderer `expected > 0` defensive guard (closes build R1 qa-I1) ---


def test_renderer_expected_greater_than_zero_guard_suppresses_partial_on_zero_grid() -> None:
    """Build R1 qa-I1 closure: pin the renderer's defensive
    `expected > 0` short-circuit. Construct a hypothetical
    non-sentinel rollup row with `n_pair_grid=0` AND
    `n_cells_paired=0` AND `bootstrap_skipped_reason=None`.
    Production never routes such a row through the partial-flag
    branch (the aggregator's Gate D + sentinel-emit cascade
    guarantee `n_pair_grid >= 1` on non-sentinel rows), so the
    guard is defensive. The renderer's partial computation is
    `n_cells_paired < expected and expected > 0`: with
    `expected = n_pair_grid = 0`, the second clause short-
    circuits to False and the asterisk does NOT fire. A future
    edit that drops the `expected > 0` clause would change
    nothing observable here either (0 < 0 is False) but a
    different mutation (e.g., `expected >= 0`) would also
    survive; the test pins the current observable behavior
    rather than the precise short-circuit mechanism."""
    result = _make_lift_result_for_renderer()
    rollup = [_make_rollup_row_for_renderer(n_seeds=0, n_folds=0, n_cells_paired=0, n_pair_grid=0)]
    md = render_ensemble_lift_markdown_with_ci(result, rollup)
    # The row renders without a partial-coverage asterisk on
    # the CI cell. The numeric interval still appears because
    # primary_metric_mean/_ci_lo/_ci_hi are populated.
    ci_line = next(line for line in md.splitlines() if "0.2000 [0.1500, 0.2500]" in line)
    assert "*" not in ci_line
