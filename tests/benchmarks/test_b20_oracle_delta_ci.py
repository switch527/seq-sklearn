"""Phase B20 D-B16.1 oracle Δ CI tests.

Covers the 13 named tests from `docs/benchmark_suite_design_b20_delta.md`
section B20.3. Six tests exercise the aggregator's new per-cell oracle Δ
bootstrap path (happy path, partial coverage, all-None fallback, sign
convention vs cross-wire, OOM gate, NaN guard); five tests exercise the
CI-variant renderer's oracle CI column (rendering, partial asterisk
fires + suppresses, sentinel + zero-paired branches); one test pins the
sentinel-emit helper's oracle defaults; one test pins the
`BOOTSTRAP_DEFAULT_SEED ^ BOOTSTRAP_ORACLE_SEED_OFFSET` derived-seed
mechanism (R-B20-2a) by re-running under offset=0 and asserting CI
bounds collapse.
"""

from pathlib import Path

import pandas as pd
import pytest
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
from benchmarks.report.bootstrap_rollup import RawRollupError
from benchmarks.report.ensemble_lift import render_ensemble_lift_markdown_with_ci
from benchmarks.run_manifest import RunManifest, build_run_manifest, write_run_manifest

from tests.benchmarks._fakes import register_all_fakes_and_get_panels

_GBM_MODEL = "gbm_constant"
_SEQ_MODEL = "seq_constant"


# =============================================================================
# Aggregator-test helpers (mirror tests/benchmarks/test_bootstrap_ensemble_lift.py)
# =============================================================================


def _make_config(
    tmp_path: Path,
    *,
    bootstrap_n_resamples: int | None = None,
) -> BenchmarkConfig:
    return BenchmarkConfig(
        datasets=("fake_binary",),
        models=("fake_constant_binary",),
        experiments=(
            ExperimentSpec(kind="raw_loss", seeds=(0,)),
            ExperimentSpec(
                kind="ensemble_lift",
                seeds=(0,),
                bootstrap_n_resamples=bootstrap_n_resamples,
            ),
        ),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )


def _make_run_manifest(config: BenchmarkConfig, output_root: Path) -> RunManifest:
    register_all_fakes_and_get_panels()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = build_run_manifest(
        config=config,
        run_id="b20-test",
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
    task_type: str = "binary",
    seed: int = 0,
    fold_index: int = 0,
    variant: str = "default",
    skipped_reason: str | None = None,
) -> dict[str, object]:
    return {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "task_type": task_type,
        "seed": seed,
        "fold_index": fold_index,
        "variant": variant,
        "skipped_reason": skipped_reason,
    }


def _setup(
    tmp_path: Path,
    *,
    bootstrap_n_resamples: int | None = None,
) -> tuple[BenchmarkConfig, Path, RunManifest]:
    config = _make_config(tmp_path, bootstrap_n_resamples=bootstrap_n_resamples)
    output_root = tmp_path / "out"
    manifest = _make_run_manifest(config, output_root)
    return config, output_root, manifest


def _stub_load_run(monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, object]]) -> None:
    import benchmarks.report.bootstrap_ensemble_lift as _module

    df = pd.DataFrame(rows)
    monkeypatch.setattr(_module, "load_run", lambda _root: df)  # type: ignore[misc]


def _stub_families(
    monkeypatch: pytest.MonkeyPatch,
    mapping: dict[str, str] | None = None,
) -> None:
    import benchmarks.report.bootstrap_ensemble_lift as _module

    families = mapping if mapping is not None else {_GBM_MODEL: "gbm", _SEQ_MODEL: "seq_sklearn"}
    monkeypatch.setattr(_module, "model_families", lambda _df: dict(families))  # type: ignore[misc]


def _stub_per_cell(
    monkeypatch: pytest.MonkeyPatch,
    result_by_dataset: dict[str, ComputePerCellLiftDeltasResult],
) -> None:
    import benchmarks.report.bootstrap_ensemble_lift as _module

    def _fake(*, dataset_name: str, **_kwargs: object) -> ComputePerCellLiftDeltasResult:
        return result_by_dataset.get(
            dataset_name,
            ComputePerCellLiftDeltasResult(cells=(), selector="log_loss"),
        )

    monkeypatch.setattr(_module, "compute_per_cell_lift_deltas", _fake)  # type: ignore[misc]


def _ok_rows_both_families(
    dataset_name: str = "fake_binary",
    n_seeds: int = 2,
    n_folds: int = 2,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for s in range(n_seeds):
        for f in range(n_folds):
            rows.append(
                _manifest_row(
                    dataset_name=dataset_name, model_name=_GBM_MODEL, seed=s, fold_index=f
                )
            )
            rows.append(
                _manifest_row(
                    dataset_name=dataset_name, model_name=_SEQ_MODEL, seed=s, fold_index=f
                )
            )
    return rows


def _cells_with_oracle(
    values: list[tuple[int, int, float, float, float | None]],
) -> ComputePerCellLiftDeltasResult:
    """Build a ComputePerCellLiftDeltasResult from
    (seed, fold, loss_gbm, loss_gbm_plus_seq, oracle_loss) tuples."""
    cells = tuple(
        PerCellLiftDelta(
            seed=s,
            fold_index=f,
            loss_gbm=lg,
            loss_gbm_plus_seq=lgs,
            delta_loss=lg - lgs,
            oracle_loss=ol,
        )
        for s, f, lg, lgs, ol in values
    )
    return ComputePerCellLiftDeltasResult(cells=cells, selector="log_loss")


# =============================================================================
# Renderer-test helpers
# =============================================================================


def _make_per_dataset_lift(*, dataset_name: str = "ds_one") -> PerDatasetLift:
    return PerDatasetLift(
        dataset_name=dataset_name,
        task_type="binary",
        primary_loss_column="log_loss",
        n_cells_paired=4,
        loss_gbm_only_mean=0.60,
        loss_gbm_plus_seq_mean=0.40,
        delta_loss_mean=0.20,
        delta_loss_std=0.05,
        oracle_loss_mean=0.30,
        oracle_delta_loss_mean=0.30,
    )


def _make_lift_result(*, dataset_name: str = "ds_one") -> EnsembleLiftExperimentResult:
    return EnsembleLiftExperimentResult(
        run_id="b20-renderer-test",
        seq_family="seq_sklearn",
        baseline_family="gbm",
        rows=(_make_per_dataset_lift(dataset_name=dataset_name),),
        wilcoxon=WilcoxonResult(
            statistic=1.0,
            p_value=0.10,
            holm_adjusted_p_value=0.10,
            n_datasets=1,
            family_size=1,
        ),
    )


def _make_rollup_row(
    *,
    dataset_name: str = "ds_one",
    n_cells_paired: int = 4,
    n_pair_grid: int = 4,
    n_oracle_cells_paired: int = 4,
    primary_metric_mean: float | None = 0.20,
    primary_metric_ci_lo: float | None = 0.15,
    primary_metric_ci_hi: float | None = 0.25,
    oracle_metric_mean: float | None = 0.10,
    oracle_metric_ci_lo: float | None = 0.08,
    oracle_metric_ci_hi: float | None = 0.12,
    bootstrap_skipped_reason: str | None = None,
):
    from benchmarks.bootstrap_manifest import EnsembleLiftRollupRow

    return EnsembleLiftRollupRow(
        dataset_name=dataset_name,
        task_type="binary",
        primary_metric="delta_loss",
        primary_loss_column="log_loss",
        n_seeds=2,
        n_folds=2,
        n_cells_paired=n_cells_paired,
        n_pair_grid=n_pair_grid,
        n_oracle_cells_paired=n_oracle_cells_paired,
        n_skipped_cells=0,
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
        manifest_fingerprint="f" * 64,
    )


# =============================================================================
# 1. Aggregator happy path: every cell has finite oracle_loss
# =============================================================================


def test_aggregator_writes_oracle_ci_on_happy_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """4 cells with finite `oracle_loss` on all. Assert
    `n_oracle_cells_paired == 4`, `oracle_metric_mean` is finite, AND
    `oracle_metric_ci_lo <= oracle_metric_mean <= oracle_metric_ci_hi`.
    Pins the happy-path B20 schema-write contract."""
    config, output_root, manifest = _setup(tmp_path)
    _stub_load_run(monkeypatch, _ok_rows_both_families())
    _stub_families(monkeypatch)
    _stub_per_cell(
        monkeypatch,
        {
            "fake_binary": _cells_with_oracle(
                [
                    (0, 0, 0.60, 0.40, 0.30),
                    (0, 1, 0.62, 0.42, 0.32),
                    (1, 0, 0.58, 0.38, 0.28),
                    (1, 1, 0.64, 0.44, 0.34),
                ]
            )
        },
    )

    env = build_run_environment(profile="smoke")
    rollup = aggregate_bootstrap_ensemble_lift_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    assert len(rollup) == 1
    row = rollup[0]
    assert row.n_oracle_cells_paired == 4
    assert row.oracle_metric_mean is not None
    assert row.oracle_metric_ci_lo is not None
    assert row.oracle_metric_ci_hi is not None
    assert row.oracle_metric_ci_lo <= row.oracle_metric_mean <= row.oracle_metric_ci_hi


# =============================================================================
# 2. Aggregator drops cells where oracle_loss is None
# =============================================================================


def test_aggregator_skips_cells_with_none_oracle_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """4 cells; 2 have `oracle_loss=None`. Assert
    `n_oracle_cells_paired == 2` (the two None cells were dropped) AND
    the oracle CI fields are populated from the remaining 2 cells."""
    config, output_root, manifest = _setup(tmp_path)
    _stub_load_run(monkeypatch, _ok_rows_both_families())
    _stub_families(monkeypatch)
    _stub_per_cell(
        monkeypatch,
        {
            "fake_binary": _cells_with_oracle(
                [
                    (0, 0, 0.60, 0.40, 0.30),
                    (0, 1, 0.62, 0.42, None),
                    (1, 0, 0.58, 0.38, 0.28),
                    (1, 1, 0.64, 0.44, None),
                ]
            )
        },
    )

    env = build_run_environment(profile="smoke")
    rollup = aggregate_bootstrap_ensemble_lift_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    row = rollup[0]
    assert row.n_oracle_cells_paired == 2
    assert row.n_cells_paired == 4  # main bootstrap unaffected
    assert row.oracle_metric_mean is not None
    assert row.oracle_metric_ci_lo is not None
    assert row.oracle_metric_ci_hi is not None


# =============================================================================
# 3. Aggregator writes None on all-None fallback
# =============================================================================


def test_aggregator_writes_oracle_none_when_all_cells_lack_oracle_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """4 cells; ALL have `oracle_loss=None`. Assert
    `n_oracle_cells_paired == 0` AND the three oracle CI fields are
    None AND the row's main `primary_metric_*` fields are still
    populated (the two bootstraps are independent)."""
    config, output_root, manifest = _setup(tmp_path)
    _stub_load_run(monkeypatch, _ok_rows_both_families())
    _stub_families(monkeypatch)
    _stub_per_cell(
        monkeypatch,
        {
            "fake_binary": _cells_with_oracle(
                [
                    (0, 0, 0.60, 0.40, None),
                    (0, 1, 0.62, 0.42, None),
                    (1, 0, 0.58, 0.38, None),
                    (1, 1, 0.64, 0.44, None),
                ]
            )
        },
    )

    env = build_run_environment(profile="smoke")
    rollup = aggregate_bootstrap_ensemble_lift_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    row = rollup[0]
    assert row.n_oracle_cells_paired == 0
    assert row.oracle_metric_mean is None
    assert row.oracle_metric_ci_lo is None
    assert row.oracle_metric_ci_hi is None
    assert row.primary_metric_mean is not None  # main bootstrap independent
    assert row.primary_metric_ci_lo is not None
    assert row.primary_metric_ci_hi is not None


# =============================================================================
# 4. Aggregator oracle sign convention is loss_gbm - oracle_loss
# =============================================================================


def test_aggregator_oracle_delta_sign_convention_is_loss_gbm_minus_oracle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asymmetric fixture: per-cell `loss_gbm=0.50, oracle_loss=0.30,
    loss_gbm_plus_seq=0.45` so per-cell oracle Δ = 0.20 AND per-cell
    main Δloss = 0.05 (intentionally distinct so a cross-wire
    mutation that writes the main bootstrap into the oracle fields
    fails the `== 0.20` assertion). Pins the sign convention AND the
    field-routing contract."""
    config, output_root, manifest = _setup(tmp_path)
    _stub_load_run(monkeypatch, _ok_rows_both_families())
    _stub_families(monkeypatch)
    _stub_per_cell(
        monkeypatch,
        {
            "fake_binary": _cells_with_oracle(
                [(s, f, 0.50, 0.45, 0.30) for s in (0, 1) for f in (0, 1)]
            )
        },
    )

    env = build_run_environment(profile="smoke")
    rollup = aggregate_bootstrap_ensemble_lift_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    row = rollup[0]
    assert row.oracle_metric_mean == pytest.approx(0.20, abs=1e-9)
    # Cross-wire mutation pin: if the main bootstrap's mean (0.05) were
    # written into the oracle fields, this would fire.
    assert row.oracle_metric_mean != pytest.approx(0.05, abs=1e-9)
    # The main bootstrap wrote the right value to the right field.
    assert row.primary_metric_mean == pytest.approx(0.05, abs=1e-9)


# =============================================================================
# 5. Aggregator oracle OOM gate raises
# =============================================================================


def test_aggregator_oracle_oom_gate_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin the oracle bootstrap's defensive OOM ceiling. In production
    the main gate fires first on the equal-or-larger main array, so
    this test forces the oracle gate by stubbing
    `entity_block_bootstrap_ci` with a side effect that drops
    `BOOTSTRAP_ROW_COUNT_CEILING` to 0 between the main and oracle
    bootstrap invocations. The oracle gate condition then becomes
    `n_oracle_cells * n_resamples > 0`, which always fires for
    positive counts."""
    config, output_root, manifest = _setup(tmp_path, bootstrap_n_resamples=2)
    _stub_load_run(monkeypatch, _ok_rows_both_families())
    _stub_families(monkeypatch)
    _stub_per_cell(
        monkeypatch,
        {
            "fake_binary": _cells_with_oracle(
                [(s, f, 0.60, 0.40, 0.30) for s in (0, 1) for f in (0, 1)]
            )
        },
    )

    import benchmarks.report.bootstrap_ensemble_lift as _module

    # Establish a monkeypatch-tracked baseline for the ceiling so the
    # post-test revert restores it even after the side-effecting stub
    # mutates the module attribute.
    monkeypatch.setattr(_module, "BOOTSTRAP_ROW_COUNT_CEILING", 100)

    def _bootstrap_with_side_effect(
        *_args: object, **_kwargs: object
    ) -> tuple[float, float, float]:
        # The aggregator's oracle gate reads the ceiling AFTER the
        # main bootstrap returns; lowering it here ensures the gate
        # fires on the oracle invocation that has not yet been called.
        _module.BOOTSTRAP_ROW_COUNT_CEILING = 0
        return (0.20, 0.15, 0.25)

    monkeypatch.setattr(_module, "entity_block_bootstrap_ci", _bootstrap_with_side_effect)

    env = build_run_environment(profile="smoke")
    with pytest.raises(RawRollupError, match=r"oracle delta"):
        aggregate_bootstrap_ensemble_lift_rollup(
            config, output_root=output_root, env=env, manifest=manifest
        )


# =============================================================================
# 6. Aggregator oracle NaN guard fires via injection seam
# =============================================================================


def test_aggregator_oracle_nan_delta_raises_via_stub(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stub `compute_per_cell_lift_deltas` (the same monkeypatch seam
    used by the B16 aggregator tests) to return a single
    `PerCellLiftDelta` record whose `loss_gbm` is NaN while
    `oracle_loss=0.30` and `delta_loss=0.20` (finite). The main
    bootstrap's `np.isfinite(deltas).all()` guard fires FIRST in
    production. To exercise the new oracle `np.isfinite` guard
    structurally, the test stubs `entity_block_bootstrap_ci` to a
    no-op so the main `delta_loss` array (finite) passes its gate
    and runs; the oracle `loss_gbm - oracle_loss` value (NaN - 0.30
    = NaN) trips the new oracle guard. Pins the guard message via
    `match=` so a guard rename is caught."""
    config, output_root, manifest = _setup(tmp_path)
    _stub_load_run(monkeypatch, _ok_rows_both_families(n_seeds=1, n_folds=1))
    _stub_families(monkeypatch)
    nan_record = (
        PerCellLiftDelta(
            seed=0,
            fold_index=0,
            loss_gbm=float("nan"),
            loss_gbm_plus_seq=float("nan") - 0.20,
            delta_loss=0.20,
            oracle_loss=0.30,
        ),
        PerCellLiftDelta(
            seed=0,
            fold_index=0,
            loss_gbm=0.60,
            loss_gbm_plus_seq=0.40,
            delta_loss=0.20,
            oracle_loss=0.30,
        ),
    )
    # Note: the schema rejects NaN delta_loss upstream; here we inject
    # via direct construction to exercise the oracle isfinite guard.
    # The main bootstrap's `np.isfinite(deltas)` array is finite
    # (both delta_loss values are 0.20), so the main guard does NOT
    # fire; only the oracle guard sees a NaN (`loss_gbm - oracle_loss`
    # = NaN - 0.30 = NaN on the first cell).
    _stub_per_cell(
        monkeypatch,
        {"fake_binary": ComputePerCellLiftDeltasResult(cells=nan_record, selector="log_loss")},
    )

    env = build_run_environment(profile="smoke")
    with pytest.raises(RawRollupError, match=r"non-finite oracle delta"):
        aggregate_bootstrap_ensemble_lift_rollup(
            config, output_root=output_root, env=env, manifest=manifest
        )


# =============================================================================
# 7. Renderer: oracle CI cell shows mean + interval; header includes "[95% CI]"
# =============================================================================


def test_renderer_oracle_ci_cell_renders_mean_and_interval() -> None:
    """Construct a rollup row with finite oracle CI values AND
    `n_oracle_cells_paired == n_pair_grid` (no oracle asterisk).
    Assert the rendered CI cell matches `0.NNNN [0.NNNN, 0.NNNN]`
    (no trailing asterisk) AND the header includes
    `Δloss(oracle) [95% CI]`."""
    result = _make_lift_result()
    rollup = [_make_rollup_row(n_oracle_cells_paired=4, n_pair_grid=4)]
    md = render_ensemble_lift_markdown_with_ci(result, rollup)
    assert "Δloss(oracle) [95% CI]" in md
    oracle_cell = "0.1000 [0.0800, 0.1200]"
    assert oracle_cell in md
    # The oracle cell has no trailing asterisk (full coverage).
    assert oracle_cell + "*" not in md


# =============================================================================
# 8. Renderer: oracle partial asterisk fires independently from main
# =============================================================================


def test_renderer_oracle_ci_cell_partial_asterisk_fires_independently() -> None:
    """Row with `n_pair_grid=4, n_cells_paired=4` (main asterisk does
    NOT fire) AND `n_oracle_cells_paired=3` (oracle asterisk SHOULD
    fire). Assert the main Δloss CI cell has no asterisk AND the
    oracle Δloss CI cell carries a trailing asterisk."""
    result = _make_lift_result()
    rollup = [_make_rollup_row(n_cells_paired=4, n_pair_grid=4, n_oracle_cells_paired=3)]
    md = render_ensemble_lift_markdown_with_ci(result, rollup)
    main_cell = "0.2000 [0.1500, 0.2500]"
    oracle_cell = "0.1000 [0.0800, 0.1200]"
    assert main_cell in md
    assert main_cell + "*" not in md
    assert oracle_cell + "*" in md


# =============================================================================
# 9. Renderer: oracle "(no CI)" on n_oracle_cells_paired=0 (non-sentinel row)
# =============================================================================


def test_renderer_oracle_ci_cell_renders_no_ci_when_n_oracle_cells_paired_is_zero() -> None:
    """Non-sentinel row (`bootstrap_skipped_reason=None`) with
    `n_oracle_cells_paired=0` AND `oracle_metric_mean=None`. Assert
    the rendered oracle CI cell is `(no CI)` AND the main Δloss CI
    cell still renders the finite interval (the two columns are
    independent)."""
    result = _make_lift_result()
    rollup = [
        _make_rollup_row(
            n_oracle_cells_paired=0,
            oracle_metric_mean=None,
            oracle_metric_ci_lo=None,
            oracle_metric_ci_hi=None,
        )
    ]
    md = render_ensemble_lift_markdown_with_ci(result, rollup)
    # Main Δloss CI cell still surfaces the numeric interval.
    assert "0.2000 [0.1500, 0.2500]" in md
    # Oracle column collapses to `(no CI)`.
    table_row = next(line for line in md.splitlines() if "0.2000 [0.1500, 0.2500]" in line)
    assert "(no CI)" in table_row


# =============================================================================
# 10. Renderer: sentinel row writes "(no CI)" to BOTH columns simultaneously
# =============================================================================


def test_renderer_oracle_ci_cell_renders_no_ci_when_sentinel_row() -> None:
    """Sentinel row (`bootstrap_skipped_reason="no_gbm_predictions"`,
    `n_oracle_cells_paired=0`, `oracle_metric_mean=None`,
    `primary_metric_mean=None`). Assert BOTH the rendered oracle CI
    cell AND the main Δloss CI cell are `(no CI)`. A renderer bug
    that routes sentinel rows through the n_oracle_cells_paired==0
    branch would produce the same oracle output but would leave the
    main CI cell rendering a numeric interval; the cross-column
    assertion catches this."""
    result = _make_lift_result()
    rollup = [
        _make_rollup_row(
            n_cells_paired=0,
            n_pair_grid=0,
            n_oracle_cells_paired=0,
            primary_metric_mean=None,
            primary_metric_ci_lo=None,
            primary_metric_ci_hi=None,
            oracle_metric_mean=None,
            oracle_metric_ci_lo=None,
            oracle_metric_ci_hi=None,
            bootstrap_skipped_reason="no_gbm_predictions",
        )
    ]
    md = render_ensemble_lift_markdown_with_ci(result, rollup)
    table_row = next(line for line in md.splitlines() if "| ds_one |" in line)
    # The table row carries TWO `(no CI)` cells under the sentinel
    # branch: one for the main Δloss column, one for the oracle.
    assert table_row.count("(no CI)") == 2


# =============================================================================
# 11. Renderer: oracle partial asterisk suppressed when fully paired
# =============================================================================


def test_renderer_oracle_partial_asterisk_suppressed_when_fully_paired() -> None:
    """Row with `n_pair_grid=4, n_oracle_cells_paired=4` (equal, full
    coverage). Assert the rendered oracle CI cell contains the
    numeric interval AND no trailing asterisk."""
    result = _make_lift_result()
    rollup = [_make_rollup_row(n_pair_grid=4, n_oracle_cells_paired=4)]
    md = render_ensemble_lift_markdown_with_ci(result, rollup)
    oracle_cell = "0.1000 [0.0800, 0.1200]"
    assert oracle_cell in md
    assert oracle_cell + "*" not in md


# =============================================================================
# 12. Aggregator sentinel-emit functional test: oracle defaults populate
# =============================================================================


def test_aggregator_emits_sentinel_row_with_default_oracle_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configure a fixture with disjoint seed sets so the aggregator
    routes through `_emit_sentinel_row`. Assert the emitted row has
    `n_oracle_cells_paired == 0` AND `oracle_metric_mean is None`
    AND `oracle_metric_ci_lo is None` AND
    `oracle_metric_ci_hi is None`. Pins the sentinel-emit helper's
    documentary defaults for the 4 new oracle fields."""
    config, output_root, manifest = _setup(tmp_path)
    _stub_load_run(monkeypatch, _ok_rows_both_families())
    _stub_families(monkeypatch)
    # Empty per-cell result -> aggregator routes through sentinel emit.
    _stub_per_cell(monkeypatch, {})

    env = build_run_environment(profile="smoke")
    rollup = aggregate_bootstrap_ensemble_lift_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    assert len(rollup) == 1
    row = rollup[0]
    assert row.bootstrap_skipped_reason == "all_cells_skipped_in_manifest"
    assert row.n_oracle_cells_paired == 0
    assert row.oracle_metric_mean is None
    assert row.oracle_metric_ci_lo is None
    assert row.oracle_metric_ci_hi is None


# =============================================================================
# 13. Aggregator: XOR seed offset produces independent PCG64 streams
# =============================================================================


def test_aggregator_oracle_and_main_ci_differ_under_xor_seed_offset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin the `BOOTSTRAP_ORACLE_SEED_OFFSET` derived-seed mechanism
    (R-B20-2a). Construct a fixture with AT LEAST 8 cells where
    per-cell `delta_loss == loss_gbm - oracle_loss` (so the main
    bootstrap and oracle bootstrap operate on IDENTICAL per-cell
    value arrays) AND set `bootstrap_n_resamples=500` (enough
    resamples + cells that two independent PCG64 streams produce
    numerically distinct CI bounds with probability ~1). Run the
    aggregator under the real offset and under
    `BOOTSTRAP_ORACLE_SEED_OFFSET=0`. Under the real offset, the
    two CIs differ; under offset=0 the two PCG64 streams collapse
    to the same seed and produce identical bounds. The `!=`
    assertion kills a regression where the XOR offset is dropped."""
    config, output_root, manifest = _setup(tmp_path, bootstrap_n_resamples=500)

    # 8 cells with non-degenerate variance. To make the main delta_loss
    # array and the oracle delta array numerically IDENTICAL, set
    # `loss_gbm_plus_seq = oracle_loss = 0.30` (constant) and vary
    # `loss_gbm`. Then per-cell delta_loss = loss_gbm - 0.30 and
    # per-cell oracle Δ = loss_gbm - 0.30 (equal). The varying
    # `loss_gbm` gives the bootstrap non-zero variance so two
    # independent PCG64 streams produce numerically distinct CI
    # bounds with probability ~1.
    seed_fold_pairs = [(s, f) for s in (0, 1, 2, 3) for f in (0, 1)]
    loss_gbm_values = [0.60, 0.65, 0.55, 0.62, 0.58, 0.61, 0.59, 0.63]

    def _run_once() -> tuple[float | None, float | None, float | None, float | None]:
        _stub_load_run(monkeypatch, _ok_rows_both_families(n_seeds=4, n_folds=2))
        _stub_families(monkeypatch)
        _stub_per_cell(
            monkeypatch,
            {
                "fake_binary": _cells_with_oracle(
                    [
                        (s, f, lg, 0.30, 0.30)
                        for (s, f), lg in zip(seed_fold_pairs, loss_gbm_values, strict=True)
                    ]
                )
            },
        )
        env = build_run_environment(profile="smoke")
        rollup = aggregate_bootstrap_ensemble_lift_rollup(
            config, output_root=output_root, env=env, manifest=manifest
        )
        return (
            rollup[0].primary_metric_ci_lo,
            rollup[0].primary_metric_ci_hi,
            rollup[0].oracle_metric_ci_lo,
            rollup[0].oracle_metric_ci_hi,
        )

    real_main_lo, real_main_hi, real_oracle_lo, real_oracle_hi = _run_once()
    # The two value arrays are identical (delta_loss = lg - 0.30 =
    # oracle Δ for every cell), so any divergence in CI bounds is
    # purely from the PCG64 stream difference under the XOR offset.
    assert (real_oracle_lo, real_oracle_hi) != (real_main_lo, real_main_hi)

    # Now collapse the seeds and re-run. With offset=0 the oracle
    # bootstrap uses the same seed as the main bootstrap; on identical
    # value arrays the two CIs collapse to the same bounds.
    import benchmarks.report._bootstrap_aggregate as _agg

    monkeypatch.setattr(_agg, "BOOTSTRAP_ORACLE_SEED_OFFSET", 0)
    collapsed_main_lo, collapsed_main_hi, collapsed_oracle_lo, collapsed_oracle_hi = _run_once()
    assert (collapsed_oracle_lo, collapsed_oracle_hi) == (
        collapsed_main_lo,
        collapsed_main_hi,
    )
