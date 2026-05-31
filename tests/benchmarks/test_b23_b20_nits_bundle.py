"""Phase B23 D-B20 NITs bundle tests (14 named; 16 collected).

Covers the three small B20 deferrals:
- D-B20.1: renderer oracle CI partial-coverage footnote (#1-#5 + #5b + #11)
- D-B20.2: stable raise-message discriminator tokens (#6, #7)
- D-B20.3: cross-field invariant validator (#8, #9, #10, #12, #13)

Test #3 is parametrized over 3 sentinel reasons; #5b was added
in the R1 build-swarm closure for qa-I2.
"""

from pathlib import Path

import pandas as pd
import pytest
from benchmarks.bootstrap_manifest import EnsembleLiftRollupRow
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
from pydantic import ValidationError

from tests.benchmarks._fakes import register_all_fakes_and_get_panels

_GBM_MODEL = "gbm_constant"
_SEQ_MODEL = "seq_constant"


# =============================================================================
# Renderer test helpers
# =============================================================================


def _make_lift_result(dataset_names: list[str]) -> EnsembleLiftExperimentResult:
    rows = tuple(
        PerDatasetLift(
            dataset_name=name,
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
        for name in dataset_names
    )
    return EnsembleLiftExperimentResult(
        run_id="b23-test",
        seq_family="seq_sklearn",
        baseline_family="gbm",
        rows=rows,
        wilcoxon=WilcoxonResult(
            statistic=1.0,
            p_value=0.10,
            holm_adjusted_p_value=0.10,
            n_datasets=len(dataset_names),
            family_size=1,
        ),
    )


def _make_rollup_row(
    *,
    dataset_name: str = "ds_one",
    n_cells_paired: int = 4,
    n_pair_grid: int = 4,
    n_oracle_cells_paired: int = 4,
    bootstrap_skipped_reason: str | None = None,
) -> EnsembleLiftRollupRow:
    return EnsembleLiftRollupRow(
        dataset_name=dataset_name,
        task_type="binary",
        primary_metric="delta_loss",
        primary_loss_column="log_loss",
        # B29 / D-B28.1 closure: n_seeds * n_folds bumped to
        # 16 to satisfy the new `n_pair_grid <= n_seeds *
        # n_folds` validator for the largest n_pair_grid used
        # in B23 tests (5 in the positive-equality boundary).
        n_seeds=4,
        n_folds=4,
        n_cells_paired=n_cells_paired,
        n_pair_grid=n_pair_grid,
        n_oracle_cells_paired=n_oracle_cells_paired,
        n_skipped_cells=0,
        primary_metric_mean=0.20 if bootstrap_skipped_reason is None else None,
        primary_metric_ci_lo=0.15 if bootstrap_skipped_reason is None else None,
        primary_metric_ci_hi=0.25 if bootstrap_skipped_reason is None else None,
        # B28 / D-B27.2 closure: oracle defaults gated on
        # n_oracle_cells_paired (not bootstrap_skipped_reason)
        # so the row satisfies the new oracle CI-sentinel
        # invariant regardless of the main bootstrap state.
        oracle_metric_mean=0.10 if n_oracle_cells_paired > 0 else None,
        oracle_metric_ci_lo=0.08 if n_oracle_cells_paired > 0 else None,
        oracle_metric_ci_hi=0.12 if n_oracle_cells_paired > 0 else None,
        bootstrap_seed=42,
        bootstrap_n_resamples=10_000,
        bootstrap_numpy_version="2.3.0",
        bootstrap_skipped_reason=bootstrap_skipped_reason,
        manifest_fingerprint="f" * 64,
    )


# =============================================================================
# 1. Renderer oracle partial-coverage footnote fires on partial row
# =============================================================================


def test_renderer_oracle_partial_footnote_fires_when_any_row_partial() -> None:
    """One dataset with n_oracle_cells_paired=3, n_pair_grid=4. The
    footnote table appears with the correct dataset row."""
    result = _make_lift_result(["ds_one"])
    rollup = [_make_rollup_row(n_oracle_cells_paired=3, n_pair_grid=4)]
    md = render_ensemble_lift_markdown_with_ci(result, rollup)
    assert "Oracle partial-coverage footnotes" in md
    assert "| ds_one | 3 | 4 | 1 |" in md


# =============================================================================
# 2. Footnote silent when no row is partial
# =============================================================================


def test_renderer_oracle_partial_footnote_silent_on_happy_path() -> None:
    """All rows have n_oracle_cells_paired == n_pair_grid. The
    footnote table does NOT appear."""
    result = _make_lift_result(["ds_one"])
    rollup = [_make_rollup_row(n_oracle_cells_paired=4, n_pair_grid=4)]
    md = render_ensemble_lift_markdown_with_ci(result, rollup)
    assert "Oracle partial-coverage footnotes" not in md


# =============================================================================
# 3. Footnote skips sentinel rows (parametrized over all 3 reasons)
# =============================================================================


@pytest.mark.parametrize(
    "sentinel_reason",
    [
        "no_gbm_predictions",
        "no_seq_predictions",
        "all_cells_skipped_in_manifest",
    ],
)
def test_renderer_oracle_partial_footnote_skips_sentinel_rows(
    sentinel_reason: str,
) -> None:
    """qa-R1-I4 + R1 build-swarm CRITICAL closure: the actual
    short-circuit mechanism is `bootstrap_skipped_reason is not
    None`, NOT `n_pair_grid==0`. Sentinel rows can carry non-zero
    `n_pair_grid` (per `bootstrap_manifest.py:592-595` for
    `all_cells_skipped_in_manifest`, and the aggregator at
    `bootstrap_ensemble_lift.py:235` forwards the real grid for
    the other two sentinels). Using `n_pair_grid=4` +
    `n_oracle_cells_paired=2` here means the trigger condition
    `n_oracle_cells_paired < n_pair_grid AND n_pair_grid > 0`
    DOES fire, so the test now actually proves the
    `bootstrap_skipped_reason is not None` guard at
    `ensemble_lift.py:436` is what suppresses the footnote.
    Without that guard the test would go RED for all three
    sentinel reasons."""
    result = _make_lift_result(["ds_sentinel", "ds_happy"])
    rollup = [
        _make_rollup_row(
            dataset_name="ds_sentinel",
            n_cells_paired=2,
            n_pair_grid=4,
            n_oracle_cells_paired=2,
            bootstrap_skipped_reason=sentinel_reason,
        ),
        _make_rollup_row(
            dataset_name="ds_happy",
            n_oracle_cells_paired=4,
            n_pair_grid=4,
        ),
    ]
    md = render_ensemble_lift_markdown_with_ci(result, rollup)
    assert "Oracle partial-coverage footnotes" not in md


# =============================================================================
# 4. Footnote sorts by dataset name ascending
# =============================================================================


def test_renderer_oracle_partial_footnote_sorts_by_dataset_name() -> None:
    """Two affected rows with dataset names "z_alpha" and "a_beta".
    The footnote table lists "a_beta" before "z_alpha"."""
    result = _make_lift_result(["z_alpha", "a_beta"])
    rollup = [
        _make_rollup_row(dataset_name="z_alpha", n_oracle_cells_paired=2, n_pair_grid=4),
        _make_rollup_row(dataset_name="a_beta", n_oracle_cells_paired=3, n_pair_grid=4),
    ]
    md = render_ensemble_lift_markdown_with_ci(result, rollup)
    # Locate the footnote table and verify "a_beta" appears before "z_alpha".
    footnote_idx = md.find("Oracle partial-coverage footnotes")
    assert footnote_idx >= 0
    footnote_block = md[footnote_idx:]
    a_beta_idx = footnote_block.find("| a_beta |")
    z_alpha_idx = footnote_block.find("| z_alpha |")
    assert a_beta_idx >= 0
    assert z_alpha_idx >= 0
    assert a_beta_idx < z_alpha_idx


# =============================================================================
# 5. n_missing column arithmetic
# =============================================================================


def test_renderer_oracle_partial_footnote_n_missing_column() -> None:
    """One row with n_oracle_cells_paired=2, n_pair_grid=5; the
    n_missing column reads "3"."""
    result = _make_lift_result(["ds_one"])
    rollup = [_make_rollup_row(n_oracle_cells_paired=2, n_pair_grid=5)]
    md = render_ensemble_lift_markdown_with_ci(result, rollup)
    assert "| ds_one | 2 | 5 | 3 |" in md


# =============================================================================
# 5b. Zero oracle cells with non-zero grid still triggers the footnote
# =============================================================================


def test_renderer_oracle_partial_footnote_fires_when_oracle_cells_zero_and_grid_nonzero() -> None:
    """qa-R1-build-I2 closure: a non-sentinel row with
    `n_oracle_cells_paired=0, n_pair_grid=4` is partial-coverage
    by design (0 of 4 oracle cells paired). The trigger predicate
    `n_oracle_cells_paired < n_pair_grid AND n_pair_grid > 0`
    evaluates true; the footnote fires with `n_missing=4`. The
    footnote covers a strictly wider domain than the
    per-cell-asterisk path: the oracle CI cell at
    `ensemble_lift.py:179-180` short-circuits to `"(no CI)"` when
    `n_oracle_cells_paired == 0` and never reaches the asterisk
    branch at `:189-192`, while the footnote fires regardless."""
    result = _make_lift_result(["ds_zero_oracle"])
    rollup = [
        _make_rollup_row(
            dataset_name="ds_zero_oracle",
            n_cells_paired=4,
            n_pair_grid=4,
            n_oracle_cells_paired=0,
        )
    ]
    md = render_ensemble_lift_markdown_with_ci(result, rollup)
    assert "Oracle partial-coverage footnotes" in md
    assert "| ds_zero_oracle | 0 | 4 | 4 |" in md


# =============================================================================
# Aggregator-test helpers (mirror B20 scaffolding)
# =============================================================================


def _make_config(
    tmp_path: Path, *, bootstrap_n_resamples: int | None = None
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
        run_id="b23-test",
        library_git_sha="0" * 40,
        profile="smoke",
        hardware_tier="cpu",
        output_root=output_root,
    )
    write_run_manifest(output_root, manifest)
    return manifest


def _ok_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for s in (0, 1):
        for f in (0, 1):
            for model in (_GBM_MODEL, _SEQ_MODEL):
                rows.append(
                    {
                        "dataset_name": "fake_binary",
                        "model_name": model,
                        "task_type": "binary",
                        "seed": s,
                        "fold_index": f,
                        "variant": "default",
                        "skipped_reason": None,
                    }
                )
    return rows


# =============================================================================
# 6. Oracle NaN raise message contains oracle_metric_mean: prefix
# =============================================================================


def test_aggregator_oracle_nan_raise_message_contains_oracle_metric_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The isfinite-guard raise message starts with the
    `oracle_metric_mean:` prefix token; pytest match= pins it."""
    config = _make_config(tmp_path)
    output_root = tmp_path / "out"
    manifest = _make_run_manifest(config, output_root)

    import benchmarks.report.bootstrap_ensemble_lift as _module

    monkeypatch.setattr(_module, "load_run", lambda _root: pd.DataFrame(_ok_rows()))  # type: ignore[misc]
    monkeypatch.setattr(
        _module,
        "model_families",
        lambda _df: {_GBM_MODEL: "gbm", _SEQ_MODEL: "seq_sklearn"},  # type: ignore[misc]
    )

    # Two records with finite delta_loss but NaN in loss_gbm so the
    # oracle delta becomes NaN; the main isfinite guard passes
    # (delta_loss is finite) and the oracle isfinite guard fires.
    nan_cells = (
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
    monkeypatch.setattr(
        _module,
        "compute_per_cell_lift_deltas",
        lambda **_k: ComputePerCellLiftDeltasResult(cells=nan_cells, selector="log_loss"),  # type: ignore[misc]
    )

    env = build_run_environment(profile="smoke")
    with pytest.raises(RawRollupError, match=r"oracle_metric_mean:"):
        aggregate_bootstrap_ensemble_lift_rollup(
            config, output_root=output_root, env=env, manifest=manifest
        )


# =============================================================================
# 7. Oracle OOM raise message contains n_oracle_cells_paired: prefix
# =============================================================================


def test_aggregator_oracle_oom_raise_message_contains_n_oracle_cells_paired_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The OOM-gate raise message starts with the
    `n_oracle_cells_paired:` prefix token; pytest match= pins it.
    Uses bootstrap_n_resamples=2 + initial ceiling=100 so the main
    gate passes (4*2 <= 100); the side-effect drops the ceiling
    to 0 between calls so the oracle gate fires."""
    config = _make_config(tmp_path, bootstrap_n_resamples=2)
    output_root = tmp_path / "out"
    manifest = _make_run_manifest(config, output_root)

    import benchmarks.report._bootstrap_aggregate as _agg
    import benchmarks.report.bootstrap_ensemble_lift as _module

    monkeypatch.setattr(_module, "load_run", lambda _root: pd.DataFrame(_ok_rows()))  # type: ignore[misc]
    monkeypatch.setattr(
        _module,
        "model_families",
        lambda _df: {_GBM_MODEL: "gbm", _SEQ_MODEL: "seq_sklearn"},  # type: ignore[misc]
    )
    happy_cells = tuple(
        PerCellLiftDelta(
            seed=s,
            fold_index=f,
            loss_gbm=0.60,
            loss_gbm_plus_seq=0.40,
            delta_loss=0.20,
            oracle_loss=0.30,
        )
        for s in (0, 1)
        for f in (0, 1)
    )
    monkeypatch.setattr(
        _module,
        "compute_per_cell_lift_deltas",
        lambda **_k: ComputePerCellLiftDeltasResult(cells=happy_cells, selector="log_loss"),  # type: ignore[misc]
    )

    # Side-effect monkeypatch to flip the ceiling AFTER the main
    # bootstrap completes so the oracle OOM gate fires.
    monkeypatch.setattr(_agg, "BOOTSTRAP_ROW_COUNT_CEILING", 100)

    def _bootstrap_with_side_effect(
        *_a: object, **_k: object
    ) -> tuple[float, float, float, str | None]:
        _agg.BOOTSTRAP_ROW_COUNT_CEILING = 0
        return (0.20, 0.15, 0.25, None)

    monkeypatch.setattr(_module, "entity_block_bootstrap_ci", _bootstrap_with_side_effect)

    env = build_run_environment(profile="smoke")
    with pytest.raises(RawRollupError, match=r"n_oracle_cells_paired:"):
        aggregate_bootstrap_ensemble_lift_rollup(
            config, output_root=output_root, env=env, manifest=manifest
        )


# =============================================================================
# 8. Validator rejects n_oracle_cells_paired > n_cells_paired
# =============================================================================


def test_ensemble_lift_validator_rejects_n_oracle_cells_paired_exceeds_n_cells_paired() -> None:
    with pytest.raises(ValidationError, match=r"n_oracle_cells_paired.*exceeds n_cells_paired"):
        _make_rollup_row(n_cells_paired=4, n_oracle_cells_paired=5, n_pair_grid=10)


# =============================================================================
# 9. Validator rejects n_oracle_cells_paired > n_pair_grid
# =============================================================================


def test_ensemble_lift_validator_rejects_n_oracle_cells_paired_exceeds_n_pair_grid() -> None:
    with pytest.raises(ValidationError, match=r"n_oracle_cells_paired.*exceeds n_pair_grid"):
        _make_rollup_row(n_cells_paired=10, n_oracle_cells_paired=5, n_pair_grid=4)


# =============================================================================
# 10. Validator accepts sentinel row with zero counts
# =============================================================================


def test_ensemble_lift_validator_accepts_sentinel_row_zero_counts() -> None:
    """Sentinel rows trivially satisfy all three invariants
    (n_cells_paired = n_pair_grid = n_oracle_cells_paired = 0)."""
    row = _make_rollup_row(
        n_cells_paired=0,
        n_pair_grid=0,
        n_oracle_cells_paired=0,
        bootstrap_skipped_reason="no_gbm_predictions",
    )
    assert row.n_cells_paired == 0
    assert row.n_oracle_cells_paired == 0


# =============================================================================
# 11. Footnote with 3 rows: mixed partial + full + sentinel (qa-R1-I1)
# =============================================================================


def test_renderer_oracle_partial_footnote_mixed_partial_and_full_rows() -> None:
    """Three rows: one fully paired, one partially paired, one
    sentinel. The footnote table contains exactly the partial row
    AND NOT the full or sentinel rows."""
    result = _make_lift_result(["ds_full", "ds_partial", "ds_sentinel"])
    rollup = [
        _make_rollup_row(dataset_name="ds_full", n_oracle_cells_paired=4, n_pair_grid=4),
        _make_rollup_row(dataset_name="ds_partial", n_oracle_cells_paired=2, n_pair_grid=4),
        _make_rollup_row(
            dataset_name="ds_sentinel",
            n_cells_paired=0,
            n_pair_grid=0,
            n_oracle_cells_paired=0,
            bootstrap_skipped_reason="no_gbm_predictions",
        ),
    ]
    md = render_ensemble_lift_markdown_with_ci(result, rollup)
    assert "Oracle partial-coverage footnotes" in md
    # Locate the footnote block.
    footnote_idx = md.find("Oracle partial-coverage footnotes")
    footnote_block = md[footnote_idx:]
    # Only ds_partial appears in the table; ds_full + ds_sentinel
    # are skipped.
    assert "| ds_partial | 2 | 4 | 2 |" in footnote_block
    # Pipe-delimited row pattern with the other two dataset names
    # must NOT appear inside the footnote block.
    assert "| ds_full |" not in footnote_block
    assert "| ds_sentinel |" not in footnote_block


# =============================================================================
# 12. Validator accepts positive-equality boundary (qa-R1-I3)
# =============================================================================


def test_ensemble_lift_validator_accepts_positive_equality_boundary() -> None:
    """Construct a row with `n_cells_paired = n_oracle_cells_paired
    = n_pair_grid = 5`. All three invariants are equality
    boundaries; the row constructs successfully. A mutation that
    changed `>` to `>=` in any of the three clauses would fail
    this test."""
    row = _make_rollup_row(n_cells_paired=5, n_oracle_cells_paired=5, n_pair_grid=5)
    assert row.n_oracle_cells_paired == 5
    assert row.n_cells_paired == 5
    assert row.n_pair_grid == 5


# =============================================================================
# 13. Validator rejects n_cells_paired > n_pair_grid (arch-R1-I1)
# =============================================================================


def test_ensemble_lift_validator_rejects_n_cells_paired_exceeds_n_pair_grid() -> None:
    """Peer invariant: n_cells_paired must not exceed n_pair_grid
    (assumed by the renderer's partial-coverage flag at
    ensemble_lift.py:166)."""
    with pytest.raises(ValidationError, match=r"n_cells_paired.*exceeds n_pair_grid"):
        _make_rollup_row(n_cells_paired=5, n_oracle_cells_paired=0, n_pair_grid=4)
