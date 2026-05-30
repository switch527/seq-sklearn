"""Phase B22 D-B16.3 per-fold CIs tests (19 tests)."""

from pathlib import Path

import pandas as pd
import pytest
from benchmarks.bootstrap_manifest import (
    EnsembleLiftRollupRow,
    FoldCI,
    HPOUpliftRollupRow,
    PairwiseRollupRow,
    RollupRow,
    TrainingTimeRollupRow,
    load_rollup,
    write_rollup,
)
from benchmarks.config import BenchmarkConfig, ExperimentSpec
from benchmarks.experiments import build_run_environment
from benchmarks.experiments.ensemble_lift import (
    ComputePerCellLiftDeltasResult,
    PerCellLiftDelta,
)
from benchmarks.report.bootstrap_ensemble_lift import (
    aggregate_bootstrap_ensemble_lift_rollup,
)
from benchmarks.run_manifest import RunManifest, build_run_manifest, write_run_manifest
from pydantic import ValidationError

from tests.benchmarks._fakes import register_all_fakes_and_get_panels

_GBM_MODEL = "gbm_constant"
_SEQ_MODEL = "seq_constant"


# =============================================================================
# Helpers
# =============================================================================


def _make_config(
    tmp_path: Path,
    *,
    per_fold_enabled: bool = False,
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
                bootstrap_per_fold_cis_enabled=per_fold_enabled,
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
        run_id="b22-test",
        library_git_sha="0" * 40,
        profile="smoke",
        hardware_tier="cpu",
        output_root=output_root,
    )
    write_run_manifest(output_root, manifest)
    return manifest


def _ok_rows(n_seeds: int = 2, n_folds: int = 2) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for s in range(n_seeds):
        for f in range(n_folds):
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


def _stub_load_run(monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, object]]) -> None:
    import benchmarks.report.bootstrap_ensemble_lift as _module

    df = pd.DataFrame(rows)
    monkeypatch.setattr(_module, "load_run", lambda _root: df)  # type: ignore[misc]


def _stub_families(monkeypatch: pytest.MonkeyPatch) -> None:
    import benchmarks.report.bootstrap_ensemble_lift as _module

    families = {_GBM_MODEL: "gbm", _SEQ_MODEL: "seq_sklearn"}
    monkeypatch.setattr(_module, "model_families", lambda _df: dict(families))  # type: ignore[misc]


def _stub_per_cell(
    monkeypatch: pytest.MonkeyPatch,
    cells: tuple[PerCellLiftDelta, ...],
    *,
    seen_no_gbm: bool = False,
    seen_no_seq: bool = False,
) -> None:
    import benchmarks.report.bootstrap_ensemble_lift as _module

    result = ComputePerCellLiftDeltasResult(
        cells=cells,
        seen_no_gbm=seen_no_gbm,
        seen_no_seq=seen_no_seq,
        selector="log_loss",
    )
    monkeypatch.setattr(_module, "compute_per_cell_lift_deltas", lambda **_k: result)  # type: ignore[misc]


def _happy_cells(n_seeds: int = 2, n_folds: int = 2) -> tuple[PerCellLiftDelta, ...]:
    return tuple(
        PerCellLiftDelta(
            seed=s,
            fold_index=f,
            loss_gbm=0.60 + 0.01 * (s + f),
            loss_gbm_plus_seq=0.40 + 0.02 * (s + f),
            delta_loss=0.20 - 0.01 * (s + f),
            oracle_loss=0.30 + 0.01 * (s + f),
        )
        for s in range(n_seeds)
        for f in range(n_folds)
    )


def _run_ensemble_lift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    per_fold_enabled: bool,
    cells: tuple[PerCellLiftDelta, ...],
    n_seeds: int = 2,
    n_folds: int = 2,
    bootstrap_n_resamples: int | None = None,
) -> list[EnsembleLiftRollupRow]:
    config = _make_config(
        tmp_path,
        per_fold_enabled=per_fold_enabled,
        bootstrap_n_resamples=bootstrap_n_resamples,
    )
    output_root = tmp_path / "out"
    manifest = _make_run_manifest(config, output_root)
    _stub_load_run(monkeypatch, _ok_rows(n_seeds=n_seeds, n_folds=n_folds))
    _stub_families(monkeypatch)
    _stub_per_cell(monkeypatch, cells)
    env = build_run_environment(profile="smoke")
    return aggregate_bootstrap_ensemble_lift_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )


# =============================================================================
# 1. FoldCI model invariants
# =============================================================================


def test_fold_ci_model_has_frozen_extra_forbid_config() -> None:
    """FoldCI.model_config matches every existing RollupRow schema:
    frozen=True + extra='forbid'."""
    assert FoldCI.model_config.get("frozen") is True
    assert FoldCI.model_config.get("extra") == "forbid"


# =============================================================================
# 2. FoldCI Field(ge=0) constraints
# =============================================================================


def test_fold_ci_field_ge_constraints_reject_negative() -> None:
    """The three count fields (fold_index, n_seeds, n_entities) all
    carry Field(ge=0) and reject negative values."""
    for field, value in (
        ("fold_index", -1),
        ("n_seeds", -1),
        ("n_entities", -1),
    ):
        kwargs = {
            "fold_index": 0,
            "n_seeds": 0,
            "n_entities": 0,
            "ci_method": "bca",
        }
        kwargs[field] = value
        with pytest.raises(ValidationError):
            FoldCI(**kwargs)


# =============================================================================
# 3. RollupRow per_fold_cis schema default is None
# =============================================================================


def test_rollup_row_per_fold_cis_schema_default_is_none() -> None:
    """Constructing each of the 5 RollupRow types WITHOUT supplying
    per_fold_cis yields None (backward-compat marker)."""
    common = dict(
        dataset_name="ds",
        task_type="binary",
        bootstrap_seed=42,
        bootstrap_n_resamples=100,
        bootstrap_numpy_version="2.0.0",
        manifest_fingerprint="f" * 64,
    )
    b5 = RollupRow(
        model_name="m",
        primary_metric="log_loss",
        n_seeds=2,
        n_cells_evaluated=0,
        n_skipped_cells=0,
        n_rows=0,
        n_entities=0,
        **common,
    )
    assert b5.per_fold_cis is None
    b6 = PairwiseRollupRow(
        model_a="a",
        model_b="b",
        primary_metric="complementarity_score",
        n_seeds=2,
        n_cells_evaluated=0,
        n_skipped_cells=0,
        **common,
    )
    assert b6.per_fold_cis is None
    b7 = TrainingTimeRollupRow(
        model_name="m",
        hardware_tier="cpu",
        primary_metric="wall_seconds",
        n_seeds=2,
        n_cells_evaluated=0,
        n_skipped_cells=0,
        **common,
    )
    assert b7.per_fold_cis is None
    b8 = HPOUpliftRollupRow(
        model_name="m",
        primary_metric="delta",
        primary_loss_column="log_loss",
        n_seeds=2,
        n_folds=2,
        n_cells_paired=0,
        n_skipped_cells=0,
        **common,
    )
    assert b8.per_fold_cis is None
    b16 = EnsembleLiftRollupRow(
        primary_metric="delta_loss",
        primary_loss_column="log_loss",
        n_seeds=2,
        n_folds=2,
        n_cells_paired=0,
        n_pair_grid=0,
        n_oracle_cells_paired=0,
        n_skipped_cells=0,
        **common,
    )
    assert b16.per_fold_cis is None


# =============================================================================
# 4. RollupRow accepts empty list
# =============================================================================


def test_rollup_row_per_fold_cis_accepts_empty_list() -> None:
    """per_fold_cis=[] is a valid state (requested but every fold
    degenerate-zero-rows). The list round-trips."""
    row = RollupRow(
        dataset_name="ds",
        model_name="m",
        task_type="binary",
        primary_metric="log_loss",
        n_seeds=2,
        n_cells_evaluated=0,
        n_skipped_cells=0,
        n_rows=0,
        n_entities=0,
        bootstrap_seed=42,
        bootstrap_n_resamples=100,
        bootstrap_numpy_version="2.0.0",
        manifest_fingerprint="f" * 64,
        per_fold_cis=[],
    )
    assert row.per_fold_cis == []


# =============================================================================
# 5. Parquet round-trip on B5 with 2 FoldCI entries (concrete values)
# =============================================================================


def test_rollup_row_per_fold_cis_round_trip_with_two_folds(tmp_path: Path) -> None:
    """qa-R1-N2 closure: concrete distinct values per FoldCI field,
    including a None vs 'p0_at_edge' pair on ci_fallback_reason to
    catch silent pd.NA coercion."""
    fold0 = FoldCI(
        fold_index=0,
        n_seeds=3,
        n_entities=15,
        metric_mean=0.50,
        metric_ci_lo=0.45,
        metric_ci_hi=0.55,
        ci_method="bca",
        ci_fallback_reason=None,
    )
    fold1 = FoldCI(
        fold_index=1,
        n_seeds=3,
        n_entities=15,
        metric_mean=0.60,
        metric_ci_lo=0.55,
        metric_ci_hi=0.65,
        ci_method="bca",
        ci_fallback_reason="p0_at_edge",
    )
    row = RollupRow(
        dataset_name="ds",
        model_name="m",
        task_type="binary",
        primary_metric="log_loss",
        n_seeds=3,
        n_cells_evaluated=6,
        n_skipped_cells=0,
        n_rows=30,
        n_entities=15,
        primary_metric_mean=0.55,
        primary_metric_ci_lo=0.50,
        primary_metric_ci_hi=0.60,
        bootstrap_seed=42,
        bootstrap_n_resamples=100,
        bootstrap_numpy_version="2.0.0",
        manifest_fingerprint="f" * 64,
        per_fold_cis=[fold0, fold1],
    )
    write_rollup(tmp_path, [row])
    loaded = load_rollup(tmp_path)
    assert len(loaded) == 1
    assert loaded[0].per_fold_cis == [fold0, fold1]


# =============================================================================
# 6. ExperimentSpec default
# =============================================================================


def test_experiment_spec_per_fold_cis_disabled_by_default() -> None:
    spec = ExperimentSpec(kind="raw_loss")
    assert spec.bootstrap_per_fold_cis_enabled is False


# =============================================================================
# 7. Aggregator off: per_fold_cis is None
# =============================================================================


def test_aggregator_writes_per_fold_cis_none_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rollup = _run_ensemble_lift(tmp_path, monkeypatch, per_fold_enabled=False, cells=_happy_cells())
    assert rollup[0].per_fold_cis is None


# =============================================================================
# 8. Aggregator on: per_fold_cis is a non-empty list of FoldCI
# =============================================================================


def test_aggregator_writes_per_fold_cis_list_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rollup = _run_ensemble_lift(tmp_path, monkeypatch, per_fold_enabled=True, cells=_happy_cells())
    row = rollup[0]
    assert row.per_fold_cis is not None
    assert len(row.per_fold_cis) >= 1
    for fc in row.per_fold_cis:
        assert isinstance(fc, FoldCI)
        assert fc.ci_method == "bca"
        assert fc.ci_fallback_reason in (None, "p0_at_edge", "a_overshoot")


# =============================================================================
# 9. Each fold uses distinct seed (qa-R1-C1 / R-B22-4)
# =============================================================================


def test_aggregator_per_fold_cis_each_fold_uses_distinct_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Monkeypatch entity_block_bootstrap_ci to capture every `seed`
    kwarg. Run with 3 folds. Assert the 3 per-fold seeds are
    distinct from each other AND from BOOTSTRAP_DEFAULT_SEED."""
    seeds_captured: list[int] = []

    import benchmarks.report.bootstrap_ensemble_lift as _module
    from benchmarks.metrics.bootstrap import (
        entity_block_bootstrap_ci as _real,
    )

    def _capture(*args: object, **kwargs: object):
        seeds_captured.append(int(kwargs["seed"]))  # type: ignore[arg-type]
        return _real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(_module, "entity_block_bootstrap_ci", _capture)
    import benchmarks.report._bootstrap_aggregate as _agg

    monkeypatch.setattr(_agg, "entity_block_bootstrap_ci", _capture)

    _run_ensemble_lift(tmp_path, monkeypatch, per_fold_enabled=True, cells=_happy_cells(n_folds=3))

    from benchmarks.report._bootstrap_aggregate import BOOTSTRAP_DEFAULT_SEED

    # B16 call order: main pooled, 3 per-fold, oracle pooled
    # (per-fold runs BETWEEN main pool and oracle pool, not after).
    assert seeds_captured[0] == BOOTSTRAP_DEFAULT_SEED  # main pooled
    per_fold_seeds = seeds_captured[1:4]  # 3 per-fold calls
    assert len(per_fold_seeds) == 3
    assert len(set(per_fold_seeds)) == 3
    for s in per_fold_seeds:
        assert s != BOOTSTRAP_DEFAULT_SEED


# =============================================================================
# 10. Exact XOR formula pin (qa-R1-C1 closure)
# =============================================================================


def test_aggregator_per_fold_cis_seed_derivation_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pin the exact XOR formula: a fold_index=N receives
    `BOOTSTRAP_DEFAULT_SEED ^ BOOTSTRAP_PER_FOLD_SEED_OFFSET ^ N`.
    Capture seeds + match against expected."""
    seeds_captured: dict[int, int] = {}

    import benchmarks.report.bootstrap_ensemble_lift as _module
    from benchmarks.metrics.bootstrap import (
        entity_block_bootstrap_ci as _real,
    )
    from benchmarks.report._bootstrap_aggregate import (
        BOOTSTRAP_DEFAULT_SEED,
        BOOTSTRAP_PER_FOLD_SEED_OFFSET,
    )

    call_count = 0
    # B16 aggregator call order (per the actual implementation):
    # (1) pooled main delta_loss, (2..N) per-fold for main delta
    # in ascending fold_index, (N+1) oracle delta (when oracle
    # data present). Per-fold runs BETWEEN main pool and oracle
    # pool, NOT after both. The per-fold calls live inside
    # `_bootstrap_aggregate.compute_per_fold_cis`, so we patch
    # BOTH module sites.
    POOLED_CALLS_BEFORE_PER_FOLD = 1  # main only

    def _capture(*args: object, **kwargs: object):
        nonlocal call_count
        call_count += 1
        seed = int(kwargs["seed"])  # type: ignore[arg-type]
        # Capture the next 3 calls after the main pool as fold 0/1/2.
        if POOLED_CALLS_BEFORE_PER_FOLD < call_count <= POOLED_CALLS_BEFORE_PER_FOLD + 3:
            fold_index = call_count - POOLED_CALLS_BEFORE_PER_FOLD - 1
            seeds_captured[fold_index] = seed
        return _real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(_module, "entity_block_bootstrap_ci", _capture)
    import benchmarks.report._bootstrap_aggregate as _agg

    monkeypatch.setattr(_agg, "entity_block_bootstrap_ci", _capture)

    _run_ensemble_lift(tmp_path, monkeypatch, per_fold_enabled=True, cells=_happy_cells(n_folds=3))

    for fold_index in (0, 1, 2):
        expected = BOOTSTRAP_DEFAULT_SEED ^ BOOTSTRAP_PER_FOLD_SEED_OFFSET ^ fold_index
        assert seeds_captured[fold_index] == expected


# =============================================================================
# 11. Zero-row fold omitted (R-B22-7)
# =============================================================================


def test_aggregator_per_fold_cis_omits_zero_row_folds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fixture with cells only on fold 0 + fold 1; fold 2 has no
    cells. Assert per_fold_cis contains 2 entries with fold_index
    0 and 1; no fold-2 entry."""
    # Cells on folds 0 and 1 only (no fold 2).
    cells = _happy_cells(n_seeds=2, n_folds=2)
    rollup = _run_ensemble_lift(
        tmp_path,
        monkeypatch,
        per_fold_enabled=True,
        cells=cells,
        n_seeds=2,
        n_folds=2,
    )
    row = rollup[0]
    assert row.per_fold_cis is not None
    fold_indices_present = [fc.fold_index for fc in row.per_fold_cis]
    assert fold_indices_present == [0, 1]
    assert 2 not in fold_indices_present


# =============================================================================
# 12. Per-fold OOM gate (R-B22-6)
# =============================================================================


def test_aggregator_per_fold_cis_oom_gate_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lower BOOTSTRAP_ROW_COUNT_CEILING to 1 on a fixture with 4
    cells + n_resamples=2. Each fold has 2 cells * 2 resamples =
    4 > 1, tripping the per-fold gate (the pooled gate also
    trips because 4*2=8 > 1; both raise the same way but the
    per-fold message names the fold_index)."""
    from benchmarks.report import _bootstrap_aggregate as _agg
    from benchmarks.report.bootstrap_rollup import RawRollupError

    # The pooled gate fires first. To exercise the PER-FOLD gate
    # path, raise the ceiling enough that the pooled gate passes
    # (4 cells * 2 resamples = 8 <= 8) but each per-fold call
    # would trip (2 cells * 2 resamples = 4 > 3). The per-fold
    # raise message names fold_index.
    monkeypatch.setattr(_agg, "BOOTSTRAP_ROW_COUNT_CEILING", 8)
    # Pool: 4 cells * 2 = 8 == ceiling, NOT > ceiling -> passes.
    # Per fold: 2 cells * 2 = 4, ceiling now needs to be < 4.
    # Mutate the ceiling between the pooled call and the per-fold
    # call via a side-effecting wrapper.
    import benchmarks.report.bootstrap_ensemble_lift as _module
    from benchmarks.metrics.bootstrap import (
        entity_block_bootstrap_ci as _real,
    )

    def _wrapper(*args: object, **kwargs: object):
        # After the pooled call, drop the ceiling so the per-fold
        # gate fires.
        _agg.BOOTSTRAP_ROW_COUNT_CEILING = 3
        return _real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(_module, "entity_block_bootstrap_ci", _wrapper)

    with pytest.raises(RawRollupError, match=r"per-fold CI"):
        _run_ensemble_lift(
            tmp_path,
            monkeypatch,
            per_fold_enabled=True,
            cells=_happy_cells(n_seeds=2, n_folds=2),
            bootstrap_n_resamples=2,
        )


# =============================================================================
# 13. Sentinel row carries per_fold_cis=None even with flag on (R-B22-8)
# =============================================================================


def test_aggregator_sentinel_row_carries_per_fold_cis_none_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """qa-R1-I2 closure: fixture with the per-fold flag ON AND the
    aggregator routes through the sentinel emit (no cells). Assert
    per_fold_cis is None (NOT empty list)."""
    rollup = _run_ensemble_lift(
        tmp_path,
        monkeypatch,
        per_fold_enabled=True,
        cells=(),
        n_seeds=2,
        n_folds=2,
    )
    assert rollup[0].per_fold_cis is None


# =============================================================================
# 14. Single-entity fold degenerate path
# =============================================================================


def test_aggregator_per_fold_cis_single_entity_fold_degenerates_correctly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fold with exactly one entity (the B16 cell-as-entity
    contract: one cell = one entity) degenerates through the
    primitive's `n_entities <= 1` path, returning
    `(mean, mean, mean, None)`. The corresponding FoldCI has
    `metric_mean == metric_ci_lo == metric_ci_hi` AND
    `ci_fallback_reason is None`."""
    # 1 cell per fold => n_entities=1 per fold => degenerate.
    rollup = _run_ensemble_lift(
        tmp_path,
        monkeypatch,
        per_fold_enabled=True,
        cells=_happy_cells(n_seeds=1, n_folds=2),
        n_seeds=1,
        n_folds=2,
    )
    row = rollup[0]
    assert row.per_fold_cis is not None
    for fc in row.per_fold_cis:
        assert fc.metric_mean == fc.metric_ci_lo == fc.metric_ci_hi
        assert fc.ci_fallback_reason is None


# =============================================================================
# 15. Order-stability: ascending by fold_index (qa-R1-C1)
# =============================================================================


def test_aggregator_per_fold_cis_list_is_sorted_by_fold_index_ascending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Build a fixture where cells are emitted in non-ascending
    fold order (fold 2, fold 0, fold 1). Assert per_fold_cis is
    sorted ascending."""
    cells = (
        PerCellLiftDelta(seed=0, fold_index=2, loss_gbm=0.6, loss_gbm_plus_seq=0.4, delta_loss=0.2),
        PerCellLiftDelta(seed=1, fold_index=2, loss_gbm=0.6, loss_gbm_plus_seq=0.4, delta_loss=0.2),
        PerCellLiftDelta(seed=0, fold_index=0, loss_gbm=0.6, loss_gbm_plus_seq=0.4, delta_loss=0.2),
        PerCellLiftDelta(seed=1, fold_index=0, loss_gbm=0.6, loss_gbm_plus_seq=0.4, delta_loss=0.2),
        PerCellLiftDelta(seed=0, fold_index=1, loss_gbm=0.6, loss_gbm_plus_seq=0.4, delta_loss=0.2),
        PerCellLiftDelta(seed=1, fold_index=1, loss_gbm=0.6, loss_gbm_plus_seq=0.4, delta_loss=0.2),
    )
    rollup = _run_ensemble_lift(
        tmp_path,
        monkeypatch,
        per_fold_enabled=True,
        cells=cells,
        n_seeds=2,
        n_folds=3,
    )
    row = rollup[0]
    assert row.per_fold_cis is not None
    assert [fc.fold_index for fc in row.per_fold_cis] == [0, 1, 2]


# =============================================================================
# 16. n_resamples propagation (qa-R1-C2)
# =============================================================================


def test_aggregator_per_fold_cis_each_fold_receives_correct_n_resamples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Capture n_resamples on every per-fold call; assert each
    captured value equals the spec's bootstrap_n_resamples value."""
    n_resamples_captured: list[int] = []

    import benchmarks.report.bootstrap_ensemble_lift as _module
    from benchmarks.metrics.bootstrap import (
        entity_block_bootstrap_ci as _real,
    )

    def _capture(*args: object, **kwargs: object):
        n_resamples_captured.append(int(kwargs["n_resamples"]))  # type: ignore[arg-type]
        return _real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(_module, "entity_block_bootstrap_ci", _capture)
    import benchmarks.report._bootstrap_aggregate as _agg

    monkeypatch.setattr(_agg, "entity_block_bootstrap_ci", _capture)

    _run_ensemble_lift(
        tmp_path,
        monkeypatch,
        per_fold_enabled=True,
        cells=_happy_cells(n_folds=3),
        bootstrap_n_resamples=137,
    )

    # B16 call order: 2 pooled (main + oracle) + 3 per-fold = 5
    # total. Every call MUST use n_resamples=137 from the spec.
    assert len(n_resamples_captured) == 5
    assert all(n == 137 for n in n_resamples_captured)


# =============================================================================
# 17. Non-B5 (ensemble_lift) aggregator on-path (qa-R1-I1)
# =============================================================================


def test_non_b5_aggregator_writes_per_fold_cis_list_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Closes qa-R1-I1: the existing aggregator under test in #8
    above IS the B16 ensemble_lift aggregator, not B5. This test
    re-asserts the same contract under the explicit non-B5
    framing so a coder reading this file by section heading sees
    the closure."""
    rollup = _run_ensemble_lift(tmp_path, monkeypatch, per_fold_enabled=True, cells=_happy_cells())
    row = rollup[0]
    assert isinstance(row, EnsembleLiftRollupRow)
    assert row.per_fold_cis is not None
    for fc in row.per_fold_cis:
        assert isinstance(fc, FoldCI)


# =============================================================================
# 18. Non-B5 schema parquet round-trip (qa-R1-I3)
# =============================================================================


def test_non_b5_rollup_row_per_fold_cis_round_trip(tmp_path: Path) -> None:
    """Closes qa-R1-I3: write + load a PairwiseRollupRow with a
    2-entry per_fold_cis list and assert every FoldCI field
    survives."""
    from benchmarks.bootstrap_manifest import (
        load_pairwise_rollup,
        write_pairwise_rollup,
    )

    fold0 = FoldCI(
        fold_index=0,
        n_seeds=2,
        n_entities=4,
        metric_mean=0.30,
        metric_ci_lo=0.25,
        metric_ci_hi=0.35,
        ci_method="bca",
        ci_fallback_reason=None,
    )
    fold1 = FoldCI(
        fold_index=1,
        n_seeds=2,
        n_entities=4,
        metric_mean=0.40,
        metric_ci_lo=0.35,
        metric_ci_hi=0.45,
        ci_method="percentile",
        ci_fallback_reason="a_overshoot",
    )
    row = PairwiseRollupRow(
        dataset_name="ds",
        model_a="a",
        model_b="b",
        task_type="binary",
        primary_metric="complementarity_score",
        n_seeds=2,
        n_cells_evaluated=8,
        n_skipped_cells=0,
        primary_metric_mean=0.35,
        primary_metric_ci_lo=0.30,
        primary_metric_ci_hi=0.40,
        bootstrap_seed=42,
        bootstrap_n_resamples=100,
        bootstrap_numpy_version="2.0.0",
        manifest_fingerprint="f" * 64,
        per_fold_cis=[fold0, fold1],
    )
    write_pairwise_rollup(tmp_path, [row])
    loaded = load_pairwise_rollup(tmp_path)
    assert len(loaded) == 1
    assert loaded[0].per_fold_cis == [fold0, fold1]


# =============================================================================
# 19. B16: per_fold_cis populated; schema does NOT carry oracle per-fold
# =============================================================================


def test_b16_aggregator_per_fold_cis_does_not_populate_oracle_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """qa-R2-I2 closure: at v1 the B16 aggregator computes per-fold
    CIs for the MAIN delta only. Oracle per-fold is deferred under
    D-B22.2. Assert:
    (a) per_fold_cis is not None on the emitted row;
    (b) the schema does NOT carry an attribute named
        per_fold_oracle_cis (D-B22.2 scope pin)."""
    rollup = _run_ensemble_lift(tmp_path, monkeypatch, per_fold_enabled=True, cells=_happy_cells())
    row = rollup[0]
    assert row.per_fold_cis is not None
    assert not hasattr(row, "per_fold_oracle_cis")
    assert "per_fold_oracle_cis" not in EnsembleLiftRollupRow.model_fields
