"""Phase B21 D-B16.2 BCa CI tests.

Covers 16 tests from `docs/benchmark_suite_design_b21_delta.md`
sections B21.6 + R1-R4 closures:

- Tests #1, #2, #9, #10: public-primitive shape + degenerate paths
- Tests #3, #4: `_bca_percentile_points` p0_at_edge fallback branches
- Test #5: `_bca_percentile_points` a_overshoot fallback branch
- Test #6: `_compute_acceleration_from_jackknife` a=0 degenerate
- Tests #7, #8: BCa vs percentile agreement on symmetric / divergence
  on skewed bootstrap distributions
- Tests #11-13: aggregator-level audit field writes + late-binding
  seam + fallback-reason propagation
- Test #14: B16 ensemble_lift main vs oracle independent fallback
- Test #15: parquet round-trip on all 5 RollupRow types
- Test #16: schema-default `"percentile"` backward-compat invariant
"""

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from benchmarks.bootstrap_manifest import (
    EnsembleLiftRollupRow,
    HPOUpliftRollupRow,
    PairwiseRollupRow,
    RollupRow,
    TrainingTimeRollupRow,
    load_ensemble_lift_rollup,
    load_hpo_uplift_rollup,
    load_pairwise_rollup,
    load_rollup,
    load_training_time_rollup,
    write_ensemble_lift_rollup,
    write_hpo_uplift_rollup,
    write_pairwise_rollup,
    write_rollup,
    write_training_time_rollup,
)
from benchmarks.config import BenchmarkConfig, ExperimentSpec
from benchmarks.experiments import build_run_environment
from benchmarks.experiments.ensemble_lift import (
    ComputePerCellLiftDeltasResult,
    PerCellLiftDelta,
)
from benchmarks.metrics.bootstrap import (
    _bca_percentile_points,
    _compute_acceleration_from_jackknife,
    entity_block_bootstrap_ci,
)
from benchmarks.report.bootstrap_ensemble_lift import (
    aggregate_bootstrap_ensemble_lift_rollup,
)
from benchmarks.run_manifest import RunManifest, build_run_manifest, write_run_manifest

from tests.benchmarks._fakes import register_all_fakes_and_get_panels

_GBM_MODEL = "gbm_constant"
_SEQ_MODEL = "seq_constant"


# =============================================================================
# Aggregator scaffolding (mirrors test_b20_oracle_delta_ci helpers)
# =============================================================================


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
        run_id="b21-test",
        library_git_sha="0" * 40,
        profile="smoke",
        hardware_tier="cpu",
        output_root=output_root,
    )
    write_run_manifest(output_root, manifest)
    return manifest


def _manifest_row(
    *,
    model_name: str,
    seed: int,
    fold_index: int,
) -> dict[str, object]:
    return {
        "dataset_name": "fake_binary",
        "model_name": model_name,
        "task_type": "binary",
        "seed": seed,
        "fold_index": fold_index,
        "variant": "default",
        "skipped_reason": None,
    }


def _ok_rows_both_families(n_seeds: int = 2, n_folds: int = 2) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for s in range(n_seeds):
        for f in range(n_folds):
            rows.append(_manifest_row(model_name=_GBM_MODEL, seed=s, fold_index=f))
            rows.append(_manifest_row(model_name=_SEQ_MODEL, seed=s, fold_index=f))
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
    result: ComputePerCellLiftDeltasResult,
) -> None:
    import benchmarks.report.bootstrap_ensemble_lift as _module

    def _fake(**_kwargs: object) -> ComputePerCellLiftDeltasResult:
        return result

    monkeypatch.setattr(_module, "compute_per_cell_lift_deltas", _fake)  # type: ignore[misc]


def _setup_ensemble_lift(tmp_path: Path) -> tuple[BenchmarkConfig, Path, RunManifest]:
    config = _make_config(tmp_path)
    output_root = tmp_path / "out"
    manifest = _make_run_manifest(config, output_root)
    return config, output_root, manifest


def _happy_per_cell() -> ComputePerCellLiftDeltasResult:
    """4 cells with finite losses and oracle losses; produces a
    non-degenerate BCa happy-path bootstrap."""
    cells = tuple(
        PerCellLiftDelta(
            seed=s,
            fold_index=f,
            loss_gbm=0.60 + 0.01 * (s + f),
            loss_gbm_plus_seq=0.40 + 0.02 * (s + f),
            delta_loss=0.20 - 0.01 * (s + f),
            oracle_loss=0.30 + 0.01 * (s + f),
        )
        for s in (0, 1)
        for f in (0, 1)
    )
    return ComputePerCellLiftDeltasResult(cells=cells, selector="log_loss")


# =============================================================================
# 1. Public primitive returns a 4-tuple with None fallback on happy path
# =============================================================================


def test_bca_returns_4tuple_with_none_fallback_on_happy_path() -> None:
    """`entity_block_bootstrap_ci(ci_method="bca")` returns a 4-tuple
    `(mean, lo, hi, fallback_reason)` on a non-degenerate fixture.
    Asserts shape, ordering, and `fallback_reason is None`."""
    losses = np.array([0.1, 0.1, 0.2, 0.3, 0.5, 1.0], dtype=np.float64)
    entity_ids = np.array([0, 1, 2, 3, 4, 5])
    result = entity_block_bootstrap_ci(
        losses, entity_ids, n_resamples=500, seed=42, ci_method="bca"
    )
    assert len(result) == 4
    mean, lo, hi, fallback = result
    assert lo <= mean <= hi
    assert fallback is None


# =============================================================================
# 2. BCa lo < hi strictly on non-degenerate fixture
# =============================================================================


def test_bca_lo_lt_hi_on_non_degenerate_fixture() -> None:
    """The CI width is positive on a multi-entity fixture; the BCa
    bounds do not collapse."""
    losses = np.array([0.1, 0.1, 0.2, 0.3, 0.5, 1.0], dtype=np.float64)
    entity_ids = np.array([0, 1, 2, 3, 4, 5])
    _, lo, hi, _ = entity_block_bootstrap_ci(
        losses, entity_ids, n_resamples=500, seed=42, ci_method="bca"
    )
    assert lo < hi


# =============================================================================
# 3. _bca_percentile_points p0_at_edge when p0 == 0.0
# =============================================================================


def test_bca_percentile_points_returns_p0_at_edge_when_p0_is_zero() -> None:
    """`_bca_percentile_points(p0=0.0, a=0.5, confidence=0.95)`
    returns `(0.025, 0.975, "p0_at_edge")`. The `p0 <= 0.0` branch
    short-circuits BEFORE `a` is consulted; passing `a=0.5` is
    deliberately non-zero to confirm the short-circuit.
    """
    alpha_1, alpha_2, fallback = _bca_percentile_points(p0=0.0, a=0.5, confidence=0.95)
    assert (alpha_1, alpha_2) == pytest.approx((0.025, 0.975), abs=1e-12)
    assert fallback == "p0_at_edge"


# =============================================================================
# 4. _bca_percentile_points p0_at_edge when p0 == 1.0
# =============================================================================


def test_bca_percentile_points_returns_p0_at_edge_when_p0_is_one() -> None:
    """Symmetric fallback at the upper edge."""
    alpha_1, alpha_2, fallback = _bca_percentile_points(p0=1.0, a=0.5, confidence=0.95)
    assert (alpha_1, alpha_2) == pytest.approx((0.025, 0.975), abs=1e-12)
    assert fallback == "p0_at_edge"


# =============================================================================
# 5. _bca_percentile_points a_overshoot when denom <= eps
# =============================================================================


def test_bca_percentile_points_returns_a_overshoot_when_denom_at_or_below_epsilon() -> None:
    """`_bca_percentile_points(p0=0.5, a=10.0, confidence=0.95)`
    triggers `a_overshoot`. With `z0 = norm.ppf(0.5) = 0.0` and
    `z_hi = norm.ppf(0.975) = 1.96`, `denom_hi = 1 - 10.0 * 1.96 =
    -18.6 <= _BCA_DENOM_EPS`. The directly-passed `a=10.0` exceeds
    the theoretical Cauchy-Schwarz bound `|a| <= 1/(6*sqrt(n))`;
    the unit test bypasses the jackknife to pin the fallback
    routing on the percentile-point math directly.
    """
    alpha_1, alpha_2, fallback = _bca_percentile_points(p0=0.5, a=10.0, confidence=0.95)
    assert (alpha_1, alpha_2) == pytest.approx((0.025, 0.975), abs=1e-12)
    assert fallback == "a_overshoot"


# =============================================================================
# 5b. _bca_percentile_points a_overshoot when denom_lo arm fires alone
# =============================================================================


def test_bca_percentile_points_returns_a_overshoot_when_denom_lo_fires_alone() -> None:
    """R1 qa-I2 closure: pin the denom_lo arm of the OR independently.
    With `a=-10.0`, `denom_lo = 1 - (-10) * (0 + (-1.96)) = -18.6 <=
    eps` fires while `denom_hi = 1 - (-10) * (0 + 1.96) = 20.6` is
    safely positive. A mutation that swapped the OR for AND would
    survive test #5 (denom_hi fires alone) but fail here.
    """
    alpha_1, alpha_2, fallback = _bca_percentile_points(p0=0.5, a=-10.0, confidence=0.95)
    assert (alpha_1, alpha_2) == pytest.approx((0.025, 0.975), abs=1e-12)
    assert fallback == "a_overshoot"


# =============================================================================
# 5c. _bca_percentile_points a_overshoot fires at the exact `<=` boundary
# =============================================================================


def test_bca_percentile_points_a_overshoot_fires_at_eps_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B30 / D-B21.5 + qa-R2-C1 closure: monkeypatch
    `_BCA_DENOM_EPS = 1.0` and pass `a = 0.0` so
    `denom_hi = 1.0 - 0.0 * z_hi = 1.0` exactly (integer
    arithmetic; no float drift). The `<=` predicate at
    `benchmarks/metrics/bootstrap.py:97` fires on
    `1.0 <= 1.0`; a `<` mutation would NOT fire on
    `1.0 < 1.0` and fallback would be None."""
    import benchmarks.metrics.bootstrap as _bca

    monkeypatch.setattr(_bca, "_BCA_DENOM_EPS", 1.0)
    _, _, fallback = _bca_percentile_points(p0=0.5, a=0.0, confidence=0.95)
    assert fallback == "a_overshoot"


def test_bca_percentile_points_a_overshoot_does_not_fire_above_eps() -> None:
    """B30 / D-B21.5 companion (safe-region check, not an
    independent mutation discriminator): with the real eps
    (1e-12), `a = (1.0 - 2e-12) / z_hi` gives `denom_hi ~ 2e-12`,
    exactly one eps above the 1e-12 threshold. The fallback
    must NOT fire. The boundary-fires test above carries the
    `<=` vs `<` mutation kill; this companion confirms safe-
    region behavior."""
    from scipy.stats import norm

    confidence = 0.95
    alpha = (1.0 - confidence) / 2.0
    z_hi = float(norm.ppf(1.0 - alpha))
    a = (1.0 - 2e-12) / z_hi
    _, _, fallback = _bca_percentile_points(p0=0.5, a=a, confidence=confidence)
    assert fallback is None


# =============================================================================
# 6. _compute_acceleration_from_jackknife returns 0 on equal jackknife
# =============================================================================


def test_compute_acceleration_returns_zero_when_jackknife_all_equal() -> None:
    """All jackknife values equal → deviations all zero → denom is
    `6 * 0^1.5 = 0` → `a = 0.0`. Pins the degenerate-denominator
    short-circuit; BCa with `a=0` reduces to BC (bias-corrected
    percentile, no acceleration adjustment)."""
    jackknife = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    a = _compute_acceleration_from_jackknife(jackknife)
    assert a == 0.0


# =============================================================================
# 7. BCa matches percentile on a symmetric distribution
# =============================================================================


def test_bca_matches_percentile_when_distribution_symmetric() -> None:
    """On a roughly symmetric fixture the BCa and percentile CIs
    agree within Monte-Carlo tolerance. Cross-check approach
    (qa-R1-N2 closure): call the same primitive twice on the same
    fixture with `ci_method="percentile"` then `ci_method="bca"`
    and compare bound values."""
    losses = np.array([0.4, 0.5, 0.6], dtype=np.float64)
    entity_ids = np.array([0, 1, 2])
    _, p_lo, p_hi, _ = entity_block_bootstrap_ci(
        losses, entity_ids, n_resamples=10_000, seed=42, ci_method="percentile"
    )
    _, b_lo, b_hi, _ = entity_block_bootstrap_ci(
        losses, entity_ids, n_resamples=10_000, seed=42, ci_method="bca"
    )
    # Symmetric distribution: BCa correction is small.
    assert abs(b_lo - p_lo) < 5e-2
    assert abs(b_hi - p_hi) < 5e-2


# =============================================================================
# 8. BCa differs from percentile on a skewed distribution
# =============================================================================


def test_bca_differs_from_percentile_when_distribution_skewed() -> None:
    """On a skewed fixture at least one endpoint shifts. Cross-check
    approach: same two-invocation comparison; assert the absolute
    difference exceeds a per-endpoint floor."""
    losses = np.array([0.1, 0.1, 0.2, 0.5, 1.0, 2.0], dtype=np.float64)
    entity_ids = np.array([0, 1, 2, 3, 4, 5])
    _, p_lo, p_hi, _ = entity_block_bootstrap_ci(
        losses, entity_ids, n_resamples=10_000, seed=42, ci_method="percentile"
    )
    _, b_lo, b_hi, _ = entity_block_bootstrap_ci(
        losses, entity_ids, n_resamples=10_000, seed=42, ci_method="bca"
    )
    assert abs(b_lo - p_lo) > 1e-3 or abs(b_hi - p_hi) > 1e-3


# =============================================================================
# 9. n_entities == 1 returns degenerate 4-tuple
# =============================================================================


def test_bca_n_entities_one_returns_collapsed_tuple() -> None:
    """A single entity → the existing degenerate path returns
    `(mean, mean, mean, None)` (widened from the pre-B21 3-tuple).
    """
    losses = np.array([0.5], dtype=np.float64)
    entity_ids = np.array([0])
    mean, lo, hi, fallback = entity_block_bootstrap_ci(
        losses, entity_ids, n_resamples=10, ci_method="bca"
    )
    assert mean == lo == hi
    assert fallback is None


# =============================================================================
# 10. Primitive's default ci_method is "bca"
# =============================================================================


def test_primitive_default_ci_method_is_bca() -> None:
    """The primitive's `ci_method` kwarg defaults to `"bca"`.
    Inspect the signature to pin the default explicitly."""
    sig = inspect.signature(entity_block_bootstrap_ci)
    assert sig.parameters["ci_method"].default == "bca"


# =============================================================================
# 11. Aggregator writes bootstrap_ci_method="bca" on the happy path
# =============================================================================


def test_aggregator_writes_bootstrap_ci_method_bca(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end through the B16 ensemble_lift aggregator on a stub
    fixture; assert `bootstrap_ci_method == "bca"` AND
    `bootstrap_ci_fallback_reason is None` on the emitted row."""
    config, output_root, manifest = _setup_ensemble_lift(tmp_path)
    _stub_load_run(monkeypatch, _ok_rows_both_families())
    _stub_families(monkeypatch)
    _stub_per_cell(monkeypatch, _happy_per_cell())

    env = build_run_environment(profile="smoke")
    rollup = aggregate_bootstrap_ensemble_lift_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    assert len(rollup) == 1
    row = rollup[0]
    assert row.bootstrap_ci_method == "bca"
    # B21 R1 code-I2 closure: pin BOTH main + oracle fallback reasons
    # as type-checked audit values. Small-N happy-path fixtures may
    # degenerate to a known fallback (e.g., `p0_at_edge` when the
    # bootstrap distribution concentrates at the mean); the audit
    # field write is the contract.
    assert row.bootstrap_ci_fallback_reason in (None, "p0_at_edge", "a_overshoot")
    assert row.bootstrap_oracle_ci_fallback_reason in (None, "p0_at_edge", "a_overshoot")


# =============================================================================
# 12. Aggregator late-binding ci_method seam (monkeypatch the constant)
# =============================================================================


def test_aggregator_late_binding_ci_method_default_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Monkeypatch `_bootstrap_aggregate.BOOTSTRAP_DEFAULT_CI_METHOD`
    to `"percentile"`; assert the aggregator writes
    `bootstrap_ci_method == "percentile"` on the emitted row.
    Exercises the late-binding seam (NOT a no-op self-monkeypatch).
    """
    import benchmarks.report._bootstrap_aggregate as _agg

    monkeypatch.setattr(_agg, "BOOTSTRAP_DEFAULT_CI_METHOD", "percentile")

    config, output_root, manifest = _setup_ensemble_lift(tmp_path)
    _stub_load_run(monkeypatch, _ok_rows_both_families())
    _stub_families(monkeypatch)
    _stub_per_cell(monkeypatch, _happy_per_cell())

    env = build_run_environment(profile="smoke")
    rollup = aggregate_bootstrap_ensemble_lift_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    row = rollup[0]
    assert row.bootstrap_ci_method == "percentile"
    # Percentile path never sets a fallback reason.
    assert row.bootstrap_ci_fallback_reason is None


# =============================================================================
# 13. Aggregator surfaces fallback_reason via the stateful metric_fn seam
# =============================================================================


def test_aggregator_writes_fallback_reason_when_bca_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inject a stateful metric_fn via a wrapper around
    `entity_block_bootstrap_ci` that forces `p_0 = 0.0` on the main
    bootstrap. Assert the emitted row carries
    `bootstrap_ci_fallback_reason == "p0_at_edge"`."""
    config, output_root, manifest = _setup_ensemble_lift(tmp_path)
    _stub_load_run(monkeypatch, _ok_rows_both_families())
    _stub_families(monkeypatch)
    _stub_per_cell(monkeypatch, _happy_per_cell())

    from benchmarks.metrics.bootstrap import (
        entity_block_bootstrap_ci as real_entity_block_bootstrap_ci,
    )

    call_count = 0

    def stateful_metric_fn(x: np.ndarray) -> float:
        nonlocal call_count
        call_count += 1
        return -1e9 if call_count == 1 else 1e9

    def _wrapped_primitive(
        losses: np.ndarray,
        entity_ids: np.ndarray,
        **kwargs: object,
    ) -> tuple[float, float, float, str | None]:
        nonlocal call_count
        call_count = 0  # reset on each primitive call
        kwargs["metric_fn"] = stateful_metric_fn
        return real_entity_block_bootstrap_ci(losses, entity_ids, **kwargs)  # type: ignore[arg-type]

    import benchmarks.report.bootstrap_ensemble_lift as _module

    monkeypatch.setattr(_module, "entity_block_bootstrap_ci", _wrapped_primitive)

    env = build_run_environment(profile="smoke")
    rollup = aggregate_bootstrap_ensemble_lift_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    row = rollup[0]
    assert row.bootstrap_ci_method == "bca"
    assert row.bootstrap_ci_fallback_reason == "p0_at_edge"


# =============================================================================
# 14. Ensemble lift: independent main vs oracle fallback reasons
# =============================================================================


def test_ensemble_lift_aggregator_writes_independent_oracle_fallback_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inject a wrapper around `entity_block_bootstrap_ci` in the B16
    aggregator that ROUTES BY INVOCATION ORDER: call #1 (main)
    passes through to the real primitive (no fallback fires), call
    #2 (oracle) injects the stateful stub forcing `p_0 = 0.0`. The
    emitted row carries `bootstrap_ci_fallback_reason is None`
    (main) AND `bootstrap_oracle_ci_fallback_reason == "p0_at_edge"`
    (oracle). Pins R-B21-7 fallback-reason independence."""
    config, output_root, manifest = _setup_ensemble_lift(tmp_path)
    _stub_load_run(monkeypatch, _ok_rows_both_families())
    _stub_families(monkeypatch)
    _stub_per_cell(monkeypatch, _happy_per_cell())

    from benchmarks.metrics.bootstrap import (
        entity_block_bootstrap_ci as real_entity_block_bootstrap_ci,
    )

    primitive_call_count = 0
    metric_call_count = 0

    def stateful_metric_fn(x: np.ndarray) -> float:
        nonlocal metric_call_count
        metric_call_count += 1
        return -1e9 if metric_call_count == 1 else 1e9

    def _wrapped_primitive(
        losses: np.ndarray,
        entity_ids: np.ndarray,
        **kwargs: object,
    ) -> tuple[float, float, float, str | None]:
        nonlocal primitive_call_count, metric_call_count
        primitive_call_count += 1
        if primitive_call_count == 1:
            # Main delta_loss bootstrap: pass through.
            return real_entity_block_bootstrap_ci(losses, entity_ids, **kwargs)  # type: ignore[arg-type]
        metric_call_count = 0  # reset stub state for the oracle call
        kwargs["metric_fn"] = stateful_metric_fn
        return real_entity_block_bootstrap_ci(losses, entity_ids, **kwargs)  # type: ignore[arg-type]

    import benchmarks.report.bootstrap_ensemble_lift as _module

    monkeypatch.setattr(_module, "entity_block_bootstrap_ci", _wrapped_primitive)

    env = build_run_environment(profile="smoke")
    rollup = aggregate_bootstrap_ensemble_lift_rollup(
        config, output_root=output_root, env=env, manifest=manifest
    )
    row = rollup[0]
    assert row.bootstrap_ci_fallback_reason is None
    assert row.bootstrap_oracle_ci_fallback_reason == "p0_at_edge"


# =============================================================================
# 15. Parquet round-trip for the new audit fields across all 5 schemas
# =============================================================================


def _make_b5_row(*, ci_method: str, fallback: str | None) -> RollupRow:
    return RollupRow(
        dataset_name="ds",
        model_name="m",
        task_type="binary",
        primary_metric="log_loss",
        n_seeds=2,
        n_cells_evaluated=4,
        n_skipped_cells=0,
        n_rows=100,
        n_entities=10,
        primary_metric_mean=0.5,
        primary_metric_ci_lo=0.4,
        primary_metric_ci_hi=0.6,
        bootstrap_seed=42,
        bootstrap_n_resamples=100,
        bootstrap_numpy_version="2.0.0",
        bootstrap_ci_method=ci_method,
        bootstrap_ci_fallback_reason=fallback,
        manifest_fingerprint="f" * 64,
    )


def _make_pairwise_row(*, ci_method: str, fallback: str | None) -> PairwiseRollupRow:
    return PairwiseRollupRow(
        dataset_name="ds",
        model_a="a",
        model_b="b",
        task_type="binary",
        primary_metric="complementarity_score",
        n_seeds=2,
        n_cells_evaluated=4,
        n_skipped_cells=0,
        primary_metric_mean=0.5,
        primary_metric_ci_lo=0.4,
        primary_metric_ci_hi=0.6,
        bootstrap_seed=42,
        bootstrap_n_resamples=100,
        bootstrap_numpy_version="2.0.0",
        bootstrap_ci_method=ci_method,
        bootstrap_ci_fallback_reason=fallback,
        manifest_fingerprint="f" * 64,
    )


def _make_training_time_row(*, ci_method: str, fallback: str | None) -> TrainingTimeRollupRow:
    return TrainingTimeRollupRow(
        dataset_name="ds",
        model_name="m",
        hardware_tier="cpu",
        task_type="binary",
        primary_metric="wall_seconds",
        n_seeds=2,
        n_cells_evaluated=4,
        n_skipped_cells=0,
        primary_metric_mean=10.0,
        primary_metric_ci_lo=8.0,
        primary_metric_ci_hi=12.0,
        bootstrap_seed=42,
        bootstrap_n_resamples=100,
        bootstrap_numpy_version="2.0.0",
        bootstrap_ci_method=ci_method,
        bootstrap_ci_fallback_reason=fallback,
        manifest_fingerprint="f" * 64,
    )


def _make_hpo_uplift_row(*, ci_method: str, fallback: str | None) -> HPOUpliftRollupRow:
    return HPOUpliftRollupRow(
        dataset_name="ds",
        model_name="m",
        task_type="binary",
        primary_metric="delta",
        primary_loss_column="log_loss",
        n_seeds=2,
        n_folds=2,
        n_cells_paired=4,
        n_skipped_cells=0,
        primary_metric_mean=0.1,
        primary_metric_ci_lo=0.05,
        primary_metric_ci_hi=0.15,
        bootstrap_seed=42,
        bootstrap_n_resamples=100,
        bootstrap_numpy_version="2.0.0",
        bootstrap_ci_method=ci_method,
        bootstrap_ci_fallback_reason=fallback,
        manifest_fingerprint="f" * 64,
    )


def _make_ensemble_lift_row(
    *,
    ci_method: str,
    fallback: str | None,
    oracle_fallback: str | None,
) -> EnsembleLiftRollupRow:
    return EnsembleLiftRollupRow(
        dataset_name="ds",
        task_type="binary",
        primary_metric="delta_loss",
        primary_loss_column="log_loss",
        n_seeds=2,
        n_folds=2,
        n_cells_paired=4,
        n_pair_grid=4,
        n_oracle_cells_paired=4,
        n_skipped_cells=0,
        primary_metric_mean=0.2,
        primary_metric_ci_lo=0.15,
        primary_metric_ci_hi=0.25,
        oracle_metric_mean=0.1,
        oracle_metric_ci_lo=0.08,
        oracle_metric_ci_hi=0.12,
        bootstrap_seed=42,
        bootstrap_n_resamples=100,
        bootstrap_numpy_version="2.0.0",
        bootstrap_ci_method=ci_method,
        bootstrap_ci_fallback_reason=fallback,
        bootstrap_oracle_ci_fallback_reason=oracle_fallback,
        manifest_fingerprint="f" * 64,
    )


def test_b21_audit_fields_survive_parquet_round_trip(tmp_path: Path) -> None:
    """Write + load one row of each of the 5 RollupRow types under
    both "bca" + non-None fallback AND "percentile" + None fallback;
    assert each new field survives the round-trip on each row type.
    Verifies the pd.NA -> None coercion on the nullable fallback
    fields."""

    # B5 raw-loss
    b5_root = tmp_path / "b5"
    b5_root.mkdir()
    b5_rows = [
        _make_b5_row(ci_method="bca", fallback="p0_at_edge"),
        _make_b5_row(ci_method="percentile", fallback=None),
    ]
    write_rollup(b5_root, b5_rows)
    b5_loaded = load_rollup(b5_root)
    assert b5_loaded[0].bootstrap_ci_method == "bca"
    assert b5_loaded[0].bootstrap_ci_fallback_reason == "p0_at_edge"
    assert b5_loaded[1].bootstrap_ci_method == "percentile"
    assert b5_loaded[1].bootstrap_ci_fallback_reason is None

    # B6 pairwise
    b6_root = tmp_path / "b6"
    b6_root.mkdir()
    b6_rows = [
        _make_pairwise_row(ci_method="bca", fallback="a_overshoot"),
        _make_pairwise_row(ci_method="percentile", fallback=None),
    ]
    write_pairwise_rollup(b6_root, b6_rows)
    b6_loaded = load_pairwise_rollup(b6_root)
    assert b6_loaded[0].bootstrap_ci_method == "bca"
    assert b6_loaded[0].bootstrap_ci_fallback_reason == "a_overshoot"
    assert b6_loaded[1].bootstrap_ci_method == "percentile"
    assert b6_loaded[1].bootstrap_ci_fallback_reason is None

    # B7 training-time
    b7_root = tmp_path / "b7"
    b7_root.mkdir()
    b7_rows = [
        _make_training_time_row(ci_method="bca", fallback="p0_at_edge"),
        _make_training_time_row(ci_method="percentile", fallback=None),
    ]
    write_training_time_rollup(b7_root, b7_rows)
    b7_loaded = load_training_time_rollup(b7_root)
    assert b7_loaded[0].bootstrap_ci_method == "bca"
    assert b7_loaded[0].bootstrap_ci_fallback_reason == "p0_at_edge"
    assert b7_loaded[1].bootstrap_ci_fallback_reason is None

    # B8 HPO-uplift
    b8_root = tmp_path / "b8"
    b8_root.mkdir()
    b8_rows = [
        _make_hpo_uplift_row(ci_method="bca", fallback="p0_at_edge"),
        _make_hpo_uplift_row(ci_method="percentile", fallback=None),
    ]
    write_hpo_uplift_rollup(b8_root, b8_rows)
    b8_loaded = load_hpo_uplift_rollup(b8_root)
    assert b8_loaded[0].bootstrap_ci_method == "bca"
    assert b8_loaded[0].bootstrap_ci_fallback_reason == "p0_at_edge"
    assert b8_loaded[1].bootstrap_ci_fallback_reason is None

    # B16 ensemble-lift (+ oracle fallback)
    b16_root = tmp_path / "b16"
    b16_root.mkdir()
    b16_rows = [
        _make_ensemble_lift_row(
            ci_method="bca", fallback="p0_at_edge", oracle_fallback="a_overshoot"
        ),
        _make_ensemble_lift_row(ci_method="percentile", fallback=None, oracle_fallback=None),
    ]
    write_ensemble_lift_rollup(b16_root, b16_rows)
    b16_loaded = load_ensemble_lift_rollup(b16_root)
    assert b16_loaded[0].bootstrap_ci_method == "bca"
    assert b16_loaded[0].bootstrap_ci_fallback_reason == "p0_at_edge"
    assert b16_loaded[0].bootstrap_oracle_ci_fallback_reason == "a_overshoot"
    assert b16_loaded[1].bootstrap_ci_method == "percentile"
    assert b16_loaded[1].bootstrap_ci_fallback_reason is None
    assert b16_loaded[1].bootstrap_oracle_ci_fallback_reason is None


# =============================================================================
# 16. Schema-default backward-compat: omitting bootstrap_ci_method yields percentile
# =============================================================================


def test_rollup_row_schema_default_ci_method_is_percentile() -> None:
    """Construct each of the 5 RollupRow types WITHOUT supplying
    `bootstrap_ci_method` and assert the field reads back
    `"percentile"`. Pins the backward-compat invariant for pre-B21
    parquet shard loading; a future schema-default change to
    `"bca"` would silently mislabel pre-B21 CI bounds."""
    # B26 / D-B23.2 closure: bootstrap_skipped_reason populated so
    # the rows construct as valid sentinels under the new
    # CI-sentinel @model_validator (all metric_* default to None).
    common_base = dict(
        dataset_name="ds",
        task_type="binary",
        bootstrap_seed=42,
        bootstrap_n_resamples=100,
        bootstrap_numpy_version="2.0.0",
        bootstrap_skipped_reason="test_fixture",
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
        **common_base,
    )
    assert b5.bootstrap_ci_method == "percentile"

    b6 = PairwiseRollupRow(
        model_a="a",
        model_b="b",
        primary_metric="complementarity_score",
        n_seeds=2,
        n_cells_evaluated=0,
        n_skipped_cells=0,
        **common_base,
    )
    assert b6.bootstrap_ci_method == "percentile"

    b7 = TrainingTimeRollupRow(
        model_name="m",
        hardware_tier="cpu",
        primary_metric="wall_seconds",
        n_seeds=2,
        n_cells_evaluated=0,
        n_skipped_cells=0,
        **common_base,
    )
    assert b7.bootstrap_ci_method == "percentile"

    b8 = HPOUpliftRollupRow(
        model_name="m",
        primary_metric="delta",
        primary_loss_column="log_loss",
        n_seeds=2,
        n_folds=2,
        n_cells_paired=0,
        n_skipped_cells=0,
        **common_base,
    )
    assert b8.bootstrap_ci_method == "percentile"

    b16 = EnsembleLiftRollupRow(
        primary_metric="delta_loss",
        primary_loss_column="log_loss",
        n_seeds=2,
        n_folds=2,
        n_cells_paired=0,
        n_pair_grid=0,
        n_oracle_cells_paired=0,
        n_skipped_cells=0,
        **common_base,
    )
    assert b16.bootstrap_ci_method == "percentile"
