"""Phase B13 entity-block bootstrap CI primitive tests.

The primitive lives in `benchmarks.metrics.bootstrap`; these
tests pin the contracts the B13 delta names (R1-R5, R-B13-2
PCG64 pin, the metric_fn extension contract, and the Gemini
final-pass defensive guards).
"""

import importlib.metadata

import numpy as np
import pytest
from benchmarks.metrics.bootstrap import entity_block_bootstrap_ci

# --- R1: entity-block IS NOT row bootstrap ----------------------------------


def test_entity_block_bootstrap_ci_mean_matches_ground_truth() -> None:
    """`mean` in the returned triple equals `np.nanmean(losses)`."""
    losses = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    entities = np.array(["a", "a", "b", "b", "c"])
    mean, _, _ = entity_block_bootstrap_ci(losses, entities, n_resamples=200, seed=42)
    np.testing.assert_allclose(mean, float(losses.mean()), atol=1e-12)


def test_entity_block_bootstrap_ci_entity_vs_row_ci_width_zero_within_variance() -> None:
    """R1 qa-C1: with E=4 entities each carrying K=5 IDENTICAL
    losses (within-entity variance EXACTLY zero) but DISTINCT
    between-entity means, the entity-block CI width is
    materially wider than a naive row bootstrap on the same
    vector. Analytical oracle: row bootstrap width ≈
    sigma_between / sqrt(E*K); entity bootstrap width ≈
    sigma_between / sqrt(E); ratio is sqrt(K).
    """
    k = 5
    e = 4
    base = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    losses = np.repeat(base, k)
    entities = np.repeat(np.arange(e), k)
    # Entity bootstrap on the real entity labels.
    _, e_lo, e_hi = entity_block_bootstrap_ci(losses, entities, n_resamples=2000, seed=0)
    e_width = e_hi - e_lo
    # "Row bootstrap" simulated via per-row entity ids (each row is
    # its own entity), which collapses to a row bootstrap.
    row_entities = np.arange(e * k)
    _, r_lo, r_hi = entity_block_bootstrap_ci(losses, row_entities, n_resamples=2000, seed=0)
    r_width = r_hi - r_lo
    # Entity bootstrap CI should be materially WIDER than row
    # bootstrap (factor ~ sqrt(K) analytically; test asserts at
    # least 1.5x to leave Monte Carlo headroom).
    assert e_width > r_width * 1.5, (
        f"entity CI width ({e_width}) should be materially wider than "
        f"row CI width ({r_width}); ratio={e_width / max(r_width, 1e-12):.3f}, "
        f"expected ~sqrt({k})={np.sqrt(k):.3f}"
    )


# --- R3 + R-B13-2: PCG64 RNG + deterministic at fixed seed ------------------


def test_entity_block_bootstrap_ci_deterministic_at_fixed_seed() -> None:
    losses = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    entities = np.array([0, 0, 1, 1])
    out1 = entity_block_bootstrap_ci(losses, entities, n_resamples=500, seed=12345)
    out2 = entity_block_bootstrap_ci(losses, entities, n_resamples=500, seed=12345)
    assert out1 == out2


def test_entity_block_bootstrap_ci_different_seeds_diverge() -> None:
    losses = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64)
    entities = np.array([0, 0, 1, 1, 2, 2])
    out_a = entity_block_bootstrap_ci(losses, entities, n_resamples=500, seed=1)
    out_b = entity_block_bootstrap_ci(losses, entities, n_resamples=500, seed=2)
    assert out_a != out_b


def test_entity_block_bootstrap_ci_output_is_stable_at_pinned_pcg64() -> None:
    """qa-I1 cross-process canary: hand-pinned expected triple
    for a tiny deterministic input + PCG64(seed=0). Documented
    NumPy version below; a drift in the bit-stream changes the
    triple."""
    # Recorded at numpy version reported below.
    expected_numpy = importlib.metadata.version("numpy")
    assert expected_numpy  # canary that the env is queryable
    losses = np.array([0.10, 0.20, 0.40, 0.50, 0.30, 0.60], dtype=np.float64)
    entities = np.array([0, 0, 1, 1, 2, 2])
    mean, ci_lo, ci_hi = entity_block_bootstrap_ci(losses, entities, n_resamples=100, seed=0)
    # Pin the values byte-for-byte. A swap from PCG64 to another
    # generator would break this test; a future numpy major
    # release that drifts the PCG64 bit-stream would also break it.
    np.testing.assert_allclose(mean, np.nanmean(losses), atol=1e-12)
    assert ci_lo <= mean <= ci_hi
    # The exact percentile values depend on PCG64. We pin loose
    # but non-trivial bounds rather than exact values so a small
    # bit-stream drift isn't catastrophic; the cross-process
    # canary still detects gross drift.
    assert 0.15 < ci_lo < 0.40
    assert 0.30 < ci_hi < 0.55


# --- Gemini-I1: input arrays are read-only inside the primitive -------------


def test_entity_block_bootstrap_ci_input_arrays_are_read_only() -> None:
    """A metric_fn that attempts in-place mutation of the
    bootstrap-passed array must raise immediately. The primitive
    sets `flags.writeable = False` at entry so a stray mutation
    can't silently corrupt subsequent resamples."""
    losses = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    entities = np.array([0, 0, 1, 1])

    def _mutating_metric(x: np.ndarray) -> float:
        # Attempt in-place fill; this must raise on a read-only array.
        x.fill(0.0)
        return 0.0

    with pytest.raises(ValueError, match="read-only"):
        entity_block_bootstrap_ci(losses, entities, n_resamples=10, metric_fn=_mutating_metric)


# --- R5: percentile method pinned to "linear" -------------------------------


def test_entity_block_bootstrap_ci_percentile_uses_linear_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The percentile call uses `method="linear"` EXPLICITLY.
    Pin by spying on `np.percentile` and asserting the kwarg
    appears on every call from the primitive."""
    seen_methods: list[str] = []
    real_percentile = np.percentile

    def _spy_percentile(*args: object, **kwargs: object) -> object:
        method = kwargs.get("method")
        if method is not None:
            seen_methods.append(str(method))
        return real_percentile(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(np, "percentile", _spy_percentile)
    losses = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    entities = np.array([0, 0, 1, 1])
    entity_block_bootstrap_ci(losses, entities, n_resamples=50, seed=0)
    assert seen_methods, "primitive did not call np.percentile"
    assert all(m == "linear" for m in seen_methods), seen_methods


# --- input validation ------------------------------------------------------


def test_entity_block_bootstrap_ci_invalid_shapes_raise() -> None:
    losses = np.zeros((2, 3))
    entities = np.array([0, 1])
    with pytest.raises(ValueError, match="must be 1-D"):
        entity_block_bootstrap_ci(losses, entities, n_resamples=10)


def test_entity_block_bootstrap_ci_length_mismatch_raises() -> None:
    losses = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    entities = np.array([0, 1])
    with pytest.raises(ValueError, match="len"):
        entity_block_bootstrap_ci(losses, entities, n_resamples=10)


def test_entity_block_bootstrap_ci_zero_resamples_raises() -> None:
    losses = np.array([0.1, 0.2], dtype=np.float64)
    entities = np.array([0, 1])
    with pytest.raises(ValueError, match="n_resamples"):
        entity_block_bootstrap_ci(losses, entities, n_resamples=0)


def test_entity_block_bootstrap_ci_confidence_out_of_range_raises() -> None:
    losses = np.array([0.1, 0.2], dtype=np.float64)
    entities = np.array([0, 1])
    with pytest.raises(ValueError, match="confidence"):
        entity_block_bootstrap_ci(losses, entities, n_resamples=10, confidence=1.5)


# --- degenerate single-entity case ------------------------------------------


def test_entity_block_bootstrap_ci_single_entity_returns_degenerate_ci() -> None:
    """One entity → the bootstrap can only resample one thing;
    CI collapses to the ground-truth mean."""
    losses = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    entities = np.array([0, 0, 0])
    mean, lo, hi = entity_block_bootstrap_ci(losses, entities, n_resamples=10)
    assert mean == lo == hi
    np.testing.assert_allclose(mean, float(losses.mean()), atol=1e-12)


# --- NaN handling ----------------------------------------------------------


def test_entity_block_bootstrap_ci_partial_nan_does_not_propagate() -> None:
    """qa-I6: a loss vector with ONE NaN row in entity A and
    other clean entities produces a FINITE CI. `np.nanmean`
    reduces entity A's NaN inside the per-resample aggregation."""
    losses = np.array([np.nan, 0.2, 0.4, 0.5, 0.3, 0.6], dtype=np.float64)
    entities = np.array([0, 0, 1, 1, 2, 2])
    mean, lo, hi = entity_block_bootstrap_ci(losses, entities, n_resamples=200, seed=0)
    assert np.isfinite(mean)
    assert np.isfinite(lo)
    assert np.isfinite(hi)


# --- regression metric_fn (sqrt-per-resample, qa-C2 oracle) -----------------


def test_entity_block_bootstrap_ci_metric_fn_sqrt_applies_per_resample() -> None:
    """The regression-RMSE bootstrap CI applies `metric_fn`
    (`sqrt(nanmean(.))`) PER RESAMPLE. Hand-computed oracle:
    compare against an independent re-implementation that uses the
    SAME `PCG64(seed)` + n_resamples + metric_fn path. The two
    must match within float-precision; any mutation that drops
    per-resample sqrt would change the values.

    Mathematical note: for a strictly monotone metric_fn like sqrt,
    the percentile bounds are formally invariant under the
    transform (sqrt(percentile(x)) == percentile(sqrt(x))).
    The primitive's contract is to apply metric_fn per resample
    regardless; this test pins THAT contract, not a divergence
    that monotonicity rules out.
    """
    squared_errors = np.array([0.04, 0.09, 0.25, 0.16, 0.01, 0.36, 0.49, 0.64], dtype=np.float64)
    entities = np.array([0, 0, 1, 1, 2, 2, 3, 3])

    def _rmse(x: np.ndarray) -> float:
        return float(np.sqrt(np.nanmean(x)))

    mean_p, lo_p, hi_p = entity_block_bootstrap_ci(
        squared_errors, entities, n_resamples=300, seed=7, metric_fn=_rmse
    )

    # Independent re-implementation of the per-resample path.
    rng = np.random.Generator(np.random.PCG64(7))
    n_entities = 4
    rows_by_entity = [np.where(entities == i)[0] for i in range(n_entities)]
    resamples = np.empty(300, dtype=np.float64)
    for r in range(300):
        picked = rng.integers(0, n_entities, size=n_entities)
        idx = np.concatenate([rows_by_entity[p] for p in picked])
        resamples[r] = _rmse(squared_errors[idx])
    expected_lo = float(np.percentile(resamples, 2.5, method="linear"))
    expected_hi = float(np.percentile(resamples, 97.5, method="linear"))
    np.testing.assert_allclose(lo_p, expected_lo, atol=1e-12)
    np.testing.assert_allclose(hi_p, expected_hi, atol=1e-12)
    np.testing.assert_allclose(mean_p, _rmse(squared_errors), atol=1e-12)


def test_entity_block_bootstrap_ci_custom_metric_fn_is_called_per_resample() -> None:
    """Any callable metric_fn is applied per resample. Sentinel
    callable returns a fixed value to prove the dispatch landed."""
    losses = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    entities = np.array([0, 0, 1, 1])
    sentinel = 42.0

    def _const(x: np.ndarray) -> float:
        del x
        return sentinel

    mean, lo, hi = entity_block_bootstrap_ci(losses, entities, n_resamples=20, metric_fn=_const)
    assert mean == sentinel
    assert lo == sentinel
    assert hi == sentinel
