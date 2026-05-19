"""Tests for the F6 synthetic data-generating process."""

import warnings

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from seq_sklearn.data.synthetic.generator import SyntheticPanelGenerator, _sigmoid


class _StubGenerator:
    """Minimal generator stub returning a degenerate Dirichlet draw.

    Used to exercise the F6 step-6 fallback deterministically: the
    returned vector concentrates all mass in two entries so every clip
    pass leaves fewer than three loaded entries, forcing the retry cap
    and the pre-clip fallback path.
    """

    def __init__(self) -> None:
        self.calls = 0

    def dirichlet(self, alpha: np.ndarray) -> np.ndarray:
        self.calls += 1
        n = len(alpha)
        vec = np.full(n, 0.001)
        vec[0] = 0.6
        vec[1] = 0.4 - 0.001 * (n - 2)
        return vec


def test_dgp_version_is_one() -> None:
    gen = SyntheticPanelGenerator(seed=42)
    assert gen.dgp_version == 1


@pytest.mark.parametrize("seed", SyntheticPanelGenerator.CANONICAL_SEEDS)
def test_panel_columns_and_dtype(seed: int) -> None:
    gen = SyntheticPanelGenerator(num_entities=20, periods_per_entity=(1, 60), seed=seed)
    panel, y = gen.generate()
    assert list(panel.columns)[:2] == ["id", "time"]
    assert np.issubdtype(panel["time"].dtype, np.datetime64)
    assert len(panel) == len(y)
    assert len(panel) > 0


@pytest.mark.parametrize("seed", SyntheticPanelGenerator.CANONICAL_SEEDS)
def test_three_seed_signal_recovery_above_chance(seed: int) -> None:
    gen = SyntheticPanelGenerator(
        target_kind="binary",
        num_entities=120,
        periods_per_entity=20,
        signal_strength=0.98,
        noise_level=0.05,
        seed=seed,
    )
    panel, y = gen.generate()
    feature_cols = [
        *gen.static_categorical_cols,
        *gen.static_real_cols,
        *gen.time_varying_real_cols,
        *gen.time_varying_categorical_cols,
    ]
    features = panel[feature_cols].to_numpy(dtype=float)
    clf = LogisticRegression(max_iter=500)
    clf.fit(features, y)
    accuracy = clf.score(features, y)
    majority = max(float(np.mean(y)), 1.0 - float(np.mean(y)))
    assert accuracy > majority + 0.03


def test_byte_reproducible_per_seed() -> None:
    a, ya = SyntheticPanelGenerator(num_entities=10, seed=42).generate()
    b, yb = SyntheticPanelGenerator(num_entities=10, seed=42).generate()
    np.testing.assert_array_equal(ya, yb)
    np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())


def test_call_time_seed_overrides_constructor() -> None:
    gen = SyntheticPanelGenerator(num_entities=8, seed=1)
    _, y_default = gen.generate()
    _, y_override = gen.generate(seed=2)
    assert not np.array_equal(y_default, y_override)


def test_none_seed_is_nondeterministic() -> None:
    gen = SyntheticPanelGenerator(num_entities=8, seed=None)
    _, y1 = gen.generate()
    _, y2 = gen.generate()
    assert not np.array_equal(y1, y2)


def test_ragged_periods_includes_one_row_entity() -> None:
    gen = SyntheticPanelGenerator(
        num_entities=200,
        periods_per_entity=(1, 60),
        seed=42,
    )
    panel, _ = gen.generate()
    per_entity = panel.groupby("id").size()
    assert per_entity.min() == 1
    assert per_entity.max() > 12


def test_callable_periods_per_entity() -> None:
    gen = SyntheticPanelGenerator(
        num_entities=5,
        periods_per_entity=lambda rng: int(rng.integers(5, 8)),
        seed=3,
    )
    panel, _ = gen.generate()
    assert len(panel) > 0


def test_zero_period_entity_skipped() -> None:
    gen = SyntheticPanelGenerator(
        num_entities=4,
        periods_per_entity=lambda rng: 0,
        seed=1,
    )
    panel, y = gen.generate()
    assert panel.empty
    assert y.shape == (0,)
    assert np.issubdtype(panel["time"].dtype, np.datetime64)


def test_multiclass_targets_in_range() -> None:
    gen = SyntheticPanelGenerator(
        target_kind="multiclass",
        num_classes=4,
        num_entities=30,
        periods_per_entity=15,
        class_distribution=(0.4, 0.3, 0.2, 0.1),
        seed=42,
    )
    _, y = gen.generate()
    assert set(np.unique(y)).issubset({0, 1, 2, 3})


def test_regression_point_is_float() -> None:
    gen = SyntheticPanelGenerator(
        target_kind="regression_point",
        num_entities=20,
        periods_per_entity=15,
        seed=42,
    )
    _, y = gen.generate()
    assert y.dtype == np.float64


def test_regression_quantile_is_float() -> None:
    gen = SyntheticPanelGenerator(
        target_kind="regression_quantile",
        num_entities=10,
        periods_per_entity=15,
        seed=7,
    )
    _, y = gen.generate()
    assert y.dtype == np.float64


@pytest.mark.parametrize("grain", ["daily", "weekly", "monthly", "quarterly", "yearly"])
def test_period_grain_produces_datetime(grain: str) -> None:
    gen = SyntheticPanelGenerator(num_entities=5, periods_per_entity=10, period_grain=grain, seed=1)
    panel, _ = gen.generate()
    assert np.issubdtype(panel["time"].dtype, np.datetime64)


def test_w_temporal_short_lookback_passthrough() -> None:
    gen = SyntheticPanelGenerator(lookback=2, seed=1)
    rng = np.random.Generator(np.random.PCG64(1))
    w = gen._build_w_temporal(rng)
    assert w.shape == (2,)
    assert np.isclose(w.sum(), 1.0)


def test_w_temporal_fallback_path() -> None:
    gen = SyntheticPanelGenerator(lookback=6, seed=1)
    stub = _StubGenerator()
    w = gen._build_w_temporal(stub)  # type: ignore[arg-type]
    assert stub.calls == 1 + 5  # one initial draw plus five retries
    assert np.isclose(w.sum(), 1.0)
    assert np.count_nonzero(w) >= 3


class _LastDrawSucceedsGenerator:
    """Returns degenerate draws until the final retry, then a loaded one.

    Exercises the post-retry-loop success return: the loop's six total
    draws are degenerate for the first five and well-loaded on the
    sixth, so the success check after the loop body (not the top-of-loop
    check) is the one that returns.
    """

    def __init__(self) -> None:
        self.calls = 0

    def dirichlet(self, alpha: np.ndarray) -> np.ndarray:
        self.calls += 1
        n = len(alpha)
        if self.calls <= 5:
            vec = np.full(n, 0.001)
            vec[0] = 0.6
            vec[1] = 0.4 - 0.001 * (n - 2)
            return vec
        return np.full(n, 1.0 / n)


def test_w_temporal_post_loop_success_return() -> None:
    gen = SyntheticPanelGenerator(lookback=6, seed=1)
    stub = _LastDrawSucceedsGenerator()
    gen._build_w_temporal(stub)  # type: ignore[arg-type]
    assert stub.calls == 1 + 5


def test_w_temporal_fallback_pre_clip_branch_hand_checked() -> None:
    # post_clip has only 2 non-zero entries (< 3), so the fallback picks
    # the 3 smallest values of the full pre-clip draw. Ties (the four
    # 0.001 entries) resolve to the lowest indices: 2, 3, 4.
    pre_clip = np.array([0.6, 0.396, 0.001, 0.001, 0.001, 0.001])
    post_clip = np.where(pre_clip < 0.05, 0.0, pre_clip)
    assert np.count_nonzero(post_clip) == 2
    w = SyntheticPanelGenerator._w_temporal_fallback(pre_clip, post_clip)
    forced = pre_clip.copy()
    forced[[2, 3, 4]] = 0.1
    np.testing.assert_allclose(w, forced / forced.sum())
    assert np.isclose(w.sum(), 1.0)


def test_w_temporal_fallback_post_clip_branch_hand_checked() -> None:
    # post_clip has >= 3 non-zero entries, so the fallback picks the 3
    # smallest NON-ZERO values of the post-clip vector: indices 5, 4, 3
    # (values 0.05, 0.07, 0.08).
    pre_clip = np.array([0.4, 0.3, 0.1, 0.08, 0.07, 0.05])
    post_clip = np.where(pre_clip < 0.05, 0.0, pre_clip)
    assert np.count_nonzero(post_clip) == 6
    w = SyntheticPanelGenerator._w_temporal_fallback(pre_clip, post_clip)
    forced = pre_clip.copy()
    forced[[3, 4, 5]] = 0.1
    np.testing.assert_allclose(w, forced / forced.sum())
    assert np.isclose(w.sum(), 1.0)


def test_w_temporal_normal_path_has_three_loaded() -> None:
    gen = SyntheticPanelGenerator(lookback=12, seed=42)
    rng = np.random.Generator(np.random.PCG64(42))
    w = gen._build_w_temporal(rng)
    assert np.count_nonzero(w) >= 3
    assert np.isclose(w.sum(), 1.0)


def test_dgp_version_bump_regression(monkeypatch: pytest.MonkeyPatch) -> None:
    gen_v1 = SyntheticPanelGenerator(num_entities=10, seed=42)
    _, y_v1 = gen_v1.generate()

    gen_v2 = SyntheticPanelGenerator(num_entities=10, seed=42)
    monkeypatch.setattr(gen_v2, "dgp_version", 2)

    def shifted_target(rng: np.random.Generator, z: np.ndarray) -> object:
        return float(np.sum(z)) + gen_v2.dgp_version

    monkeypatch.setattr(gen_v2, "_sample_target", shifted_target)
    _, y_v2 = gen_v2.generate()

    assert gen_v2.dgp_version != gen_v1.dgp_version
    assert not np.array_equal(y_v1, y_v2)


def test_invalid_config_raises() -> None:
    with pytest.raises(ValueError, match="num_entities"):
        SyntheticPanelGenerator(num_entities=0)
    with pytest.raises(ValueError, match="signal_strength"):
        SyntheticPanelGenerator(signal_strength=1.5)
    with pytest.raises(ValueError, match="lookback"):
        SyntheticPanelGenerator(lookback=0)
    with pytest.raises(ValueError, match="categorical_cardinality"):
        SyntheticPanelGenerator(categorical_cardinality=0)
    with pytest.raises(ValueError, match="categorical_embed_dim"):
        SyntheticPanelGenerator(categorical_embed_dim=0)
    with pytest.raises(ValueError, match="class_balance"):
        SyntheticPanelGenerator(class_balance=0.0)


def test_multiclass_config_validation() -> None:
    with pytest.raises(ValueError, match="num_classes"):
        SyntheticPanelGenerator(target_kind="multiclass", num_classes=1)
    with pytest.raises(ValueError, match="class_distribution length"):
        SyntheticPanelGenerator(
            target_kind="multiclass", num_classes=3, class_distribution=(0.5, 0.5)
        )
    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        SyntheticPanelGenerator(
            target_kind="multiclass",
            num_classes=3,
            class_distribution=(0.5, 0.5, 0.5),
        )


def test_multiclass_uniform_when_no_distribution() -> None:
    gen = SyntheticPanelGenerator(target_kind="multiclass", num_classes=3, num_entities=10, seed=1)
    _, y = gen.generate()
    assert set(np.unique(y)).issubset({0, 1, 2})


def test_only_static_categorical_zero() -> None:
    gen = SyntheticPanelGenerator(
        num_static_categorical=0,
        num_time_varying_categorical=2,
        num_entities=5,
        periods_per_entity=10,
        seed=1,
    )
    panel, y = gen.generate()
    assert gen.static_categorical_cols == []
    assert len(gen.time_varying_categorical_cols) == 2
    assert len(panel) == len(y) > 0


def test_only_time_varying_categorical_zero() -> None:
    gen = SyntheticPanelGenerator(
        num_static_categorical=2,
        num_time_varying_categorical=0,
        num_entities=5,
        periods_per_entity=10,
        seed=1,
    )
    panel, y = gen.generate()
    assert len(gen.static_categorical_cols) == 2
    assert gen.time_varying_categorical_cols == []
    assert len(panel) == len(y) > 0


def test_no_categorical_or_real_columns() -> None:
    gen = SyntheticPanelGenerator(
        num_static_categorical=0,
        num_static_real=0,
        num_time_varying_real=1,
        num_time_varying_categorical=0,
        num_entities=5,
        periods_per_entity=10,
        seed=1,
    )
    panel, y = gen.generate()
    assert len(panel) == len(y) > 0


def test_sigmoid_no_overflow_on_extreme_logits() -> None:
    z = np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = _sigmoid(z)
    assert out[0] == pytest.approx(0.0)
    assert out[2] == pytest.approx(0.5)
    assert out[4] == pytest.approx(1.0)
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_class_distribution_zero_entry_raises() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        SyntheticPanelGenerator(
            target_kind="multiclass",
            num_classes=2,
            class_distribution=(0.0, 1.0),
            seed=1,
        )


def test_generator_has_no_prediction_step_and_emits_n_periods() -> None:
    """#12: F6 generator is unconditionally contemporaneous.

    Decoupled from the TabularToSequence clamp test #5 so neither can
    be silently omitted while the other is written.
    """
    import inspect

    sig = inspect.signature(SyntheticPanelGenerator.__init__)
    assert "prediction_step" not in sig.parameters
    with pytest.raises(TypeError):
        SyntheticPanelGenerator(prediction_step=1)  # type: ignore[call-arg]

    # One row per period per entity (n_periods, NOT n_periods - 1): the
    # old prediction_step=1 skip-guard would have dropped each entity's
    # final window. Fixed period count makes the count exact.
    gen = SyntheticPanelGenerator(num_entities=7, periods_per_entity=10, seed=42)
    panel, y = gen.generate(seed=42)
    per_entity = panel.groupby(gen.id_col).size()
    assert (per_entity == 10).all()
    assert len(panel) == 7 * 10
    assert len(y) == len(panel)

    # Single-period entities now APPEAR (entity-id SET change, not just
    # a count delta): the old skip-guard emitted zero rows for them.
    ragged = SyntheticPanelGenerator(num_entities=150, periods_per_entity=(1, 40), seed=7)
    rpanel, _ = ragged.generate(seed=7)
    rcounts = rpanel.groupby(ragged.id_col).size()
    assert rcounts.min() == 1
