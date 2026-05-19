"""F1 caller-input-row-order restore contract (Phase 9 refactor).

Every caller-facing prediction surface must return rows in the
caller's input ``X`` row order, not the transform's internal
``(id, time)`` sort order. Int-id synthetic panels coincidentally
hide the bug, so these tests use shuffled STRING-id panels where the
internal sort is a non-identity permutation.

Mandatory tests #3, #6, #7, #8, #11 (refactor_prediction_step.md).
"""

import logging

import numpy as np
import pandas as pd
import pytest
import torch

from seq_sklearn.config._adapters import SchedulerParams, TabularConfigParams
from seq_sklearn.data.synthetic.generator import SyntheticPanelGenerator
from seq_sklearn.errors import DataContractError
from seq_sklearn.logging import Event
from seq_sklearn.models.transformer.tft.classifier import TFTClassifier
from seq_sklearn.models.transformer.tft.regressor import TFTRegressor

pytestmark = pytest.mark.integration


def _force_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from seq_sklearn.hardware import HardwareTier

    monkeypatch.setattr("seq_sklearn.training.trainer.detect", lambda: HardwareTier.CPU)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


def _gen(target_kind: str = "binary") -> SyntheticPanelGenerator:
    return SyntheticPanelGenerator(
        target_kind=target_kind,  # type: ignore[arg-type]
        num_entities=18,
        periods_per_entity=16,
        signal_strength=0.85,
        lookback=6,
        seed=42,
    )


def _tab(gen: SyntheticPanelGenerator) -> TabularConfigParams:
    return TabularConfigParams(
        id_col=gen.id_col,
        time_col=gen.time_col,
        static_categorical_cols=tuple(gen.static_categorical_cols),
        static_real_cols=tuple(gen.static_real_cols),
        time_varying_real_cols=tuple(gen.time_varying_real_cols),
        time_varying_categorical_cols=tuple(gen.time_varying_categorical_cols),
        lookback=gen.lookback,
        min_periods=1,
        min_periods_predict=1,
        max_categorical_cardinality=10_000,
    )


_COMMON = {
    "scheduler": SchedulerParams(name="constant", warmup_steps=0),
    "hidden_size": 8,
    "attention_heads": 2,
    "max_epochs": 1,
    "batch_size": 32,
    "val_fraction": 0.2,
    "precision": "32-true",
    "verbose": False,
    "seed": 42,
}


def _string_ids(panel: pd.DataFrame, id_col: str) -> pd.DataFrame:
    out = panel.copy()
    out[id_col] = "e" + out[id_col].astype(str)
    return out


def _shuffle(
    panel: pd.DataFrame, y: np.ndarray, seed: int
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return (shuffled panel, shuffled y, perm).

    ``perm`` is the index used as ``panel.iloc[perm]``: shuffled row
    ``i`` is original row ``perm[i]``, so for any original-order
    output array ``a``, ``a[perm]`` is the shuffled-order output. The
    row-order oracle asserts ``a[perm] == predict(shuffled)``.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(panel))
    sp = panel.iloc[perm].reset_index(drop=True)
    return sp, y[perm], perm


def test_predict_output_row_order_shuffled_panel(monkeypatch: pytest.MonkeyPatch) -> None:
    """#3: predict / predict_proba follow the caller's input row order."""
    _force_cpu(monkeypatch)
    gen = _gen("binary")
    panel, y = gen.generate(seed=42)
    panel = _string_ids(panel, gen.id_col)
    est = TFTClassifier(
        task_type="binary", tabular_config=_tab(gen), cal_fraction=0.0, **_COMMON
    ).fit(panel, y)

    proba = est.predict_proba(panel)
    labels = est.predict(panel)
    assert len(proba) == len(panel)
    assert len(labels) == len(panel)

    # Permuting the caller rows must permute the outputs identically.
    sp, _, perm = _shuffle(panel, y, seed=7)
    proba_s = est.predict_proba(sp)
    labels_s = est.predict(sp)
    np.testing.assert_allclose(proba[perm], proba_s, rtol=1e-5, atol=1e-6, equal_nan=True)
    np.testing.assert_array_equal(labels[perm], labels_s)


def test_below_min_periods_predict_entity_nan_filled_preserves_count(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """#6: a sub-floor entity is NaN-filled per row, never dropped."""
    _force_cpu(monkeypatch)
    gen = _gen("binary")
    panel, y = gen.generate(seed=42)
    panel = _string_ids(panel, gen.id_col)
    # min_periods=1 keeps the short entity at fit; min_periods_predict=4
    # NaN-fills any entity with < 4 rows. Truncate one entity to 2 rows.
    short_id = panel[gen.id_col].iloc[-1]
    keep_short = panel[panel[gen.id_col] == short_id].index[:2]
    drop = [i for i in panel[panel[gen.id_col] == short_id].index if i not in set(keep_short)]
    panel = panel.drop(index=drop).reset_index(drop=True)
    y = np.delete(y, drop)

    tab = TabularConfigParams(
        **{**_tab(gen).get_params(deep=False), "min_periods": 1, "min_periods_predict": 4}
    )
    est = TFTClassifier(task_type="binary", tabular_config=tab, cal_fraction=0.0, **_COMMON).fit(
        panel, y
    )
    with caplog.at_level(logging.WARNING, logger="seq_sklearn"):
        proba = est.predict_proba(panel)

    assert len(proba) == len(panel)
    short_mask = (panel[gen.id_col] == short_id).to_numpy()
    # All sub-floor rows NaN (2 rows, not one collapsed row, not dropped).
    assert short_mask.sum() == 2
    assert np.isnan(proba[short_mask]).all()
    # Above-floor rows finite and index-aligned.
    assert np.isfinite(proba[~short_mask]).all()
    # The breach warning is AGGREGATED per transform call: every record
    # carries the entity COUNT (one sub-floor entity here; all others
    # have 16 rows >= 4), never one warning per below-floor row/entity.
    # That aggregation is the #6 contract; the predict path may invoke
    # transform more than once internally, so assert the property on
    # every record rather than coupling to that internal call count.
    breach = [
        r
        for r in caplog.records
        if getattr(r, "event", None) == Event.DATA_DUPLICATE_FLOOR_BREACH_COUNT
    ]
    assert len(breach) >= 1
    assert all(r.payload["count"] == 1 for r in breach)
    assert all(r.payload["min_periods_predict"] == 4 for r in breach)


def _binary_panels(
    gen: SyntheticPanelGenerator, seed: int
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    panel, y = gen.generate(seed=seed)
    panel = _string_ids(panel, gen.id_col)
    n = len(panel)
    cut = n - n // 3
    return (
        panel.iloc[:cut].reset_index(drop=True),
        y[:cut],
        panel.iloc[cut:].reset_index(drop=True),
        y[cut:],
    )


def test_calibration_fold_alignment_unsorted_x_cal(monkeypatch: pytest.MonkeyPatch) -> None:
    """#7: explicit calibration_set pairs outputs with y_cal in caller order."""
    _force_cpu(monkeypatch)
    gen = _gen("binary")
    x_tr, y_tr, x_cal, y_cal = _binary_panels(gen, seed=42)

    # Shuffle x_cal into NON-(id,time) order so the transform's internal
    # sort is a non-identity permutation.
    x_cal_s, y_cal_s, _ = _shuffle(x_cal, y_cal, seed=11)
    # Pre-sorted twin of the SAME calibration data.
    order = x_cal_s.sort_values([gen.id_col, gen.time_col]).index.to_numpy()
    x_cal_sorted = x_cal_s.iloc[order].reset_index(drop=True)
    y_cal_sorted = y_cal_s[order]

    common = dict(
        task_type="binary",
        tabular_config=_tab(gen),
        threshold_tuning=True,
        cal_fraction=0.0,
        **_COMMON,
    )
    est_shuf = TFTClassifier(**common).fit(x_tr, y_tr, calibration_set=(x_cal_s, y_cal_s))
    est_sort = TFTClassifier(**common).fit(x_tr, y_tr, calibration_set=(x_cal_sorted, y_cal_sorted))

    # The fix makes the shuffled and pre-sorted calibration folds yield
    # the SAME tuned threshold (both pair outputs with y_cal in a
    # consistent order). Non-degenerate: threshold strictly interior.
    assert 0.0 < est_shuf.decision_threshold_ < 1.0
    assert est_shuf.decision_threshold_ == pytest.approx(est_sort.decision_threshold_, abs=1e-6)

    # Mispairing sensitivity: a deliberately shuffled y_cal (broken
    # pairing) must move the threshold measurably, so the test would
    # fail if the fold mispaired sorted outputs with caller-order y_cal.
    rng = np.random.default_rng(99)
    y_mis = y_cal_s[rng.permutation(len(y_cal_s))]
    est_mis = TFTClassifier(**common).fit(x_tr, y_tr, calibration_set=(x_cal_s, y_mis))
    assert est_mis.decision_threshold_ != pytest.approx(est_shuf.decision_threshold_, abs=1e-6)


def test_calibration_fold_internal_split_sorted_consistent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#11: the default internal-split cal fold stays sorted-space consistent."""
    _force_cpu(monkeypatch)
    gen = _gen("binary")
    panel, y = gen.generate(seed=42)
    panel = _string_ids(panel, gen.id_col)

    shuf, y_shuf, _ = _shuffle(panel, y, seed=5)
    order = shuf.sort_values([gen.id_col, gen.time_col]).index.to_numpy()
    pre = shuf.iloc[order].reset_index(drop=True)
    y_pre = y_shuf[order]

    common = dict(
        task_type="binary",
        tabular_config=_tab(gen),
        threshold_tuning=True,
        cal_fraction=0.25,
        **_COMMON,
    )
    est_shuf = TFTClassifier(**common).fit(shuf, y_shuf)
    est_pre = TFTClassifier(**common).fit(pre, y_pre)

    # The internal-split branch indexes outputs and targets by the same
    # sorted-space `keep`; input_row_order is NOT applied there. Both
    # fits derive the SAME sorted batch, so the tuned threshold is
    # identical. A regression that threads input_row_order through the
    # internal-split branch would break this identity.
    assert 0.0 < est_shuf.decision_threshold_ < 1.0
    assert est_shuf.decision_threshold_ == pytest.approx(est_pre.decision_threshold_, abs=1e-6)


def test_predict_with_attention_row_order_shuffled_panel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#8: every per-row AttentionOutput field follows caller row order."""
    _force_cpu(monkeypatch)
    gen = _gen("binary")
    panel, y = gen.generate(seed=42)
    panel = _string_ids(panel, gen.id_col)
    est = TFTClassifier(
        task_type="binary", tabular_config=_tab(gen), cal_fraction=0.0, **_COMMON
    ).fit(panel, y)

    a = est.predict_with_attention(panel)
    sp, _, perm = _shuffle(panel, y, seed=3)
    b = est.predict_with_attention(sp)

    per_row = (
        "predictions",
        "probabilities",
        "logits",
        "var_selection_weights",
        "static_var_selection_weights",
        "attention_weights",
        "padding_mask",
        "entity_id",
    )
    for field in per_row:
        va = np.asarray(getattr(a, field))
        vb = np.asarray(getattr(b, field))
        assert len(va) == len(panel), field
        np.testing.assert_allclose(
            va[perm], vb, rtol=1e-5, atol=1e-6, equal_nan=True, err_msg=field
        )

    # Regressor variant: per-row fields reorder; quantiles_used does NOT.
    rgen = _gen("regression_quantile")
    rpanel, ry = rgen.generate(seed=42)
    rpanel = _string_ids(rpanel, rgen.id_col)
    rest = TFTRegressor(
        task_type="regression_quantile",
        tabular_config=_tab(rgen),
        quantiles=(0.1, 0.5, 0.9),
        cal_fraction=0.0,
        **_COMMON,
    ).fit(rpanel, ry.astype(float))
    ra = rest.predict_with_attention(rpanel)
    rsp, _, rperm = _shuffle(rpanel, ry, seed=8)
    rb = rest.predict_with_attention(rsp)
    for field in (
        "predictions",
        "var_selection_weights",
        "static_var_selection_weights",
        "attention_weights",
        "padding_mask",
        "entity_id",
    ):
        va = np.asarray(getattr(ra, field))
        vb = np.asarray(getattr(rb, field))
        np.testing.assert_allclose(
            va[rperm], vb, rtol=1e-5, atol=1e-6, equal_nan=True, err_msg=field
        )
    # quantiles_used is fit-time metadata, shuffle-INVARIANT.
    assert ra.quantiles_used == rb.quantiles_used


def test_predict_quantiles_row_order_shuffled_panel(monkeypatch: pytest.MonkeyPatch) -> None:
    """#3 (regressor array surface): predict / predict_quantiles follow caller order.

    Closes the ledger #3 `predict_quantiles` clause and gives the
    `_calibrated_matrix` array-path reorder (`_regressor.py`) an
    adversarial oracle: existing int-id regressor tests coincidentally
    hide the bug because the (id, time) sort is identity there.
    """
    _force_cpu(monkeypatch)
    rgen = _gen("regression_quantile")
    panel, ry = rgen.generate(seed=42)
    panel = _string_ids(panel, rgen.id_col)
    ry = ry.astype(float)
    est = TFTRegressor(
        task_type="regression_quantile",
        tabular_config=_tab(rgen),
        quantiles=(0.1, 0.5, 0.9),
        cal_fraction=0.0,
        **_COMMON,
    ).fit(panel, ry)

    pq = est.predict_quantiles(panel)
    pt = est.predict(panel)
    assert len(pq) == len(panel)
    assert len(pt) == len(panel)
    # Per-row variance so the [perm] oracle is non-vacuous: a no-op
    # reorder (regression) would leave both calls in sorted order, and
    # since the underlying data is identical the shuffled call would
    # equal the unshuffled one, so pq[perm] != pq_s for a non-identity
    # perm -> the assertion below fails. That is the mutation the test
    # is built to catch.
    assert not np.allclose(pq, pq[::-1])

    sp, _, perm = _shuffle(panel, ry, seed=4)
    pq_s = est.predict_quantiles(sp)
    pt_s = est.predict(sp)
    np.testing.assert_allclose(pq[perm], pq_s, rtol=1e-5, atol=1e-6, equal_nan=True)
    np.testing.assert_allclose(pt[perm], pt_s, rtol=1e-5, atol=1e-6, equal_nan=True)


def test_explicit_calibration_set_length_mismatch_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deferral 2a: len(x_cal) != len(y_cal) fails fast with DataContractError.

    Without the early guard the mispaired arrays fail late and
    cryptically inside the calibrator / threshold-tuner fit.
    """
    _force_cpu(monkeypatch)
    gen = _gen("binary")
    x_tr, y_tr, x_cal, y_cal = _binary_panels(gen, seed=42)
    est = TFTClassifier(
        task_type="binary",
        tabular_config=_tab(gen),
        threshold_tuning=True,
        cal_fraction=0.0,
        **_COMMON,
    )
    with pytest.raises(DataContractError, match="calibration_set length mismatch"):
        est.fit(x_tr, y_tr, calibration_set=(x_cal, y_cal[:-1]))


def test_explicit_calibration_set_keeps_below_floor_entity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deferral 2b: explicit calibration_set is caller-owned, NOT floor-filtered.

    The explicit branch intentionally does not drop ``< min_periods_predict``
    entities (real ``y_cal``, no sentinel hazard) - unlike the internal-split
    recomputed fold, which raises when its fold is all below-floor. Here every
    calibration entity is below the floor yet calibration still succeeds,
    pinning the deliberate asymmetry.
    """
    _force_cpu(monkeypatch)
    gen = _gen("binary")
    x_tr, y_tr, x_cal, y_cal = _binary_panels(gen, seed=42)
    # Truncate EVERY calibration entity to 2 rows; min_periods_predict=4
    # makes them all below-floor, min_periods=1 keeps them at fit.
    # _binary_panels returns x_cal with a 0..n-1 positional index aligned
    # to y_cal, so the retained rows' index selects the matching labels.
    x_cal_trunc = x_cal.groupby(gen.id_col, sort=False, group_keys=False).head(2)
    y_cal_trunc = np.asarray(y_cal)[x_cal_trunc.index.to_numpy()]
    x_cal = x_cal_trunc.reset_index(drop=True)
    assert (x_cal.groupby(gen.id_col).size() == 2).all()

    tab = TabularConfigParams(
        **{**_tab(gen).get_params(deep=False), "min_periods": 1, "min_periods_predict": 4}
    )
    est = TFTClassifier(
        task_type="binary",
        tabular_config=tab,
        threshold_tuning=True,
        cal_fraction=0.0,
        **_COMMON,
    )
    # No raise, no silent drop-to-empty: calibration completes and a
    # threshold is set even though every cal entity is below the floor.
    est.fit(x_tr, y_tr, calibration_set=(x_cal, y_cal_trunc))
    assert hasattr(est, "decision_threshold_")
    assert 0.0 <= est.decision_threshold_ <= 1.0
