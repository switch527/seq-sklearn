"""TransformerSequenceEstimator family-base contract (Phase 6b, A2 / A15.1).

The ``Classifier`` / ``Regressor`` mixins compose with the Phase-6a
family bases and add ``predict_with_attention`` WITHOUT breaking the
base ``predict`` / ``predict_proba`` / ``predict_quantiles`` contract.
The introspection tensors are non-degenerate; below-floor entities are
NaN-filled on the prediction surface exactly as the base predict path;
the ``device=`` keyword flips every field to an on-device tensor; and
``predict_with_attention`` raises ``NotFittedError`` before ``fit``.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from seq_sklearn.config._adapters import SchedulerParams, TabularConfigParams
from seq_sklearn.errors import NotFittedError
from seq_sklearn.inference.attention import AttentionOutput, RegressionAttentionOutput
from tests._test_models._dummy_estimator import (
    _DummyTransformerClassifier,
    _DummyTransformerRegressor,
)


def _force_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from seq_sklearn.hardware import HardwareTier

    monkeypatch.setattr("seq_sklearn.training.trainer.detect", lambda: HardwareTier.CPU)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


def _panel(n_entities: int = 16, rows: int = 6) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    out: list[dict[str, object]] = []
    y: list[int] = []
    for e in range(n_entities):
        label = e % 2
        for t in range(rows):
            out.append(
                {
                    "id": e,
                    "time": pd.Timestamp("2021-01-01") + pd.offsets.Day(t),
                    "a": float(label) + 0.2 * rng.normal(),
                    "b": float(1 - label) + 0.2 * rng.normal(),
                }
            )
            y.append(label)
    return pd.DataFrame(out), np.asarray(y)


def _tab() -> TabularConfigParams:
    return TabularConfigParams(
        time_varying_real_cols=("a", "b"),
        lookback=3,
        min_periods=1,
        min_periods_predict=1,
    )


_COMMON: dict[str, object] = {
    "scheduler": SchedulerParams(name="constant", warmup_steps=0),
    "max_epochs": 1,
    "batch_size": 16,
    "val_fraction": 0.2,
    "cal_fraction": 0.0,
    "precision": "32-true",
    "verbose": False,
    "seed": 42,
}


def _clf(**kw: object) -> _DummyTransformerClassifier:
    return _DummyTransformerClassifier(task_type="binary", tabular_config=_tab(), **_COMMON, **kw)


def test_classifier_mixin_does_not_break_base_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_cpu(monkeypatch)
    x, y = _panel()
    est = _clf().fit(x, y)
    proba = est.predict_proba(x)
    assert proba.shape == (96, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert set(np.unique(est.predict(x))).issubset({0, 1})


def test_classifier_predict_with_attention_shapes_and_consistency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_cpu(monkeypatch)
    x, y = _panel()
    est = _clf().fit(x, y)
    out = est.predict_with_attention(x)
    assert isinstance(out, AttentionOutput)
    n = 96
    length = 3  # lookback
    assert out.probabilities.shape == (n, 2)
    assert out.logits.shape == (n, 1)  # binary head out_dim=1, pre-activation
    assert out.predictions.shape == (n,)
    assert out.var_selection_weights.shape == (n, length, 2)
    assert out.static_var_selection_weights.shape == (n, 2)
    assert out.attention_weights.shape == (n, 2, length, length)
    assert out.padding_mask.shape == (n, length)
    assert out.entity_id.shape == (n,)
    # predictions are A15.1 class INDICES, consistent with the base
    # predict (mapped through classes_).
    mapped = np.asarray(est.classes_)[out.predictions]
    assert np.array_equal(mapped, est.predict(x))
    # probabilities equal the base predict_proba (one shared forward).
    assert np.allclose(out.probabilities, est.predict_proba(x), equal_nan=True)


def test_introspection_tensors_are_non_degenerate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_cpu(monkeypatch)
    x, y = _panel()
    out = _clf().fit(x, y).predict_with_attention(x)
    vsw = out.var_selection_weights
    assert np.allclose(vsw.sum(axis=-1), 1.0)  # softmax over the 2 vars
    assert vsw.std() > 0.0  # not all-uniform
    ssw = out.static_var_selection_weights
    assert np.allclose(ssw.sum(axis=-1), 1.0)
    aw = out.attention_weights
    # every query row over valid keys sums to 1 (all keys valid here),
    # and attention is not the uniform 1/L everywhere.
    assert np.allclose(aw.sum(axis=-1), 1.0)
    assert aw.std() > 0.0


def test_below_floor_nan_fill_matches_base(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_cpu(monkeypatch)
    x_fit, y_fit = _panel()
    est = _DummyTransformerClassifier(
        task_type="binary",
        tabular_config=TabularConfigParams(
            time_varying_real_cols=("a", "b"),
            lookback=3,
            min_periods=1,
            min_periods_predict=3,
        ),
        **_COMMON,
    ).fit(x_fit, y_fit)
    x_pred, _ = _panel(n_entities=4, rows=2)  # all 2-row -> below floor
    out = est.predict_with_attention(x_pred)
    assert np.isnan(out.probabilities).all()
    assert np.isnan(out.logits).all()
    # below-floor predictions still map identically to base predict
    # (NaN proba -> index 0 in the shared _index_from_proba, A2)
    assert np.array_equal(np.asarray(est.classes_)[out.predictions], est.predict(x_pred))
    # introspection tensors stay finite (diagnostics, not predictions)
    assert np.isfinite(out.attention_weights).all()


def test_device_keyword_returns_on_device_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_cpu(monkeypatch)
    x, y = _panel()
    out = _clf().fit(x, y).predict_with_attention(x, device="cpu")
    for field in (
        out.probabilities,
        out.logits,
        out.predictions,
        out.var_selection_weights,
        out.attention_weights,
        out.static_var_selection_weights,
        out.padding_mask,
        out.entity_id,
    ):
        assert isinstance(field, torch.Tensor)
        assert field.device.type == "cpu"


def test_classifier_predict_with_attention_before_fit_raises() -> None:
    est = _clf()
    x = pd.DataFrame({"id": [0], "time": [pd.Timestamp("2021-01-01")], "a": [0.0], "b": [0.0]})
    with pytest.raises(NotFittedError):
        est.predict_with_attention(x)


def test_point_regressor_predict_with_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_cpu(monkeypatch)
    x, _ = _panel()
    y = np.arange(96, dtype=float)
    est = _DummyTransformerRegressor(
        task_type="regression_point", tabular_config=_tab(), **_COMMON
    ).fit(x, y)
    out = est.predict_with_attention(x)
    assert isinstance(out, RegressionAttentionOutput)
    assert out.predictions.shape == (96,)
    assert out.quantiles_used is None
    assert np.allclose(out.predictions, est.predict(x), equal_nan=True)
    assert not hasattr(out, "logits")


def test_quantile_regressor_predict_with_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_cpu(monkeypatch)
    x, _ = _panel()
    y = np.arange(96, dtype=float)
    est = _DummyTransformerRegressor(
        task_type="regression_quantile",
        tabular_config=_tab(),
        quantiles=(0.1, 0.5, 0.9),
        **_COMMON,
    ).fit(x, y)
    out = est.predict_with_attention(x)
    assert out.predictions.shape == (96, 3)
    assert out.quantiles_used == (0.1, 0.5, 0.9)
    assert np.allclose(out.predictions, est.predict_quantiles(x), equal_nan=True)


def test_predict_with_attention_independent_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Kills the shared-seam trap: the `out.probabilities == predict_proba`
    # assertions are vacuous because both delegate to `_forward_backbone`.
    # Pin a KNOWN fixed representation into the fitted backbone and check
    # the output against an independent recomputation through the real
    # head, and that a different representation yields a different result.
    _force_cpu(monkeypatch)
    x, y = _panel()
    est = _clf().fit(x, y)
    backbone = est._module.backbone
    head = est._module.head
    orig = backbone.forward

    def _fixed(value: float):
        def _fwd(batch: dict[str, torch.Tensor]):  # type: ignore[no-untyped-def]
            o = orig(batch)
            o.representation = torch.full_like(o.representation, value)
            return o

        return _fwd

    monkeypatch.setattr(backbone, "forward", _fixed(1.0))
    out1 = est.predict_with_attention(x)
    with torch.no_grad():
        rep = torch.full_like(torch.zeros(1, head.proj.in_features), 1.0)  # (1, hidden)
        exp_logit = float(head(rep).reshape(-1)[0])
    exp_p1 = 1.0 / (1.0 + np.exp(-exp_logit))
    # constant representation -> identical probability on every row
    assert np.allclose(out1.probabilities[:, 1], exp_p1, atol=1e-5)
    assert np.allclose(out1.logits.reshape(-1), exp_logit, atol=1e-5)

    monkeypatch.setattr(backbone, "forward", _fixed(-1.0))
    out2 = est.predict_with_attention(x)
    # a different representation must move the output (catches a mutant
    # that ignores / zeroes the backbone representation)
    assert not np.allclose(out1.probabilities, out2.probabilities)


def test_regressor_predict_with_attention_independent_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regressor twin of the classifier oracle: the regressor half of the
    # _calibrate_raw seam also needs an oracle independent of
    # predict/predict_quantiles (both delegate to the same seam). Point
    # mode, no calibrator, so predictions == head(representation).
    _force_cpu(monkeypatch)
    x, _ = _panel()
    y = np.arange(96, dtype=float)
    est = _DummyTransformerRegressor(
        task_type="regression_point", tabular_config=_tab(), **_COMMON
    ).fit(x, y)
    assert est.calibrator_ is None
    backbone = est._module.backbone
    head = est._module.head
    orig = backbone.forward

    def _fixed(value: float):
        def _fwd(batch: dict[str, torch.Tensor]):  # type: ignore[no-untyped-def]
            o = orig(batch)
            o.representation = torch.full_like(o.representation, value)
            return o

        return _fwd

    monkeypatch.setattr(backbone, "forward", _fixed(1.0))
    out1 = est.predict_with_attention(x)
    with torch.no_grad():
        rep = torch.full_like(torch.zeros(1, head.proj.in_features), 1.0)
        exp = float(head(rep).reshape(-1)[0])
    assert np.allclose(out1.predictions, exp, atol=1e-5)

    monkeypatch.setattr(backbone, "forward", _fixed(-1.0))
    out2 = est.predict_with_attention(x)
    assert not np.allclose(out1.predictions, out2.predictions)


def _patch_fixed_representation(monkeypatch: pytest.MonkeyPatch, est: object, value: float):
    backbone = est._module.backbone  # type: ignore[attr-defined]
    orig = backbone.forward

    def _fwd(batch: dict[str, torch.Tensor]):  # type: ignore[no-untyped-def]
        o = orig(batch)
        o.representation = torch.full_like(o.representation, value)
        return o

    monkeypatch.setattr(backbone, "forward", _fwd)


def test_classifier_threshold_branch_independent_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The _index_from_proba threshold branch (proba[:,1] >=
    # decision_threshold_) is its own shared-seam trap: both
    # predict_with_attention and predict route through it, so a >= -> <
    # mutation inverts both and the consistency assertion stays green.
    # Pin a fixed representation, compute the threshold decision
    # independently, and require a threshold-straddling representation
    # to flip the indices.
    _force_cpu(monkeypatch)
    x, y = _panel()
    est = _DummyTransformerClassifier(
        task_type="binary",
        tabular_config=_tab(),
        threshold_tuning=True,  # calibration_strategy stays "none"
        cal_fraction=0.2,
        **{k: v for k, v in _COMMON.items() if k != "cal_fraction"},
    ).fit(x, y)
    thr = est.decision_threshold_
    assert isinstance(thr, float)
    head = est._module.head

    def _expected_idx(value: float) -> int:
        with torch.no_grad():
            rep = torch.full_like(torch.zeros(1, head.proj.in_features), value)
            logit = float(head(rep).reshape(-1)[0])
        p1 = 1.0 / (1.0 + np.exp(-logit))
        return int(p1 >= thr)  # independent `>=`; a `<` mutant diverges

    # Two representations on opposite sigmoid extremes. The independent
    # `>=` recomputation must match production for BOTH; a `>=` -> `<`
    # mutation in _index_from_proba would invert production only and
    # fail at least one (p1 is continuous, never exactly == thr here).
    for value in (10.0, -10.0):
        _patch_fixed_representation(monkeypatch, est, value)
        out = est.predict_with_attention(x)
        assert np.all(out.predictions == _expected_idx(value))


def test_calibrated_quantile_regressor_independent_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The calibrated arm of _calibrate_raw (calibrator_.transform) is the
    # last un-oracled regressor seam: skipping the transform survives a
    # predict-vs-attention consistency check. Recompute
    # calibrator_.transform(head(rep)) independently and require a
    # different representation to move the calibrated output.
    _force_cpu(monkeypatch)
    x, _ = _panel()
    y = np.arange(96, dtype=float)
    est = _DummyTransformerRegressor(
        task_type="regression_quantile",
        tabular_config=_tab(),
        quantiles=(0.1, 0.5, 0.9),
        calibration_strategy="isotonic_quantile",
        cal_fraction=0.2,
        **{k: v for k, v in _COMMON.items() if k != "cal_fraction"},
    ).fit(x, y)
    cal = est.calibrator_
    assert cal is not None
    head = est._module.head

    def _expected(value: float) -> np.ndarray:
        with torch.no_grad():
            rep = torch.full_like(torch.zeros(1, head.proj.in_features), value)
            raw_row = head(rep).reshape(1, -1)
            return cal.transform(raw_row).numpy()[0]

    _patch_fixed_representation(monkeypatch, est, 1.0)
    out1 = est.predict_with_attention(x)
    exp1 = _expected(1.0)
    assert out1.predictions.shape == (96, 3)
    assert np.allclose(out1.predictions, exp1, atol=1e-5)

    _patch_fixed_representation(monkeypatch, est, -1.0)
    out2 = est.predict_with_attention(x)
    assert not np.allclose(out1.predictions, out2.predictions)


def test_classifier_calibrated_predict_with_attention_matches_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_cpu(monkeypatch)
    x, y = _panel()
    est = _DummyTransformerClassifier(
        task_type="binary",
        tabular_config=_tab(),
        calibration_strategy="temperature",
        cal_fraction=0.2,
        **{k: v for k, v in _COMMON.items() if k != "cal_fraction"},
    ).fit(x, y)
    assert est.calibrator_ is not None  # calibrated branch is real
    out = est.predict_with_attention(x)
    assert np.allclose(out.probabilities, est.predict_proba(x), equal_nan=True)
    mapped = np.asarray(est.classes_)[out.predictions]
    assert np.array_equal(mapped, est.predict(x))


def test_regressor_calibrated_predict_with_attention_matches_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_cpu(monkeypatch)
    x, _ = _panel()
    y = np.arange(96, dtype=float)
    est = _DummyTransformerRegressor(
        task_type="regression_quantile",
        tabular_config=_tab(),
        quantiles=(0.1, 0.5, 0.9),
        calibration_strategy="isotonic_quantile",
        cal_fraction=0.2,
        **{k: v for k, v in _COMMON.items() if k != "cal_fraction"},
    ).fit(x, y)
    assert est.calibrator_ is not None
    out = est.predict_with_attention(x)
    assert np.allclose(out.predictions, est.predict_quantiles(x), equal_nan=True)


def test_binary_threshold_tuned_predict_with_attention_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_cpu(monkeypatch)
    x, y = _panel()
    est = _DummyTransformerClassifier(
        task_type="binary",
        tabular_config=_tab(),
        calibration_strategy="platt",
        threshold_tuning=True,
        cal_fraction=0.2,
        **{k: v for k, v in _COMMON.items() if k != "cal_fraction"},
    ).fit(x, y)
    assert isinstance(est.decision_threshold_, float)
    out = est.predict_with_attention(x)
    # the _index_from_proba threshold branch must agree with base predict
    mapped = np.asarray(est.classes_)[out.predictions]
    assert np.array_equal(mapped, est.predict(x))


def test_regressor_below_floor_nan_fill_matches_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_cpu(monkeypatch)
    x_fit, _ = _panel()
    y_fit = np.arange(96, dtype=float)
    est = _DummyTransformerRegressor(
        task_type="regression_quantile",
        tabular_config=TabularConfigParams(
            time_varying_real_cols=("a", "b"),
            lookback=3,
            min_periods=1,
            min_periods_predict=3,
        ),
        quantiles=(0.1, 0.5, 0.9),
        **_COMMON,
    ).fit(x_fit, y_fit)
    x_pred, _ = _panel(n_entities=4, rows=2)  # all 2-row -> below floor
    out = est.predict_with_attention(x_pred)
    assert np.isnan(out.predictions).all()
    assert np.isfinite(out.attention_weights).all()


def test_device_output_values_equal_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_cpu(monkeypatch)
    x, y = _panel()
    est = _clf().fit(x, y)
    npy = est.predict_with_attention(x)
    dev = est.predict_with_attention(x, device="cpu")
    for name in (
        "probabilities",
        "logits",
        "predictions",
        "var_selection_weights",
        "static_var_selection_weights",
        "attention_weights",
        "padding_mask",
        "entity_id",
    ):
        a = np.asarray(getattr(npy, name))
        b = getattr(dev, name).numpy()
        assert np.allclose(a, b, equal_nan=True), name
