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
