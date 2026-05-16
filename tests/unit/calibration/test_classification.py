"""Classification calibrators (A9 / F5 / N1).

Each calibrator is exercised standalone with hand-crafted
``(logits, y_true)`` tensors (no estimator, per A9). Coverage includes
the JSON-roundtripped serialize / deserialize byte-equality (N1), the
F11 ``calibration.fit`` / ``calibration.small_set`` emission, the
pre-fit error paths, and a hypothesis shape/finiteness property.
"""

import json
import logging

import hypothesis.strategies as st
import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays

from seq_sklearn.calibration._protocol import _Calibrator
from seq_sklearn.calibration.classification import (
    IsotonicCalibrator,
    PlattScaling,
    TemperatureScaling,
)
from seq_sklearn.errors import NotFittedError, TrainingError
from seq_sklearn.logging import Event

_CAL = "seq_sklearn.calibration"


def _binary_data(n: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(n, generator=g, dtype=torch.float64)
    y = (logits + 0.3 * torch.randn(n, generator=g, dtype=torch.float64) > 0).long()
    return logits, y


def _multiclass_data(n: int = 90, k: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(n, k, generator=g, dtype=torch.float64)
    y = logits.argmax(dim=1)
    return logits, y


# --- TemperatureScaling --------------------------------------------------


def test_temperature_invalid_task_raises() -> None:
    with pytest.raises(ValueError, match=r"binary.*multiclass"):
        TemperatureScaling("regression_point")


@pytest.mark.parametrize("task", ["binary", "multiclass"])
def test_temperature_fit_transform_shapes(task: str) -> None:
    logits, y = _binary_data() if task == "binary" else _multiclass_data()
    cal = TemperatureScaling(task)
    cal.fit(logits, y)
    out = cal.transform(logits)
    if task == "binary":
        assert out.shape == (logits.shape[0],)
        assert torch.all((out >= 0) & (out <= 1))
    else:
        assert out.shape == logits.shape
        assert torch.allclose(out.sum(dim=1), torch.ones(logits.shape[0], dtype=torch.float64))
    assert torch.isfinite(out).all()


def test_temperature_emits_calibration_fit_and_small_set(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logits, y = _binary_data(40)  # < 100 -> small_set warning
    with caplog.at_level(logging.INFO, logger=_CAL):
        TemperatureScaling("binary").fit(logits, y)
    small = [r for r in caplog.records if r.event == Event.CALIBRATION_SMALL_SET]
    fit = [r for r in caplog.records if r.event == Event.CALIBRATION_FIT]
    assert len(small) == 1
    assert small[0].levelno == logging.WARNING
    assert small[0].payload == {
        "strategy": "temperature",
        "cal_size": 40,
        "min_recommended": 100,
    }
    assert len(fit) == 1
    assert set(fit[0].payload) == {"strategy", "cal_size", "pre_ece", "post_ece"}


def test_temperature_large_set_no_small_set_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logits, y = _multiclass_data(120, 3)
    with caplog.at_level(logging.INFO, logger=_CAL):
        TemperatureScaling("multiclass").fit(logits, y)
    assert not [r for r in caplog.records if r.event == Event.CALIBRATION_SMALL_SET]
    assert [r for r in caplog.records if r.event == Event.CALIBRATION_FIT]


def test_temperature_non_finite_input_raises_training_error() -> None:
    logits, y = _binary_data()
    logits[0] = float("inf")
    with pytest.raises(TrainingError, match="diverged"):
        TemperatureScaling("binary").fit(logits, y)


def test_temperature_transform_before_fit_raises() -> None:
    with pytest.raises(NotFittedError):
        TemperatureScaling("binary").transform(torch.zeros(4))


def test_temperature_serialize_before_fit_raises() -> None:
    with pytest.raises(NotFittedError):
        TemperatureScaling("binary").serialize()


@pytest.mark.parametrize("task", ["binary", "multiclass"])
def test_temperature_json_roundtrip_byte_equal(task: str) -> None:
    logits, y = _binary_data() if task == "binary" else _multiclass_data()
    cal = TemperatureScaling(task)
    cal.fit(logits, y)
    restored = TemperatureScaling.deserialize(json.loads(json.dumps(cal.serialize())))
    assert torch.equal(cal.transform(logits), restored.transform(logits))


# --- PlattScaling --------------------------------------------------------


def test_platt_fit_transform_and_roundtrip() -> None:
    logits, y = _binary_data()
    cal = PlattScaling()
    cal.fit(logits, y)
    out = cal.transform(logits)
    assert out.shape == (logits.shape[0],)
    assert torch.all((out >= 0) & (out <= 1))
    restored = PlattScaling.deserialize(json.loads(json.dumps(cal.serialize())))
    assert torch.equal(cal.transform(logits), restored.transform(logits))


def test_platt_non_finite_input_raises_training_error() -> None:
    logits, y = _binary_data()
    logits[1] = float("nan")
    with pytest.raises(TrainingError, match="diverged"):
        PlattScaling().fit(logits, y)


def test_platt_before_fit_errors() -> None:
    with pytest.raises(NotFittedError):
        PlattScaling().transform(torch.zeros(4))
    with pytest.raises(NotFittedError):
        PlattScaling().serialize()


def test_platt_accepts_2d_column_logits() -> None:
    logits, y = _binary_data()
    cal = PlattScaling()
    cal.fit(logits.reshape(-1, 1), y.reshape(-1, 1))
    assert cal.transform(logits.reshape(-1, 1)).shape == (logits.shape[0],)


# --- IsotonicCalibrator --------------------------------------------------


def test_isotonic_invalid_task_raises() -> None:
    with pytest.raises(ValueError, match=r"binary.*multiclass"):
        IsotonicCalibrator("regression_point")


def test_isotonic_binary_fit_transform_roundtrip() -> None:
    logits, y = _binary_data()
    cal = IsotonicCalibrator("binary")
    cal.fit(logits, y)
    out = cal.transform(logits)
    assert out.shape == (logits.shape[0],)
    assert torch.all((out >= 0) & (out <= 1))
    restored = IsotonicCalibrator.deserialize(json.loads(json.dumps(cal.serialize())))
    assert torch.equal(cal.transform(logits), restored.transform(logits))


def test_isotonic_multiclass_rows_sum_to_one_and_roundtrip() -> None:
    logits, y = _multiclass_data()
    cal = IsotonicCalibrator("multiclass")
    cal.fit(logits, y)
    out = cal.transform(logits)
    assert out.shape == logits.shape
    assert torch.allclose(out.sum(dim=1), torch.ones(logits.shape[0], dtype=torch.float64))
    restored = IsotonicCalibrator.deserialize(json.loads(json.dumps(cal.serialize())))
    assert torch.equal(cal.transform(logits), restored.transform(logits))


def test_isotonic_multiclass_zero_row_falls_back_to_uniform() -> None:
    # Defensive branch: every per-class regressor maps the query to 0
    # (clipped below its min knot) so the row sum is 0 -> uniform.
    cal = IsotonicCalibrator("multiclass")
    cal.fit(*_multiclass_data())
    for m in cal._models:  # type: ignore[union-attr]
        # Force each regressor to predict 0 everywhere.
        m.fit(np.array([0.0, 1.0]), np.array([0.0, 0.0]))
    out = cal.transform(_multiclass_data()[0])
    assert torch.allclose(out, torch.full_like(out, 1.0 / out.shape[1]))


def test_isotonic_before_fit_errors() -> None:
    with pytest.raises(NotFittedError):
        IsotonicCalibrator("binary").transform(torch.zeros(4))
    with pytest.raises(NotFittedError):
        IsotonicCalibrator("binary").serialize()


# --- protocol + hypothesis ----------------------------------------------


@settings(max_examples=30, deadline=None)
@given(
    arrays(np.float64, st.integers(16, 48).map(lambda n: (n,)), elements=st.floats(-4, 4)),
)
def test_binary_calibrators_preserve_shape_and_finiteness(logits_np: np.ndarray) -> None:
    logits = torch.from_numpy(logits_np)
    g = torch.Generator().manual_seed(7)
    # Labels correlated with the logits (20% flip) so the temperature
    # fit stays well-posed; the property under test is shape/finiteness,
    # not robustness to a degenerate (separable) calibration fold.
    flip = torch.rand(logits.shape[0], generator=g) < 0.2
    y = ((logits > 0) ^ flip).long()
    for cal in (TemperatureScaling("binary"), PlattScaling(), IsotonicCalibrator("binary")):
        assert isinstance(cal, _Calibrator)
        cal.fit(logits, y)
        out = cal.transform(logits)
        assert out.shape == (logits.shape[0],)
        assert torch.isfinite(out).all()
