"""Calibrators are CPU-internal (Gemini final-pass CRITICAL).

The concrete calibrators are numpy / sklearn bound. ``fit`` /
``transform`` must accept a ``raw_output`` on any device, normalize it
to CPU at the boundary, and return a CPU ``float64`` tensor, so a
GPU-trained backbone's CUDA calibration-fold outputs do not crash the
fit path (cross-device op or ``Tensor.numpy()`` on CUDA). The CPU
assertions run everywhere; the CUDA round-trip is ``pytest.mark.gpu``
(CI default is CPU-only per testing.md).
"""

import pytest
import torch

from seq_sklearn.calibration.classification import (
    IsotonicCalibrator,
    PlattScaling,
    TemperatureScaling,
)
from seq_sklearn.calibration.regression import (
    ConformalCalibrator,
    IsotonicQuantileCalibrator,
)
from seq_sklearn.calibration.threshold import ThresholdTuner

_Q = (0.1, 0.5, 0.9)


def _binary() -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(64, generator=g)
    y = (torch.arange(64) % 2).long()
    return logits, y


def _multiclass() -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(60, 3, generator=g)
    y = (torch.arange(60) % 3).long()
    return logits, y


def _reg() -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(2)
    y = torch.randn(64, generator=g)
    pred = (y.reshape(-1, 1) + torch.tensor([-1.0, 0.0, 1.0])).sort(dim=1).values
    return pred, y


def _calibrator_cases() -> list[tuple[object, torch.Tensor, torch.Tensor]]:
    b_logits, b_y = _binary()
    m_logits, m_y = _multiclass()
    r_pred, r_y = _reg()
    temp_b = TemperatureScaling("binary")
    temp_b.fit(b_logits, b_y)
    temp_m = TemperatureScaling("multiclass")
    temp_m.fit(m_logits, m_y)
    platt = PlattScaling()
    platt.fit(b_logits, b_y)
    iso_b = IsotonicCalibrator("binary")
    iso_b.fit(b_logits, b_y)
    iso_m = IsotonicCalibrator("multiclass")
    iso_m.fit(m_logits, m_y)
    conf = ConformalCalibrator(_Q)
    conf.fit(r_pred, r_y)
    isoq = IsotonicQuantileCalibrator(_Q)
    isoq.fit(r_pred, r_y)
    return [
        (temp_b, b_logits, b_y),
        (temp_m, m_logits, m_y),
        (platt, b_logits, b_y),
        (iso_b, b_logits, b_y),
        (iso_m, m_logits, m_y),
        (conf, r_pred, r_y),
        (isoq, r_pred, r_y),
    ]


def test_transform_returns_cpu_float64_tensor() -> None:
    for cal, raw, _ in _calibrator_cases():
        out = cal.transform(raw)  # type: ignore[attr-defined]
        assert out.device.type == "cpu"
        assert out.dtype == torch.float64


def test_calibrators_accept_float32_input() -> None:
    # float32 logits (the real estimator dtype) must fit/transform
    # without a dtype crash; the calibrator upcasts internally.
    logits, y = _binary()
    for cal in (TemperatureScaling("binary"), PlattScaling(), IsotonicCalibrator("binary")):
        cal.fit(logits.to(torch.float32), y)
        out = cal.transform(logits.to(torch.float32))
        assert torch.isfinite(out).all()


def test_threshold_tuner_accepts_arbitrary_device_proba() -> None:
    tuner = ThresholdTuner("f1")
    tuner.fit(torch.tensor([0.1, 0.9, 0.4, 0.8]), torch.tensor([0, 1, 0, 1]))
    assert tuner.threshold_ is not None


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_calibrators_accept_cuda_input_and_return_cpu() -> None:
    # The Phase 6 GPU path: a CUDA backbone yields CUDA calibration
    # outputs. fit must not crash (no cross-device op, no .numpy() on
    # CUDA) and transform must return a CPU float64 tensor.
    for cal, raw, y in _calibrator_cases():
        fresh = type(cal)(*_ctor_args(cal))
        fresh.fit(raw.cuda(), y.cuda())  # type: ignore[attr-defined]
        out = fresh.transform(raw.cuda())  # type: ignore[attr-defined]
        assert out.device.type == "cpu"
        assert out.dtype == torch.float64
        assert torch.isfinite(out).all()


def _ctor_args(cal: object) -> tuple[object, ...]:
    if isinstance(cal, TemperatureScaling | IsotonicCalibrator):
        return (cal.task,)
    if isinstance(cal, ConformalCalibrator | IsotonicQuantileCalibrator):
        return (cal.quantiles,)
    return ()
