"""Calibrators are CPU-internal: the device-normalization contract.

The concrete calibrators are numpy / sklearn bound. ``fit`` /
``transform`` accept a ``raw_output`` on any device, normalize it to
CPU at the boundary, and return a CPU ``float64`` tensor, so a
GPU-trained backbone's CUDA calibration-fold outputs do not crash the
fit path (cross-device op or ``Tensor.numpy()`` on CUDA).

The boundary normalization is mutation-pinned on the CPU-default
runner: a ``requires_grad`` input must come back detached, which is
only true if ``.detach().cpu()`` actually ran (CPU-in / CPU-out alone
cannot catch a dropped ``_cpu`` call). The CUDA round-trip is an
additional ``pytest.mark.gpu`` guard for runners that have a GPU.
"""

from collections.abc import Callable

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


def test_transform_detaches_requires_grad_input_cpu_visible() -> None:
    # CPU-visible mutation pin for the boundary normalization: a
    # requires_grad input can only come back without a grad_fn if
    # .detach() (inside _cpu / the .detach().cpu() boundary) actually
    # ran. CPU-in / CPU-out alone cannot catch a dropped _cpu() call;
    # this can, on the CPU-default runner.
    for cal, raw, _ in _calibrator_cases():
        out = cal.transform(raw.clone().requires_grad_(True))  # type: ignore[attr-defined]
        assert out.grad_fn is None
        assert out.requires_grad is False
        assert out.device.type == "cpu"


@pytest.mark.parametrize(
    ("cal_factory", "raw", "y"),
    [
        (lambda: TemperatureScaling("binary"), *_binary()),
        (lambda: TemperatureScaling("multiclass"), *_multiclass()),
        (PlattScaling, *_binary()),
        (lambda: IsotonicCalibrator("binary"), *_binary()),
        (lambda: IsotonicCalibrator("multiclass"), *_multiclass()),
        (lambda: ConformalCalibrator(_Q), *_reg()),
        (lambda: IsotonicQuantileCalibrator(_Q), *_reg()),
    ],
)
def test_calibrators_accept_float32_input(
    cal_factory: Callable[[], object], raw: torch.Tensor, y: torch.Tensor
) -> None:
    # float32 is the real estimator output dtype; every calibrator
    # (classification AND regression) must fit/transform without a
    # dtype crash and upcast internally to float64.
    cal = cal_factory()
    cal.fit(raw.to(torch.float32), y)  # type: ignore[attr-defined]
    out = cal.transform(raw.to(torch.float32))  # type: ignore[attr-defined]
    assert out.dtype == torch.float64
    assert torch.isfinite(out).all()


def test_threshold_tuner_accepts_requires_grad_proba() -> None:
    # ThresholdTuner is also numpy/sklearn bound: a requires_grad (and,
    # on a GPU runner, CUDA) proba must not crash .numpy().
    tuner = ThresholdTuner("f1")
    proba = torch.tensor([0.1, 0.9, 0.4, 0.8]).requires_grad_(True)
    tuner.fit(proba, torch.tensor([0, 1, 0, 1]))
    assert isinstance(tuner.threshold_, float)
    assert 0.0 <= tuner.threshold_ <= 1.0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_calibrators_accept_cuda_input_and_return_cpu() -> None:
    # GPU backbone path: a CUDA backbone yields CUDA calibration
    # outputs. fit must not crash (no cross-device op, no .numpy() on
    # CUDA) and transform must return a CPU float64 tensor.
    for cal, raw, y in _calibrator_cases():
        fresh = type(cal)(*_ctor_args(cal))
        fresh.fit(raw.cuda(), y.cuda())  # type: ignore[attr-defined]
        out = fresh.transform(raw.cuda())  # type: ignore[attr-defined]
        assert out.device.type == "cpu"
        assert out.dtype == torch.float64
        assert torch.isfinite(out).all()
    # ThresholdTuner is not a _Calibrator (no transform); exercise its
    # own CUDA boundary separately.
    b_logits, b_y = _binary()
    tuner = ThresholdTuner("f1")
    tuner.fit(torch.sigmoid(b_logits).cuda(), b_y.cuda())
    assert isinstance(tuner.threshold_, float)


def _ctor_args(cal: object) -> tuple[object, ...]:
    if isinstance(cal, TemperatureScaling | IsotonicCalibrator):
        return (cal.task,)
    if isinstance(cal, ConformalCalibrator | IsotonicQuantileCalibrator):
        return (cal.quantiles,)
    return ()
