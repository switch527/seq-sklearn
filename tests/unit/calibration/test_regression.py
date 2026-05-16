"""Regression calibrators (A9 / F5 / F9 / N1).

Covers the F9 non-monotone ``TrainingError`` (the deliberately
non-monotone ``(N, len(quantiles))`` tensor from A9), the JSON
round-trip byte-equality, quantile-vector validation, shape validation,
and the F11 coverage emission.
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
from seq_sklearn.calibration.regression import (
    ConformalCalibrator,
    IsotonicQuantileCalibrator,
)
from seq_sklearn.errors import NotFittedError, TrainingError
from seq_sklearn.logging import Event

_CAL = "seq_sklearn.calibration"
_Q = (0.1, 0.5, 0.9)


def _reg_data(n: int = 80) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(2)
    y = torch.randn(n, generator=g, dtype=torch.float64)
    # Monotone-by-construction predicted quantiles around y.
    base = y.reshape(-1, 1)
    offs = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    pred = base + offs + 0.1 * torch.randn(n, 3, generator=g, dtype=torch.float64)
    pred, _ = pred.sort(dim=1)
    return pred, y


# --- quantile-vector validation -----------------------------------------


@pytest.mark.parametrize(
    ("bad", "match"),
    [
        ((), "non-empty"),
        ((0.5, 1.5), r"strictly in \(0, 1\)"),
        ((0.0, 0.5), r"strictly in \(0, 1\)"),
        ((0.5, 0.5), "strictly increasing"),
        ((0.9, 0.1), "strictly increasing"),
    ],
)
def test_invalid_quantiles_raise(bad: tuple[float, ...], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ConformalCalibrator(bad)


# --- ConformalCalibrator ------------------------------------------------


def test_conformal_fit_transform_and_roundtrip() -> None:
    pred, y = _reg_data()
    cal = ConformalCalibrator(_Q)
    cal.fit(pred, y)
    out = cal.transform(pred)
    assert out.shape == pred.shape
    assert torch.isfinite(out).all()
    restored = ConformalCalibrator.deserialize(json.loads(json.dumps(cal.serialize())))
    assert torch.equal(cal.transform(pred), restored.transform(pred))


def test_conformal_non_monotone_raises_training_error() -> None:
    # A9's exact construction: a deliberately non-monotone (N, 3) tensor.
    pred = torch.tensor([[0.5, 0.3, 0.7]], dtype=torch.float64).expand(64, 3)
    y = torch.zeros(64, dtype=torch.float64)
    with pytest.raises(TrainingError, match="non-monotone"):
        ConformalCalibrator(_Q).fit(pred, y)


def test_conformal_two_quantile_crossing_raises_training_error() -> None:
    # Q == 2 with pred[:, 0] > pred[:, 1] every row: pins the
    # `pred.shape[1] > 1` guard (a `> 2` mutation would skip this).
    pred = torch.tensor([[0.9, 0.1]], dtype=torch.float64).expand(50, 2)
    y = torch.zeros(50, dtype=torch.float64)
    with pytest.raises(TrainingError, match="non-monotone"):
        ConformalCalibrator((0.25, 0.75)).fit(pred, y)


def test_conformal_single_quantile_skips_monotone_check() -> None:
    g = torch.Generator().manual_seed(3)
    pred = torch.randn(50, 1, generator=g, dtype=torch.float64)
    y = pred.reshape(-1) + 0.1
    cal = ConformalCalibrator((0.5,))
    cal.fit(pred, y)
    assert cal.transform(pred).shape == (50, 1)


@pytest.mark.parametrize(
    "bad",
    [
        torch.zeros(10, 2, dtype=torch.float64),  # wrong column count
        torch.zeros(10, dtype=torch.float64),  # 1-D: ndim != 2 disjunct
    ],
)
def test_conformal_transform_shape_mismatch_raises(bad: torch.Tensor) -> None:
    pred, y = _reg_data()
    cal = ConformalCalibrator(_Q)
    cal.fit(pred, y)
    with pytest.raises(ValueError, match="shape"):
        cal.transform(bad)


def test_conformal_before_fit_errors() -> None:
    cal = ConformalCalibrator(_Q)
    with pytest.raises(NotFittedError):
        cal.transform(torch.zeros(4, 3))
    with pytest.raises(NotFittedError):
        cal.serialize()


def test_conformal_emits_coverage_payload(caplog: pytest.LogCaptureFixture) -> None:
    pred, y = _reg_data(40)
    with caplog.at_level(logging.INFO, logger=_CAL):
        ConformalCalibrator(_Q).fit(pred, y)
    fit = [r for r in caplog.records if r.event == Event.CALIBRATION_FIT]
    small = [r for r in caplog.records if r.event == Event.CALIBRATION_SMALL_SET]
    assert len(small) == 1
    assert set(fit[0].payload) == {
        "strategy",
        "cal_size",
        "pre_coverage",
        "post_coverage",
    }


# --- IsotonicQuantileCalibrator -----------------------------------------


def test_isotonic_quantile_fit_transform_and_roundtrip() -> None:
    pred, y = _reg_data()
    cal = IsotonicQuantileCalibrator(_Q)
    cal.fit(pred, y)
    out = cal.transform(pred)
    assert out.shape == pred.shape
    assert torch.isfinite(out).all()
    restored = IsotonicQuantileCalibrator.deserialize(json.loads(json.dumps(cal.serialize())))
    assert torch.equal(cal.transform(pred), restored.transform(pred))


def test_isotonic_quantile_high_tau_uses_last_knot() -> None:
    # tau == 0.999 so the fitted residual CDF never reaches it on a
    # small fold -> the "reached empty -> idx = n-1" branch.
    g = torch.Generator().manual_seed(4)
    pred = torch.randn(20, 1, generator=g, dtype=torch.float64)
    y = pred.reshape(-1) + 0.05
    cal = IsotonicQuantileCalibrator((0.999,))
    cal.fit(pred, y)
    assert cal.transform(pred).shape == (20, 1)


def test_isotonic_quantile_before_fit_errors() -> None:
    cal = IsotonicQuantileCalibrator(_Q)
    with pytest.raises(NotFittedError):
        cal.transform(torch.zeros(4, 3))
    with pytest.raises(NotFittedError):
        cal.serialize()


def test_isotonic_quantile_emits_coverage_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pred, y = _reg_data(150)  # >= 100 -> no small_set
    with caplog.at_level(logging.INFO, logger=_CAL):
        IsotonicQuantileCalibrator(_Q).fit(pred, y)
    assert not [r for r in caplog.records if r.event == Event.CALIBRATION_SMALL_SET]
    fit = [r for r in caplog.records if r.event == Event.CALIBRATION_FIT]
    assert set(fit[0].payload) == {
        "strategy",
        "cal_size",
        "pre_coverage",
        "post_coverage",
    }


@settings(max_examples=25, deadline=None)
@given(
    arrays(
        np.float64,
        st.integers(20, 60).map(lambda n: (n, 3)),
        elements=st.floats(-3, 3),
    ),
)
def test_regression_calibrators_preserve_shape(pred_np: np.ndarray) -> None:
    pred = torch.from_numpy(np.sort(pred_np, axis=1))
    g = torch.Generator().manual_seed(9)
    y = pred[:, 1] + 0.2 * torch.randn(pred.shape[0], generator=g, dtype=torch.float64)
    # Pre-sorted columns => the F9 crossing guard never fires, so both
    # regression strategies are exercised (implementation_plan.md:780).
    for cal in (ConformalCalibrator(_Q), IsotonicQuantileCalibrator(_Q)):
        assert isinstance(cal, _Calibrator)
        cal.fit(pred, y)
        out = cal.transform(pred)
        assert out.shape == pred.shape
        assert torch.isfinite(out).all()
