"""Fast internals: calibrator dispatch + activation branches (Phase 6a).

Targets the no-training branches that the e2e fits don't reach: the
defensive invalid-strategy guards, the no-calibrator activation paths,
and the conformal ``_build_calibrator_from`` (the dummy's crossing
quantiles stop conformal ever reaching save / load).
"""

import json

import numpy as np
import pytest
import torch

from seq_sklearn.calibration.regression import ConformalCalibrator
from tests._test_models._dummy_estimator import (
    _DummySequenceClassifier,
    _DummySequenceRegressor,
)


def test_classifier_make_calibrator_rejects_non_classifier_strategy() -> None:
    est = _DummySequenceClassifier(task_type="binary", calibration_strategy="conformal")
    with pytest.raises(ValueError, match="not a classifier strategy"):
        est._make_calibrator()


def test_regressor_make_calibrator_rejects_non_regressor_strategy() -> None:
    est = _DummySequenceRegressor(
        task_type="regression_quantile", calibration_strategy="temperature"
    )
    with pytest.raises(ValueError, match="not a regressor strategy"):
        est._make_calibrator()


def test_classifier_proba_multiclass_no_calibrator_softmaxes() -> None:
    est = _DummySequenceClassifier(task_type="multiclass")
    est.calibrator_ = None
    raw = torch.randn(5, 3)
    proba = est._proba_from_raw(raw)
    assert proba.shape == (5, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_classifier_proba_binary_no_calibrator_sigmoids() -> None:
    est = _DummySequenceClassifier(task_type="binary")
    est.calibrator_ = None
    raw = torch.randn(4, 1)
    proba = est._proba_from_raw(raw)
    assert proba.shape == (4, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_regressor_build_calibrator_from_conformal_blob() -> None:
    est = _DummySequenceRegressor(
        task_type="regression_quantile",
        quantiles=(0.1, 0.5, 0.9),
        calibration_strategy="conformal",
    )
    # Monotone synthetic quantiles so ConformalCalibrator fits cleanly.
    g = torch.Generator().manual_seed(0)
    y = torch.randn(64, generator=g, dtype=torch.float64)
    pred = (y.reshape(-1, 1) + torch.tensor([-1.0, 0.0, 1.0])).sort(dim=1).values
    cal = ConformalCalibrator((0.1, 0.5, 0.9))
    cal.fit(pred, y)
    blob = json.loads(json.dumps(cal.serialize()))
    restored = est._build_calibrator_from(blob)
    assert isinstance(restored, ConformalCalibrator)
    assert torch.equal(restored.transform(pred), cal.transform(pred))
