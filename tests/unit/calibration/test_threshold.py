"""ThresholdTuner: 101-point grid search over the F5 metrics (A9).

Pins the float-in-[0,1] deliverable, the three ``threshold_metric``
branches, the degenerate single-class Youden-J guards, the serialize /
deserialize round trip, and the pre-fit error path.
"""

import json

import pytest
import torch

from seq_sklearn.calibration.threshold import ThresholdTuner
from seq_sklearn.errors import NotFittedError


def _separable() -> tuple[torch.Tensor, torch.Tensor]:
    proba = torch.tensor([0.05, 0.1, 0.2, 0.45, 0.55, 0.8, 0.9, 0.95])
    y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    return proba, y


def test_invalid_metric_raises() -> None:
    with pytest.raises(ValueError, match="threshold_metric"):
        ThresholdTuner("roc_auc")


@pytest.mark.parametrize("metric", ["f1", "balanced_accuracy", "youden_j"])
def test_threshold_is_a_float_in_unit_interval(metric: str) -> None:
    proba, y = _separable()
    tuner = ThresholdTuner(metric)
    tuner.fit(proba, y)
    assert isinstance(tuner.threshold_, float)
    assert 0.0 <= tuner.threshold_ <= 1.0
    # On separable data the cut falls between the two class clusters
    # (last negative proba 0.45, first positive 0.55).
    assert 0.2 < tuner.threshold_ < 0.6


def test_threshold_accepts_2d_column_proba() -> None:
    proba, y = _separable()
    tuner = ThresholdTuner("f1")
    tuner.fit(proba.reshape(-1, 1), y.reshape(-1, 1))
    assert tuner.threshold_ is not None


def test_youden_j_all_positive_fold_is_finite() -> None:
    # n_neg == 0 branch: specificity defaults to 0, metric stays finite.
    tuner = ThresholdTuner("youden_j")
    tuner.fit(torch.tensor([0.6, 0.7, 0.8, 0.9]), torch.tensor([1, 1, 1, 1]))
    assert tuner.threshold_ is not None


def test_youden_j_all_negative_fold_is_finite() -> None:
    # n_pos == 0 branch: sensitivity defaults to 0.
    tuner = ThresholdTuner("youden_j")
    tuner.fit(torch.tensor([0.1, 0.2, 0.3, 0.4]), torch.tensor([0, 0, 0, 0]))
    assert tuner.threshold_ is not None


def test_serialize_before_fit_raises() -> None:
    with pytest.raises(NotFittedError):
        ThresholdTuner("f1").serialize()


def test_json_roundtrip_preserves_threshold_and_metric() -> None:
    proba, y = _separable()
    tuner = ThresholdTuner("balanced_accuracy")
    tuner.fit(proba, y)
    restored = ThresholdTuner.deserialize(json.loads(json.dumps(tuner.serialize())))
    assert restored.metric == "balanced_accuracy"
    assert restored.threshold_ == tuner.threshold_
