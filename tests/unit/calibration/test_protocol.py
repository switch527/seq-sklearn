"""``_Calibrator`` structural-conformance checks (A9).

Every concrete calibrator must satisfy the runtime-checkable protocol
so the Phase 6 save / load path can ``isinstance``-guard a
deserialized object. ``ThresholdTuner`` deliberately does NOT (it has a
different consume/return contract) and that exclusion is pinned here.
"""

import torch

from seq_sklearn.calibration._protocol import _Calibrator
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


def test_every_calibrator_satisfies_the_protocol() -> None:
    instances = [
        TemperatureScaling("binary"),
        PlattScaling(),
        IsotonicCalibrator("multiclass"),
        ConformalCalibrator((0.1, 0.5, 0.9)),
        IsotonicQuantileCalibrator((0.5,)),
    ]
    for inst in instances:
        assert isinstance(inst, _Calibrator)


def test_threshold_tuner_is_not_a_calibrator() -> None:
    # Different contract (proba -> scalar, no transform); must not be
    # mistaken for a _Calibrator by the save/load isinstance guard.
    assert not isinstance(ThresholdTuner(), _Calibrator)
    # It still works standalone on a synthetic pair.
    tuner = ThresholdTuner("f1")
    tuner.fit(torch.tensor([0.1, 0.9, 0.4, 0.8]), torch.tensor([0, 1, 0, 1]))
    assert tuner.threshold_ is not None
