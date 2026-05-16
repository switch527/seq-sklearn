"""Post-hoc calibration + threshold tuning (per architecture A9 / F5).

Concrete calibrators implement the internal
:class:`~seq_sklearn.calibration._protocol._Calibrator` contract and
are constructed by the estimator base from ``calibration_strategy``;
:class:`~seq_sklearn.calibration.threshold.ThresholdTuner` is the
separate binary decision-threshold search. Import the concrete classes
from their submodules; this package exposes no re-export surface (cf.
``seq_sklearn.models``).
"""
