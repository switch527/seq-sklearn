"""``BaseSequenceClassifier``: classifier overlay on the estimator shell (A2 / F1.1).

Adds the `ClassifierMixin` contract: ``classes_`` (LabelEncoder-sorted),
``predict_proba`` / ``predict``, the optional binary
``decision_threshold_`` (ABSENT on multiclass or when threshold tuning
is off, per A2), and the temperature / platt / isotonic calibrator
dispatch. Still abstract: a concrete family supplies
:meth:`_build_backbone_head`.
"""

from typing import cast

import numpy as np
import pandas as pd
import torch
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from torch import Tensor

from seq_sklearn.calibration._protocol import _Calibrator
from seq_sklearn.calibration.classification import (
    IsotonicCalibrator,
    PlattScaling,
    TemperatureScaling,
)
from seq_sklearn.calibration.threshold import ThresholdTuner
from seq_sklearn.data.tabular_to_sequence import TabularToSequence
from seq_sklearn.models._base import BaseSequenceEstimator

__all__ = ["BaseSequenceClassifier"]


class BaseSequenceClassifier(ClassifierMixin, BaseSequenceEstimator):
    """Classifier family base (ClassifierMixin + the estimator shell)."""

    # Classifier fit-state (F1.1); decision_threshold_ stays ABSENT
    # unless binary + threshold_tuning (A2), so it is annotation-only.
    classes_: np.ndarray
    n_classes_: int
    decision_threshold_: float
    _label_encoder: LabelEncoder

    def _calibrator_task(self) -> str:
        return "binary" if self.task_type == "binary" else "multiclass"

    def _encode_targets(self, y: object) -> np.ndarray:
        """Label-encode ``y``; fit the encoder once (training), reuse after.

        The training call fits the :class:`LabelEncoder` and pins
        ``classes_``; a later call (an explicit ``calibration_set``)
        transforms with that same encoder so calibration labels map to
        the training class codes.
        """
        y_arr = np.asarray(y).reshape(-1)
        if not hasattr(self, "_label_encoder"):
            self._label_encoder = LabelEncoder().fit(y_arr)
            self.classes_ = np.asarray(self._label_encoder.classes_)
        return self._label_encoder.transform(y_arr).astype(np.int64)

    def _set_target_fit_state(self, y: np.ndarray) -> None:  # noqa: ARG002
        # classes_ is pinned in _encode_targets; head width is derived
        # from it by the concrete _build_backbone_head.
        self.n_classes_ = int(np.asarray(self.classes_).shape[0])

    def _head_out_dim(self) -> int:
        """Tensor projection width: 1 for binary, ``n_classes`` otherwise."""
        return 1 if self.task_type == "binary" else self.n_classes_

    def _make_calibrator(self) -> _Calibrator | None:
        strategy = self.calibration_strategy
        if strategy == "none":
            return None
        if strategy == "temperature":
            return TemperatureScaling(self._calibrator_task())
        if strategy == "platt":
            return PlattScaling()
        if strategy == "isotonic":
            return IsotonicCalibrator(self._calibrator_task())
        raise ValueError(f"calibration_strategy={strategy!r} is not a classifier strategy")

    def _build_calibrator_from(self, blob: dict[str, object]) -> _Calibrator:
        strategy = self.calibration_strategy
        if strategy == "temperature":
            return TemperatureScaling.deserialize(blob)
        if strategy == "platt":
            return PlattScaling.deserialize(blob)
        return IsotonicCalibrator.deserialize(blob)

    def _family_state(self) -> dict[str, object]:
        return {
            "classes_": np.asarray(self.classes_).tolist(),
            "n_classes_": self.n_classes_,
            "decision_threshold_": getattr(self, "decision_threshold_", None),
        }

    def _restore_family_state(self, state: dict[str, object]) -> None:
        self.classes_ = np.asarray(state["classes_"])
        self.n_classes_ = int(state["n_classes_"])  # type: ignore[arg-type]
        threshold = state.get("decision_threshold_")
        if threshold is not None:
            self.decision_threshold_ = float(threshold)  # type: ignore[arg-type]

    def _proba_from_raw(self, raw: Tensor) -> np.ndarray:
        """Calibrated (or activated) class-probability matrix ``(N, n_classes)``."""
        if self.calibrator_ is not None:
            calibrated = self.calibrator_.transform(raw)
            if self.task_type == "binary":
                pos = calibrated.numpy().reshape(-1)
                return np.column_stack([1.0 - pos, pos])
            return calibrated.numpy()
        if self.task_type == "binary":
            pos = torch.sigmoid(raw.reshape(-1)).numpy()
            return np.column_stack([1.0 - pos, pos])
        return torch.softmax(raw, dim=1).numpy()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        """Class probabilities ``(n_samples, n_classes)`` (F1).

        Rows of entities below ``min_periods_predict`` are NaN-filled
        (correct shape, never zero-filled) per the F NaN-in-output
        contract; the aggregated breach WARNING fires once per call
        inside ``transform``.
        """
        raw, below = self._predict_raw(X)
        proba = self._proba_from_raw(raw)
        proba[below] = np.nan
        return proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        """Class labels; binary honours ``decision_threshold_`` when tuned (F1)."""
        proba = self.predict_proba(X)
        if self.task_type == "binary" and hasattr(self, "decision_threshold_"):
            idx = (proba[:, 1] >= self.decision_threshold_).astype(np.int64)
        else:
            idx = proba.argmax(axis=1)
        return np.asarray(self.classes_)[idx]

    def _post_fit(
        self,
        transformer: TabularToSequence,
        X: pd.DataFrame,  # noqa: N803
        calibration_set: tuple[pd.DataFrame, object] | None,
    ) -> None:
        """Tune ``decision_threshold_`` on the cal fold (binary + tuning only).

        Left ABSENT on multiclass or when ``threshold_tuning`` is off so
        ``hasattr(est, 'decision_threshold_')`` is ``False`` there (A2).
        """
        if self.task_type != "binary" or not self.threshold_tuning:
            return
        raw, targets = self._calibration_fold(transformer, X, calibration_set)
        if self.calibrator_ is not None:
            proba_pos = self.calibrator_.transform(raw).reshape(-1)
        else:
            proba_pos = torch.sigmoid(raw.reshape(-1))
        tuner = ThresholdTuner(self.threshold_metric)
        tuner.fit(proba_pos, targets)
        # ThresholdTuner.fit always sets threshold_; cast (not assert,
        # which python -O strips) narrows the float | None for pyright.
        self.decision_threshold_ = float(cast("float", tuner.threshold_))
