"""Decision-threshold tuner (per architecture A9 / requirements F5).

:class:`ThresholdTuner` is the binary-classifier post-hoc threshold
search. It runs AFTER calibration on the same calibration fold and is
independent of the loss / imbalance / calibration strategy (F5). It is
NOT a :class:`~seq_sklearn.calibration._protocol._Calibrator`: it
consumes calibrated POSITIVE-CLASS PROBABILITIES (not logits) and
returns a scalar threshold, a different contract.

The search is the F5 fixed grid: 101 evenly spaced points on
``[0, 1]``; the threshold maximising ``threshold_metric`` is kept, ties
broken toward the lower threshold (higher recall) by taking the first
argmax.
"""

import logging

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch import Tensor

from seq_sklearn.errors import NotFittedError

__all__ = ["ThresholdTuner"]

logger = logging.getLogger(__name__)

_GRID = np.linspace(0.0, 1.0, 101)
_METRICS = ("f1", "balanced_accuracy", "youden_j")


def _youden_j(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Sensitivity + specificity - 1, computed from the 2x2 counts."""
    pos = y_true == 1
    neg = ~pos
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    # A degenerate single-class calibration fold makes one rate
    # undefined; treat the absent class's rate as 0 so the metric stays
    # finite and the grid search still returns a threshold.
    sensitivity = float((y_pred[pos] == 1).mean()) if n_pos else 0.0
    specificity = float((y_pred[neg] == 0).mean()) if n_neg else 0.0
    return sensitivity + specificity - 1.0


def _score(metric: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if metric == "f1":
        return float(f1_score(y_true, y_pred, zero_division=0))  # type: ignore[arg-type]
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    return _youden_j(y_true, y_pred)


class ThresholdTuner:
    """Grid-search the decision threshold maximising ``metric``.

    ``metric`` is one of ``"f1"``, ``"balanced_accuracy"``,
    ``"youden_j"`` (the F5 ``threshold_metric`` domain). The fitted
    threshold is exposed as :attr:`threshold_` and consulted by
    ``predict`` (raw probabilities are unaffected).
    """

    def __init__(self, metric: str = "f1") -> None:
        if metric not in _METRICS:
            raise ValueError(f"threshold_metric must be one of {_METRICS}, got {metric!r}")
        self.metric = metric
        self.threshold_: float | None = None

    def fit(self, proba: Tensor, y_true: Tensor) -> None:
        """Pick the grid threshold maximising ``metric`` on ``(proba, y)``.

        ``proba`` is ``(N,)`` / ``(N, 1)`` calibrated positive-class
        probability; ``y_true`` is ``(N,)`` binary labels.
        """
        # .cpu() before .numpy(): the tuner is sklearn/numpy bound and
        # CPU-internal, so a CUDA proba / y_true from a GPU estimator
        # does not crash Tensor.numpy().
        p = proba.detach().cpu().reshape(-1).to(torch.float64).numpy()
        y = y_true.detach().cpu().reshape(-1).long().numpy()
        scores = [_score(self.metric, y, (p >= t).astype(int)) for t in _GRID]
        best = int(np.argmax(scores))
        self.threshold_ = float(_GRID[best])
        logger.info(
            "threshold tuned: metric=%s threshold=%.4f score=%.4f",
            self.metric,
            self.threshold_,
            scores[best],
        )

    def serialize(self) -> dict[str, object]:
        if self.threshold_ is None:
            raise NotFittedError("ThresholdTuner.serialize before fit")
        return {"metric": self.metric, "threshold": self.threshold_}

    @classmethod
    def deserialize(cls, blob: dict[str, object]) -> "ThresholdTuner":
        obj = cls(metric=str(blob["metric"]))
        obj.threshold_ = float(blob["threshold"])  # type: ignore[arg-type]
        return obj
