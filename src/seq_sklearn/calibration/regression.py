"""Regression calibrators (per architecture A9 / requirements F5).

Two concrete ``_Calibrator`` implementations applied at
``predict_quantiles`` time on the held-out calibration fold. Both
return calibrated quantile VALUES (``pred`` plus an additive
correction, per the :mod:`seq_sklearn.calibration._protocol`
contract); they differ in how the correction is estimated:

* :class:`ConformalCalibrator` (split-conformal): per-column
  recentring offsets, but only after rejecting a crossing
  predicted-quantile vector with
  :class:`~seq_sklearn.errors.TrainingError` per the F9 contract (the
  check runs on the raw predictions, before the offset that would mask
  it).
* :class:`IsotonicQuantileCalibrator`: a per-column offset read off an
  isotonic-regressed residual CDF (a monotone-smoothed empirical CDF),
  inverted on its own knot grid so no cross-column interpolation is
  introduced (v1 forbids quantile interpolation, requirements F5). It
  does not enforce monotonicity (the F9 contract assigns the
  crossing-quantile failure to conformal only).

Richer joint quantile recalibration (CQR, cross-quantile
interpolation) is intentionally out of v1 scope.
"""

import logging
from itertools import pairwise

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from torch import Tensor

from seq_sklearn.calibration._events import emit_calibration_fit
from seq_sklearn.calibration._metrics import mean_quantile_coverage
from seq_sklearn.errors import NotFittedError, TrainingError

__all__ = ["ConformalCalibrator", "IsotonicQuantileCalibrator"]

logger = logging.getLogger(__name__)


def _validate_quantiles(quantiles: tuple[float, ...]) -> tuple[float, ...]:
    """Reject an empty / out-of-range / non-increasing quantile vector."""
    if not quantiles:
        raise ValueError("quantiles must be a non-empty tuple")
    q = tuple(float(v) for v in quantiles)
    if any(not (0.0 < v < 1.0) for v in q):
        raise ValueError(f"quantiles must lie strictly in (0, 1), got {q}")
    if any(b <= a for a, b in pairwise(q)):
        raise ValueError(f"quantiles must be strictly increasing, got {q}")
    return q


def _as_pred_matrix(pred_quantiles: Tensor, n_quantiles: int, where: str) -> np.ndarray:
    """Detach to a float64 ``(N, Q)`` numpy array, validating ``Q``.

    ``.cpu()`` before ``.numpy()``: regression calibrators are numpy
    bound and CPU-internal (cf. ``classification._cpu``); a CUDA
    predicted-quantile matrix from a GPU backbone would otherwise crash
    ``Tensor.numpy()``.
    """
    arr = pred_quantiles.detach().cpu().to(torch.float64).numpy()
    if arr.ndim != 2 or arr.shape[1] != n_quantiles:
        raise ValueError(
            f"{where}: expected predicted quantiles of shape (N, {n_quantiles}), "
            f"got {tuple(arr.shape)}"
        )
    if arr.shape[0] == 0:
        raise ValueError(f"{where}: calibration fold is empty (N=0)")
    return arr


class ConformalCalibrator:
    """Split-conformal per-quantile additive offset (A9 / F9).

    The F9 contract is checked FIRST: a quantile regressor that yields
    non-monotone predicted quantiles on the calibration fold (the
    undertrained-regressor failure mode) is rejected with
    :class:`TrainingError` before any offset is computed, since a
    per-quantile recentring offset would otherwise silently mask the
    crossing. Once the predicted quantiles are monotone, the offset is
    the empirical ``tau_j``-quantile of the residuals ``y - pred_j``
    (``method="higher"`` so the offset is an observed residual, no
    interpolation), recentring each column for marginal coverage.
    """

    def __init__(self, quantiles: tuple[float, ...]) -> None:
        self.quantiles = _validate_quantiles(quantiles)
        self._offsets: list[float] | None = None

    def fit(self, raw_output: Tensor, y_true: Tensor) -> None:
        raw_output, y_true = raw_output.detach().cpu(), y_true.detach().cpu()
        pred = _as_pred_matrix(raw_output, len(self.quantiles), "ConformalCalibrator.fit")
        if pred.shape[1] > 1 and bool(np.any(np.diff(pred, axis=1) < 0.0)):
            raise TrainingError(
                "non-monotone quantiles: the regressor's predicted "
                "quantiles cross on the calibration fold; the quantile "
                "regressor is likely undertrained for this fold"
            )
        y = y_true.detach().to(torch.float64).reshape(-1, 1).numpy()
        residuals = y - pred  # (N, Q)
        offsets = [
            float(np.quantile(residuals[:, j], tau, method="higher"))
            for j, tau in enumerate(self.quantiles)
        ]
        self._offsets = offsets
        cal_size = int(y.shape[0])
        pre = mean_quantile_coverage(raw_output, y_true)
        post = mean_quantile_coverage(self.transform(raw_output), y_true)
        emit_calibration_fit(logger, "conformal", cal_size, pre_coverage=pre, post_coverage=post)

    def transform(self, raw_output: Tensor) -> Tensor:
        if self._offsets is None:
            raise NotFittedError("ConformalCalibrator.transform before fit")
        pred = _as_pred_matrix(raw_output, len(self.quantiles), "ConformalCalibrator.transform")
        return torch.from_numpy(pred + np.asarray(self._offsets)).to(torch.float64)

    def serialize(self) -> dict[str, object]:
        if self._offsets is None:
            raise NotFittedError("ConformalCalibrator.serialize before fit")
        return {
            "type": "conformal",
            "quantiles": list(self.quantiles),
            "offsets": list(self._offsets),
        }

    @classmethod
    def deserialize(cls, blob: dict[str, object]) -> "ConformalCalibrator":
        obj = cls(quantiles=tuple(blob["quantiles"]))  # type: ignore[arg-type]
        obj._offsets = [float(v) for v in blob["offsets"]]  # type: ignore[union-attr]
        return obj


class IsotonicQuantileCalibrator:
    """Isotonic-smoothed residual-CDF additive offset (A9).

    Per column ``j`` an :class:`~sklearn.isotonic.IsotonicRegression`
    is fit mapping sorted residuals to their empirical CDF; ``delta_j``
    is the smallest knot whose fitted CDF reaches ``tau_j``. Inverting
    on the knot grid (not by interpolation) keeps v1's no-interpolation
    quantile contract. Unlike conformal this does not enforce
    monotonicity: the isotonic residual CDF is monotone by construction
    and the v1 contract assigns the crossing-quantile failure to
    conformal only.
    """

    def __init__(self, quantiles: tuple[float, ...]) -> None:
        self.quantiles = _validate_quantiles(quantiles)
        self._offsets: list[float] | None = None

    def fit(self, raw_output: Tensor, y_true: Tensor) -> None:
        raw_output, y_true = raw_output.detach().cpu(), y_true.detach().cpu()
        pred = _as_pred_matrix(raw_output, len(self.quantiles), "IsotonicQuantileCalibrator.fit")
        y = y_true.detach().to(torch.float64).reshape(-1, 1).numpy()
        residuals = y - pred  # (N, Q)
        n = residuals.shape[0]
        cdf_targets = np.arange(1, n + 1, dtype=np.float64) / n
        offsets: list[float] = []
        for j, tau in enumerate(self.quantiles):
            r_sorted = np.sort(residuals[:, j])
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            fitted_cdf = iso.fit(r_sorted, cdf_targets).predict(r_sorted)
            reached = np.flatnonzero(fitted_cdf >= tau)
            idx = int(reached[0]) if reached.size else n - 1
            offsets.append(float(r_sorted[idx]))
        self._offsets = offsets
        cal_size = int(y.shape[0])
        pre = mean_quantile_coverage(raw_output, y_true)
        post = mean_quantile_coverage(self.transform(raw_output), y_true)
        emit_calibration_fit(
            logger,
            "isotonic_quantile",
            cal_size,
            pre_coverage=pre,
            post_coverage=post,
        )

    def transform(self, raw_output: Tensor) -> Tensor:
        if self._offsets is None:
            raise NotFittedError("IsotonicQuantileCalibrator.transform before fit")
        pred = _as_pred_matrix(
            raw_output, len(self.quantiles), "IsotonicQuantileCalibrator.transform"
        )
        return torch.from_numpy(pred + np.asarray(self._offsets)).to(torch.float64)

    def serialize(self) -> dict[str, object]:
        if self._offsets is None:
            raise NotFittedError("IsotonicQuantileCalibrator.serialize before fit")
        return {
            "type": "isotonic_quantile",
            "quantiles": list(self.quantiles),
            "offsets": list(self._offsets),
        }

    @classmethod
    def deserialize(cls, blob: dict[str, object]) -> "IsotonicQuantileCalibrator":
        obj = cls(quantiles=tuple(blob["quantiles"]))  # type: ignore[arg-type]
        obj._offsets = [float(v) for v in blob["offsets"]]  # type: ignore[union-attr]
        return obj
