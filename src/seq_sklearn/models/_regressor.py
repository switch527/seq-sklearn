"""``BaseSequenceRegressor``: regressor overlay on the estimator shell (A2 / F1 / N1).

Adds the `RegressorMixin` contract: point ``predict``, ``quantiles_``
(quantile mode only), ``predict_quantiles`` with the three N1 error
paths (point-mode :class:`PredictionError`, pre-fit
:class:`NotFittedError`, off-fit-vector :class:`ValueError`), and the
conformal / isotonic-quantile calibrator dispatch. Still abstract: a
concrete family supplies :meth:`_build_backbone_head`.
"""

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from torch import Tensor

from seq_sklearn.calibration._protocol import _Calibrator
from seq_sklearn.calibration.regression import (
    ConformalCalibrator,
    IsotonicQuantileCalibrator,
)
from seq_sklearn.errors import PredictionError
from seq_sklearn.models._base import BaseSequenceEstimator

__all__ = ["BaseSequenceRegressor"]


class BaseSequenceRegressor(RegressorMixin, BaseSequenceEstimator):
    """Regressor family base (RegressorMixin + the estimator shell)."""

    # quantiles_ stays ABSENT for point regressors (A2 / F1.1), so it is
    # annotation-only here and set in _set_target_fit_state for quantile.
    quantiles_: np.ndarray

    def _is_quantile(self) -> bool:
        return self.task_type == "regression_quantile"

    def _encode_targets(self, y: object) -> np.ndarray:
        return np.asarray(y, dtype=np.float64).reshape(-1)

    def _set_target_fit_state(self, y: np.ndarray) -> None:  # noqa: ARG002
        # quantiles_ is set for quantile regressors only; ABSENT
        # otherwise so hasattr(est, 'quantiles_') is False (A2 / F1.1).
        if self._is_quantile():
            self.quantiles_ = np.asarray(self.quantiles, dtype=np.float64)

    def _head_out_dim(self) -> int:
        return 1

    def _n_quantiles(self) -> int:
        """Head quantile width: ``len(quantiles)`` quantile-mode, else 1."""
        return len(self.quantiles) if self._is_quantile() and self.quantiles else 1

    def _make_calibrator(self) -> _Calibrator | None:
        strategy = self.calibration_strategy
        if strategy == "none":
            return None
        quantiles = tuple(self.quantiles) if self.quantiles is not None else ()
        if strategy == "conformal":
            return ConformalCalibrator(quantiles)
        if strategy == "isotonic_quantile":
            return IsotonicQuantileCalibrator(quantiles)
        raise ValueError(f"calibration_strategy={strategy!r} is not a regressor strategy")

    def _build_calibrator_from(self, blob: dict[str, object]) -> _Calibrator:
        if self.calibration_strategy == "conformal":
            return ConformalCalibrator.deserialize(blob)
        return IsotonicQuantileCalibrator.deserialize(blob)

    def _family_state(self) -> dict[str, object]:
        if hasattr(self, "quantiles_"):
            return {"quantiles_": np.asarray(self.quantiles_).tolist()}
        return {}

    def _restore_family_state(self, state: dict[str, object]) -> None:
        if "quantiles_" in state:
            self.quantiles_ = np.asarray(state["quantiles_"], dtype=np.float64)

    def _raw_predict_matrix(self, X: pd.DataFrame) -> Tensor:  # noqa: N803
        """``(N, Q)`` raw quantile matrix (Q == 1 for point regression)."""
        raw = self._predict_raw(X)
        return raw.reshape(raw.shape[0], -1)

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        """Point predictions; quantile mode returns the median column (F1)."""
        mat = self._raw_predict_matrix(X)
        if not self._is_quantile():
            return mat.reshape(-1).numpy()
        q = np.asarray(self.quantiles_, dtype=np.float64)
        median_col = int(np.argmin(np.abs(q - 0.5)))
        return mat[:, median_col].numpy()

    def predict_quantiles(
        self,
        X: pd.DataFrame,  # noqa: N803
        quantiles: list[float] | None = None,
    ) -> np.ndarray:
        """Calibrated quantile estimates per row (F1 / N1).

        Raises:
            NotFittedError: ``fit`` has not run.
            PredictionError: the regressor is point mode (use ``predict``).
            ValueError: a requested quantile is not in the fit-time vector
                (v1 does not interpolate); the message lists that vector.
        """
        self._check_fitted()
        if not self._is_quantile():
            raise PredictionError(
                "predict_quantiles requires a quantile regressor "
                "(loss_strategy='pinball'); this is a point-mode regressor, "
                "use predict instead"
            )
        fit_q = np.asarray(self.quantiles_, dtype=np.float64)
        raw = self._raw_predict_matrix(X)
        calibrated = (
            self.calibrator_.transform(raw) if self.calibrator_ is not None else raw
        ).numpy()
        if quantiles is None:
            return calibrated
        requested = np.asarray(quantiles, dtype=np.float64)
        cols = []
        for q in requested:
            match = np.flatnonzero(np.isclose(fit_q, q))
            if match.size == 0:
                raise ValueError(
                    f"quantile {q} is not in the fit-time quantile vector "
                    f"{fit_q.tolist()}; v1 does not interpolate"
                )
            cols.append(int(match[0]))
        return calibrated[:, cols]
