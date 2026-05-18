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

    def _calibrated_matrix(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:  # noqa: N803
        """``(calibrated (N, Q) matrix, below-floor row mask)``.

        Calibrated when a calibrator is fitted, else raw. Shared so the
        point estimate (``predict``) is the same median
        ``predict_quantiles`` returns; otherwise the two would disagree
        by the calibration offset. Below-floor entity rows are
        NaN-filled here so every regressor predict path satisfies the
        F NaN-in-output contract (never zero-filled).
        """
        raw, below, input_row_order = self._predict_raw(X)
        # _calibrate_raw NaN-fills in sorted (transform) space; restore
        # caller X row order at this array boundary, AFTER the fill (F1).
        # _calibrate_raw itself must NOT reorder (it is shared with
        # predict_with_attention, which reorders per-field in Step 4;
        # reordering there would double-permute the attention path).
        calibrated = self._calibrate_raw(raw, below)
        return calibrated[input_row_order], below[input_row_order]

    def _calibrate_raw(self, raw: Tensor, below: np.ndarray) -> np.ndarray:
        """Calibrated ``(N, Q)`` matrix from raw head logits + below mask.

        Split from :meth:`_calibrated_matrix` so the transformer
        family's ``predict_with_attention`` reuses the identical
        calibrate-then-NaN-fill logic on the single backbone forward
        pass it already holds, instead of a second predict pass.
        """
        mat = raw.reshape(raw.shape[0], -1)
        if self.calibrator_ is not None:
            mat = self.calibrator_.transform(mat)
        out = mat.numpy()
        out[below] = np.nan
        return out

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        """Point predictions; quantile mode returns the calibrated median (F1)."""
        mat, _ = self._calibrated_matrix(X)
        if not self._is_quantile():
            return mat.reshape(-1)
        q = np.asarray(self.quantiles_, dtype=np.float64)
        median_col = int(np.argmin(np.abs(q - 0.5)))
        return mat[:, median_col]

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
        calibrated, _ = self._calibrated_matrix(X)
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
