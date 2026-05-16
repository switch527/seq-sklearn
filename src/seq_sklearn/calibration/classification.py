"""Classification calibrators (per architecture A9 / requirements F5).

Three concrete ``_Calibrator`` implementations applied at
``predict_proba`` time on the held-out calibration fold:

* :class:`TemperatureScaling` (binary + multiclass): one scalar ``T``
  fit by LBFGS on calibration-set NLL.
* :class:`PlattScaling` (binary only): a logistic ``a * logit + b``
  fit by LBFGS on calibration-set BCE.
* :class:`IsotonicCalibrator` (binary + multiclass): a non-parametric
  monotone fit; multiclass is one-vs-rest with row renormalisation.

Every calibrator consumes RAW logits and returns CALIBRATED
PROBABILITIES (binary: ``(N,)`` positive-class; multiclass: ``(N, K)``
row-stochastic), per the :mod:`seq_sklearn.calibration._protocol`
contract. Each ``fit`` emits the F11 ``calibration.fit`` record (and
``calibration.small_set`` when the fold is under-sized) so the path is
observable when exercised standalone.
"""

import logging

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from torch import Tensor
from torch.nn import functional as torch_functional

from seq_sklearn.calibration._events import emit_calibration_fit
from seq_sklearn.calibration._metrics import expected_calibration_error
from seq_sklearn.errors import NotFittedError, TrainingError

__all__ = ["IsotonicCalibrator", "PlattScaling", "TemperatureScaling"]

logger = logging.getLogger(__name__)

# LBFGS budget for the scalar/2-param convex fits. Implementation
# constants (cf. _NAN_ABORT_LIMIT), not user hyperparameters: there is
# no calibration config class in v1 and these never reach get_params.
_LBFGS_MAX_ITER = 100
_LBFGS_LR = 0.1


def _as_binary_logits(logits: Tensor) -> Tensor:
    """Flatten ``(N,)`` / ``(N, 1)`` binary logits to ``(N,)`` float64."""
    return logits.detach().reshape(-1).to(torch.float64)


def _require_nonempty(raw_output: Tensor, where: str) -> None:
    """Reject a zero-row calibration fold at the ``fit`` boundary.

    Raised before any optimizer step or state mutation so a degenerate
    fold fails loud with a ``ValueError`` and never leaves a
    partially-fitted calibrator.
    """
    if raw_output.shape[0] == 0:
        raise ValueError(f"{where}: calibration fold is empty (N=0)")


class TemperatureScaling:
    """Single-scalar temperature on the calibration-set NLL (A9).

    ``task`` is ``"binary"`` or ``"multiclass"``. The scalar ``T`` is
    parameterised as ``exp(log_T)`` so the LBFGS search stays on
    ``(0, inf)`` without a constraint; a non-finite optimum (a diverged
    calibration set) raises :class:`TrainingError`.
    """

    def __init__(self, task: str) -> None:
        if task not in ("binary", "multiclass"):
            raise ValueError(f"TemperatureScaling supports 'binary' / 'multiclass', got {task!r}")
        self.task = task
        self._temperature: float | None = None

    def fit(self, raw_output: Tensor, y_true: Tensor) -> None:
        _require_nonempty(raw_output, "TemperatureScaling.fit")
        log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=_LBFGS_LR, max_iter=_LBFGS_MAX_ITER)
        if self.task == "binary":
            x = _as_binary_logits(raw_output)
            y = y_true.detach().reshape(-1).to(torch.float64)

            def closure() -> Tensor:
                opt.zero_grad()
                loss = torch_functional.binary_cross_entropy_with_logits(x / log_t.exp(), y)
                loss.backward()
                return loss
        else:
            x = raw_output.detach().to(torch.float64)
            y = y_true.detach().reshape(-1).long()

            def closure() -> Tensor:
                opt.zero_grad()
                loss = torch_functional.cross_entropy(x / log_t.exp(), y)
                loss.backward()
                return loss

        opt.step(closure)
        temperature = float(log_t.detach().exp())
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise TrainingError(
                f"temperature scaling diverged on the calibration set "
                f"(T={temperature!r}); the calibration fold may be "
                f"degenerate or too small"
            )
        self._temperature = temperature
        cal_size = int(y_true.reshape(-1).shape[0])
        pre = expected_calibration_error(_probs(raw_output, self.task, 1.0), y_true)
        post = expected_calibration_error(self.transform(raw_output), y_true)
        emit_calibration_fit(logger, "temperature", cal_size, pre_ece=pre, post_ece=post)

    def transform(self, raw_output: Tensor) -> Tensor:
        if self._temperature is None:
            raise NotFittedError("TemperatureScaling.transform before fit")
        return _probs(raw_output, self.task, self._temperature)

    def serialize(self) -> dict[str, object]:
        if self._temperature is None:
            raise NotFittedError("TemperatureScaling.serialize before fit")
        return {
            "type": "temperature",
            "task": self.task,
            "temperature": self._temperature,
        }

    @classmethod
    def deserialize(cls, blob: dict[str, object]) -> "TemperatureScaling":
        obj = cls(task=str(blob["task"]))
        obj._temperature = float(blob["temperature"])  # type: ignore[arg-type]
        return obj


def _probs(logits: Tensor, task: str, temperature: float) -> Tensor:
    """Temperature-scaled activation: sigmoid (binary) / softmax (mc)."""
    if task == "binary":
        return torch.sigmoid(_as_binary_logits(logits) / temperature)
    scaled = logits.detach().to(torch.float64) / temperature
    return torch_functional.softmax(scaled, dim=1)


class PlattScaling:
    """Binary-only logistic recalibration ``sigmoid(a * logit + b)`` (A9).

    ``a`` and ``b`` are fit by LBFGS on calibration-set BCE. Binary
    only by construction (no ``task`` argument): Platt scaling has no
    canonical multiclass form in v1 (use temperature or isotonic). The
    validity matrix already restricts ``platt`` to binary, so the
    estimator never constructs this for a multiclass task.
    """

    def __init__(self) -> None:
        self._a: float | None = None
        self._b: float | None = None

    def fit(self, raw_output: Tensor, y_true: Tensor) -> None:
        _require_nonempty(raw_output, "PlattScaling.fit")
        x = _as_binary_logits(raw_output)
        y = y_true.detach().reshape(-1).to(torch.float64)
        a = torch.ones(1, dtype=torch.float64, requires_grad=True)
        b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([a, b], lr=_LBFGS_LR, max_iter=_LBFGS_MAX_ITER)

        def closure() -> Tensor:
            opt.zero_grad()
            loss = torch_functional.binary_cross_entropy_with_logits(a * x + b, y)
            loss.backward()
            return loss

        opt.step(closure)
        a_val, b_val = float(a.detach()), float(b.detach())
        if not (np.isfinite(a_val) and np.isfinite(b_val)):
            raise TrainingError(
                f"Platt scaling diverged on the calibration set (a={a_val!r}, b={b_val!r})"
            )
        self._a, self._b = a_val, b_val
        cal_size = int(y.shape[0])
        pre = expected_calibration_error(torch.sigmoid(x), y_true)
        post = expected_calibration_error(self.transform(raw_output), y_true)
        emit_calibration_fit(logger, "platt", cal_size, pre_ece=pre, post_ece=post)

    def transform(self, raw_output: Tensor) -> Tensor:
        if self._a is None or self._b is None:
            raise NotFittedError("PlattScaling.transform before fit")
        x = _as_binary_logits(raw_output)
        return torch.sigmoid(self._a * x + self._b)

    def serialize(self) -> dict[str, object]:
        if self._a is None or self._b is None:
            raise NotFittedError("PlattScaling.serialize before fit")
        return {"type": "platt", "a": self._a, "b": self._b}

    @classmethod
    def deserialize(cls, blob: dict[str, object]) -> "PlattScaling":
        obj = cls()
        obj._a = float(blob["a"])  # type: ignore[arg-type]
        obj._b = float(blob["b"])  # type: ignore[arg-type]
        return obj


class IsotonicCalibrator:
    """Non-parametric monotone calibration (A9).

    Binary fits one :class:`~sklearn.isotonic.IsotonicRegression` on the
    positive-class probability. Multiclass fits one regressor per class
    one-vs-rest on the softmax probabilities, then renormalises each row
    to sum to one (a row that maps to all-zero falls back to uniform so
    ``predict_proba`` never returns a non-stochastic row).

    State is persisted as the public interpolation knots
    (``X_thresholds_`` / ``y_thresholds_``); :meth:`deserialize`
    rebuilds each regressor by fitting on its own knots, which
    reproduces the exact piecewise-linear map without touching sklearn
    private attributes.
    """

    def __init__(self, task: str) -> None:
        if task not in ("binary", "multiclass"):
            raise ValueError(f"IsotonicCalibrator supports 'binary' / 'multiclass', got {task!r}")
        self.task = task
        self._models: list[IsotonicRegression] | None = None

    @staticmethod
    def _new_regressor() -> IsotonicRegression:
        return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, raw_output: Tensor, y_true: Tensor) -> None:
        _require_nonempty(raw_output, "IsotonicCalibrator.fit")
        y = y_true.detach().reshape(-1).long().numpy()
        if self.task == "binary":
            p = torch.sigmoid(_as_binary_logits(raw_output)).numpy()
            model = self._new_regressor().fit(p, (y == 1).astype(np.float64))
            self._models = [model]
        else:
            probs = torch_functional.softmax(raw_output.detach().to(torch.float64), dim=1).numpy()
            n_classes = probs.shape[1]
            self._models = [
                self._new_regressor().fit(probs[:, k], (y == k).astype(np.float64))
                for k in range(n_classes)
            ]
        cal_size = int(y.shape[0])
        pre = expected_calibration_error(_raw_probs(raw_output, self.task), y_true)
        post = expected_calibration_error(self.transform(raw_output), y_true)
        emit_calibration_fit(logger, "isotonic", cal_size, pre_ece=pre, post_ece=post)

    def transform(self, raw_output: Tensor) -> Tensor:
        if self._models is None:
            raise NotFittedError("IsotonicCalibrator.transform before fit")
        if self.task == "binary":
            p = torch.sigmoid(_as_binary_logits(raw_output)).numpy()
            return torch.from_numpy(self._models[0].predict(p)).to(torch.float64)
        probs = torch_functional.softmax(raw_output.detach().to(torch.float64), dim=1).numpy()
        cols = np.stack(
            [m.predict(probs[:, k]) for k, m in enumerate(self._models)],
            axis=1,
        )
        row_sum = cols.sum(axis=1, keepdims=True)
        n_classes = cols.shape[1]
        calibrated = np.divide(
            cols,
            row_sum,
            out=np.full_like(cols, 1.0 / n_classes),
            where=row_sum > 0.0,
        )
        return torch.from_numpy(calibrated).to(torch.float64)

    def serialize(self) -> dict[str, object]:
        if self._models is None:
            raise NotFittedError("IsotonicCalibrator.serialize before fit")
        return {
            "type": "isotonic",
            "task": self.task,
            "models": [
                {
                    "x": m.X_thresholds_.astype(float).tolist(),
                    "y": m.y_thresholds_.astype(float).tolist(),
                }
                for m in self._models
            ],
        }

    @classmethod
    def deserialize(cls, blob: dict[str, object]) -> "IsotonicCalibrator":
        obj = cls(task=str(blob["task"]))
        models: list[IsotonicRegression] = []
        for knots in blob["models"]:  # type: ignore[union-attr]
            x = np.asarray(knots["x"], dtype=np.float64)  # type: ignore[index]
            y = np.asarray(knots["y"], dtype=np.float64)  # type: ignore[index]
            models.append(cls._new_regressor().fit(x, y))
        obj._models = models
        return obj


def _raw_probs(logits: Tensor, task: str) -> Tensor:
    """Uncalibrated activation (temperature 1.0) for the pre-ECE scalar."""
    return _probs(logits, task, 1.0)
